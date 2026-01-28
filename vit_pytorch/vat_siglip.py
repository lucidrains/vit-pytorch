from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, cat, stack, tensor, einsum
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_context = None,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        norm_eps = 1e-6,
        gate_attn = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim, eps = norm_eps)

        self.is_cross_attn = exists(dim_context)
        dim_context = default(dim_context, dim)
        self.norm_context = nn.LayerNorm(dim_context, eps = norm_eps) if self.is_cross_attn else None

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2)

        self.to_out_gates = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b ... h -> b h ... 1'),
            nn.Sigmoid()
        ) if gate_attn else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None):
        x = self.norm(x)

        if self.is_cross_attn:
            assert exists(context)
            context = self.norm_context(context)
        else:
            context = x

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        if exists(self.to_out_gates):
            out = out * self.to_out_gates(x) # https://arxiv.org/abs/2505.06708

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

def FeedForward(
    dim,
    dim_inner,
    norm_eps = 1e-6
):
    return nn.Sequential(
        nn.LayerNorm(dim, eps = norm_eps),
        nn.Linear(dim, dim_inner),
        nn.GELU(approximate = 'tanh'),
        nn.Linear(dim_inner, dim)
    )

class SigLIP(Module):
    def __init__(
        self,
        image_size = 224,
        patch_size = 14,
        dim = 1152,
        depth = 27,
        heads = 16,
        mlp_dim = 4304,
        norm_eps = 1e-6
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        num_patches = (image_size // patch_size) ** 2
        dim_head = dim // heads

        self.to_patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_size * patch_size * 3, dim)
        )

        self.pos_embed = nn.Parameter(torch.randn(num_patches, dim))

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, norm_eps = norm_eps),
                FeedForward(dim = dim, dim_inner = mlp_dim, norm_eps = norm_eps)
            ]))

        self.norm = nn.LayerNorm(dim, eps = norm_eps)

    def forward(self, x, return_hiddens = False):
        x = self.to_patch_embed(x)
        num_patches = x.shape[1]

        x = x + self.pos_embed[:num_patches]

        hiddens = []

        for attn, ff in self.layers:
            hiddens.append(x)

            x = attn(x) + x
            x = ff(x) + x

        out = self.norm(x)

        if not return_hiddens:
            return out

        return out, stack(hiddens)

class FiLM(Module):
    def __init__(self, dim):
        super().__init__()
        proj = nn.Linear(dim, dim * 2)

        self.to_gamma_beta = nn.Sequential(
            proj,
            Rearrange('b (two d) -> two b 1 d', two = 2)
        )

        nn.init.zeros_(proj.weight)
        nn.init.zeros_(proj.bias)

    def forward(self, tokens, cond):
        gamma, beta = self.to_gamma_beta(cond)
        return tokens * gamma + beta

class SigLIPVAT(Module):
    def __init__(
        self,
        *,
        dim = 512,
        depth = 27,
        heads = 8,
        dim_head = 64,
        dim_action = 32,
        mlp_dim = 2048,
        num_views = 1,
        num_tasks = None,
        dim_extra_token = None,
        num_register_tokens = 4,
        action_chunk_len = 50,
        time_seq_len = 1,
        dropout = 0.,
        add_self_attn = True,
        self_attn_heads = 4,
        self_attn_dim_head = 32,
        vit_layer_indices: tuple[int, ...] | None = None,
        siglip_image_size = 224,
        siglip_patch_size = 14,
        siglip_dim = 1152,
        siglip_depth = 27,
        siglip_heads = 16,
        siglip_mlp_dim = 4304,
        siglip_norm_eps = 1e-6,
    ):
        super().__init__()

        self.vit = SigLIP(
            image_size = siglip_image_size,
            patch_size = siglip_patch_size,
            dim = siglip_dim,
            depth = siglip_depth,
            heads = siglip_heads,
            mlp_dim = siglip_mlp_dim,
            norm_eps = siglip_norm_eps
        )

        vit_dim = siglip_dim
        self.vit_dim = vit_dim

        vit_layer_indices = default(vit_layer_indices, tuple(range(depth)))
        assert len(vit_layer_indices) == depth, f'number of vit layer indices {len(vit_layer_indices)} does not much the VAT depth {depth}'
        self.register_buffer('layer_indices', tensor(vit_layer_indices), persistent = False)

        # handle maybe multiple frames

        is_video = time_seq_len > 1
        self.is_video = is_video
        self.time_seq_len = time_seq_len
        self.time_pos_emb = nn.Parameter(torch.randn(time_seq_len, vit_dim) * 1e-2) if is_video else None

        # maybe view embeddings

        self.view_emb = nn.Parameter(torch.randn(num_views, vit_dim) * 1e-2) if exists(num_views) and num_views > 1 else None

        # handle maybe task conditioning

        self.has_tasks = exists(num_tasks)
        if self.has_tasks:
            self.task_emb = nn.Parameter(torch.randn(num_tasks, dim) * 1e-2)

        # register tokens

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

        # to action tokens

        self.action_pos_emb = nn.Parameter(torch.randn(action_chunk_len, dim) * 1e-2)

        self.layers = ModuleList([])
        for _ in range(depth):
            maybe_film = FiLM(dim = dim) if self.has_tasks else None
            maybe_self_attn = Attention(dim = dim, heads = self_attn_heads, dim_head = self_attn_dim_head, dropout = dropout) if add_self_attn else None

            self.layers.append(ModuleList([
                maybe_film,
                maybe_self_attn,
                Attention(dim = dim, dim_context = vit_dim, heads = heads, dim_head = dim_head, dropout = dropout, gate_attn = True),
                FeedForward(dim = dim, dim_inner = mlp_dim)
            ]))

        self.final_norm = nn.LayerNorm(dim)
        self.to_pred_action = nn.Linear(dim, dim_action, bias = False)

        # handle the extra token

        self.accept_extra_token = exists(dim_extra_token)
        if exists(dim_extra_token):
            self.to_extra_token = nn.Linear(dim_extra_token, dim)

    def load_siglip(
        self,
        repo_id = 'google/siglip-so400m-patch14-224',
        folder = 'checkpoints/siglip'
    ):
        folder = Path(folder)
        if not folder.exists():
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id = repo_id,
                local_dir = folder,
                allow_patterns = ['config.json', 'model.safetensors']
            )

        from safetensors import safe_open
        weights_path = folder / 'model.safetensors'

        # Auto-detect prefix based on keys
        with safe_open(weights_path, framework = 'pt') as f:
            keys = f.keys()
            
            vi_p = ''
            if any(k.startswith('paligemma_with_expert.paligemma.model.vision_tower.vision_model') for k in keys):
                vi_p = 'paligemma_with_expert.paligemma.model.vision_tower.vision_model.'
            elif any(k.startswith('vision_model') for k in keys):
                vi_p = 'vision_model.'
            
            pz_state = self.vit.state_dict()

            def copy_weight_bias(pz_prefix, vi_prefix):
                pz_state[f'{pz_prefix}.weight'].copy_(f.get_tensor(f'{vi_prefix}.weight'))
                pz_state[f'{pz_prefix}.bias'].copy_(f.get_tensor(f'{vi_prefix}.bias'))

            # patch embedding
            patch_weight = rearrange(f.get_tensor(f'{vi_p}embeddings.patch_embedding.weight'), 'd c h w -> d (h w c)')
            pz_state['to_patch_embed.1.weight'].copy_(patch_weight)
            pz_state['to_patch_embed.1.bias'].copy_(f.get_tensor(f'{vi_p}embeddings.patch_embedding.bias'))

            # position embedding
            pz_state['pos_embed'].copy_(f.get_tensor(f'{vi_p}embeddings.position_embedding.weight'))

            # transformer layers
            for i in range(self.vit.depth):
                v_pi = f'{vi_p}encoder.layers.{i}'
                v_pz = f'layers.{i}'

                # attention
                copy_weight_bias(f'{v_pz}.0.norm', f'{v_pi}.layer_norm1')
                copy_weight_bias(f'{v_pz}.0.to_q', f'{v_pi}.self_attn.q_proj')

                vk, vv = [f.get_tensor(f'{v_pi}.self_attn.{x}_proj.weight') for x in ('k', 'v')]
                bk, bv = [f.get_tensor(f'{v_pi}.self_attn.{x}_proj.bias') for x in ('k', 'v')]

                pz_state[f'{v_pz}.0.to_kv.weight'].copy_(cat((vk, vv), dim = 0))
                pz_state[f'{v_pz}.0.to_kv.bias'].copy_(cat((bk, bv), dim = 0))

                copy_weight_bias(f'{v_pz}.0.to_out.0', f'{v_pi}.self_attn.out_proj')

                # feedforward
                copy_weight_bias(f'{v_pz}.1.0', f'{v_pi}.layer_norm2')
                copy_weight_bias(f'{v_pz}.1.1', f'{v_pi}.mlp.fc1')
                copy_weight_bias(f'{v_pz}.1.3', f'{v_pi}.mlp.fc2')

            # post-layernorm
            copy_weight_bias('norm', f'{vi_p}post_layernorm')

            self.vit.load_state_dict(pz_state)

        print(f'Successfully loaded SigLIP weights from {repo_id}')

    def forward(
        self,
        video_or_image,   # (b v? c t? h w)
        *,
        extra = None,
        tasks = None,
        actions = None,
        return_hiddens = False,
        freeze_vit = False
    ):
        batch = video_or_image.shape[0]
        return_loss = exists(actions)

        # handle some various input dimensions

        if video_or_image.ndim == 4:
            video_or_image = rearrange(video_or_image, 'b 1 c h w')

        if video_or_image.ndim == 5:
            video_or_image = rearrange(video_or_image, 'b v c h w -> b v c 1 h w')

        assert video_or_image.shape[3] == self.time_seq_len

        # to images

        images = rearrange(video_or_image, 'b v c t h w -> b v t c h w')
        images, packed_shape = pack([images], '* c h w')

        # get representation trajectory from vit

        vit_forward_context = torch.no_grad if freeze_vit else nullcontext

        with vit_forward_context():
            embed, hiddens = self.vit(images, return_hiddens = True)

        hiddens = cat((hiddens, embed[None, ...]))

        # extract the hiddens needed for the action cross attention

        hiddens = hiddens[self.layer_indices]

        # pack temporarily for embedding

        hiddens, = unpack(hiddens, packed_shape, 'l * n d') # l for layers

        # maybe add time embeddings

        if self.is_video:
            time_pos_emb = rearrange(self.time_pos_emb, 't d -> t 1 d')
            hiddens = hiddens + time_pos_emb

        # maybe view embeddings

        if exists(self.view_emb):
            view_emb = rearrange(self.view_emb, 'v d -> v 1 1 d')
            hiddens = hiddens + view_emb

        # maybe tasks

        if exists(tasks):
            task_emb = self.task_emb[tasks]

        # cross from actions to representation trajectory

        context = rearrange(hiddens, 'l b v t n d -> l b (v t n) d')

        # get main action tokens and maybe append extra

        action_tokens = repeat(self.action_pos_emb, 'k d -> b k d', b = batch)

        has_extra = exists(extra)
        if has_extra:
            extra_token = self.to_extra_token(extra)
            action_tokens, packed_extra = pack([action_tokens, extra_token], 'b * d')

        # register tokens

        register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = batch)
        action_tokens, registers_packed_shape = pack((register_tokens, action_tokens), 'b * d')

        # cross attention

        vat_hiddens = [action_tokens]

        for (maybe_film, maybe_self_attn, cross_attn, ff), layer_context in zip(self.layers, context):

            if exists(tasks):
                action_tokens = maybe_film(action_tokens, task_emb)

            action_tokens = cross_attn(action_tokens, layer_context) + action_tokens

            if exists(maybe_self_attn):
                action_tokens = maybe_self_attn(action_tokens) + action_tokens

            action_tokens = ff(action_tokens) + action_tokens

            vat_hiddens.append(action_tokens)

        # unpack registers

        _, action_tokens = unpack(action_tokens, registers_packed_shape, 'b * d')

        # maybe unpack extra

        if has_extra:
            action_tokens, _ = unpack(action_tokens, packed_extra, 'b * d')

        # norm and prediction

        action_tokens = self.final_norm(action_tokens)
        pred_action = self.to_pred_action(action_tokens)

        if not return_loss:
            if not return_hiddens:
                return pred_action

            return pred_action, stack(vat_hiddens)

        assert pred_action.shape[1] == actions.shape[1]

        return F.l1_loss(pred_action, actions)

# quick test

if __name__ == '__main__':
    vat = SigLIPVAT(
        num_tasks = 4,
        dim_extra_token = 32,
        time_seq_len = 2,
        num_views = 2,
        depth = 4,
        vit_layer_indices = (               # extending on the paper, allow for any order of hiddens, and also allow for depth index (which equates to the final embedding output from the vit)
            0, 1, 26, 27
        )
    )

    vat.load_siglip() # load siglip weights from hf

    # inputs

    images = torch.randn(1, 2, 3, 2, 224, 224) # (b, v, c, t, h, w)
    tasks = torch.randint(0, 4, (1,))
    extra = torch.randn(1, 32)

    actions = torch.randn(1, 50, 32) # actions for learning

    loss = vat(images, actions = actions, tasks = tasks, extra = extra, freeze_vit = True)
    loss.backward()

    # after much training

    pred_actions = vat(images, tasks = tasks, extra = extra)
    
    assert pred_actions.shape == (1, 50, 32)
