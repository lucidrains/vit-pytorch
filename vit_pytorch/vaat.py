# vision-audio-action transformer - vaat

from __future__ import annotations
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn, cat, stack, arange, tensor
from torch.nn import Module, ModuleList

from torchaudio.transforms import Spectrogram

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 2d sinusoidal positional embedding
# simple vit paper shows it is good enough compared to learned

def posemb_sincos_2d(
    patches,
    temperature = 10000,
    dtype = torch.float32
):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'

    omega = arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = temperature ** -omega

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    pe = pe.type(dtype)

    return rearrange(pe, '(h w) d -> h w d', h = h, w = w)

# classes

class FiLM(Module):
    def __init__(
        self,
        dim,
    ):
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

class FeedForward(Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        dim_context = None,
        cross_attend = False
    ):
        super().__init__()
        dim_context = default(dim_context, dim)
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.cross_attend = cross_attend
        self.context_norm = nn.LayerNorm(dim_context) if cross_attend else None

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, context = None):

        assert not (self.cross_attend ^ exists(context)), 'context must be passed in if cross attending, or vice versa'

        x = self.norm(x)

        # handle norming of context for cross attention

        kv_input = x

        if self.cross_attend:
            context = self.context_norm(context)
            kv_input = context

        # project for queries, keys, values

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout = 0.
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(
        self,
        x,
        return_hiddens = False
    ):

        hiddens = []

        for attn, ff in self.layers:
            hiddens.append(x)

            x = attn(x) + x
            x = ff(x) + x

        x = self.norm(x)

        if not return_hiddens:
            return x

        return x, hiddens

class AST(Module):
    # audio spectrogram transformer https://arxiv.org/abs/2104.01778

    def __init__(
        self,
        dim,
        depth,
        mlp_dim,
        num_classes = None,
        patch_size = 16,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        accept_spec = False,
        accept_spec_time_first = True,
        spec_n_fft = 128,
        spec_power = 2,
        spec_win_length = 24,
        spec_hop_length = None,
        spec_pad = 0,
        spec_center = True,
        spec_pad_mode = 'reflect',
        num_register_tokens = 4
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        patch_height, patch_width = pair(patch_size)
        patch_input_dim = patch_height * patch_width

        self.patch_size = (patch_height, patch_width)

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b (h p1) (w p2) -> b h w (p1 p2)', p1 = self.patch_size[0], p2 = self.patch_size[1]),
            nn.LayerNorm(patch_input_dim),
            nn.Linear(patch_input_dim, dim),
            nn.LayerNorm(dim)
        )

        self.accept_spec = accept_spec
        self.accept_spec_time_first = accept_spec_time_first

        self.spec = Spectrogram(
            n_fft = spec_n_fft,
            power = spec_power,
            win_length = spec_win_length,
            hop_length = spec_hop_length,
            pad = spec_pad,
            center = spec_center,
            pad_mode = spec_pad_mode
        )

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            mlp_dim = mlp_dim,
            dropout = dropout,
        )

        self.final_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Linear(dim, num_classes) if exists(num_classes) else nn.Identity()

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

    def forward(
        self,
        raw_audio_or_spec, # (b t) | (b f t)
        return_hiddens = False
    ):
        batch, device = raw_audio_or_spec.shape[0], raw_audio_or_spec.device

        assert (self.accept_spec and raw_audio_or_spec.ndim == 3) or (not self.accept_spec and raw_audio_or_spec.ndim == 2)

        if self.accept_spec:
            spec = rearrange(raw_audio_or_spec, 'b t f -> b f t')
        else:
            spec = self.spec(raw_audio_or_spec)

        # automatically crop if audio does not yield a 2d spectrogram that is divisible by patch sizes

        height, width = spec.shape[-2:]
        patch_height, patch_width = self.patch_size

        rounded_height = height // patch_height * patch_height
        rounded_width = width // patch_width * patch_width

        spec = spec[..., :rounded_height, :rounded_width]

        # to patches

        tokens = self.to_patch_tokens(spec)

        # get number of patches along height and width

        _, num_patch_height, num_patch_width, _ = tokens.shape

        # 2d sinusoidal positional embedding

        tokens = tokens + posemb_sincos_2d(tokens)

        tokens = rearrange(tokens, 'b ... c -> b (...) c')

        # register tokens

        register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = batch)

        tokens, packed_shape = pack((register_tokens, tokens), 'b * d')

        # attention

        attended, hiddens = self.transformer(tokens, return_hiddens = True)

        # final global average and norm (most recent papers show this is superior to CLS token)

        normed = self.final_norm(attended)

        if return_hiddens:
            return normed, stack(hiddens)

        register_tokens, normed = unpack(normed, packed_shape, 'b * d')

        pooled = reduce(normed, 'b n d -> b d', 'mean')

        maybe_logits = self.mlp_head(pooled)

        return maybe_logits

class ViT(Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        num_register_tokens = 0
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

    def forward(self, img, return_hiddens = False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:n]

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = b)

        x, packed_shape = pack((register_tokens, cls_tokens, x), 'b * d')

        x = self.dropout(x)

        x, hiddens = self.transformer(x, return_hiddens = True)

        # return the representation trajectory

        if return_hiddens:
            return x, stack(hiddens)

        register_tokens, cls_tokens, x = unpack(x, packed_shape, 'b * d')

        x = x.mean(dim = 1) if self.pool == 'mean' else cls_tokens

        x = self.to_latent(x)

        return self.mlp_head(x)

# proposed VAT

# https://openreview.net/forum?id=TalHOvvLZu
# simple way to get SOTA on Libero dataset (beating fine-tuned pi-zero)

class VAAT(Module):
    def __init__(
        self,
        vit: ViT | dict,
        ast: AST | dict,
        *,
        dim,
        depth,
        heads,
        dim_head,
        dim_action,
        mlp_dim,
        num_image_views = None,
        num_audio_views = None,
        num_tasks = None,
        dim_extra_token = None,
        num_register_tokens = 4,
        action_chunk_len = 7,
        time_seq_len = 1,
        dropout = 0.,
        add_self_attn = True,  # in the paper, they didn't have any ways for the action token to exchange information with the extra token, so we'll just add it as an option
        self_attn_heads = 4,
        self_attn_dim_head = 32,
        ast_layer_indices: tuple[int, ...] | None = None,
        vit_layer_indices: tuple[int, ...] | None = None
    ):
        super().__init__()

        # vit

        if isinstance(vit, dict):
            vit = ViT(**vit)

        self.vit = vit

        vit_dim = vit.dim

        assert vit.depth == depth or exists(vit_layer_indices), f'if the VAAT depth is not equal to the ViT depth, you must pass in the indices from the ViT to be layered to the VAAT in order from bottom to top'

        vit_layer_indices = default(vit_layer_indices, tuple(range(depth)))

        assert len(vit_layer_indices) == depth, f'number of vit layer indices {len(vit_layer_indices)} does not much the VAT depth {depth}'

        self.register_buffer('vit_layer_indices', tensor(vit_layer_indices), persistent = False)

        # ast

        if isinstance(ast, dict):
            ast = AST(**ast)

        self.ast = ast

        ast_dim = ast.dim

        self.ast_accept_spec = ast.accept_spec

        assert ast.depth == depth or exists(ast_layer_indices), f'if the VAAT depth is not equal to the AST depth, you must pass in the indices from the AST to be layered to the VAAT in order from bottom to top'

        ast_layer_indices = default(ast_layer_indices, tuple(range(depth)))

        assert len(ast_layer_indices) == depth, f'number of ast layer indices {len(ast_layer_indices)} does not much the VAAT depth {depth}'

        self.register_buffer('ast_layer_indices', tensor(vit_layer_indices), persistent = False)

        # handle maybe multiple frames

        is_video = time_seq_len > 1

        self.is_video = is_video
        self.time_seq_len = time_seq_len
        self.time_pos_emb = nn.Parameter(torch.randn(time_seq_len, vit_dim) * 1e-2) if is_video else None

        # maybe view embeddings

        self.image_view_emb = nn.Parameter(torch.randn(num_image_views, vit_dim) * 1e-2) if exists(num_image_views) and num_image_views > 1 else None

        self.audio_view_emb = nn.Parameter(torch.randn(num_audio_views, ast_dim) * 1e-2) if exists(num_audio_views) and num_audio_views > 1 else None

        # handle maybe task conditioning

        self.has_tasks = exists(num_tasks)

        if self.has_tasks:
            self.task_emb = nn.Parameter(torch.randn(num_tasks, dim) * 1e-2)

        # register tokens from Darcet et al.

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
                Attention(dim = dim, dim_context = vit_dim, heads = heads, dim_head = dim_head, dropout = dropout, cross_attend = True),
                Attention(dim = dim, dim_context = ast_dim, heads = heads, dim_head = dim_head, dropout = dropout, cross_attend = True),
                FeedForward(dim = dim, hidden_dim = mlp_dim, dropout = dropout)
            ]))

        self.final_norm = nn.LayerNorm(dim)
        self.to_pred_action = nn.Linear(dim, dim_action, bias = False)

        # handle the extra token

        self.accept_extra_token = exists(dim_extra_token)

        if exists(dim_extra_token):
            self.to_extra_token = nn.Linear(dim_extra_token, dim)

    def forward(
        self,
        video_or_image,   # (b v? c t? h w)      - batch, views [wrist + third person or more], channels, maybe time, height, width
        audio_or_spec,    # (b v? t) | (b v?f t) - batch, audio len | batch, spec freq, time
        *,
        extra = None,     # (b d)                - batch, dim extra     
        tasks = None,     # (b)
        actions = None,   # (b k d)              - batch, action chunk length, action dimension
        return_hiddens = False,
        freeze_vit = False,
        freeze_ast = False
    ):
        batch = video_or_image.shape[0]
        return_loss = exists(actions)

        # handle some various input dimensions

        if video_or_image.ndim == 4:
            video_or_image = rearrange(video_or_image, 'b 1 c h w')

        assert (
            (video_or_image.ndim == 5 and not self.is_video) or
            (video_or_image.ndim == 6 and self.is_video)
        )

        if video_or_image.ndim == 5:
            video_or_image = rearrange(video_or_image, 'b v c h w -> b v c 1 h w')

        assert video_or_image.shape[3] == self.time_seq_len

        # audio shapes - adding view if impliciy to be 1

        if audio_or_spec.ndim == 2 and not self.ast_accept_spec:
            audio_or_spec = rearrange(audio_or_spec, 'b t -> b 1 t')

        elif audio_or_spec.ndim == 3 and self.ast_accept_spec:
            audio_or_spec = rearrange(audio_or_spec, 'b f t -> b 1 f t')

        # to images

        images = rearrange(video_or_image, 'b v c t h w -> b v t c h w')

        images, image_packed_shape = pack([images], '* c h w')

        # to audio

        if self.ast_accept_spec:
            audio_or_spec, audio_packed_shape = pack([audio_or_spec], '* f t')
        else:
            audio_or_spec, audio_packed_shape = pack([audio_or_spec], '* t')

        # get representation trajectory from vit

        vit_forward_context = torch.no_grad if freeze_vit else nullcontext

        with vit_forward_context():
            embed, hiddens = self.vit(images, return_hiddens = True)

        hiddens = cat((hiddens, embed[None, ...]))

        # extract the hiddens needed for the action cross attention

        hiddens = hiddens[self.vit_layer_indices]

        # unpack temporarily for embedding

        hiddens, = unpack(hiddens, image_packed_shape, 'l * n d') # l for layers

        # maybe add time embeddings

        if self.is_video:
            time_pos_emb = rearrange(self.time_pos_emb, 't d -> t 1 d')
            hiddens = hiddens + time_pos_emb

        # maybe view embeddings

        if exists(self.image_view_emb):
            assert self.image_view_emb.shape[0] == hiddens.shape[2]

            image_view_emb = rearrange(self.image_view_emb, 'v d -> v 1 1 d')
            hiddens = hiddens + image_view_emb

        # get representation trajectory from ast

        ast_forward_context = torch.no_grad if freeze_ast else nullcontext

        with ast_forward_context():
            audio_embed, audio_hiddens = self.ast(audio_or_spec, return_hiddens = True)

        audio_hiddens = cat((audio_hiddens, audio_embed[None, ...]))

        # extract the hiddens needed for the action cross attention

        audio_hiddens = audio_hiddens[self.ast_layer_indices]

        # unpack audio temporarily for embedding

        audio_hiddens, = unpack(audio_hiddens, audio_packed_shape, 'l * n d') # l for layers

        # maybe audio view embeddings

        if exists(self.audio_view_emb):
            assert self.audio_view_emb.shape[0] == audio_hiddens.shape[2]

            audio_view_emb = rearrange(self.audio_view_emb, 'v d -> v 1 1 d')
            audio_hiddens = audio_hiddens + audio_view_emb

        # maybe tasks

        if exists(tasks):
            assert self.has_tasks, f'`num_tasks` must be set on `VAT` for task conditioning'

            task_emb = self.task_emb[tasks]

        # cross from actions to representation trajectory

        image_context = rearrange(hiddens, 'l b v t n d -> l b (v t n) d')

        audio_context = rearrange(audio_hiddens, 'l b v n d -> l b (v n) d')

        # get main action tokens and maybe append extra

        action_tokens = repeat(self.action_pos_emb, 'k d -> b k d', b = batch)

        has_extra = exists(extra)

        if has_extra:
            assert self.accept_extra_token

            extra_token = self.to_extra_token(extra)

            action_tokens, packed_extra = pack([action_tokens, extra_token], 'b * d')

        # register tokens

        register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = batch)

        action_tokens, registers_packed_shape = pack((register_tokens, action_tokens), 'b * d')

        # cross attention

        hiddens = [action_tokens]

        for (maybe_film, maybe_self_attn, image_cross_attn, audio_cross_attn, ff), image_layer_context, audio_layer_context in zip(self.layers, image_context, audio_context):

            if exists(tasks):
                action_tokens = maybe_film(action_tokens, task_emb)

            action_tokens = image_cross_attn(action_tokens, image_layer_context) + action_tokens

            action_tokens = audio_cross_attn(action_tokens, audio_layer_context) + action_tokens

            if exists(maybe_self_attn):
                action_tokens = maybe_self_attn(action_tokens) + action_tokens

            action_tokens = ff(action_tokens) + action_tokens

            hiddens.append(action_tokens)

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

            return pred_action, stack(hiddens)

        assert pred_action.shape[1] == actions.shape[1]

        # they found l1 loss suffices

        return F.l1_loss(pred_action, actions)

# quick test

if __name__ == '__main__':

    vit = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 384,
        heads = 8,
        depth = 4,
        mlp_dim = 384 * 4
    )

    ast = AST(
        dim = 384,
        depth = 4,
        heads = 8,
        num_classes = 1000,
        patch_size = 16,
        mlp_dim = 384 * 4
    )

    vaat = VAAT(
        vit,
        ast,
        dim = 512,
        depth = 9,
        heads = 8,
        dim_head = 64,
        mlp_dim = 2048,
        dim_action = 20,
        action_chunk_len = 7,
        time_seq_len = 4,
        num_image_views = 2,
        num_audio_views = 2,
        num_tasks = 4,
        add_self_attn = True,
        dim_extra_token = 33,               # extra token with some variable dimension
        vit_layer_indices = (               # extending on the paper, allow for any order of hiddens, and also allow for depth index (which equates to the final embedding output from the vit)
            0, 0, 1, 1, 2, 2, 3, 3, 4
        ),
        ast_layer_indices = (
            1, 1, 1, 2, 2, 2, 3, 3, 3
        )
    )

    images = torch.randn(2, 2, 3, 4, 256, 256) # (2 views with 4 frames)
    audio = torch.randn(2, 2, 14_100 * 5)

    tasks = torch.randint(0, 4, (2,))
    extra = torch.randn(2, 33)                 # extra internal state

    actions = torch.randn(2, 7, 20)         # actions for learning

    loss = vaat(images, audio, actions = actions, tasks = tasks, extra = extra, freeze_vit = True)
    loss.backward()

    # after much training

    pred_actions, hiddens = vaat(images, audio, tasks = tasks, extra = extra, return_hiddens = True)

    assert pred_actions.shape == (2, 7, 20)
