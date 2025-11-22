from __future__ import annotations
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn, cat, stack, tensor
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
        dim_context = None,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
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

class VAT(Module):
    def __init__(
        self,
        vit: ViT | dict,
        *,
        dim,
        depth,
        heads,
        dim_head,
        dim_action,
        mlp_dim,
        num_views = None,
        num_tasks = None,
        dim_extra_token = None,
        num_register_tokens = 4,
        action_chunk_len = 7,
        time_seq_len = 1,
        dropout = 0.,
        add_self_attn = True,  # in the paper, they didn't have any ways for the action token to exchange information with the extra token, so we'll just add it as an option
        self_attn_heads = 4,
        self_attn_dim_head = 32,
        vit_layer_indices: tuple[int, ...] | None = None
    ):
        super().__init__()

        if isinstance(vit, dict):
            vit = ViT(**vit)

        self.vit = vit

        vit_dim = vit.dim

        assert vit.depth == depth or exists(vit_layer_indices), f'if the VAT depth is not equal to the ViT depth, you must pass in the indices from the ViT to be layered to the VAT in order from bottom to top'

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
        video_or_image,   # (b v? c t? h w) - batch, views [wrist + third person or more], channels, maybe time, height, width
        *,
        extra = None,     # (b d)           - batch, dim extra     
        tasks = None,     # (b)
        actions = None,   # (b k d)         - batch, action chunk length, action dimension
        return_hiddens = False,
        freeze_vit = False
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
            assert self.view_emb.shape[0] == hiddens.shape[2]

            view_emb = rearrange(self.view_emb, 'v d -> v 1 1 d')
            hiddens = hiddens + view_emb

        # maybe tasks

        if exists(tasks):
            assert self.has_tasks, f'`num_tasks` must be set on `VAT` for task conditioning'

            task_emb = self.task_emb[tasks]

        # cross from actions to representation trajectory

        context = rearrange(hiddens, 'l b v t n d -> l b (v t n) d')

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

        for (maybe_film, maybe_self_attn, cross_attn, ff), layer_context in zip(self.layers, context):

            if exists(tasks):
                action_tokens = maybe_film(action_tokens, task_emb)

            action_tokens = cross_attn(action_tokens, layer_context) + action_tokens

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
        dim = 256,
        heads = 8,
        depth = 4,
        mlp_dim = 1024
    )

    vat = VAT(
        vit,
        dim = 512,
        depth = 9,
        heads = 8,
        dim_head = 64,
        mlp_dim = 2048,
        dim_action = 20,
        action_chunk_len = 7,
        time_seq_len = 4,
        num_views = 2,
        num_tasks = 4,
        add_self_attn = True,
        dim_extra_token = 33,               # extra token with some variable dimension
        vit_layer_indices = (               # extending on the paper, allow for any order of hiddens, and also allow for depth index (which equates to the final embedding output from the vit)
            0, 0, 1, 1, 2, 2, 3, 3, 4
        )
    )

    images = torch.randn(2, 2, 3, 4, 256, 256) # (2 views with 4 frames)
    tasks = torch.randint(0, 4, (2,))
    extra = torch.randn(2, 33)                 # extra internal state

    actions = torch.randn(2, 7, 20)         # actions for learning

    loss = vat(images, actions = actions, tasks = tasks, extra = extra, freeze_vit = True)
    loss.backward()

    # after much training

    pred_actions, hiddens = vat(images, tasks = tasks, extra = extra, return_hiddens = True)

    assert pred_actions.shape == (2, 7, 20)
