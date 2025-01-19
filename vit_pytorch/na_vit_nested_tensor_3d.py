from __future__ import annotations

from typing import List
from functools import partial

import torch
import packaging.version as pkg_version

from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.nested import nested_tensor

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# feedforward

def FeedForward(dim, hidden_dim, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim, bias = False),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., qk_norm = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim, bias = False)

        dim_inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head

        self.to_queries = nn.Linear(dim, dim_inner, bias = False)
        self.to_keys = nn.Linear(dim, dim_inner, bias = False)
        self.to_values = nn.Linear(dim, dim_inner, bias = False)

        # in the paper, they employ qk rmsnorm, a way to stabilize attention
        # will use layernorm in place of rmsnorm, which has been shown to work in certain papers. requires l2norm on non-ragged dimension to be supported in nested tensors

        self.query_norm = nn.LayerNorm(dim_head, bias = False) if qk_norm else nn.Identity()
        self.key_norm = nn.LayerNorm(dim_head, bias = False) if qk_norm else nn.Identity()

        self.dropout = dropout

        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self, 
        x,
        context: Tensor | None = None
    ):

        x = self.norm(x)

        # for attention pooling, one query pooling to entire sequence

        context = default(context, x)

        # queries, keys, values

        query = self.to_queries(x)
        key = self.to_keys(context)
        value = self.to_values(context)

        # split heads

        def split_heads(t):
            return t.unflatten(-1, (self.heads, self.dim_head))

        def transpose_head_seq(t):
            return t.transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))

        # qk norm for attention stability

        query = self.query_norm(query)
        key = self.key_norm(key)

        query, key, value = map(transpose_head_seq, (query, key, value))

        # attention

        out = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p = self.dropout if self.training else 0.
        )

        # merge heads

        out = out.transpose(1, 2).flatten(-2)

        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., qk_norm = True):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, qk_norm = qk_norm),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

        self.norm = nn.LayerNorm(dim, bias = False)

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class NaViT(Module):
    def __init__(
        self,
        *,
        image_size,
        max_frames,
        patch_size,
        frame_patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        num_registers = 4,
        qk_rmsnorm = True,
        token_dropout_prob: float | None = None
    ):
        super().__init__()
        image_height, image_width = pair(image_size)

        if pkg_version.parse(torch.__version__) < pkg_version.parse('2.5'):
            print('nested tensor NaViT was tested on pytorch 2.5')

        # what percent of tokens to dropout
        # if int or float given, then assume constant dropout prob
        # otherwise accept a callback that in turn calculates dropout prob from height and width

        self.token_dropout_prob = token_dropout_prob

        # calculate patching related stuff

        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'
        assert divisible_by(max_frames, frame_patch_size)

        patch_frame_dim, patch_height_dim, patch_width_dim = (max_frames // frame_patch_size), (image_height // patch_size), (image_width // patch_size)

        patch_dim = channels * (patch_size ** 2) * frame_patch_size

        self.channels = channels
        self.patch_size = patch_size
        self.to_patches = Rearrange('c (f pf) (h p1) (w p2) -> f h w (c p1 p2 pf)', p1 = patch_size, p2 = patch_size, pf = frame_patch_size)

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embed_frame = nn.Parameter(torch.zeros(patch_frame_dim, dim))
        self.pos_embed_height = nn.Parameter(torch.zeros(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.zeros(patch_width_dim, dim))

        # register tokens

        self.register_tokens = nn.Parameter(torch.zeros(num_registers, dim))

        nn.init.normal_(self.pos_embed_frame, std = 0.02)
        nn.init.normal_(self.pos_embed_height, std = 0.02)
        nn.init.normal_(self.pos_embed_width, std = 0.02)
        nn.init.normal_(self.register_tokens, std = 0.02)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, qk_rmsnorm)

        # final attention pooling queries

        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        # output to logits

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim, bias = False),
            nn.Linear(dim, num_classes, bias = False)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        volumes: List[Tensor], # different resolution images / CT scans
    ):
        batch, device = len(volumes), self.device
        arange = partial(torch.arange, device = device)

        assert all([volume.ndim == 4 and volume.shape[0] == self.channels for volume in volumes]), f'all volumes must have {self.channels} channels and number of dimensions of {self.channels} (channels, frame, height, width)'

        all_patches = [self.to_patches(volume) for volume in volumes]

        # prepare factorized positional embedding height width indices

        positions = []

        for patches in all_patches:
            patch_frame, patch_height, patch_width = patches.shape[:3]
            fhw_indices = torch.stack(torch.meshgrid((arange(patch_frame), arange(patch_height), arange(patch_width)), indexing = 'ij'), dim = -1)
            fhw_indices = rearrange(fhw_indices, 'f h w c -> (f h w) c')

            positions.append(fhw_indices)

        # need the sizes to compute token dropout + positional embedding

        tokens = [rearrange(patches, 'f h w d -> (f h w) d') for patches in all_patches]

        # handle token dropout

        seq_lens = torch.tensor([i.shape[0] for i in tokens], device = device)

        if self.training and self.token_dropout_prob > 0:

            keep_seq_lens = ((1. - self.token_dropout_prob) * seq_lens).int().clamp(min = 1)

            kept_tokens = []
            kept_positions = []

            for one_image_tokens, one_image_positions, seq_len, num_keep in zip(tokens, positions, seq_lens, keep_seq_lens):
                keep_indices = torch.randn((seq_len,), device = device).topk(num_keep, dim = -1).indices

                one_image_kept_tokens = one_image_tokens[keep_indices]
                one_image_kept_positions = one_image_positions[keep_indices]

                kept_tokens.append(one_image_kept_tokens)
                kept_positions.append(one_image_kept_positions)

            tokens, positions, seq_lens = kept_tokens, kept_positions, keep_seq_lens

        # add all height and width factorized positions


        frame_indices, height_indices, width_indices = torch.cat(positions).unbind(dim = -1)
        frame_embed, height_embed, width_embed = self.pos_embed_frame[frame_indices], self.pos_embed_height[height_indices], self.pos_embed_width[width_indices]

        pos_embed = frame_embed + height_embed + width_embed

        tokens = torch.cat(tokens)

        # linear projection to patch embeddings

        tokens = self.to_patch_embedding(tokens)

        # absolute positions

        tokens = tokens + pos_embed

        # add register tokens

        tokens = tokens.split(seq_lens.tolist())

        tokens = [torch.cat((self.register_tokens, one_tokens)) for one_tokens in tokens]

        # use nested tensor for transformers and save on padding computation

        tokens = nested_tensor(tokens, layout = torch.jagged, device = device)

        # embedding dropout

        tokens = self.dropout(tokens)

        # transformer

        tokens = self.transformer(tokens)

        # attention pooling
        # will use a jagged tensor for queries, as SDPA requires all inputs to be jagged, or not

        attn_pool_queries = [rearrange(self.attn_pool_queries, '... -> 1 ...')] * batch

        attn_pool_queries = nested_tensor(attn_pool_queries, layout = torch.jagged)

        pooled = self.attn_pool(attn_pool_queries, tokens)

        # back to unjagged

        logits = torch.stack(pooled.unbind())

        logits = rearrange(logits, 'b 1 d -> b d')

        logits = self.to_latent(logits)

        return self.mlp_head(logits)

# quick test

if __name__ == '__main__':

    # works for torch 2.5

    v = NaViT(
        image_size = 256,
        max_frames = 8,
        patch_size = 32,
        frame_patch_size = 2,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.,
        emb_dropout = 0.,
        token_dropout_prob = 0.1
    )

    # 5 volumetric data (videos or CT scans) of different resolutions - List[Tensor]

    volumes = [
        torch.randn(3, 2, 256, 256), torch.randn(3, 8, 128, 128),
        torch.randn(3, 4, 128, 256), torch.randn(3, 2, 256, 128),
        torch.randn(3, 4, 64, 256)
    ]

    assert v(volumes).shape == (5, 1000)

    v(volumes).sum().backward()
