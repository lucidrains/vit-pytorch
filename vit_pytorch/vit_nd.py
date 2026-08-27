from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import Module

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from vit_pytorch.vit_nd_masking import get_nd_causal_mask_fn, create_nd_block_mask, nd_attention

# helpers

def exists(val):
    return val is not None

def join(arr, delimiter = ' '):
    return delimiter.join(arr)

def ensure_tuple(t, length):
    if isinstance(t, (tuple, list)):
        assert len(t) == length, f'Expected tuple of length {length}, got {len(t)}'
        return tuple(t)
    return (t,) * length

# classes

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, block_mask = None, mask_fn = None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        out = nd_attention(q, k, v, mask_fn = mask_fn, block_mask = block_mask, scale = self.scale)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, block_mask = None, mask_fn = None):
        for attn, ff in self.layers:
            x = attn(x, block_mask, mask_fn) + x
            x = ff(x) + x
        return self.norm(x)

class ViTND(Module):
    def __init__(
        self,
        *,
        ndim: int,
        input_shape: int | tuple[int, ...],
        patch_size: int | tuple[int, ...],
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = 'cls',
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.,
        emb_dropout: float = 0.,
        causal_dims: int | tuple[int, ...] | None = None
    ):
        super().__init__()

        assert 1 <= ndim <= 7, 'ndim must be between 1 and 7'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.ndim = ndim
        self.pool = pool

        input_shape = ensure_tuple(input_shape, ndim)
        patch_size = ensure_tuple(patch_size, ndim)

        for i, (inp_dim, patch_dim) in enumerate(zip(input_shape, patch_size)):
            assert inp_dim % patch_dim == 0, f'Input dimension {i} ({inp_dim}) must be divisible by patch size ({patch_dim})'

        num_patches_per_dim = [inp_dim // patch_dim for inp_dim, patch_dim in zip(input_shape, patch_size)]
        self.num_patches_per_dim = num_patches_per_dim
        num_patches = 1
        for n in num_patches_per_dim:
            num_patches *= n

        patch_dim = channels
        for p in patch_size:
            patch_dim *= p

        dim_names = 'fghijkl'[:ndim]
        self.dim_names = dim_names

        input_dims = [f'({d} p{i})' for i, d in enumerate(dim_names)]
        patch_dims = [f'p{i}' for i in range(ndim)]

        input_pattern = f'b c {join(input_dims)}'
        output_pattern = f'b ({join(dim_names)}) ({join(patch_dims)} c)'
        rearrange_str = f'{input_pattern} -> {output_pattern}'

        rearrange_kwargs = {f'p{i}': p for i, p in enumerate(patch_size)}

        self.to_patch_embedding = nn.Sequential(
            Rearrange(rearrange_str, **rearrange_kwargs),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        nd_patches_pattern = f'b ({join(dim_names)}) d -> b d {join(dim_names)}'
        self.to_nd_patches = Rearrange(nd_patches_pattern, **dict(zip(dim_names, num_patches_per_dim)))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # causal masking along one or more dimensions

        self.causal_mask_fn = get_nd_causal_mask_fn(num_patches_per_dim, causal_dims, ndim = ndim, has_cls = True)

        self.heads = heads
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x: Tensor, return_nd_patches: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        x = self.to_patch_embedding(x)
        b, _, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x, ps = pack((cls_tokens, x), 'b * d')

        x = x + repeat(self.pos_embedding, '1 n d -> b n d', b = b)
        x = self.dropout(x)

        block_mask = None
        mask_fn = self.causal_mask_fn

        if exists(self.causal_mask_fn):
            seq_len = x.shape[1]
            block_mask = create_nd_block_mask(self.causal_mask_fn, self.heads, seq_len, device = x.device)

        x = self.transformer(x, block_mask, mask_fn)

        # unpack cls token and nd patches

        cls_token, patches = unpack(x, ps, 'b * d')
        cls_token = rearrange(cls_token, 'b 1 d -> b d')

        # return cls token and nd patch embeddings separately

        if return_nd_patches:
            patches = self.to_nd_patches(patches)

            return cls_token, patches

        # pool to single token and classify

        if self.pool == 'mean':
            pooled = reduce(patches, 'b n d -> b d', 'mean')
        else:
            pooled = cls_token

        pooled = self.to_latent(pooled)
        logits = self.mlp_head(pooled)

        return logits


if __name__ == '__main__':

    model = ViTND(
        ndim = 4,
        input_shape = (8, 16, 32, 64),
        patch_size = (2, 4, 4, 8),
        num_classes = 1000,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        channels = 3,
        dropout = 0.1,
        emb_dropout = 0.1,
        causal_dims = (0,) # time dimension is causal
    )

    occupancy_time = torch.randn(2, 3, 8, 16, 32, 64)

    if torch.cuda.is_available():
        model = torch.compile(model)

    logits = model(occupancy_time)
    print(logits.shape) # (2, 1000)

    cls_token, patch_embeddings = model(occupancy_time, return_nd_patches = True)
    print(cls_token.shape) # (2, 512)
    print(patch_embeddings.shape) # (2, 512, 4, 4, 8, 8) - nd patch dims, cls token spliced out
