# LeVJEPA, generalized to an arbitrary number of dimensions
# https://arxiv.org/abs/2608.27395, https://arxiv.org/abs/2511.08544

from collections import namedtuple
from math import pi, prod

import torch
import torch.nn.functional as F
from torch import arange, cat, einsum, nn, stack
from torch.nn import Module, ModuleList
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from vit_pytorch.vit_nd_masking import flex_attention_supported

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def cast_tuple(t, length = 1):
    return tuple(t) if isinstance(t, (tuple, list)) else ((t,) * length)

# sigreg loss - https://arxiv.org/abs/2511.08544

def sigreg_loss(x, num_proj = 1024, knots = 17):
    x = x.float() if x.dtype != torch.float64 else x
    batch, dim, device, dtype = *x.shape[-2:], x.device, x.dtype

    rand_projs = F.normalize(torch.randn(dim, num_proj, device = device, dtype = dtype), dim = 0)

    t = torch.linspace(0., 3., knots, device = device, dtype = dtype)
    phi = (-0.5 * t.square()).exp()

    x_t = einsum('... n d, d m -> ... n m', x, rand_projs)
    x_t = x_t[..., None] * t

    ecf_cos, ecf_sin = x_t.cos().mean(dim = -3), x_t.sin().mean(dim = -3)
    err = (ecf_cos - phi).square() + ecf_sin.square()

    return torch.trapezoid(err * phi, t, dim = -1).mean() * batch * 2

# nd block-causal masking
# CLS attends to all; patches are causal along each axis in `causal_dims` (q_coord >= k_coord)

def causal_condition(q_coord, k_coord):
    return (q_coord == -1) | ((k_coord >= 0) & (q_coord >= k_coord))

def nd_causal_mask(q_coord, k_coord, causal_dims):
    mask = causal_condition(q_coord[..., causal_dims[0]], k_coord[..., causal_dims[0]])
    for d in causal_dims[1:]:
        mask = mask & causal_condition(q_coord[..., d], k_coord[..., d])
    return mask

def create_nd_block_causal_mask_fn(coords, causal_dims):
    def block_fn(batch, heads, query_idx, kv_idx):
        return nd_causal_mask(coords[batch, query_idx], coords[batch, kv_idx], causal_dims)

    return block_fn

def create_dense_nd_mask(coords, causal_dims):
    return nd_causal_mask(coords[:, :, None], coords[:, None, :], causal_dims)

def nd_attention(q, k, v, block_mask = None, dense_mask = None, scale = None):
    if exists(block_mask):
        return flex_attention(q, k, v, block_mask = block_mask, scale = scale)

    attn_mask = dense_mask[:, None] if exists(dense_mask) else None
    return F.scaled_dot_product_attention(q, k, v, attn_mask = attn_mask, scale = scale)

# factorized nd rotary

class NDRotaryEmbedding(Module):
    def __init__(self, dim, ndim, max_freq = 10.):
        super().__init__()
        num_freq = dim // (2 * ndim)
        assert num_freq >= 1, f'dim_head ({dim}) too small for rotary on {ndim} dimensions'

        scales = torch.linspace(1., max_freq / 2, num_freq)
        self.register_buffer('scales', scales)

    def forward(self, coords):
        # coords: [b, n, ndim] in [-1, 1]

        theta = coords[..., None] * self.scales * pi
        theta = rearrange(theta, 'b n i j -> b n (i j)')

        sin, cos = theta.sin(), theta.cos()
        return tuple(repeat(t, 'b n d -> b 1 n (d j)', j = 2) for t in (sin, cos))

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary(t, sin, cos):
    dim_rotary = sin.shape[-1]
    t_rot, t_pass = t[..., :dim_rotary], t[..., dim_rotary:]
    t_rot = (t_rot * cos) + (rotate_every_two(t_rot) * sin)
    return cat((t_rot, t_pass), dim = -1).type_as(t)

# attention

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x, rotary_pos_emb, block_mask = None, dense_mask = None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = (rearrange(t, 'b n (h d) -> b h n d', h = self.heads) for t in qkv)

        sin, cos = rotary_pos_emb
        q, k = apply_rotary(q, sin, cos), apply_rotary(k, sin, cos)

        out = nd_attention(q, k, v, block_mask = block_mask, dense_mask = dense_mask, scale = self.scale)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, rotary_pos_emb, block_mask = None, dense_mask = None):
        for attn, ff in self.layers:
            x = attn(x, rotary_pos_emb, block_mask = block_mask, dense_mask = dense_mask) + x
            x = ff(x) + x
        return x

# nd transformer encoder

class NDTransformer(Module):
    def __init__(
        self,
        *,
        ndim,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        input_shape = None,
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        rotary_max_freq = 10.,
        causal_dims = (0,)
    ):
        super().__init__()
        assert 1 <= ndim <= 7, 'ndim must be between 1 and 7'

        patch_size = cast_tuple(patch_size, ndim)

        if exists(input_shape):
            for i, (inp_dim, patch_dim) in enumerate(zip(cast_tuple(input_shape, ndim), patch_size)):
                assert divisible_by(inp_dim, patch_dim), f'input dimension {i} ({inp_dim}) must be divisible by patch size ({patch_dim})'

        causal_dims = cast_tuple(causal_dims) if exists(causal_dims) else ()
        assert all(0 <= i < ndim for i in causal_dims), 'invalid causal dimension'

        self.ndim = ndim
        self.dim = dim
        self.patch_size = patch_size
        self.causal_dims = causal_dims

        patch_dim = channels * prod(patch_size)

        dim_names = 'fghijkl'[:ndim]
        patch_names = [f'p{i}' for i in range(ndim)]

        input_pattern = 'b c ' + ' '.join(f'({d} p{i})' for i, d in enumerate(dim_names))
        output_pattern = f"b {' '.join(dim_names)} (c {' '.join(patch_names)})"
        rearrange_kwargs = {f'p{i}': p for i, p in enumerate(patch_size)}

        self.to_patch_embedding = nn.Sequential(
            Rearrange(f'{input_pattern} -> {output_pattern}', **rearrange_kwargs),
            nn.LayerNorm(patch_dim, bias = False),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim, bias = False)
        )

        self.cls_token = nn.Parameter(torch.randn(dim))
        self.rotary_emb = NDRotaryEmbedding(dim_head, ndim, max_freq = rotary_max_freq)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout = dropout)
        self.norm = nn.RMSNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x, drop_ratio = None):
        assert x.ndim == self.ndim + 2, f'expected {self.ndim + 2}D input tensor, got {x.ndim}D'

        for i, (inp_dim, patch_dim) in enumerate(zip(x.shape[2:], self.patch_size)):
            assert divisible_by(inp_dim, patch_dim), f'input dimension {i} ({inp_dim}) must be divisible by patch size ({patch_dim})'

        batch, device = x.shape[0], x.device

        x = self.to_patch_embedding(x)
        patch_shape = x.shape[1:-1]
        x = self.dropout(rearrange(x, 'b ... d -> b (...) d'))

        # nd coordinates: normalized [-1, 1] for rotary, raw integer for causal masking

        grid = stack(torch.meshgrid(*(arange(n, device = device) for n in patch_shape), indexing = 'ij'), dim = -1)
        grid = rearrange(grid, '... i -> (...) i')

        coords_norm = (grid + 0.5) / torch.tensor(patch_shape, dtype = torch.float, device = device)
        coords_norm = repeat(coords_norm * 2. - 1., 'n i -> b n i', b = batch)
        coords_raw = repeat(grid, 'n i -> b n i', b = batch)

        # uniform random token dropping (training only)

        if exists(drop_ratio) and drop_ratio > 0. and self.training:
            assert 0. <= drop_ratio < 1.
            num_tokens, dim = x.shape[1], x.shape[-1]
            num_keep = max(1, int(num_tokens * (1 - drop_ratio)))

            indices = torch.rand(batch, num_tokens, device = device).topk(num_keep, dim = -1).indices

            x = x.gather(1, repeat(indices, 'b n -> b n d', d = dim))
            coords_norm = coords_norm.gather(1, repeat(indices, 'b n -> b n i', i = self.ndim))
            coords_raw = coords_raw.gather(1, repeat(indices, 'b n -> b n i', i = self.ndim))

        # prepend CLS token (never dropped, norm coord 0, raw coord -1)

        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = batch)
        x = cat((cls_tokens, x), dim = 1)
        coords_norm = cat((torch.zeros(batch, 1, self.ndim, device = device), coords_norm), dim = 1)
        coords_raw = cat((torch.full((batch, 1, self.ndim), -1, dtype = grid.dtype, device = device), coords_raw), dim = 1)

        seq_len = x.shape[1]

        # block-causal mask along causal_dims

        block_mask, dense_mask = None, None

        if len(self.causal_dims) > 0:
            if flex_attention_supported(device):
                mask_fn = create_nd_block_causal_mask_fn(coords_raw, self.causal_dims)
                block_mask = create_block_mask(mask_fn, batch, None, seq_len, seq_len, device = device)
            else:
                dense_mask = create_dense_nd_mask(coords_raw, self.causal_dims)

        rotary_pos_emb = self.rotary_emb(coords_norm)

        x = self.transformer(x, rotary_pos_emb, block_mask = block_mask, dense_mask = dense_mask)
        return self.norm(x)

# byol works even without batch statistics - richemond et al. https://arxiv.org/abs/2010.10241

class WeightStandardizedLinear(Module):
    def __init__(self, dim, dim_out, bias = True, eps = 1e-4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(dim_out, dim) * dim ** -0.5)

        self.bias = nn.Parameter(torch.zeros(dim_out)) if bias else None

    def forward(self, x):
        w, bias, eps = self.weight, self.bias, self.eps

        mean = w.mean(dim = -1, keepdim = True)
        var = w.var(dim = -1, keepdim = True, unbiased = False)

        w = (w - mean) / var.clamp_min(eps).sqrt()

        return F.linear(x, w, bias)

# main class - LeVJEPA for any number of dimensions

LeVJEPANDLosses = namedtuple('LeVJEPANDLosses', ['invariance_loss', 'sigreg_loss'])

class LeVJEPAND(Module):
    def __init__(
        self,
        encoder,
        *,
        input_shape = None,
        augment_src = None,
        augment_tgt = None,
        num_target_aug_views = 1,
        num_classes_K = 256,
        projection_hidden = 256,
        drop_ratio = 0.95,
        use_batch_norm = True,
        norm_fn = None,
        use_weight_standardization = False,
        invariance_loss_weight = 1.,
        sigreg_loss_weight = 0.02,
        sigreg_loss_kwargs = dict(num_proj = 1024, knots = 17)
    ):
        super().__init__()
        assert num_target_aug_views >= 1

        self.encoder = encoder
        self.augment_src = default(augment_src, nn.Identity())
        self.augment_tgt = default(augment_tgt, nn.Identity())

        norm_fn = default(norm_fn, nn.BatchNorm1d if use_batch_norm else nn.RMSNorm)
        linear = WeightStandardizedLinear if use_weight_standardization else nn.Linear

        self.projector = nn.Sequential(
            linear(encoder.dim, projection_hidden),
            norm_fn(projection_hidden),
            nn.GELU(),
            nn.Linear(projection_hidden, num_classes_K)
        )

        self.num_target_aug_views = num_target_aug_views
        self.drop_ratio = drop_ratio

        self.invariance_loss_weight = invariance_loss_weight
        self.sigreg_loss_weight = sigreg_loss_weight
        self.sigreg_loss_kwargs = sigreg_loss_kwargs

    def forward(self, x, return_embedding = False, return_loss_breakdown = False):
        if return_embedding:
            return self.encoder(x)[:, 0]

        batch = x.shape[0]

        src_view = self.augment_src(x)
        tgt_views = cat([self.augment_tgt(x) for _ in range(self.num_target_aug_views)], dim = 0)

        src_cls = self.encoder(src_view, drop_ratio = self.drop_ratio)[:, 0]
        tgt_cls = self.encoder(tgt_views, drop_ratio = self.drop_ratio)[:, 0]

        cls_tokens = cat((
            rearrange(src_cls, 'b d -> b 1 d'),
            rearrange(tgt_cls, '(v b) d -> b v d', b = batch)
        ), dim = 1)

        projections = self.projector(rearrange(cls_tokens, 'b v d -> (b v) d'))
        projections = rearrange(projections, '(b v) d -> b v d', b = batch)
        proj_src = projections[:, :1]

        # invariance loss

        invariance_loss = F.mse_loss(projections, proj_src.expand_as(projections))

        # sigreg loss

        sigreg = sigreg_loss(rearrange(projections, 'b v d -> v b d'), **self.sigreg_loss_kwargs)

        # total loss

        invariance_loss = invariance_loss * self.invariance_loss_weight
        sigreg = sigreg * self.sigreg_loss_weight
        total_loss = invariance_loss + sigreg

        if not return_loss_breakdown:
            return total_loss

        return total_loss, LeVJEPANDLosses(invariance_loss, sigreg)

# quick run

if __name__ == '__main__':
    augment = lambda x: x + torch.randn_like(x) * 0.1

    # 3d video (b c t h w)

    encoder_3d = NDTransformer(
        ndim = 3,
        input_shape = (8, 64, 64),
        patch_size = (1, 16, 16),
        dim = 128,
        depth = 2,
        heads = 4,
        mlp_dim = 256
    )

    learner_3d = LeVJEPAND(encoder_3d, augment_src = augment, augment_tgt = augment)

    video = torch.randn(2, 3, 8, 64, 64)
    loss_3d = learner_3d(video)
    loss_3d.backward()

    embed = learner_3d(video, return_embedding = True)
    assert embed.shape == (2, 128)

    # 4d volumes (b c t x y z)

    encoder_4d = NDTransformer(
        ndim = 4,
        input_shape = (8, 16, 32, 32),
        patch_size = (1, 4, 8, 8),
        dim = 128,
        depth = 2,
        heads = 4,
        mlp_dim = 256
    )

    learner_4d = LeVJEPAND(encoder_4d, augment_src = augment, augment_tgt = augment)

    volumetric = torch.randn(2, 3, 8, 16, 32, 32)
    loss_4d = learner_4d(volumetric)
    loss_4d.backward()
