# LeVJEPA, generalized number of dimensions
# https://arxiv.org/abs/2608.27395, https://arxiv.org/abs/2511.08544

from __future__ import annotations

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

def ensure_tuple(t, length):
    return tuple(t) if isinstance(t, (tuple, list)) else (t,) * length

# sigreg loss - https://arxiv.org/abs/2511.08544

def sigreg_loss(x, num_proj = 1024, knots = 17):
    device = dict(device = x.device)
    batch, dim = x.shape[-2], x.shape[-1]

    rand_projs = torch.randn(dim, num_proj, **device)
    rand_projs = rand_projs / rand_projs.norm(dim = 0)

    t = torch.linspace(0., 3., knots, **device)
    phi = (-0.5 * t.square()).exp()

    x_t = einsum('... n d, d m -> ... n m', x, rand_projs)
    x_t = x_t[..., None] * t

    ecf_cos, ecf_sin = x_t.cos().mean(-3), x_t.sin().mean(-3)

    err = (ecf_cos - phi).square() + ecf_sin.square()
    return torch.trapezoid(err * phi, t, dim = -1).mean() * batch * 2

# nd block-causal masking
# CLS attends to all; patches are causal along each axis in `causal_dims` (q_coord >= k_coord)

def causal_condition(q_coord, k_coord):
    return (q_coord == -1) | ((k_coord >= 0) & (q_coord >= k_coord))

def create_nd_block_causal_mask_fn(coords, causal_dims):
    # coords: [b, n, ndim] raw patch positions, cls row is all -1

    def block_fn(batch, heads, query_idx, kv_idx):
        q_coord = coords[batch, query_idx]
        k_coord = coords[batch, kv_idx]

        mask = torch.ones(query_idx.shape, dtype = torch.bool, device = coords.device)

        for causal_dim in causal_dims:
            mask = mask & causal_condition(q_coord[..., causal_dim], k_coord[..., causal_dim])

        return mask

    return block_fn

def dense_nd_mask(coords, causal_dims):
    batch, n, _ = coords.shape

    q_coord = coords[:, :, None, :]
    k_coord = coords[:, None, :, :]

    mask = torch.ones(batch, n, n, dtype = torch.bool, device = coords.device)

    for causal_dim in causal_dims:
        mask = mask & causal_condition(q_coord[..., causal_dim], k_coord[..., causal_dim])

    return mask

# attention - flex when a block mask is available, dense masked fallback everywhere else

def nd_attention(q, k, v, block_mask = None, dense_mask = None, scale = None):
    if exists(block_mask):
        try:
            return flex_attention(q, k, v, block_mask = block_mask, scale = scale)
        except Exception:
            pass

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    if exists(dense_mask):
        scores = scores.masked_fill(~dense_mask[:, None], -torch.finfo(scores.dtype).max)

    return torch.matmul(scores.softmax(dim = -1), v)

# factorized nd rotary embedding, generalization of the video rotary in levjepa

class NDRotaryEmbedding(Module):
    def __init__(self, dim, ndim, max_freq = 10.):
        super().__init__()
        num_freq = (dim - dim % (2 * ndim)) // (2 * ndim)
        assert num_freq >= 1, f'dim ({dim}) too small for rotary on {ndim} dimensions'

        scales = torch.linspace(1., max_freq / 2, num_freq)
        self.register_buffer('scales', scales)

    def forward(self, coords):
        # coords: [b, n, ndim] in [-1, 1], cls row all zeros

        theta = coords[..., None] * self.scales * pi
        theta = rearrange(theta, 'b n i j -> b n (i j)')

        sin, cos = theta.sin(), theta.cos()
        return (repeat(t, 'b n d -> b n (d j)', j = 2) for t in (sin, cos))

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary(q, k, sin, cos):
    sin, cos = sin[:, None], cos[:, None]
    dim_rotary = sin.shape[-1]

    q_rot, q_pass = q[..., :dim_rotary], q[..., dim_rotary:]
    k_rot, k_pass = k[..., :dim_rotary], k[..., dim_rotary:]

    q_rot = (q_rot * cos) + (rotate_every_two(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_every_two(k_rot) * sin)

    return cat((q_rot, q_pass), dim = -1), cat((k_rot, k_pass), dim = -1)

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

    def forward(self, x, rotary_emb, pos, block_mask, dense_mask):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = (rearrange(t, 'b n (h d) -> b h n d', h = self.heads) for t in qkv)

        if exists(rotary_emb):
            sin, cos = rotary_emb(pos)
            q, k = apply_rotary(q, k, sin, cos)

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

    def forward(self, x, rotary_emb, pos, block_mask, dense_mask):
        for attn, ff in self.layers:
            x = attn(x, rotary_emb, pos, block_mask, dense_mask) + x
            x = ff(x) + x

        return x

# main transformer - patch dims channel-first to match the original video implementation, tokens row-major across dims

class NDTransformer(Module):
    def __init__(
        self,
        *,
        ndim,
        input_shape,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        rotary_max_freq = 10.,
        causal_dims = (0,),
    ):
        super().__init__()

        assert 1 <= ndim <= 7, 'ndim must be between 1 and 7'

        input_shape = ensure_tuple(input_shape, ndim)
        patch_size = ensure_tuple(patch_size, ndim)

        for i, (inp_dim, patch_dim) in enumerate(zip(input_shape, patch_size)):
            assert (inp_dim % patch_dim) == 0, f'input dimension {i} ({inp_dim}) must be divisible by patch size ({patch_dim})'

        causal_dims = (causal_dims,) if isinstance(causal_dims, int) else tuple(causal_dims)
        assert all(0 <= i < ndim for i in causal_dims), 'invalid causal dimension'

        self.ndim = ndim
        self.dim = dim
        self.heads = heads
        self.causal_dims = causal_dims

        self.num_patches_per_dim = tuple(inp_dim // patch_dim for inp_dim, patch_dim in zip(input_shape, patch_size))
        self.num_patches = prod(self.num_patches_per_dim)

        patch_dim = channels * prod(patch_size)

        # nd patch embedding

        dim_names = 'fghijkl'[:ndim]
        patch_names = [f'p{i}' for i in range(ndim)]

        input_pattern = f'b c ' + ' '.join(f'({d} p{i})' for i, d in enumerate(dim_names))
        output_pattern = f'b ' + ' '.join(dim_names) + f' (c ' + ' '.join(patch_names) + ')'
        rearrange_kwargs = {f'p{i}': p for i, p in enumerate(patch_size)}

        self.to_patch_embedding = nn.Sequential(
            Rearrange(f'{input_pattern} -> {output_pattern}', **rearrange_kwargs),
            nn.LayerNorm(patch_dim, bias = False),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim, bias = False)
        )

        self.cls_token = nn.Parameter(torch.randn(dim))
        self.rotary_emb = NDRotaryEmbedding(dim_head, ndim, max_freq = rotary_max_freq)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.norm = nn.RMSNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x, drop_ratio = None):
        batch = x.shape[0]
        device = dict(device = x.device)

        x = self.to_patch_embedding(x).reshape(batch, self.num_patches, self.dim)

        # nd coordinates: normalized [-1, 1] for rotary, raw (cls row -1) for causal masking

        grid = stack(torch.meshgrid(*(arange(n, **device) for n in self.num_patches_per_dim), indexing = 'ij'), dim = -1).reshape(-1, self.ndim)

        coords_norm = grid / torch.tensor(self.num_patches_per_dim, dtype = torch.float, **device)
        coords_norm = repeat((coords_norm - 0.5) * 2, 'n i -> b n i', b = batch)
        coords_norm = cat((torch.zeros(batch, 1, self.ndim, **device), coords_norm), dim = 1)

        coords_raw = repeat(grid.float(), 'n i -> b n i', b = batch)
        coords_raw = cat((torch.full((batch, 1, self.ndim), -1, **device), coords_raw), dim = 1)

        # CLS token, never dropped

        x = cat((repeat(self.cls_token, 'd -> b 1 d', b = batch), self.dropout(x)), dim = 1)

        # uniform random token dropping (training only)

        if exists(drop_ratio) and self.training:
            total_tokens, dim = x.shape[1] - 1, x.shape[-1]
            num_keep = max(1, int(total_tokens * (1 - drop_ratio)))

            rand = torch.rand(batch, total_tokens, **device)
            patch_indices = rand.topk(num_keep, dim = -1).indices + 1
            indices = cat((torch.zeros(batch, 1, dtype = patch_indices.dtype, **device), patch_indices), dim = 1)

            x = torch.gather(x, 1, repeat(indices, 'b n -> b n d', d = dim))
            coords_norm = torch.gather(coords_norm, 1, repeat(indices, 'b n -> b n i', i = self.ndim))
            coords_raw = torch.gather(coords_raw, 1, repeat(indices, 'b n -> b n i', i = self.ndim))

        seq_len = x.shape[1]

        # always block-causal - flex block mask when supported, dense fallback elsewhere

        mask_fn = create_nd_block_causal_mask_fn(coords_raw, self.causal_dims)

        block_mask, dense_mask = None, None

        if flex_attention_supported(x.device):
            try:
                block_mask = create_block_mask(mask_fn, batch, self.heads, seq_len, seq_len, **device)
            except Exception:
                block_mask = None

        if not exists(block_mask):
            dense_mask = dense_nd_mask(coords_raw, self.causal_dims)

        x = self.transformer(x, self.rotary_emb, coords_norm, block_mask, dense_mask)
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
        input_shape,
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
        sigreg_loss_kwargs = dict(num_proj = 1024, knots = 17),
    ):
        super().__init__()

        input_shape = ensure_tuple(input_shape, encoder.ndim)

        self.encoder = encoder
        self.augment_src = default(augment_src, nn.Identity())
        self.augment_tgt = default(augment_tgt, nn.Identity())

        # small projector, discarded after pretraining

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
        dims = ' '.join('fghijkl'[:self.encoder.ndim])

        # source and target views, augmented independently

        src_view = self.augment_src(x)
        tgt_views = self.augment_tgt(repeat(x, f'b c {dims} -> (b v) c {dims}', v = self.num_target_aug_views))

        projections = self.projector(self.encoder(cat((src_view, tgt_views), dim = 0), drop_ratio = self.drop_ratio)[:, 0])
        projection = rearrange(projections, '(b v) d -> b v d', b = batch, v = self.num_target_aug_views + 1)

        # invariance loss

        invariance_loss = F.mse_loss(projection, projection[:, :1].expand_as(projection))

        # sigreg loss

        sigreg = sigreg_loss(rearrange(projection, 'b v d -> v b d'), **self.sigreg_loss_kwargs)

        # total loss

        invariance_loss = invariance_loss * self.invariance_loss_weight
        sigreg = sigreg * self.sigreg_loss_weight
        total_loss = invariance_loss + sigreg

        if not return_loss_breakdown:
            return total_loss

        return total_loss, LeVJEPANDLosses(invariance_loss, sigreg)

# quick run

if __name__ == '__main__':
    # works for any number of dimensions - 1d audio (b c t), 2d (b c t h), 3d video (b c t h w), 4d volumes (b c t x y z)

    # contrived augmentation for source and target views - random noise

    augment = lambda x: x + torch.randn_like(x) * 0.1

    # 3d video (b c t h w)

    learner_3d = LeVJEPAND(NDTransformer(ndim = 3, input_shape = (8, 64, 64), patch_size = (1, 16, 16), dim = 128, depth = 2, heads = 4, mlp_dim = 256), input_shape = (8, 64, 64), augment_src = augment, augment_tgt = augment)

    video = torch.randn(2, 3, 8, 64, 64)
    loss_3d = learner_3d(video)
    loss_3d.backward()

    embed = learner_3d(video, return_embedding = True)
    assert embed.shape == (2, 128)

    # 4d volumes (b c t x y z) - e.g. time series of ct / mri scans

    learner_4d = LeVJEPAND(NDTransformer(ndim = 4, input_shape = (8, 16, 32, 32), patch_size = (1, 4, 8, 8), dim = 128, depth = 2, heads = 4, mlp_dim = 256), input_shape = (8, 16, 32, 32), augment_src = augment, augment_tgt = augment)

    volumetric = torch.randn(2, 3, 8, 16, 32, 32)
    loss_4d = learner_4d(volumetric)
    loss_4d.backward()
