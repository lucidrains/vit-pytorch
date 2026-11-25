# Lukas Kuhn et al. - https://arxiv.org/abs/2608.27395

import random
from collections import namedtuple
from math import pi

import torch
import torch.nn.functional as F
from torch import arange, cat, einsum, nn, stack
from torch.nn import Module, ModuleList
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from torchvision import transforms as T

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from vit_pytorch.vit_nd_masking import flex_attention_supported

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(num, den):
    return (num % den) == 0

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# block causal mask - bidirectional within frame, causal across frames, CLS attends to all

def causal_condition(query_frame, key_frame):
    return (query_frame == -1) | ((key_frame >= 0) & (query_frame >= key_frame))

def create_block_causal_mask_fn(frame_idx):
    def block_fn(batch, heads, query_idx, kv_idx):
        return causal_condition(frame_idx[batch, query_idx], frame_idx[batch, kv_idx])

    return block_fn

def create_dense_block_causal_mask(frame_idx):
    return causal_condition(frame_idx[:, :, None], frame_idx[:, None, :])

def video_attention(q, k, v, block_mask = None, dense_mask = None, scale = None):
    if exists(block_mask):
        return flex_attention(q, k, v, block_mask = block_mask, scale = scale)

    attn_mask = dense_mask[:, None] if exists(dense_mask) else None
    return F.scaled_dot_product_attention(q, k, v, attn_mask = attn_mask, scale = scale)

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

# augmentation utils

class RandomApply(Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

def augment_video(video, net):
    frames = [net(rearrange(v, 'c t h w -> t c h w')) for v in video]
    return rearrange(stack(frames), 'b t c h w -> b c t h w')

class NormalizeVideo(Module):
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):
        super().__init__()
        self.normalize = T.Normalize(mean, std)

    def forward(self, video):
        batch = video.shape[0]
        frames = rearrange(video, 'b c t h w -> (b t) c h w')
        return rearrange(self.normalize(frames), '(b t) c h w -> b c t h w', b = batch)

class PhotometricAugment(Module):
    def __init__(
        self,
        color_jitter_prob = 0.8,
        grayscale_prob = 0.2,
        gaussian_blur_prob = 0.0,
        hflip = True,
        mean = (0.485, 0.456, 0.406),
        std = (0.229, 0.224, 0.225)
    ):
        super().__init__()
        self.net = Sequential(
            T.RandomHorizontalFlip(p = 0.5) if hflip else None,
            RandomApply(T.ColorJitter(0.4, 0.4, 0.2, 0.1), p = color_jitter_prob) if color_jitter_prob > 0 else None,
            T.RandomGrayscale(p = grayscale_prob) if grayscale_prob > 0 else None,
            RandomApply(T.GaussianBlur(9, (0.1, 2.0)), p = gaussian_blur_prob) if gaussian_blur_prob > 0 else None,
            T.Normalize(mean, std)
        )

    def forward(self, video):
        return augment_video(video, self.net)

class RandomVideoCrop(Module):
    def __init__(self, size, scale, ratio_range = (0.75, 1.3333)):
        super().__init__()
        self.crop = T.RandomResizedCrop(
            pair(size),
            scale = scale,
            ratio = ratio_range,
            interpolation = T.InterpolationMode.BICUBIC,
            antialias = True
        )

    def forward(self, video):
        return augment_video(video, self.crop)

# factorized 3d rotary

class VideoRotaryEmbedding(Module):
    def __init__(self, dim, max_freq = 10.):
        super().__init__()
        num_freq = dim // 6
        assert num_freq >= 1, f'dim_head ({dim}) must be at least 6 for 3d rotary'

        scales = torch.linspace(1., max_freq / 2, num_freq)
        self.register_buffer('scales', scales)

    def forward(self, coords):
        # coords: [b, n, 3] in [-1, 1]

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

        out = video_attention(q, k, v, block_mask = block_mask, dense_mask = dense_mask, scale = self.scale)
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

# video transformer encoder

class VideoTransformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        mlp_dim,
        image_size,
        patch_size,
        frames = None,
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        rotary_max_freq = 10.,
        block_causal = True
    ):
        super().__init__()
        image_h, image_w = pair(image_size)
        patch_h, patch_w = pair(patch_size)

        assert divisible_by(image_h, patch_h) and divisible_by(image_w, patch_w)

        patch_dim = channels * patch_h * patch_w

        self.dim = dim
        self.patch_h, self.patch_w = patch_h, patch_w
        self.block_causal = block_causal

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c f (h p1) (w p2) -> b (f h w) (c p1 p2)', p1 = patch_h, p2 = patch_w),
            nn.LayerNorm(patch_dim, bias = False),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim, bias = False)
        )

        self.cls_token = nn.Parameter(torch.randn(dim))
        self.rotary_emb = VideoRotaryEmbedding(dim_head, max_freq = rotary_max_freq)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout = dropout)
        self.norm = nn.RMSNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, video, drop_ratio = None):
        batch, _, frames, height, width = video.shape
        device = video.device

        assert divisible_by(height, self.patch_h) and divisible_by(width, self.patch_w)

        x = self.dropout(self.to_patch_embedding(video))
        tm, hm, wm = frames, height // self.patch_h, width // self.patch_w

        # factorized [frame, height, width] coordinates in [-1, 1] and frame indices

        grid = torch.cartesian_prod(
            arange(tm, device = device),
            arange(hm, device = device),
            arange(wm, device = device)
        )

        coords = (grid + 0.5) / torch.tensor([tm, hm, wm], dtype = torch.float, device = device)
        coords = repeat(coords * 2. - 1., 'n i -> b n i', b = batch)
        frame_idx = repeat(arange(tm, device = device), 't -> b (t s)', b = batch, s = hm * wm)

        # uniform random token dropping (training only)

        if exists(drop_ratio) and drop_ratio > 0. and self.training:
            assert 0. <= drop_ratio < 1.
            num_tokens, dim = x.shape[1], x.shape[-1]
            num_keep = max(1, int(num_tokens * (1 - drop_ratio)))

            indices = torch.rand(batch, num_tokens, device = device).topk(num_keep, dim = -1).indices

            x = x.gather(1, repeat(indices, 'b n -> b n d', d = dim))
            coords = coords.gather(1, repeat(indices, 'b n -> b n i', i = 3))
            frame_idx = frame_idx.gather(1, indices)

        # prepend CLS token (never dropped, coord 0, frame_idx -1)

        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = batch)
        x = cat((cls_tokens, x), dim = 1)
        coords = cat((torch.zeros(batch, 1, 3, device = device), coords), dim = 1)
        frame_idx = cat((torch.full((batch, 1), -1, device = device), frame_idx), dim = 1)

        seq_len = x.shape[1]

        # block-causal mask

        block_mask, dense_mask = None, None

        if self.block_causal:
            if flex_attention_supported(device):
                block_fn = create_block_causal_mask_fn(frame_idx)
                block_mask = create_block_mask(block_fn, batch, None, seq_len, seq_len, device = device)
            else:
                dense_mask = create_dense_block_causal_mask(frame_idx)

        rotary_pos_emb = self.rotary_emb(coords)

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

# main class - LeVJEPA

LeVJEPALosses = namedtuple('LeVJEPALosses', ['invariance_loss', 'sigreg_loss'])

class LeVJEPA(Module):
    def __init__(
        self,
        encoder,
        image_size,
        local_size = 96,
        num_local_views = 4,
        num_classes_K = 256,
        projection_hidden = 256,
        local_crop_scale = (0.02, 0.4),
        global_crop_scale = (0.8, 1.0),
        drop_ratio = 0.95,
        use_batch_norm = True,
        norm_fn = None,
        use_weight_standardization = False,
        invariance_loss_weight = 1.,
        sigreg_loss_weight = 0.02,
        sigreg_loss_kwargs = dict(num_proj = 1024, knots = 17),
        local_augment_fn = None,
        global_augment_fn = None,
        crop_ratio_range = (0.75, 1.3333)
    ):
        super().__init__()
        assert num_local_views >= 1

        self.encoder = encoder
        self.global_augment = default(global_augment_fn, NormalizeVideo())
        self.local_augment = default(local_augment_fn, PhotometricAugment())

        self.global_crop = RandomVideoCrop(image_size, scale = global_crop_scale, ratio_range = crop_ratio_range)
        self.local_crop = RandomVideoCrop(local_size, scale = local_crop_scale, ratio_range = crop_ratio_range)

        norm_fn = default(norm_fn, nn.BatchNorm1d if use_batch_norm else nn.RMSNorm)
        linear = WeightStandardizedLinear if use_weight_standardization else nn.Linear

        self.projector = nn.Sequential(
            linear(encoder.dim, projection_hidden),
            norm_fn(projection_hidden),
            nn.GELU(),
            nn.Linear(projection_hidden, num_classes_K)
        )

        self.num_local_views = num_local_views
        self.drop_ratio = drop_ratio

        self.invariance_loss_weight = invariance_loss_weight
        self.sigreg_loss_weight = sigreg_loss_weight
        self.sigreg_loss_kwargs = sigreg_loss_kwargs

    def forward(self, video, return_embedding = False, return_loss_breakdown = False):
        if return_embedding:
            return self.encoder(video)[:, 0]

        batch = video.shape[0]

        global_view = self.global_augment(self.global_crop(video))
        local_views = cat([self.local_augment(self.local_crop(video)) for _ in range(self.num_local_views)], dim = 0)

        global_cls = self.encoder(global_view, drop_ratio = self.drop_ratio)[:, 0]
        local_cls = self.encoder(local_views, drop_ratio = self.drop_ratio)[:, 0]

        cls_tokens = cat((
            rearrange(global_cls, 'b d -> b 1 d'),
            rearrange(local_cls, '(v b) d -> b v d', b = batch)
        ), dim = 1)

        projections = self.projector(rearrange(cls_tokens, 'b v d -> (b v) d'))
        projections = rearrange(projections, '(b v) d -> b v d', b = batch)
        proj_global = projections[:, :1]

        # invariance loss

        invariance_loss = F.mse_loss(projections, proj_global.expand_as(projections))

        # sigreg loss

        sigreg = sigreg_loss(rearrange(projections, 'b v d -> v b d'), **self.sigreg_loss_kwargs)

        # total loss

        invariance_loss = invariance_loss * self.invariance_loss_weight
        sigreg = sigreg * self.sigreg_loss_weight
        total_loss = invariance_loss + sigreg

        if not return_loss_breakdown:
            return total_loss

        return total_loss, LeVJEPALosses(invariance_loss, sigreg)

# quick run

if __name__ == '__main__':
    encoder = VideoTransformer(
        image_size = 128,
        patch_size = 16,
        frames = 8,
        dim = 256,
        depth = 3,
        heads = 8,
        mlp_dim = 512
    )

    learner = LeVJEPA(encoder, image_size = 128, local_size = 96)
    video = torch.randn(2, 3, 8, 128, 128)

    loss = learner(video)
    loss.backward()

    embed = learner(video, return_embedding = True)
    assert embed.shape == (2, 256)
