# https://openreview.net/forum?id=Co6SCyBIjo
# applied at https://arxiv.org/abs/2605.03269 - 50-85% jump in pick-place moving conveyer belt

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def l2norm(t):
    return F.normalize(t, dim = -1)

# normalization helpers

class ChanLayerNorm(Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + self.eps).rsqrt() * self.gamma

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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., use_flash_attn = True, causal = False):
        super().__init__()
        self.use_flash_attn = use_flash_attn
        self.dropout_p = dropout
        self.causal = causal
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def flash_attn(self, q, k, v, mask = None):
        is_causal = self.causal and q.shape[-2] > 1

        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout_p,
                is_causal = is_causal,
                scale = self.scale
            )
        return out

    def forward(self, x, mask = None, cache = None, return_cache = False):
        is_causal = self.causal and x.shape[-2] > 1
        assert not (is_causal and exists(mask)), 'causal attention is not compatible with key padding mask'

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        if exists(cache):
            ck, cv = cache
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')

        if self.use_flash_attn:
            out = self.flash_attn(q, k, v, mask = mask)
        else:
            dots = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale
            if exists(mask):
                dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

            if self.causal:
                i, j = dots.shape[-2:]
                causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(j - i + 1)
                dots = dots.masked_fill(causal_mask, -torch.finfo(dots.dtype).max)

            attn = self.attend(dots)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if not return_cache:
            return out

        return out, (k, v)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., use_flash_attn = True, causal = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, use_flash_attn = use_flash_attn, causal = causal),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, mask = None, cache = None, return_cache = False):
        new_caches = []
        cache = default(cache, (None,) * len(self.layers))

        for (attn, ff), layer_cache in zip(self.layers, cache):
            attn_out, next_cache = attn(x, mask = mask, cache = layer_cache, return_cache = True)
            new_caches.append(next_cache)

            x = attn_out + x
            x = ff(x) + x

        x = self.norm(x)

        if not return_cache:
            return x

        return x, tuple(new_caches)

# moss specific classes

class STSSEncoder(Module):
    def __init__(self, dim, local_time = 3, local_height = 3, local_width = 3, hidden_dim = 64):
        super().__init__()

        self.spatial_to_hidden = nn.Linear(local_height * local_width, hidden_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1),
            ChanLayerNorm(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1),
            ChanLayerNorm(hidden_dim),
            nn.GELU()
        )

        self.time_to_out = nn.Linear(local_time * hidden_dim, dim)

    def forward(self, sim):
        b, t, h, w, lt, lh, lw = sim.shape

        x = rearrange(sim, 'b t h w lt lh lw -> b t h w lt (lh lw)')
        x = self.spatial_to_hidden(x)

        x = rearrange(x, 'b t h w lt d -> (b t lt) d h w')
        x = self.conv(x)
        x = rearrange(x, '(b t lt) d h w -> b t h w (lt d)', b = b, t = t, lt = lt)

        return self.time_to_out(x)

class MOSS(Module):
    def __init__(
        self,
        dim,
        local_time = 3,
        local_height = 3,
        local_width = 3,
        hidden_dim = 64,
        orders = 2,
        causal = False
    ):
        super().__init__()
        assert is_odd(local_time) and is_odd(local_height) and is_odd(local_width), 'MOSS local dimensions must be odd'

        self.local_time = local_time
        self.local_height = local_height
        self.local_width = local_width
        self.causal = causal

        self.encoders = ModuleList([STSSEncoder(dim, local_time, local_height, local_width, hidden_dim) for _ in range(orders)])
        self.to_order_out = ModuleList([nn.Linear(dim, dim) for _ in range(orders)])
        self.to_out = nn.Linear(dim, dim)

    def stss_transform(self, x, cache = None, return_cache = False):
        assert not (exists(cache) and not self.causal), 'cache cannot be passed in if MOSS is not causal'

        lt, lh, lw = self.local_time, self.local_height, self.local_width
        _, _, h, w, _ = x.shape

        x = l2norm(x)
        x = rearrange(x, 'b t h w c -> b c t h w')

        pad_h, pad_w = lh // 2, lw // 2
        pad_t_past, pad_t_future = (lt - 1, 0) if self.causal else (lt // 2, lt // 2)

        has_cache = self.causal and exists(cache)
        x_temporal = torch.cat((cache, x), dim = 2) if has_cache else x

        padding = (pad_w, pad_w, pad_h, pad_h, 0 if has_cache else pad_t_past, pad_t_future)
        padded_x = F.pad(x_temporal, padding)

        windows = padded_x.unfold(2, lt, 1).unfold(3, lh, 1).unfold(4, lw, 1)

        sim = einsum(x, windows, 'b c t h w, b c t h w l u v -> b t h w l u v')

        if not return_cache:
            return sim

        new_cache = padded_x[..., -(lt - 1):, pad_h:(pad_h + h), pad_w:(pad_w + w)] if self.causal else None
        return sim, new_cache

    def forward(
        self,
        x,
        cache = None,
        return_cache = False
    ):
        assert not (exists(cache) and not self.causal), 'cache cannot be passed in if MOSS is not causal'

        out = self.to_out(x)

        new_caches = []
        cache = default(cache, (None,) * len(self.encoders))

        for encoder, to_order_out, layer_cache in zip(self.encoders, self.to_order_out, cache):
            sim, next_cache = self.stss_transform(x, cache = layer_cache, return_cache = True)
            new_caches.append(next_cache)

            x = encoder(sim)
            out = out + to_order_out(x)

        if not return_cache:
            return out

        return out, tuple(new_caches)

# main architecture

class ViViT(Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        spatial_depth,
        temporal_depth,
        heads,
        mlp_dim,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        use_flash_attn: bool = True,
        moss_local_time = 3,
        moss_local_height = 3,
        moss_local_width = 3,
        moss_hidden_dim = 64,
        moss_orders = 2,
        moss_causal = True,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.patch_size = patch_height, patch_width = pair(image_patch_size)

        assert divisible_by(image_height, patch_height) and divisible_by(image_width, patch_width), 'Image dimensions must be divisible by the patch size.'
        assert divisible_by(frames, frame_patch_size), 'Frames must be divisible by frame patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        num_image_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = frames // frame_patch_size
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.frame_patch_size = frame_patch_size
        self.patch_h = image_height // patch_height
        self.patch_w = image_width // patch_width
        self.global_average_pool = pool == 'mean'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (pf p1 p2 c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.has_cls = not self.global_average_pool
        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if self.has_cls else None
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if self.has_cls else None

        self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout, use_flash_attn, causal = False)
        self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout, use_flash_attn, causal = moss_causal)

        self.moss = MOSS(
            dim,
            local_time = moss_local_time,
            local_height = moss_local_height,
            local_width = moss_local_width,
            hidden_dim = moss_hidden_dim,
            orders = moss_orders,
            causal = moss_causal
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video, mask = None):
        assert not (exists(mask) and self.moss.causal), 'mask cannot be passed if MOSS is causal'

        x = self.to_patch_embedding(video)
        batch, frames, seq, _ = x.shape

        x = x + self.pos_embedding[:, :frames, :seq]

        if self.has_cls:
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b = batch, f = frames)
            x = torch.cat((spatial_cls_tokens, x), dim = 2)

        x = self.dropout(x)

        # temporal mask

        temporal_mask = None
        if exists(mask):
            temporal_mask = reduce(mask, 'b (f patch) -> b f', 'all', patch = self.frame_patch_size)

        x = rearrange(x, 'b f n d -> (b f) n d')

        # attend across space

        x = self.spatial_transformer(x)
        x = rearrange(x, '(b f) n d -> b f n d', b = batch)

        # moss integration over spatial patch tokens

        if self.has_cls:
            spatial_cls_tokens, patch_tokens = x[:, :, :1], x[:, :, 1:]
        else:
            patch_tokens = x

        patch_tokens = rearrange(patch_tokens, 'b f (h w) d -> b f h w d', h = self.patch_h, w = self.patch_w)
        patch_tokens = self.moss(patch_tokens)
        patch_tokens = rearrange(patch_tokens, 'b f h w d -> b f (h w) d')

        # pool spatial features

        moss_pooled = reduce(patch_tokens, 'b f n d -> b f d', 'mean')

        if self.has_cls:
            x = rearrange(spatial_cls_tokens, 'b f 1 d -> b f d') + moss_pooled
        else:
            x = moss_pooled

        # append temporal cls tokens

        if self.has_cls:
            temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d -> b 1 d', b = batch)
            x = torch.cat((temporal_cls_tokens, x), dim = 1)

            if exists(temporal_mask):
                temporal_mask = F.pad(temporal_mask, (1, 0), value = True)

        # attend across time

        x = self.temporal_transformer(x, mask = temporal_mask)

        # temporal pooling

        x = x[:, 0] if self.has_cls else reduce(x, 'b f d -> b d', 'mean')

        return self.mlp_head(x)

if __name__ == '__main__':

    vivit = ViViT(
        dim = 512,
        spatial_depth = 2,
        temporal_depth = 2,
        heads = 4,
        mlp_dim = 2048,
        image_size = 256,
        image_patch_size = 32,
        frames = 8,
        frame_patch_size = 2,
        num_classes = 1000,
        moss_causal = True
    )

    video = torch.randn(2, 3, 8, 256, 256)
    logits = vivit(video, mask = None)
    assert logits.shape == (2, 1000)

    moss = MOSS(
        dim = 512,
        local_time = 3,
        local_height = 3,
        local_width = 3,
        hidden_dim = 64,
        orders = 2,
        causal = True
    )

    moss_input = torch.randn(2, 8, 16, 16, 512) # (batch, frames, height, width, dim)
    moss_output = moss(moss_input)
    assert moss_output.shape == (2, 8, 16, 16, 512)
