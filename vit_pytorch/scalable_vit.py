from functools import partial
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 3, stride = 2, padding = 1)

    def forward(self, x):
        return self.conv(x)

class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1)

    def forward(self, x):
        return self.proj(x) + x

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor = 4, dropout = 0.):
        super().__init__()
        inner_dim = dim * expansion_factor
        self.net = nn.Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, inner_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention

class ScalableSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_key = 32,
        dim_value = 32,
        dropout = 0.,
        reduction_factor = 1
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_key ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.norm = ChanLayerNorm(dim)
        self.to_q = nn.Conv2d(dim, dim_key * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_key * heads, reduction_factor, stride = reduction_factor, bias = False)
        self.to_v = nn.Conv2d(dim, dim_value * heads, reduction_factor, stride = reduction_factor, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(dim_value * heads, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        height, width, heads = *x.shape[-2:], self.heads

        x = self.norm(x)

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # similarity

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # attention

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # aggregate values

        out = torch.matmul(attn, v)

        # merge back heads

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = height, y = width)
        return self.to_out(out)

class InteractiveWindowedSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        heads = 8,
        dim_key = 32,
        dim_value = 32,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_key ** -0.5
        self.window_size = window_size
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.norm = ChanLayerNorm(dim)
        self.local_interactive_module = nn.Conv2d(dim_value * heads, dim_value * heads, 3, padding = 1)

        self.to_q = nn.Conv2d(dim, dim_key * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_key * heads, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim_value * heads, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(dim_value * heads, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        height, width, heads, wsz = *x.shape[-2:], self.heads, self.window_size

        x = self.norm(x)

        wsz_h, wsz_w = default(wsz, height), default(wsz, width)
        assert (height % wsz_h) == 0 and (width % wsz_w) == 0, f'height ({height}) or width ({width}) of feature map is not divisible by the window size ({wsz_h}, {wsz_w})'

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # get output of LIM

        local_out = self.local_interactive_module(v)

        # divide into window (and split out heads) for efficient self attention

        q, k, v = map(lambda t: rearrange(t, 'b (h d) (x w1) (y w2) -> (b x y) h (w1 w2) d', h = heads, w1 = wsz_h, w2 = wsz_w), (q, k, v))

        # similarity

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # attention

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # aggregate values

        out = torch.matmul(attn, v)

        # reshape the windows back to full feature map (and merge heads)

        out = rearrange(out, '(b x y) h (w1 w2) d -> b (h d) (x w1) (y w2)', x = height // wsz_h, y = width // wsz_w, w1 = wsz_h, w2 = wsz_w)

        # add LIM output 

        out = out + local_out

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        ff_expansion_factor = 4,
        dropout = 0.,
        ssa_dim_key = 32,
        ssa_dim_value = 32,
        ssa_reduction_factor = 1,
        iwsa_dim_key = 32,
        iwsa_dim_value = 32,
        iwsa_window_size = None,
        norm_output = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for ind in range(depth):
            is_first = ind == 0

            self.layers.append(nn.ModuleList([
                ScalableSelfAttention(dim, heads = heads, dim_key = ssa_dim_key, dim_value = ssa_dim_value, reduction_factor = ssa_reduction_factor, dropout = dropout),
                FeedForward(dim, expansion_factor = ff_expansion_factor, dropout = dropout),
                PEG(dim) if is_first else None,
                FeedForward(dim, expansion_factor = ff_expansion_factor, dropout = dropout),
                InteractiveWindowedSelfAttention(dim, heads = heads, dim_key = iwsa_dim_key, dim_value = iwsa_dim_value, window_size = iwsa_window_size, dropout = dropout)
            ]))

        self.norm = ChanLayerNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for ssa, ff1, peg, iwsa, ff2 in self.layers:
            x = ssa(x) + x
            x = ff1(x) + x

            if exists(peg):
                x = peg(x)

            x = iwsa(x) + x
            x = ff2(x) + x

        return self.norm(x)

class ScalableViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        heads,
        reduction_factor,
        window_size = None,
        iwsa_dim_key = 32,
        iwsa_dim_value = 32,
        ssa_dim_key = 32,
        ssa_dim_value = 32,
        ff_expansion_factor = 4,
        channels = 3,
        dropout = 0.
    ):
        super().__init__()
        self.to_patches = nn.Conv2d(channels, dim, 7, stride = 4, padding = 3)

        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        num_stages = len(depth)
        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))

        hyperparams_per_stage = [
            heads,
            ssa_dim_key,
            ssa_dim_value,
            reduction_factor,
            iwsa_dim_key,
            iwsa_dim_value,
            window_size,
        ]

        hyperparams_per_stage = list(map(partial(cast_tuple, length = num_stages), hyperparams_per_stage))
        assert all(tuple(map(lambda arr: len(arr) == num_stages, hyperparams_per_stage)))

        self.layers = nn.ModuleList([])

        for ind, (layer_dim, layer_depth, layer_heads, layer_ssa_dim_key, layer_ssa_dim_value, layer_ssa_reduction_factor, layer_iwsa_dim_key, layer_iwsa_dim_value, layer_window_size) in enumerate(zip(dims, depth, *hyperparams_per_stage)):
            is_last = ind == (num_stages - 1)

            self.layers.append(nn.ModuleList([
                Transformer(dim = layer_dim, depth = layer_depth, heads = layer_heads, ff_expansion_factor = ff_expansion_factor, dropout = dropout, ssa_dim_key = layer_ssa_dim_key, ssa_dim_value = layer_ssa_dim_value, ssa_reduction_factor = layer_ssa_reduction_factor, iwsa_dim_key = layer_iwsa_dim_key, iwsa_dim_value = layer_iwsa_dim_value, iwsa_window_size = layer_window_size, norm_output = not is_last),
                Downsample(layer_dim, layer_dim * 2) if not is_last else None
            ]))

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, img):
        x = self.to_patches(img)

        for transformer, downsample in self.layers:
            x = transformer(x)

            if exists(downsample):
                x = downsample(x)

        return self.mlp_head(x)
