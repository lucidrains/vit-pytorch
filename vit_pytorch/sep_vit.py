from functools import partial

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# helpers

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

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = ChanLayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class OverlappingPatchEmbed(nn.Module):
    def __init__(self, dim_in, dim_out, stride = 2):
        super().__init__()
        kernel_size = stride * 2 - 1
        padding = kernel_size // 2
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride = stride, padding = padding)

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
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention

class DSSA(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        inner_dim = dim_head * heads

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)

        # window tokens

        self.window_tokens = nn.Parameter(torch.randn(dim))

        # prenorm and non-linearity for window tokens
        # then projection to queries and keys for window tokens

        self.window_tokens_to_qk = nn.Sequential(
            nn.LayerNorm(dim_head),
            nn.GELU(),
            Rearrange('b h n c -> b (h c) n'),
            nn.Conv1d(inner_dim, inner_dim * 2, 1),
            Rearrange('b (h c) n -> b h n c', h = heads),
        )

        # window attention

        self.window_attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        einstein notation

        b - batch
        c - channels
        w1 - window size (height)
        w2 - also window size (width)
        i - sequence dimension (source)
        j - sequence dimension (target dimension to be reduced)
        h - heads
        x - height of feature map divided by window size
        y - width of feature map divided by window size
        """

        batch, height, width, heads, wsz = x.shape[0], *x.shape[-2:], self.heads, self.window_size
        assert (height % wsz) == 0 and (width % wsz) == 0, f'height {height} and width {width} must be divisible by window size {wsz}'
        num_windows = (height // wsz) * (width // wsz)

        # fold in windows for "depthwise" attention - not sure why it is named depthwise when it is just "windowed" attention

        x = rearrange(x, 'b c (h w1) (w w2) -> (b h w) c (w1 w2)', w1 = wsz, w2 = wsz)

        # add windowing tokens

        w = repeat(self.window_tokens, 'c -> b c 1', b = x.shape[0])
        x = torch.cat((w, x), dim = -1)

        # project for queries, keys, value

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # scale

        q = q * self.scale

        # similarity

        dots = einsum('b h i d, b h j d -> b h i j', q, k)

        # attention

        attn = self.attend(dots)

        # aggregate values

        out = torch.matmul(attn, v)

        # split out windowed tokens

        window_tokens, windowed_fmaps = out[:, :, 0], out[:, :, 1:]

        # early return if there is only 1 window

        if num_windows == 1:
            fmap = rearrange(windowed_fmaps, '(b x y) h (w1 w2) d -> b (h d) (x w1) (y w2)', x = height // wsz, y = width // wsz, w1 = wsz, w2 = wsz)
            return self.to_out(fmap)

        # carry out the pointwise attention, the main novelty in the paper

        window_tokens = rearrange(window_tokens, '(b x y) h d -> b h (x y) d', x = height // wsz, y = width // wsz)
        windowed_fmaps = rearrange(windowed_fmaps, '(b x y) h n d -> b h (x y) n d', x = height // wsz, y = width // wsz)

        # windowed queries and keys (preceded by prenorm activation)

        w_q, w_k = self.window_tokens_to_qk(window_tokens).chunk(2, dim = -1)

        # scale

        w_q = w_q * self.scale

        # similarities

        w_dots = einsum('b h i d, b h j d -> b h i j', w_q, w_k)

        w_attn = self.window_attend(w_dots)

        # aggregate the feature maps from the "depthwise" attention step (the most interesting part of the paper, one i haven't seen before)

        aggregated_windowed_fmap = einsum('b h i j, b h j w d -> b h i w d', w_attn, windowed_fmaps)

        # fold back the windows and then combine heads for aggregation

        fmap = rearrange(aggregated_windowed_fmap, 'b h (x y) (w1 w2) d -> b (h d) (x w1) (y w2)', x = height // wsz, y = width // wsz, w1 = wsz, w2 = wsz)
        return self.to_out(fmap)

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 32,
        heads = 8,
        ff_mult = 4,
        dropout = 0.,
        norm_output = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, DSSA(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mult = ff_mult, dropout = dropout)),
            ]))

        self.norm = ChanLayerNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class SepViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        heads,
        window_size = 7,
        dim_head = 32,
        ff_mult = 4,
        channels = 3,
        dropout = 0.
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (channels, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        strides = (4, *((2,) * (num_stages - 1)))

        hyperparams_per_stage = [heads, window_size]
        hyperparams_per_stage = list(map(partial(cast_tuple, length = num_stages), hyperparams_per_stage))
        assert all(tuple(map(lambda arr: len(arr) == num_stages, hyperparams_per_stage)))

        self.layers = nn.ModuleList([])

        for ind, ((layer_dim_in, layer_dim), layer_depth, layer_stride, layer_heads, layer_window_size) in enumerate(zip(dim_pairs, depth, strides, *hyperparams_per_stage)):
            is_last = ind == (num_stages - 1)

            self.layers.append(nn.ModuleList([
                OverlappingPatchEmbed(layer_dim_in, layer_dim, stride = layer_stride),
                PEG(layer_dim),
                Transformer(dim = layer_dim, depth = layer_depth, heads = layer_heads, ff_mult = ff_mult, dropout = dropout, norm_output = not is_last),
            ]))

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):

        for ope, peg, transformer in self.layers:
            x = ope(x)
            x = peg(x)
            x = transformer(x)

        return self.mlp_head(x)
