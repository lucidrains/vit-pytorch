from math import sqrt, pi, log

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.cuda.amp import autocast

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# rotary embeddings

@autocast(enabled = False)
def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        scales = torch.linspace(1., max_freq / 2, self.dim // 4)
        self.register_buffer('scales', scales)

    @autocast(enabled = False)
    def forward(self, x):
        device, dtype, n = x.device, x.dtype, int(sqrt(x.shape[-2]))

        seq = torch.linspace(-1., 1., steps = n, device = device)
        seq = seq.unsqueeze(-1)

        scales = self.scales[(*((None,) * (len(seq.shape) - 1)), Ellipsis)]
        scales = scales.to(x)

        seq = seq * scales * pi

        x_sinu = repeat(seq, 'i d -> i j d', j = n)
        y_sinu = repeat(seq, 'j d -> i j d', i = n)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos))
        return sin, cos

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

# helper classes

class SpatialConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel, bias = False):
        super().__init__()
        self.conv = DepthWiseConv2d(dim_in, dim_out, kernel, padding = kernel // 2, bias = False)
        self.cls_proj = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x, fmap_dims):
        cls_token, x = x[:, :1], x[:, 1:]
        x = rearrange(x, 'b (h w) d -> b d h w', **fmap_dims)
        x = self.conv(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        cls_token = self.cls_proj(cls_token)
        return torch.cat((cls_token, x), dim = 1)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return F.gelu(gates) * x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., use_glu = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim * 2 if use_glu else hidden_dim),
            GEGLU() if use_glu else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., use_rotary = True, use_ds_conv = True, conv_query_kernel = 5):
        super().__init__()
        inner_dim = dim_head *  heads
        self.use_rotary = use_rotary
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.use_ds_conv = use_ds_conv

        self.to_q = SpatialConv(dim, inner_dim, conv_query_kernel, bias = False) if use_ds_conv else nn.Linear(dim, inner_dim, bias = False)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, pos_emb, fmap_dims):
        b, n, _, h = *x.shape, self.heads

        to_q_kwargs = {'fmap_dims': fmap_dims} if self.use_ds_conv else {}

        x = self.norm(x)

        q = self.to_q(x, **to_q_kwargs)

        qkv = (q, *self.to_kv(x).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

        if self.use_rotary:
            # apply 2d rotary embeddings to queries and keys, excluding CLS tokens

            sin, cos = pos_emb
            dim_rotary = sin.shape[-1]

            (q_cls, q), (k_cls, k) = map(lambda t: (t[:, :1], t[:, 1:]), (q, k))

            # handle the case where rotary dimension < head dimension

            (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k))
            q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
            q, k = map(lambda t: torch.cat(t, dim = -1), ((q, q_pass), (k, k_pass)))

            # concat back the CLS tokens

            q = torch.cat((q_cls, q), dim = 1)
            k = torch.cat((k_cls, k), dim = 1)

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, image_size, dropout = 0., use_rotary = True, use_ds_conv = True, use_glu = True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = AxialRotaryEmbedding(dim_head, max_freq = image_size)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, use_rotary = use_rotary, use_ds_conv = use_ds_conv),
                FeedForward(dim, mlp_dim, dropout = dropout, use_glu = use_glu)
            ]))
    def forward(self, x, fmap_dims):
        pos_emb = self.pos_emb(x[:, 1:])

        for attn, ff in self.layers:
            x = attn(x, pos_emb = pos_emb, fmap_dims = fmap_dims) + x
            x = ff(x) + x
        return x

# Rotary Vision Transformer

class RvT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., use_rotary = True, use_ds_conv = True, use_glu = True):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, image_size, dropout, use_rotary, use_ds_conv, use_glu)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        b, _, h, w, p = *img.shape, self.patch_size

        x = self.to_patch_embedding(img)
        n = x.shape[1]

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

        fmap_dims = {'h': h // p, 'w': w // p}
        x = self.transformer(x, fmap_dims = fmap_dims)

        return self.mlp_head(x[:, 0])
