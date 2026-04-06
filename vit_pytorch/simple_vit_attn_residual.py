from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def last(arr):
    return arr[-1]

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(num, den):
    return (num % den) == 0

def posemb_sincos_2d(h, w, dim, temperature = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing = 'ij')
    assert divisible_by(dim, 4), 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# classes

class FeedForward(Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, cross_attend = False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim) if cross_attend else None

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, context = None):
        x = self.norm(x)

        if exists(context):
            context = self.norm_context(context)
        else:
            context = x

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim = -1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class AttentionResidual(Module):
    def __init__(self, fn, dim, heads = 8, dim_head = 64, learned_query = True, disable = False):
        super().__init__()
        self.fn = fn
        self.disable = disable

        if disable:
            return

        self.attn = Attention(dim, heads = heads, dim_head = dim_head, cross_attend = True)
        self.learned_query = nn.Parameter(torch.randn(dim)) if learned_query else None

    def forward(self, history: list[Tensor]) -> Tensor:
        if self.disable:
            return self.fn(last(history))

        batch, seq_len = history[0].shape[:2]

        context = torch.stack(history, dim = 2)
        context = rearrange(context, 'b n l d -> (b n) l d')

        if exists(self.learned_query):
            q = repeat(self.learned_query, 'd -> (b n) 1 d', b = batch, n = seq_len)
        else:
            q = rearrange(last(history), 'b n d -> (b n) 1 d')

        pooled = self.attn(q, context = context)
        pooled = rearrange(pooled, '(b n) 1 d -> b n d', b = batch, n = seq_len)

        return self.fn(pooled)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, learned_query = True):
        super().__init__()

        self.layers = ModuleList([])
        for ind in range(depth):
            is_first = ind == 0

            self.layers.append(ModuleList([
                AttentionResidual(Attention(dim, heads = heads, dim_head = dim_head), dim, heads = heads, dim_head = dim_head, learned_query = learned_query, disable = is_first),
                AttentionResidual(FeedForward(dim, mlp_dim), dim, heads = heads, dim_head = dim_head, learned_query = learned_query),
            ]))

        self.final_pool = AttentionResidual(nn.LayerNorm(dim), dim, heads = heads, dim_head = dim_head, learned_query = learned_query)

    def forward(
        self,
        x,
        history: list[Tensor] | None = None,
        return_history = False
    ):
        history = [*default(history, [])]

        history.append(x)

        for attn_residual, ff_residual in self.layers:
            history.append(attn_residual(history))
            history.append(ff_residual(history))

        out = self.final_pool(history)

        if return_history:
            return out, history

        return out

class SimpleViTAttnResidual(Module):
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
        channels = 3,
        dim_head = 64,
        learned_query = True
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert divisible_by(image_height, patch_height) and divisible_by(image_width, patch_width), 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, learned_query = learned_query)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(
        self,
        img,
        history: list[Tensor] | None = None,
        return_history = False
    ):
        device, dtype = img.device, img.dtype

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype = dtype)

        x = self.transformer(x, history = history, return_history = return_history)

        if return_history:
            x, history = x

        x = x.mean(dim = 1)

        x = self.to_latent(x)
        out = self.linear_head(x)

        if return_history:
            return out, history

        return out

if __name__ == '__main__':
    for learned_query in (True, False):
        v = SimpleViTAttnResidual(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            learned_query = learned_query
        )

        img = torch.randn(2, 3, 256, 256)
        preds, history = v(img, return_history = True)

        assert preds.shape == (2, 1000)

        preds, _ = v(img, history = history, return_history = True)

        assert preds.shape == (2, 1000)
