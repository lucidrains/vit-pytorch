from __future__ import annotations

# Alexis Marouani et al. https://arxiv.org/abs/2602.08626

import torch
from torch import nn, cat, Tensor, is_tensor
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class Specialized(Module):
    def __init__(self, modules: list[Module]):
        super().__init__()
        self.fns = ModuleList(modules)

    def forward(
        self,
        x: Tensor | list[Tensor],
        token_lens: tuple[int, ...] = None
    ):
        if is_tensor(x):
            assert exists(token_lens)
            x = x.split(token_lens, dim = 1)

        assert len(self.fns) == len(x)

        out = tuple(fn(t) for fn, t in zip(self.fns, x))

        if is_tensor:
            out = cat(out, dim = 1)

        return out

class FeedForward(Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = Specialized([
            nn.LayerNorm(dim),
            nn.LayerNorm(dim)
        ])

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x, token_lens = None):
        x = self.norm(x, token_lens = token_lens)
        return self.net(x)

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, specialize_qkv = False):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = Specialized([
            nn.LayerNorm(dim),
            nn.LayerNorm(dim)
        ])

        self.attend = nn.Softmax(dim = -1)
        self.specialize_qkv = specialize_qkv

        if specialize_qkv:
            self.to_qkv = Specialized([
                nn.Linear(dim, inner_dim * 3, bias = False),
                nn.Linear(dim, inner_dim * 3, bias = False)
            ])
        else:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, token_lens = None):
        x = self.norm(x, token_lens = token_lens)

        if self.specialize_qkv:
            qkv = self.to_qkv(x, token_lens = token_lens).chunk(3, dim = -1)
        else:
            qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = (rearrange(t, 'b n (h d) -> b h n d', h = self.heads) for t in qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, specialize_qkv_depth):
        super().__init__()
        self.norm = Specialized([nn.LayerNorm(dim), nn.LayerNorm(dim)])

        self.layers = ModuleList([])

        for ind in range(depth):
            specialize_qkv = ind < specialize_qkv_depth
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, specialize_qkv = specialize_qkv),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x, token_lens = None):
        for attn, ff in self.layers:
            x = attn(x, token_lens = token_lens) + x
            x = ff(x, token_lens = token_lens) + x

        return self.norm(x, token_lens = token_lens)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, specialize_qkv_depth = None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        )

        self.cls_token = nn.Parameter(torch.randn(dim) * 1e-2)

        specialize_qkv_depth = default(specialize_qkv_depth, depth // 3) # author found just first third of transformer having specialized qkv projection for cls token is enough

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, specialize_qkv_depth)

        self.pool = 'cls'
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = b)
        x = cat((cls_tokens, x), dim = 1)

        x = self.transformer(x, token_lens = (1, n))

        x = x[:, 0]

        x = self.to_latent(x)
        return self.linear_head(x)

if __name__ == '__main__':
    v = SimpleViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048
    )

    img = torch.randn(1, 3, 256, 256)
    out = v(img)

    assert out.shape == (1, 1000)
