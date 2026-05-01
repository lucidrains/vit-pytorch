# DetPool ViT - a vit that accepts an object mask and attends and pools only using that mask - table 1
# Dantong Niu et al. - https://openreview.net/forum?id=NZDaMcpXZm

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange, Reduce

# helpers

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def masked_mean(t, mask, dim = 1, eps = 1e-5):
    if not exists(mask):
        return t.mean(dim = dim)

    mask = rearrange(mask.bool(), '... -> ... 1')
    t = t.masked_fill(~mask, 0.)
    return t.sum(dim = dim) / mask.sum(dim = dim).clamp(min = eps)

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
        inner_dim = dim_head *  heads
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

    def forward(self, x, mask = None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = (rearrange(t, 'b n (h d) -> b h n d', h = self.heads) for t in qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(~mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return self.norm(x)

class ViTDetPool(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, use_cls_token = True, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.patch_height = patch_height
        self.patch_width = patch_width

        self.downsample_mask = Reduce('b (h p1) (w p2) -> b (h w)', 'max', p1 = patch_height, p2 = patch_width)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # maybe cls

        self.use_cls_token = use_cls_token

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(dim) * 1e-2)

        self.pos_embedding = nn.Parameter(torch.randn(num_patches, dim) * 1e-2)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes) if num_classes > 0 else None

    def forward(self, img, object_mask = None):
        has_cls = self.use_cls_token

        batch = img.shape[0]
        tokens = self.to_patch_embedding(img)

        seq = tokens.shape[1]
        tokens = tokens + self.pos_embedding[:seq]

        if has_cls:
            cls_token = repeat(self.cls_token, 'd -> b d', b = batch)
            tokens, packed_shape = pack((cls_token, tokens), 'b * d')

        tokens = self.dropout(tokens)

        # handle the attention mask, and for final pooling

        mask = None

        if exists(object_mask):
            assert object_mask.ndim in {3, 2}

            if object_mask.ndim == 3:
                mask = self.downsample_mask(object_mask)

            elif object_mask.ndim == 2:
                mask = object_mask

            assert mask.shape == (batch, seq)

            if has_cls:
                mask = F.pad(mask, (1, 0), value = True)

        # attend with maybe mask

        tokens = self.transformer(tokens, mask = mask)

        if not exists(self.mlp_head):
            return tokens

        # splice out cls

        if has_cls:
            _, tokens = unpack(tokens, packed_shape, 'b * d')

            if exists(mask):
                mask = mask[..., 1:]

        # pooling with the mask

        pooled = masked_mean(tokens, mask, dim = 1)

        pooled = self.to_latent(pooled)
        return self.mlp_head(pooled)

# quick test

if __name__ == '__main__':
    v = ViTDetPool(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    img = torch.randn(1, 3, 256, 256)
    object_mask = torch.randint(0, 2, (1, 256, 256)).bool()

    preds = v(img, object_mask = object_mask)
    assert preds.shape == (1, 1000)

    preds_no_mask = v(img)
    assert preds_no_mask.shape == (1, 1000)

    v_no_cls = ViTDetPool(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        use_cls_token = False,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    preds_no_cls = v_no_cls(img, object_mask = object_mask)
    assert preds_no_cls.shape == (1, 1000)
