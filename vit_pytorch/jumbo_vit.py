# Simpler Fast Vision Transformers with a Jumbo CLS Token
# https://arxiv.org/abs/2502.15021

import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(num, den):
    return (num % den) == 0

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert divisible_by(dim, 4), "feature dimension must be multiple of 4 for sincos emb"

    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = temperature ** -omega

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pos_emb = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)

    return pos_emb.type(dtype)

# classes

def FeedForward(dim, mult = 4.):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim),
    )

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class JumboViT(Module):
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
        num_jumbo_cls = 1,  # differing from paper, allow for multiple jumbo cls, so one could break it up into 2 jumbo cls tokens with 3x the dim, as an example
        jumbo_cls_k = 6,    # they use a CLS token with this factor times the dimension - 6 was the value they settled on
        jumbo_ff_mult = 2,  # expansion factor of the jumbo cls token feedforward
        channels = 3,
        dim_head = 64
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert divisible_by(image_height, patch_height) and divisible_by(image_width, patch_width), 'Image dimensions must be divisible by the patch size.'

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

        jumbo_cls_dim = dim * jumbo_cls_k

        self.jumbo_cls_token = nn.Parameter(torch.zeros(num_jumbo_cls, jumbo_cls_dim))

        jumbo_cls_to_tokens = Rearrange('b n (k d) -> b (n k) d', k = jumbo_cls_k)
        self.jumbo_cls_to_tokens = jumbo_cls_to_tokens

        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        # attention and feedforwards

        self.jumbo_ff = nn.Sequential(
            Rearrange('b (n k) d -> b n (k d)', k = jumbo_cls_k),
            FeedForward(jumbo_cls_dim, int(jumbo_cls_dim * jumbo_ff_mult)), # they use separate parameters for the jumbo feedforward, weight tied for parameter efficient
            jumbo_cls_to_tokens
        )

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim),
            ]))

        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):

        batch, device = img.shape[0], img.device

        x = self.to_patch_embedding(img)

        # pos embedding

        pos_emb = self.pos_embedding.to(device, dtype = x.dtype)

        x = x + pos_emb

        # add cls tokens

        cls_tokens = repeat(self.jumbo_cls_token, 'nj d -> b nj d', b = batch)

        jumbo_tokens = self.jumbo_cls_to_tokens(cls_tokens)

        x, cls_packed_shape = pack([jumbo_tokens, x], 'b * d')

        # attention and feedforwards

        for layer, (attn, ff) in enumerate(self.layers, start = 1):
            is_last = layer == len(self.layers)

            x = attn(x) + x

            # jumbo feedforward

            jumbo_cls_tokens, x = unpack(x, cls_packed_shape, 'b * d')

            x = ff(x) + x
            jumbo_cls_tokens = self.jumbo_ff(jumbo_cls_tokens) + jumbo_cls_tokens

            if is_last:
                continue

            x, _ = pack([jumbo_cls_tokens, x], 'b * d')

        pooled = reduce(jumbo_cls_tokens, 'b n d -> b d', 'mean')

        # normalization and project to logits

        embed = self.norm(pooled)

        embed = self.to_latent(embed)
        logits = self.linear_head(embed)
        return logits

# copy pasteable file

if __name__ == '__main__':

    v = JumboViT(
        num_classes = 1000,
        image_size = 64,
        patch_size = 8,
        dim = 16,
        depth = 2,
        heads = 2,
        mlp_dim = 32,
        jumbo_cls_k = 3,
        jumbo_ff_mult = 2,
    )

    images = torch.randn(1, 3, 64, 64)

    logits = v(images)
    assert logits.shape == (1, 1000)
