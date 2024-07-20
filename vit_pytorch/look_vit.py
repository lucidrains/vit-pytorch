import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

# simple vit sinusoidal pos emb

def posemb_sincos_2d(t, temperature = 10000):
    h, w, d, device = *t.shape[1:], t.device
    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (d % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(d // 4, device = device) / (d // 4 - 1)
    omega = temperature ** -omega

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pos = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)

    return pos.float()

# bias-less layernorm with unit offset trick (discovered by Ohad Rubin)

class LayerNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        normed = self.ln(x)
        return normed * (self.gamma + 1)

# mlp

def MLP(dim, factor = 4, dropout = 0.):
    hidden_dim = int(dim * factor)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        cross_attend = False,
        reuse_attention = False
    ):
        super().__init__()
        inner_dim = dim_head *  heads

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.reuse_attention = reuse_attention
        self.cross_attend = cross_attend

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)

        self.norm = LayerNorm(dim) if not reuse_attention else nn.Identity()
        self.norm_context = LayerNorm(dim) if cross_attend else nn.Identity()

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False) if not reuse_attention else None
        self.to_k = nn.Linear(dim, inner_dim, bias = False) if not reuse_attention else None
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        return_qk_sim = False,
        qk_sim = None
    ):
        x = self.norm(x)

        assert not (exists(context) ^ self.cross_attend)

        if self.cross_attend:
            context = self.norm_context(context)
        else:
            context = x

        v = self.to_v(context)
        v = self.split_heads(v)

        if not self.reuse_attention:
            qk = (self.to_q(x), self.to_k(context))
            q, k = tuple(self.split_heads(t) for t in qk)

            q = q * self.scale
            qk_sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        else:
            assert exists(qk_sim), 'qk sim matrix must be passed in for reusing previous attention'

        attn = self.attend(qk_sim)
        attn = self.dropout(attn)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        out = self.to_out(out)

        if not return_qk_sim:
            return out

        return out, qk_sim

# LookViT

class LookViT(Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        num_classes,
        depth = 3,
        patch_size = 16,
        heads = 8,
        mlp_factor = 4,
        dim_head = 64,
        highres_patch_size = 12,
        highres_mlp_factor = 4,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        patch_conv_kernel_size = 7,
        dropout = 0.1,
        channels = 3
    ):
        super().__init__()
        assert divisible_by(image_size, highres_patch_size)
        assert divisible_by(image_size, patch_size)
        assert patch_size > highres_patch_size, 'patch size of the main vision transformer should be smaller than the highres patch sizes (that does the `lookup`)'
        assert not divisible_by(patch_conv_kernel_size, 2)

        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size

        kernel_size = patch_conv_kernel_size
        patch_dim = (highres_patch_size * highres_patch_size) * channels

        self.to_patches = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = highres_patch_size, p2 = highres_patch_size),
            nn.Conv2d(patch_dim, dim, kernel_size, padding = kernel_size // 2),
            Rearrange('b c h w -> b h w c'),
            LayerNorm(dim),
        )

        # absolute positions

        num_patches = (image_size // highres_patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(num_patches, dim))

        # lookvit blocks

        layers = ModuleList([])

        for _ in range(depth):
            layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout),
                MLP(dim = dim, factor = mlp_factor, dropout = dropout),
                Attention(dim = dim, dim_head = cross_attn_dim_head, heads = cross_attn_heads, dropout = dropout, cross_attend = True),
                Attention(dim = dim, dim_head = cross_attn_dim_head, heads = cross_attn_heads, dropout = dropout, cross_attend = True, reuse_attention = True),
                LayerNorm(dim),
                MLP(dim = dim, factor = highres_mlp_factor, dropout = dropout)
            ]))

        self.layers = layers

        self.norm = LayerNorm(dim)
        self.highres_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, num_classes, bias = False)

    def forward(self, img):
        assert img.shape[-2:] == (self.image_size, self.image_size)

        # to patch tokens and positions

        highres_tokens = self.to_patches(img)
        size = highres_tokens.shape[-2]

        pos_emb = posemb_sincos_2d(highres_tokens)
        highres_tokens = highres_tokens + rearrange(pos_emb, '(h w) d -> h w d', h = size)

        tokens = F.interpolate(
            rearrange(highres_tokens, 'b h w d -> b d h w'),
            img.shape[-1] // self.patch_size,
            mode = 'bilinear'
        )

        tokens = rearrange(tokens, 'b c h w -> b (h w) c')
        highres_tokens = rearrange(highres_tokens, 'b h w c -> b (h w) c')

        # attention and feedforwards

        for attn, mlp, lookup_cross_attn, highres_attn, highres_norm, highres_mlp in self.layers:

            # main tokens cross attends (lookup) on the high res tokens

            lookup_out, qk_sim = lookup_cross_attn(tokens, highres_tokens, return_qk_sim = True)  # return attention as they reuse the attention matrix
            tokens = lookup_out + tokens

            tokens = attn(tokens) + tokens
            tokens = mlp(tokens) + tokens

            # attention-reuse

            qk_sim = rearrange(qk_sim, 'b h i j -> b h j i') # transpose for reverse cross attention

            highres_tokens = highres_attn(highres_tokens, tokens, qk_sim = qk_sim) + highres_tokens
            highres_tokens = highres_norm(highres_tokens)

            highres_tokens = highres_mlp(highres_tokens) + highres_tokens

        # to logits

        tokens = self.norm(tokens)
        highres_tokens = self.highres_norm(highres_tokens)

        tokens = reduce(tokens, 'b n d -> b d', 'mean')
        highres_tokens = reduce(highres_tokens, 'b n d -> b d', 'mean')

        return self.to_logits(tokens + highres_tokens)

# main

if __name__ == '__main__':
    v = LookViT(
        image_size = 256,
        num_classes = 1000,
        dim = 512,
        depth = 2,
        heads = 8,
        dim_head = 64,
        patch_size = 32,
        highres_patch_size = 8,
        highres_mlp_factor = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        dropout = 0.1
    ).cuda()

    img = torch.randn(2, 3, 256, 256).cuda()
    pred = v(img)

    assert pred.shape == (2, 1000)
