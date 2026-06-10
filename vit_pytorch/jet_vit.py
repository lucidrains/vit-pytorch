import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, reduce, repeat, einsum
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

def linear_attn(q, k, v):
    q = F.relu(q)
    k = F.relu(k)

    context = einsum(k, v, 'b h n d, b h n e -> b h d e')
    normalizer = einsum(q, k.sum(dim=2), 'b h n d, b h d -> b h n')
    attn = einsum(q, context, 'b h n d, b h d e -> b h n e')
    return (attn / normalizer.unsqueeze(-1).clamp(min=1e-6))


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

class SqueezeDynamicConv(Module):
    def __init__(self, dim, h_s, w_s, kernel_size = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.dim = dim
        self.h_s = h_s
        self.w_s = w_s

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, dim * kernel_size * kernel_size)
        )

    def forward(self, v):
        b, heads, _, _ = v.shape
        
        # Squeeze global context
        v_mean = reduce(v, 'b h n d -> b (h d)', 'mean')

        weight = self.mlp(v_mean) 

        weight = rearrange(weight, 'b (c k1 k2) -> (b c) 1 k1 k2', c = self.dim, k1 = self.kernel_size, k2 = self.kernel_size)

        v_spatial = rearrange(v, 'b h (h_s w_s) d -> 1 (b h d) h_s w_s', h_s = self.h_s, w_s = self.w_s)

        out = F.conv2d(v_spatial, weight, padding = self.padding, groups = b * self.dim)
        
        return rearrange(out, '1 (b h d) h_s w_s -> b h (h_s w_s) d', b = b, h = heads, h_s = self.h_s, w_s = self.w_s)
    

class JetViTLinearAttention(Module):
    def __init__(self, dim, h_s, w_s, heads = 8, dim_head = 64, dropout = 0., kernel_size = 3):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.h_s = h_s
        self.w_s = w_s

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.dynamic_conv = SqueezeDynamicConv(inner_dim, h_s, w_s, kernel_size = kernel_size)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        
        seq_len = x.shape[1]
        has_cls = (self.h_s * self.w_s + 1 == seq_len)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        linear_out = linear_attn(q, k, v)
        linear_out = rearrange(linear_out, 'b h n d -> b n (h d)')

        v_img = v[:, :, 1:, :] if has_cls else v
            
        conv_out = self.dynamic_conv(v_img)
        conv_out = rearrange(conv_out, 'b h n d -> b n (h d)')

        if has_cls:
            cls_zeros = torch.zeros((x.shape[0], 1, conv_out.shape[-1]), device=x.device, dtype=x.dtype)
            conv_out = torch.cat((cls_zeros, conv_out), dim = 1)

        return self.to_out(linear_out + conv_out)
    

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

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, h_s, w_s, dropout = 0., full_attn_layers = ()):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for i in range(depth):
            if i in full_attn_layers:
                attn = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
            else:
                attn = JetViTLinearAttention(dim, h_s, w_s, heads = heads, dim_head = dim_head, dropout = dropout)

            self.layers.append(ModuleList([
                attn,
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., full_attn_layers = ()):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.patch_size = patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        h_s = image_height // patch_height
        w_s = image_width // patch_width

        num_patches = h_s * w_s
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, h_s, w_s, dropout, full_attn_layers)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes) if num_classes > 0 else None

    def forward(self, img):
        batch = img.shape[0]
        
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)
        x = torch.cat((cls_tokens, x), dim = 1)

        seq = x.shape[1]

        x = x + self.pos_embedding[:seq]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.mlp_head is None:
            return x

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)