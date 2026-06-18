import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, ModuleList
from math import sqrt, pi

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.amp import autocast

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def exists(val):
    return val is not None

@autocast('cuda', enabled = False)
def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, num_registers=4, base=100000, reg_base=100):
        super().__init__()
        self.dim = dim
        self.num_registers = num_registers

        half_dim = dim // 2

        patch_inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer('patch_inv_freq', patch_inv_freq)

        reg_inv_freq = 1.0 / (reg_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('reg_inv_freq', reg_inv_freq)

    @autocast('cuda', enabled=False)
    def forward(self, x):
        device, dtype = x.device, x.dtype
        num_patches = x.shape[-2] - self.num_registers
        n = int(sqrt(num_patches))

        seq = torch.arange(n, device=device, dtype=dtype)

        freqs = torch.einsum('i, j -> i j', seq, self.patch_inv_freq.to(dtype))

        freqs_x = repeat(freqs, 'i d -> i j d', j=n)
        freqs_y = repeat(freqs, 'j d -> i j d', i=n)

        patch_freqs = torch.cat((freqs_x, freqs_y), dim=-1)
        patch_freqs = rearrange(patch_freqs, 'i j d -> (i j) d')

        # registers
        reg_pos = torch.arange(self.num_registers, device=device, dtype=dtype)
        reg_freqs = torch.einsum('i, j -> i j', reg_pos, self.reg_inv_freq.to(dtype))

        all_freqs = torch.cat((reg_freqs, patch_freqs), dim=0)

        sin = all_freqs.sin()
        cos = all_freqs.cos()

        sin, cos = map(lambda t: repeat(t, 'n d -> () () n (d j)', j=2), (sin, cos))
        return sin, cos

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = RMSNorm(dim)

        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, pos_emb=None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if exists(pos_emb):
            sin, cos = pos_emb
            (q_cls, q), (k_cls, k) = map(lambda t: (t[:, :, :1], t[:, :, 1:]), (q, k))
            q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
            q = torch.cat((q_cls, q), dim = 2)
            k = torch.cat((k_cls, k), dim = 2)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.layers = ModuleList([])
        
        self.ls1 = nn.ParameterList([])
        self.ls2 = nn.ParameterList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
            self.ls1.append(nn.Parameter(torch.ones(dim) * 1e-4))
            self.ls2.append(nn.Parameter(torch.ones(dim) * 1e-4))

    def forward(self, x, pos_emb=None):
        for (attn, ff), ls1, ls2 in zip(self.layers, self.ls1, self.ls2):
            x = attn(x, pos_emb=pos_emb) * ls1 + x
            x = ff(x) * ls2 + x

        return self.norm(x)

class ViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., num_registers = 4):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.patch_size = patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.num_registers = num_registers

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, dim))

        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.randn(num_registers, dim))

        self.pos_embedding = nn.Parameter(torch.randn(num_patches + 1 + num_registers, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.rotary_emb = AxialRotaryEmbedding(dim=dim_head, num_registers=num_registers)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes) if num_classes > 0 else None

    def forward(self, img):
        batch = img.shape[0]
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)

        if self.num_registers > 0:
            register_tokens = repeat(self.register_tokens, '... d -> b ... d', b = batch)
            x = torch.cat((cls_tokens, register_tokens, x), dim = 1)
        else:
            x = torch.cat((cls_tokens, x), dim = 1)

        seq = x.shape[1]

        x = x + self.pos_embedding[:seq]
        x = self.dropout(x)

        pos_emb = self.rotary_emb(x[:, 1:])
        x = self.transformer(x, pos_emb=pos_emb)

        if self.mlp_head is None:
            return x

        x = x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)
