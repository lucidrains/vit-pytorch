"""
multiway attention residuals for ViT, combining
https://arxiv.org/abs/2603.15031 (Attention Residuals, Kimi)
https://arxiv.org/abs/2607.27230 (Multi-Head Attention Residuals, Luo et al.)
https://arxiv.org/abs/2502.06785 (DeepCrossAttention, DCA)

softmax over preceding depth outputs with zero-init pseudo-queries (AttnRes)
q / k / v each get their own depth mix (DCA), v doubles as the residual

improvisations
- head_groups: one depth mix per group of heads (1 = shared, heads = per-head)
- ffn_heads: one depth mix and proj_in slice per ffn head
- gate: headwise sigmoid gate on attention output
"""

import math

import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import einsum, rearrange, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(v):
    return v is not None

def last(arr):
    return arr[-1]

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(num, den):
    return (num % den) == 0

# pos emb

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    assert divisible_by(dim, 4), 'feature dimension must be multiple of 4 for sincos emb'

    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = temperature ** -omega

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing = 'ij')
    y = rearrange(y, 'h w -> (h w) 1') * omega
    x = rearrange(x, 'h w -> (h w) 1') * omega

    pos = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = -1)
    return pos.type(dtype)

# classes

class GroupedLinear(Module):
    """ block-diagonal linear, same params / flops as dense """
    def __init__(self, dim, out_dim, groups, bias = False):
        super().__init__()
        assert divisible_by(out_dim, groups), 'out_dim must be divisible by groups'

        self.split_groups = Rearrange('(g e) d -> g e d', g = groups)
        self.merge_groups = Rearrange('b n g e -> b n (g e)')

        self.weight = nn.Parameter(torch.empty(out_dim, dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a = math.sqrt(5))

    def forward(self, x):
        # g b n d -> b n (g e)

        weight = self.split_groups(self.weight)
        out = einsum(x, weight, 'g b n d, g e d -> b n g e')
        out = self.merge_groups(out)

        return out + self.bias if exists(self.bias) else out

def linear(dim_in, dim_out, groups = 1, bias = False):
    if groups > 1:
        return GroupedLinear(dim_in, dim_out, groups, bias = bias)

    return nn.Linear(dim_in, dim_out, bias = bias)

# attn res / mhar

class AttentionResidual(Module):
    def __init__(self, dim, streams = 3, groups = 1, merge_groups = False, eps = 1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps = eps)
        self.pseudo_queries = nn.Parameter(torch.zeros(streams, groups, dim)) # zero -> uniform mix
        self.merge_groups = Rearrange('s 1 b n d -> s b n d') if merge_groups else None

    def forward(self, history):
        values = torch.stack(history) # l b n d
        keys = self.norm(values)

        logits = einsum(keys, self.pseudo_queries, 'l b n d, s g d -> s g l b n')
        weights = logits.softmax(dim = 2) # over depth

        out = einsum(weights, values, 's g l b n, l b n d -> s g b n d')

        if exists(self.merge_groups):
            out = self.merge_groups(out)

        return out.unbind()

def FeedForward(dim, hidden_dim, heads = 1):
    return nn.Sequential(
        nn.LayerNorm(dim),
        linear(dim, hidden_dim, heads, bias = True),
        nn.GELU(),
        linear(hidden_dim, dim, bias = True)
    )

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, head_groups = 1, gate = True, eps = 1e-6):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        assert 1 <= head_groups <= heads and divisible_by(heads, head_groups), 'head_groups must divide heads'

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.norm_q = nn.RMSNorm(dim, eps = eps)
        self.norm_k = nn.RMSNorm(dim, eps = eps)
        self.norm_v = nn.RMSNorm(dim, eps = eps)

        self.to_q = linear(dim, inner_dim, head_groups)
        self.to_k = linear(dim, inner_dim, head_groups)
        self.to_v = linear(dim, inner_dim, head_groups)
        self.to_out = linear(inner_dim, dim)

        # headwise gate, each group from its own depth mix

        self.to_gate = nn.Sequential(
            linear(dim, heads, head_groups),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if gate else None

    def forward(self, xq, xk, xv):
        # (b n d) or (g b n d) when head_groups > 1

        xq, xk, xv = self.norm_q(xq), self.norm_k(xk), self.norm_v(xv)

        q, k, v = map(self.split_heads, (self.to_q(xq), self.to_k(xk), self.to_v(xv)))

        logits = einsum(q, k, 'b h n d, b h m d -> b h n m') * self.scale
        attn = logits.softmax(dim = -1)

        out = einsum(attn, v, 'b h n m, b h m d -> b h n d')

        if exists(self.to_gate):
            out = out * self.to_gate(xq)

        return self.to_out(self.merge_heads(out))

class TransformerBlock(Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, head_groups = 1, ffn_heads = 1, gate = True, eps = 1e-6):
        super().__init__()
        self.head_groups = head_groups

        self.attn_res = AttentionResidual(
            dim,
            streams = 3,
            groups = head_groups,
            merge_groups = head_groups == 1,
            eps = eps
        )
        self.attn = Attention(dim, heads = heads, dim_head = dim_head, head_groups = head_groups, gate = gate, eps = eps)

        self.ffn_res = AttentionResidual(dim, streams = 1, groups = ffn_heads, eps = eps) if ffn_heads > 1 else None
        self.ff = FeedForward(dim, mlp_dim, heads = ffn_heads)

    def forward(self, history):
        xq, xk, xv = self.attn_res(history)

        # group value mix as residual

        res = reduce(xv, 'g b n d -> b n d', 'mean') if self.head_groups > 1 else xv
        x = self.attn(xq, xk, xv) + res

        # each ffn head can mix depth too

        ffn_in = last(self.ffn_res([*history, x])) if exists(self.ffn_res) else x
        return x + self.ff(ffn_in)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, head_groups = 1, ffn_heads = 1, gate = True, eps = 1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps = eps)
        self.layers = ModuleList([
            TransformerBlock(dim, heads, dim_head, mlp_dim, head_groups = head_groups, ffn_heads = ffn_heads, gate = gate, eps = eps)
            for _ in range(depth)
        ])

    def forward(self, history):
        for layer in self.layers:
            history.append(layer(history))

        return self.norm(last(history))

class SimpleViT(Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        dim_head = 64,
        mlp_dim,
        channels = 3,
        head_groups = 1,
        ffn_heads = 1,
        gate = True,
        eps = 1e-6,
        temperature = 10000
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert divisible_by(image_height, patch_height) and divisible_by(image_width, patch_width), 'image dimensions must be divisible by the patch size'
        assert ffn_heads >= 1, 'ffn_heads must be at least 1'
        assert divisible_by(mlp_dim, ffn_heads), 'mlp_dim must be divisible by ffn_heads'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim, bias = False),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim, bias = False),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
            temperature = temperature
        )

        self.initial_ff = FeedForward(dim, mlp_dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, head_groups = head_groups, ffn_heads = ffn_heads, gate = gate, eps = eps)

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device, dtype = img.device, img.dtype

        x = self.to_patch_embedding(img)
        x = x + self.pos_embedding.to(device, dtype = dtype)

        history = [x, self.initial_ff(x)] # two depth outputs to start
        x = self.transformer(history)

        x = reduce(x, 'b n d -> b d', 'mean')
        return self.linear_head(x)

# main

if __name__ == '__main__':
    for head_groups, ffn_heads in ((1, 1), (2, 4), (4, 2)):
        vit = SimpleViT(
            image_size = 32,
            patch_size = 8,
            num_classes = 10,
            dim = 32,
            depth = 2,
            heads = 4,
            dim_head = 8,
            mlp_dim = 64,
            head_groups = head_groups,
            ffn_heads = ffn_heads
        )

        images = torch.randn(2, 3, 32, 32)
        logits = vit(images)

        assert logits.shape == (2, 10)

        logits.sum().backward()

        block = last(vit.transformer.layers)
        assert exists(block.attn_res.pseudo_queries.grad)
        assert not exists(block.ffn_res) or exists(block.ffn_res.pseudo_queries.grad)
        assert exists(block.attn.to_gate[0].weight.grad)

        label = 'shared' if head_groups == 1 else f'{head_groups} groups'
        print(f'multiway attn residual ({label}, ffn_heads = {ffn_heads}): ok')
