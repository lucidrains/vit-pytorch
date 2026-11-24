"""
multihead attention residuals for ViT, combining
https://arxiv.org/abs/2603.15031 (Attention Residuals, Kimi)
https://arxiv.org/abs/2607.27230 (Multi-Head Attention Residuals, Luo et al.)
https://arxiv.org/abs/2502.06785 (DeepCrossAttention, DCA)

softmax over preceding depth outputs with zero-init pseudo-queries (AttnRes)
q / k / v each get their own depth mix (DCA); v and ff mixes double as residuals

improvisations
- attn_res_heads: depth mix heads for attention (1 = shared, heads = per-head)
- ff_res_heads: depth mix heads and proj_in slices for feedforward (1 = shared)
- gate: headwise sigmoid gate on attention output
"""

import torch
from torch import cat, stack
from torch.nn import Module, ModuleList, Sequential, Linear, RMSNorm, LayerNorm, GELU, Sigmoid, Parameter

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

def posemb_sincos_2d(h, w, dim, temperature = 10000, dtype = torch.float32):
    assert divisible_by(dim, 4), 'feature dimension must be multiple of 4 for sincos emb'

    omega = torch.arange(dim // 4) / max(dim // 4 - 1, 1)
    omega = temperature ** -omega

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing = 'ij')
    y = rearrange(y, 'h w -> (h w) 1') * omega
    x = rearrange(x, 'h w -> (h w) 1') * omega

    pos = cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = -1)
    return pos.type(dtype)

# classes

# block-diagonal linear across groups, same params / flops as dense

class GroupedLinear(Module):
    def __init__(self, dim_in, dim_out, groups = 1, bias = False):
        super().__init__()
        assert groups >= 1 and divisible_by(dim_out, groups), 'dim_out must be divisible by groups'
        dim_group = dim_out // groups

        bound = dim_in ** -0.5
        self.weight = Parameter(torch.empty(groups, dim_group, dim_in).uniform_(-bound, bound))
        self.bias = Parameter(torch.zeros(dim_out)) if bias else None

    def forward(self, x):
        # g ... d -> ... (g e)

        out = einsum(x, self.weight, 'g ... d, g e d -> ... g e')
        out = rearrange(out, '... g e -> ... (g e)')

        return out + self.bias if exists(self.bias) else out

# multihead attn residual

class MultiHeadAttentionResidual(Module):
    def __init__(self, dim, num_streams = 1, heads = 1, eps = 1e-6):
        super().__init__()
        assert num_streams >= 1 and heads >= 1
        self.num_streams = num_streams

        self.norm = RMSNorm(dim, eps = eps)
        self.pseudo_queries = Parameter(torch.zeros(num_streams, heads, dim)) # zero -> uniform mix

    def forward(self, history):
        values = stack(history) # l b n d
        keys = self.norm(values)

        sim = einsum(keys, self.pseudo_queries, 'l b n d, s h d -> s h b n l')
        attn = sim.softmax(dim = -1) # over depth

        out = einsum(attn, values, 's h b n l, l b n d -> s h b n d')

        if self.num_streams == 1:
            return rearrange(out, '1 ... -> ...')

        return out.unbind()

def FeedForward(dim, hidden_dim, groups = None, eps = 1e-6):
    proj_in = GroupedLinear(dim, hidden_dim, groups, bias = True) if exists(groups) else Linear(dim, hidden_dim)

    return Sequential(
        RMSNorm(dim, eps = eps),
        proj_in,
        GELU(),
        Linear(hidden_dim, dim)
    )

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        res_heads = 1,
        gate = True,
        eps = 1e-6
    ):
        super().__init__()
        assert 1 <= res_heads <= heads and divisible_by(heads, res_heads), 'res_heads must divide heads'
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.norm_q = RMSNorm(dim, eps = eps)
        self.norm_k = RMSNorm(dim, eps = eps)
        self.norm_v = RMSNorm(dim, eps = eps)

        self.to_q = GroupedLinear(dim, inner_dim, res_heads)
        self.to_k = GroupedLinear(dim, inner_dim, res_heads)
        self.to_v = GroupedLinear(dim, inner_dim, res_heads)

        # headwise gate from query depth mix

        self.to_gate = Sequential(
            GroupedLinear(dim, heads, res_heads),
            Rearrange('b n h -> b h n 1'),
            Sigmoid()
        ) if gate else None

        self.to_out = Linear(inner_dim, dim, bias = False)

    def forward(self, q_input, k_input, v_input):
        # h b n d

        q_input = self.norm_q(q_input)
        k_input = self.norm_k(k_input)
        v_input = self.norm_v(v_input)

        q, k, v = self.to_q(q_input), self.to_k(k_input), self.to_v(v_input)
        q, k, v = map(self.split_heads, (q, k, v))

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        if exists(self.to_gate):
            out = out * self.to_gate(q_input)

        return self.to_out(self.merge_heads(out))

class Transformer(Module):
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        dim_head = 64,
        mlp_dim = 2048,
        attn_res_heads = 1,
        ff_res_heads = 1,
        gate = True,
        eps = 1e-6
    ):
        super().__init__()
        self.init_ff = FeedForward(dim, mlp_dim, eps = eps)

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                MultiHeadAttentionResidual(dim, num_streams = 3, heads = attn_res_heads, eps = eps),
                Attention(dim, heads = heads, dim_head = dim_head, res_heads = attn_res_heads, gate = gate, eps = eps),
                MultiHeadAttentionResidual(dim, num_streams = 1, heads = ff_res_heads, eps = eps),
                FeedForward(dim, mlp_dim, groups = ff_res_heads, eps = eps)
            ]))

        self.norm = RMSNorm(dim, eps = eps)

    def forward(self, x):
        history = [x, self.init_ff(x)] # two depth outputs to start

        for attn_res, attn, ff_res, ff in self.layers:
            q_input, k_input, v_input = attn_res(history)

            # value mix doubles as residual

            residual = reduce(v_input, 'h b n d -> b n d', 'mean')
            x = attn(q_input, k_input, v_input) + residual

            # ff mixes history and post-attn state

            ff_input = ff_res([*history, x])
            residual = reduce(ff_input, 'h b n d -> b n d', 'mean')
            x = ff(ff_input) + residual

            history.append(x)

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
        mlp_dim,
        channels = 3,
        dim_head = 64,
        attn_res_heads = 1,
        ff_res_heads = 1,
        gate = True,
        eps = 1e-6,
        temperature = 10000
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert divisible_by(image_height, patch_height) and divisible_by(image_width, patch_width), 'image dimensions must be divisible by the patch size'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            LayerNorm(patch_dim, bias = False),
            Linear(patch_dim, dim),
            LayerNorm(dim, bias = False)
        )

        self.register_buffer('pos_embedding', posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
            temperature = temperature
        ), persistent = False)

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            mlp_dim = mlp_dim,
            attn_res_heads = attn_res_heads,
            ff_res_heads = ff_res_heads,
            gate = gate,
            eps = eps
        )

        self.linear_head = Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        x = x + self.pos_embedding.to(x.dtype)

        x = self.transformer(x)
        x = reduce(x, 'b n d -> b d', 'mean')

        return self.linear_head(x)

# main

if __name__ == '__main__':
    for attn_res_heads, ff_res_heads, gate in ((1, 1, True), (2, 4, True), (4, 2, False)):
        vit = SimpleViT(
            image_size = 32,
            patch_size = 8,
            num_classes = 10,
            dim = 32,
            depth = 2,
            heads = 4,
            dim_head = 8,
            mlp_dim = 64,
            attn_res_heads = attn_res_heads,
            ff_res_heads = ff_res_heads,
            gate = gate
        )

        images = torch.randn(2, 3, 32, 32)
        logits = vit(images)

        assert logits.shape == (2, 10)

        logits.sum().backward()

        attn_res, attn, ff_res, _ = last(vit.transformer.layers)
        assert exists(attn_res.pseudo_queries.grad) and attn_res.pseudo_queries.grad.abs().sum() > 0
        assert exists(ff_res.pseudo_queries.grad) and ff_res.pseudo_queries.grad.abs().sum() > 0
        assert not gate or exists(attn.to_gate[0].weight.grad)

        label = 'shared' if attn_res_heads == 1 else f'{attn_res_heads} heads'
        print(f'multihead attn residual ({label}, ff_res_heads = {ff_res_heads}, gate = {gate}): ok')
