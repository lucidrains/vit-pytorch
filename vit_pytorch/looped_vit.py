# Looped Diffusion Transformer
# https://arxiv.org/abs/2609.40305

from __future__ import annotations
from math import prod

import torch
from torch import nn, Tensor, stack
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def join(arr, delimiter = ' '):
    return delimiter.join(arr)

def ensure_tuple(t, length):
    if isinstance(t, (tuple, list)):
        assert len(t) == length, f'expected tuple of length {length}, got {len(t)}'
        return tuple(t)
    return (t,) * length

# deep supervision

deep_supervision_weightings = dict(
    final = lambda n: torch.cat((torch.zeros(n - 1), torch.ones(1))),
    exponential = lambda n: 2. ** (torch.arange(n) - (n - 1)),
    final_mean = lambda n: torch.cat((torch.full((n - 1,), 1. / max(n - 1, 1)), torch.ones(1)))
)

def deep_supervision_weights(num_loops, weighting = 'final_mean'):
    if isinstance(weighting, str):
        assert weighting in deep_supervision_weightings, f'unknown weighting {weighting}'
        return deep_supervision_weightings[weighting](num_loops)

    weights = torch.as_tensor(weighting, dtype = torch.float32)
    assert weights.shape == (num_loops,), f'expected {num_loops} loop weights, one per loop'
    return weights

def per_loop_losses(logits, labels, label_smoothing = 0.):
    if logits.ndim == 2:
        logits = rearrange(logits, 'b ... -> 1 b ...')

    loops = logits.shape[0]

    loss = F.cross_entropy(
        rearrange(logits, 'l b ... -> (l b) ...'),
        repeat(labels, 'b -> (l b)', l = loops),
        reduction = 'none',
        label_smoothing = label_smoothing
    )

    return rearrange(loss, '(l b) -> l b', l = loops)

def deep_supervision_loss(logits, labels, weighting = 'final_mean', label_smoothing = 0.):
    if logits.ndim == 2:
        return F.cross_entropy(logits, labels, label_smoothing = label_smoothing)

    loops = logits.shape[0]

    losses = per_loop_losses(logits, labels, label_smoothing = label_smoothing)
    weights = deep_supervision_weights(loops, weighting).to(losses)

    return (losses * rearrange(weights, 'l -> l 1')).sum(dim = 0).mean()

@torch.no_grad()
def adaptive_exit_targets(logits, labels, label_smoothing = 0.):
    loops = logits.shape[0]
    assert loops > 1, 'need at least 2 loops to compute adaptive exit targets'

    losses = per_loop_losses(logits, labels, label_smoothing = label_smoothing)

    loop_index = torch.arange(loops, device = losses.device)
    r, k = rearrange(loop_index, 'l -> l 1 1'), rearrange(loop_index, 'l -> 1 l 1')

    diffs = rearrange(losses, 'l b -> l 1 b') - rearrange(losses, 'l b -> 1 l b')
    gains = torch.where(k > r, diffs / (k - r).clamp(min = 1), 0.)

    return gains.clamp(min = 0.).amax(dim = 1)[:-1]

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
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = dropout

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_gate = nn.Linear(dim, heads, bias = False)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias = False), nn.Dropout(dropout))

    def forward(self, x):
        normed = self.norm(x)

        q, k, v = self.to_qkv(normed).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        out = F.scaled_dot_product_attention(
            q, k, v,
            scale = self.scale,
            dropout_p = self.dropout if self.training else 0.
        )

        # exclusive self-attention

        v_hat = F.normalize(v, dim = -1)
        out = out - (out * v_hat).sum(dim = -1, keepdim = True) * v_hat

        # headwise gate

        gate = self.to_gate(normed).sigmoid()
        out = out * rearrange(gate, 'b n h -> b h n 1')

        out = self.merge_heads(out)
        return self.to_out(out)

class Transformer(Module):
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        dim_head = 64,
        mlp_dim = 2048,
        dropout = 0.
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(
                    dim,
                    heads = heads,
                    dim_head = dim_head,
                    dropout = dropout
                ),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class AdaptiveLoopGate(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        num_queries = 4,
        mlp_mult = 2,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = dropout

        self.queries = nn.Parameter(torch.randn(heads, num_queries, dim_head) * dim_head ** -0.5)

        self.norm_kv = nn.LayerNorm(dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_queries = Rearrange('b h q d -> b q (h d)')

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mlp_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_mult, 1),
            Rearrange('... 1 -> ...')
        )

    def forward(self, x):
        batch = x.shape[0]

        q = repeat(self.queries, 'h q d -> b h q d', b = batch)
        k, v = self.to_kv(self.norm_kv(x)).chunk(2, dim = -1)
        k, v = map(self.split_heads, (k, v))

        out = F.scaled_dot_product_attention(
            q, k, v,
            scale = self.scale,
            dropout_p = self.dropout if self.training else 0.
        )
        out = self.merge_queries(out)

        out = self.to_out(out).mean(dim = 1)
        return self.mlp(out)

class LoopedViT(Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        ndim = 2,
        input_shape = None,
        patch_size = None,
        image_size = None,
        pre_loop_depth = 6,
        num_looped_blocks = 5,
        post_loop_depth = 6,
        loop_depth = 4,
        deep_supervision_weighting = 'final_mean',
        pool = 'cls',
        heads = 8,
        dim_head = 64,
        mlp_dim = 2048,
        channels = 3,
        dropout = 0.,
        emb_dropout = 0.,
        adaptive_loop = False,
        exit_threshold = 0.
    ):
        super().__init__()

        input_shape = default(input_shape, image_size)
        assert exists(input_shape), 'input_shape (or image_size) must be specified'

        if isinstance(input_shape, (tuple, list)):
            ndim = len(input_shape)

        assert 1 <= ndim <= 7, 'ndim must be between 1 and 7'
        assert pool in {'cls', 'mean'}, 'pool must be cls (cls token) or mean (mean pooling)'

        self.pool = pool
        self.loop_depth = loop_depth
        self.exit_threshold = exit_threshold

        self.register_buffer(
            'loop_weights',
            deep_supervision_weights(loop_depth, deep_supervision_weighting),
            persistent = False
        )

        input_shape = ensure_tuple(input_shape, ndim)
        patch_size = ensure_tuple(patch_size, ndim)

        divisible = all(inp % patch == 0 for inp, patch in zip(input_shape, patch_size))
        assert divisible, 'input dimensions must be divisible by the patch sizes'

        dims = 'fghijkl'[:ndim]
        src = join(f'({d} p{i})' for i, d in enumerate(dims))
        dst = join(f'p{i}' for i in range(ndim))

        patch_pattern = f'b c {src} -> b ({join(dims)}) ({dst} c)'
        patch_sizes = {f'p{i}': p for i, p in enumerate(patch_size)}

        self.to_patch_embedding = nn.Sequential(
            Rearrange(patch_pattern, **patch_sizes),
            nn.Linear(channels * prod(patch_size), dim),
            nn.LayerNorm(dim)
        )

        num_patches = prod(inp // patch for inp, patch in zip(input_shape, patch_size))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.pre_loop = Transformer(
            dim,
            pre_loop_depth,
            heads = heads,
            dim_head = dim_head,
            mlp_dim = mlp_dim,
            dropout = dropout
        )
        self.looped = Transformer(
            dim,
            num_looped_blocks,
            heads = heads,
            dim_head = dim_head,
            mlp_dim = mlp_dim,
            dropout = dropout
        )
        self.post_loop = Transformer(
            dim,
            post_loop_depth,
            heads = heads,
            dim_head = dim_head,
            mlp_dim = mlp_dim,
            dropout = dropout
        )

        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

        exit_gate = AdaptiveLoopGate(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.exit_gate = exit_gate if adaptive_loop else None

    def decode(self, x):
        return self.norm(self.post_loop(x))

    def forward(
        self,
        x,
        loop_depth = None,
        return_all_logits = False,
        return_exit_benefits = False,
        adaptive_exit = False,
        exit_threshold = None
    ):
        max_loops = default(loop_depth, self.loop_depth)
        threshold = default(exit_threshold, self.exit_threshold)

        assert max_loops >= 1, 'loop depth must be at least 1'

        needs_benefits = adaptive_exit or return_exit_benefits

        if needs_benefits:
            assert exists(self.exit_gate), 'exit gate requires adaptive_loop = True on init'
            assert max_loops > 1, 'adaptive looping with a single loop has nothing to gate'

        x = self.to_patch_embedding(x)
        batch = x.shape[0]

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch)
        x = torch.cat((cls_tokens, x), dim = 1)

        x = self.dropout(x + self.pos_embedding)
        x = self.pre_loop(x)

        logits_per_loop = []
        exit_benefits = []

        for loop in range(max_loops):
            x = self.looped(x)

            is_last = loop == (max_loops - 1)

            # decode intermediates only when deep supervising

            if not (is_last or needs_benefits or return_all_logits):
                continue

            decoded = self.decode(x)
            pooled = decoded[:, 1:].mean(dim = 1) if self.pool == 'mean' else decoded[:, 0]
            logits_per_loop.append(self.mlp_head(pooled))

            if is_last or not needs_benefits:
                continue

            benefit = self.exit_gate(decoded)
            exit_benefits.append(benefit)

            # early exit if benefit below threshold

            if adaptive_exit and bool(benefit.max() <= threshold):
                break

        logits = stack(logits_per_loop) if return_all_logits else logits_per_loop[-1]

        return (logits, stack(exit_benefits)) if return_exit_benefits else logits

if __name__ == '__main__':

    model = LoopedViT(
        image_size = 256,
        patch_size = 16,
        num_classes = 1000,
        dim = 512,
        heads = 8,
        dim_head = 64,
        mlp_dim = 2048,
        pre_loop_depth = 6,
        num_looped_blocks = 5,
        post_loop_depth = 6,
        loop_depth = 4,
        deep_supervision_weighting = 'final_mean',
        adaptive_loop = True
    )

    img = torch.randn(2, 3, 256, 256)
    labels = torch.randint(0, 1000, (2,))

    # deep supervision across loops

    logits, exit_benefits = model(
        img,
        return_all_logits = True,
        return_exit_benefits = True
    )

    loss = deep_supervision_loss(logits, labels, weighting = 'final_mean')

    # train adaptive loop exit gate

    targets = adaptive_exit_targets(logits, labels)
    gate_loss = F.mse_loss(exit_benefits, targets)

    total_loss = loss + gate_loss
    total_loss.backward()

    # inference - dial loop depth

    logits = model(img, loop_depth = 8)

    # inference - early exit

    logits_adaptive = model(img, adaptive_exit = True, exit_threshold = 0.05)
