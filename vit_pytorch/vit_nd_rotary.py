from __future__ import annotations

import torch
from torch import nn, arange, cat, stack, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def join(arr, delimiter = ' '):
    return delimiter.join(arr)

def ensure_tuple(t, length):
    if isinstance(t, (tuple, list)):
        assert len(t) == length, f'Expected tuple of length {length}, got {len(t)}'
        return tuple(t)
    return (t,) * length

# golden gate rotary - Jerry Xiong, PhD student at UIUC
# https://jerryxio.ng/posts/nd-rope/

def _phi(m: int) -> float:
    x = 2.0
    for _ in range(10):
        x = (1 + x) ** (1.0 / (m + 1.0))
    return x

def make_directions(n: int, d: int) -> Tensor:
    g = _phi(d)
    alpha = (1.0 / g) ** arange(1, d + 1, dtype = torch.float64)
    i = arange(1, n + 1, dtype = torch.float64).unsqueeze(1)
    z = torch.fmod(i * alpha, 1.0)
    directions = torch.erfinv(2.0 * z - 1.0)
    directions = l2norm(directions)
    return directions.float()

class GoldenGateRoPENd(Module):
    def __init__(
        self,
        dim_pos: int,
        heads: int,
        dim_head: int,
        rope_min_freq: float = 1.0,
        rope_max_freq: float = 10000.0,
        rope_p_zero_freqs: float = 0.0, # proportion of frequencies set to 0
    ):
        super().__init__()
        n_freqs = dim_head // 2
        n_zero_freqs = round(rope_p_zero_freqs * n_freqs)

        omega = cat((
            torch.zeros(n_zero_freqs),
            rope_min_freq * (rope_max_freq / rope_min_freq) ** torch.linspace(0, 1, n_freqs - n_zero_freqs),
        ))

        directions = rearrange(
            make_directions(heads * n_freqs, dim_pos),
            '(h f) p -> h f p',
            h = heads
        )

        omega_expanded = rearrange(omega, 'f -> f 1')
        self.register_buffer('freqs', directions * omega_expanded)  # shape: (h, f, p)

    def forward(self, input: Tensor, pos: Tensor) -> Tensor:
        # input shape: (b, h, n, d) where d = head_dim
        # pos shape: (b, n, p) where p = pos_dim
        # self.freqs shape: (h, f, p) where f = d // 2
        
        x, y = input.float().chunk(2, dim = -1)  # both (b, h, n, f)
        
        # Expand dimensions for broadcasting
        freqs = rearrange(self.freqs, 'h f p -> 1 h 1 f p')
        positions = rearrange(pos.float(), 'b n p -> b 1 n 1 p')
        
        # Compute theta for each (batch, head, seq, freq)
        theta = reduce(freqs * positions, 'b h n f p -> b h n f', 'sum')
        
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # Apply rotation
        x_out = x * cos_theta - y * sin_theta
        y_out = x * sin_theta + y * cos_theta
        
        output = cat((x_out, y_out), dim=-1)
        return output.type_as(input)

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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., rotary_emb = None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.rotary_emb = rotary_emb
        
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x, pos = None):
        x = self.norm(x)
        qkv = (*self.to_qk(x).chunk(2, dim = -1), self.to_v(x))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        # Apply rotary embeddings if available
        if exists(self.rotary_emb):
            assert exists(pos)
            q = self.rotary_emb(q, pos)
            k = self.rotary_emb(k, pos)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., rotary_emb = None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, rotary_emb = rotary_emb),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    
    def forward(self, x, pos = None):
        for attn, ff in self.layers:
            x = attn(x, pos) + x
            x = ff(x) + x
        return self.norm(x)

class ViTND(Module):
    def __init__(
        self,
        *,
        ndim: int,
        input_shape: int | tuple[int, ...],
        patch_size: int | tuple[int, ...],
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.,
        emb_dropout: float = 0.,
        rope_min_freq: float = 1.0,
        rope_max_freq: float = 10000.0,
        rope_p_zero_freqs: float = 0.0
    ):
        super().__init__()
        
        assert 1 <= ndim <= 7, 'ndim must be between 1 and 7'
        
        self.ndim = ndim
        
        input_shape = ensure_tuple(input_shape, ndim)
        patch_size = ensure_tuple(patch_size, ndim)
        
        for i, (inp_dim, patch_dim) in enumerate(zip(input_shape, patch_size)):
            assert inp_dim % patch_dim == 0, f'Input dimension {i} ({inp_dim}) must be divisible by patch size ({patch_dim})'
        
        num_patches_per_dim = [inp_dim // patch_dim for inp_dim, patch_dim in zip(input_shape, patch_size)]
        num_patches = 1
        for n in num_patches_per_dim:
            num_patches *= n
        
        patch_dim = channels
        for p in patch_size:
            patch_dim *= p
        
        dim_names = 'fghijkl'[:ndim]
        
        input_dims = [f'({d} p{i})' for i, d in enumerate(dim_names)]
        patch_dims = [f'p{i}' for i in range(ndim)]
        
        input_pattern = f'b c {join(input_dims)}'
        output_pattern = f'b {join(dim_names)} ({join(patch_dims)} c)'
        rearrange_str = f'{input_pattern} -> {output_pattern}'
        
        rearrange_kwargs = {f'p{i}': p for i, p in enumerate(patch_size)}
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange(rearrange_str, **rearrange_kwargs),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        self.dropout = nn.Dropout(emb_dropout)
        
        # Create rotary embeddings
        self.rotary_emb = GoldenGateRoPENd(
            dim_pos = ndim,
            heads = heads,
            dim_head = dim_head,
            rope_min_freq = rope_min_freq,
            rope_max_freq = rope_max_freq,
            rope_p_zero_freqs = rope_p_zero_freqs
        )
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, rotary_emb = self.rotary_emb)
        
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)
    
    def muon_parameters(self):
        params = []

        for m in self.modules():
            if isinstance(m, Attention):
                params.extend([
                    m.to_v.weight,
                    m.to_out[0].weight
                ])
            elif isinstance(m, FeedForward):
                params.extend([
                    m.net[1].weight,
                    m.net[-2].weight
                ])

        return params

    def forward(
        self,
        x,
        return_embed = False
    ):
        x = self.to_patch_embedding(x) # (b, *spatial_dims, patch_dim)
        
        batch, *spatial_dims, _, device = *x.shape, x.device
        
        # Generate position coordinates

        grids = [arange(d, device = device, dtype = torch.float32) for d in spatial_dims]
        grid = torch.meshgrid(*grids, indexing = 'ij')
        pos = stack(grid, dim = -1)  # (*spatial_dims, ndim)

        # flatten spatial dimensions for attention with nd rotary
        
        pos = repeat(pos, '... p -> b (...) p', b = batch)
        x, packed_shape = pack([x], 'b * d')

        x = self.dropout(x)
        
        embed = self.transformer(x, pos)

        # return the embed with reconstituted patch shape

        if return_embed:
            embed, = unpack(embed, packed_shape, 'b * d')
            return embed

        # pooling to logits

        pooled = reduce(embed, 'b n d -> b d', 'mean')

        pooled = self.to_latent(pooled)
        return self.mlp_head(pooled)


if __name__ == '__main__':
  
    model = ViTND(
        ndim = 5,
        input_shape = (4, 8, 16, 32, 64),
        patch_size = (2, 2, 4, 4, 8),
        num_classes = 1000,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        channels = 3,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    data = torch.randn(2, 3, 4, 8, 16, 32, 64)

    logits = model(data)

    embed = model(data, return_embed = True) # (2, 2, 4, 4, 8, 8, 512)
