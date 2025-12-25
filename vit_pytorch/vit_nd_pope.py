from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import pi, nn, arange, cat, stack, Tensor
from torch.nn import Module, ModuleList
from torch.amp import autocast

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

# but using polar version instead
# Gopalakrishnan et al. https://arxiv.org/abs/2509.10534

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

class GoldenGatePoPENd(Module):
    def __init__(
        self,
        dim_pos: int,
        heads: int,
        dim_head: int,
        min_freq: float = 1.0,
        max_freq: float = 10000.0,
        p_zero_freqs: float = 0.0, # proportion of frequencies set to 0
        init_learned_bias_uniform = False
    ):
        super().__init__()
        n_freqs = dim_head
        n_zero_freqs = round(p_zero_freqs * n_freqs)

        omega = cat((
            torch.zeros(n_zero_freqs),
            min_freq * (max_freq / min_freq) ** torch.linspace(0, 1, n_freqs - n_zero_freqs),
        ))

        directions = rearrange(
            make_directions(heads * n_freqs, dim_pos),
            '(h f) p -> h f p',
            h = heads
        )

        omega_expanded = rearrange(omega, 'f -> f 1')
        self.register_buffer('freqs', directions * omega_expanded)  # shape: (h, f, p)

        self.learned_bias = nn.Parameter(torch.zeros(heads, dim_head))

        if init_learned_bias_uniform:
            self.learned_bias.uniform_(-2. * pi, 0.)

    @autocast('cuda', enabled = False)
    def forward(self, pos):

        freqs = rearrange(self.freqs, 'h f p -> 1 h 1 f p')
        positions = rearrange(pos.float(), 'b n p -> b 1 n 1 p')

        # compute theta for each (batch, head, seq, freq)

        theta = reduce(freqs * positions, 'b h n f p -> b h n f', 'sum')

        bias = self.learned_bias.clamp(-2. * pi, 0.)
        bias = rearrange(bias, 'h d -> h 1 d')

        return theta, bias

@autocast('cuda', enabled = False)
def apply_polar_pos_emb(t, freqs):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype
    freqs = freqs[:, -seq_len:]

    t = t.float()

    t = F.softplus(t)
    out = cat((t * freqs.cos(), t * freqs.sin()), dim = -1)

    return out.type(orig_dtype)

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
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x, polar_pos_emb = None):
        x = self.norm(x)
        qkv = (*self.to_qk(x).chunk(2, dim = -1), self.to_v(x))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        if exists(polar_pos_emb):
            freqs, bias = polar_pos_emb
            q = apply_polar_pos_emb(q, freqs)
            k = apply_polar_pos_emb(k, freqs + bias)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., polar_emb = None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.polar_emb = polar_emb

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    
    def forward(self, x, pos = None):

        # pope embedding

        polar_pos_emb = None
        if exists(pos) and exists(self.polar_emb):
            polar_pos_emb = self.polar_emb(pos)

        # transformer layers

        for attn, ff in self.layers:
            x = attn(x, polar_pos_emb) + x
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
        pope_min_freq: float = 1.0,
        pope_max_freq: float = 10000.0,
        pope_p_zero_freqs: float = 0.0,
        pope_init_learned_bias_uniform = False
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
        
        # golden gate pope

        self.polar_emb = GoldenGatePoPENd(
            dim_pos = ndim,
            heads = heads,
            dim_head = dim_head,
            min_freq = pope_min_freq,
            max_freq = pope_max_freq,
            p_zero_freqs = pope_p_zero_freqs,
            init_learned_bias_uniform = pope_init_learned_bias_uniform
        )
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, polar_emb = self.polar_emb)
        
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

    data = torch.randn(3, 3, 4, 8, 16, 32, 64)

    logits = model(data)

    embed = model(data, return_embed = True) # (2, 2, 4, 4, 8, 8, 512)
