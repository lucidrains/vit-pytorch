import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(num, den):
    return (num % den) == 0

def posemb_sincos_2d(h, w, dim, temperature = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing = 'ij')
    assert divisible_by(dim, 4), 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

def embed_small_init(*shape, std = 0.02):
    return nn.Parameter(torch.randn(*shape) * std)

# classes

class RelativeProjPositionalBias(Module):
    """
    relative position bias via learned distance basis projection
    (Inkling model from Thinking Machines, generalized to N-D)
    """

    def __init__(
        self,
        dim,
        heads,
        dim_pos = 2,
        num_distance_basis = 16,
        max_dist = 512
    ):
        super().__init__()
        self.heads = heads
        self.dim_pos = dim_pos
        self.num_distance_basis = num_distance_basis
        self.max_dist = max_dist

        self.to_distance_weights = nn.Sequential(
            nn.Linear(dim, heads * dim_pos * num_distance_basis),
            Rearrange('b ... (w h d) -> w b h (...) d', w = dim_pos, h = heads)
        )

        self.distance_banks = embed_small_init(dim_pos, num_distance_basis, 2 * max_dist - 1)

    def forward(self, x):
        spatial_shape = x.shape[1:-1]
        dim_pos = len(spatial_shape)
        assert dim_pos == self.dim_pos, f'expected input with {self.dim_pos} spatial dimensions, but got {dim_pos}'

        b, device, heads = x.shape[0], x.device, self.heads

        # grid coordinates

        grid_coords = torch.meshgrid(
            *(torch.arange(s, device = device) for s in spatial_shape),
            indexing = 'ij'
        )
        flat_coords = [g.flatten() for g in grid_coords]

        # relative distances

        rel_dists = [pos[:, None] - pos[None, :] + self.max_dist - 1 for pos in flat_coords]

        valid_masks = [(rel >= 0) & (rel < 2 * self.max_dist - 1) for rel in rel_dists]
        valid_mask = torch.stack(valid_masks).all(dim = 0)

        clamped_dists = torch.stack([rel.clamp(0, 2 * self.max_dist - 2) for rel in rel_dists], dim = 0)

        # distance weight projection

        weights = self.to_distance_weights(x)

        # curves and gather

        curves = einsum('w b h n d, w d r -> w b h n r', weights, self.distance_banks)

        biases = curves.gather(-1, repeat(clamped_dists, 'w i j -> w b h i j', b = b, h = heads))

        bias = reduce(biases, 'w b h i j -> b h i j', 'sum')
        return bias.masked_fill(~valid_mask, 0.)

class FeedForward(Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, num_distance_basis = 16, max_dist = 512):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.rel_pos_bias = RelativeProjPositionalBias(
            dim = dim,
            heads = heads,
            dim_pos = 2,
            num_distance_basis = num_distance_basis,
            max_dist = max_dist
        )

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        # relative position bias

        bias = self.rel_pos_bias(x)

        # attention

        x, ps = pack([x], 'b * d')
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = dots + bias

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        out, = unpack(out, ps, 'b * d')
        return out

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_distance_basis = 16, max_dist = 512):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, num_distance_basis = num_distance_basis, max_dist = max_dist),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

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
        num_distance_basis = 16,
        max_dist = 512
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.patch_size = patch_height, patch_width = pair(patch_size)

        assert divisible_by(image_height, patch_height) and divisible_by(image_width, patch_width), 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width
        self.h = image_height // patch_height
        self.w = image_width // patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = self.h,
            w = self.w,
            dim = dim,
        )

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            mlp_dim = mlp_dim,
            num_distance_basis = num_distance_basis,
            max_dist = max_dist
        )

        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype = x.dtype)

        x = rearrange(x, 'b (h w) d -> b h w d', h = self.h, w = self.w)

        x = self.transformer(x)
        x = reduce(x, 'b h w d -> b d', 'mean')

        x = self.to_latent(x)
        return self.linear_head(x)

if __name__ == '__main__':
    v = SimpleViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 2,
        heads = 8,
        mlp_dim = 2048
    )

    img = torch.randn(2, 3, 256, 256)
    out = v(img)
    assert out.shape == (2, 1000)

    out.sum().backward()

    # test 1d and 3d relative position bias

    rel_pos_bias_1d = RelativeProjPositionalBias(dim = 64, heads = 4, dim_pos = 1)
    seq_1d = torch.randn(2, 16, 64)
    bias_1d = rel_pos_bias_1d(seq_1d)
    assert bias_1d.shape == (2, 4, 16, 16)

    rel_pos_bias_3d = RelativeProjPositionalBias(dim = 64, heads = 4, dim_pos = 3)
    vol_3d = torch.randn(2, 4, 4, 4, 64)
    bias_3d = rel_pos_bias_3d(vol_3d)
    assert bias_3d.shape == (2, 4, 64, 64)
