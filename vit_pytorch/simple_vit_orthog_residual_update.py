# Revisiting Residual Connections: Orthogonal Updates for Stable and Efficient Deep Networks
# Giyeong Oh et al. https://arxiv.org/abs/2505.11881

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def exists(v):
    return v is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

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

class OrthogonalResidualUpdate(Module):
    def __init__(
        self,
        block: Module,
        dim = None,
        double_precision = True,
        learned = False
    ):
        super().__init__()
        self.block = block
        self.double_precision = double_precision

        self.learned = learned

        if learned:
            assert exists(dim)
            self.to_modulation = nn.Linear(dim, 2)

    def orthog_proj(self, block_out, residual):
        use_double, dtype = self.double_precision, residual.dtype

        if use_double:
            residual, block_out = residual.double(), block_out.double()

        # get orthogonal projection of the attention or feedforward output respect to residual

        unit = F.normalize(residual, dim = -1)
        parallel = (block_out * unit).sum(dim = -1, keepdim = True) * unit
        orthogonal = block_out - parallel

        # back to original dtype if double precision

        if use_double:
            parallel, orthogonal = parallel.to(dtype), orthogonal.to(dtype)

        return parallel, orthogonal

    def forward(self, residual):
        block_out = self.block(residual)

        parallel_update, orthog_update = self.orthog_proj(block_out, residual)

        if self.learned:
            parallel_mod, orthog_mod = self.to_modulation(block_out).sigmoid().split(1, dim = -1)
            parallel_update = parallel_update * parallel_mod
            orthog_update = orthog_update * orthog_mod
        else:
            parallel_update = 0

        return residual + parallel_update + orthog_update

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, orthog_residual_update_kwargs: dict = dict()):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            attn = Attention(dim, heads = heads, dim_head = dim_head)
            ff = FeedForward(dim, mlp_dim)

            self.layers.append(ModuleList([
                OrthogonalResidualUpdate(attn, dim = dim, **orthog_residual_update_kwargs),
                OrthogonalResidualUpdate(ff, dim = dim, **orthog_residual_update_kwargs)
            ]))

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return self.norm(x)

class SimpleViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, orthog_residual_update_kwargs: dict = dict()):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

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

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, orthog_residual_update_kwargs)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

# quick test

if __name__ == '__main__':
    vit = SimpleViT(
        image_size = 256,
        patch_size = 16,
        num_classes = 10,
        dim = 512,
        depth = 2,
        heads = 4,
        mlp_dim = 2048,
        orthog_residual_update_kwargs = dict(
            learned = True
        )
    )

    images = torch.randn(2, 3, 256, 256)

    assert vit(images).shape == (2, 10)
