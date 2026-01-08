from collections import namedtuple

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.nn.attention import SDPBackend, sdpa_kernel

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def divisible_by(num, den):
    return (num % den) == 0

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., use_flash_attn = True):
        super().__init__()
        self.use_flash_attn = use_flash_attn
        self.dropout_p = dropout
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

    def flash_attn(self, q, k, v, mask = None):

        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]):

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout_p,
                is_causal = False,
                scale = self.scale
            )

        return out

    def forward(self, x, mask = None):
        batch, seq, _ = x.shape

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')

        if self.use_flash_attn:
            out =  self.flash_attn(q, k, v, mask = mask)

        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            if exists(mask):
                mask = rearrange(mask, 'b j -> b 1 1 j')
                dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., use_flash_attn = True):
        super().__init__()
        self.use_flash_attn = use_flash_attn

        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, mask = None):

        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return self.norm(x)

class FactorizedTransformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., use_flash_attn = True):
        super().__init__()
        self.use_flash_attn = use_flash_attn

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, use_flash_attn = use_flash_attn),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, use_flash_attn = use_flash_attn),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, mask = None):
        batch, frames, seq, _ = x.shape

        if exists(mask):
            mask = repeat(mask, 'b ... -> (b space) ...', space = x.shape[2])

        for spatial_attn, temporal_attn, ff in self.layers:
            x = rearrange(x, 'b f n d -> (b f) n d')
            x = spatial_attn(x) + x
            x = rearrange(x, '(b f) n d -> (b n) f d', b = batch, f = frames)
            x = temporal_attn(x, mask = mask) + x
            x = ff(x) + x
            x = rearrange(x, '(b n) f d -> b f n d', b = batch, n = seq)

        return self.norm(x)

class ViViT(Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        spatial_depth,
        temporal_depth,
        heads,
        mlp_dim,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        variant = 'factorized_encoder',
        use_flash_attn: bool = True,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert divisible_by(image_height, patch_height) and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert divisible_by(frames, frame_patch_size), 'Frames must be divisible by frame patch size'
        assert variant in ('factorized_encoder', 'factorized_self_attention'), f'variant = {variant} is not implemented'

        num_image_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = (frames // frame_patch_size)

        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.frame_patch_size = frame_patch_size

        self.global_average_pool = pool == 'mean'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (pf p1 p2 c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None

        if variant == 'factorized_encoder':
            self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
            self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout, use_flash_attn)
            self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout, use_flash_attn)
        elif variant == 'factorized_self_attention':
            assert spatial_depth == temporal_depth, 'Spatial and temporal depth must be the same for factorized self-attention'
            self.factorized_transformer = FactorizedTransformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout, use_flash_attn)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
        self.variant = variant

    def forward(self, video, mask = None):
        device = video.device

        x = self.to_patch_embedding(video)
        batch, frames, seq, _ = x.shape

        x = x + self.pos_embedding[:, :frames, :seq]

        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b = batch, f = frames)
            x = cat((spatial_cls_tokens, x), dim = 2)

        x = self.dropout(x)

        # maybe temporal mask

        temporal_mask = None

        if exists(mask):
            temporal_mask = reduce(mask, 'b (f patch) -> b f', 'all', patch = self.frame_patch_size)

        # the two variants

        if self.variant == 'factorized_encoder':
            x = rearrange(x, 'b f n d -> (b f) n d')

            # attend across space

            x = self.spatial_transformer(x)
            x = rearrange(x, '(b f) n d -> b f n d', b = batch)

            # excise out the spatial cls tokens or average pool for temporal attention

            x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')

            # append temporal CLS tokens

            if exists(self.temporal_cls_token):
                temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b = batch)

                x = cat((temporal_cls_tokens, x), dim = 1)

                if exists(temporal_mask):
                    temporal_mask = F.pad(temporal_mask, (1, 0), value = True)

            # attend across time

            x = self.temporal_transformer(x, mask = temporal_mask)

            # excise out temporal cls token or average pool

            x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

        elif self.variant == 'factorized_self_attention':

            x = self.factorized_transformer(x, mask = temporal_mask)

            x = x[:, 0, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b d', 'mean')

        x = self.to_latent(x)
        return self.mlp_head(x)

# main

if __name__ == '__main__':

    vivit = ViViT(
        dim = 512,
        spatial_depth = 2,
        temporal_depth = 2,
        heads = 4,
        mlp_dim = 2048,
        image_size = 256,
        image_patch_size = 16,
        frames = 8,
        frame_patch_size = 2,
        num_classes = 1000,
        variant = 'factorized_encoder',
    )

    video = torch.randn(3, 3, 8, 256, 256)
    mask = torch.randint(0, 2, (3, 8)).bool()

    logits = vivit(video, mask = None)
    assert logits.shape == (3, 1000)
