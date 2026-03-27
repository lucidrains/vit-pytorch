import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def last(arr):
    return arr[-1]

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
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, context = None):
        x = self.norm(x)
        context = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = tuple(rearrange(t, 'b n (h d) -> b h n d', h = self.heads) for t in (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class AttentionPool(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, use_learned_query = True):
        super().__init__()
        self.use_learned_query = use_learned_query
        self.norm_context = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads = heads, dim_head = dim_head)

        if use_learned_query:
            self.query = nn.Parameter(torch.randn(dim))

    def forward(self, context, query = None):
        batch = context.shape[0]

        context = self.norm_context(context)

        if self.use_learned_query:
            q = repeat(self.query, 'd -> b 1 d', b = batch)
        else:
            q = query

        return self.attn(q, context = context)

class AttentionResidual(Module):
    """
    replaces the standard residual connection.
    pools from a growing history of all previous outputs via attention,
    then passes the result through the wrapped module (attn or ff).
    the output is appended back to history (in-place list mutation).
    """

    def __init__(self, fn, dim, heads = 8, dim_head = 64, use_learned_query = True):
        super().__init__()
        self.fn = fn
        self.attn_pool = AttentionPool(dim, heads = heads, dim_head = dim_head, use_learned_query = use_learned_query)

    def forward(self, history):
        context = torch.stack(history, dim = 2)
        b, n, l, d = context.shape

        context = rearrange(context, 'b n l d -> (b n) l d')

        last_out = last(history)
        query = rearrange(last_out, 'b n d -> (b n) 1 d')

        pooled = self.attn_pool(context, query = query)

        pooled = rearrange(pooled, '(b n) 1 d -> b n d', b = b, n = n)

        out = self.fn(pooled)

        # mutate history for subsequent layers
        history.append(out)

        return history

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, use_learned_query = True):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                AttentionResidual(Attention(dim, heads = heads, dim_head = dim_head), dim, heads = heads, dim_head = dim_head, use_learned_query = use_learned_query),
                AttentionResidual(FeedForward(dim, mlp_dim), dim, heads = heads, dim_head = dim_head, use_learned_query = use_learned_query)
            ]))

        self.final_attn_pool = AttentionResidual(nn.LayerNorm(dim), dim, heads = heads, dim_head = dim_head, use_learned_query = use_learned_query)

    def forward(self, tokens):
        history = [tokens]

        for attn_res, ff_res in self.layers:
            history = attn_res(history)
            history = ff_res(history)

        history = self.final_attn_pool(history)

        return last(history)

class SimpleViTAttnResidual(Module):
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
        use_learned_query = True
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert divisible_by(image_height, patch_height) and divisible_by(image_width, patch_width), 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, use_learned_query = use_learned_query)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device, dtype = img.device, img.dtype

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype = dtype)

        x = self.transformer(x)

        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

if __name__ == '__main__':
    for use_learned_query in (True, False):
        v = SimpleViTAttnResidual(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            use_learned_query = use_learned_query
        )

        img = torch.randn(2, 3, 256, 256)
        preds = v(img)

        assert preds.shape == (2, 1000)
