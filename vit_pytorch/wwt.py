from functools import partial
from collections import namedtuple

import torch
from torch import nn, cat
from torch.nn import Module, ModuleList
from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange

# constants

LinearNoBias = partial(nn.Linear, bias = False)
LayerNormNoBias = partial(nn.LayerNorm, bias = False)

WWTBlockReturn = namedtuple('WWTBlockReturn', ['tokens', 'slots', 'mask'])
WWTReturn = namedtuple('WWTReturn', ['slot_logits', 'token_logits'])
WWTFeatureReturn = namedtuple('WWTFeatureReturn', ['slots', 'tokens', 'mask'])

# functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(num, den):
    return (num % den) == 0

# classes

def FeedForward(dim, hidden_dim, dropout = 0., out_dim = None):
    out_dim = default(out_dim, dim)
    return nn.Sequential(
        LayerNormNoBias(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
        nn.Dropout(dropout)
    )

class WWTBlock(Module):
    def __init__(self, dim, num_slots, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.heads = heads
        inner_dim = heads * dim_head

        self.norm_tokens = LayerNormNoBias(dim)
        self.norm_slots = LayerNormNoBias(dim)

        self.to_q_v_tokens = LinearNoBias(dim, inner_dim * 2)
        self.to_k_v_slots = LinearNoBias(dim, inner_dim * 2)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.scale = dim_head ** -0.5

        self.to_out_tokens = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.to_out_slots = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.mlp_tokens = FeedForward(dim, mlp_dim, dropout = dropout)
        self.mlp_slots = FeedForward(dim, mlp_dim, dropout = dropout)

        self.mlp_mask = FeedForward(heads * num_slots + dim, mlp_dim, dropout = dropout, out_dim = heads * num_slots)

    def forward(self, tokens, slots, mask):
        h = self.heads

        # normalize

        tokens_norm = self.norm_tokens(tokens)
        slots_norm = self.norm_slots(slots)

        # queries, keys, values

        q, v_tokens = self.to_q_v_tokens(tokens_norm).chunk(2, dim = -1)
        k, v_slots = self.to_k_v_slots(slots_norm).chunk(2, dim = -1)

        # split heads

        q, k, v_slots, v_tokens = map(self.split_heads, (q, k, v_slots, v_tokens))

        # mutual attention

        dots = einsum(q, k, 'b h t d, b h s d -> b h t s') * self.scale

        mask_prime = mask + dots

        attn_tokens = mask_prime.softmax(dim = -2)

        # softmax over slots, as in slot attention - https://arxiv.org/abs/2006.15055

        attn_slots = mask_prime.softmax(dim = -1)

        # aggregate

        tokens_out = einsum(attn_tokens, v_slots, 'b h t s, b h s d -> b h t d')
        tokens_out = self.merge_heads(tokens_out)
        tokens_out = self.to_out_tokens(tokens_out)

        tokens_prime = tokens + tokens_out

        slots_out = einsum(attn_slots, v_tokens, 'b h t s, b h t d -> b h s d')
        slots_out = self.merge_heads(slots_out)
        slots_out = self.to_out_slots(slots_out)

        slots_prime = slots + slots_out

        # feedforward

        tokens_next = tokens_prime + self.mlp_tokens(tokens_prime)
        slots_next = slots_prime + self.mlp_slots(slots_prime)

        # mask update

        mask_prime_reshaped = rearrange(mask_prime, 'b h t s -> b t (h s)')
        concat_mask_tokens = cat((mask_prime_reshaped, tokens_prime), dim = -1)
        mask_next_reshaped = self.mlp_mask(concat_mask_tokens)
        mask_next = rearrange(mask_next_reshaped, 'b t (h s) -> b h t s', h = h)

        return WWTBlockReturn(tokens_next, slots_next, mask_next)

# Yoshihashi et al. - https://arxiv.org/abs/2605.12021

class WWT(Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        num_slots,
        heads = 8,
        dim_head = 64,
        mlp_dim = None,
        channels = 3,
        dropout = 0.,
        return_tokens = False,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert divisible_by(image_height, patch_height) and divisible_by(image_width, patch_width), 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        mlp_dim = default(mlp_dim, dim * 4)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            LayerNormNoBias(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNormNoBias(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(num_patches, dim))
        self.slots = nn.Parameter(torch.randn(num_slots, dim))

        self.heads = heads

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(WWTBlock(dim, num_slots, heads, dim_head, mlp_dim, dropout = dropout))

        self.mlp_head = nn.Sequential(
            LayerNormNoBias(dim),
            nn.Linear(dim, num_classes)
        )

        self.return_tokens = return_tokens

        if not self.return_tokens:
            return

        self.mlp_head_tokens = nn.Sequential(
            LayerNormNoBias(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, return_embeddings = False):

        # get tokens

        tokens = self.to_patch_embedding(img)
        b, t, _ = tokens.shape

        # pos embedding

        tokens = tokens + self.pos_embedding

        # get slots and mask

        slots = repeat(self.slots, 's d -> b s d', b = b)
        _, s, _ = slots.shape

        mask = tokens.new_zeros(b, self.heads, t, s)

        # layers

        for block in self.layers:
            tokens, slots, mask = block(tokens, slots, mask)

        if return_embeddings:
            tokens_out = tokens if self.return_tokens else None
            return WWTFeatureReturn(slots, tokens_out, mask)

        # classification

        slot_logits = self.mlp_head(slots)
        pooled_slot_logits = reduce(slot_logits, 'b s c -> b c', 'mean')

        if not self.return_tokens:
            return pooled_slot_logits

        token_logits = self.mlp_head_tokens(tokens)
        pooled_token_logits = reduce(token_logits, 'b t c -> b c', 'mean')

        return WWTReturn(pooled_slot_logits, pooled_token_logits)

if __name__ == '__main__':
    model = WWT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        num_slots = 64,
        heads = 16,
        mlp_dim = 2048,
        return_tokens = True
    )

    img = torch.randn(1, 3, 256, 256)
    slot_preds, token_preds = model(img)

    assert slot_preds.shape == (1, 1000)
    assert token_preds.shape == (1, 1000)

    print("WWT forward pass successful!")
