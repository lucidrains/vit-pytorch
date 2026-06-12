from functools import partial
from collections import namedtuple

import torch
from torch import nn, cat
from torch.nn import Module, ModuleList
from einops import rearrange, repeat, reduce, einsum, pack, unpack
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

def l1norm(t, dim = -1, eps = 1e-8):
    return t / t.sum(dim = dim, keepdim = True).clamp(min = eps)

def pack_with_inverse(t, pattern):
    packed, packed_shape = pack(t, pattern)

    def inverse(packed_out):
        return unpack(packed_out, packed_shape, pattern)

    return packed, inverse

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
    def __init__(
        self,
        dim,
        num_slots,
        heads,
        dim_head,
        mlp_dim,
        dropout = 0.,
        l1norm_after_tokens_softmax = False,
        token_softmax_over_slots = False
    ):
        super().__init__()
        self.heads = heads
        self.l1norm_after_tokens_softmax = l1norm_after_tokens_softmax
        self.token_softmax_over_slots = token_softmax_over_slots

        self.q_groups = 2 if token_softmax_over_slots else 1

        inner_dim = heads * dim_head

        self.norm_tokens = LayerNormNoBias(dim)
        self.norm_slots = LayerNormNoBias(dim)

        self.to_q_v_tokens = LinearNoBias(dim, inner_dim * (self.q_groups + 1))
        self.to_k_v_slots = LinearNoBias(dim, inner_dim * 2)

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

        self.mlp_mask = FeedForward(self.q_groups * heads * num_slots + dim, mlp_dim, dropout = dropout, out_dim = self.q_groups * heads * num_slots)

    def forward(self, tokens, slots, mask):
        h, g = self.heads, self.q_groups

        # normalize

        tokens_norm = self.norm_tokens(tokens)
        slots_norm = self.norm_slots(slots)

        # queries, keys, values

        tokens_q_v = rearrange(self.to_q_v_tokens(tokens_norm), 'b t (g h d) -> g b h t d', h = h, g = g + 1)
        q, v_tokens = tokens_q_v[:-1], tokens_q_v[-1]
        q = rearrange(q, 'g b h t d -> b g h t d')

        k, v_slots = rearrange(self.to_k_v_slots(slots_norm), 'b s (kv h d) -> kv b h s d', h = h, kv = 2)

        # mutual attention

        sim = einsum(q, k, 'b g h t d, b h s d -> b g h t s') * self.scale

        mask_prime = mask + sim

        if self.token_softmax_over_slots:
            mask_prime_slots, mask_prime_tokens = mask_prime.unbind(dim = 1)
            attn_tokens = mask_prime_tokens.softmax(dim = -1)
        else:
            mask_prime_slots = rearrange(mask_prime, 'b 1 h t s -> b h t s')
            attn_tokens = mask_prime_slots.softmax(dim = -2)

        # softmax over slots, as in slot attention - https://arxiv.org/abs/2006.15055

        attn_slots = mask_prime_slots.softmax(dim = -1)

        if self.l1norm_after_tokens_softmax:
            attn_slots = l1norm(attn_slots, dim = -2)

        # aggregate

        tokens_out = einsum(attn_tokens, v_slots, 'b h t s, b h s d -> b h t d')
        tokens_out = rearrange(tokens_out, 'b h t d -> b t (h d)')
        tokens_out = self.to_out_tokens(tokens_out)

        tokens_prime = tokens + tokens_out

        slots_out = einsum(attn_slots, v_tokens, 'b h t s, b h t d -> b h s d')
        slots_out = rearrange(slots_out, 'b h s d -> b s (h d)')
        slots_out = self.to_out_slots(slots_out)

        slots_prime = slots + slots_out

        # feedforward

        tokens_next = tokens_prime + self.mlp_tokens(tokens_prime)
        slots_next = slots_prime + self.mlp_slots(slots_prime)

        # mask update

        mask_prime_reshaped = rearrange(mask_prime, 'b g h t s -> b t (g h s)')
        concat_mask_tokens = cat((mask_prime_reshaped, tokens_prime), dim = -1)
        mask_next_reshaped = self.mlp_mask(concat_mask_tokens)
        mask_next = rearrange(mask_next_reshaped, 'b t (g h s) -> b g h t s', h = h, g = g)

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
        l1norm_after_tokens_softmax = False,
        token_softmax_over_slots = False,
        num_register_tokens = 0,
        num_register_slots = 0,
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

        self.num_register_tokens = num_register_tokens
        self.num_register_slots = num_register_slots

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))
        self.register_slots = nn.Parameter(torch.randn(num_register_slots, dim))

        self.heads = heads
        self.token_softmax_over_slots = token_softmax_over_slots
        self.q_groups = 2 if token_softmax_over_slots else 1

        total_slots = num_slots + num_register_slots

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(WWTBlock(
                dim,
                total_slots,
                heads,
                dim_head,
                mlp_dim,
                dropout = dropout,
                l1norm_after_tokens_softmax = l1norm_after_tokens_softmax,
                token_softmax_over_slots = token_softmax_over_slots
            ))

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

        # registers

        reg_tokens = repeat(self.register_tokens, 'n d -> b n d', b = b)
        tokens, inverse_pack_tokens = pack_with_inverse([reg_tokens, tokens], 'b * d')

        reg_slots = repeat(self.register_slots, 'n d -> b n d', b = b)
        slots, inverse_pack_slots = pack_with_inverse([reg_slots, slots], 'b * d')

        t_packed = tokens.shape[-2]
        s_packed = slots.shape[-2]

        mask = tokens.new_zeros(b, self.q_groups, self.heads, t_packed, s_packed)

        # layers

        for block in self.layers:
            tokens, slots, mask = block(tokens, slots, mask)

        _, tokens = inverse_pack_tokens(tokens)
        _, slots = inverse_pack_slots(slots)

        mask = mask[..., self.num_register_tokens:, self.num_register_slots:]

        if not self.token_softmax_over_slots:
            mask = rearrange(mask, 'b 1 h t s -> b h t s')

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
    for token_softmax_over_slots in (False, True):
        print(f"Testing with token_softmax_over_slots = {token_softmax_over_slots}")
        model = WWT(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 256,
            depth = 2,
            num_slots = 32,
            heads = 4,
            mlp_dim = 512,
            return_tokens = True,
            l1norm_after_tokens_softmax = True,
            token_softmax_over_slots = token_softmax_over_slots,
            num_register_tokens = 4,
            num_register_slots = 4
        )

        img = torch.randn(1, 3, 256, 256)
        slot_preds, token_preds = model(img)

        assert slot_preds.shape == (1, 1000)
        assert token_preds.shape == (1, 1000)
