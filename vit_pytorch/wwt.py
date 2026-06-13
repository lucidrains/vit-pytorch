from __future__ import annotations

from functools import partial
from collections import namedtuple

import torch
from torch import nn, cat, is_tensor
from torch.nn import Module, ModuleList, ParameterList
from einops import rearrange, repeat, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange

# constants

LinearNoBias = partial(nn.Linear, bias = False)
LayerNormNoBias = partial(nn.LayerNorm, bias = False)

WWTReturn = namedtuple('WWTReturn', ['slot_logits', 'token_logits'])
WWTFeatureReturn = namedtuple('WWTFeatureReturn', ['slots', 'tokens', 'masks'])

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

class AutoencodingHead(Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        decoder: Module | None = None,
        channel_first = False
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.grid_h = image_height // patch_height
        self.grid_w = image_width // patch_width

        self.channel_first = channel_first
        self.decoder = default(decoder, nn.Identity())

    def forward(self, all_x, masks, interactions):
        feature_maps = []
        pattern = 'b (h w) d -> b d h w' if self.channel_first else 'b (h w) d -> b h w d'

        for mask, (i, j) in zip(masks, interactions):
            is_patch_interaction = i == 0

            if not is_patch_interaction:
                continue

            mask_softmax = reduce(mask, 'b ... t s -> b t s', 'mean').softmax(dim = -1)
            dense_features = einsum(mask_softmax, all_x[j], 'b t s, b s d -> b t d')

            spatial = rearrange(dense_features, pattern, h = self.grid_h, w = self.grid_w)
            feature_maps.append(self.decoder(spatial))

        assert len(feature_maps) > 0, 'no interactions found that chain to the main image'

        return feature_maps[0] if len(feature_maps) == 1 else tuple(feature_maps)

def FeedForward(dim, hidden_dim, dropout = 0., out_dim = None):
    return nn.Sequential(
        LayerNormNoBias(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, default(out_dim, dim)),
        nn.Dropout(dropout)
    )

class MutualAttention(Module):
    def __init__(
        self,
        dim,
        num_slots,
        heads,
        dim_head,
        mlp_dim,
        dropout = 0.,
        l1norm_after_tokens_softmax = False,
        token_softmax_over_slots = False,
        project_mask_groups = False
    ):
        super().__init__()
        self.heads = heads
        self.l1norm_after_tokens_softmax = l1norm_after_tokens_softmax
        self.token_softmax_over_slots = token_softmax_over_slots

        self.q_groups = 2 if token_softmax_over_slots else 1

        self.project_mask_groups = project_mask_groups and token_softmax_over_slots
        self.mask_groups = 1 if self.project_mask_groups else self.q_groups

        inner_dim = heads * dim_head

        self.to_q_v_tokens = LinearNoBias(dim, inner_dim * (self.q_groups + 1))
        self.to_k_v_slots = LinearNoBias(dim, inner_dim * 2)

        self.scale = dim_head ** -0.5

        self.to_out_tokens = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.to_out_slots = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        if self.project_mask_groups:
            self.mask_project = nn.Conv2d(self.q_groups * heads, heads, 1)

        self.mlp_mask = FeedForward(
            self.mask_groups * heads * num_slots + dim,
            mlp_dim,
            dropout = dropout,
            out_dim = self.mask_groups * heads * num_slots
        )

    def forward(self, tokens, slots, mask):
        h, g = self.heads, self.q_groups

        # queries, keys, values

        tokens_q_v = rearrange(self.to_q_v_tokens(tokens), 'b t (g h d) -> g b h t d', h = h, g = g + 1)
        q, v_tokens = tokens_q_v[:-1], tokens_q_v[-1]
        q = rearrange(q, 'g b h t d -> b g h t d')

        k, v_slots = rearrange(self.to_k_v_slots(slots), 'b s (kv h d) -> kv b h s d', h = h, kv = 2)

        # mutual attention

        sim = einsum(q, k, 'b g h t d, b h s d -> b g h t s') * self.scale
        mask_prime = mask + sim

        if self.token_softmax_over_slots:
            mask_prime_slots, mask_prime_tokens = mask_prime.unbind(dim = 1)
            attn_tokens = mask_prime_tokens.softmax(dim = -1)
        else:
            mask_prime_slots = rearrange(mask_prime, 'b 1 h t s -> b h t s')
            attn_tokens = mask_prime_slots.softmax(dim = -2)

        # slot softmax - https://arxiv.org/abs/2006.15055

        attn_slots = mask_prime_slots.softmax(dim = -1)

        if self.l1norm_after_tokens_softmax:
            attn_slots = l1norm(attn_slots, dim = -2)

        # aggregate

        tokens_out = self.to_out_tokens(rearrange(einsum(attn_tokens, v_slots, 'b h t s, b h s d -> b h t d'), 'b h t d -> b t (h d)'))
        slots_out = self.to_out_slots(rearrange(einsum(attn_slots, v_tokens, 'b h t s, b h t d -> b h s d'), 'b h s d -> b s (h d)'))

        # mask update

        if self.project_mask_groups:
            mask_prime = self.mask_project(rearrange(mask_prime, 'b g h t s -> b (g h) t s'))
            mask_prime = rearrange(mask_prime, 'b h t s -> b 1 h t s')

        mask_next_reshaped = self.mlp_mask(cat((rearrange(mask_prime, 'b g h t s -> b t (g h s)'), tokens + tokens_out), dim = -1))
        mask_next = rearrange(mask_next_reshaped, 'b t (g h s) -> b g h t s', h = h, g = self.mask_groups)

        return tokens_out, slots_out, mask_next

class WWTBlock(Module):
    def __init__(
        self,
        dim,
        num_hierarchies,
        seq_lengths,
        interactions,
        heads,
        dim_head,
        mlp_dim,
        dropout = 0.,
        l1norm_after_tokens_softmax = False,
        token_softmax_over_slots = False,
        project_mask_groups = False
    ):
        super().__init__()
        self.interactions = interactions

        self.attns = ModuleList([MutualAttention(
            dim = dim,
            num_slots = seq_lengths[j],
            heads = heads,
            dim_head = dim_head,
            mlp_dim = mlp_dim,
            dropout = dropout,
            l1norm_after_tokens_softmax = l1norm_after_tokens_softmax,
            token_softmax_over_slots = token_softmax_over_slots,
            project_mask_groups = project_mask_groups
        ) for _, j in interactions])

        self.norms = ModuleList([LayerNormNoBias(dim) for _ in range(num_hierarchies)])
        self.mlps = ModuleList([FeedForward(dim, mlp_dim, dropout = dropout) for _ in range(num_hierarchies)])

    def forward(self, x, masks):
        norm_x = [norm(seq) for norm, seq in zip(self.norms, x)]

        delta_x = [0.] * len(x)
        next_masks = []

        for mask, (i, j), attn in zip(masks, self.interactions, self.attns):
            tokens_out, slots_out, next_mask = attn(norm_x[i], norm_x[j], mask)

            delta_x[i] = delta_x[i] + tokens_out
            delta_x[j] = delta_x[j] + slots_out
            next_masks.append(next_mask)

        return [seq + delta + mlp(seq + delta) for seq, delta, mlp in zip(x, delta_x, self.mlps)], next_masks

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
        num_slots: int | tuple[int, ...],
        interactions: tuple[tuple[int, int], ...] | None = None,
        heads = 8,
        dim_head = 64,
        mlp_dim = None,
        channels = 3,
        dropout = 0.,
        return_tokens = False,
        l1norm_after_tokens_softmax = False,
        token_softmax_over_slots = False,
        project_mask_groups = False,
        num_register_tokens = 0,
        num_register_slots: int | tuple[int, ...] = 0,
        task_heads: tuple[Module, ...] | list[Module] = (),
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

        num_slots = (num_slots,) if isinstance(num_slots, int) else tuple(num_slots)

        # ensure slots are in decreasing order to establish a part-whole hierarchy

        for s1, s2 in zip(num_slots[:-1], num_slots[1:]):
            assert s1 > s2, 'to establish a part-whole hierarchy, the number of slots must be strictly decreasing across levels'
        num_hierarchies = 1 + len(num_slots)

        self.interactions = default(interactions, tuple((0, i + 1) for i in range(len(num_slots))))
        self.interactions = tuple(tuple(interaction) for interaction in self.interactions)
        assert len(set(self.interactions)) == len(self.interactions), 'interactions must be unique'

        for i, j in self.interactions:
            assert i < j, 'each interaction must be in strictly ascending order (from lower index to higher index)'
        self.slots = ParameterList([nn.Parameter(torch.randn(n, dim)) for n in num_slots])

        num_register_slots = (num_register_slots,) * len(num_slots) if isinstance(num_register_slots, int) else tuple(num_register_slots)
        assert len(num_register_slots) == len(num_slots)

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))
        self.register_slots = ParameterList([nn.Parameter(torch.randn(n, dim)) for n in num_register_slots])
        self.num_regs = (num_register_tokens, *num_register_slots)

        self.heads = heads
        self.token_softmax_over_slots = token_softmax_over_slots
        self.q_groups = 2 if token_softmax_over_slots else 1

        self.project_mask_groups = project_mask_groups and token_softmax_over_slots
        self.mask_groups = 1 if self.project_mask_groups else self.q_groups

        self.seq_lengths = (num_patches + num_register_tokens, *(s + r for s, r in zip(num_slots, num_register_slots)))

        self.layers = ModuleList([WWTBlock(
            dim = dim,
            num_hierarchies = num_hierarchies,
            seq_lengths = self.seq_lengths,
            interactions = self.interactions,
            heads = heads,
            dim_head = dim_head,
            mlp_dim = mlp_dim,
            dropout = dropout,
            l1norm_after_tokens_softmax = l1norm_after_tokens_softmax,
            token_softmax_over_slots = token_softmax_over_slots,
            project_mask_groups = project_mask_groups
        ) for _ in range(depth)])

        self.mlp_head = nn.Sequential(LayerNormNoBias(dim), nn.Linear(dim, num_classes))
        self.task_heads = ModuleList(task_heads)
        self.has_task_heads = len(self.task_heads) > 0
        self.return_tokens = return_tokens

        if self.return_tokens:
            self.mlp_head_tokens = nn.Sequential(LayerNormNoBias(dim), nn.Linear(dim, num_classes))

    def forward(self, img, return_embeddings = False):
        b = img.shape[0]

        # get tokens

        tokens = self.to_patch_embedding(img) + self.pos_embedding
        x = [
            tokens,
            *(repeat(p, 's d -> b s d', b = b) for p in self.slots)
        ]

        # pack registers

        all_regs = [
            repeat(self.register_tokens, 'n d -> b n d', b = b),
            *(repeat(p, 'n d -> b n d', b = b) for p in self.register_slots)
        ]

        packs = (pack_with_inverse([reg, seq], 'b * d') for reg, seq in zip(all_regs, x))
        x, inverse_packs = zip(*packs)

        # initial masks

        masks = [tokens.new_zeros(b, self.mask_groups, self.heads, self.seq_lengths[i], self.seq_lengths[j]) for (i, j) in self.interactions]

        # layers

        for block in self.layers:
            x, masks = block(x, masks)

        # unpack registers

        unpacked_seqs = (inv(seq) for seq, inv in zip(x, inverse_packs))
        tokens_out, *slots_out = (seq for _, seq in unpacked_seqs)
        slots_out = tuple(slots_out)
        all_x = (tokens_out, *slots_out)

        # process masks

        processed_masks = []
        for mask, (i, j) in zip(masks, self.interactions):
            mask = mask[..., self.num_regs[i]:, self.num_regs[j]:]
            if not self.token_softmax_over_slots or self.project_mask_groups:
                mask = rearrange(mask, 'b 1 h t s -> b h t s')
            processed_masks.append(mask)

        if return_embeddings:
            return WWTFeatureReturn(slots_out, tokens_out if self.return_tokens else None, processed_masks)

        # classification

        pooled_slot_logits = sum(reduce(self.mlp_head(s), 'b s c -> b c', 'mean') for s in slots_out) / len(slots_out)

        if not self.return_tokens:
            out = pooled_slot_logits
        else:
            pooled_token_logits = reduce(self.mlp_head_tokens(tokens_out), 'b t c -> b c', 'mean')
            out = WWTReturn(pooled_slot_logits, pooled_token_logits)

        # task heads

        if not self.has_task_heads:
            return out

        return (out, *(head(all_x, processed_masks, self.interactions) for head in self.task_heads))

if __name__ == '__main__':
    configs = (
        # token_softmax_over_slots, project_mask_groups, channel_first
        (False, False, False),
        (True, False, False),
        (True, True, True),
    )

    for token_softmax_over_slots, project_mask_groups, channel_first in configs:
        print(f"Testing with token_softmax_over_slots = {token_softmax_over_slots}, project_mask_groups = {project_mask_groups}, channel_first = {channel_first}")

        autoencoding_head = AutoencodingHead(
            image_size = 256,
            patch_size = 32,
            channel_first = channel_first
        )

        model = WWT(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 256,
            depth = 2,
            num_slots = (32, 16),
            interactions = ((0, 1), (0, 2), (1, 2)),
            heads = 4,
            mlp_dim = 512,
            return_tokens = True,
            l1norm_after_tokens_softmax = True,
            token_softmax_over_slots = token_softmax_over_slots,
            project_mask_groups = project_mask_groups,
            num_register_tokens = 4,
            num_register_slots = (4, 2),
            task_heads = [autoencoding_head]
        )

        img = torch.randn(1, 3, 256, 256)

        out, dense_feature_maps = model(img)
        slot_preds, token_preds = out

        assert slot_preds.shape == (1, 1000)
        assert token_preds.shape == (1, 1000)

        assert isinstance(dense_feature_maps, tuple) and len(dense_feature_maps) == 2
        dense_high, dense_low = dense_feature_maps

        if channel_first:
            assert dense_high.shape == (1, 256, 8, 8)
            assert dense_low.shape == (1, 256, 8, 8)
        else:
            assert dense_high.shape == (1, 8, 8, 256)
            assert dense_low.shape == (1, 8, 8, 256)

        print('success')
