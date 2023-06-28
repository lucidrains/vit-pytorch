import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce

# helpers

def exists(val):
    return val is not None

def prob_mask_like(t, prob):
    batch, seq_length, _ = t.shape
    return torch.zeros((batch, seq_length)).float().uniform_(0, 1) < prob

def get_mask_subset_with_prob(patched_input, prob):
    batch, seq_len, _, device = *patched_input.shape, patched_input.device
    max_masked = math.ceil(prob * seq_len)

    rand = torch.rand((batch, seq_len), device=device)
    _, sampled_indices = rand.topk(max_masked, dim=-1)

    new_mask = torch.zeros((batch, seq_len), device=device)
    new_mask.scatter_(1, sampled_indices, 1)
    return new_mask.bool()


# mpp loss


class MPPLoss(nn.Module):
    def __init__(
        self,
        patch_size,
        channels,
        output_channel_bits,
        max_pixel_val,
        mean,
        std
    ):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.output_channel_bits = output_channel_bits
        self.max_pixel_val = max_pixel_val

        self.mean = torch.tensor(mean).view(-1, 1, 1) if mean else None
        self.std = torch.tensor(std).view(-1, 1, 1) if std else None

    def forward(self, predicted_patches, target, mask):
        p, c, mpv, bits, device = self.patch_size, self.channels, self.max_pixel_val, self.output_channel_bits, target.device
        bin_size = mpv / (2 ** bits)

        # un-normalize input
        if exists(self.mean) and exists(self.std):
            target = target * self.std + self.mean

        # reshape target to patches
        target = target.clamp(max = mpv) # clamp just in case
        avg_target = reduce(target, 'b c (h p1) (w p2) -> b (h w) c', 'mean', p1 = p, p2 = p).contiguous()

        channel_bins = torch.arange(bin_size, mpv, bin_size, device = device)
        discretized_target = torch.bucketize(avg_target, channel_bins)

        bin_mask = (2 ** bits) ** torch.arange(0, c, device = device).long()
        bin_mask = rearrange(bin_mask, 'c -> () () c')

        target_label = torch.sum(bin_mask * discretized_target, dim = -1)

        loss = F.cross_entropy(predicted_patches[mask], target_label[mask])
        return loss


# main class


class MPP(nn.Module):
    def __init__(
        self,
        transformer,
        patch_size,
        dim,
        output_channel_bits=3,
        channels=3,
        max_pixel_val=1.0,
        mask_prob=0.15,
        replace_prob=0.5,
        random_patch_prob=0.5,
        mean=None,
        std=None
    ):
        super().__init__()
        self.transformer = transformer
        self.loss = MPPLoss(patch_size, channels, output_channel_bits,
                            max_pixel_val, mean, std)

        # extract patching function
        self.patch_to_emb = nn.Sequential(transformer.to_patch_embedding[1:])

        # output transformation
        self.to_bits = nn.Linear(dim, 2**(output_channel_bits * channels))

        # vit related dimensions
        self.patch_size = patch_size

        # mpp related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_patch_prob = random_patch_prob

        # token ids
        self.mask_token = nn.Parameter(torch.randn(1, 1, channels * patch_size ** 2))

    def forward(self, input, **kwargs):
        transformer = self.transformer
        # clone original image for loss
        img = input.clone().detach()

        # reshape raw image to patches
        p = self.patch_size
        input = rearrange(input,
                          'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                          p1=p,
                          p2=p)

        mask = get_mask_subset_with_prob(input, self.mask_prob)

        # mask input with mask patches with probability of `replace_prob` (keep patches the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

        # if random token probability > 0 for mpp
        if self.random_patch_prob > 0:
            random_patch_sampling_prob = self.random_patch_prob / (
                1 - self.replace_prob)
            random_patch_prob = prob_mask_like(input,
                                               random_patch_sampling_prob).to(mask.device)

            bool_random_patch_prob = mask * (random_patch_prob == True)
            random_patches = torch.randint(0,
                                           input.shape[1],
                                           (input.shape[0], input.shape[1]),
                                           device=input.device)
            randomized_input = masked_input[
                torch.arange(masked_input.shape[0]).unsqueeze(-1),
                random_patches]
            masked_input[bool_random_patch_prob] = randomized_input[
                bool_random_patch_prob]

        # [mask] input
        replace_prob = prob_mask_like(input, self.replace_prob).to(mask.device)
        bool_mask_replace = (mask * replace_prob) == True
        masked_input[bool_mask_replace] = self.mask_token

        # linear embedding of patches
        masked_input = self.patch_to_emb(masked_input)

        # add cls token to input sequence
        b, n, _ = masked_input.shape
        cls_tokens = repeat(transformer.cls_token, '() n d -> b n d', b=b)
        masked_input = torch.cat((cls_tokens, masked_input), dim=1)

        # add positional embeddings to input
        masked_input += transformer.pos_embedding[:, :(n + 1)]
        masked_input = transformer.dropout(masked_input)

        # get generator output and get mpp loss
        masked_input = transformer.transformer(masked_input, **kwargs)
        cls_logits = self.to_bits(masked_input)
        logits = cls_logits[:, 1:, :]

        mpp_loss = self.loss(logits, img, mask)

        return mpp_loss
