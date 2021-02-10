import math
from functools import reduce

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from vit_pytorch import MPPLoss

# helpers

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

# main class

class MPP(nn.Module):
    def __init__(
        self,
        transformer,
        patch_size,
        dim,
        channels = 3,
        mask_prob = 0.15,
        replace_prob = 0.5,
        random_patch_prob = 0.5):
        super().__init__()

        self.transformer = transformer
        self.loss = MPPLoss(patch_size)
        
        # vit related dimensions
        self.patch_size = patch_size

        # mpp related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_patch_prob = random_patch_prob

        # token ids
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim * channels))

    def forward(self, input, **kwargs):
        # clone original image for loss
        img = input.clone().detach()

        # reshape raw image to patches
        p = self.patch_size
        input = rearrange(input, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)

        mask = get_mask_subset_with_prob(input, self.mask_prob)

        # mask input with mask patches with probability of `replace_prob` (keep patches the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

         # if random token probability > 0 for mpp
        if self.random_patch_prob > 0:
            random_patch_sampling_prob = self.random_patch_prob / (1 - self.replace_prob)
            random_patch_prob = prob_mask_like(input, random_patch_sampling_prob)
            bool_random_patch_prob = mask * random_patch_prob == True
            random_patches = torch.randint(0, input.shape[1], (input.shape[0], input.shape[1]), device=input.device)
            randomized_input = masked_input[torch.arange(masked_input.shape[0]).unsqueeze(-1), random_patches]
            masked_input[bool_random_patch_prob] = randomized_input[bool_random_patch_prob]

        # [mask] input
        replace_prob = prob_mask_like(input, self.replace_prob)
        bool_mask_replace = (mask * replace_prob) == True
        masked_input[bool_mask_replace] = self.mask_token

        # get labels for input patches that were masked
        bool_mask = mask == True
        labels = input[bool_mask]

        # get generator output and get mpp loss
        cls_logits = self.transformer(masked_input, mpp=True, **kwargs)
        logits = cls_logits[:,1:,:]

        mpp_loss = self.loss(logits, img, mask)

        return mpp_loss

        