import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class MPPLoss(nn.Module):
    def __init__(self, patch_size):
        super(MPPLoss, self).__init__()
        self.patch_size = patch_size

    def forward(self, predicted_patches, target, mask):
        # reshape target to patches
        p = self.patch_size
        target = rearrange(target, "b c (h p1) (w p2) -> b (h w) c (p1 p2) ", p1 = p, p2 = p)

        channel_bins = torch.tensor([0.333, 0.666, 1.0])
        target = torch.bucketize(target, channel_bins, right=True)
        target = target.float().mean(dim=3)

        predicted_patches = predicted_patches[mask]
        target = target[mask]

        loss = F.mse_loss(predicted_patches, target)
        return loss