import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedPredictionLoss(nn.Module):
    def __init__(self,
                 patch_size,
                 img_size,
                 device="cpu"):
        super(MaskedPredictionLoss, self).__init__()
        self.patch_size = patch_size
        self.num_patch_axis = img_size // patch_size

    def _transform_targets(self, targets, masked_patches):
        masked_patches_dims = []
        for masked_patch in masked_patches:
            height_offset = self.patch_size * ((masked_patch) // self.num_patch_axis)
            width_offset = self.patch_size * ((masked_patch) % self.num_patch_axis)
            masked_patches_dims.append([height_offset, width_offset])

        target_patches = []
        for target in range(targets.shape[0]):
            for masked_patch in masked_patches_dims:
                height_offset, width_offset = masked_patch
                extracted_patch = targets[target, :,
                                        height_offset:height_offset + self.patch_size,
                                        width_offset:width_offset + self.patch_size]
                target_patches.append(extracted_patch)

        target_patches_tensor = torch.stack(target_patches)
        target_patches_tensor = (target_patches_tensor // 0.34).long()
        encoded_targets = F.one_hot(target_patches_tensor, 3)
        n, c, w, h, e = encoded_targets.shape
        encoded_targets = torch.reshape(encoded_targets, [n, w, h, c * e])
        mean_targets = torch.mean(encoded_targets.float(),
                                dim=[1, 2],
                                keepdim=True).view(n, c * e)
        return mean_targets

    def _transform_outputs(self, outputs, masked_patches):
        output_dim = outputs.shape[-1]

        masked_patches_shifted = [
            masked_patch + 1 for masked_patch in masked_patches
        ]
        outputs = outputs[:, masked_patches_shifted, :]
        outputs = outputs.view(-1, output_dim)
        return outputs

    def forward(self, x, y, masked_patches):
        transformed_outputs = self._transform_outputs(x, masked_patches)
        transformed_targets = self._transform_targets(y, masked_patches)
        loss = F.mse_loss(transformed_outputs, transformed_targets)
        return loss