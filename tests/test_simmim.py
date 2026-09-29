import torch
from torch import nn

from vit_pytorch.simmim import SimMIM


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, 5, 4))
        self.to_patch_embedding = nn.Sequential(
            nn.Identity(),
            nn.Identity(),
            nn.Linear(2, 4),
        )
        self.transformer = nn.Identity()


def _constant_miss(masking_ratio):
    model = SimMIM(encoder = _Encoder(), masking_ratio = masking_ratio)
    with torch.no_grad():
        model.to_pixels.weight.zero_()
        model.to_pixels.bias.zero_()
    # four patches, two pixels each, every pixel off by 1
    return model(torch.ones(1, 4, 2))


def test_a_constant_pixel_error_does_not_shrink_with_the_mask_count():
    half = _constant_miss(0.5)
    three_quarters = _constant_miss(0.75)
    assert half.item() == 1.0
    assert three_quarters.item() == 1.0
