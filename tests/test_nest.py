import pytest
import torch

from vit_pytorch.nest import NesT


def test_nest():
    net = NesT(
        image_size = 224,
        patch_size = 4,
        dim = 96,
        heads = 3,
        num_hierarchies = 3,
        block_repeats = (2, 2, 8),
        num_classes = 1000
    )

    img = torch.randn(1, 3, 224, 224)
    assert net(img).shape == (1, 1000), 'correct logits outputted'


def test_nest_rejects_indivisible_feature_map():
    # image_size // patch_size (= 14) is not divisible by 2 ** (num_hierarchies - 1)
    # (= 4), so the feature map cannot be partitioned across hierarchies. This used
    # to raise an opaque einops error from deep inside forward(); it should fail
    # fast at construction instead.
    with pytest.raises(AssertionError):
        NesT(
            image_size = 224,
            patch_size = 16,
            dim = 96,
            heads = 3,
            num_hierarchies = 3,
            block_repeats = (2, 2, 8),
            num_classes = 1000
        )
