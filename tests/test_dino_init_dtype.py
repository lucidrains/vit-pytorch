import pytest
import torch
from torch import nn

from vit_pytorch import ViT
from vit_pytorch.dino import Dino


@pytest.mark.parametrize('backbone_dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('default_dtype', (torch.float32, torch.float64))
def test_dino_initialization_image_follows_the_backbone(
    backbone_dtype, default_dtype
):
    previous_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        net = ViT(
            image_size=16, patch_size=4, num_classes=4,
            dim=8, depth=1, heads=2, mlp_dim=16,
        ).to(backbone_dtype)
        learner = Dino(
            net,
            image_size=16,
            hidden_layer='to_latent',
            projection_hidden_size=16,
            projection_layers=2,
            num_classes_K=8,
            augment_fn=nn.Identity(),
            augment_fn2=nn.Identity(),
        )
        images = torch.rand(2, 3, 16, 16, dtype=backbone_dtype)
        loss = learner(images)
        assert torch.isfinite(loss)
        loss.backward()
        assert {parameter.dtype for parameter in net.parameters()} == {backbone_dtype}
        assert learner.teacher_centers.dtype == default_dtype
        assert learner.last_teacher_centers.dtype == default_dtype
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in net.parameters()
            if parameter.grad is not None
        )
        learner.update_moving_average()
        assert torch.isfinite(learner.teacher_centers).all()
    finally:
        torch.set_default_dtype(previous_default)
