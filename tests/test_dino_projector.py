import pytest
import torch
from torch import nn

from vit_pytorch import ViT
from vit_pytorch.dino import Dino, MLP


@pytest.mark.parametrize('dim,hidden_size', [(8, 16), (16, 16), (16, 8)])
@pytest.mark.parametrize('layers', (1, 2, 4))
def test_dino_projector_width(dim, hidden_size, layers):
    projector = MLP(dim, 6, num_layers=layers, hidden_size=hidden_size)
    inputs = torch.randn(2, dim, requires_grad=True)
    output = projector(inputs)

    assert output.shape == (2, 6)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_dino_trains_with_a_single_projection_layer():
    net = ViT(
        image_size=16, patch_size=4, num_classes=4,
        dim=8, depth=1, heads=2, mlp_dim=16,
    )
    learner = Dino(
        net,
        image_size=16,
        hidden_layer='to_latent',
        projection_hidden_size=16,
        projection_layers=1,
        num_classes_K=8,
        augment_fn=nn.Identity(),
        augment_fn2=nn.Identity(),
    )
    loss = learner(torch.rand(2, 3, 16, 16))

    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in net.parameters())
