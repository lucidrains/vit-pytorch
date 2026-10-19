import pytest
import torch
from torch import nn

from vit_pytorch.distill import (
    DistillableViT,
    DistillableT2TViT,
    DistillableEfficientViT,
    DistillWrapper,
)


def make_student(kind, pool='cls'):
    common = dict(image_size=16, num_classes=4, dim=8, pool=pool)
    if kind == 't2t':
        return DistillableT2TViT(
            **common, depth=1, heads=2, mlp_dim=16, t2t_layers=((3, 2),)
        )
    if kind == 'efficient':
        return DistillableEfficientViT(
            **common, patch_size=4, transformer=nn.Identity()
        )
    return DistillableViT(
        **common, patch_size=4, depth=1, heads=2, mlp_dim=16
    )


@pytest.mark.parametrize('kind', ('vit', 't2t', 'efficient'))
@pytest.mark.parametrize('pool', ('cls', 'mean'))
@pytest.mark.parametrize('distilling', (False, True))
@pytest.mark.parametrize('image_size', (8, 16))
def test_distillable_backbone_token_layouts(kind, pool, distilling, image_size):
    torch.manual_seed(0)
    student = make_student(kind, pool).eval()
    images = torch.randn(2, 3, image_size, image_size)
    token = nn.Parameter(torch.randn(1, 8)) if distilling else None
    output = student(images, distill_token=token)

    if distilling:
        logits, hidden = output
        assert hidden.shape == (2, 8)
        loss = logits.sum() + hidden.sum()
    else:
        logits = output
        native = student.to_vit().eval()
        torch.testing.assert_close(logits, native(images))
        loss = logits.sum()

    assert logits.shape == (2, 4)
    loss.backward()
    assert student.cls_token.grad.shape == student.cls_token.shape
    assert torch.isfinite(student.cls_token.grad).all()


@pytest.mark.parametrize('kind', ('vit', 't2t', 'efficient'))
@pytest.mark.parametrize('hard', (False, True))
def test_distillation_wrapper_trains_supported_backbones(kind, hard):
    student = make_student(kind)
    teacher = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 4))
    wrapper = DistillWrapper(teacher=teacher, student=student, hard=hard)
    loss = wrapper(torch.randn(2, 3, 16, 16), torch.tensor([1, 2]))

    assert torch.isfinite(loss)
    loss.backward()
    assert wrapper.distillation_token.grad is not None
    assert torch.isfinite(wrapper.distillation_token.grad).all()
    assert all(parameter.grad is None for parameter in teacher.parameters())
