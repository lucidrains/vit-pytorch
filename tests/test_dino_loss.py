import pytest
import torch
import torch.nn.functional as F

from vit_pytorch.dino import loss_fn


def test_loss_matches_probability_reference_in_float32():
    teacher_logits = torch.tensor(
        [[1.2, -0.4, 0.7], [-0.3, 0.8, 1.1]],
        dtype = torch.float32
    )
    student_logits = torch.tensor(
        [[-0.5, 0.9, 0.2], [0.6, -0.1, 1.4]],
        dtype = torch.float32,
        requires_grad = True
    )
    centers = torch.tensor([[0.1, -0.2, 0.3]], dtype = torch.float32)
    teacher_temp = 0.7
    student_temp = 0.9
    eps = 1e-20

    expected_teacher_probs = ((teacher_logits - centers) / teacher_temp).softmax(dim = -1)
    expected_student_probs = (student_logits / student_temp).softmax(dim = -1)
    expected = -(
        expected_teacher_probs * torch.log(expected_student_probs + eps)
    ).sum(dim = -1).mean()

    actual = loss_fn(
        teacher_logits,
        student_logits,
        teacher_temp = teacher_temp,
        student_temp = student_temp,
        centers = centers,
        eps = eps
    )

    assert torch.allclose(actual, expected, rtol = 1e-5, atol = 1e-6)


@pytest.mark.parametrize('dtype', (torch.float16, torch.bfloat16))
def test_loss_and_gradient_stay_finite_for_low_precision_logits(dtype):
    # A 100-point logit gap makes softmax underflow the unlikely class in
    # both low-precision formats, while the exact cross-entropy remains 100.
    teacher_logits = torch.tensor([[50., -50.]], dtype = dtype)
    student_logits = torch.tensor([[-50., 50.]], dtype = dtype, requires_grad = True)
    centers = torch.zeros((1, 2), dtype = dtype)

    loss = loss_fn(
        teacher_logits,
        student_logits,
        teacher_temp = 1.,
        student_temp = 1.,
        centers = centers
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(student_logits.grad).all()
    assert torch.allclose(loss.float(), torch.tensor(100.), atol = 1.)


def test_loss_gradient_matches_log_softmax_reference():
    teacher_logits = torch.tensor([[0.4, -0.2, 0.8]], dtype = torch.float64)
    student_logits = torch.tensor(
        [[-0.7, 0.3, 1.1]], dtype = torch.float64,
        requires_grad = True
    )
    centers = torch.tensor([[0.1, 0.2, -0.1]], dtype = torch.float64)

    actual = loss_fn(
        teacher_logits,
        student_logits,
        teacher_temp = 0.6,
        student_temp = 0.8,
        centers = centers
    )
    actual.backward()

    teacher_probs = ((teacher_logits - centers) / 0.6).softmax(dim = -1)
    reference_logits = student_logits.detach().clone().requires_grad_()
    reference = -(
        teacher_probs * F.log_softmax(reference_logits / 0.8, dim = -1)
    ).sum(dim = -1).mean()
    reference.backward()

    assert torch.allclose(actual.detach(), reference.detach(), rtol = 1e-12, atol = 1e-12)
    assert torch.allclose(student_logits.grad, reference_logits.grad, rtol = 1e-12, atol = 1e-12)
