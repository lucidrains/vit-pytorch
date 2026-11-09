import pytest
import torch
from torch import nn

from vit_pytorch.accept_video_wrapper import AcceptVideoWrapper


class ImageNet(nn.Module):
    def forward(self, images):
        logits = images.mean(dim = (1, 2, 3)).unsqueeze(-1)
        return dict(logits = logits, aux_loss = images.mean())


@pytest.mark.parametrize('batch, frames', [(1, 1), (1, 2), (2, 1)])
def test_single_logit_video_output(batch, frames):
    wrapper = AcceptVideoWrapper(ImageNet())
    video = torch.ones(batch, 3, frames, 2, 2)

    outputs = wrapper(video)

    assert outputs['logits'].shape == (batch, frames, 1)
    torch.testing.assert_close(outputs['logits'], torch.ones(batch, frames, 1))
    assert outputs['aux_loss'].shape == ()
    torch.testing.assert_close(outputs['aux_loss'], torch.tensor(1.))
