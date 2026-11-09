import pytest
import torch

from vit_pytorch.rvt import RvT


@pytest.mark.parametrize('dtype', (torch.float16, torch.bfloat16))
def test_rvt_preserves_low_precision_input_dtype(dtype):
    model = RvT(
        image_size = 32,
        patch_size = 8,
        num_classes = 5,
        dim = 64,
        depth = 2,
        heads = 4,
        mlp_dim = 128
    ).to(dtype)

    img = torch.randn(1, 3, 32, 32, dtype = dtype)
    preds = model(img)

    assert preds.shape == (1, 5), 'correct logits outputted'
    assert preds.dtype == dtype, 'the axial rotary position grid must not upcast a low-precision input'
