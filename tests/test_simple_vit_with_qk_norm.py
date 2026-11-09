import torch
from vit_pytorch.simple_vit_with_qk_norm import SimpleViT

def test_simple_vit_with_qk_norm():
    v = SimpleViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img)
    assert preds.shape == (1, 1000), 'correct logits outputted'
