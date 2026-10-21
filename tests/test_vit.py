import torch
from vit_pytorch import ViT

def test_vit():
    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img)
    assert preds.shape == (1, 1000), 'correct logits outputted'

def test_masked_pretraining_wrappers():
    from vit_pytorch import MAE
    from vit_pytorch.simmim import SimMIM
    from vit_pytorch.mpp import MPP

    v = ViT(
        image_size = 64,
        patch_size = 16,
        num_classes = 10,
        dim = 64,
        depth = 1,
        heads = 2,
        mlp_dim = 128
    )

    img = torch.randn(2, 3, 64, 64)

    mae = MAE(encoder = v, decoder_dim = 32)
    mae(img).backward()

    simmim = SimMIM(encoder = v)
    simmim(img).backward()

    mpp = MPP(transformer = v, patch_size = 16, dim = 64)
    mpp(img).backward()
