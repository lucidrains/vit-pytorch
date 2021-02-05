import torch
from vit_pytorch import ViT, MaskedPredictionLoss

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

l = MaskedPredictionLoss(patch_size=32, img_size=256)

img = torch.randn(1, 3, 256, 256)
mask = [1,2,3,4] # optional mask, designating which patch to attend to

preds = v(img, prediction_mask=mask) # (1, 1000)
loss = l(preds, img, mask)

print(preds.shape)
