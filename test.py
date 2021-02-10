import torch
from vit_pytorch import MPP, ViT
# from vit_pytorch import ViT, MaskedPredictionLoss

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 3,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# l = MaskedPredictionLoss(patch_size=32, img_size=256)

img = torch.randn(2, 3, 256, 256)
# mask = [1,2,3,4] # optional mask, designating which patch to attend to

# preds = v(img) # (1, 1000)
# loss = l(preds, img, mask)

# print(preds.shape)


trainer = MPP(
    transformer = v,
    patch_size = 32,
    dim = 1024,
    mask_prob = 0.15,           # masking probability for masked language modeling
    random_patch_prob=0.30,
    replace_prob = 0.50,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
)

# data = torch.rand((2, 3, 10, 10))

loss = trainer(img)

print(loss)

