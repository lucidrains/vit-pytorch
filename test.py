import torch
from vit_pytorch import MPP
# from vit_pytorch import ViT, MaskedPredictionLoss

# v = ViT(
#     image_size = 256,
#     patch_size = 32,
#     num_classes = 1000,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )

# l = MaskedPredictionLoss(patch_size=32, img_size=256)

# img = torch.randn(1, 3, 256, 256)
# mask = [1,2,3,4] # optional mask, designating which patch to attend to

# preds = v(img) # (1, 1000)
# loss = l(preds, img, mask)

# print(preds.shape)

transformer = 5


trainer = MPP(
    transformer,
    5,
    75,
    mask_prob = 0.15,           # masking probability for masked language modeling
    random_patch_prob=0.05,
    replace_prob = 0.90,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
)

data = torch.rand((2, 3, 10, 10))

loss = trainer(data)

