# /// script
# dependencies = [
#   "accelerate",
#   "vit-pytorch",
#   "wandb"
# ]
# ///

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.datasets import CIFAR100

# constants

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 10
DECORR_LOSS_WEIGHT = 1e-1

TRACK_EXPERIMENT_ONLINE = False

# helpers

def exists(v):
    return v is not None

# data

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CIFAR100(
    root = 'data',
    download = True,
    train = True,
    transform = transform
)

dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

# model

from vit_pytorch.vit_with_decorr import ViT

vit = ViT(
    dim = 128,
    num_classes = 100,
    image_size = 32,
    patch_size = 4,
    depth = 6,
    heads = 8,
    dim_head = 64,
    mlp_dim = 128 * 4,
    decorr_sample_frac = 1. # use all tokens
)

# optim

from torch.optim import Adam

optim = Adam(vit.parameters(), lr = LEARNING_RATE)

# prepare

from accelerate import Accelerator

accelerator = Accelerator()

vit, optim, dataloader = accelerator.prepare(vit, optim, dataloader)

# experiment

import wandb

wandb.init(
    project = 'vit-decorr',
    mode = 'disabled' if not TRACK_EXPERIMENT_ONLINE else 'online'
)

wandb.run.name = 'baseline'

# loop

for _ in range(EPOCHS):
    for images, labels in dataloader:

        logits, decorr_aux_loss = vit(images)
        loss = F.cross_entropy(logits, labels)


        total_loss = (
            loss +
            decorr_aux_loss * DECORR_LOSS_WEIGHT
        )

        wandb.log(dict(loss = loss, decorr_loss = decorr_aux_loss))

        accelerator.print(f'loss: {loss.item():.3f} | decorr aux loss: {decorr_aux_loss.item():.3f}')

        accelerator.backward(total_loss)
        optim.step()
        optim.zero_grad()
