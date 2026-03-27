import random
from functools import wraps

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from torchvision import transforms as T
from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def l2norm(t, eps = 1e-6):
    return F.normalize(t, dim = -1, eps = eps)

# loss function

def sigreg_loss(
    x,
    num_slices = 1024,
    domain = (-5, 5),
    num_knots = 17
):
    # Randall Balestriero - https://arxiv.org/abs/2511.08544

    dim, device = x.shape[-1], x.device

    # slice sampling

    rand_projs = torch.randn((num_slices, dim), device = device)
    rand_projs = l2norm(rand_projs)

    # integration points

    t = torch.linspace(*domain, num_knots, device = device)

    # theoretical CF for N(0, 1) and Gauss. window

    exp_f = (-0.5 * t.square()).exp()

    # empirical CF

    x_t = torch.einsum('... d, m d -> ... m', x, rand_projs)
    x_t = rearrange(x_t, '... m -> (...) m')

    x_t = rearrange(x_t, 'n m -> n m 1') * t
    ecf = (1j * x_t).exp().mean(dim = 0)

    # weighted L2 distance

    err = ecf.sub(exp_f).abs().square().mul(exp_f)

    return torch.trapezoid(err, t, dim = -1).mean()

# augmentation utils

class RandomApply(Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# MLP class for projector

class L2Norm(Module):
    def forward(self, x, eps = 1e-6):
        return l2norm(x, eps)

class MLP(Module):
    def __init__(self, dim, dim_out, num_layers, hidden_size = 256):
        super().__init__()

        layers = []
        dims = (dim, *((hidden_size,) * (num_layers - 1)))

        for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = ind == (len(dims) - 1)

            layers.extend([
                nn.Linear(layer_dim_in, layer_dim_out),
                nn.GELU() if not is_last else nn.Identity()
            ])

        self.net = nn.Sequential(
            *layers,
            L2Norm(),
            nn.Linear(hidden_size, dim_out)
        )

    def forward(self, x):
        return self.net(x)

# wrapper

class NetWrapper(Module):
    def __init__(self, net, output_dim, projection_hidden_size, projection_num_layers, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_hidden_size = projection_hidden_size
        self.projection_num_layers = projection_num_layers
        self.output_dim = output_dim

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output.flatten(1)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.output_dim, self.projection_num_layers, self.projection_hidden_size)
        return projector.to(hidden)

    def get_embedding(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        embed = self.get_embedding(x)
        if not return_projection:
            return embed

        projector = self._get_projector(embed)
        return projector(embed), embed

# main class

class LeJEPA(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_hidden_size = 256,
        num_classes_K = 65336,
        projection_layers = 4,
        local_upper_crop_scale = 0.4,
        global_lower_crop_scale = 0.5,
        target_loss_weight = 1.,
        sigreg_loss_weight = 1.,
        sigreg_loss_kwargs = dict(
            num_slices = 1024,
            domain = (-5, 5),
            num_knots = 17
        ),
        augment_fn = None,
        augment_fn2 = None
    ):
        super().__init__()
        self.net = net

        # default BYOL augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, DEFAULT_AUG)

        # local and global crops

        self.local_crop = T.RandomResizedCrop((image_size, image_size), scale = (0.05, local_upper_crop_scale))
        self.global_crop = T.RandomResizedCrop((image_size, image_size), scale = (global_lower_crop_scale, 1.))

        self.encoder = NetWrapper(net, num_classes_K, projection_hidden_size, projection_layers, layer = hidden_layer)

        self.target_loss_weight = target_loss_weight
        self.sigreg_loss_weight = sigreg_loss_weight
        self.sigreg_loss_kwargs = sigreg_loss_kwargs

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True
    ):
        if return_embedding:
            return self.encoder(x, return_projection = return_projection)

        image_one, image_two = self.augment1(x), self.augment2(x)

        local_image_one, local_image_two   = self.local_crop(image_one),  self.local_crop(image_two)
        global_image_one, global_image_two = self.global_crop(image_one), self.global_crop(image_two)

        local_images = torch.cat((local_image_one, local_image_two), dim = 0)
        proj_locals, _ = self.encoder(local_images)
        proj_local_one, proj_local_two = proj_locals.chunk(2, dim = 0)

        with torch.no_grad():
            global_images = torch.cat((global_image_one, global_image_two), dim = 0)
            proj_globals, _ = self.encoder(global_images)
            proj_global_one, proj_global_two = proj_globals.chunk(2, dim = 0)

        # invariance loss

        mse_loss = F.mse_loss(proj_local_one, proj_global_two) + F.mse_loss(proj_local_two, proj_global_one)

        # sigreg loss

        sreg_loss = sigreg_loss(proj_locals, **self.sigreg_loss_kwargs)

        return mse_loss * self.target_loss_weight + sreg_loss * self.sigreg_loss_weight

# quick run

if __name__ == '__main__':
    from vit_pytorch import ViT

    model = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048
    )

    learner = LeJEPA(
        model,
        image_size = 256,
        hidden_layer = 'to_latent',          # layer name where output is hidden dimension
        projection_hidden_size = 256,        # projector network hidden dimension
        projection_layers = 4,               # number of layers in projection network
        num_classes_K = 65336,               # output dimension
        target_loss_weight = 1.0,
        sigreg_loss_weight = 1.0
    )

    opt = torch.optim.Adam(learner.parameters(), lr = 3e-4)

    images = torch.randn(8, 3, 256, 256)

    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()

    print('loss:', loss.item())
