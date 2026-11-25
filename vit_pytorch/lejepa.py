# Randall Balestriero, Yann LeCun - https://arxiv.org/abs/2511.08544

import random
from functools import wraps

import torch
import torch.nn.functional as F
from torch import cat, einsum, nn
from torch.nn import Module

from torchvision import transforms as T
from einops import rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if exists(instance):
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

# sigreg loss

def sigreg_loss(
    x,
    num_slices = 1024,
    domain = (-5, 5),
    num_knots = 17
):
    x = x.float() if x.dtype != torch.float64 else x
    dim, device, dtype = x.shape[-1], x.device, x.dtype

    rand_projs = l2norm(torch.randn((num_slices, dim), device = device, dtype = dtype))

    t = torch.linspace(*domain, num_knots, device = device, dtype = dtype)
    exp_f = (-0.5 * t.square()).exp()

    x_t = einsum('... d, m d -> ... m', x, rand_projs)
    x_t = rearrange(x_t, '... m -> (...) m 1') * t

    ecf_cos, ecf_sin = x_t.cos().mean(dim = 0), x_t.sin().mean(dim = 0)
    err = ((ecf_cos - exp_f).square() + ecf_sin.square()) * exp_f

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

# MLP projector

class L2Norm(Module):
    def forward(self, x, eps = 1e-6):
        return l2norm(x, eps)

class MLP(Module):
    def __init__(self, dim, dim_out, num_layers, hidden_size = 256):
        super().__init__()
        assert num_layers >= 1

        layers = []
        dims = (dim, *((hidden_size,) * (num_layers - 1)))

        for ind, (dim_in, dim_next) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = ind == (len(dims) - 2)
            layers.append(nn.Linear(dim_in, dim_next))
            if not is_last:
                layers.append(nn.GELU())

        self.net = nn.Sequential(
            *layers,
            L2Norm(),
            nn.Linear(dims[-1], dim_out)
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
        if isinstance(self.layer, str):
            return dict(self.net.named_modules()).get(self.layer, None)
        elif isinstance(self.layer, int):
            return [*self.net.children()][self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden[output.device] = output.flatten(1)

    def _register_hook(self):
        layer = self._find_layer()
        assert exists(layer), f'hidden layer ({self.layer}) not found'
        layer.register_forward_hook(self._hook)
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
        hidden = self.hidden.get(x.device)
        self.hidden.clear()

        assert exists(hidden), f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        embed = self.get_embedding(x)
        if not return_projection:
            return embed

        projector = self._get_projector(embed)
        return projector(embed), embed

# main class

class LeJEPA(Module):
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

        default_aug = nn.Sequential(
            RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p = 0.3),
            T.RandomGrayscale(p = 0.2),
            T.RandomHorizontalFlip(),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p = 0.2),
            T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        )

        self.augment1 = default(augment_fn, default_aug)
        self.augment2 = default(augment_fn2, default_aug)

        image_h, image_w = pair(image_size)

        self.local_crop = T.RandomResizedCrop((image_h, image_w), scale = (0.05, local_upper_crop_scale))
        self.global_crop = T.RandomResizedCrop((image_h, image_w), scale = (global_lower_crop_scale, 1.))

        self.encoder = NetWrapper(net, num_classes_K, projection_hidden_size, projection_layers, layer = hidden_layer)

        self.target_loss_weight = target_loss_weight
        self.sigreg_loss_weight = sigreg_loss_weight
        self.sigreg_loss_kwargs = sigreg_loss_kwargs

        device = get_module_device(net)
        self.to(device)

        # mock forward to instantiate singleton projector

        with torch.no_grad():
            self.forward(torch.randn(2, 3, image_h, image_w, device = device))

    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True
    ):
        if return_embedding:
            return self.encoder(x, return_projection = return_projection)

        image_one, image_two = self.augment1(x), self.augment2(x)

        local_image_one, local_image_two = self.local_crop(image_one), self.local_crop(image_two)
        global_image_one, global_image_two = self.global_crop(image_one), self.global_crop(image_two)

        local_images = cat((local_image_one, local_image_two), dim = 0)
        proj_locals, _ = self.encoder(local_images)
        proj_local_one, proj_local_two = proj_locals.chunk(2, dim = 0)

        with torch.no_grad():
            global_images = cat((global_image_one, global_image_two), dim = 0)
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
        hidden_layer = 'to_latent',
        projection_hidden_size = 256,
        projection_layers = 4,
        num_classes_K = 65336,
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
