import copy
import random
from functools import wraps, partial

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T

# helper functions

def exists(val):
    return val is not None

def default(val, default):
    return val if exists(val) else default

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

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss function # (algorithm 1 in the paper)

def loss_fn(
    teacher_logits,
    student_logits,
    teacher_temp,
    student_temp,
    centers,
    eps = 1e-20
):
    teacher_logits = teacher_logits.detach()
    student_probs = (student_logits / student_temp).softmax(dim = -1)
    teacher_probs = ((teacher_logits - centers) / teacher_temp).softmax(dim = -1)
    return - (teacher_probs * torch.log(student_probs + eps)).sum(dim = -1).mean()

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

class L2Norm(nn.Module):
    def forward(self, x, eps = 1e-6):
        norm = x.norm(dim = 1, keepdim = True).clamp(min = eps)
        return x / norm

class MLP(nn.Module):
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

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
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

class Dino(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_hidden_size = 256,
        num_classes_K = 65336,
        projection_layers = 4,
        student_temp = 0.9,
        teacher_temp = 0.04,
        local_upper_crop_scale = 0.4,
        global_lower_crop_scale = 0.5,
        moving_average_decay = 0.9,
        center_moving_average_decay = 0.9,
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

        self.student_encoder = NetWrapper(net, num_classes_K, projection_hidden_size, projection_layers, layer = hidden_layer)

        self.teacher_encoder = None
        self.teacher_ema_updater = EMA(moving_average_decay)

        self.register_buffer('teacher_centers', torch.zeros(1, num_classes_K))
        self.register_buffer('last_teacher_centers',  torch.zeros(1, num_classes_K))

        self.teacher_centering_ema_updater = EMA(center_moving_average_decay)

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('teacher_encoder')
    def _get_teacher_encoder(self):
        teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(teacher_encoder, False)
        return teacher_encoder

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

        new_teacher_centers = self.teacher_centering_ema_updater.update_average(self.teacher_centers, self.last_teacher_centers)
        self.teacher_centers.copy_(new_teacher_centers)

    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True,
        student_temp = None,
        teacher_temp = None
    ):
        if return_embedding:
            return self.student_encoder(x, return_projection = return_projection)

        image_one, image_two = self.augment1(x), self.augment2(x)

        local_image_one, local_image_two   = self.local_crop(image_one),  self.local_crop(image_two)
        global_image_one, global_image_two = self.global_crop(image_one), self.global_crop(image_two)

        student_proj_one, _ = self.student_encoder(local_image_one)
        student_proj_two, _ = self.student_encoder(local_image_two)

        with torch.no_grad():
            teacher_encoder = self._get_teacher_encoder()
            teacher_proj_one, _ = teacher_encoder(global_image_one)
            teacher_proj_two, _ = teacher_encoder(global_image_two)

        loss_fn_ = partial(
            loss_fn,
            student_temp = default(student_temp, self.student_temp),
            teacher_temp = default(teacher_temp, self.teacher_temp),
            centers = self.teacher_centers
        )

        teacher_logits_avg = torch.cat((teacher_proj_one, teacher_proj_two)).mean(dim = 0)
        self.last_teacher_centers.copy_(teacher_logits_avg)

        loss = (loss_fn_(teacher_proj_one, student_proj_two) + loss_fn_(teacher_proj_two, student_proj_one)) / 2
        return loss
