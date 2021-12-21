import torch
from torch import nn

def exists(val):
    return val is not None

class Extractor(nn.Module):
    def __init__(self, vit, device = None):
        super().__init__()
        self.vit = vit

        self.data = None
        self.latents = None
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

    def _hook(self, _, input, output):
        self.latents = output.clone().detach()

    def _register_hook(self):
        handle = self.vit.transformer.register_forward_hook(self._hook)
        self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self):
        del self.latents
        self.latents = None

    def forward(self, img):
        assert not self.ejected, 'extractor has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        pred = self.vit(img)

        target_device = self.device if exists(self.device) else img.device
        latents = self.latents.to(target_device)
        return pred, latents
