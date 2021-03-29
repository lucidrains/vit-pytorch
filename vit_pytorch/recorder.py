from functools import wraps
import torch

def exists(val):
    return val is not None

def record_wrapper(fn):
    @wraps(fn)
    def inner(model, img, **kwargs):
        rec = kwargs.pop('rec', None)
        if exists(rec):
            rec.clear()

        out = fn(model, img, rec = rec, **kwargs)

        if exists(rec):
            rec.finalize()
        return out
    return inner

class Recorder():
    def __init__(self):
        super().__init__()
        self._layer_attns = []
        self.attn = None

    def clear(self):
        self._layer_attns.clear()
        self.attn = None

    def finalize(self):
        self.attn = torch.stack(self._layer_attns, dim = 1)

    def record(self, attn):
        self._layer_attns.append(attn.clone().detach())
