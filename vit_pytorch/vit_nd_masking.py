import math
import os
import sys

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# helpers

def exists(val):
    return val is not None

def nd_causal_mask_fn(num_patches_per_dim, causal_dims, has_cls = False):
    # returns a mask function (b, h, q_idx, k_idx) -> bool for use with flex attention
    # tokens are laid out in row-major fashion, with dimension 0 as the outermost (slowest varying) axis
    # a token may only attend to other tokens that are not ahead of it along any causal dimension

    num_patches_per_dim = tuple(num_patches_per_dim)

    def mask_fn(b, h, q_idx, k_idx):
        if has_cls:
            # cls token attends to all, and is attended to by all

            q_is_cls = q_idx == 0
            k_is_cls = k_idx == 0

            q_idx = (q_idx - 1).clamp(min = 0)
            k_idx = (k_idx - 1).clamp(min = 0)

        mask = torch.ones(q_idx.shape, dtype = torch.bool, device = q_idx.device)

        for causal_dim in causal_dims:
            stride = math.prod(num_patches_per_dim[causal_dim + 1:])
            q_coord = (q_idx // stride) % num_patches_per_dim[causal_dim]
            k_coord = (k_idx // stride) % num_patches_per_dim[causal_dim]
            mask = mask & (q_coord >= k_coord)

        if has_cls:
            mask = mask | q_is_cls | k_is_cls

        return mask

    return mask_fn

def get_nd_causal_mask_fn(num_patches_per_dim, causal_dims, ndim = None, has_cls = False):
    # returns the causal mask function, or None if causal_dims is not specified

    if not exists(causal_dims):
        return None

    causal_dims = (causal_dims,) if isinstance(causal_dims, int) else tuple(causal_dims)

    if exists(ndim):
        assert all(0 <= dim < ndim for dim in causal_dims), f'each causal dimension must be between 0 and {ndim - 1}'

    return nd_causal_mask_fn(num_patches_per_dim, causal_dims, has_cls = has_cls)

def flex_attention_supported(device) -> bool:
    # the compiled flex attention kernel is only lowered by inductor on select platforms
    # - cuda, or x86 cpu with avx2 (and not darwin)
    # elsewhere, fall back to manual attention so that torch.compile keeps working

    device = torch.device(device)

    if device.type == 'cuda':
        return True

    if device.type == 'cpu':
        avx2 = getattr(torch.cpu, '_is_avx2_supported', lambda: False)()
        return (
            avx2
            and not torch.xpu.is_available()
            and sys.platform != 'darwin'
            and os.getenv('ATEN_CPU_CAPABILITY') != 'default'
        )

    return False

def create_nd_block_mask(mask_fn, heads, seq_len, device):
    # only build the block mask when the compiled flex kernel is available

    if flex_attention_supported(device):
        return create_block_mask(mask_fn, 1, heads, seq_len, seq_len, device = device)

    return None

def nd_attention(q, k, v, mask_fn = None, block_mask = None, scale = None):
    # flex attention when the platform supports the compiled kernel,
    # otherwise manual attention with the dense mask (works under torch.compile everywhere)

    if flex_attention_supported(q.device):
        return flex_attention(q, k, v, block_mask = block_mask, scale = scale)

    n = q.shape[-2]

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    if exists(mask_fn):
        q_idx = torch.arange(n, device = q.device)[:, None]
        k_idx = torch.arange(n, device = q.device)[None, :]
        mask = mask_fn(0, 0, q_idx, k_idx)
        scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)

    attn = scores.softmax(dim = -1)

    return torch.matmul(attn, v)
