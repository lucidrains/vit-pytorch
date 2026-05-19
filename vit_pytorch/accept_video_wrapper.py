from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import is_tensor, randn
from torch.nn import Module, Linear, Parameter
from torch.utils._pytree import tree_flatten, tree_unflatten

from vit_pytorch.vivit_with_moss import MOSS

from einops import rearrange, repeat

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class AcceptVideoWrapper(Module):
    def __init__(
        self,
        image_net: Module,
        forward_function = 'forward',
        add_time_pos_emb = False,
        dim_emb = None,
        time_seq_len = None,
        embed_is_channel_first = False,
        output_pos_add_pos_emb = 0, # defaults to first output position to add embedding
        proj_embed_to_dim = None,
        patch_size = None,
        moss: Module | dict | None = None
    ):
        super().__init__()
        self.image_net = image_net
        self.forward_function = forward_function # for openclip, used in TRI-LBM

        self.add_time_pos_emb = add_time_pos_emb
        self.output_pos_add_pos_emb = output_pos_add_pos_emb

        # maybe project the image embedding

        self.embed_proj = None

        if exists(proj_embed_to_dim):
            assert exists(dim_emb), '`dim_emb` must be passed in'
            self.embed_proj = Linear(dim_emb, proj_embed_to_dim)

        # time positional embedding

        if add_time_pos_emb:
            assert exists(dim_emb) and exists(time_seq_len), '`dim_emb` and `time_seq_len` must be set if adding positional embeddings to the output'
            self.time_seq_len = time_seq_len

            dim_pos_emb = default(proj_embed_to_dim, dim_emb)

            self.pos_emb = Parameter(randn(time_seq_len, dim_pos_emb) * 1e-2)

        self.embed_is_channel_first = embed_is_channel_first

        # patch size and moss

        if not exists(patch_size):
            if hasattr(image_net, 'patch_size'):
                patch_size = image_net.patch_size
            elif hasattr(image_net, 'vit') and hasattr(image_net.vit, 'patch_size'):
                patch_size = image_net.vit.patch_size

        self.patch_size = patch_size

        if isinstance(moss, dict):
            moss = MOSS(**moss)

        self.moss = moss

        if exists(self.moss):
            assert exists(self.patch_size), '`patch_size` must be provided either on the `image_net` or passed in explicitly if using MOSS'

    def forward(
        self,
        video, # (b c t h w)
        eval_with_no_grad = False,
        forward_kwargs = dict()
    ):
        add_time_pos_emb = self.add_time_pos_emb
        time = video.shape[2]

        # maybe validate time positional embedding

        if add_time_pos_emb:
            assert time <= self.time_seq_len, f'received video with {time} frames but `time_seq_len` ({self.time_seq_len}) is too low'

        video_height, video_width = video.shape[-2:]

        video = rearrange(video, 'b c t h w -> b t c h w')

        video = rearrange(video, 'b t ... -> (b t) ...')

        # forward through image net for outputs

        func = getattr(self.image_net, self.forward_function)

        if eval_with_no_grad:
            self.image_net.eval()

        context = torch.no_grad if eval_with_no_grad else nullcontext

        with context():
            outputs = func(video, **forward_kwargs)

        # handle multiple outputs, say logits and embeddings returned from extractor - also handle some reduce aux loss being returned

        outputs, tree_spec = tree_flatten(outputs)

        outputs = tuple(rearrange(t, '(b t) ... -> b t ...', t = time) if is_tensor(t) and t.numel() > 1 else t for t in outputs)

        # maybe project embedding

        if exists(self.embed_proj):
            outputs = list(outputs)

            embed = outputs[self.output_pos_add_pos_emb]

            outputs[self.output_pos_add_pos_emb] = self.embed_proj(embed)

        # maybe add time positional embedding

        if add_time_pos_emb:

            outputs = list(outputs)
            embed = outputs[self.output_pos_add_pos_emb]

            pos_emb = rearrange(self.pos_emb, 't d -> 1 t d')

            # handle the network outputting embeddings with spatial dimensions intact - assume embedded dimension is last

            dims_to_unsqueeze = embed.ndim - pos_emb.ndim

            one_dims = ((1,) * dims_to_unsqueeze)

            if self.embed_is_channel_first:
                pos_emb = pos_emb.reshape(*pos_emb.shape, *one_dims)
            else:
                pos_emb = pos_emb.reshape(*pos_emb.shape[:2], *one_dims, pos_emb.shape[-1])

            pos_emb = pos_emb[:, :embed.shape[1]]

            embed = embed + pos_emb

            outputs[self.output_pos_add_pos_emb] = embed

        # moss - stack of ssts
        # https://openreview.net/forum?id=Co6SCyBIjo

        if exists(self.moss):
            outputs = list(outputs)
            embed = outputs[self.output_pos_add_pos_emb]

            patch_h, patch_w = pair(self.patch_size)
            num_h, num_w = video_height // patch_h, video_width // patch_w
            num_patches = num_h * num_w

            num_cls_tokens = embed.shape[-2] - num_patches
            cls_tokens, patch_tokens = embed[:, :, :num_cls_tokens], embed[:, :, num_cls_tokens:]

            patch_tokens = rearrange(patch_tokens, 'b t (h w) d -> b t h w d', h = num_h, w = num_w)
            patch_tokens = self.moss(patch_tokens)
            patch_tokens = rearrange(patch_tokens, 'b t h w d -> b t (h w) d')

            embed = torch.cat((cls_tokens, patch_tokens), dim = -2)
            outputs[self.output_pos_add_pos_emb] = embed

        return tree_unflatten(outputs, tree_spec)

# main

if __name__ == '__main__':
    from vit_pytorch import ViT

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

    videos = torch.randn(1, 3, 7, 256, 256)

    # step up the difficulty and return embeddings for robotics

    from vit_pytorch.extractor import Extractor

    v = Extractor(v)

    moss_kwargs = dict(
        dim = 512,
        local_time = 3,
        local_height = 3,
        local_width = 3,
        hidden_dim = 64,
        orders = 2,
        causal = True
    )

    video_acceptor = AcceptVideoWrapper(
        v,
        add_time_pos_emb = True,
        output_pos_add_pos_emb = 1,
        time_seq_len = 12,
        dim_emb = 1024,
        proj_embed_to_dim = 512,
        moss = moss_kwargs
    )

    logits, embeddings = video_acceptor(videos, eval_with_no_grad = True) # always (batch, channels, time, height, width) - time is always dimension 2

    assert logits.shape == (1, 7, 1000)
    assert embeddings.shape == (1, 7, 65, 512)
