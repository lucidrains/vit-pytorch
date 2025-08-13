from contextlib import nullcontext

import torch
from torch import is_tensor, randn
from torch.nn import Module, Linear, Parameter
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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
        proj_embed_to_dim = None
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

    video_acceptor = AcceptVideoWrapper(v, add_time_pos_emb = True, output_pos_add_pos_emb = 1, time_seq_len = 12, dim_emb = 1024, proj_embed_to_dim = 512)

    logits, embeddings = video_acceptor(videos, eval_with_no_grad = True) # always (batch, channels, time, height, width) - time is always dimension 2

    assert logits.shape == (1, 7, 1000)
    assert embeddings.shape == (1, 7, 65, 512)
