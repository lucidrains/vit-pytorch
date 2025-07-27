import torch
from torch import is_tensor, randn
from torch.nn import Module, Parameter
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
        output_pos_add_pos_emb = 0 # defaults to first output position to add embedding 
    ):
        super().__init__()
        self.image_net = image_net
        self.forward_function = forward_function # for openclip, used in TRI-LBM

        self.add_time_pos_emb = add_time_pos_emb
        self.output_pos_add_pos_emb = output_pos_add_pos_emb

        if add_time_pos_emb:
            assert exists(dim_emb) and exists(time_seq_len), '`dim_emb` and `time_seq_len` must be set if adding positional embeddings to the output'
            self.time_seq_len = time_seq_len

            self.pos_emb = Parameter(randn(time_seq_len, dim_emb) * 1e-2)

    def forward(
        self,
        video # (b c t h w)
    ):
        add_time_pos_emb = self.add_time_pos_emb
        batch, time = video.shape[0], video.shape[2]

        # maybe validate time positional embedding

        if add_time_pos_emb:
            assert time <= self.time_seq_len, f'received video with {time} frames but `time_seq_len` ({self.time_seq_len}) is too low'

        video = rearrange(video, 'b c t h w -> b t c h w')

        video = rearrange(video, 'b t ... -> (b t) ...')

        func = getattr(self.image_net, self.forward_function)

        outputs = func(video)

        # handle multiple outputs, say logits and embeddings returned from extractor - also handle some reduce aux loss being returned

        outputs, tree_spec = tree_flatten(outputs)

        outputs = tuple(rearrange(t, '(b t) ... -> b t ...', t = time) if is_tensor(t) and t.numel() > 1 else t for t in outputs)

        # maybe add time positional embedding

        if add_time_pos_emb:
            pos_emb = repeat(self.pos_emb, 't d -> b t 1 d', b = batch)

            outputs = list(outputs)
            embed = outputs[self.output_pos_add_pos_emb]

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

    videos = torch.randn(1, 3, 10, 256, 256)

    # step up the difficulty and return embeddings for robotics

    from vit_pytorch.extractor import Extractor
    v = Extractor(v)

    video_acceptor = AcceptVideoWrapper(v, add_time_pos_emb = True, output_pos_add_pos_emb = 1, time_seq_len = 10, dim_emb = 1024)

    logits, embeddings = video_acceptor(videos) # always (batch, channels, time, height, width) - time is always dimension 2

    assert logits.shape == (1, 10, 1000)
    assert embeddings.shape == (1, 10, 65, 1024)
