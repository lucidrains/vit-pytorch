import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.0.0'):
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()

from vit_pytorch.vit import ViT
from vit_pytorch.simple_vit import SimpleViT

from vit_pytorch.mae import MAE
from vit_pytorch.dino import Dino
