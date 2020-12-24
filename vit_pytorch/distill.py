import torch
import torch.nn.functional as F
from torch import nn
from vit_pytorch.vit_pytorch import ViT

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

# classes

class DistillableViT(ViT):
    def __init__(self, *args, **kwargs):
        super(DistillableViT, self).__init__(*args, **kwargs)
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def forward(self, img, distill_token, mask = None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        x += self.pos_embedding[:, :(n + 1)]

        distill_tokens = repeat(distill_token, '() n d -> b n d', b = b)
        x = torch.cat((x, distill_tokens), dim = 1)

        x = self.dropout(x)

        x = self.transformer(x, mask)

        x, distill_tokens = x[:, :-1], x[:, -1]
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x), distill_tokens

class DistillWrapper(nn.Module):
    def __init__(
        self,
        *,
        teacher,
        student,
        temperature = 1.,
        alpha = 0.5
    ):
        super().__init__()
        assert isinstance(student, DistillableViT), 'student must be a vision transformer'
        self.teacher = teacher
        self.student = student

        dim = student.dim
        num_classes = student.num_classes
        self.temperature = temperature
        self.alpha = alpha

        self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))

        self.distill_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, labels, temperature = None, alpha = None, **kwargs):
        b, *_ = img.shape
        alpha = alpha if exists(alpha) else self.alpha
        T = temperature if exists(temperature) else self.temperature

        with torch.no_grad():
            teacher_logits = self.teacher(img)

        student_logits, distill_tokens = self.student(img, distill_token = self.distillation_token, **kwargs)
        distill_logits = self.distill_mlp(distill_tokens)

        loss = F.cross_entropy(student_logits, labels)

        distill_loss = F.kl_div(
            F.log_softmax(distill_logits / T, dim = -1),
            F.softmax(teacher_logits / T, dim = -1).detach(),
        reduction = 'batchmean')

        distill_loss *= T ** 2

        return loss * alpha + distill_loss * (1 - alpha)
