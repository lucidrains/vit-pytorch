import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn


class SimMIM(nn.Module):
    def __init__(self, *, encoder, encoder_stride, in_chans=3, masking_ratio=0.5):
        super().__init__()
        assert (
            masking_ratio > 0 and masking_ratio < 1
        ), "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        self.in_chans = in_chans
        self.encoder_stride = encoder_stride
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=encoder_dim,
                out_channels=self.encoder_stride ** 2 * 3,
                kernel_size=1,
            ),
            nn.PixelShuffle(self.encoder_stride),
        )
        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_pixels = nn.Linear(encoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # for indexing purposes

        batch_range = torch.arange(batch, device=device)[:, None]

        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1 : (num_patches + 1)]

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb

        # prepare mask tokens

        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = (
            torch.rand(batch, num_patches, device=device)
            .topk(k=num_masked, dim=-1)
            .indices
        )
        masked_bool_mask = (
            torch.zeros((batch, num_patches), device=device)
            .scatter_(-1, masked_indices, 1)
            .bool()
        )

        # mask tokens
        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer
        encoded = self.encoder.transformer(tokens)

        # encoded = encoded[:, 1:]
        B, L, C = encoded.shape
        H = W = int(L ** 0.5)
        z = encoded.permute(0, 2, 1).reshape(B, C, H, W)

        x_rec = self.decoder(z)

        loss_recon = F.l1_loss(img, x_rec)
        return loss_recon / self.in_chans
        # mask weight
        # patch_size = int(num_patches ** 0.5)
        # mask_lst = []
        # for i, masked_indice in enumerate(masked_indices):
        #     mask = torch.ones(num_patches)
        #     mask[masked_indice] = 0
        #     mask_lst.append(mask.view(patch_size, patch_size))
        # mask = torch.stack(mask_lst, dim=0).to(device)

        # mask = (
        #     mask.repeat_interleave(patch_size, 1)
        #     .repeat_interleave(patch_size, 2)
        #     .unsqueeze(1)
        #     .contiguous()
        # )
        # loss_recon = F.l1_loss(img, x_rec, reduction="none")
        # return (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        # # simple head
        # # get the masked tokens
        # encoded_mask_tokens = encoded[batch_range, masked_indices]
        # # small linear projection for predicted pixel values
        # pred_pixel_values = self.to_pixels(encoded_mask_tokens)

        # # get the masked patches for the final reconstruction loss

        # masked_patches = patches[batch_range, masked_indices]

        # # calculate reconstruction loss

        # recon_loss = F.l1_loss(pred_pixel_values, masked_patches) / num_masked
        # return recon_loss


if __name__ == "__main__":
    import torch

    from vit import ViT

    v = ViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
    )

    mim = SimMIM(
        encoder=v,
        encoder_stride=16,  # for swin transformer, it should be 32
        masking_ratio=0.5,  # they found 50% to yield the best results
    )

    images = torch.randn(8, 3, 224, 224)

    loss = mim(images)
    loss.backward()
    print(loss)
