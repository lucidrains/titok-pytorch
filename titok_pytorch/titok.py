from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops.layers.torch import Rearrange
from einops import rearrange, repeat, pack, unpack

from vector_quantize_pytorch import (
    VectorQuantize as VQ,
    LFQ
)

from x_transformers import Encoder

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def pack_square_height_width(t):
    assert t.ndim == 4
    return rearrange(t, 'b h w d -> b (h w) d')

def unpack_square_height_width(t):
    assert t.ndim == 3
    hw = int(sqrt(t.shape[1]))
    return rearrange(t, 'b (h w) d -> b h w d', h = hw, w = hw)

# tokenizer

class TiTokTokenizer(Module):
    def __init__(
        self,
        *,
        dim,
        image_size = 256,
        patch_size = 32,
        channels = 3,
        num_latent_tokens = 32,
        enc_depth = 6,
        dec_depth = 6,
        codebook_size = 8192,
        enc_kwargs: dict = dict(),
        dec_kwargs: dict = dict(),
        vq_kwargs: dict = dict()
    ):
        super().__init__()
        """
        ein notation:
        b - batch
        c - channels
        p - patch
        h - height
        w - width
        l - latents
        """

        self.image_size = image_size

        assert divisible_by(image_size, patch_size)

        dim_patch = channels * patch_size ** 2
        num_tokens = (image_size // patch_size) ** 2

        self.latents = nn.Parameter(torch.zeros(num_latent_tokens, dim))
        self.pos_emb = nn.Parameter(torch.zeros(num_tokens, dim))
        self.mask_tokens = nn.Parameter(torch.zeros(num_tokens, dim))

        nn.init.normal_(self.latents, std = 0.02)
        nn.init.normal_(self.pos_emb, std = 0.02)
        nn.init.normal_(self.mask_tokens, std = 0.02)

        self.image_to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(dim_patch, dim)
        )

        self.encoder = Encoder(
            dim = dim,
            depth = enc_depth,
            **enc_kwargs
        )

        self.vq = VQ(
            dim = dim,
            codebook_dim = dim,
            codebook_size = codebook_size,
            **vq_kwargs
        )

        self.decoder = Encoder(
            dim = dim,
            depth = dec_depth,
            **dec_kwargs
        )

        self.tokens_to_image = nn.Sequential(
            nn.Linear(dim, dim_patch),
            Rearrange('b h w (c p1 p2) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size)
        )

    @torch.no_grad()
    def tokenize(self, images):
        return self.forward(images, return_codebook_ids = True)

    def codebook_ids_to_images(self, token_ids):
        codes = self.vq.get_output_from_indices(token_ids)
        return self.decode(codes)

    def decode(self, latents):
        batch = latents.shape[0]

        # append mask tokens

        mask_tokens = repeat(self.mask_tokens, 'n d -> b n d', b = batch)

        tokens, mask_packed_shape = pack([mask_tokens, latents], 'b * d')

        # decode

        tokens = self.decoder(tokens)

        tokens, _ = unpack(tokens, mask_packed_shape, 'b * d')

        tokens = unpack_square_height_width(tokens)

        # tokens to image patches

        recon = self.tokens_to_image(tokens)
        return recon

    def forward(
        self,
        images,
        return_codebook_ids = False,
        return_recon_images = False
    ):
        assert images.ndim == 4 and images.shape[-2:] == ((self.image_size,) * 2)

        batch = images.shape[0]
        orig_images = images

        # image patches to tokens

        tokens = self.image_to_tokens(images)

        tokens = pack_square_height_width(tokens)

        # add absolute positions

        pos_emb = repeat(self.pos_emb, 'n d -> b n d', b = batch)

        tokens = tokens + pos_emb

        # concat latents

        latents = repeat(self.latents, 'l d -> b l d', b = batch)

        tokens, latents_packed_shape = pack([tokens, latents], 'b * d')

        # encoder

        tokens = self.encoder(tokens)

        # slice out latents and pass through vq as codes
        # this is the important line of code and main proposal of the paper

        _, latents = unpack(tokens, latents_packed_shape, 'b * d')

        # vq - usually tokens here, but they do the latents

        quantized, indices, _ = self.vq(latents)

        # whether to early return

        if return_codebook_ids:
            return indices

        recon_images = self.decode(quantized)

        # reconstruction loss

        recon_loss = F.mse_loss(recon_images, orig_images)

        if not return_recon_images:
            return recon_loss

        return recon_loss, recon_images
