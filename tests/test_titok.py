import pytest
import torch
from titok_pytorch import TiTokTokenizer

def test_titok():

    images = torch.randn(2, 3, 256, 256)

    titok = TiTokTokenizer(
        dim = 512,
        num_latent_tokens = 32
    )

    loss = titok(images)
    loss.backward()

    # after much training
    # extract codes for gpt, maskgit, whatever

    codes = titok.tokenize(images)

    assert codes.shape == (2, 32)

    # reconstructing images from codes

    recon_images = titok.codebook_ids_to_images(codes)

    assert recon_images.shape == images.shape
