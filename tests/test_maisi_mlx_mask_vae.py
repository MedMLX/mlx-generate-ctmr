"""Paired-mask MAISI VAE parity against pinned Torch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from golden_fixtures import seed_torch
from medmlx_core.ops import as_fp32
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

from mlx_generate_ctmr._arrays import as_numpy
from mlx_generate_ctmr.maisi_mlx_feasibility import VAE_ATOL, VAE_RTOL
from mlx_generate_ctmr.mask_vae import decode_mask_vae_ncdhw, encode_mask_vae_ncdhw

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Mlx, VaeConstructor


def test_mask_vae_encode_decode_matches_pinned_torch(mx: Mlx) -> None:
    """Compare mask VAE encoding and decoding with the source."""
    constructor: VaeConstructor = {
        "spatial_dims": 3,
        "in_channels": 8,
        "out_channels": 125,
        "latent_channels": 4,
        "num_channels": [32, 64, 128],
        "num_res_blocks": [1, 2, 2],
        "norm_num_groups": 32,
        "norm_eps": 1e-6,
        "attention_levels": [False, False, False],
        "with_encoder_nonlocal_attn": False,
        "with_decoder_nonlocal_attn": False,
        "use_flash_attention": False,
        "use_checkpointing": False,
        "use_convtranspose": True,
        "norm_float16": False,
        "num_splits": 1,
        "dim_split": 1,
        "print_info": False,
        "save_mem": False,
    }
    module = AutoencoderKlMaisi(**constructor).eval()
    seed_torch(23)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0.0, 0.02)
    packed = torch.randn(1, 8, 16, 16, 16)
    with torch.inference_mode():
        z_mu_t, z_sigma_t = module.encode(packed)
        logits_t = module.decode(z_mu_t)
    weights = {name: as_fp32(value.numpy(), mx) for name, value in module.state_dict().items()}
    z_mu, z_sigma = encode_mask_vae_ncdhw(
        as_fp32(packed.numpy(), mx), weights, constructor=constructor, mx=mx
    )
    _post_quant, logits = decode_mask_vae_ncdhw(z_mu, weights, constructor=constructor, mx=mx)
    for actual, expected in ((z_mu, z_mu_t), (z_sigma, z_sigma_t), (logits, logits_t)):
        np.testing.assert_allclose(
            as_numpy(actual), expected.numpy(), rtol=VAE_RTOL, atol=VAE_ATOL
        )
