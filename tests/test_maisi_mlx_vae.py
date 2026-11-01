"""MAISI MLX VAE decode/encode parity against recorded goldens and pinned Torch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from golden_fixtures import load_boundary, mlx_weights, seed_torch
from medmlx_core.ops import as_fp32
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

from mlx_generate_ctmr._arrays import as_numpy
from mlx_generate_ctmr.maisi_mlx_feasibility import VAE_ATOL, VAE_RTOL
from mlx_generate_ctmr.vae import decode_vae_ncdhw, encode_vae_ncdhw

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Mlx, VaeConstructor


def test_vae_decode_matches_torch_golden(mx: Mlx) -> None:
    """Compare native VAE decode output with the Torch recording."""
    manifest, arrays = load_boundary("vae")
    z, weights = as_fp32(arrays["z"], mx), mlx_weights(arrays, mx)
    post_quant, decoded = decode_vae_ncdhw(z, weights, constructor=manifest["constructor"], mx=mx)
    for actual, key in ((post_quant, "post_quant"), (decoded, "decoded")):
        np.testing.assert_allclose(as_numpy(actual), arrays[key], rtol=VAE_RTOL, atol=VAE_ATOL)


@pytest.mark.parametrize(
    "num_splits,image_size",
    [pytest.param(1, 16, id="unsplit"), pytest.param(2, 32, id="partitioned")],
)
def test_vae_encode_decode_smoke_matches_pinned_torch(
    mx: Mlx, num_splits: int, image_size: int
) -> None:
    """Compare a small native VAE encode-decode case with the source."""
    constructor: VaeConstructor = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "num_res_blocks": [1, 1, 1],
        "num_channels": [32, 32, 32],
        "attention_levels": [False, False, False],
        "latent_channels": 4,
        "norm_num_groups": 32,
        "norm_eps": 1e-6,
        "with_encoder_nonlocal_attn": False,
        "with_decoder_nonlocal_attn": False,
        "include_fc": False,
        "use_combined_linear": False,
        "use_flash_attention": False,
        "use_checkpointing": False,
        "use_convtranspose": False,
        "num_splits": 1,
        "dim_split": 1,
        "norm_float16": False,
        "print_info": False,
        "save_mem": False,
    }
    module = AutoencoderKlMaisi(**constructor).to("cpu").eval()
    seed_torch(23)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0.0, 0.02)
    image = torch.randn(1, 1, image_size, image_size, image_size)
    with torch.inference_mode():
        z_mu_t, z_sigma_t = module.encode(image)
        decoded_t = module.decode(z_mu_t)
    weights = {name: as_fp32(value.numpy(), mx) for name, value in module.state_dict().items()}
    # The Torch oracle and native decoder stay unsplit; vary encoder chunking only.
    native_constructor: VaeConstructor = {**constructor, "num_splits": num_splits}
    native_image = as_fp32(image.numpy(), mx)
    z_mu, z_sigma = encode_vae_ncdhw(native_image, weights, constructor=native_constructor, mx=mx)
    _post_quant, decoded = decode_vae_ncdhw(z_mu, weights, constructor=constructor, mx=mx)
    for actual, expected in ((z_mu, z_mu_t), (z_sigma, z_sigma_t), (decoded, decoded_t)):
        np.testing.assert_allclose(
            as_numpy(actual), expected.numpy(), rtol=VAE_RTOL, atol=VAE_ATOL
        )
