"""Shared CT/MR rflow constructors and CT-from-mask parity tolerances."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import ControlnetConstructor, RflowConstructor, VaeConstructor

from mlx_generate_ctmr.maisi_mlx_feasibility import (
    UNET_ATOL,
    UNET_RTOL,
    VAE_ATOL,
    VAE_RTOL,
)

# Full-size constructors for the released networks, from config_network_rflow.json.
# Keep flash attention off and the VAE affine path in fp32. Do not put the tiny
# test-network configurations here.
PRODUCTION_VAE_CONSTRUCTOR: Final[VaeConstructor] = {
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 1,
    "latent_channels": 4,
    "num_channels": [64, 128, 256],
    "num_res_blocks": [2, 2, 2],
    "norm_num_groups": 32,
    "norm_eps": 1e-6,
    "attention_levels": [False, False, False],
    "with_encoder_nonlocal_attn": False,
    "with_decoder_nonlocal_attn": False,
    "include_fc": False,
    "use_combined_linear": False,
    "use_flash_attention": False,
    "use_checkpointing": False,
    "use_convtranspose": False,
    "num_splits": 4,
    "dim_split": 1,
    "norm_float16": False,
    "print_info": False,
    "save_mem": False,
}
PRODUCTION_UNET_CONSTRUCTOR: Final[RflowConstructor] = {
    "spatial_dims": 3,
    "in_channels": 4,
    "out_channels": 4,
    "num_channels": [64, 128, 256, 512],
    "attention_levels": [False, False, True, True],
    "num_head_channels": [0, 0, 32, 32],
    "num_res_blocks": 2,
    "norm_num_groups": 32,
    "norm_eps": 1e-6,
    "use_flash_attention": False,
    "include_top_region_index_input": False,
    "include_bottom_region_index_input": False,
    "include_spacing_input": True,
    "num_class_embeds": 128,
    "resblock_updown": True,
    "include_fc": True,
    "with_conditioning": False,
    "use_combined_linear": False,
}
PRODUCTION_CONTROLNET_CONSTRUCTOR: Final[ControlnetConstructor] = {
    "spatial_dims": 3,
    "in_channels": 4,
    "num_channels": [64, 128, 256, 512],
    "attention_levels": [False, False, True, True],
    "num_head_channels": [0, 0, 32, 32],
    "num_res_blocks": 2,
    "norm_num_groups": 32,
    "norm_eps": 1e-6,
    "use_flash_attention": False,
    "conditioning_embedding_in_channels": 8,
    "conditioning_embedding_num_channels": [8, 32, 64],
    "num_class_embeds": 128,
    "resblock_updown": True,
    "include_fc": True,
    "with_conditioning": False,
    "use_combined_linear": False,
}

# The two-step golden path runs UNet and ControlNet before VAE decode. Use the
# same fp32 tolerance budgets as the named component checks.
E2E_LATENT_RTOL: Final = UNET_RTOL
E2E_LATENT_ATOL: Final = UNET_ATOL
E2E_DECODE_RTOL: Final = VAE_RTOL
E2E_DECODE_ATOL: Final = VAE_ATOL
# HU = clip(decoded) * 2000 - 1000, so absolute drift scales with the affine.
E2E_HU_RTOL: Final = VAE_RTOL
E2E_HU_ATOL: Final = VAE_ATOL * 2000.0
