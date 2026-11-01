"""Dedicated paired-mask MAISI autoencoder wrappers on top of shared VAE blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, Mlx, VaeConstructor

from collections.abc import Mapping

from mlx_generate_ctmr.arithmetic_mode import vae_arithmetic
from mlx_generate_ctmr.vae import (
    decode_vae_ncdhw,
    encode_vae_ncdhw,
)

MASK_LOGIT_CHANNELS = 125

# Full-size topology of the released mask autoencoder, from config_network_rflow.json.
# Keep group norm in fp32 and leave checkpointing off.
PRODUCTION_MASK_VAE_CONSTRUCTOR: VaeConstructor = {
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
    "num_splits": 4,
    "dim_split": 1,
    "print_info": False,
    "save_mem": False,
}


def encode_mask_vae_ncdhw(
    packed_label: Array,
    weights: Mapping[str, Array],
    *,
    constructor: VaeConstructor,
    mx: Mlx,
) -> tuple[Array, Array]:
    """Encode packed label bits to ``(mu, sigma)`` latents.

    With the production constructor, input is ``(N, 8, D, H, W)``; when the
    spatial dimensions are divisible by 4, each latent is
    ``(N, 4, D/4, H/4, W/4)``. The packing is the one produced by
    ``condition.binarize_labels_ncdhw``. Custom constructors determine the latent
    channels and spatial reduction.
    """

    return encode_vae_ncdhw(
        packed_label,
        weights,
        constructor=_encoder_constructor(constructor),
        mx=mx,
    )


def decode_mask_vae_ncdhw(
    latent: Array,
    weights: Mapping[str, Array],
    *,
    constructor: VaeConstructor,
    mx: Mlx,
) -> tuple[Array, Array]:
    """Decode mask latents to 125-channel logits.

    The latent must already be divided by the mask autoencoder's scale factor.
    For the production constructor, input shape is ``(N, 4, D, H, W)`` and logits
    are ``(N, 125, 4D, 4H, 4W)``. Custom constructors determine latent channels and
    spatial expansion. Returns ``(post_quant, logits)``.
    """

    _refuse_unsupported(constructor)
    post_quant, logits = decode_vae_ncdhw(
        latent,
        weights,
        constructor=constructor,
        mx=mx,
        arithmetic=vae_arithmetic(),
    )
    if int(logits.shape[1]) != MASK_LOGIT_CHANNELS:
        raise ValueError(
            f"mask VAE logits must have {MASK_LOGIT_CHANNELS} channels; got {int(logits.shape[1])}"
        )
    return post_quant, logits


def _encoder_constructor(constructor: VaeConstructor) -> VaeConstructor:
    """Derive the encoder constructor from the mask VAE configuration."""
    resolved = cast("VaeConstructor", dict(constructor))
    resolved["use_convtranspose"] = False
    return resolved


def _refuse_unsupported(constructor: VaeConstructor) -> None:
    """Reject mask VAE options outside the implemented contract."""
    if not bool(constructor.get("use_convtranspose")):
        raise ValueError("mask VAE dest requires ConvTranspose3d upsampling")
    if bool(constructor.get("with_encoder_nonlocal_attn")) or bool(
        constructor.get("with_decoder_nonlocal_attn")
    ):
        raise ValueError("mask VAE dest has no nonlocal attention")
    if bool(constructor.get("use_flash_attention")):
        raise ValueError("mask VAE dest does not use flash attention")
