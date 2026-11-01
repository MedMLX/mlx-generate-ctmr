"""MLX port of ``ControlNetMaisi`` (NCDHW, rflow from-mask dest)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, ControlnetConstructor, Mlx, UnetArithmetic

from collections.abc import Mapping

from medmlx_core.layout import require_ncdhw

from mlx_generate_ctmr import source_fp32
from mlx_generate_ctmr.arithmetic_mode import unet_arithmetic
from mlx_generate_ctmr.embed import (
    add_class_to_time,
    embed_class,
)
from mlx_generate_ctmr.unet import (
    apply_down_blocks,
    apply_mid_block,
    apply_silu_mlp,
    bool_sequence,
    int_sequence,
    per_level,
    required_weight,
)


def controlnet_residuals_ncdhw(
    x: Array,
    timesteps: Array,
    class_labels: Array | None,
    cond: Array,
    weights: Mapping[str, Array],
    *,
    constructor: ControlnetConstructor,
    mx: Mlx,
    conditioning_scale: float = 1.0,
    conditioning_embedding: Array | None = None,
) -> tuple[list[Array], Array]:
    """Run the ControlNet; returns ``(down, mid)`` residuals for the UNet.

    ``x`` is the noisy latent ``(N, in_channels, D, H, W)``. ``cond`` is the packed
    label condition ``(N, 8, D*s, H*s, W*s)`` (see ``condition.binarize_labels_ncdhw``)
    with ``s = 2 ** (len(conditioning_embedding_num_channels) - 1)``, i.e. 4 for the
    released network. ``down`` has one residual per UNet skip connection and
    ``mid`` one for the mid block; both are multiplied by ``conditioning_scale``.
    ``timesteps`` and ``class_labels`` are as in ``unet.denoise_unet_ncdhw``.
    ``conditioning_embedding`` may reuse ``encode_controlnet_condition_ncdhw``'s
    float32 ``(N, num_channels[0], D, H, W)`` result for the same condition and
    weights. Only the mask encoder is reused; latent/time-dependent blocks run
    on every call. Shape and dtype are checked, but provenance remains caller-owned.

    Raises:
        ValueError: If shapes do not match the constructor, a required input is
            missing, or the constructor requests an unsupported feature.
    """

    require_ncdhw(x, name="controlnet latent")
    require_ncdhw(cond, name="controlnet cond")
    _refuse_unsupported(constructor)
    if int(x.shape[1]) != int(constructor["in_channels"]):
        raise ValueError(
            f"controlnet latent channels {int(x.shape[1])} != "
            f"in_channels {constructor['in_channels']}"
        )
    cond_in = int(constructor["conditioning_embedding_in_channels"])
    if int(cond.shape[1]) != cond_in:
        raise ValueError(f"controlnet cond channels {int(cond.shape[1])} != {cond_in}")
    cond_channels = int_sequence(
        constructor["conditioning_embedding_num_channels"],
        name="conditioning_embedding_num_channels",
    )
    scale = 2 ** (len(cond_channels) - 1)
    latent_spatial = tuple(int(dim) for dim in x.shape[2:])
    cond_spatial = tuple(int(dim) for dim in cond.shape[2:])
    expected = tuple(dim * scale for dim in latent_spatial)
    if cond_spatial != expected:
        raise ValueError(
            f"controlnet cond spatial {cond_spatial} must be {scale}x latent {latent_spatial}"
        )

    channels = int_sequence(constructor["num_channels"], name="num_channels")
    n_levels = len(channels)
    res_blocks = per_level(constructor["num_res_blocks"], n_levels, "num_res_blocks")
    attn_levels = bool_sequence(constructor["attention_levels"], name="attention_levels")
    if len(attn_levels) != n_levels:
        raise ValueError("attention_levels length must match num_channels")
    heads = per_level(constructor["num_head_channels"], n_levels, "num_head_channels")
    groups = int(constructor["norm_num_groups"])
    eps = float(constructor["norm_eps"])
    include_fc = bool(constructor.get("include_fc", True))
    resblock_updown = bool(constructor.get("resblock_updown", True))
    if conditioning_embedding is not None:
        expected_embedding = (int(x.shape[0]), channels[0], *latent_spatial)
        if tuple(conditioning_embedding.shape) != expected_embedding:
            raise ValueError(
                f"controlnet conditioning embedding shape {conditioning_embedding.shape} "
                f"must be {expected_embedding}"
            )
        if conditioning_embedding.dtype != mx.float32:
            raise ValueError("controlnet conditioning embedding must be float32")

    arithmetic = unet_arithmetic()
    t_emb = arithmetic.sinusoidal_timestep_embedding(timesteps, channels[0], mx=mx)
    emb = apply_silu_mlp(t_emb, weights, "time_embed", mx, arithmetic=arithmetic)
    if constructor.get("num_class_embeds") is not None:
        if class_labels is None:
            raise ValueError("class_labels should be provided when num_class_embeds > 0")
        class_emb = embed_class(
            class_labels.astype(mx.int32),
            required_weight(weights, "class_embedding.weight"),
            mx,
        )
        emb = add_class_to_time(emb, class_emb)

    hidden = arithmetic.conv3d_ncdhw(
        x,
        required_weight(weights, "conv_in.conv.weight"),
        required_weight(weights, "conv_in.conv.bias"),
        padding=1,
        mx=mx,
    )
    hidden = hidden + (
        _cond_embed(cond, weights, cond_channels, mx, arithmetic)
        if conditioning_embedding is None
        else conditioning_embedding
    )
    hidden, skips = apply_down_blocks(
        hidden,
        emb,
        weights,
        channels=channels,
        res_blocks=res_blocks,
        attn_levels=attn_levels,
        heads=heads,
        groups=groups,
        eps=eps,
        include_fc=include_fc,
        resblock_updown=resblock_updown,
        mx=mx,
        down_block_additional_residuals=None,
    )
    hidden = apply_mid_block(
        hidden,
        emb,
        weights,
        heads=heads[-1],
        groups=groups,
        eps=eps,
        include_fc=include_fc,
        mx=mx,
    )
    down = [
        _zero_conv(skip, weights, f"controlnet_down_blocks.{index}", mx, arithmetic)
        for index, skip in enumerate(skips)
    ]
    mid = _zero_conv(hidden, weights, "controlnet_mid_block", mx, arithmetic)
    if conditioning_scale != 1.0:
        down = [residual * conditioning_scale for residual in down]
        mid = mid * conditioning_scale
    return down, mid


def encode_controlnet_condition_ncdhw(
    cond: Array,
    weights: Mapping[str, Array],
    *,
    constructor: ControlnetConstructor,
    mx: Mlx,
) -> Array:
    """Encode a packed label condition once for a sequence of denoising steps.

    Input is ``(N, conditioning_embedding_in_channels, D*s, H*s, W*s)``;
    output is float32 ``(N, num_channels[0], D, H, W)``, where
    ``s = 2**(len(conditioning_embedding_num_channels) - 1)``. The result depends
    only on the supplied condition and weights, and stays lazy until evaluated.
    Invalid channel counts or spatial sizes raise ``ValueError``.
    """
    require_ncdhw(cond, name="controlnet cond")
    _refuse_unsupported(constructor)
    cond_in = int(constructor["conditioning_embedding_in_channels"])
    if int(cond.shape[1]) != cond_in:
        raise ValueError(f"controlnet cond channels {int(cond.shape[1])} != {cond_in}")
    channels = int_sequence(
        constructor["conditioning_embedding_num_channels"],
        name="conditioning_embedding_num_channels",
    )
    scale = 2 ** (len(channels) - 1)
    if any(int(size) % scale for size in cond.shape[2:]):
        raise ValueError(f"controlnet cond spatial sizes must be multiples of {scale}")
    return _cond_embed(cond, weights, channels, mx, unet_arithmetic())


def _cond_embed(
    cond: Array,
    weights: Mapping[str, Array],
    channels: tuple[int, ...],
    mx: Mlx,
    arithmetic: UnetArithmetic,
) -> Array:
    """Build the ControlNet conditioning embedding from the input mask."""
    conv3d_ncdhw = arithmetic.conv3d_ncdhw

    def silu(array: Array, mx: Mlx) -> Array:
        # MONAI's Swish here rounds the sigmoid separately from Torch SiLU's division.
        """Apply the selected arithmetic mode's SiLU activation."""
        if arithmetic is source_fp32:
            return source_fp32.swish(array, mx)
        return arithmetic.silu(array, mx)

    hidden = silu(
        conv3d_ncdhw(
            cond,
            required_weight(weights, "controlnet_cond_embedding.conv_in.conv.weight"),
            required_weight(weights, "controlnet_cond_embedding.conv_in.conv.bias"),
            padding=1,
            mx=mx,
        ),
        mx,
    )
    block = 0
    for _ in range(len(channels) - 1):
        hidden = silu(
            conv3d_ncdhw(
                hidden,
                required_weight(weights, f"controlnet_cond_embedding.blocks.{block}.conv.weight"),
                required_weight(weights, f"controlnet_cond_embedding.blocks.{block}.conv.bias"),
                padding=1,
                mx=mx,
            ),
            mx,
        )
        block += 1
        hidden = silu(
            conv3d_ncdhw(
                hidden,
                required_weight(weights, f"controlnet_cond_embedding.blocks.{block}.conv.weight"),
                required_weight(weights, f"controlnet_cond_embedding.blocks.{block}.conv.bias"),
                padding=1,
                stride=2,
                mx=mx,
            ),
            mx,
        )
        block += 1
    return conv3d_ncdhw(
        hidden,
        required_weight(weights, "controlnet_cond_embedding.conv_out.conv.weight"),
        required_weight(weights, "controlnet_cond_embedding.conv_out.conv.bias"),
        padding=1,
        mx=mx,
    )


def _zero_conv(
    array: Array,
    weights: Mapping[str, Array],
    prefix: str,
    mx: Mlx,
    arithmetic: UnetArithmetic,
) -> Array:
    """Apply a named zero-convolution projection."""
    wrapped = f"{prefix}.conv.weight"
    raw = f"{prefix}.weight"
    if wrapped in weights:
        weight, bias = weights[wrapped], required_weight(weights, f"{prefix}.conv.bias")
    elif raw in weights:
        weight, bias = weights[raw], required_weight(weights, f"{prefix}.bias")
    else:
        raise ValueError(f"ControlNet weights missing: {wrapped} or {raw}")
    return arithmetic.conv3d_ncdhw(array, weight, bias, padding=0, mx=mx)


def _refuse_unsupported(constructor: ControlnetConstructor) -> None:
    """``resblock_updown=False`` (``ddpm-ct``) is ported; see ``apply_down_blocks``."""

    if int(constructor.get("spatial_dims", 3)) != 3:
        raise ValueError("MAISI ControlNet dest is 3D NCDHW")
    if bool(constructor.get("use_flash_attention")):
        raise ValueError("CUDA flash attention is not an MLX destination")
    if bool(constructor.get("with_conditioning")):
        raise ValueError("MAISI ControlNet dest has with_conditioning=false")
    if bool(constructor.get("use_combined_linear")):
        raise ValueError("MAISI ControlNet dest uses separate QKV projections")
