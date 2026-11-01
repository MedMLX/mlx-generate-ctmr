"""MLX port of ``DiffusionModelUNetMaisi`` (NCDHW, rflow from-mask dest)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, Mlx, RflowConstructor

from collections.abc import Mapping, Sequence

from medmlx_core.layout import require_ncdhw
from medmlx_core.ops import (
    concat_channels_ncdhw,
)

from mlx_generate_ctmr import fast_fp32
from mlx_generate_ctmr.arithmetic_mode import unet_arithmetic
from mlx_generate_ctmr.attention import spatial_attention_ncdhw
from mlx_generate_ctmr.embed import (
    add_class_to_time,
    concat_bottom_region_index,
    concat_spacing,
    concat_top_region_index,
    embed_class,
)
from mlx_generate_ctmr.residual import residual_block_ncdhw
from mlx_generate_ctmr.unet_common import (
    apply_silu_mlp,
    bool_sequence,
    downsample_conv_ncdhw,
    int_sequence,
    per_level,
    required_weight,
    upsample_conv_ncdhw,
)
from mlx_generate_ctmr.unet_common import (
    prefixed_weights as _prefixed,
)


def denoise_unet_ncdhw(
    x: Array,
    timesteps: Array,
    class_labels: Array | None,
    spacing: Array | None,
    weights: Mapping[str, Array],
    *,
    constructor: RflowConstructor,
    mx: Mlx,
    down_block_additional_residuals: Sequence[Array] | None = None,
    mid_block_additional_residual: Array | None = None,
    top_region_index: Array | None = None,
    bottom_region_index: Array | None = None,
) -> tuple[Array, Array]:
    """Run one denoiser pass; returns ``(t_emb, y)``.

    ``x`` is the noisy latent ``(N, in_channels, D, H, W)`` float32 and ``y`` the
    network output ``(N, out_channels, D, H, W)`` (a velocity for rflow-CT/MR, the
    noise estimate for ddpm-ct). ``t_emb`` is the sinusoidal timestep embedding
    ``(N, num_channels[0])``. ``timesteps`` is ``(N,)`` in 0..1000 and
    ``class_labels`` ``(N,)`` int. ``spacing`` is ``(N, 3)`` voxel size in mm
    multiplied by 100 (see ``condition.spacing_tensor_ncdhw``). ``weights`` is a
    flat state dict with Torch tensor layouts (``Linear`` ``(out, in)``).
    ``down_block_additional_residuals`` and ``mid_block_additional_residual`` are
    the ControlNet outputs added to the skip connections and the mid block.

    ``top_region_index``/``bottom_region_index`` are the ``(N, 4)`` ddpm-ct
    body-region one-hot inputs (``include_top_region_index_input`` and
    ``include_bottom_region_index_input``); rflow-ct leaves both ``None``. The
    timestep range, one-hot values, and matching batch sizes are caller
    expectations rather than validations performed by this entry point.

    Raises:
        ValueError: If an input has the wrong shape, a required conditioning input
            is missing, or the constructor requests an unsupported feature.
    """

    require_ncdhw(x, name="unet input")
    _refuse_unsupported(constructor)
    if int(x.shape[1]) != int(constructor["in_channels"]):
        raise ValueError(
            f"unet input channels {int(x.shape[1])} != in_channels {constructor['in_channels']}"
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

    arithmetic = unet_arithmetic()
    t_emb, emb = _condition_embedding(
        timesteps,
        class_labels,
        top_region_index,
        bottom_region_index,
        spacing,
        weights,
        constructor=constructor,
        mx=mx,
    )
    hidden = arithmetic.conv3d_ncdhw(
        x,
        required_weight(weights, "conv_in.conv.weight"),
        required_weight(weights, "conv_in.conv.bias"),
        padding=1,
        mx=mx,
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
        down_block_additional_residuals=down_block_additional_residuals,
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
    if mid_block_additional_residual is not None:
        require_ncdhw(mid_block_additional_residual, name="unet mid residual")
        hidden = hidden + mid_block_additional_residual
    hidden = _apply_up_blocks(
        hidden,
        emb,
        skips,
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
    )
    if arithmetic is fast_fp32:
        y = fast_fp32.norm_conv3d_ncdhw(
            hidden,
            required_weight(weights, "out.2.conv.weight"),
            required_weight(weights, "out.2.conv.bias"),
            norm=(required_weight(weights, "out.0.weight"), required_weight(weights, "out.0.bias")),
            num_groups=groups,
            eps=eps,
            apply_silu=True,
            mx=mx,
        )
        return t_emb, y
    hidden = arithmetic.group_norm_ncdhw(
        hidden,
        required_weight(weights, "out.0.weight"),
        required_weight(weights, "out.0.bias"),
        num_groups=groups,
        eps=eps,
        mx=mx,
    )
    hidden = arithmetic.silu(hidden, mx)
    y = arithmetic.conv3d_ncdhw(
        hidden,
        required_weight(weights, "out.2.conv.weight"),
        required_weight(weights, "out.2.conv.bias"),
        padding=1,
        mx=mx,
    )
    return t_emb, y


def _condition_embedding(
    timesteps: Array,
    class_labels: Array | None,
    top_region_index: Array | None,
    bottom_region_index: Array | None,
    spacing: Array | None,
    weights: Mapping[str, Array],
    *,
    constructor: RflowConstructor,
    mx: Mlx,
) -> tuple[Array, Array]:
    """Build the UNet conditioning vector from the timestep and optional side inputs.

    Returns the sinusoidal timestep embedding and the full embedding: the
    time-MLP output, plus class, top/bottom region-index (N, 4) and spacing
    (N, 3) embeddings as enabled by the constructor. Raises ValueError when an
    enabled input is missing or has the wrong shape.
    """
    arithmetic = unet_arithmetic()
    t_emb = arithmetic.sinusoidal_timestep_embedding(
        timesteps, int(constructor["num_channels"][0]), mx=mx
    )
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
    if constructor.get("include_top_region_index_input"):
        if top_region_index is None:
            raise ValueError(
                "top_region_index should be provided when include_top_region_index_input is true"
            )
        if int(top_region_index.ndim) != 2 or int(top_region_index.shape[-1]) != 4:
            raise ValueError(
                "top_region_index must be [N, 4]; got shape="
                f"{tuple(int(d) for d in top_region_index.shape)}"
            )
        emb = concat_top_region_index(
            emb,
            apply_silu_mlp(
                top_region_index, weights, "top_region_index_layer", mx, arithmetic=arithmetic
            ),
            mx,
        )
    if constructor.get("include_bottom_region_index_input"):
        if bottom_region_index is None:
            raise ValueError(
                "bottom_region_index should be provided when "
                "include_bottom_region_index_input is true"
            )
        if int(bottom_region_index.ndim) != 2 or int(bottom_region_index.shape[-1]) != 4:
            raise ValueError(
                "bottom_region_index must be [N, 4]; got shape="
                f"{tuple(int(d) for d in bottom_region_index.shape)}"
            )
        emb = concat_bottom_region_index(
            emb,
            apply_silu_mlp(
                bottom_region_index,
                weights,
                "bottom_region_index_layer",
                mx,
                arithmetic=arithmetic,
            ),
            mx,
        )
    if constructor.get("include_spacing_input"):
        if spacing is None:
            raise ValueError("spacing should be provided when include_spacing_input is true")
        if int(spacing.ndim) != 2 or int(spacing.shape[-1]) != 3:
            raise ValueError(
                f"spacing must be [N, 3]; got shape={tuple(int(d) for d in spacing.shape)}"
            )
        emb = concat_spacing(
            emb, apply_silu_mlp(spacing, weights, "spacing_layer", mx, arithmetic=arithmetic), mx
        )
    return t_emb, emb


def apply_down_blocks(
    hidden: Array,
    emb: Array,
    weights: Mapping[str, Array],
    *,
    channels: Sequence[int],
    res_blocks: Sequence[int],
    attn_levels: Sequence[bool],
    heads: Sequence[int],
    groups: int,
    eps: float,
    include_fc: bool,
    mx: Mlx,
    down_block_additional_residuals: Sequence[Array] | None,
    resblock_updown: bool = True,
) -> tuple[Array, list[Array]]:
    """Run the NCDHW encoder blocks and collect their intermediate skip tensors.

    When supplied, ControlNet residuals are added to the corresponding skips;
    their count must match the skip count. The returned hidden state includes the
    residual added to the last skip.
    """
    arithmetic = unet_arithmetic()
    skips: list[Array] = [hidden]
    n_levels = len(channels)
    for index, (_out_ch, n_res, has_attn, n_heads) in enumerate(
        zip(channels, res_blocks, attn_levels, heads, strict=True)
    ):
        for res_index in range(n_res):
            hidden = residual_block_ncdhw(
                hidden,
                emb,
                _prefixed(weights, f"down_blocks.{index}.resnets.{res_index}"),
                num_groups=groups,
                eps=eps,
                mx=mx,
                arithmetic=arithmetic,
            )
            if has_attn:
                hidden = spatial_attention_ncdhw(
                    hidden,
                    _prefixed(weights, f"down_blocks.{index}.attentions.{res_index}"),
                    num_groups=groups,
                    eps=eps,
                    num_head_channels=n_heads,
                    include_fc=include_fc,
                    mx=mx,
                    arithmetic=arithmetic,
                )
            skips.append(hidden)
        if index != n_levels - 1:
            if resblock_updown:
                hidden = residual_block_ncdhw(
                    hidden,
                    emb,
                    _prefixed(weights, f"down_blocks.{index}.downsampler"),
                    num_groups=groups,
                    eps=eps,
                    mx=mx,
                    down=True,
                    arithmetic=arithmetic,
                )
            else:
                hidden = downsample_conv_ncdhw(
                    hidden,
                    weights,
                    f"down_blocks.{index}.downsampler",
                    mx=mx,
                    arithmetic=arithmetic,
                )
            skips.append(hidden)
    if down_block_additional_residuals is not None:
        extras = list(down_block_additional_residuals)
        if len(extras) != len(skips):
            raise ValueError(
                f"ControlNet residuals length {len(extras)} != UNet skip count {len(skips)}"
            )
        for residual in extras:
            require_ncdhw(residual, name="unet down residual")
        skips = [skip + residual for skip, residual in zip(skips, extras, strict=True)]
        # MONAI's in-place addition also updates h, which aliases the last skip.
        hidden = skips[-1]
    return hidden, skips


def apply_mid_block(
    hidden: Array,
    emb: Array,
    weights: Mapping[str, Array],
    *,
    heads: int,
    groups: int,
    eps: float,
    include_fc: bool,
    mx: Mlx,
) -> Array:
    """Run the conditioned middle residual-attention-residual blocks in NCDHW order."""
    arithmetic = unet_arithmetic()
    hidden = residual_block_ncdhw(
        hidden,
        emb,
        _prefixed(weights, "middle_block.resnet_1"),
        num_groups=groups,
        eps=eps,
        mx=mx,
        arithmetic=arithmetic,
    )
    hidden = spatial_attention_ncdhw(
        hidden,
        _prefixed(weights, "middle_block.attention"),
        num_groups=groups,
        eps=eps,
        num_head_channels=heads,
        include_fc=include_fc,
        mx=mx,
        arithmetic=arithmetic,
    )
    return residual_block_ncdhw(
        hidden,
        emb,
        _prefixed(weights, "middle_block.resnet_2"),
        num_groups=groups,
        eps=eps,
        mx=mx,
        arithmetic=arithmetic,
    )


def _apply_up_blocks(
    hidden: Array,
    emb: Array,
    skips: list[Array],
    weights: Mapping[str, Array],
    *,
    channels: Sequence[int],
    res_blocks: Sequence[int],
    attn_levels: Sequence[bool],
    heads: Sequence[int],
    groups: int,
    eps: float,
    include_fc: bool,
    mx: Mlx,
    resblock_updown: bool = True,
) -> Array:
    """Run the decoder half of the UNet, consuming `skips` (from the down path) in reverse.

    Each level concatenates a skip along channels before every resblock, so
    `n_res + 1` skips are used per level. Raises ValueError if the skip list is
    too short or has leftovers.
    """
    arithmetic = unet_arithmetic()
    n_levels = len(channels)
    reversed_res = list(reversed(res_blocks))
    reversed_attn = list(reversed(attn_levels))
    reversed_heads = list(reversed(heads))
    remaining = skips
    for index in range(n_levels):
        n_res = reversed_res[index] + 1
        if len(remaining) < n_res:
            raise ValueError(
                f"UNet skip list too short: need {n_res} for "
                f"up_blocks.{index}, have {len(remaining)}"
            )
        res_samples = remaining[-n_res:]
        remaining = remaining[:-n_res]
        has_attn = reversed_attn[index]
        n_heads = reversed_heads[index]
        for res_index in range(n_res):
            skip = res_samples[-1]
            res_samples = res_samples[:-1]
            hidden = concat_channels_ncdhw(hidden, skip, mx)
            hidden = residual_block_ncdhw(
                hidden,
                emb,
                _prefixed(weights, f"up_blocks.{index}.resnets.{res_index}"),
                num_groups=groups,
                eps=eps,
                mx=mx,
                arithmetic=arithmetic,
            )
            if has_attn:
                hidden = spatial_attention_ncdhw(
                    hidden,
                    _prefixed(weights, f"up_blocks.{index}.attentions.{res_index}"),
                    num_groups=groups,
                    eps=eps,
                    num_head_channels=n_heads,
                    include_fc=include_fc,
                    mx=mx,
                    arithmetic=arithmetic,
                )
        if index != n_levels - 1:
            if resblock_updown:
                hidden = residual_block_ncdhw(
                    hidden,
                    emb,
                    _prefixed(weights, f"up_blocks.{index}.upsampler"),
                    num_groups=groups,
                    eps=eps,
                    mx=mx,
                    up=True,
                    arithmetic=arithmetic,
                )
            else:
                hidden = upsample_conv_ncdhw(
                    hidden,
                    weights,
                    f"up_blocks.{index}.upsampler",
                    mx=mx,
                    arithmetic=arithmetic,
                )
    if remaining:
        raise ValueError(f"UNet skip list leftover after up blocks: {len(remaining)}")
    return hidden


def _refuse_unsupported(constructor: RflowConstructor) -> None:
    """Refuse constructors with no MLX destination: CUDA flash attention, the
    mask-generation cross-attention conditioning path, and combined QKV.
    ``resblock_updown=False`` (``ddpm-ct``) and the ``ddpm-ct`` top/bottom
    region-index inputs are both ported; see ``denoise_unet_ncdhw``.
    """

    if int(constructor.get("spatial_dims", 3)) != 3:
        raise ValueError("MAISI UNet dest is 3D NCDHW")
    if bool(constructor.get("use_flash_attention")):
        raise ValueError("CUDA flash attention is not an MLX destination")
    if bool(constructor.get("with_conditioning")):
        raise ValueError("MAISI UNet dest has with_conditioning=false")
    if bool(constructor.get("use_combined_linear")):
        raise ValueError("MAISI UNet dest uses separate QKV projections")
