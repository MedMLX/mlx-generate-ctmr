"""MLX port of the paired-mask conditioned ``DiffusionModelUNet``."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import (
        Array,
        ArrayConstructor,
        HostArray,
        MaskUnetConstructor,
        Mlx,
        ResidualArithmetic,
    )

from collections.abc import Mapping, Sequence

from medmlx_core import ops
from medmlx_core.layout import require_ncdhw
from medmlx_core.ops import concat_channels_ncdhw

from mlx_generate_ctmr import fast_fp32
from mlx_generate_ctmr.arithmetic_mode import active_arithmetic_mode
from mlx_generate_ctmr.attention import spatial_transformer_ncdhw
from mlx_generate_ctmr.embed import sinusoidal_timestep_embedding
from mlx_generate_ctmr.residual import residual_block_ncdhw
from mlx_generate_ctmr.unet_common import (
    apply_silu_mlp,
    bool_sequence,
    int_sequence,
    per_level,
    prefixed_weights,
    required_weight,
)

# One cross-attention token of 10 anatomy-size slots (five organs, five tumors); see
# ``mask_condition.MASK_ANATOMY_SIZE_SLOTS`` for the slot order.
MASK_CONDITION_DIM = 10

PRODUCTION_MASK_UNET_CONSTRUCTOR: MaskUnetConstructor = {
    "spatial_dims": 3,
    "in_channels": 4,
    "out_channels": 4,
    "num_res_blocks": 2,
    "channels": [64, 128, 256, 512],
    "attention_levels": [False, False, True, True],
    "norm_num_groups": 32,
    "norm_eps": 1e-6,
    "resblock_updown": False,
    "num_head_channels": [0, 0, 32, 32],
    "with_conditioning": True,
    "transformer_num_layers": 1,
    "cross_attention_dim": MASK_CONDITION_DIM,
    "num_class_embeds": None,
    "upcast_attention": True,
    "dropout_cattn": 0.0,
    "include_fc": True,
    "use_combined_linear": False,
    "use_flash_attention": False,
}


def anatomy_size_context(
    anatomy_size: HostArray | Array,
    *,
    mx: Mlx,
    cross_attention_dim: int = MASK_CONDITION_DIM,
) -> Array:
    """Convert anatomy context to float32 with trailing ``cross_attention_dim``.

    A vector becomes ``(1, 1, C)`` and a matrix ``(B, 1, C)``. A 3D input
    ``(B, tokens, C)`` is returned with its token dimension unchanged; ``C`` is
    10 for the production constructor. Values are not range-checked here.
    """

    context = cast("ArrayConstructor", mx.array)(anatomy_size, dtype=mx.float32)
    if int(context.ndim) == 1:
        if int(context.shape[0]) != cross_attention_dim:
            raise ValueError(
                "anatomy_size length "
                f"{int(context.shape[0])} != cross_attention_dim {cross_attention_dim}"
            )
        return mx.reshape(context, (1, 1, cross_attention_dim))
    if int(context.ndim) == 2:
        if int(context.shape[1]) != cross_attention_dim:
            raise ValueError(
                "anatomy_size last dim "
                f"{int(context.shape[1])} != cross_attention_dim {cross_attention_dim}"
            )
        return mx.reshape(context, (int(context.shape[0]), 1, cross_attention_dim))
    if int(context.ndim) == 3 and int(context.shape[-1]) == cross_attention_dim:
        return context
    raise ValueError(
        "anatomy_size must be [10], [B, 10], or [B, 1, 10]; "
        f"got shape={tuple(int(dim) for dim in context.shape)}"
    )


def denoise_mask_unet_ncdhw(
    x: Array,
    timesteps: Array,
    context: Array,
    weights: Mapping[str, Array],
    *,
    constructor: MaskUnetConstructor,
    mx: Mlx,
) -> tuple[Array, Array]:
    """Run the paired-mask UNet once; returns ``(t_emb, eps)``.

    ``x`` is the noisy mask latent ``(N, in_channels, D, H, W)``, ``timesteps`` is
    ``(N,)`` and ``context`` the anatomy-size condition (see
    ``anatomy_size_context``), with batch size 1 or N. ``eps`` is predicted noise
    with ``out_channels`` channels and the input spatial shape; the production
    constructor uses 4 for both channel counts. ``t_emb`` is the sinusoidal
    timestep embedding before its MLP.

    Raises:
        ValueError: If shapes do not match the constructor or it requests an
            unsupported feature.
    """

    require_ncdhw(x, name="mask UNet input")
    _refuse_unsupported(constructor)
    if int(x.shape[1]) != int(constructor["in_channels"]):
        raise ValueError(
            "mask UNet input channels "
            f"{int(x.shape[1])} != in_channels {constructor['in_channels']}"
        )
    if int(timesteps.ndim) != 1:
        raise ValueError("timesteps must be a rank-1 array")
    if int(timesteps.shape[0]) != int(x.shape[0]):
        raise ValueError(
            f"timesteps batch {int(timesteps.shape[0])} != input batch {int(x.shape[0])}"
        )
    cross_attention_dim = int(constructor["cross_attention_dim"])
    conditioning = anatomy_size_context(
        context,
        mx=mx,
        cross_attention_dim=cross_attention_dim,
    )
    if int(conditioning.shape[0]) not in {1, int(x.shape[0])}:
        raise ValueError(
            f"context batch {int(conditioning.shape[0])} must be 1 or input batch {int(x.shape[0])}"
        )

    channels = _channels(constructor)
    n_levels = len(channels)
    res_blocks = per_level(constructor["num_res_blocks"], n_levels, "num_res_blocks")
    attn_levels = bool_sequence(constructor["attention_levels"], name="attention_levels")
    if len(attn_levels) != n_levels:
        raise ValueError("attention_levels length must match channels")
    heads = per_level(constructor["num_head_channels"], n_levels, "num_head_channels")
    groups = int(constructor["norm_num_groups"])
    eps = float(constructor["norm_eps"])
    include_fc = bool(constructor.get("include_fc", True))

    t_emb = sinusoidal_timestep_embedding(timesteps.astype(mx.float32), channels[0], mx=mx)
    emb = apply_silu_mlp(t_emb, weights, "time_embed", mx)
    hidden = _arithmetic().conv3d_ncdhw(
        x,
        required_weight(weights, "conv_in.conv.weight"),
        required_weight(weights, "conv_in.conv.bias"),
        padding=1,
        mx=mx,
    )
    hidden, skips = _apply_down_blocks(
        hidden,
        emb,
        conditioning,
        weights,
        channels=channels,
        res_blocks=res_blocks,
        attn_levels=attn_levels,
        heads=heads,
        groups=groups,
        eps=eps,
        cross_attention_dim=cross_attention_dim,
        include_fc=include_fc,
        mx=mx,
    )
    hidden = _apply_mid_block(
        hidden,
        emb,
        conditioning,
        weights,
        num_head_channels=heads[-1],
        groups=groups,
        eps=eps,
        cross_attention_dim=cross_attention_dim,
        include_fc=include_fc,
        mx=mx,
    )
    hidden = _apply_up_blocks(
        hidden,
        emb,
        conditioning,
        skips,
        weights,
        channels=channels,
        res_blocks=res_blocks,
        attn_levels=attn_levels,
        heads=heads,
        groups=groups,
        eps=eps,
        cross_attention_dim=cross_attention_dim,
        include_fc=include_fc,
        mx=mx,
    )
    norm = (required_weight(weights, "out.0.weight"), required_weight(weights, "out.0.bias"))
    out_weight = required_weight(weights, "out.2.conv.weight")
    out_bias = required_weight(weights, "out.2.conv.bias")
    if _arithmetic() is fast_fp32:
        eps_pred = fast_fp32.norm_conv3d_ncdhw(
            hidden,
            out_weight,
            out_bias,
            norm=norm,
            num_groups=groups,
            eps=eps,
            apply_silu=True,
            mx=mx,
        )
        return t_emb, eps_pred
    hidden = ops.group_norm_ncdhw(hidden, *norm, num_groups=groups, eps=eps, mx=mx)
    hidden = ops.silu(hidden, mx)
    eps_pred = ops.conv3d_ncdhw(hidden, out_weight, out_bias, padding=1, mx=mx)
    return t_emb, eps_pred


def _arithmetic() -> ResidualArithmetic:
    """``fast_fp32`` under the default fast mode; the generic ``medmlx_core.ops`` under source.

    The source mode keeps this network on the generic ops it was ported with;
    unlike the image UNet it has no CUDA-ordered arithmetic.
    """
    return fast_fp32 if active_arithmetic_mode() == "fast" else ops


def _apply_down_blocks(
    hidden: Array,
    emb: Array,
    context: Array,
    weights: Mapping[str, Array],
    *,
    channels: Sequence[int],
    res_blocks: Sequence[int],
    attn_levels: Sequence[bool],
    heads: Sequence[int],
    groups: int,
    eps: float,
    cross_attention_dim: int,
    include_fc: bool,
    mx: Mlx,
) -> tuple[Array, list[Array]]:
    """Run the encoder half of the paired-mask UNet on NCDHW data.

    Returns the hidden state and the skip list (input, every resblock/attention
    output, every downsample output) consumed by `_apply_up_blocks`.
    """
    skips: list[Array] = [hidden]
    n_levels = len(channels)
    for index, (_out_channels, n_res, has_attn, n_heads) in enumerate(
        zip(channels, res_blocks, attn_levels, heads, strict=True)
    ):
        for res_index in range(n_res):
            hidden = residual_block_ncdhw(
                hidden,
                emb,
                prefixed_weights(weights, f"down_blocks.{index}.resnets.{res_index}"),
                num_groups=groups,
                eps=eps,
                mx=mx,
                arithmetic=_arithmetic(),
            )
            if has_attn:
                hidden = spatial_transformer_ncdhw(
                    hidden,
                    context,
                    prefixed_weights(weights, f"down_blocks.{index}.attentions.{res_index}"),
                    num_groups=groups,
                    eps=eps,
                    num_head_channels=n_heads,
                    cross_attention_dim=cross_attention_dim,
                    include_fc=include_fc,
                    mx=mx,
                )
            skips.append(hidden)
        if index != n_levels - 1:
            hidden = _downsample_conv_ncdhw(
                hidden,
                prefixed_weights(weights, f"down_blocks.{index}.downsampler"),
                mx=mx,
            )
            skips.append(hidden)
    return hidden, skips


def _apply_mid_block(
    hidden: Array,
    emb: Array,
    context: Array,
    weights: Mapping[str, Array],
    *,
    num_head_channels: int,
    groups: int,
    eps: float,
    cross_attention_dim: int,
    include_fc: bool,
    mx: Mlx,
) -> Array:
    """Resblock, cross-attention transformer, resblock at the lowest resolution."""
    hidden = residual_block_ncdhw(
        hidden,
        emb,
        prefixed_weights(weights, "middle_block.resnet_1"),
        num_groups=groups,
        eps=eps,
        mx=mx,
        arithmetic=_arithmetic(),
    )
    hidden = spatial_transformer_ncdhw(
        hidden,
        context,
        prefixed_weights(weights, "middle_block.attention"),
        num_groups=groups,
        eps=eps,
        num_head_channels=num_head_channels,
        cross_attention_dim=cross_attention_dim,
        include_fc=include_fc,
        mx=mx,
    )
    return residual_block_ncdhw(
        hidden,
        emb,
        prefixed_weights(weights, "middle_block.resnet_2"),
        num_groups=groups,
        eps=eps,
        mx=mx,
        arithmetic=_arithmetic(),
    )


def _apply_up_blocks(
    hidden: Array,
    emb: Array,
    context: Array,
    skips: list[Array],
    weights: Mapping[str, Array],
    *,
    channels: Sequence[int],
    res_blocks: Sequence[int],
    attn_levels: Sequence[bool],
    heads: Sequence[int],
    groups: int,
    eps: float,
    cross_attention_dim: int,
    include_fc: bool,
    mx: Mlx,
) -> Array:
    """Run the decoder half of the paired-mask UNet, consuming `skips` in reverse.

    Each level concatenates a skip along channels before every resblock, so
    `n_res + 1` skips are used per level. Raises ValueError if the skip list is
    too short or has leftovers.
    """
    n_levels = len(channels)
    reversed_res = list(reversed(res_blocks))
    reversed_attn = list(reversed(attn_levels))
    reversed_heads = list(reversed(heads))
    remaining = skips
    for index in range(n_levels):
        n_res = reversed_res[index] + 1
        if len(remaining) < n_res:
            raise ValueError(
                f"mask UNet skip list too short: need {n_res} for up_blocks.{index}, "
                f"have {len(remaining)}"
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
                prefixed_weights(weights, f"up_blocks.{index}.resnets.{res_index}"),
                num_groups=groups,
                eps=eps,
                mx=mx,
                arithmetic=_arithmetic(),
            )
            if has_attn:
                hidden = spatial_transformer_ncdhw(
                    hidden,
                    context,
                    prefixed_weights(weights, f"up_blocks.{index}.attentions.{res_index}"),
                    num_groups=groups,
                    eps=eps,
                    num_head_channels=n_heads,
                    cross_attention_dim=cross_attention_dim,
                    include_fc=include_fc,
                    mx=mx,
                )
        if index != n_levels - 1:
            hidden = _upsample_postconv_ncdhw(
                hidden,
                prefixed_weights(weights, f"up_blocks.{index}.upsampler"),
                mx=mx,
            )
    if remaining:
        raise ValueError(f"mask UNet skip list leftover after up blocks: {len(remaining)}")
    return hidden


def _downsample_conv_ncdhw(hidden: Array, weights: Mapping[str, Array], *, mx: Mlx) -> Array:
    """Apply the mask UNet downsampling convolution."""
    return _arithmetic().conv3d_ncdhw(
        hidden,
        required_weight(weights, "op.conv.weight"),
        required_weight(weights, "op.conv.bias"),
        padding=1,
        stride=2,
        mx=mx,
    )


def _upsample_postconv_ncdhw(hidden: Array, weights: Mapping[str, Array], *, mx: Mlx) -> Array:
    """Nearest-neighbour 2x upsample followed by the `postconv` convolution (NCDHW)."""
    weight = required_weight(weights, "postconv.conv.weight")
    bias = required_weight(weights, "postconv.conv.bias")
    if _arithmetic() is fast_fp32:
        # Folds the nearest upsampling into the weights instead of forming it.
        return fast_fp32.norm_conv3d_ncdhw(
            hidden,
            weight,
            bias,
            norm=None,
            num_groups=1,
            eps=0.0,
            apply_silu=False,
            upsample=True,
            mx=mx,
        )
    hidden = ops.upsample_nearest_ncdhw(hidden, mx=mx)
    return ops.conv3d_ncdhw(hidden, weight, bias, padding=1, mx=mx)


def _channels(constructor: MaskUnetConstructor) -> tuple[int, ...]:
    """Resolve the configured channel widths for the mask UNet."""
    raw = constructor.get("channels", constructor.get("num_channels"))
    if raw is None:
        raise ValueError("constructor must define channels")
    return int_sequence(raw, name="channels")


def _refuse_unsupported(constructor: MaskUnetConstructor) -> None:
    """Raise ValueError for constructor options other than the supported paired-mask UNet (3D,
    conditioned, separate QKV, no class/spacing embedding).
    """
    if int(constructor.get("spatial_dims", 3)) != 3:
        raise ValueError("paired-mask UNet dest is 3D NCDHW")
    if bool(constructor.get("use_flash_attention")):
        raise ValueError("CUDA flash attention is not an MLX destination")
    if not bool(constructor.get("with_conditioning")):
        raise ValueError("paired-mask UNet dest requires with_conditioning=true")
    if bool(constructor.get("use_combined_linear")):
        raise ValueError("paired-mask UNet dest uses separate QKV projections")
    if bool(constructor.get("resblock_updown")):
        raise ValueError("paired-mask UNet dest requires resblock_updown=false")
    if constructor.get("num_class_embeds") is not None:
        raise ValueError("paired-mask UNet dest has no class embedding")
    if "include_spacing_input" in constructor and bool(constructor.get("include_spacing_input")):
        raise ValueError("paired-mask UNet dest has no spacing embedding")
    if int(constructor.get("transformer_num_layers", 1)) != 1:
        raise ValueError("paired-mask UNet dest requires transformer_num_layers=1")
    if int(constructor.get("cross_attention_dim", MASK_CONDITION_DIM)) != MASK_CONDITION_DIM:
        raise ValueError(f"paired-mask UNet dest requires cross_attention_dim={MASK_CONDITION_DIM}")
