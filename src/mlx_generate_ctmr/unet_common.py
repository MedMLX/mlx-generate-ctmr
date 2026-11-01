"""Shared MAISI UNet helper functions. Importing this module does not import MLX."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, MlpArithmetic, Mlx, ResidualArithmetic

from medmlx_core import ops

from mlx_generate_ctmr.embed import silu_mlp


def apply_silu_mlp(
    array: Array,
    weights: Mapping[str, Array],
    prefix: str,
    mx: Mlx,
    *,
    arithmetic: MlpArithmetic = ops,
) -> Array:
    """Apply the common ``Linear -> SiLU -> Linear`` weight prefix."""

    return silu_mlp(
        array,
        required_weight(weights, f"{prefix}.0.weight"),
        required_weight(weights, f"{prefix}.0.bias"),
        required_weight(weights, f"{prefix}.2.weight"),
        required_weight(weights, f"{prefix}.2.bias"),
        mx,
        arithmetic=arithmetic,
    )


def downsample_conv_ncdhw(
    x: Array,
    weights: Mapping[str, Array],
    prefix: str,
    *,
    mx: Mlx,
    arithmetic: ResidualArithmetic = ops,
) -> Array:
    """``DiffusionUnetDownsample``: a plain stride-2 conv, used when ``resblock_updown`` is false.

    Unlike the resblock-based downsampler, this path does not touch the
    timestep embedding (MONAI's ``forward`` drops ``emb`` for this module).
    """

    return arithmetic.conv3d_ncdhw(
        x,
        required_weight(weights, f"{prefix}.op.conv.weight"),
        required_weight(weights, f"{prefix}.op.conv.bias"),
        padding=1,
        stride=2,
        mx=mx,
    )


def upsample_conv_ncdhw(
    x: Array,
    weights: Mapping[str, Array],
    prefix: str,
    *,
    mx: Mlx,
    arithmetic: ResidualArithmetic = ops,
) -> Array:
    """``WrappedUpsample``: nearest 2x upsample then a 3x3 conv, used when
    ``resblock_updown`` is false. Like the downsample path, ``emb`` is unused.
    """

    hidden = arithmetic.upsample_nearest_ncdhw(x, mx=mx)
    return arithmetic.conv3d_ncdhw(
        hidden,
        required_weight(weights, f"{prefix}.postconv.conv.weight"),
        required_weight(weights, f"{prefix}.postconv.conv.bias"),
        padding=1,
        mx=mx,
    )


def prefixed_weights(weights: Mapping[str, Array], prefix: str) -> dict[str, Array]:
    """Return a new dict with ``prefix.`` stripped; array values are shared."""

    start = prefix + "."
    return {key[len(start) :]: value for key, value in weights.items() if key.startswith(start)}


def required_weight(weights: Mapping[str, Array], key: str) -> Array:
    """Fetch one required tensor or fail fast with its full key."""

    if key not in weights:
        raise ValueError(f"UNet weights missing: {key}")
    return weights[key]


def int_sequence(values: object, *, name: str) -> tuple[int, ...]:
    """Validate a plain integer sequence."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of integers")
    result: list[int] = []
    for value in cast(Sequence[object], values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must contain only integers")
        result.append(value)
    return tuple(result)


def bool_sequence(values: object, *, name: str) -> tuple[bool, ...]:
    """Validate a plain boolean sequence."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of booleans")
    result: list[bool] = []
    for value in cast(Sequence[object], values):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must contain only booleans")
        result.append(value)
    return tuple(result)


def per_level(value: object, n_levels: int, name: str) -> tuple[int, ...]:
    """Expand one int or validate one per-level sequence."""

    if isinstance(value, int) and not isinstance(value, bool):
        return (int(value),) * n_levels
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        out = int_sequence(cast(Sequence[object], value), name=name)
        if len(out) != n_levels:
            raise ValueError(f"{name} length {len(out)} != num_channels {n_levels}")
        return out
    raise ValueError(f"{name} must be int or sequence")
