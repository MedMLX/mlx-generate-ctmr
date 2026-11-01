"""From-mask label packing and spacing/class tensors. No MLX import."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, ArrayValue, ClassFactory, Mlx

from collections.abc import Sequence

from medmlx_core.layout import require_ncdhw

from mlx_generate_ctmr.configurations import get_configuration

BITS = 8
SPACING_SCALE = 100.0
CT_CLASS_INDEX = get_configuration().default_modality


def binarize_labels_ncdhw(label: Array, *, mx: Mlx, bits: int = BITS) -> Array:
    """Pack integer labels into the 8-channel ControlNet condition.

    ``label`` is ``(N, 1, D, H, W)`` with integer ids (0..132 and body 200).
    Channel ``k`` of the ``(N, 8, D, H, W)`` float32 result holds bit ``k`` of the
    id (least significant bit first) as 0.0 or 1.0. ``bits`` must be 8. This
    helper casts to int32 but does not validate integer dtype or allowed label IDs;
    the from-mask sampler validates those before calling it.

    Raises:
        ValueError: If ``label`` is not single-channel NCDHW or ``bits`` is not 8.
    """

    require_ncdhw(label, name="maisi label")
    if int(label.shape[1]) != 1:
        raise ValueError(f"maisi label must have 1 channel; got {int(label.shape[1])}")
    if int(bits) != BITS:
        raise ValueError(f"from-mask dest packs {BITS} bits; got {bits}")
    ids = label.astype(mx.int32)
    powers = mx.left_shift(mx.ones((BITS,), dtype=mx.int32), mx.arange(BITS, dtype=mx.int32))
    expanded = mx.expand_dims(ids, axis=-1)
    mask = mx.reshape(powers, (1, 1, 1, 1, 1, BITS))
    packed = mx.squeeze((mx.bitwise_and(expanded, mask) != 0), axis=1)
    return mx.transpose(packed, (0, 4, 1, 2, 3)).astype(mx.float32)


def spacing_tensor_ncdhw(spacing_mm: Sequence[float], *, mx: Mlx) -> Array:
    """Return ``spacing_mm * 100`` as a ``(1, 3)`` float32 array.

    ``spacing_mm`` is the voxel size in mm, in the label array's axis order. No
    FP16 rounding is applied here; the image-only path in ``image.py`` rounds the
    scaled spacing through FP16 instead.

    Raises:
        ValueError: If there are not three finite, positive values.
    """

    if len(spacing_mm) != 3:
        raise ValueError(f"spacing_mm must have 3 values; got {tuple(spacing_mm)}")
    values = [float(axis) * SPACING_SCALE for axis in spacing_mm]
    if any(not _finite(value) or value <= 0.0 for value in values):
        raise ValueError(f"spacing_mm * {SPACING_SCALE} must be finite and > 0; got {values}")
    return mx.array(cast("ArrayValue", [values]), dtype=mx.float32)


def modality_class_labels[Tensor, Dtype](
    *,
    mx: ClassFactory[Tensor, Dtype],
    batch: int = 1,
    configuration: str = "ct",
    modality: str | int | None = None,
) -> Tensor:
    """Return the modality embedding index repeated ``batch`` times, shape ``(batch,)`` int32.

    ``modality`` is a name or index from the configuration's modality table;
    ``None`` selects the configuration default (1 for CT).

    Raises:
        ValueError: If ``batch`` is less than 1.
        InvalidInputError: If the modality is not supported by ``configuration``.
    """

    if int(batch) < 1:
        raise ValueError(f"batch must be >= 1; got {batch}")
    index = get_configuration(configuration).modality_index(modality)
    return mx.full((int(batch),), index, dtype=mx.int32)


def _finite(value: float) -> bool:
    """Return whether a scalar is neither NaN nor infinity."""
    return value == value and value not in (float("inf"), float("-inf"))
