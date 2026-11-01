"""Sliding-window VAE decode for from-mask CT. No MLX import at module load."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, IndexedUpdates, Mlx, VaeArithmetic, VaeConstructor

import math
from collections.abc import Callable, Mapping, Sequence
from itertools import product

from medmlx_core import ops
from medmlx_core.layout import require_ncdhw

from mlx_generate_ctmr.maisi_mlx_weights import MASK_AUTOENCODER_SCALE_FACTOR
from mlx_generate_ctmr.vae import decode_vae_ncdhw

# Window sizes are in latent voxels (D, H, W); the decoder output is Z_SCALE times
# larger along each axis.
DEFAULT_ROI_SIZE: tuple[int, int, int] = (48, 48, 64)
DEFAULT_OVERLAP = 0.6666
DEFAULT_SIGMA_SCALE = 0.125
DEFAULT_MODE = "gaussian"
Z_SCALE = 4
_GAUSSIAN_FLOOR = 1e-3


def scan_interval(
    image_size: Sequence[int],
    roi: Sequence[int],
    overlap: float,
) -> tuple[int, int, int]:
    """``int(roi * (1-overlap))``, or 1 if 0; ``roi`` when it equals that axis."""

    _check_roi(roi)
    _check_overlap(overlap)
    spatial = _spatial3(image_size, name="image_size")
    interval: list[int] = []
    for size, window in zip(spatial, _check_roi(roi), strict=True):
        if window == size:
            interval.append(window)
            continue
        step = int(window * (1.0 - float(overlap)))
        interval.append(step if step > 0 else 1)
    return interval[0], interval[1], interval[2]


def dense_patch_slices(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    interval: Sequence[int],
) -> tuple[tuple[slice, slice, slice], ...]:
    """Window slices; last start is pulled back so the patch stays inside."""

    spatial = _spatial3(image_size, name="image_size")
    patch = _check_roi(patch_size)
    steps = _spatial3(interval, name="interval")
    starts: list[list[int]] = []
    for size, window, step in zip(spatial, patch, steps, strict=True):
        if step < 1:
            raise ValueError(f"interval must be positive; got {tuple(steps)}")
        count = int(math.ceil(float(size) / float(step)))
        last = next(
            (index for index in range(count) if index * step + window >= size),
            None,
        )
        n_starts = 1 if last is None else last + 1
        axis_starts: list[int] = []
        for index in range(n_starts):
            start = index * step
            start -= max(start + window - size, 0)
            axis_starts.append(start)
        starts.append(axis_starts)
    windows: list[tuple[slice, slice, slice]] = []
    for depth, height, width in product(*starts):
        windows.append(
            (
                slice(depth, depth + patch[0]),
                slice(height, height + patch[1]),
                slice(width, width + patch[2]),
            )
        )
    return tuple(windows)


def gaussian_importance_ncdhw(
    roi: Sequence[int],
    *,
    mx: Mlx,
    sigma_scale: float = DEFAULT_SIGMA_SCALE,
) -> Array:
    """1-D Gaussians, outer product, floor ``1e-3``. Shape ``[1, 1, D, H, W]``."""

    windows = _check_roi(roi)
    if float(sigma_scale) <= 0.0:
        raise ValueError(f"sigma_scale must be > 0; got {sigma_scale}")
    axes: list[Array] = []
    for length in windows:
        half = (length - 1) / 2.0
        x = mx.arange(-half, half + 1.0, dtype=mx.float32)
        sigma = float(sigma_scale) * float(length)
        weight = mx.exp((x * x) / (-2.0 * sigma * sigma))
        axes.append(weight)
    depth, height, width = axes
    importance = (
        mx.reshape(depth, (1, 1, windows[0], 1, 1))
        * mx.reshape(height, (1, 1, 1, windows[1], 1))
        * mx.reshape(width, (1, 1, 1, 1, windows[2]))
    )
    return mx.maximum(importance, mx.array(_GAUSSIAN_FLOOR, dtype=mx.float32))


def sliding_window_ncdhw(
    latent: Array,
    predict: Callable[[Array], Array],
    *,
    mx: Mlx,
    roi: Sequence[int] = DEFAULT_ROI_SIZE,
    overlap: float = DEFAULT_OVERLAP,
    sigma_scale: float = DEFAULT_SIGMA_SCALE,
    mode: str = DEFAULT_MODE,
    z_scale: int = Z_SCALE,
) -> Array:
    """Apply ``predict`` to overlapping latent windows and blend with a Gaussian weight.

    ``latent`` is ``(N, C, D, H, W)``; ``roi`` is the window size in latent voxels.
    ``predict`` maps one latent window to an output window ``z_scale`` times larger
    along each spatial axis. A latent smaller than ``roi`` is zero-padded (centered)
    and the result cropped back. Returns ``(N, C_out, D*z_scale, H*z_scale,
    W*z_scale)`` float32, already evaluated.

    Raises:
        ValueError: If ``mode`` is not ``"gaussian"``, ``z_scale`` is not 4, the
            window or overlap is invalid, or ``predict`` returns the wrong size.
    """

    require_ncdhw(latent, name="sliding-window latent")
    if mode != DEFAULT_MODE:
        raise ValueError(f"from-mask dest mode is {DEFAULT_MODE!r}; got {mode!r}")
    if int(z_scale) != Z_SCALE:
        raise ValueError(f"image VAE dest z_scale is {Z_SCALE}; got {z_scale}")
    roi_size = _check_roi(roi)
    _check_overlap(overlap)
    padded, pads = _pad_latent_to_roi(latent, roi_size, mx)
    spatial = (int(padded.shape[2]), int(padded.shape[3]), int(padded.shape[4]))
    interval = scan_interval(spatial, roi_size, overlap)
    windows = dense_patch_slices(spatial, roi_size, interval)
    weights = gaussian_importance_ncdhw(roi_size, mx=mx, sigma_scale=sigma_scale)
    weights = _nearest_resize_ncdhw(weights, z_scale, mx)
    output: Array | None = None
    weight_sum: Array | None = None
    for depth, height, width in windows:
        pred = predict(padded[:, :, depth, height, width])
        require_ncdhw(pred, name="sliding-window prediction")
        _require_output_scale(pred, roi_size, z_scale)
        out_depth = slice(depth.start * z_scale, depth.stop * z_scale)
        out_height = slice(height.start * z_scale, height.stop * z_scale)
        out_width = slice(width.start * z_scale, width.stop * z_scale)
        if output is None:
            out_spatial = tuple(axis * z_scale for axis in spatial)
            output = mx.zeros(
                (int(pred.shape[0]), int(pred.shape[1]), *out_spatial),
                dtype=mx.float32,
            )
            weight_sum = mx.zeros((1, 1, *out_spatial), dtype=mx.float32)
        if cast(object, output) is None or weight_sum is None:
            raise RuntimeError("sliding-window accumulator initialization failed")
        output = cast("IndexedUpdates", output.at)[:, :, out_depth, out_height, out_width].add(
            pred * weights
        )
        weight_sum = cast("IndexedUpdates", weight_sum.at)[
            :, :, out_depth, out_height, out_width
        ].add(weights)
        # Evaluate per window so the lazy graph does not keep every window's decoder
        # activations alive at once; this is what bounds peak memory when tiling.
        mx.eval(output, weight_sum)
    if output is None or weight_sum is None:
        raise ValueError("sliding window produced no patches")
    stitched = output / weight_sum
    stitched = _crop_decoded_pads(stitched, pads, z_scale)
    mx.eval(stitched)
    return stitched


def sliding_window_decode_ncdhw(
    latent: Array,
    weights: Mapping[str, Array],
    *,
    constructor: VaeConstructor,
    mx: Mlx,
    roi: Sequence[int] = DEFAULT_ROI_SIZE,
    overlap: float = DEFAULT_OVERLAP,
    sigma_scale: float = DEFAULT_SIGMA_SCALE,
    mode: str = DEFAULT_MODE,
    scale_factor: float | None = None,
    arithmetic: VaeArithmetic = ops,
) -> Array:
    """Decode latents with the VAE, tiling with a sliding window when they are large.

    If ``scale_factor`` is given the latent is divided by it first; with ``None``
    the latent is assumed to be unscaled already. There is no default scale factor:
    it must come from the UNet checkpoint. A latent no larger than ``roi`` (by
    voxel count) is decoded in one pass; otherwise windows are clamped to the latent
    size and blended by ``sliding_window_ncdhw``. Returns an evaluated decoded
    array. The production constructor yields ``(N, C_out, 4D, 4H, 4W)``; tiled
    decoding requires this 4x scale, while a direct decode uses the supplied
    constructor's expansion without checking the output scale.

    The direct one-pass branch validates ``roi`` but does not inspect ``overlap``,
    ``sigma_scale`` or ``mode``; those options are validated only when tiling.

    Raises:
        ValueError: If ``scale_factor`` is zero or non-finite, or the window
            arguments are invalid.
    """

    require_ncdhw(latent, name="vae latent")
    roi_size = _check_roi(roi)
    scaled = _maybe_scale_latent(latent, scale_factor, mx)

    def _decode_window(window: Array) -> Array:
        """Decode one latent window with the selected VAE settings."""
        _post_quant, decoded = decode_vae_ncdhw(
            window, weights, constructor=constructor, mx=mx, arithmetic=arithmetic
        )
        return decoded

    spatial = (int(scaled.shape[2]), int(scaled.shape[3]), int(scaled.shape[4]))
    if _prod(spatial) <= _prod(roi_size):
        decoded = _decode_window(scaled)
        mx.eval(decoded)
        return decoded
    adjusted = tuple(min(window, size) for window, size in zip(roi_size, spatial, strict=True))
    return sliding_window_ncdhw(
        scaled,
        _decode_window,
        mx=mx,
        roi=adjusted,
        overlap=overlap,
        sigma_scale=sigma_scale,
        mode=mode,
        z_scale=Z_SCALE,
    )


def to_hu_ncdhw(image: Array, label: Array, mx: Mlx) -> Array:
    """Map decoded intensities to float32 Hounsfield units.

    ``image`` is ``(N, 1, D, H, W)`` in the decoder's normalized range; it is
    clipped to [0, 1] and mapped linearly to [-1000, 1000] HU. Voxels where
    ``label == 0`` (background) are set to -1000 HU.
    """

    require_ncdhw(image, name="hu image")
    clipped = mx.clip(image, 0.0, 1.0)
    hu = clipped * 2000.0 - 1000.0
    return mx.where(label == 0, mx.array(-1000.0, dtype=mx.float32), hu)


def _maybe_scale_latent(latent: Array, scale_factor: float | None, mx: Mlx) -> Array:
    # If the caller omits the scale, keep the latent as-is. Do not invent the
    # mask-AE scale factor 1.0055984258651733 here.
    """Apply the optional inverse latent scale before decoding."""
    if scale_factor is None:
        return latent
    value = float(scale_factor)
    if not math.isfinite(value) or value == 0.0:
        raise ValueError(
            "scale_factor must be a finite non-zero caller value; "
            f"refuse to invent {MASK_AUTOENCODER_SCALE_FACTOR}; got {scale_factor!r}"
        )
    return latent / mx.array(value, dtype=mx.float32)


def _pad_latent_to_roi(
    latent: Array, roi: tuple[int, int, int], mx: Mlx
) -> tuple[Array, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """Zero-pad the spatial axes of an NCDHW latent up to the ROI window.

    Axes already at least as large as the window are untouched. The extra voxels
    are split with the smaller half in front. Returns the latent and the
    (before, after) latent-voxel pads for `_crop_decoded_pads`.
    """
    pads: list[tuple[int, int]] = []
    for axis, window in enumerate(roi):
        size = int(latent.shape[axis + 2])
        extra = max(window - size, 0)
        before = extra // 2
        pads.append((before, extra - before))
    pad_d, pad_h, pad_w = pads[0], pads[1], pads[2]
    if any(before or after for before, after in (pad_d, pad_h, pad_w)):
        latent = mx.pad(latent, [(0, 0), (0, 0), pad_d, pad_h, pad_w])
    return latent, (pad_d, pad_h, pad_w)


def _crop_decoded_pads(
    decoded: Array,
    pads: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    z_scale: int,
) -> Array:
    """Crop the pads added by `_pad_latent_to_roi` from an NCDHW decode.

    `z_scale` is the latent-to-voxel upsampling factor.
    """
    slices: list[slice] = []
    spatial = tuple(int(size) for size in decoded.shape[2:])
    for (before, after), size in zip(pads, spatial, strict=True):
        start = before * z_scale
        stop = int(size) - after * z_scale
        slices.append(slice(start, stop))
    if all(
        item.start == 0 and item.stop == size for item, size in zip(slices, spatial, strict=True)
    ):
        return decoded
    return decoded[:, :, slices[0], slices[1], slices[2]]


def _nearest_resize_ncdhw(array: Array, scale: int, mx: Mlx) -> Array:
    """Resize NCDHW spatial axes by nearest-neighbor indexing."""
    output = array
    for axis in (2, 3, 4):
        output = mx.repeat(output, scale, axis=axis)
    return output


def _require_output_scale(pred: Array, roi: tuple[int, int, int], z_scale: int) -> None:
    """Validate that a decoded window matches the configured spatial scale."""
    expected = tuple(window * z_scale for window in roi)
    got = (int(pred.shape[2]), int(pred.shape[3]), int(pred.shape[4]))
    if got != expected:
        raise ValueError(f"decoder output spatial {got} is not {z_scale}x latent ROI {roi}")


def _check_roi(roi: Sequence[int]) -> tuple[int, int, int]:
    """Validate and return a three-axis decoder ROI."""
    if len(roi) != 3:
        raise ValueError(f"roi must be three positive ints; got {tuple(roi)}")
    windows = (int(roi[0]), int(roi[1]), int(roi[2]))
    if any(window < 1 for window in windows):
        raise ValueError(f"roi must be three positive ints; got {windows}")
    return windows


def _check_overlap(overlap: float) -> None:
    """Validate the decoder window overlap fraction."""
    value = float(overlap)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"overlap must be >= 0 and < 1; got {overlap}")


def _spatial3(values: Sequence[int], *, name: str) -> tuple[int, int, int]:
    """Convert a three-element spatial sequence to an integer tuple."""
    if len(values) != 3:
        raise ValueError(f"{name} must have 3 spatial axes; got {tuple(values)}")
    spatial = (int(values[0]), int(values[1]), int(values[2]))
    if any(size < 1 for size in spatial):
        raise ValueError(f"{name} must be positive; got {spatial}")
    return spatial


def _prod(values: Sequence[int]) -> int:
    """Multiply the dimensions in a spatial sequence."""
    total = 1
    for value in values:
        total *= int(value)
    return total
