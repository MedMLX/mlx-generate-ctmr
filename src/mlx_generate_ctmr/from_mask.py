"""Compose ControlNet, UNet, rflow sampling and VAE decode for from-mask CT.

Volumes are NCDHW. For a canonical-RAS NIfTI label map the array axes
``(X, Y, Z)`` occupy the ``(D, H, W)`` positions. The latent has 4 channels and a
quarter of the label size along each spatial axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_generate_ctmr._arrays import Float64Array
    from mlx_generate_ctmr._types import (
        Array,
        ArrayConstructor,
        ControlnetConstructor,
        HostArray,
        Mlx,
        RflowConstructor,
        VaeConstructor,
    )

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from medmlx_core.layout import require_ncdhw
from medmlx_core.maisi import MAISI_LABELS
from medmlx_core.ops import as_fp32

from mlx_generate_ctmr.arithmetic_mode import vae_arithmetic
from mlx_generate_ctmr.condition import (
    binarize_labels_ncdhw,
    modality_class_labels,
    spacing_tensor_ncdhw,
)
from mlx_generate_ctmr.controlnet import (
    controlnet_residuals_ncdhw,
    encode_controlnet_condition_ncdhw,
)
from mlx_generate_ctmr.decode import (
    DEFAULT_OVERLAP,
    DEFAULT_ROI_SIZE,
    Z_SCALE,
    sliding_window_decode_ncdhw,
    to_hu_ncdhw,
)
from mlx_generate_ctmr.maisi_mlx_weights import MASK_AUTOENCODER_SCALE_FACTOR
from mlx_generate_ctmr.rflow import (
    init_latent_ncdhw,
    sample_rflow,
    set_timesteps_rflow,
)
from mlx_generate_ctmr.unet import denoise_unet_ncdhw

_VALID_LABELS = frozenset(item.value for item in MAISI_LABELS)
IDENTITY_AFFINE = np.eye(4, dtype=np.float64)
DEFAULT_STEPS = 30
SEAM_NOT_APPLICABLE = "not_applicable_below_roi"
SUPPORTED_PRODUCTION_SHAPES = ((256, 256, 128), (256, 256, 256))


@dataclass(frozen=True, slots=True)
class FromMaskArrays:
    """In-memory from-mask outputs. Not a NIfTI/run-record writer.

    ``latent`` is the sampled latent ``(N, 4, D/4, H/4, W/4)`` before division by
    the scale factor; ``decoded`` is the VAE output ``(N, 1, D, H, W)`` in the
    decoder's normalized intensity range; ``hu`` is float32 Hounsfield units, same
    shape, clipped to [-1000, 1000] with label 0 forced to -1000. ``affine`` is the
    float64 4x4 voxel-to-world matrix passed through (identity if none was given).
    ``seam_audit`` is ``"tiled"`` when the decode used more than one window, else
    ``"not_applicable_below_roi"``. ``stage_timings_s`` is wall-clock seconds.
    """

    latent: Array
    decoded: Array
    hu: Array
    affine: Float64Array
    scale_factor: float
    num_inference_steps: int
    seam_audit: str
    stage_timings_s: Mapping[str, float]


def require_supported_production_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    """Return ``shape`` as a 3-tuple if this port supports it, else raise ``ValueError``.

    Only 256x256x128 and 256x256x256 are accepted. Larger grids are refused before
    any allocation because the dense FP32 attention path has not been checked at
    those sizes.
    """

    spatial = tuple(int(value) for value in shape)
    if len(spatial) != 3:
        raise ValueError(f"MLX NV-Generate-CT mask shape must be 3D; got {spatial}")
    if spatial not in SUPPORTED_PRODUCTION_SHAPES:
        raise ValueError(
            f"MLX NV-Generate-CT from_mask shape {spatial} is not production-proven; "
            "supported shapes: (256, 256, 128) and (256, 256, 256). The "
            "current dense FP32 SDPA path blocks 512^3 and larger before allocation."
        )
    return spatial[0], spatial[1], spatial[2]


def sample_ct_from_mask_ncdhw(
    label: HostArray,
    spacing_mm: Sequence[float],
    *,
    unet_weights: Mapping[str, Array],
    controlnet_weights: Mapping[str, Array],
    vae_weights: Mapping[str, Array],
    unet_constructor: RflowConstructor,
    controlnet_constructor: ControlnetConstructor,
    vae_constructor: VaeConstructor,
    mx: Mlx,
    scale_factor: float,
    seed: int | None = None,
    latent: Array | None = None,
    num_inference_steps: int = DEFAULT_STEPS,
    guidance_scale: float = 0.0,
    affine: HostArray | None = None,
    roi: Sequence[int] = DEFAULT_ROI_SIZE,
    overlap: float = DEFAULT_OVERLAP,
) -> FromMaskArrays:
    """Generate a CT latent and volume from a MAISI label map with MLX.

    ``label`` is a host integer array ``(N, 1, D, H, W)`` of MAISI label ids (0..132
    plus body 200, which must be present) with each spatial size a multiple of 4.
    ``spacing_mm`` is the voxel size in mm along the same axes. ``scale_factor`` is
    the UNet checkpoint's latent scale (no default is supplied). Provide either
    ``seed`` (seeds MLX's global RNG) or a ready ``latent`` of shape
    ``(N, 4, D/4, H/4, W/4)``. ``num_inference_steps`` is the rflow step count;
    ``guidance_scale`` must be 0 (the CT path uses ControlNet, not guidance).
    ``roi``/``overlap`` set the VAE decode window in latent voxels. ``affine`` is
    carried through to the result unchanged. Label validation and affine handling
    use host arrays; neural inference uses MLX. Result MLX arrays are evaluated.

    Raises:
        ValueError: If ``label``, ``latent``, ``spacing_mm``, ``affine``,
            ``scale_factor`` or ``guidance_scale`` is invalid, or neither ``seed``
            nor ``latent`` is given.
    """

    if float(guidance_scale) != 0.0:
        raise ValueError("CFG>0 majority vote is not an MLX destination")
    factor = require_scale_factor(scale_factor)
    host_label = _require_label(label)
    label_mx = cast("ArrayConstructor", mx.array)(host_label)
    require_ncdhw(label_mx, name="maisi label")
    cond = binarize_labels_ncdhw(label_mx, mx=mx)
    spacing = spacing_tensor_ncdhw(spacing_mm, mx=mx)
    class_labels = modality_class_labels(mx=mx, batch=int(host_label.shape[0]))
    latent_shape = _latent_shape(host_label)
    hidden = latent
    if hidden is None:
        if seed is None:
            raise ValueError("seed is required when latent is omitted")
        hidden = init_latent_ncdhw(latent_shape, seed=int(seed), mx=mx)
    else:
        require_ncdhw(hidden, name="from-mask latent")
        if tuple(int(dim) for dim in hidden.shape) != latent_shape:
            raise ValueError(
                f"latent shape {tuple(int(dim) for dim in hidden.shape)} != {latent_shape}"
            )
    timesteps, _next = set_timesteps_rflow(
        num_inference_steps=int(num_inference_steps),
        input_img_size_numel=_prod(latent_shape[2:]),
    )
    inference_started = time.perf_counter()
    rflow_started = time.perf_counter()
    # The label map and ControlNet weights are fixed throughout this generation.
    # Materialize the embedding once so its convolution intermediates can be freed.
    conditioning_embedding = encode_controlnet_condition_ncdhw(
        cond, controlnet_weights, constructor=controlnet_constructor, mx=mx
    )
    mx.eval(conditioning_embedding)

    def _velocity(state: Array, timestep: float) -> Array:
        """Predict one CT diffusion velocity for the current timestep."""
        t = cast("ArrayConstructor", mx.array)([float(timestep)], dtype=mx.float32)
        down, mid = controlnet_residuals_ncdhw(
            state,
            t,
            class_labels,
            cond,
            controlnet_weights,
            constructor=controlnet_constructor,
            mx=mx,
            conditioning_embedding=conditioning_embedding,
        )
        _t_emb, velocity = denoise_unet_ncdhw(
            state,
            t,
            class_labels,
            spacing,
            unet_weights,
            constructor=unet_constructor,
            mx=mx,
            down_block_additional_residuals=down,
            mid_block_additional_residual=mid,
        )
        return velocity

    final = sample_rflow(hidden, _velocity, timesteps, mx=mx, guidance_scale=0.0)
    mx.eval(final)
    rflow_s = time.perf_counter() - rflow_started
    decode_started = time.perf_counter()
    decoded = sliding_window_decode_ncdhw(
        final,
        vae_weights,
        constructor=vae_constructor,
        mx=mx,
        roi=roi,
        overlap=overlap,
        scale_factor=factor,
        arithmetic=vae_arithmetic(),
    )
    mx.eval(decoded)
    decode_s = time.perf_counter() - decode_started
    hu_started = time.perf_counter()
    hu = to_hu_ncdhw(decoded, as_fp32(host_label, mx), mx)
    mx.eval(hu)
    hu_s = time.perf_counter() - hu_started
    spatial = tuple(int(dim) for dim in decoded.shape[2:])
    roi_image = tuple(int(axis) * Z_SCALE for axis in roi)
    seam = SEAM_NOT_APPLICABLE if _prod(spatial) <= _prod(roi_image) else "tiled"
    out_affine = IDENTITY_AFFINE.copy() if affine is None else np.asarray(affine, dtype=np.float64)
    if out_affine.shape != (4, 4):
        raise ValueError(f"affine must be 4x4; got {out_affine.shape}")
    return FromMaskArrays(
        latent=final,
        decoded=decoded,
        hu=hu,
        affine=out_affine,
        scale_factor=factor,
        num_inference_steps=int(num_inference_steps),
        seam_audit=seam,
        stage_timings_s={
            "rflow_controlnet_unet": rflow_s,
            "vae_decode": decode_s,
            "hu_postprocess": hu_s,
            "total_synchronized_inference": time.perf_counter() - inference_started,
        },
    )


def require_scale_factor(scale_factor: float) -> float:
    """Return ``scale_factor`` as a float, refusing zero or non-finite values.

    The latent scale comes from the UNet checkpoint; a fixed default is never used.
    """
    value = float(scale_factor)
    if not np.isfinite(value) or value == 0.0:
        raise ValueError(
            "scale_factor must be a finite non-zero caller value; "
            f"refuse to invent {MASK_AUTOENCODER_SCALE_FACTOR}; got {scale_factor!r}"
        )
    return value


def _require_label(label: HostArray) -> HostArray:
    """Validate and materialize a categorical CT conditioning mask."""
    host = np.asarray(label)
    if host.ndim != 5 or int(host.shape[1]) != 1:
        raise ValueError(f"label must be NCDHW with 1 channel; got {host.shape}")
    if not np.issubdtype(host.dtype, np.integer):
        raise ValueError(f"label must be integer; got {host.dtype}")
    unique = {int(value) for value in np.unique(host)}
    unknown = sorted(value for value in unique if value not in _VALID_LABELS)
    if unknown:
        raise ValueError(
            "MAISI mask contains labels outside 0..132 plus body label 200: "
            + ", ".join(str(value) for value in unknown[:10])
        )
    if 200 not in unique:
        raise ValueError("MAISI mask is missing body-envelope label 200; add it before generation")
    if any(int(size) % Z_SCALE != 0 for size in host.shape[2:]):
        spatial = tuple(int(size) for size in host.shape[2:])
        if any(size % Z_SCALE != 0 for size in spatial):
            raise ValueError(f"label spatial {spatial} must be multiples of {Z_SCALE}")
    return host


def _latent_shape(label: HostArray) -> tuple[int, int, int, int, int]:
    """Derive the CT diffusion latent grid from the requested spatial size."""
    return (
        int(label.shape[0]),
        4,
        int(label.shape[2]) // Z_SCALE,
        int(label.shape[3]) // Z_SCALE,
        int(label.shape[4]) // Z_SCALE,
    )


def _prod(values: Sequence[int]) -> int:
    """Multiply a sequence of dimensions."""
    total = 1
    for value in values:
        total *= int(value)
    return total


__all__ = [
    "DEFAULT_STEPS",
    "IDENTITY_AFFINE",
    "SUPPORTED_PRODUCTION_SHAPES",
    "FromMaskArrays",
    "require_supported_production_shape",
    "sample_ct_from_mask_ncdhw",
]
