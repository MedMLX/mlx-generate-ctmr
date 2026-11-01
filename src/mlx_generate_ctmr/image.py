"""Image-only NV-Generate rflow inference with modality classifier-free guidance."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_generate_ctmr._arrays import Float64Array, RealArray
    from mlx_generate_ctmr._types import (
        Array,
        ArrayFactory,
        Dtype,
        HostArray,
        IntArray,
        Mlx,
        RflowConstructor,
        VaeConstructor,
    )

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from mlx_generate_ctmr.arithmetic_mode import vae_arithmetic
from mlx_generate_ctmr.condition import SPACING_SCALE, modality_class_labels
from mlx_generate_ctmr.configurations import Configuration, ModelConfiguration, get_configuration
from mlx_generate_ctmr.decode import Z_SCALE, sliding_window_decode_ncdhw
from mlx_generate_ctmr.from_mask import require_scale_factor
from mlx_generate_ctmr.rflow import init_latent_ncdhw, sample_rflow, set_timesteps_rflow
from mlx_generate_ctmr.unet import denoise_unet_ncdhw

IMAGE_ROI = (80, 80, 80)
IMAGE_OVERLAP = 0.4


@dataclass(frozen=True, slots=True)
class ImageArrays:
    """In-memory image-only outputs.

    ``latent`` is the sampled latent before scale-factor division and ``decoded`` the
    VAE output ``(1, 1, D, H, W)`` in the decoder's normalized range. ``image`` is the
    int16 volume of the same shape (CT in HU, MR in arbitrary units; see
    ``output_intensities``). ``affine`` is the float64 4x4 diagonal matrix built
    from ``spacing_mm`` (no rotation or origin). ``stage_timings_s`` is seconds.
    """

    latent: Array
    decoded: Array
    image: IntArray
    affine: Float64Array
    stage_timings_s: Mapping[str, float]


def output_intensities(decoded: HostArray, *, configuration: str = "ct") -> IntArray:
    """Convert decoded VAE output to the int16 image the upstream image-only path writes.

    CT: ``clip(x * 2000 - 1000, -1000, 1000)`` HU. MR and MR-Brain:
    ``max(x * 1000, 0)`` in arbitrary units, with no upper clip, so extreme values
    can exceed the int16 range just as they do upstream. The float result is
    truncated to int16.

    Raises:
        ValueError: If ``decoded`` contains NaN or infinity.
    """
    config = get_configuration(configuration)
    data = np.asarray(decoded, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("decoded image must be finite")
    if config.name is Configuration.CT:
        scaled = np.clip(data * 2000.0 - 1000.0, -1000.0, 1000.0)
    else:
        scaled = np.clip(data * 1000.0, 0.0, None)
    return scaled.astype(np.int16)


def image_spacing_tensor[Tensor, Dtype](
    spacing_mm: Sequence[float] | RealArray, *, mx: ArrayFactory[Tensor, Dtype]
) -> Tensor:
    """Return ``spacing_mm * 100`` rounded through FP16, as a ``(1, 3)`` float32 array.

    The upstream image-only path casts the scaled spacing to half precision before
    the network sees it; this reproduces that rounding.

    Raises:
        ValueError: If there are not three positive, finite values whose scaled
            values are at most the largest finite FP16 value.

    Values small enough to underflow may round to zero in FP16.
    """
    if len(spacing_mm) != 3:
        raise ValueError("spacing_mm must have three values")
    scaled = np.asarray([spacing_mm], dtype=np.float64) * SPACING_SCALE
    if (
        not np.isfinite(scaled).all()
        or np.any(scaled <= 0)
        or np.any(scaled > np.finfo(np.float16).max)
    ):
        raise ValueError("scaled spacing must be positive, finite and representable in FP16")
    # FP64 -> FP16 directly, as in torch.from_numpy(...).half(); rounding first
    # through FP32 changes values immediately beside a half-precision midpoint.
    return mx.array(scaled.astype(np.float16).astype(np.float32), dtype=mx.float32)


def guided_velocity(
    state: Array,
    timestep: float,
    *,
    class_labels: Array,
    spacing: Array,
    weights: Mapping[str, Array],
    constructor: RflowConstructor,
    mx: Mlx,
    guidance_scale: float,
) -> Array:
    """Velocity for one sampler step, with modality classifier-free guidance.

    With ``guidance_scale == 0`` this is a single conditional pass. Otherwise the
    batch is doubled with the unconditional copy labelled class 0 (spacing and time
    are identical for both), and the result is
    ``unconditional + guidance_scale * (conditional - unconditional)``.
    """
    t = mx.full((int(state.shape[0]),), float(timestep), dtype=mx.float32)
    if guidance_scale == 0.0:
        return denoise_unet_ncdhw(
            state,
            t,
            class_labels,
            spacing,
            weights,
            constructor=constructor,
            mx=mx,
        )[1]
    batch = int(state.shape[0])
    classes = mx.concatenate([class_labels, mx.zeros_like(class_labels)])
    _, velocity = denoise_unet_ncdhw(
        mx.concatenate([state, state]),
        mx.concatenate([t, t]),
        classes,
        mx.concatenate([spacing, spacing]),
        weights,
        constructor=constructor,
        mx=mx,
    )
    conditional, unconditional = velocity[:batch], velocity[batch:]
    return unconditional + guidance_scale * (conditional - unconditional)


def _prepare_latent(
    output_size: Sequence[int],
    *,
    constructor: RflowConstructor,
    mx: Mlx,
    seed: int | None,
    latent: Array | None,
) -> Array:
    """Validate the requested output size and prepare initial diffusion noise."""
    values = tuple(output_size)
    divisor = Z_SCALE * 2 ** (len(constructor["num_channels"]) - 1)
    if len(values) != 3 or any(
        isinstance(v, bool) or not isinstance(cast(object, v), int) or v <= 0 or v % divisor
        for v in values
    ):
        raise ValueError(f"output_size must be three positive multiples of {divisor}")
    shape = (1, int(constructor["in_channels"]), *(v // Z_SCALE for v in values))
    if latent is not None:
        if tuple(latent.shape) != shape:
            raise ValueError(f"image-only latent shape {tuple(latent.shape)} != {shape}")
        return latent
    if isinstance(cast(object, seed), bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer when latent is omitted")
    return init_latent_ncdhw(shape, seed=seed, mx=mx)


def sample_image_ncdhw(
    output_size: Sequence[int],
    spacing_mm: Sequence[float] | RealArray,
    *,
    unet_weights: Mapping[str, Array],
    vae_weights: Mapping[str, Array],
    unet_constructor: RflowConstructor,
    vae_constructor: VaeConstructor,
    mx: Mlx,
    scale_factor: float,
    configuration: str = "ct",
    modality: str | int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    latent: Array | None = None,
    num_inference_steps: int = 30,
    roi: Sequence[int] = IMAGE_ROI,
    overlap: float = IMAGE_OVERLAP,
) -> ImageArrays:
    """Sample an image from noise (no ControlNet, mask or CT-specific QC).

    ``output_size`` is the output volume shape in voxels (array axes D, H, W); each
    value must be a positive multiple of ``4 * 2**(len(num_channels) - 1)`` from
    ``unet_constructor`` (32 for the production constructor). ``spacing_mm`` is the voxel size in mm
    along the same axes and is fed to the network as ``spacing * 100`` rounded
    through FP16. The latent is ``(1, in_channels, D/4, H/4, W/4)``. Provide a
    non-negative ``seed`` (seeds MLX's global RNG) or a ready ``latent``.
    ``guidance_scale=None`` uses the configuration default; ``modality=None`` the
    configuration's default modality. ``scale_factor`` is the UNet checkpoint's
    latent scale. ``roi``/``overlap`` set the VAE decode window in latent voxels.

    Raises:
        ValueError: If a size, seed, latent, scale or guidance value is invalid.
        InvalidInputError: If the configuration or modality is unsupported.
    """
    config = get_configuration(configuration)
    factor = require_scale_factor(scale_factor)
    guidance = guidance_scale_value(config, guidance_scale)
    hidden = _prepare_latent(
        output_size, constructor=unet_constructor, mx=mx, seed=seed, latent=latent
    )
    classes = modality_class_labels(mx=mx, configuration=configuration, modality=modality)
    spacing = image_spacing_tensor(spacing_mm, mx=cast("ArrayFactory[Array, Dtype]", mx))
    timesteps, _ = set_timesteps_rflow(
        num_inference_steps=num_inference_steps,
        input_img_size_numel=math.prod(hidden.shape[2:]),
    )

    def velocity(state: Array, timestep: float) -> Array:
        """Compute guided image diffusion velocity for one timestep."""
        return guided_velocity(
            state,
            timestep,
            class_labels=classes,
            spacing=spacing,
            weights=unet_weights,
            constructor=unet_constructor,
            mx=mx,
            guidance_scale=guidance,
        )

    started = time.perf_counter()
    final = sample_rflow(hidden, velocity, timesteps, mx=mx)
    rflow_s = time.perf_counter() - started
    decode_started = time.perf_counter()
    decoded = sliding_window_decode_ncdhw(
        final,
        vae_weights,
        constructor=vae_constructor,
        mx=mx,
        scale_factor=factor,
        roi=roi,
        overlap=overlap,
        arithmetic=vae_arithmetic(),
    )
    mx.eval(decoded)
    image = output_intensities(np.asarray(decoded, dtype=np.float32), configuration=configuration)
    affine = np.diag((*map(float, spacing_mm), 1.0))
    return ImageArrays(
        latent=final,
        decoded=decoded,
        image=image,
        affine=affine,
        stage_timings_s={"rflow_unet": rflow_s, "vae_decode": time.perf_counter() - decode_started},
    )


def guidance_scale_value(config: ModelConfiguration, value: float | None) -> float:
    """Return ``value`` if given, else the configuration's default guidance scale.

    Raises:
        ValueError: If ``value`` is not a finite, non-negative number.
    """
    if value is None:
        return config.guidance_scale
    if (
        isinstance(cast(object, value), bool)
        or not isinstance(cast(object, value), int | float)
        or not math.isfinite(value)
    ):
        raise ValueError("guidance_scale must be finite and non-negative")
    if value < 0:
        raise ValueError("guidance_scale must be finite and non-negative")
    return float(value)
