"""Native MAISI MRI sampling, source controls, decode, and intensity export."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, Mlx, RflowConstructor, VaeConstructor

import math
from collections.abc import Mapping, Sequence

import numpy as np
from medmlx_core.errors import InvalidInputError
from medmlx_core.layout import require_ncdhw
from numpy.typing import NDArray

from mlx_generate_ctmr.arithmetic_mode import vae_arithmetic
from mlx_generate_ctmr.decode import sliding_window_decode_ncdhw
from mlx_generate_ctmr.maisi_mlx_from_mask import (
    PRODUCTION_UNET_CONSTRUCTOR,
    PRODUCTION_VAE_CONSTRUCTOR,
)
from mlx_generate_ctmr.rflow import set_timesteps_rflow, step_rflow
from mlx_generate_ctmr.unet import denoise_unet_ncdhw

DECODE_ROI = (80, 80, 80)
DECODE_OVERLAP = 0.4


def mr_timesteps(*, latent_numel: int, num_inference_steps: int = 30) -> NDArray[np.float32]:
    """Rflow timesteps for an MR latent with ``latent_numel`` voxels (D*H*W).

    MONAI stores the transformed schedule in float16 before the model casts it to
    FP32; ``set_timesteps_rflow`` reproduces that rounding.
    """
    timesteps, _ = set_timesteps_rflow(
        num_inference_steps=num_inference_steps,
        input_img_size_numel=latent_numel,
    )
    return timesteps


def mr_spacing_tensor_ncdhw(spacing_mm: Sequence[float], *, mx: Mlx) -> Array:
    """Return ``spacing_mm * 100`` rounded through FP16 as a ``(1, 3)`` float32 array.

    Mirrors the source's control ingest: scale on the host, round to half
    precision, then feed the network in FP32. Finite positive inputs can overflow
    the FP16 conversion for extreme spacings; the result is not revalidated here.

    Raises:
        InvalidInputError: If there are not three finite, positive values.
    """
    if len(spacing_mm) != 3 or any(
        not math.isfinite(float(value)) or float(value) <= 0 for value in spacing_mm
    ):
        raise InvalidInputError("MRI spacing requires three finite positive values")
    rounded = np.asarray([float(value) * 100.0 for value in spacing_mm], dtype=np.float16)
    return mx.array(rounded.astype(np.float32).reshape(1, 3), dtype=mx.float32)


def sample_mr_latent_ncdhw(
    latent: Array,
    weights: Mapping[str, Array],
    *,
    modality_code: int,
    spacing_mm: Sequence[float],
    guidance_scale: float,
    mx: Mlx,
    num_inference_steps: int = 30,
    constructor: RflowConstructor = PRODUCTION_UNET_CONSTRUCTOR,
) -> Array:
    """Sample an MR latent with modality classifier-free guidance.

    ``latent`` is the initial noise ``(1, 4, D, H, W)``. Each step runs a doubled
    batch, conditioned class (``modality_code``, an index of 8 or more) first and
    unconditional class 0 second, with identical spacing and time, then applies
    ``unconditional + guidance_scale * (conditioned - unconditioned)``. Returns the
    final latent, not yet divided by the scale factor (``decode_mr_ncdhw`` does that).

    Raises:
        InvalidInputError: If the latent shape, modality code, guidance scale or
            spacing is invalid.
    """
    require_ncdhw(latent, name="MRI latent")
    if tuple(latent.shape[:2]) != (1, 4):
        raise InvalidInputError("MRI latent requires batch=1 and four latent channels")
    if modality_code < 8 or modality_code >= int(cast(int, constructor["num_class_embeds"])):
        raise InvalidInputError("MRI modality is outside the pinned class embedding table")
    if not math.isfinite(guidance_scale) or guidance_scale <= 0:
        raise InvalidInputError("MRI classifier-free guidance must be finite and positive")
    spacing = mr_spacing_tensor_ncdhw(spacing_mm, mx=mx)
    # Upstream duplicates spacing and time for both branches, changing only the class.
    spacing = mx.concatenate([spacing, spacing], axis=0)
    labels = mx.array([modality_code, 0], dtype=mx.int32)
    timesteps = mr_timesteps(
        num_inference_steps=num_inference_steps,
        latent_numel=math.prod(int(size) for size in latent.shape[2:]),
    )
    next_timesteps = np.concatenate((timesteps[1:], np.zeros(1, dtype=np.float32)))
    hidden = latent.astype(mx.float32)
    for timestep, next_timestep in zip(timesteps.tolist(), next_timesteps.tolist(), strict=True):
        batch = mx.concatenate([hidden, hidden], axis=0)
        times = mx.full((2,), float(timestep), dtype=mx.float32)
        _, predictions = denoise_unet_ncdhw(
            batch,
            times,
            labels,
            spacing,
            weights,
            constructor=constructor,
            mx=mx,
        )
        conditioned, unconditioned = mx.split(predictions, 2, axis=0)
        velocity = unconditioned + guidance_scale * (conditioned - unconditioned)
        hidden, _, _ = step_rflow(hidden, velocity, float(timestep), float(next_timestep))
        # Materialize each completed step so the lazy graph does not retain the whole trajectory.
        mx.eval(hidden)
    return hidden


def decode_mr_ncdhw(
    latent: Array,
    weights: Mapping[str, Array],
    *,
    scale_factor: float,
    mx: Mlx,
    constructor: VaeConstructor = PRODUCTION_VAE_CONSTRUCTOR,
) -> Array:
    """Decode an MR latent ``(1, 4, D, H, W)`` to ``(1, 1, 4D, 4H, 4W)`` float32.

    The latent is divided by ``scale_factor`` first. Decoding uses 80x80x80
    latent-voxel Gaussian windows with 0.4 overlap when the latent is larger than
    one window, as in the source's dynamic inference. Like the other image
    decoders and ``sample_mr_latent_ncdhw``, it uses the active
    ``arithmetic_mode``: ``fast_fp32`` by default, or the CUDA-matched
    ``source_vae`` under ``arithmetic_mode("source")``.

    Raises:
        InvalidInputError: If the latent contains non-finite values.
    """
    if not bool(mx.all(mx.isfinite(latent)).item()):
        raise InvalidInputError("MRI sampled latent contains non-finite values")
    decoded = sliding_window_decode_ncdhw(
        latent,
        weights,
        constructor=constructor,
        mx=mx,
        roi=DECODE_ROI,
        overlap=DECODE_OVERLAP,
        scale_factor=scale_factor,
        arithmetic=vae_arithmetic(),
    )
    mx.eval(decoded)
    return decoded


def export_mr_int16(image: Array, *, mx: Mlx) -> NDArray[np.int16]:
    """Convert a decoded MR image ``(1, 1, D, H, W)`` to an int16 ``(D, H, W)`` array.

    Multiplies by 1000, clamps negative values to 0 and truncates to int16. There is
    no upper clip, so values beyond the int16 range are not clipped (as upstream).

    Raises:
        InvalidInputError: If the shape is not single-image or values are non-finite.
    """
    require_ncdhw(image, name="decoded MRI")
    if tuple(image.shape[:2]) != (1, 1):
        raise InvalidInputError("Decoded MRI requires a single image channel and batch")
    if not bool(mx.all(mx.isfinite(image)).item()):
        raise InvalidInputError("MRI decoder produced non-finite intensities")
    intensities = mx.maximum(image * 1000.0, 0.0)
    mx.eval(intensities)
    host = np.asarray(intensities, dtype=np.float32)
    if not np.isfinite(host).all():
        raise InvalidInputError("MRI decoder produced non-finite intensities")
    return host[0, 0].astype(np.int16)
