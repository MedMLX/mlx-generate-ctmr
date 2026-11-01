"""NIfTI boundary for image-only MR configurations; called by medmlx.run."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, ArrayFactory, Dtype, Mlx, VaeConstructor

import numpy as np
from medmlx_core.errors import AssetNotReadyError, InvalidInputError, ModelExecutionError
from medmlx_core.runtime import import_mlx

from mlx_generate_ctmr._nifti import nib
from mlx_generate_ctmr._records import RunResult
from mlx_generate_ctmr.configurations import ModelConfiguration, get_configuration
from mlx_generate_ctmr.image import (
    IMAGE_OVERLAP,
    IMAGE_ROI,
    guidance_scale_value,
    sample_image_ncdhw,
)
from mlx_generate_ctmr.maisi_mlx_common import source_pins
from mlx_generate_ctmr.maisi_mlx_from_mask import (
    PRODUCTION_UNET_CONSTRUCTOR,
    PRODUCTION_VAE_CONSTRUCTOR,
)
from mlx_generate_ctmr.maisi_mlx_weights import load_catalog
from mlx_generate_ctmr.weights import (
    load_network,
    network_file,
    require_configuration,
)

type PathArg = str | PathLike[str]


def _geometry(
    config: ModelConfiguration,
    output_size: Sequence[int] | None,
    spacing_mm: Sequence[float] | None,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Resolve the MR output size (voxels) and spacing (mm), defaulting to the configuration.

    Raises InvalidInputError unless the size is three multiples of 32 within the
    dense FP32 graph limits and the spacing is three positive finite values.
    """
    size = config.output_size if output_size is None else tuple(output_size)
    spacing = config.spacing_mm if spacing_mm is None else tuple(spacing_mm)
    if len(size) != 3 or any(
        isinstance(v, bool) or not isinstance(cast(object, v), int) or v < 32 or v % 32
        for v in size
    ):
        raise InvalidInputError("MR output_size must be three positive multiples of 32")
    # Same maximum latent volume as the existing dense FP32 CT graph, before allocation.
    if np.prod(size) > 256**3 or any(v > 512 for v in size):
        raise InvalidInputError("MR dense FP32 graph requires at most 256**3 voxels and axes <=512")
    if len(spacing) != 3 or any(
        isinstance(v, bool)
        or not isinstance(cast(object, v), int | float)
        or not np.isfinite(v)
        or v <= 0
        or v * 100 > np.finfo(np.float16).max
        for v in spacing
    ):
        raise InvalidInputError("spacing_mm must be three positive finite FP16-scalable values")
    return size, tuple(float(v) for v in spacing)


def _load_networks(
    directory: Path, config: ModelConfiguration, mx: Mlx
) -> tuple[dict[str, Array], dict[str, Array], float]:
    """Load VAE and UNet weights checked against the catalog; returns `(vae, unet,
    unet_scale_factor)`.

    Raises AssetNotReadyError on a catalog mismatch, unreadable weights or a missing scale
    factor.
    """
    catalog = load_catalog(config.name)
    try:
        vae, _ = load_network(
            network_file(directory),
            network="vae",
            mx=cast("ArrayFactory[Array, Dtype]", mx),
            catalog=catalog,
            configuration=config.name,
        )
        unet, factor = load_network(
            network_file(directory),
            network="unet",
            mx=cast("ArrayFactory[Array, Dtype]", mx),
            catalog=catalog,
            configuration=config.name,
        )
    except (InvalidInputError, OSError, ValueError) as exc:
        raise AssetNotReadyError(
            f"{config.name} weights do not match their catalog: {exc}",
            reason=str(exc),
            hint="Download the matching safetensors weights again.",
        ) from exc
    if factor is None:
        raise AssetNotReadyError(
            "UNet scale_factor is missing",
            reason="missing scale_factor",
            hint="Download the matching safetensors weights again.",
        )
    mx.eval(*vae.values(), *unet.values())
    return vae, unet, factor


@dataclass(frozen=True, slots=True)
class ImageRequest:
    """A validated image-only MR request: grid, conditioning, decoder plan and destination."""

    config: ModelConfiguration
    size: tuple[int, ...]
    spacing: tuple[float, ...]
    modality: int
    guidance: float
    seed: int
    num_inference_steps: int
    roi: tuple[int, ...]
    overlap: float
    constructor: VaeConstructor
    destination: Path


def check_image_request(
    *,
    output_dir: PathArg,
    configuration: str,
    modality: str | int | None,
    output_size: Sequence[int] | None,
    spacing_mm: Sequence[float] | None,
    guidance_scale: float | None,
    seed: int,
    num_inference_steps: int | None,
    sliding_window_size: Sequence[int] | None,
    sliding_window_overlap: float | None,
    autoencoder_tp_num_splits: int | None,
    overwrite: bool,
) -> ImageRequest:
    """Validate an image-only MR request without reading weights or loading MLX.

    ``None`` selects configuration defaults for geometry, guidance and step count,
    and the image path's defaults for the sliding-window arguments.

    Raises:
        InvalidInputError: If an argument is invalid, ``output_dir`` is a file, or
            the output already exists and ``overwrite`` is false.
    """
    from mlx_generate_ctmr.medmlx import parse_options, refuse_existing

    config = get_configuration(configuration)
    size, spacing = _geometry(config, output_size, spacing_mm)
    index = config.modality_index(modality)
    try:
        guidance = guidance_scale_value(config, guidance_scale)
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
    options = parse_options(
        generation_mode="from_mask",
        anatomy=(),
        anatomy_size=None,
        seed=seed,
        num_inference_steps=num_inference_steps,
        mask_generation_num_inference_steps=None,
        sliding_window_size=sliding_window_size,
        sliding_window_overlap=sliding_window_overlap,
        autoencoder_tp_num_splits=autoencoder_tp_num_splits,
    )
    if not isinstance(cast(object, overwrite), bool):
        raise InvalidInputError("overwrite must be a boolean")
    roi = options.sliding_window_size or IMAGE_ROI
    overlap = (
        IMAGE_OVERLAP if options.sliding_window_overlap is None else options.sliding_window_overlap
    )
    constructor = cast("VaeConstructor", dict(PRODUCTION_VAE_CONSTRUCTOR))
    if options.autoencoder_tp_num_splits is not None:
        constructor["num_splits"] = options.autoencoder_tp_num_splits
    latent = tuple(axis // 4 for axis in size)
    window = latent if math.prod(latent) <= math.prod(roi) else tuple(
        min(axis, bound) for axis, bound in zip(latent, roi, strict=True)
    )
    split_length = window[int(constructor["dim_split"])]
    splits = int(constructor["num_splits"])
    # MAISI's split/crop needs three voxels of context per chunk. Use one split
    # for an incompatible default; an explicit split must be safe as requested.
    if splits > 1 and (split_length % splits or split_length // splits < 3):
        if options.autoencoder_tp_num_splits is not None:
            raise InvalidInputError(
                f"MR decoder window length {split_length} cannot safely use {splits} splits; "
                "choose autoencoder_tp_num_splits=1 or a compatible window"
            )
        constructor["num_splits"] = 1
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise InvalidInputError(f"output_dir is not a directory: {output}")
    destination = output / f"synthetic_{configuration.replace('-', '_')}.nii.gz"
    refuse_existing(output, {"synthetic_image": destination}, overwrite=overwrite)
    return ImageRequest(
        config=config,
        size=size,
        spacing=spacing,
        modality=index,
        guidance=guidance,
        seed=options.seed,
        num_inference_steps=options.num_inference_steps or config.num_inference_steps,
        roi=tuple(roi),
        overlap=overlap,
        constructor=constructor,
        destination=destination,
    )


def run_image(request: ImageRequest, *, weights_path: PathArg) -> RunResult[str]:
    """Generate one validated image-only MR volume and write it as an int16 NIfTI.

    Checks the converted weights in ``weights_path`` and writes
    ``synthetic_mr.nii.gz`` or ``synthetic_mr_brain.nii.gz`` (``request.destination``).
    The saved affine is diagonal with the requested millimeter spacing and zero
    origin; no reference-image geometry is copied. The NIfTI header's spatial
    unit is left unknown even though the spacing values are specified in mm. Returns
    ``{"outputs": {"synthetic_image": path}, "details": {...}}``.

    Raises:
        InvalidInputError: If ``output_dir`` is the weights folder.
        AssetNotReadyError: If the converted weights are missing or do not match
            the pinned catalog.
        ModelExecutionError: If sampling, decoding or writing fails.
    """
    from mlx_generate_ctmr.medmlx import require_weights_dir

    configuration = request.config.name.value
    directory = require_weights_dir(
        weights_path,
        networks=request.config.networks,
        catalog=load_catalog(configuration),
        label=f"NV-Generate-{configuration} weights",
        hint="Download the matching safetensors weights from MedMLX/mlx-generate-ctmr.",
    )
    try:
        require_configuration(directory, configuration=configuration)
    except InvalidInputError as exc:
        raise AssetNotReadyError(
            str(exc), reason=str(exc),
            hint="Download the matching safetensors weights and conversion.json.",
        ) from exc
    if request.destination.parent == directory:
        raise InvalidInputError("output_dir must differ from weights_path")
    return _execute_image(
        request.destination,
        directory,
        request.config,
        size=request.size,
        spacing=request.spacing,
        modality=request.modality,
        guidance=request.guidance,
        seed=request.seed,
        num_inference_steps=request.num_inference_steps,
        roi=request.roi,
        overlap=request.overlap,
        constructor=request.constructor,
    )


def _execute_image(
    destination: Path,
    directory: Path,
    config: ModelConfiguration,
    *,
    size: tuple[int, ...],
    spacing: tuple[float, ...],
    modality: int,
    guidance: float,
    seed: int,
    num_inference_steps: int,
    roi: Sequence[int],
    overlap: float,
    constructor: VaeConstructor,
) -> RunResult[str]:
    """Sample one image-only volume and save it to `destination` as NIfTI.

    `size` is in voxels and `spacing` in mm. The saved array is int16 on the
    requested grid; the affine comes from the sampler. Raises ModelExecutionError
    if sampling, validation or saving fails.
    """
    started = time.perf_counter()
    mx = cast("Mlx", import_mlx())
    mx.reset_peak_memory()
    vae, unet, factor = _load_networks(directory, config, mx)
    try:
        result = sample_image_ncdhw(
            size,
            spacing,
            configuration=config.name,
            modality=modality,
            guidance_scale=guidance,
            seed=seed,
            num_inference_steps=num_inference_steps,
            scale_factor=factor,
            mx=mx,
            unet_weights=unet,
            vae_weights=vae,
            unet_constructor=PRODUCTION_UNET_CONSTRUCTOR,
            vae_constructor=constructor,
            roi=roi,
            overlap=overlap,
        )
        if result.image.shape != (1, 1, *size) or result.image.dtype != np.int16:
            raise ValueError("image-only output does not match the requested grid/int16 dtype")
        destination.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(result.image[0, 0], result.affine), destination)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ModelExecutionError(f"NV-Generate-{config.name} MLX failed: {exc}") from exc
    return {
        "outputs": {"synthetic_image": destination.as_posix()},
        "details": {
            "configuration": config.name.value,
            "generation_mode": "image",
            "modality": modality,
            "seed": seed,
            "guidance_scale": guidance,
            "num_inference_steps": num_inference_steps,
            "shape": list(size),
            "spacing_mm": list(spacing),
            "dtype": "int16",
            "intensity_units": "MR arbitrary units (decoded * 1000, lower clip only)",
            "scale_factor": factor,
            "sliding_window_size": list(roi),
            "sliding_window_overlap": overlap,
            "autoencoder_tp_num_splits": int(constructor["num_splits"]),
            "source": source_pins(config.name),
            "stage_timings_s": dict(result.stage_timings_s),
            "peak_memory_bytes": mx.get_peak_memory(),
            "elapsed_s": time.perf_counter() - started,
        },
    }
