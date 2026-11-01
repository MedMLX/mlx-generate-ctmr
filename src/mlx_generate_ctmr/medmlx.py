"""MedMLX model declaration and end-to-end runner for NV-Generate (MAISI CT, MR, MR-Brain).

The runner covers mode dispatch, the from-mask CT stage, the paired mask and
composition stages, NIfTI readers/writers, MAISI mask preparation and QC,
output validation, and the paired child-seed derivation. MR configurations are
dispatched to ``mlx_generate_ctmr.image_runner``.

Run records, generation specs, asset staging, the model registry, and readiness
belong to the calling host application (for example the ``medmlx`` package).
Weights are explicit native safetensors directories. The image arithmetic needs the Apple
Silicon Metal GPU; there is no CPU fallback. Generated volumes are synthetic
research data, not for clinical use.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from mlx_generate_ctmr._arrays import HostArray, NumericArray

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import (
        Array,
        ArrayConstructor,
        ArrayFactory,
        Dtype,
        Mlx,
        VaeConstructor,
    )

import numpy as np
from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MissingDependencyError,
    ModelExecutionError,
)
from medmlx_core.maisi import (
    MAISI_LABELS,
    NV_GENERATE_CT_SIZE_CONTROL_PATHOLOGIES,
    NvGenerateAnatomySize,
    maisi_label,
    normalize_maisi_anatomy_name,
    validate_maisi_geometry,
)
from medmlx_core.runtime import import_mlx, mlx_default_device_name, probe_mlx_runtime

from mlx_generate_ctmr._morphology import fill_label_envelope_holes
from mlx_generate_ctmr._nifti import ImageFileError, NiftiImage, nib
from mlx_generate_ctmr._records import (
    Catalog,
    ImageFacts,
    JsonObject,
    MaskFacts,
    PairFacts,
    RunResult,
)
from mlx_generate_ctmr.arithmetic_mode import (
    ARITHMETIC_MODES,
    active_arithmetic_mode,
    arithmetic_mode,
)
from mlx_generate_ctmr.attention_budget import attention_memory_budget
from mlx_generate_ctmr.configurations import Configuration, get_configuration
from mlx_generate_ctmr.decode import DEFAULT_OVERLAP, DEFAULT_ROI_SIZE
from mlx_generate_ctmr.from_mask import (
    require_supported_production_shape,
    sample_ct_from_mask_ncdhw,
)
from mlx_generate_ctmr.maisi_mlx_common import (
    PINNED_MONAI_REVISION,
    PINNED_NV_GENERATE_REVISION,
    source_pins,
)
from mlx_generate_ctmr.maisi_mlx_from_mask import (
    PRODUCTION_CONTROLNET_CONSTRUCTOR,
    PRODUCTION_UNET_CONSTRUCTOR,
    PRODUCTION_VAE_CONSTRUCTOR,
)
from mlx_generate_ctmr.maisi_mlx_mask_weights import load_catalog as load_mask_catalog
from mlx_generate_ctmr.maisi_mlx_weights import load_catalog
from mlx_generate_ctmr.mask_archive import (
    MaskGeometry,
    load_mask_archive,
    resample_mask_geometry,
    select_archive_mask,
)
from mlx_generate_ctmr.mask_augmentation import augment_archive_mask
from mlx_generate_ctmr.mask_condition import (
    MASK_ANATOMY_SIZE_SLOTS,
    prepare_mask_generation_controls,
    select_canonical_mask_labels_ncdhw,
)
from mlx_generate_ctmr.mask_ddpm import PRODUCTION_MASK_LATENT_SHAPE, sample_mask_labels_ncdhw
from mlx_generate_ctmr.mask_decode import load_mask_label_id_map
from mlx_generate_ctmr.mask_postprocess import postprocess_mask
from mlx_generate_ctmr.mask_qc import validate_write_and_require_maisi_mask
from mlx_generate_ctmr.mask_unet import PRODUCTION_MASK_UNET_CONSTRUCTOR
from mlx_generate_ctmr.mask_vae import PRODUCTION_MASK_VAE_CONSTRUCTOR
from mlx_generate_ctmr.weights import (
    load_mask_network,
    load_network,
    network_file,
)

MODEL_ID: Final = "mlx-generate-ctmr"
MASK_SPACING_MM: Final = (1.5, 1.5, 1.5)
MASK_ORIENTATION: Final = ("R", "A", "S")
WARNING: Final = (
    "Synthetic model output is not patient data, not ground truth, and must be labeled "
    "before an application turns it into teaching content."
)

# Paired-mask grid (voxels) and the released mask-generation network definitions.
_MASK_SHAPE = (256, 256, 256)
_MASK_LATENT_SHAPE = PRODUCTION_MASK_LATENT_SHAPE
_MASK_UNET_CONSTRUCTOR = PRODUCTION_MASK_UNET_CONSTRUCTOR

_SYNTHETIC_CT = "synthetic_ct.nii.gz"
_SYNTHETIC_LABEL = "synthetic_ct_label.nii.gz"
_PREPARED_MASK = "input_mask_ras.nii.gz"
_MASK_QC = "mask_qc.json"
_MASK_STAGE_DIR = ("stages", "mask")
_GENERATED_MASK = "generated_mask.nii.gz"
_CONDITIONING_MASK = "conditioning_mask.nii.gz"
_CT_NETWORKS = ("vae", "unet", "controlnet")
_MASK_NETWORKS = ("unet", "vae")
_VALID_LABELS = frozenset((*range(133), 200))
_RUNTIME_ERRORS = (OSError, RuntimeError, ValueError)
_CT_WEIGHTS_HINT = (
    "Download the CT safetensors weights from MedMLX/mlx-generate-ctmr."
)
_MASK_WEIGHTS_HINT = (
    "Download the paired-mask safetensors weights from MedMLX/mlx-generate-ctmr."
)
_MASK_WEIGHTS_LABEL = "NV-Generate-CT mask-generation weights"
_MASK_DATABASE_LABEL = "MAISI candidate-mask database"
_LABEL_DICT_LABEL = "MAISI mask label remap (configs/label_dict_124_to_132.json)"
_SIZE_DATABASE_LABEL = "anatomy-size database (datasets/all_anatomy_size_conditions.json)"

CT_MODEL: Final[dict[str, object]] = {
    "id": "mlx-generate-ctmr-ct",
    "label": "NV-Generate CT",
    "task": "generation",
    "run": "mlx_generate_ctmr.medmlx:run_ct",
    "summary": (
        "Synthetic CT and its MAISI label map, from a supplied MAISI label mask or "
        "from requested anatomy with optional size controls."
    ),
    # The names output_label_table writes for the label maps.
    "labels": {str(item.value): item.name for item in MAISI_LABELS if item.value != 0},
    # README: fast CT from-mask at 256³; convert the recorded GiB to decimal GB.
    "peak_memory_gb": 27.9 * 1024**3 / 1e9,
}
MR_MODEL: Final[dict[str, str]] = {
    "id": "mlx-generate-ctmr-mr",
    "label": "NV-Generate MR",
    "task": "generation",
    "run": "mlx_generate_ctmr.medmlx:run_mr",
    "summary": "Synthetic image-only body MRI (T1, T2 or FLAIR), without a label map.",
}
MR_BRAIN_MODEL: Final[dict[str, str | float]] = {
    "id": "mlx-generate-ctmr-mr-brain",
    "label": "NV-Generate MR-Brain",
    "task": "generation",
    "run": "mlx_generate_ctmr.medmlx:run_mr_brain",
    "summary": (
        "Synthetic image-only brain MRI (T1, T2, FLAIR, MRA or SWI, optionally "
        "skull-stripped), without a label map."
    ),
    # README: fast guided MR-Brain at 256x256x128; GiB to decimal GB.
    "peak_memory_gb": 23.8 * 1024**3 / 1e9,
}

type _PathArg = str | PathLike[str]


@dataclass(frozen=True, slots=True)
class _Options:
    """Validated, normalized run options shared by the CT from-mask and paired paths."""

    generation_mode: str
    anatomy: tuple[str, ...]
    anatomy_size: tuple[NvGenerateAnatomySize, ...]
    seed: int
    num_inference_steps: int | None
    mask_generation_num_inference_steps: int | None
    sliding_window_size: tuple[int, int, int] | None
    sliding_window_overlap: float | None
    autoencoder_tp_num_splits: int | None
    body_regions: tuple[str, ...] = ()
    geometry: MaskGeometry | None = None


@dataclass(frozen=True, slots=True)
class SeedLineage:
    """Stable child seeds derived from one root seed (see ``derive_seed_lineage``).

    ``mask`` seeds mask generation (or archive selection and augmentation) and ``ct``
    seeds the CT stage. ``pet`` is derived for a downstream PET stage; nothing in
    this package consumes it, but it is part of the recorded lineage.
    """

    root: int
    mask: int
    ct: int
    pet: int

    def to_payload(self) -> dict[str, int]:
        """Serialize the root and derived generation-stage seeds."""
        return {"root": self.root, "mask": self.mask, "ct": self.ct, "pet": self.pet}


def run(
    *,
    output_dir: _PathArg,
    weights_path: _PathArg | None = None,
    configuration: str = "ct",
    modality: str | int | None = None,
    output_size: Sequence[int] | None = None,
    spacing_mm: Sequence[float] | None = None,
    guidance_scale: float | None = None,
    mask_path: _PathArg | None = None,
    anatomy: Sequence[str] | None = None,
    anatomy_size: Mapping[str, float] | None = None,
    mask_weights_path: _PathArg | None = None,
    mask_label_dict_path: _PathArg | None = None,
    anatomy_size_conditions_path: _PathArg | None = None,
    generation_mode: str | None = None,
    body_regions: Sequence[str] = (),
    mask_database_path: _PathArg | None = None,
    mask_directory: _PathArg | None = None,
    seed: int = 0,
    num_inference_steps: int | None = None,
    mask_generation_num_inference_steps: int | None = None,
    sliding_window_size: Sequence[int] | None = None,
    sliding_window_overlap: float | None = None,
    autoencoder_tp_num_splits: int | None = None,
    overwrite: bool = False,
    ct_intensity_qc: bool | None = None,
    ct_qc_statistics_path: _PathArg | None = None,
    ct_qc_max_attempts: int = 2,
    attention_scratch_bytes: int | None = None,
    arithmetic: str = "fast",
) -> RunResult[str]:
    """Run NV-Generate for one configuration and write NIfTI outputs.

    Dispatches on ``configuration`` (``"ct"``, ``"mr"`` or ``"mr-brain"``) and, for
    CT, on ``generation_mode``. A supplied CT mask selects ``from_mask`` by default;
    otherwise CT defaults to ``paired``. MR produces an image without a label map.

    For CT, each generated image is checked against per-region median HU
    statistics (``ct_intensity_qc``, on by default). A rejected image is regenerated
    with a derived seed, up to ``ct_qc_max_attempts`` attempts in total (default 2);
    if every attempt is rejected a ``ModelExecutionError`` is raised. Passing
    ``ct_intensity_qc=False`` records an explicit opt-out. CT QC cannot be enabled
    for MR; a statistics override or nondefault attempt count requires enabled QC.

    ``attention_scratch_bytes`` opts into request-scoped query chunk sizing for
    dense self/cross-attention, including every QC retry. It budgets three FP32
    score matrices; weights, Q/K/V, outputs, allocator cache and backend workspace
    are excluded. See ``attention_memory_budget`` for the exact scope. The
    original token-threshold policy remains the default. Under
    ``arithmetic="fast"`` the image networks' self-attention computes scores block
    by block and needs no budget; mask-generation attention still uses it.

    ``arithmetic`` selects the image UNet/ControlNet/VAE-decoder arithmetic (see
    ``mlx_generate_ctmr.arithmetic_mode``). ``"fast"`` (default) uses native FP32
    operations, with M5 neural-accelerator kernels where hardware and macOS support
    them and ordinary MLX GPU operations otherwise. It accumulates in a different
    reduction order and decodes without ``autoencoder_tp_num_splits`` splits, which
    only bound memory; the paired-mask UNet and mask VAE use the same kernels.
    ``"source"`` keeps the CUDA operation order the seeded references and the
    retained pretrained MR-Brain comparison used, and the mask UNet's generic
    MLX ops.

    Args:
        output_dir: Directory for generated NIfTI files and associated records.
        weights_path: Optional selected native model folder. Omit to download the
            ``ct``, ``mr`` or ``mr-brain`` subfolder of ``MedMLX/mlx-generate-ctmr``;
            CT also downloads its paired-mask companion folder. Uses
            ``model.safetensors`` with ``vae.``/``unet.`` tensor prefixes
            (also ``controlnet.`` for CT). The UNet must include ``scale_factor``.
            Conversion receipts are not required.
        configuration: ``ct``, ``mr``, or ``mr-brain``.
        modality: Supported modality name or index for the configuration; ``None``
            selects its default (T1 for MR).
        output_size: Requested three-axis voxel shape for CT paired or MR generation.
            CT from-mask takes its shape from the mask and rejects this argument.
        spacing_mm: Requested voxel spacing in mm for CT paired or MR generation.
            CT from-mask takes spacing from the mask and rejects this argument.
        guidance_scale: MR classifier-free guidance scale; rejected for CT.
        mask_path: CT from-mask input NIfTI (``.nii`` or ``.nii.gz``) with MAISI
            labels 0..132 and required body-envelope label 200.
        anatomy: Canonical MAISI names to retain in CT paired output; required for
            paired mode and rejected for from-mask and MR.
        anatomy_size: CT paired size controls keyed by selected anatomy, with scales
            0..1 or -1; a nonempty mapping selects learned mask generation.
            Unspecified slots, including tumors, come from the selected source
            database row, so a mask can contain a tumor that was not requested.
        mask_weights_path: Native mask UNet and VAE directory for learned paired CT.
            Automatic CT downloads use the managed root's ``mask`` folder beside
            ``ct``. With explicit local weights, ``<weights_path>/mask`` is used
            when present. An explicit value overrides discovery.
            Uses mask-generation ``model.safetensors`` with ``unet.``/``vae.`` prefixes.
            The UNet must include ``scale_factor``; no conversion receipt is required.
        mask_label_dict_path: Source label-remapping JSON for learned paired CT.
        anatomy_size_conditions_path: Source anatomy-size database for learned paired CT.
        generation_mode: CT ``from_mask`` or ``paired``; MR accepts ``image`` or None.
        body_regions: Body-region filters for archive-based paired CT; cannot be
            combined with anatomy-size controls.
        mask_database_path: Candidate-mask database for archive-based paired CT.
        mask_directory: Directory containing the candidate masks for archive-based paired CT.
        seed: Nonnegative root seed; paired CT derives separate mask and CT seeds.
        num_inference_steps: Image-generation steps, 1..1000; None uses the default.
        mask_generation_num_inference_steps: Mask-generation steps, 1..1000; not
            accepted for CT from-mask or MR.
        sliding_window_size: Three positive multiples of 16 for latent decoding,
            or None for the configuration default.
        sliding_window_overlap: Decoder window overlap in [0, 1), or None for the default.
        autoencoder_tp_num_splits: Decoder split count 1, 2, 4, 8 or 16, or None for default.
        overwrite: Allow existing output files to be replaced.
        ct_intensity_qc: Enable or disable CT intensity QC; None enables it for CT.
        ct_qc_statistics_path: Optional CT QC statistics file; requires enabled CT QC.
        ct_qc_max_attempts: Maximum positive number of CT generation attempts, default 2.
        attention_scratch_bytes: Optional positive byte budget for attention score
            matrices, with exclusions described above.
        arithmetic: ``"fast"`` (default) or ``"source"``, described above.

    Returns ``{"outputs": {role: absolute posix path}, "details": {...}}``. The
    details record seeds, shapes, dtypes, timings, peak MLX memory and the upstream
    and weight pins. Outputs are synthetic and need human review.

    Raises:
        InvalidInputError: If an argument is invalid or an output exists and
            ``overwrite`` is false.
        AssetNotReadyError: If converted weights or reference files are missing or
            do not match the pinned catalog.
        MissingDependencyError: If MLX or its Metal GPU is unavailable.
        ModelExecutionError: If generation fails or every CT QC attempt is rejected.
    """

    if attention_scratch_bytes is not None and (
        type(attention_scratch_bytes) is not int or attention_scratch_bytes < 1
    ):
        raise InvalidInputError("attention_scratch_bytes must be a positive integer or None")
    if arithmetic not in ARITHMETIC_MODES:
        raise InvalidInputError(f"arithmetic must be one of {ARITHMETIC_MODES}")
    mode = arithmetic
    if ct_intensity_qc is not None and not isinstance(cast(object, ct_intensity_qc), bool):
        raise InvalidInputError("ct_intensity_qc must be a boolean")
    if (
        isinstance(ct_qc_max_attempts, bool)
        or not isinstance(cast(object, ct_qc_max_attempts), int)
        or ct_qc_max_attempts < 1
    ):
        raise InvalidInputError("ct_qc_max_attempts must be a positive integer")
    qc_enabled = configuration == "ct" if ct_intensity_qc is None else ct_intensity_qc
    if not qc_enabled and (ct_qc_statistics_path is not None or ct_qc_max_attempts != 2):
        raise InvalidInputError(
            "CT QC statistics and attempt settings require ct_intensity_qc=true"
        )
    if qc_enabled and configuration != "ct":
        raise InvalidInputError("CT intensity QC only applies to CT configurations")
    config = get_configuration(configuration)
    config.modality_index(modality)
    if not isinstance(cast(object, overwrite), bool):
        raise InvalidInputError("overwrite must be a boolean")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise InvalidInputError(f"output_dir is not a directory: {output}")
    statistics = None
    planned_outputs = [output]
    if qc_enabled:
        from mlx_generate_ctmr.ct_qc import attempt_directory, load_ct_statistics

        try:
            statistics = load_ct_statistics(
                None if ct_qc_statistics_path is None else Path(ct_qc_statistics_path)
            )
        except (OSError, ValueError) as error:
            raise InvalidInputError(f"Invalid CT QC statistics: {error}") from error
        # Each QC attempt writes its own outputs; refuse any conflict before the first.
        planned_outputs = [attempt_directory(output, index) for index in range(ct_qc_max_attempts)]
    # Check every argument before downloading weights, so a typo fails at once.
    if config.name is Configuration.CT:
        options, resolved_mask = _check_ct_request(
            guidance_scale=guidance_scale,
            mask_path=mask_path,
            anatomy=anatomy,
            anatomy_size=anatomy_size,
            generation_mode=generation_mode,
            body_regions=body_regions,
            output_size=output_size,
            spacing_mm=spacing_mm,
            seed=seed,
            num_inference_steps=num_inference_steps,
            mask_generation_num_inference_steps=mask_generation_num_inference_steps,
            sliding_window_size=sliding_window_size,
            sliding_window_overlap=sliding_window_overlap,
            autoencoder_tp_num_splits=autoencoder_tp_num_splits,
        )
        for directory in planned_outputs:
            _refuse_ct_outputs(directory, resolved_mask, overwrite=overwrite)
        _check_supplied_ct_assets(
            options,
            mask_weights_path=mask_weights_path,
            mask_label_dict_path=mask_label_dict_path,
            anatomy_size_conditions_path=anatomy_size_conditions_path,
            mask_database_path=mask_database_path,
            mask_directory=mask_directory,
        )
    else:
        from mlx_generate_ctmr.image_runner import check_image_request

        if any(
            value is not None
            for value in (
                mask_path,
                anatomy,
                anatomy_size,
                mask_weights_path,
                mask_label_dict_path,
                anatomy_size_conditions_path,
                mask_generation_num_inference_steps,
                mask_database_path,
                mask_directory,
                body_regions or None,
            )
        ):
            raise InvalidInputError("MR image generation does not accept mask or anatomy controls")
        if generation_mode not in {None, "image"}:
            raise InvalidInputError("MR configurations require generation_mode='image'")
        check_image_request(
            output_dir=output,
            configuration=configuration,
            modality=modality,
            output_size=output_size,
            spacing_mm=spacing_mm,
            guidance_scale=guidance_scale,
            seed=seed,
            num_inference_steps=num_inference_steps,
            sliding_window_size=sliding_window_size,
            sliding_window_overlap=sliding_window_overlap,
            autoencoder_tp_num_splits=autoencoder_tp_num_splits,
            overwrite=overwrite,
        )

    from medmlx_core.weights import weights_dir, weights_revision

    if weights_path is None:
        folder = weights_dir(f"mlx-generate-ctmr-{configuration}")
        managed_root = folder.parent
    else:
        managed_root = None
        folder = Path(weights_path).expanduser()
    weights_path = folder
    if configuration == "ct":
        # Managed downloads keep mask/ beside the selected ct/ folder; explicit
        # local model folders can instead contain mask/ themselves.
        if mask_weights_path is None:
            for candidate in (
                folder / "mask", None if managed_root is None else managed_root / "mask"
            ):
                if candidate is not None and candidate.is_dir():
                    mask_weights_path = candidate
                    break
        if mask_label_dict_path is None:
            mask_label_dict_path = _bundle_asset(folder, "label_dict_124_to_132.json", "configs")
        if anatomy_size_conditions_path is None:
            anatomy_size_conditions_path = _bundle_asset(
                folder, "all_anatomy_size_conditions.json", "datasets",
            )
        if mask_database_path is None:
            for filename in ("database.json", "candidate_masks_flexible_size_and_spacing_4000.json",
                             "candidate_masks_flexible_size_and_spacing_3000.json"):
                mask_database_path = _bundle_asset(folder, filename, "datasets")
                if mask_database_path is not None:
                    break
        if mask_directory is None:
            for candidate in (folder / "masks", folder / "all_masks_flexible_size_and_spacing_4000",
                              folder / "datasets" / "all_masks_flexible_size_and_spacing_4000"):
                if candidate.is_dir():
                    mask_directory = candidate
                    break

    def generate(attempt_output: Path, attempt_seed: int) -> RunResult[str]:
        """One generation attempt under the chosen arithmetic mode and attention budget, annotated
        with model id and weights revision.
        """
        context = (
            attention_memory_budget(attention_scratch_bytes)
            if attention_scratch_bytes is not None
            else nullcontext()
        )
        with context, arithmetic_mode(mode):
            result = _run_once(
                output_dir=attempt_output,
                seed=attempt_seed,
                weights_path=weights_path,
                configuration=configuration,
                modality=modality,
                output_size=output_size,
                spacing_mm=spacing_mm,
                guidance_scale=guidance_scale,
                mask_path=mask_path,
                anatomy=anatomy,
                anatomy_size=anatomy_size,
                mask_weights_path=mask_weights_path,
                mask_label_dict_path=mask_label_dict_path,
                anatomy_size_conditions_path=anatomy_size_conditions_path,
                generation_mode=generation_mode,
                body_regions=body_regions,
                mask_database_path=mask_database_path,
                mask_directory=mask_directory,
                num_inference_steps=num_inference_steps,
                mask_generation_num_inference_steps=mask_generation_num_inference_steps,
                sliding_window_size=sliding_window_size,
                sliding_window_overlap=sliding_window_overlap,
                autoencoder_tp_num_splits=autoencoder_tp_num_splits,
                overwrite=overwrite,
            )
        if attention_scratch_bytes is not None:
            result["details"]["attention_scratch_bytes"] = attention_scratch_bytes
        result["details"]["arithmetic"] = mode
        result["details"]["model_id"] = f"mlx-generate-ctmr-{configuration}"
        result["details"]["weights_revision"] = weights_revision(folder)
        return result

    if statistics is None:
        result = generate(output, seed)
        if configuration == "ct":
            result["details"]["ct_intensity_qc"] = {
                "verdict": "disabled",
                "explicit_opt_out": ct_intensity_qc is False,
            }
        return result
    from mlx_generate_ctmr.ct_qc import generate_with_ct_qc

    return generate_with_ct_qc(
        generate, output=output, seed=seed, statistics=statistics, max_attempts=ct_qc_max_attempts
    )


def _run_once(
    *,
    output_dir: _PathArg,
    weights_path: _PathArg,
    configuration: str = "ct",
    modality: str | int | None = None,
    output_size: Sequence[int] | None = None,
    spacing_mm: Sequence[float] | None = None,
    guidance_scale: float | None = None,
    mask_path: _PathArg | None = None,
    anatomy: Sequence[str] | None = None,
    anatomy_size: Mapping[str, float] | None = None,
    mask_weights_path: _PathArg | None = None,
    mask_label_dict_path: _PathArg | None = None,
    anatomy_size_conditions_path: _PathArg | None = None,
    generation_mode: str | None = None,
    body_regions: Sequence[str] = (),
    mask_database_path: _PathArg | None = None,
    mask_directory: _PathArg | None = None,
    seed: int = 0,
    num_inference_steps: int | None = None,
    mask_generation_num_inference_steps: int | None = None,
    sliding_window_size: Sequence[int] | None = None,
    sliding_window_overlap: float | None = None,
    autoencoder_tp_num_splits: int | None = None,
    overwrite: bool = False,
) -> RunResult[str]:
    """Generate a synthetic CT and its MAISI label map on the MLX Metal GPU.

    ``configuration='mr'`` or ``'mr-brain'`` instead generates one image without
    labels (``generation_mode='image'``). ``modality`` is an upstream name or index;
    omitted, it defaults to T1 (9). ``output_size``, ``spacing_mm`` and
    ``guidance_scale`` default to the pinned variant JSON. MR weights contain VAE
    and UNet tensors in ``model.safetensors``. The output is
    int16 MR intensity on a diagonal RAS affine, without an upper intensity clip.

    ``from_mask`` (the default when ``mask_path`` is given) renders a CT from a 3D
    MAISI label NIfTI with labels 0..132 plus body label 200, at 256x256x128 or
    256x256x256. The mask is reoriented to its closest canonical RAS grid
    (``input_mask_ras.nii.gz``), and the CT and label map are written on that grid:
    the input's voxels with permuted or flipped axes and the matching affine.
    Background enclosed by the body in an axial slice is filled with label 200
    before conditioning.

    ``paired`` (the default otherwise) generates a learned mask when size controls
    are supplied, or selects and augments a mask from ``mask_database_path`` and
    ``mask_directory`` otherwise. Anatomy selections use canonical MAISI names;
    size scales are 0..1 or -1, with at most one controlled tumor. Archive selection
    accepts ``body_regions``. Requested output geometry defaults to 256x256x256 at
    1.5 mm and is resampled with nearest-neighbor spacing and centered crop/pad.
    Delivered labels keep requested anatomy and fold other anatomy into body 200.

    For CT, ``weights_path`` contains VAE, UNet and ControlNet tensors in
    ``model.safetensors``. Learned paired runs also need ``mask_weights_path``
    with mask-generation ``model.safetensors``, the source
    ``configs/label_dict_124_to_132.json`` as ``mask_label_dict_path``, and
    ``datasets/all_anatomy_size_conditions.json`` as ``anatomy_size_conditions_path``.
    Archive-based paired runs instead require the mask database and directory.

    ``seed`` seeds the from-mask latent; a paired run derives distinct mask and CT
    seeds from it. Inference steps are 1..1000; defaults are 30 rflow steps and
    1000 mask DDPM steps. Every output requires human review and is not ground truth.
    """

    config = get_configuration(configuration)
    config.modality_index(modality)
    if config.name is not Configuration.CT:
        from mlx_generate_ctmr.image_runner import check_image_request, run_image

        request = check_image_request(
            output_dir=output_dir,
            configuration=configuration,
            modality=modality,
            output_size=output_size,
            spacing_mm=spacing_mm,
            guidance_scale=guidance_scale,
            seed=seed,
            num_inference_steps=num_inference_steps,
            sliding_window_size=sliding_window_size,
            sliding_window_overlap=sliding_window_overlap,
            autoencoder_tp_num_splits=autoencoder_tp_num_splits,
            overwrite=overwrite,
        )
        return run_image(request, weights_path=weights_path)
    options, resolved_mask = _check_ct_request(
        guidance_scale=guidance_scale,
        mask_path=mask_path,
        anatomy=anatomy,
        anatomy_size=anatomy_size,
        generation_mode=generation_mode,
        body_regions=body_regions,
        output_size=output_size,
        spacing_mm=spacing_mm,
        seed=seed,
        num_inference_steps=num_inference_steps,
        mask_generation_num_inference_steps=mask_generation_num_inference_steps,
        sliding_window_size=sliding_window_size,
        sliding_window_overlap=sliding_window_overlap,
        autoencoder_tp_num_splits=autoencoder_tp_num_splits,
    )
    output = Path(output_dir).expanduser().resolve()
    _refuse_ct_outputs(output, resolved_mask, overwrite=overwrite)
    weights_dir = require_weights_dir(
        weights_path,
        networks=_CT_NETWORKS,
        catalog=load_catalog(),
        label="NV-Generate-CT from-mask weights",
        hint=_CT_WEIGHTS_HINT,
    )
    if resolved_mask is None:
        if not options.anatomy_size:
            database = _require_asset_file(
                _required_argument(mask_database_path, "mask_database_path", "archive paired"),
                label=_MASK_DATABASE_LABEL,
            )
            directory = (
                Path(_required_argument(mask_directory, "mask_directory", "archive paired"))
                .expanduser()
                .resolve()
            )
            return _run_archive_paired(output, weights_dir, database, directory, options)
        mask_weights_dir = require_weights_dir(
            _required_argument(mask_weights_path, "mask_weights_path", "paired"),
            networks=_MASK_NETWORKS,
            catalog=load_mask_catalog(),
            label=_MASK_WEIGHTS_LABEL,
            hint=_MASK_WEIGHTS_HINT,
        )
        label_dict = _require_asset_file(
            _required_argument(mask_label_dict_path, "mask_label_dict_path", "paired"),
            label=_LABEL_DICT_LABEL,
        )
        size_database = _require_asset_file(
            _required_argument(
                anatomy_size_conditions_path, "anatomy_size_conditions_path", "paired"
            ),
            label=_SIZE_DATABASE_LABEL,
        )
        return _run_paired(
            output=output,
            weights_dir=weights_dir,
            mask_weights_dir=mask_weights_dir,
            label_dict_path=label_dict,
            size_database_path=size_database,
            options=options,
        )
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    stage = _run_from_mask(
        output=output,
        weights_dir=weights_dir,
        mask_path=resolved_mask,
        options=options,
        seed=options.seed,
    )
    details: JsonObject = {
        **stage["details"],
        "generation_mode": "from_mask",
        "input": {"mask": resolved_mask.as_posix()},
        "options": options_payload(options),
        "started_at": started_at.isoformat(timespec="seconds"),
        "elapsed_s": round(time.perf_counter() - started, 4),
    }
    return {"outputs": _posix(stage["outputs"]), "details": details}


def _check_ct_request(
    *,
    guidance_scale: float | None,
    mask_path: _PathArg | None,
    anatomy: Sequence[str] | None,
    anatomy_size: Mapping[str, float] | None,
    generation_mode: str | None,
    body_regions: Sequence[str],
    output_size: Sequence[int] | None,
    spacing_mm: Sequence[float] | None,
    seed: int,
    num_inference_steps: int | None,
    mask_generation_num_inference_steps: int | None,
    sliding_window_size: Sequence[int] | None,
    sliding_window_overlap: float | None,
    autoencoder_tp_num_splits: int | None,
) -> tuple[_Options, Path | None]:
    """Validate the CT options and return them with the resolved from-mask input.

    The mask is ``None`` for paired generation. Reads no weights and writes nothing.

    Raises:
        InvalidInputError: If an option is invalid or conflicts with the mode, or the
            from-mask input is missing or not a NIfTI file.
    """
    if guidance_scale is not None:
        raise InvalidInputError("CT ControlNet inference does not use classifier-free guidance")
    mode = generation_mode or ("from_mask" if mask_path is not None else "paired")
    options = parse_options(
        generation_mode=mode,
        anatomy=() if anatomy is None else anatomy,
        anatomy_size=anatomy_size,
        seed=seed,
        num_inference_steps=num_inference_steps,
        mask_generation_num_inference_steps=mask_generation_num_inference_steps,
        sliding_window_size=sliding_window_size,
        sliding_window_overlap=sliding_window_overlap,
        autoencoder_tp_num_splits=autoencoder_tp_num_splits,
    )
    options = _with_mask_geometry(options, body_regions, output_size, spacing_mm)
    if options.generation_mode == "paired":
        if mask_path is not None:
            raise InvalidInputError("paired generation does not accept mask_path")
        return options, None
    if mask_path is None:
        raise InvalidInputError("generation_mode='from_mask' requires mask_path")
    resolved_mask = Path(mask_path).expanduser().resolve()
    if not resolved_mask.is_file():
        raise InvalidInputError(f"Missing MAISI mask: {resolved_mask}")
    if not (resolved_mask.name.endswith(".nii") or resolved_mask.name.endswith(".nii.gz")):
        raise InvalidInputError("MAISI mask_path must be a .nii or .nii.gz NIfTI file")
    return options, resolved_mask


def _refuse_ct_outputs(output: Path, mask: Path | None, *, overwrite: bool) -> None:
    """Refuse a file ``output``, an output that would replace ``mask``, and existing outputs.

    ``mask`` is the from-mask input, or ``None`` for paired generation.
    """
    if output.exists() and not output.is_dir():
        raise InvalidInputError(f"output_dir is not a directory: {output}")
    outputs = _paired_outputs(output) if mask is None else _from_mask_outputs(output)
    if mask in outputs.values():
        raise InvalidInputError("NV-Generate-CT outputs must not overwrite the input mask")
    refuse_existing(output, outputs, overwrite=overwrite)


def _check_supplied_ct_assets(
    options: _Options,
    *,
    mask_weights_path: _PathArg | None,
    mask_label_dict_path: _PathArg | None,
    anatomy_size_conditions_path: _PathArg | None,
    mask_database_path: _PathArg | None,
    mask_directory: _PathArg | None,
) -> None:
    """Check the paired-CT assets the caller supplied; omitted ones come with the weights.

    Raises:
        AssetNotReadyError: If supplied mask weights or a supplied asset file is missing.
        InvalidInputError: If a supplied ``mask_directory`` is not a directory.
    """
    if options.generation_mode != "paired":
        return
    if options.anatomy_size:
        if mask_weights_path is not None:
            require_weights_dir(
                mask_weights_path,
                networks=_MASK_NETWORKS,
                catalog=load_mask_catalog(),
                label=_MASK_WEIGHTS_LABEL,
                hint=_MASK_WEIGHTS_HINT,
            )
        if mask_label_dict_path is not None:
            _require_asset_file(mask_label_dict_path, label=_LABEL_DICT_LABEL)
        if anatomy_size_conditions_path is not None:
            _require_asset_file(anatomy_size_conditions_path, label=_SIZE_DATABASE_LABEL)
        return
    if mask_database_path is not None:
        _require_asset_file(mask_database_path, label=_MASK_DATABASE_LABEL)
    if mask_directory is not None and not Path(mask_directory).expanduser().is_dir():
        raise InvalidInputError(f"mask_directory is not a directory: {mask_directory}")


def _run_from_mask(
    *,
    output: Path,
    weights_dir: Path,
    mask_path: Path,
    options: _Options,
    seed: int,
    output_anatomy: Sequence[str] = (),
    mask_from_model: bool = False,
) -> RunResult[Path]:
    """Run one explicit-MLX from-mask job and write the canonical CT/label pair.

    Supplied masks get an axial body-envelope hole fill before conditioning. A
    mask from the learned model (``mask_from_model``) is used unchanged, as the
    pinned source does: on the seed-0 capture the fill accounted for nearly all
    mask disagreement with upstream.
    """

    vae_ctor = cast("VaeConstructor", dict(PRODUCTION_VAE_CONSTRUCTOR))
    if options.autoencoder_tp_num_splits is not None:
        vae_ctor["num_splits"] = int(options.autoencoder_tp_num_splits)
    roi = options.sliding_window_size or DEFAULT_ROI_SIZE
    overlap = (
        DEFAULT_OVERLAP
        if options.sliding_window_overlap is None
        else options.sliding_window_overlap
    )
    output.mkdir(parents=True, exist_ok=True)
    prepared = output / _PREPARED_MASK
    try:
        mask_facts = prepare_maisi_mask(mask_path, prepared)
        try:
            require_supported_production_shape(mask_facts["shape"])
        except ValueError as exc:
            raise InvalidInputError(str(exc)) from exc
    except InvalidInputError as exc:
        if mask_from_model:
            raise ModelExecutionError(f"NV-Generate-CT MLX failed: {exc}") from exc
        raise
    started = time.perf_counter()
    try:
        label, spacing_mm, affine = load_mask_ncdhw(prepared)
        validate_write_and_require_maisi_mask(
            output / _MASK_QC,
            label[0, 0],
            spacing=spacing_mm,
            orientation=("R", "A", "S"),
        )
        if mask_from_model:
            mask_preprocessing: JsonObject = {"recipe": "none", "filled_voxels": 0}
        else:
            original_background = label == 0
            label[0, 0] = fill_label_envelope_holes(label[0, 0], slice_axis=2, body_label=200)
            mask_preprocessing = {
                "recipe": "axial_body_envelope_holes",
                "filled_voxels": int(np.count_nonzero(original_background & (label != 0))),
            }
        mx = cast("Mlx", import_mlx())
        host = probe_mlx_runtime()
        default_device = mlx_default_device_name(mx)
        mx.reset_peak_memory()
        load_started = time.perf_counter()
        vae_weights, unet_weights, controlnet_weights, scale_factor = _load_ct_networks(
            weights_dir, mx
        )
        mx.eval(*vae_weights.values(), *unet_weights.values(), *controlnet_weights.values())
        load_s = time.perf_counter() - load_started
        result = sample_ct_from_mask_ncdhw(
            label,
            spacing_mm,
            unet_weights=unet_weights,
            controlnet_weights=controlnet_weights,
            vae_weights=vae_weights,
            unet_constructor=PRODUCTION_UNET_CONSTRUCTOR,
            controlnet_constructor=PRODUCTION_CONTROLNET_CONSTRUCTOR,
            vae_constructor=vae_ctor,
            mx=mx,
            scale_factor=scale_factor,
            seed=seed,
            num_inference_steps=options.num_inference_steps or 30,
            affine=affine,
            roi=roi,
            overlap=overlap,
        )
        export_started = time.perf_counter()
        hu_host = np.asarray(result.hu, dtype=np.float32)
        write_hu_nifti(output / _SYNTHETIC_CT, hu_host, prepared)
        export_labels = label
        if output_anatomy:
            # Source paired CT conditions on the full map and selects labels only for delivery.
            selected = select_canonical_mask_labels_ncdhw(
                cast("ArrayConstructor", mx.array)(label, dtype=mx.int32),
                anatomy=output_anatomy,
                mx=mx,
            )
            export_labels = np.asarray(selected, dtype=np.int16)
        write_label_nifti(prepared, output / _SYNTHETIC_LABEL, export_labels)
        export_s = time.perf_counter() - export_started
        facts = validate_output_pair(output / _SYNTHETIC_CT, output / _SYNTHETIC_LABEL)
        elapsed_s = round(time.perf_counter() - started, 4)
        peak_memory_bytes = mx.get_peak_memory()
    except (AssetNotReadyError, MissingDependencyError):
        raise
    except InvalidInputError as exc:
        if mask_from_model:
            raise ModelExecutionError(f"NV-Generate-CT MLX failed: {exc}") from exc
        raise
    except _RUNTIME_ERRORS as exc:
        raise ModelExecutionError(f"NV-Generate-CT MLX failed: {exc}") from exc
    return {
        "outputs": {
            "synthetic_image": output / _SYNTHETIC_CT,
            "target_mask": output / _SYNTHETIC_LABEL,
            "prepared_mask": prepared,
            "mask_qc": output / _MASK_QC,
        },
        "details": {
            "model_id": MODEL_ID,
            "seed": seed,
            "num_inference_steps": result.num_inference_steps,
            "scale_factor": scale_factor,
            "mask_preprocessing": mask_preprocessing,
            "input_mask": {
                "shape": mask_facts["shape"],
                "spacing_mm": mask_facts["spacing"],
                "labels": mask_facts["labels"],
            },
            "output_kind_by_role": {
                "synthetic_image": "synthetic_image",
                "target_mask": "label_map",
            },
            "output_shape_by_role": {
                "synthetic_image": facts["shape"],
                "target_mask": facts["shape"],
            },
            "output_dtype_by_role": {
                "synthetic_image": facts["image_dtype"],
                "target_mask": facts["label_dtype"],
            },
            "output_labels_by_role": {"target_mask": output_label_table(facts["labels"])},
            "affine": [[float(value) for value in row] for row in affine],
            "spacing_mm": [float(value) for value in spacing_mm],
            "decode": {
                "roi": [int(value) for value in roi],
                "overlap": float(overlap),
                "autoencoder_tp_num_splits": int(vae_ctor["num_splits"]),
                "seam_audit": result.seam_audit,
            },
            "timings_s": {
                "cold_weight_load": load_s,
                **dict(result.stage_timings_s),
                "host_export": export_s,
                "total_stage": elapsed_s,
            },
            "peak_memory_bytes": peak_memory_bytes,
            "mlx_default_device": default_device,
            "mlx_host": cast(JsonObject, host.to_payload()),
            "runtime_optimizations": {
                "dtype": "float32",
                # Fast arithmetic runs every UNet and ControlNet self-attention through the
                # flash kernel, which the supported grids always fit (FP32, 32-wide heads,
                # token counts divisible by 32). Source arithmetic does not use it.
                "flash_attention": active_arithmetic_mode() == "fast",
                "cfg_execution": "disabled",
            },
            "implementation": "NVIDIA-Medtech/NV-Generate-CTMR",
            "variant": "rflow-ct",
            "source_ref": PINNED_NV_GENERATE_REVISION,
            "framework_source_ref": PINNED_MONAI_REVISION,
            # Upstream checkpoint pins; run() records the folder's own revision.
            "source": source_pins("ct"),
            "weights_path": weights_dir.as_posix(),
            "requires_human_review": True,
            "warning": WARNING,
        },
    }


def _run_paired(
    *,
    output: Path,
    weights_dir: Path,
    mask_weights_dir: Path,
    label_dict_path: Path,
    size_database_path: Path,
    options: _Options,
) -> RunResult[str]:
    """Generate a mask with the learned model, then run the from-mask CT stage on it."""

    started_at = datetime.now(UTC)
    started = time.perf_counter()
    mask_output = output.joinpath(*_MASK_STAGE_DIR)
    lineage = derive_seed_lineage(options.seed)
    mask_stage = _run_paired_mask_stage(
        output=mask_output,
        mask_weights_dir=mask_weights_dir,
        label_dict_path=label_dict_path,
        size_database_path=size_database_path,
        options=options,
        seed=lineage.mask,
    )
    ct_options = _Options(
        generation_mode="from_mask",
        anatomy=(),
        anatomy_size=(),
        seed=lineage.ct,
        num_inference_steps=options.num_inference_steps,
        mask_generation_num_inference_steps=None,
        sliding_window_size=options.sliding_window_size,
        sliding_window_overlap=options.sliding_window_overlap,
        autoencoder_tp_num_splits=options.autoencoder_tp_num_splits,
    )
    conditioning_mask = mask_stage["outputs"]["conditioning_mask"]
    ct_stage = _run_from_mask(
        output=output,
        weights_dir=weights_dir,
        mask_path=conditioning_mask,
        options=ct_options,
        seed=lineage.ct,
        output_anatomy=options.anatomy,
        mask_from_model=True,
    )
    outputs = {
        **ct_stage["outputs"],
        "conditioning_mask_qc": ct_stage["outputs"]["mask_qc"],
        "generated_mask": mask_stage["outputs"]["generated_mask"],
        "conditioning_mask": conditioning_mask,
        "mask_qc": mask_stage["outputs"]["mask_qc"],
    }
    details: JsonObject = dict(ct_stage["details"])
    for role, labels in (
        ("generated_mask", mask_stage["details"]["labels"]),
        ("conditioning_mask", mask_stage["details"]["conditioning_labels"]),
    ):
        cast(dict[str, str], details["output_kind_by_role"])[role] = "label_map"
        cast(dict[str, list[int]], details["output_shape_by_role"])[role] = cast(
            list[int], mask_stage["details"]["shape"]
        )
        cast(dict[str, str], details["output_dtype_by_role"])[role] = "int16"
        cast(dict[str, list[dict[str, int | str]]], details["output_labels_by_role"])[role] = (
            output_label_table(cast(list[int], labels))
        )
    details.update(
        {
            "generation_mode": "paired",
            "input": {
                "anatomy": list(options.anatomy),
                "anatomy_size": [
                    {"anatomy": item.anatomy, "scale": item.scale} for item in options.anatomy_size
                ],
            },
            "mask_generation": {
                **mask_stage["details"],
                "random_seed": lineage.mask,
            },
            "seed_lineage": lineage.to_payload(),
            "options": options_payload(options),
            "started_at": started_at.isoformat(timespec="seconds"),
            "elapsed_s": round(time.perf_counter() - started, 4),
        }
    )
    return {"outputs": _posix(outputs), "details": details}


def _run_paired_mask_stage(
    *,
    output: Path,
    mask_weights_dir: Path,
    label_dict_path: Path,
    size_database_path: Path,
    options: _Options,
    seed: int,
) -> RunResult[Path]:
    """Sample a mask with the learned model, post-process it, run structural QC and write it.

    Writes the generated mask (requested anatomy kept, other anatomy folded into
    body 200), the conditioning mask (full post-processed map) and the QC receipt.
    A mask that fails structural QC raises ``ModelExecutionError`` after both masks
    and the receipt are written.
    """

    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    mask_path = output / _GENERATED_MASK
    conditioning_path = output / _CONDITIONING_MASK
    qc_path = output / _MASK_QC
    try:
        controls = prepare_mask_generation_controls(
            anatomy=options.anatomy,
            body_regions=(),
            anatomy_sizes=options.anatomy_size,
            anatomy_size_database_path=size_database_path,
        )
        if controls.anatomy_size is None:
            raise InvalidInputError("paired Path A did not produce an anatomy-size condition")
        resolved_anatomy_size = dict(
            zip(MASK_ANATOMY_SIZE_SLOTS, controls.anatomy_size.tolist(), strict=True)
        )
        mx = cast("Mlx", import_mlx())
        unet_weights, scale_factor = _load_mask_network(mask_weights_dir, "unet", mx)
        vae_weights, _ = _load_mask_network(mask_weights_dir, "vae", mx)
        if scale_factor is None:
            raise AssetNotReadyError(
                "converted paired-mask UNet is missing scale_factor",
                reason=f"missing scale_factor in {mask_weights_dir / 'model.safetensors'}",
                hint=_MASK_WEIGHTS_HINT,
            )
        label_id_map = load_mask_label_id_map(label_dict_path)
        mx.eval(*unet_weights.values(), *vae_weights.values())
        mx.reset_peak_memory()
        _latent, labels = sample_mask_labels_ncdhw(
            anatomy_size=controls.anatomy_size,
            unet_weights=unet_weights,
            vae_weights=vae_weights,
            label_id_map=label_id_map,
            scale_factor=scale_factor,
            constructor=_MASK_UNET_CONSTRUCTOR,
            vae_constructor=PRODUCTION_MASK_VAE_CONSTRUCTOR,
            mx=mx,
            num_inference_steps=options.mask_generation_num_inference_steps or 1000,
            latent_shape=_MASK_LATENT_SHAPE,
            seed=seed,
            decode_roi=options.sliding_window_size or DEFAULT_ROI_SIZE,
            decode_overlap=(
                DEFAULT_OVERLAP
                if options.sliding_window_overlap is None
                else options.sliding_window_overlap
            ),
        )
        raw_labels = np.asarray(labels[0, 0], dtype=np.int16)
        processed_labels = postprocess_mask(
            raw_labels, target_tumor_label=controls.target_tumor_label
        )
        affine = np.diag((*MASK_SPACING_MM, 1.0)).astype(np.float64)
        if options.geometry is not None:
            processed_labels, affine = resample_mask_geometry(
                processed_labels, affine, target=options.geometry
            )
        processed_mlx = cast("ArrayConstructor", mx.array)(
            processed_labels[None, None], dtype=mx.int32
        )
        selected = select_canonical_mask_labels_ncdhw(
            processed_mlx,
            anatomy=options.anatomy,
            mx=mx,
        )
        host_labels = np.asarray(selected[0, 0], dtype=np.int16)
        target_shape = _MASK_SHAPE if options.geometry is None else options.geometry.shape
        target_spacing = MASK_SPACING_MM if options.geometry is None else options.geometry.spacing
        if host_labels.shape != target_shape:
            raise InvalidInputError(
                f"paired mask shape {host_labels.shape} != production {_MASK_SHAPE}"
            )
        _write_geometry_mask(conditioning_path, processed_labels, affine)
        _write_geometry_mask(mask_path, host_labels, affine)
        receipt = validate_write_and_require_maisi_mask(
            qc_path,
            host_labels,
            spacing=target_spacing,
            orientation=MASK_ORIENTATION,
            anatomy=options.anatomy,
        )
        peak = mx.get_peak_memory()
    except (AssetNotReadyError, MissingDependencyError):
        raise
    except _RUNTIME_ERRORS as error:
        raise ModelExecutionError(f"MAISI paired-mask MLX failed: {error}") from error
    return {
        "outputs": {
            "generated_mask": mask_path,
            "conditioning_mask": conditioning_path,
            "mask_qc": qc_path,
        },
        "details": {
            "shape": list(target_shape),
            "spacing_mm": list(target_spacing),
            "orientation": list(MASK_ORIENTATION),
            "labels": np.unique(host_labels).tolist(),
            "conditioning_labels": np.unique(processed_labels).tolist(),
            "resolved_anatomy_size": resolved_anatomy_size,
            "num_inference_steps": options.mask_generation_num_inference_steps or 1000,
            "mask_qc": cast(JsonObject, receipt.to_payload()),
            "elapsed_s": round(time.perf_counter() - started, 4),
            "peak_memory_bytes": peak,
            "mask_weights_path": mask_weights_dir.as_posix(),
        },
    }


def parse_options(
    *,
    generation_mode: str,
    anatomy: Sequence[str],
    anatomy_size: Mapping[str, float] | None,
    seed: int,
    num_inference_steps: int | None,
    mask_generation_num_inference_steps: int | None,
    sliding_window_size: Sequence[int] | None,
    sliding_window_overlap: float | None,
    autoencoder_tp_num_splits: int | None,
) -> _Options:
    """Validate the CT-mode options and return them normalized.

    Anatomy names are resolved to canonical MAISI names; ``anatomy_size`` maps names
    to scales (see ``NvGenerateAnatomySize``). Step counts are 1..1000,
    ``autoencoder_tp_num_splits`` is one of 1, 2, 4, 8, 16, ``sliding_window_size``
    is three positive multiples of 16 (latent voxels) and ``sliding_window_overlap``
    is in [0, 1). ``from_mask`` accepts no anatomy controls; ``paired`` requires an
    anatomy selection.

    Raises:
        InvalidInputError: If any option is invalid.
    """

    if not isinstance(cast(object, generation_mode), str) or generation_mode not in {
        "paired",
        "from_mask",
    }:
        raise InvalidInputError(
            "MLX NV-Generate-CT generation_mode must be paired or from_mask; "
            f"got {generation_mode!r}"
        )
    if isinstance(anatomy, str) or not all(
        isinstance(cast(object, value), str) for value in anatomy
    ):
        raise InvalidInputError("NV-Generate-CT anatomy must be a list of names")
    if any(not value.strip() for value in anatomy):
        raise InvalidInputError("NV-Generate-CT anatomy names must not be empty")
    try:
        normalized = tuple(normalize_maisi_anatomy_name(value) for value in anatomy)
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
    if len(set(normalized)) != len(normalized):
        raise InvalidInputError("NV-Generate-CT anatomy names must be unique")
    sizes = _parse_anatomy_sizes(anatomy_size)
    if len({item.anatomy for item in sizes}) != len(sizes):
        raise InvalidInputError("NV-Generate-CT controllable anatomy names must be unique")
    controlled = tuple(item.anatomy for item in sizes)
    if len(NV_GENERATE_CT_SIZE_CONTROL_PATHOLOGIES.intersection(controlled)) > 1:
        raise InvalidInputError(
            "NV-Generate-CT size control supports at most one pathology per run"
        )
    if controlled and not set(controlled).issubset(normalized):
        raise InvalidInputError("NV-Generate-CT anatomy must include every anatomy-size target")
    if isinstance(cast(object, seed), bool) or not isinstance(cast(object, seed), int) or seed < 0:
        raise InvalidInputError("NV-Generate-CT seed must be a non-negative integer")
    for field, value in (
        ("num_inference_steps", num_inference_steps),
        ("mask_generation_num_inference_steps", mask_generation_num_inference_steps),
        ("autoencoder_tp_num_splits", autoencoder_tp_num_splits),
    ):
        if value is not None and (
            isinstance(cast(object, value), bool) or not isinstance(cast(object, value), int)
        ):
            raise InvalidInputError(f"NV-Generate-CT {field} must be an integer")
        if value is not None and value <= 0:
            raise InvalidInputError(f"NV-Generate-CT {field} must be positive")
    if num_inference_steps is not None and num_inference_steps > 1000:
        raise InvalidInputError("NV-Generate-CT num_inference_steps must be in 1..1000")
    if (
        mask_generation_num_inference_steps is not None
        and mask_generation_num_inference_steps > 1000
    ):
        raise InvalidInputError(
            "NV-Generate-CT mask_generation_num_inference_steps must be in 1..1000"
        )
    if autoencoder_tp_num_splits not in {None, 1, 2, 4, 8, 16}:
        raise InvalidInputError(
            "NV-Generate-CT autoencoder_tp_num_splits must be 1, 2, 4, 8, or 16"
        )
    window: tuple[int, int, int] | None = None
    if sliding_window_size is not None:
        values = tuple(sliding_window_size)
        if len(values) != 3 or any(
            isinstance(cast(object, value), bool) or not isinstance(cast(object, value), int)
            for value in values
        ):
            raise InvalidInputError("NV-Generate-CT sliding_window_size must be three integers")
        if any(value <= 0 or value % 16 for value in values):
            raise InvalidInputError(
                "NV-Generate-CT sliding-window dimensions must be positive multiples of 16"
            )
        window = (values[0], values[1], values[2])
    overlap: float | None = None
    if sliding_window_overlap is not None:
        if isinstance(cast(object, sliding_window_overlap), bool) or not isinstance(
            cast(object, sliding_window_overlap), int | float
        ):
            raise InvalidInputError("NV-Generate-CT sliding_window_overlap must be a number")
        overlap = float(sliding_window_overlap)
        if not 0.0 <= overlap < 1.0:
            raise InvalidInputError(
                "NV-Generate-CT sliding_window_overlap must be at least 0 and less than 1"
            )
    if generation_mode == "from_mask":
        if normalized or sizes:
            raise InvalidInputError(
                "from_mask uses the supplied label geometry and does not accept anatomy controls"
            )
        if mask_generation_num_inference_steps is not None:
            raise InvalidInputError("NV-Generate-CT MLX from_mask does not run mask generation")
    else:
        if not normalized:
            raise InvalidInputError("paired generation requires a nonempty anatomy selection")
    return _Options(
        generation_mode=generation_mode,
        anatomy=normalized,
        anatomy_size=sizes,
        seed=seed,
        num_inference_steps=num_inference_steps,
        mask_generation_num_inference_steps=mask_generation_num_inference_steps,
        sliding_window_size=window,
        sliding_window_overlap=overlap,
        autoencoder_tp_num_splits=autoencoder_tp_num_splits,
    )


def _parse_anatomy_sizes(
    anatomy_size: Mapping[str, float] | None,
) -> tuple[NvGenerateAnatomySize, ...]:
    """Convert the `{anatomy name: size scale}` mapping to validated entries; raises
    InvalidInputError for non-mappings, non-string names or non-numeric scales.
    """
    if anatomy_size is None:
        return ()
    if not isinstance(cast(object, anatomy_size), Mapping):
        raise InvalidInputError(
            'NV-Generate-CT anatomy_size must map names to scales, e.g. {"hepatic tumor": 0.3}'
        )
    sizes: list[NvGenerateAnatomySize] = []
    for name, scale in anatomy_size.items():
        if not isinstance(cast(object, name), str):
            raise InvalidInputError("NV-Generate-CT anatomy_size names must be strings")
        if isinstance(scale, bool) or not isinstance(cast(object, scale), int | float):
            raise InvalidInputError(f"NV-Generate-CT anatomy_size[{name!r}] must be a number")
        try:
            sizes.append(NvGenerateAnatomySize(name, float(scale)))
        except ValueError as exc:
            raise InvalidInputError(str(exc)) from exc
    return tuple(sizes)


def options_payload(options: _Options) -> JsonObject:
    """Return the effective options (defaults filled in) as a JSON-serializable record."""

    return {
        "generation_mode": options.generation_mode,
        "num_output_samples": 1,
        "anatomy": list(options.anatomy),
        "body_regions": list(options.body_regions),
        "output_size": None if options.geometry is None else list(options.geometry.shape),
        "spacing_mm": None if options.geometry is None else list(options.geometry.spacing),
        "controllable_anatomy_size": [[item.anatomy, item.scale] for item in options.anatomy_size],
        "random_seed": options.seed,
        "num_inference_steps": options.num_inference_steps or 30,
        "mask_generation_num_inference_steps": (
            None
            if options.generation_mode == "from_mask"
            else options.mask_generation_num_inference_steps or 1000
        ),
        "sliding_window_size": list(options.sliding_window_size or DEFAULT_ROI_SIZE),
        "sliding_window_overlap": (
            DEFAULT_OVERLAP
            if options.sliding_window_overlap is None
            else options.sliding_window_overlap
        ),
        "cfg_guidance_scale": 0.0,
        "autoencoder_tp_num_splits": options.autoencoder_tp_num_splits
        or int(PRODUCTION_VAE_CONSTRUCTOR["num_splits"]),
        "runtime": "mlx",
    }


def derive_seed_lineage(root_seed: int) -> SeedLineage:
    """Derive stable, distinct non-negative 31-bit child seeds from ``root_seed``.

    Each child is the first four bytes of a SHA-256 over a fixed salt, the root seed
    and the stage name (``mask``, ``ct``, ``pet``), masked to 31 bits.

    Raises:
        InvalidInputError: If ``root_seed`` is not a non-negative integer.
    """

    if isinstance(root_seed, bool) or not isinstance(cast(object, root_seed), int) or root_seed < 0:
        raise InvalidInputError("pipeline root seed must be a non-negative integer")
    children = tuple(_child_seed(root_seed, stage) for stage in ("mask", "ct", "pet"))
    if len(set(children)) != len(children):
        raise RuntimeError("pipeline child-seed derivation produced a collision")
    mask, ct, pet = children
    return SeedLineage(root=root_seed, mask=mask, ct=ct, pet=pet)


def _child_seed(root_seed: int, stage: str) -> int:
    # Fixed salt: a seed deterministically reproduces the same child seeds. The salt
    # keeps the package's pre-rename name so existing seeds still map to the same
    # children; changing it would change every paired run.
    """Derive a deterministic stage-specific seed from a root seed."""
    digest = hashlib.sha256(f"mlx_generate.maisi_pet.v1:{root_seed}:{stage}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def _mask_in_millimeters(image: NiftiImage) -> NiftiImage:
    """Normalize physical coordinates without resampling or changing categorical labels.

    An unspecified spatial unit retains the runner's established mm interpretation.
    Both coded forms and their translations follow the same unit conversion.
    """
    try:
        spatial_unit, temporal_unit = image.header.get_xyzt_units()
        scale = {"unknown": 1.0, "mm": 1.0, "meter": 1000.0, "micron": 0.001}[spatial_unit]
    except KeyError as error:
        raise InvalidInputError(
            "MAISI mask NIfTI units must be mm, meter, micron or unknown"
        ) from error
    if spatial_unit == "mm":
        return image
    affine_mm = np.array(image.affine, dtype=np.float64, copy=True)
    affine_mm[:3] *= scale
    header = image.header.copy()
    header.set_xyzt_units("mm", temporal_unit)
    normalized = nib.Nifti1Image(np.asanyarray(image.dataobj), affine_mm, header)
    for read_form, write_form in (
        (image.get_qform, normalized.set_qform),
        (image.get_sform, normalized.set_sform),
    ):
        form, code = read_form(coded=True)
        form_mm = np.array(affine_mm if form is None else form, copy=True)
        if form is not None:
            form_mm[:3] *= scale
        write_form(form_mm, code=int(code))
    return normalized


def prepare_maisi_mask(source: Path, destination: Path) -> MaskFacts:
    """Validate decoded MAISI labels and save their canonical RAS copy in millimeters.

    Normalize spatial units, including affine translations and coded forms, without
    resampling. Unspecified spatial units retain the established mm interpretation.
    The copy at ``destination`` has its axes permuted or flipped to RAS and an
    integer dtype (exactly integer-valued floating-point labels become uint8).
    Validate the decoded labels before narrowing their dtype.
    Returns the canonical ``shape``, header ``spacing`` (mm) and the sorted unique
    ``labels``.

    Raises:
        InvalidInputError: If the file is unreadable or not 3D, the grid or spacing
            is outside the MAISI mask contract, or the labels are non-integral,
            outside 0..132 plus 200, or lack body label 200.
    """

    try:
        image = nib.load(str(source))
    except (OSError, ValueError, ImageFileError) as exc:
        raise InvalidInputError(f"Could not read MAISI mask NIfTI: {source}") from exc
    if len(image.shape) != 3:
        raise InvalidInputError(f"MAISI mask must be 3D; got shape {image.shape}")
    canonical = nib.as_closest_canonical(_mask_in_millimeters(image))
    shape = (
        int(canonical.shape[0]),
        int(canonical.shape[1]),
        int(canonical.shape[2]),
    )
    zooms = canonical.header.get_zooms()
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    try:
        validate_maisi_geometry(shape, spacing)
    except ValueError as error:
        raise InvalidInputError(str(error)) from error
    data = np.asanyarray(canonical.dataobj)
    # NIfTI scaling can decode integer storage into fractional floating labels.
    floating_labels = np.issubdtype(data.dtype, np.floating)
    if floating_labels:
        if not np.all(np.isfinite(data)):
            raise InvalidInputError("MAISI mask contains non-finite label values")
        rounded = np.rint(data)
        if not np.array_equal(data, rounded):
            raise InvalidInputError(
                "MAISI mask floating-point voxels must be exactly integer-valued"
            )
    elif not np.issubdtype(data.dtype, np.integer):
        raise InvalidInputError(
            f"MAISI mask must contain categorical integer labels; got {data.dtype}"
        )
    labels = tuple(int(value) for value in np.unique(data))
    unknown = tuple(value for value in labels if value not in _VALID_LABELS)
    if unknown:
        raise InvalidInputError(
            "MAISI mask contains labels outside 0..132 plus body label 200: "
            + ", ".join(str(value) for value in unknown[:10])
        )
    if 200 not in labels:
        raise InvalidInputError(
            "MAISI mask is missing body-envelope label 200; add it before generation"
        )
    if floating_labels:
        data = data.astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = canonical.header.copy()
    header.set_data_dtype(data.dtype)
    nib.save(nib.Nifti1Image(data, canonical.affine, header), str(destination))
    return {
        "shape": list(shape),
        "spacing": list(spacing),
        "labels": list(labels),
    }


def validate_output_pair(image_path: Path, label_path: Path) -> PairFacts:
    """Check the CT/label pair before replacing a valid floating label map with uint8."""

    image, _image_data, image_facts = _load_validated_image(image_path)
    label = nib.load(str(label_path))
    if image.shape != label.shape:
        raise RuntimeError(
            f"NV-Generate-CT/label shape mismatch: image={image.shape}, label={label.shape}"
        )
    if not np.allclose(image.affine, label.affine, atol=1e-4):
        raise RuntimeError("NV-Generate-CT and label do not share the same NIfTI affine")
    label_data = np.asanyarray(label.dataobj)
    if not np.issubdtype(label_data.dtype, np.integer):
        if not np.issubdtype(label_data.dtype, np.floating):
            raise RuntimeError(f"MAISI label output must be numeric; got {label_data.dtype}")
        if not np.isfinite(label_data).all():
            raise RuntimeError("MAISI label output contains NaN or infinite voxels")
        rounded = np.rint(label_data)
        if not np.array_equal(label_data, rounded):
            raise RuntimeError("MAISI label output contains non-integral label values")
    labels = {int(value) for value in np.unique(label_data)}
    if not labels.issubset(_VALID_LABELS):
        raise RuntimeError("MAISI label output contains values outside its 132-class vocabulary")
    if not any(value != 0 for value in labels):
        raise RuntimeError("MAISI label output contains no labeled voxels")
    if np.issubdtype(label_data.dtype, np.floating):
        label_data = label_data.astype(np.uint8)
        header = label.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(label_data, label.affine, header), str(label_path))
    return {
        "shape": image_facts["shape"],
        "image_dtype": image_facts["dtype"],
        "label_dtype": str(label_data.dtype),
        "labels": sorted(labels),
    }


def output_label_table(values: Sequence[int]) -> list[dict[str, int | str]]:
    """Name every nonzero MAISI label value."""

    return [{"name": maisi_label(value).name, "value": value} for value in values if value != 0]


def _load_validated_image(image_path: Path) -> tuple[NiftiImage, HostArray, ImageFacts]:
    """Load the output NIfTI and reject non-finite, empty/constant, or all-air (every voxel <= -1000
    HU) data with RuntimeError.
    """
    image = nib.load(str(image_path))
    image_data = np.asanyarray(image.dataobj)
    numeric_image = cast(NumericArray, image_data)
    if not np.isfinite(image_data).all():
        raise RuntimeError("NV-Generate CT contains NaN or infinite voxels")
    if image_data.size == 0 or float(numeric_image.min()) == float(numeric_image.max()):
        raise RuntimeError("NV-Generate CT is empty or constant-valued")
    if int(np.count_nonzero(numeric_image > -1000.0)) == 0:
        raise RuntimeError("NV-Generate-CT contains no non-air voxels")
    return (
        image,
        image_data,
        {
            "shape": [int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            "dtype": str(image_data.dtype),
        },
    )


def load_mask_ncdhw(path: Path) -> tuple[HostArray, tuple[float, float, float], HostArray]:
    """Load a NIfTI-1 label map as int16 ``(1, 1, X, Y, Z)`` without resampling.

    Preserve the stored voxel-axis order. Return spacings in millimeters in that
    order and the float64 millimeter affine alongside the labels. This helper
    checks rank and spacing length; it casts label values without checking their meaning.
    """
    image = _mask_in_millimeters(_load_nifti1(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise InvalidInputError(f"MAISI mask must be 3D; got {data.shape}")
    label = np.asarray(data, dtype=np.int16)[None, None, ...]
    zooms = image.header.get_zooms()
    if len(zooms) < 3:
        raise InvalidInputError(f"MAISI mask needs 3D spacing; got {zooms}")
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return label, spacing, np.asarray(image.affine, dtype=np.float64)


def write_hu_nifti(destination: Path, hu: HostArray, reference: Path) -> None:
    """Write supplied HU values as float32 on the reference affine and header.

    Require shape ``(1, 1, X, Y, Z)`` and remove the batch/channel axes for
    storage. Spatial shape and alignment with the reference are not checked
    here; the caller must supply the intended grid.
    """
    image = _load_nifti1(reference)
    volume = np.asarray(hu, dtype=np.float32)
    if volume.ndim != 5 or volume.shape[0] != 1 or volume.shape[1] != 1:
        raise RuntimeError(f"HU volume must be [1,1,X,Y,Z]; got {volume.shape}")
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(volume[0, 0], image.affine, header), str(destination))


def write_label_nifti(source: Path, destination: Path, label: HostArray) -> None:
    """Write ``label[0, 0]`` using the source NIfTI's affine and copied header.

    The caller is responsible for the expected ``(1, 1, X, Y, Z)`` shape and grid
    alignment; they are not checked here. The copied header controls the saved
    storage dtype, which need not match the input array's dtype.
    """
    reference = _load_nifti1(source)
    output_image = nib.Nifti1Image(label[0, 0], reference.affine, reference.header.copy())
    nib.save(output_image, str(destination))


def write_mask(path: Path, labels: HostArray) -> None:
    """Write a 3D label map as int16 NIfTI with a diagonal affine at 1.5 mm spacing."""
    affine = np.diag((*MASK_SPACING_MM, 1.0)).astype(np.float64)
    image = nib.Nifti1Image(labels, affine)
    image.header.set_data_dtype(np.int16)
    nib.save(image, str(path))


def _load_nifti1(path: Path) -> NiftiImage:
    """Load and validate a NIfTI-1 image."""
    image: object = nib.load(str(path))
    if not isinstance(image, nib.Nifti1Image):
        raise InvalidInputError(f"MAISI mask must be NIfTI-1; got {type(image).__name__}")
    return image


def _load_ct_networks(
    weights_dir: Path, mx: Mlx
) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array], float]:
    """Load VAE, UNet and ControlNet weights checked against the catalog; returns them plus the UNet
    scale factor.

    Raises AssetNotReadyError when the weights do not match the catalog or the scale factor is
    missing.
    """
    catalog = load_catalog()
    try:
        vae, _ = load_network(
            network_file(weights_dir),
            network="vae",
            mx=cast("ArrayFactory[Array, Dtype]", mx),
            catalog=catalog,
        )
        unet, scale_factor = load_network(
            network_file(weights_dir),
            network="unet",
            mx=cast("ArrayFactory[Array, Dtype]", mx),
            catalog=catalog,
        )
        controlnet, _ = load_network(
            network_file(weights_dir),
            network="controlnet",
            mx=cast("ArrayFactory[Array, Dtype]", mx),
            catalog=catalog,
        )
    except InvalidInputError as exc:
        raise AssetNotReadyError(
            f"NV-Generate-CT from-mask weights in {weights_dir} do not match the catalog: {exc}",
            reason=str(exc),
            hint=_CT_WEIGHTS_HINT,
        ) from exc
    if scale_factor is None:
        raise AssetNotReadyError(
            "converted UNet is missing scale_factor",
            reason=f"missing scale_factor in {weights_dir / 'model.safetensors'}",
            hint=_CT_WEIGHTS_HINT,
        )
    return vae, unet, controlnet, scale_factor


def _load_mask_network(
    weights_dir: Path, network: str, mx: Mlx
) -> tuple[dict[str, Array], float | None]:
    """Load one paired-mask network's weights (and its scale factor, if stored); raises
    AssetNotReadyError on a catalog mismatch.
    """
    try:
        return load_mask_network(
            network_file(weights_dir),
            network=network,
            mx=cast("ArrayFactory[Array, Dtype]", mx),
            catalog=load_mask_catalog(),
        )
    except InvalidInputError as exc:
        raise AssetNotReadyError(
            f"NV-Generate-CT mask-generation weights in {weights_dir} do not match the "
            f"catalog: {exc}",
            reason=str(exc),
            hint=_MASK_WEIGHTS_HINT,
        ) from exc


def require_weights_dir(
    path: _PathArg,
    *,
    networks: Sequence[str],
    catalog: Catalog,
    label: str,
    hint: str,
) -> Path:
    """Return the supplied weights folder after expanding '~' and checking its native file.

    Uses ``model.safetensors``, selecting each network's tensors by its
    ``<network>.`` prefix. Each selected
    network must hold exactly the catalog's tensor names (plus ``scale_factor``
    for the UNet); contents and conversion receipts are not otherwise validated here.

    Raises:
        AssetNotReadyError: If the directory or a file is missing, unreadable, or its
            tensor names differ from the catalog.
    """

    from safetensors import SafetensorError, safe_open

    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise AssetNotReadyError(
            f"{label} directory is missing: {directory}",
            reason=f"missing {directory.as_posix()}",
            hint=hint,
        )
    specs = catalog["networks"]
    for network in networks:
        file = network_file(directory)
        if not file.is_file():
            raise AssetNotReadyError(
                f"{label} are missing {file}",
                reason=f"missing {file.as_posix()}",
                hint=hint,
            )
        try:
            with safe_open(file, framework="np") as tensors:
                prefix = network + "."
                names = tensors.keys()
                present = {key.removeprefix(prefix) for key in names if key.startswith(prefix)}
        except (OSError, ValueError, SafetensorError) as exc:
            raise AssetNotReadyError(
                f"{label} file {file} is not a readable tensor archive: {exc}",
                reason=f"unreadable {file.as_posix()}",
                hint=hint,
            ) from exc
        expected = {str(item["name"]) for item in specs[network]["tensors"]}
        if network == "unet":
            expected.add("scale_factor")
        missing = sorted(expected - present)
        unexpected = sorted(present - expected)
        if missing or unexpected:
            problem = (
                f"is missing {missing[0]}" if missing else f"has unexpected tensor {unexpected[0]}"
            )
            raise AssetNotReadyError(
                f"{label} file {file} {problem} ({len(missing)} missing, "
                f"{len(unexpected)} unexpected)",
                reason=f"mismatched {file.as_posix()}",
                hint=hint,
            )
    return directory


def _with_mask_geometry(
    options: _Options,
    body_regions: Sequence[str],
    output_size: Sequence[int] | None,
    spacing_mm: Sequence[float] | None,
) -> _Options:
    """Validate mask-generation geometry and attach it to the options.

    `output_size` (voxels) and `spacing_mm` default to the production mask geometry. `from_mask`
    runs take geometry from the mask and reject `body_regions`, `output_size` and `spacing_mm`. The
    CT field of view in X must be at least 256 mm. Raises InvalidInputError for invalid or
    conflicting values.
    """
    from dataclasses import replace

    from mlx_generate_ctmr.mask_condition import normalize_body_region_indices

    if options.generation_mode == "from_mask":
        if body_regions or output_size is not None or spacing_mm is not None:
            raise InvalidInputError(
                "from_mask takes its geometry from the mask and does not accept body_regions"
            )
        return options
    if options.anatomy_size and body_regions:
        raise InvalidInputError("body_regions cannot be combined with anatomy_size controls")
    try:
        normalize_body_region_indices(body_regions)
        shape = tuple(output_size) if output_size is not None else _MASK_SHAPE
        spacing = tuple(spacing_mm) if spacing_mm is not None else MASK_SPACING_MM
        if len(shape) != 3 or len(spacing) != 3:
            raise ValueError("output_size and spacing_mm must have three values")
        if any(isinstance(v, bool) or not isinstance(cast(object, v), int) for v in shape):
            raise ValueError("output_size must contain integers")
        geometry = MaskGeometry(
            (shape[0], shape[1], shape[2]), (spacing[0], spacing[1], spacing[2])
        )
        validate_maisi_geometry(geometry.shape, geometry.spacing)
        if geometry.shape[0] * geometry.spacing[0] < 256:
            raise ValueError("CT output field of view in X/Y must be at least 256 mm")
        require_supported_production_shape(geometry.shape)
    except ValueError as error:
        raise InvalidInputError(str(error)) from error
    return replace(options, body_regions=tuple(body_regions), geometry=geometry)


def _write_geometry_mask(path: Path, labels: HostArray, affine: HostArray) -> None:
    """Write categorical mask values with the supplied affine."""
    image = nib.Nifti1Image(labels.astype(np.int16), affine)
    image.header.set_data_dtype(np.int16)
    nib.save(image, str(path))


def _run_archive_paired(
    output: Path,
    weights_dir: Path,
    database: Path,
    directory: Path,
    options: _Options,
) -> RunResult[str]:
    """Select and resample an archive mask, then execute the same learned MLX CT action."""

    geometry = options.geometry or MaskGeometry(_MASK_SHAPE, MASK_SPACING_MM)
    lineage = derive_seed_lineage(options.seed)
    try:
        candidates = load_mask_archive(database, directory)
        candidate, labels, affine = select_archive_mask(
            candidates,
            anatomy=options.anatomy,
            body_regions=options.body_regions,
            target=geometry,
            seed=lineage.mask,
        )
    except (OSError, ValueError) as error:
        raise InvalidInputError(f"MAISI archive mask selection failed: {error}") from error
    labels = augment_archive_mask(labels, seed=lineage.mask)
    stage = output.joinpath(*_MASK_STAGE_DIR)
    stage.mkdir(parents=True, exist_ok=True)
    conditioning = stage / _CONDITIONING_MASK
    _write_geometry_mask(conditioning, labels, affine)
    qc = stage / _MASK_QC
    receipt = validate_write_and_require_maisi_mask(
        qc,
        labels,
        spacing=geometry.spacing,
        orientation=MASK_ORIENTATION,
        anatomy=options.anatomy,
    )
    ct = _run_from_mask(
        output=output,
        weights_dir=weights_dir,
        mask_path=conditioning,
        options=options,
        seed=lineage.ct,
        output_anatomy=options.anatomy,
    )
    details: JsonObject = dict(ct["details"])
    details.update(
        {
            "generation_mode": "paired",
            "seed_lineage": lineage.to_payload(),
            "options": options_payload(options),
            "mask_generation": {
                "path": "archive",
                "augmentation": "source_body_zoom_or_tumor_geometry",
                "augmentation_seed": lineage.mask,
                "augmentation_max_attempts": 1000,
                "source_mask": candidate.path.as_posix(),
                "database": database.as_posix(),
                "mask_qc": cast(JsonObject, receipt.to_payload()),
            },
        }
    )
    outputs = {**ct["outputs"], "conditioning_mask": conditioning, "stage_mask_qc": qc}
    return {"outputs": _posix(outputs), "details": details}


def _require_asset_file(path: _PathArg, *, label: str) -> Path:
    """Resolve and validate a required asset file path."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise AssetNotReadyError(
            f"Missing {label}: {resolved}",
            reason=f"missing {resolved.as_posix()}",
            hint="Download it from the pinned NV-Generate-CTMR source or nvidia/NV-Generate-CT.",
        )
    return resolved


def _required_argument(value: _PathArg | None, name: str, mode: str) -> _PathArg:
    """Require a named option for the selected generation mode."""
    if value is None:
        raise InvalidInputError(f"{mode} generation requires {name}")
    return value


def _from_mask_outputs(output: Path) -> dict[str, Path]:
    """Return expected output paths for CT generation from a mask."""
    return {
        "synthetic_image": output / _SYNTHETIC_CT,
        "target_mask": output / _SYNTHETIC_LABEL,
        "prepared_mask": output / _PREPARED_MASK,
        "mask_qc": output / _MASK_QC,
    }


def _paired_outputs(output: Path) -> dict[str, Path]:
    """Return expected output paths for paired mask and CT generation."""
    stage = output.joinpath(*_MASK_STAGE_DIR)
    return {
        **_from_mask_outputs(output),
        "generated_mask": stage / _GENERATED_MASK,
        "conditioning_mask": stage / _CONDITIONING_MASK,
        "stage_mask_qc": stage / _MASK_QC,
    }


def refuse_existing(output: Path, outputs: Mapping[str, Path], *, overwrite: bool) -> None:
    """Raise ``InvalidInputError`` if any planned output exists and ``overwrite`` is false."""
    if overwrite:
        return
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        raise InvalidInputError(
            f"{existing[0]} already exists in {output}; pass overwrite=true to replace outputs"
        )


def _posix(outputs: Mapping[str, Path]) -> dict[str, str]:
    """Convert output paths to absolute POSIX path strings."""
    return {role: Path(path).resolve().as_posix() for role, path in outputs.items()}


__all__ = [
    "CT_MODEL",
    "MR_BRAIN_MODEL",
    "MR_MODEL",
    "SeedLineage",
    "derive_seed_lineage",
    "options_payload",
    "output_label_table",
    "prepare_maisi_mask",
    "run",
    "run_ct",
    "run_mr",
    "run_mr_brain",
    "validate_output_pair",
]


def _bundle_asset(folder: Path, filename: str, subdirectory: str) -> Path | None:
    """Find a bundled auxiliary file without resolving a cache symlink."""
    for path in (folder / filename, folder / subdirectory / filename):
        if path.is_file():
            return path
    return None


def run_ct(
    *,
    output_dir: _PathArg,
    weights_path: _PathArg | None = None,
    modality: str | int | None = None,
    output_size: Sequence[int] | None = None,
    spacing_mm: Sequence[float] | None = None,
    guidance_scale: float | None = None,
    mask_path: _PathArg | None = None,
    anatomy: Sequence[str] | None = None,
    anatomy_size: Mapping[str, float] | None = None,
    mask_weights_path: _PathArg | None = None,
    mask_label_dict_path: _PathArg | None = None,
    anatomy_size_conditions_path: _PathArg | None = None,
    generation_mode: str | None = None,
    body_regions: Sequence[str] = (),
    mask_database_path: _PathArg | None = None,
    mask_directory: _PathArg | None = None,
    seed: int = 0,
    num_inference_steps: int | None = None,
    mask_generation_num_inference_steps: int | None = None,
    sliding_window_size: Sequence[int] | None = None,
    sliding_window_overlap: float | None = None,
    autoencoder_tp_num_splits: int | None = None,
    overwrite: bool = False,
    ct_intensity_qc: bool | None = None,
    ct_qc_statistics_path: _PathArg | None = None,
    ct_qc_max_attempts: int = 2,
    attention_scratch_bytes: int | None = None,
    arithmetic: str = 'fast',
) -> RunResult[str]:
    """Generate ct images; omitted weights are downloaded automatically."""
    return run(
        configuration='ct',
        output_dir=output_dir,
        weights_path=weights_path,
        modality=modality,
        output_size=output_size,
        spacing_mm=spacing_mm,
        guidance_scale=guidance_scale,
        mask_path=mask_path,
        anatomy=anatomy,
        anatomy_size=anatomy_size,
        mask_weights_path=mask_weights_path,
        mask_label_dict_path=mask_label_dict_path,
        anatomy_size_conditions_path=anatomy_size_conditions_path,
        generation_mode=generation_mode,
        body_regions=body_regions,
        mask_database_path=mask_database_path,
        mask_directory=mask_directory,
        seed=seed,
        num_inference_steps=num_inference_steps,
        mask_generation_num_inference_steps=mask_generation_num_inference_steps,
        sliding_window_size=sliding_window_size,
        sliding_window_overlap=sliding_window_overlap,
        autoencoder_tp_num_splits=autoencoder_tp_num_splits,
        overwrite=overwrite,
        ct_intensity_qc=ct_intensity_qc,
        ct_qc_statistics_path=ct_qc_statistics_path,
        ct_qc_max_attempts=ct_qc_max_attempts,
        attention_scratch_bytes=attention_scratch_bytes,
        arithmetic=arithmetic,
    )


def run_mr(
    *,
    output_dir: _PathArg,
    weights_path: _PathArg | None = None,
    modality: str | int | None = None,
    output_size: Sequence[int] | None = None,
    spacing_mm: Sequence[float] | None = None,
    guidance_scale: float | None = None,
    seed: int = 0,
    num_inference_steps: int | None = None,
    sliding_window_size: Sequence[int] | None = None,
    sliding_window_overlap: float | None = None,
    autoencoder_tp_num_splits: int | None = None,
    overwrite: bool = False,
    attention_scratch_bytes: int | None = None,
    arithmetic: str = 'fast',
) -> RunResult[str]:
    """Generate mr images; omitted weights are downloaded automatically."""
    return run(
        configuration='mr',
        output_dir=output_dir,
        weights_path=weights_path,
        modality=modality,
        output_size=output_size,
        spacing_mm=spacing_mm,
        guidance_scale=guidance_scale,
        seed=seed,
        num_inference_steps=num_inference_steps,
        sliding_window_size=sliding_window_size,
        sliding_window_overlap=sliding_window_overlap,
        autoencoder_tp_num_splits=autoencoder_tp_num_splits,
        overwrite=overwrite,
        attention_scratch_bytes=attention_scratch_bytes,
        arithmetic=arithmetic,
    )


def run_mr_brain(
    *,
    output_dir: _PathArg,
    weights_path: _PathArg | None = None,
    modality: str | int | None = None,
    output_size: Sequence[int] | None = None,
    spacing_mm: Sequence[float] | None = None,
    guidance_scale: float | None = None,
    seed: int = 0,
    num_inference_steps: int | None = None,
    sliding_window_size: Sequence[int] | None = None,
    sliding_window_overlap: float | None = None,
    autoencoder_tp_num_splits: int | None = None,
    overwrite: bool = False,
    attention_scratch_bytes: int | None = None,
    arithmetic: str = 'fast',
) -> RunResult[str]:
    """Generate mr-brain images; omitted weights are downloaded automatically."""
    return run(
        configuration='mr-brain',
        output_dir=output_dir,
        weights_path=weights_path,
        modality=modality,
        output_size=output_size,
        spacing_mm=spacing_mm,
        guidance_scale=guidance_scale,
        seed=seed,
        num_inference_steps=num_inference_steps,
        sliding_window_size=sliding_window_size,
        sliding_window_overlap=sliding_window_overlap,
        autoencoder_tp_num_splits=autoencoder_tp_num_splits,
        overwrite=overwrite,
        attention_scratch_bytes=attention_scratch_bytes,
        arithmetic=arithmetic,
    )
