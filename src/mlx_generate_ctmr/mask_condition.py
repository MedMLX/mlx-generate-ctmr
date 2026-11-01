"""Strict paired-mask anatomy controls for MLX MAISI."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, HostArray, Mlx

import numpy as np
from medmlx_core.layout import require_ncdhw
from medmlx_core.maisi import (
    NV_GENERATE_CT_SIZE_CONTROL_PATHOLOGIES,
    MaisiLabel,
    NvGenerateAnatomySize,
    maisi_label,
)

BODY_LABEL = maisi_label("body").value
MASK_ANATOMY_SIZE_SLOTS: tuple[str, ...] = (
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "colon",
    "lung tumor",
    "pancreatic tumor",
    "hepatic tumor",
    "colon cancer primaries",
    "bone lesion",
)
_MASK_TUMOR_LABEL_ORDER = tuple(
    maisi_label(name).value
    for name in (
        "lung tumor",
        "pancreatic tumor",
        "hepatic tumor",
        "colon cancer primaries",
        "bone lesion",
    )
)

_ANATOMY_SIZE_INDEX = {name: index for index, name in enumerate(MASK_ANATOMY_SIZE_SLOTS)}
_BODY_REGION_INDEX = {
    "head": 0,
    "chest": 1,
    "thorax": 1,
    "chest/thorax": 1,
    "abdomen": 2,
    "pelvis": 3,
    "lower": 3,
    "pelvis/lower": 3,
}


@dataclass(frozen=True, slots=True)
class MaskGenerationControls:
    """Validated controls for one paired-mask run.

    ``anatomy`` is the requested labels; ``anatomy_size`` the 10-element size
    condition (``None`` when no sizes were given, i.e. the archive path);
    ``body_region_indices`` the requested archive body regions; and
    ``target_tumor_label`` the last tumor id whose condition is not -1 in
    source slot order, including values inherited from the database row, if any.
    """

    anatomy: tuple[MaisiLabel, ...]
    anatomy_size: HostArray | None
    body_region_indices: tuple[int, ...]
    target_tumor_label: int | None


def prepare_mask_generation_controls(
    *,
    anatomy: Sequence[str],
    body_regions: Sequence[str] = (),
    anatomy_sizes: Sequence[NvGenerateAnatomySize] = (),
    anatomy_size_database_path: Path | None = None,
) -> MaskGenerationControls:
    """Validate paired-mask controls and, if sizes are given, build the size condition.

    Anatomy and size targets must be unique, sizes must name requested anatomy, at
    most one tumor may be explicitly size-controlled, and body regions cannot be
    combined with sizes. Sizes need ``anatomy_size_database_path`` (the pinned
    anatomy-size JSON).

    Raises:
        ValueError: If any of those conditions fails or the database is unusable.
    """

    labels = _resolve_anatomy(anatomy)
    region_indices = normalize_body_region_indices(body_regions)
    sizes = tuple(anatomy_sizes)
    size_names = tuple(item.anatomy for item in sizes)
    if len(set(size_names)) != len(size_names):
        raise ValueError("paired-mask anatomy-size targets must be unique")
    if len(NV_GENERATE_CT_SIZE_CONTROL_PATHOLOGIES.intersection(size_names)) > 1:
        raise ValueError("paired-mask anatomy sizes support at most one tumor")
    anatomy_names = tuple(label.name for label in labels)
    if sizes and not set(size_names).issubset(anatomy_names):
        raise ValueError("paired-mask anatomy must include every anatomy-size target")
    if sizes and region_indices:
        raise ValueError(
            "paired-mask body-region controls cannot be combined with anatomy-size controls"
        )

    condition: HostArray | None = None
    target_tumor_label: int | None = None
    if sizes:
        if anatomy_size_database_path is None:
            raise ValueError("paired-mask anatomy-size controls require the anatomy-size database")
        condition = prepare_anatomy_size_condition(
            sizes,
            database_path=anatomy_size_database_path,
        )
        for tumor_label, tumor_size in zip(
            _MASK_TUMOR_LABEL_ORDER,
            condition[5:10].tolist(),
            strict=True,
        ):
            if float(tumor_size) != -1.0:
                target_tumor_label = tumor_label

    return MaskGenerationControls(
        anatomy=labels,
        anatomy_size=condition,
        body_region_indices=region_indices,
        target_tumor_label=target_tumor_label,
    )


def prepare_anatomy_size_condition(
    anatomy_sizes: Sequence[NvGenerateAnatomySize],
    *,
    database_path: Path,
) -> HostArray:
    """Build the 10-element size condition from the requested sizes.

    Starts from the database row nearest to the requested values (sum of absolute
    differences over the specified slots), then overwrites the specified slots with
    the requested values. Slot order is ``MASK_ANATOMY_SIZE_SLOTS``.
    """

    controls = tuple(anatomy_sizes)
    if not controls:
        raise ValueError("paired-mask anatomy-size controls must not be empty")
    names = tuple(item.anatomy for item in controls)
    if len(set(names)) != len(names):
        raise ValueError("paired-mask anatomy-size targets must be unique")
    provided: list[float | None] = [None] * len(MASK_ANATOMY_SIZE_SLOTS)
    for item in controls:
        slot = _ANATOMY_SIZE_INDEX.get(item.anatomy)
        if slot is None:
            raise ValueError(f"unsupported paired-mask anatomy-size target: {item.anatomy}")
        provided[slot] = float(item.scale)

    rows = _load_json_list(database_path, label="anatomy-size database")
    conditions = tuple(_parse_anatomy_size_row(row, index=index) for index, row in enumerate(rows))
    if not conditions:
        raise ValueError("paired-mask anatomy-size database is empty")

    def _distance(condition: HostArray) -> float:
        """Sum absolute differences for the explicitly requested size slots."""
        return sum(
            abs(float(condition[index]) - value)
            for index, value in enumerate(provided)
            if value is not None
        )

    nearest = min(conditions, key=_distance).copy()
    for index, value in enumerate(provided):
        if value is not None:
            nearest[index] = value
    return nearest


def normalize_body_region_indices(body_regions: Sequence[str]) -> tuple[int, ...]:
    """Map pinned region names to ordered MAISI region indices."""

    indices: list[int] = []
    for region in body_regions:
        normalized = " ".join(region.split()).casefold()
        index = _BODY_REGION_INDEX.get(normalized)
        if index is None:
            choices = ", ".join(_BODY_REGION_INDEX)
            raise ValueError(
                f"unsupported paired-mask body region {region!r}; expected one of: {choices}"
            )
        if index in indices:
            raise ValueError("paired-mask body-region controls must be unique")
        indices.append(index)
    return tuple(indices)


def select_canonical_mask_labels_ncdhw(
    labels: Array,
    *,
    anatomy: Sequence[str],
    mx: Mlx,
) -> Array:
    """Keep requested IDs and fold other positive anatomy into body ID 200.

    NVIDIA's final sequential filter drops 200 unless ``body`` is requested.
    The explicit-MLX CT stage requires body label 200, so this implementation
    preserves background 0 and maps discarded positive anatomy to body 200.
    """

    require_ncdhw(labels, name="paired-mask labels")
    resolved = _resolve_anatomy(anatomy)
    keep = labels == BODY_LABEL
    for label in resolved:
        if label.value != BODY_LABEL:
            keep = mx.logical_or(keep, labels == label.value)
    body_or_background = mx.where(
        labels > 0,
        mx.array(BODY_LABEL, dtype=mx.int32),
        mx.zeros_like(labels),
    )
    selected = mx.where(keep, labels, body_or_background).astype(mx.int32)
    mx.eval(selected)
    return selected


def _resolve_anatomy(anatomy: Sequence[str]) -> tuple[MaisiLabel, ...]:
    """Resolve unique requested anatomy names to label records."""
    if not anatomy:
        raise ValueError("paired-mask anatomy selection must not be empty")
    labels = tuple(maisi_label(name) for name in anatomy)
    names = tuple(label.name for label in labels)
    if len(set(names)) != len(names):
        raise ValueError("paired-mask anatomy selections must be unique")
    return labels


def _load_json_list(path: Path, *, label: str) -> list[object]:
    """Load a JSON list or report an invalid paired-mask asset."""
    if not path.is_file():
        raise ValueError(f"paired-mask {label} is missing: {path}")
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid paired-mask {label}: {path}") from error
    if not isinstance(raw, list):
        raise ValueError(f"paired-mask {label} must be a JSON list")
    return cast(list[object], raw)


def _parse_anatomy_size_row(row: object, *, index: int) -> HostArray:
    """Parse a ten-slot finite anatomy-size row and its tumor condition."""
    if not isinstance(row, Mapping):
        raise ValueError(f"anatomy-size database row {index} must be an object")
    raw = cast(Mapping[object, object], row).get("organ_size")
    if not isinstance(raw, list):
        raise ValueError(f"anatomy-size database row {index} must contain 10 organ_size values")
    raw_values = cast(list[object], raw)
    if len(raw_values) != len(MASK_ANATOMY_SIZE_SLOTS):
        raise ValueError(f"anatomy-size database row {index} must contain 10 organ_size values")
    values = np.asarray(
        [_finite_number(value, label=f"anatomy-size database row {index}") for value in raw_values],
        dtype=np.float32,
    )
    if int(np.count_nonzero(values[5:10] != -1.0)) > 1:
        raise ValueError(f"anatomy-size database row {index} contains more than one tumor target")
    return values


def _finite_number(value: object, *, label: str) -> float:
    """Require a finite numeric anatomy-size value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must contain finite numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must contain finite numbers")
    return result
