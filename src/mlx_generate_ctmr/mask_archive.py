"""Typed MAISI archive selection and host-only categorical geometry actions."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from medmlx_core.maisi import maisi_label

from mlx_generate_ctmr._arrays import Float64Array, HostArray, IntArray
from mlx_generate_ctmr._nifti import nib
from mlx_generate_ctmr.mask_condition import normalize_body_region_indices


@dataclass(frozen=True, slots=True)
class MaskGeometry:
    """One categorical grid, parsed at the API boundary."""

    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Validate positive three-dimensional mask shape and spacing."""
        if len(self.shape) != 3 or len(self.spacing) != 3:
            raise ValueError("mask geometry requires three dimensions and spacings")
        if any(
            isinstance(v, bool) or not isinstance(cast(object, v), int) or v <= 0
            for v in self.shape
        ):
            raise ValueError("mask dimensions must be positive integers")
        if any(
            isinstance(v, bool)
            or not isinstance(cast(object, v), (int, float))
            or not math.isfinite(v)
            or v <= 0
            for v in self.spacing
        ):
            raise ValueError("mask spacing must be positive finite numbers")


@dataclass(frozen=True, slots=True)
class MaskCandidate:
    """Archive mask entity with declared anatomy and optional region extent."""

    path: Path
    geometry: MaskGeometry
    labels: frozenset[int]
    region_extent: tuple[int, int] | None


def _triplet(raw: object, *, integer: bool, field: str) -> tuple[float, float, float]:
    """Parse three finite positive magnitudes, optionally requiring integers."""
    if not isinstance(raw, list):
        raise ValueError(f"mask archive {field} must contain three numbers")
    raw = cast(list[object], raw)
    if len(raw) != 3:
        raise ValueError(f"mask archive {field} must contain three numbers")
    values = raw
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise ValueError(f"mask archive {field} must contain three numbers")
    parsed = tuple(abs(float(cast(float, value))) for value in values)
    if any(not math.isfinite(v) or v <= 0 or (integer and not v.is_integer()) for v in parsed):
        raise ValueError(
            f"mask archive {field} must contain positive {'integers' if integer else 'numbers'}"
        )
    return parsed[0], parsed[1], parsed[2]


def _region_index(raw: object) -> int:
    """Decode a four-element one-hot body-region index."""
    if not isinstance(raw, list):
        raise ValueError("mask archive region indices must be four-element one-hot vectors")
    values = cast(list[object], raw)
    if (
        len(values) != 4
        or values.count(1) != 1
        or any(isinstance(v, bool) or v not in (0, 1) for v in values)
    ):
        raise ValueError("mask archive region indices must be four-element one-hot vectors")
    return values.index(1)


def load_mask_archive(database: Path, mask_directory: Path) -> tuple[MaskCandidate, ...]:
    """Parse the pinned database JSON once; never extract or mutate source assets."""

    if not database.is_file() or not mask_directory.is_dir():
        raise ValueError(
            "mask archive requires an existing database JSON and extracted mask directory"
        )
    raw: object = json.loads(database.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("mask archive database must be a JSON list")
    root = mask_directory.resolve()
    return tuple(_parse_candidate(row, root) for row in cast(list[object], raw))


def _parse_candidate(raw: object, root: Path) -> MaskCandidate:
    """Validate one archive row and resolve its path within the mask directory."""
    if not isinstance(raw, Mapping):
        raise ValueError("mask archive rows must be objects")
    row = cast(Mapping[str, object], raw)
    filename = row.get("pseudo_label_filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError("mask archive row requires pseudo_label_filename")
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError("mask archive filename must stay inside mask_directory")
    shape = tuple(int(v) for v in _triplet(row.get("dim"), integer=True, field="dim"))
    spacing = _triplet(row.get("spacing"), integer=False, field="spacing")
    geometry = MaskGeometry((shape[0], shape[1], shape[2]), spacing)
    labels_raw = row.get("label_list")
    if not isinstance(labels_raw, list):
        raise ValueError("mask archive label_list must be a list")
    labels = cast(list[object], labels_raw)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in labels):
        raise ValueError("mask archive label_list must contain integer label IDs")
    resolved = frozenset(maisi_label(cast(int, v)).value for v in labels)
    extent = None
    if "top_region_index" in row:
        extent = (
            _region_index(row["top_region_index"]),
            _region_index(row.get("bottom_region_index")),
        )
        if extent[0] > extent[1]:
            raise ValueError("mask archive top region cannot exceed bottom region")
    return MaskCandidate(path, geometry, resolved, extent)


def _matches(candidate: MaskCandidate, required: frozenset[int], regions: tuple[int, ...]) -> bool:
    """Check requested anatomy, tumor exclusions, and available body regions."""
    if not required.issubset(candidate.labels):
        return False
    tumors = frozenset(
        maisi_label(name).value
        for name in (
            "lung tumor",
            "pancreatic tumor",
            "hepatic tumor",
            "colon cancer primaries",
            "bone lesion",
        )
    )
    if (candidate.labels & tumors) - required:
        return False
    if candidate.region_extent is None:
        return True
    top, bottom = candidate.region_extent
    return all(top <= region <= bottom for region in regions)


def _distance(candidate: MaskCandidate, target: MaskGeometry) -> float:
    """Score geometric mismatch using physical extent, shape, and spacing."""
    source = candidate.geometry
    return sum(
        abs(sd * ss - td * ts) / 10 + abs(sd - td) / 100 + abs(ss - ts)
        for sd, ss, td, ts in zip(
            source.shape, source.spacing, target.shape, target.spacing, strict=True
        )
    )


def select_archive_mask(
    candidates: Sequence[MaskCandidate],
    *,
    anatomy: Sequence[str],
    body_regions: Sequence[str],
    target: MaskGeometry,
    seed: int,
) -> tuple[MaskCandidate, IntArray, Float64Array]:
    """Pick an archive mask and resample it to ``target``.

    Candidates must contain every requested label, no unrequested tumor label, and
    cover the requested body regions. An exact geometry match is preferred (chosen
    with ``seed``); otherwise candidates that are fewer than 128 voxels smaller
    than the target on every axis are ranked by field-of-view, dimension and spacing
    distance, and the first five that still contain the requested anatomy after
    resampling are shuffled with ``seed`` and one is returned. Returns
    ``(candidate, labels, affine)`` with ``labels`` int16 of shape ``target.shape``.

    Raises:
        ValueError: If no candidate retains the requested anatomy, or a mask file
            disagrees with the database or is not a categorical 3D label volume.
    """

    required = frozenset(maisi_label(name).value for name in anatomy)
    regions = normalize_body_region_indices(body_regions)
    matching = [c for c in candidates if _matches(c, required, regions)]
    exact = [c for c in matching if c.geometry == target]
    ranked = exact or sorted(
        (
            c
            for c in matching
            if all(s > t - 128 for s, t in zip(c.geometry.shape, target.shape, strict=True))
        ),
        key=lambda c: _distance(c, target),
    )
    if exact:
        random.Random(seed).shuffle(ranked)
    surviving: list[tuple[MaskCandidate, IntArray, Float64Array]] = []
    for candidate in ranked:
        source = nib.load(str(candidate.path))
        if tuple(source.shape) != candidate.geometry.shape or not np.allclose(
            source.header.get_zooms()[:3], candidate.geometry.spacing, rtol=1e-4, atol=1e-4
        ):
            raise ValueError(f"archive geometry metadata differs from mask: {candidate.path}")
        image = nib.as_closest_canonical(source)
        data = np.asanyarray(image.dataobj)
        if data.ndim != 3 or not np.isfinite(data).all() or not np.array_equal(data, np.rint(data)):
            raise ValueError(
                f"archive mask must contain 3D finite categorical labels: {candidate.path}"
            )
        for value in np.unique(data):
            maisi_label(int(value))
        labels = np.asarray(data, dtype=np.int16)
        resampled, affine = resample_mask_geometry(labels, np.asarray(image.affine), target=target)
        if required.issubset(int(v) for v in np.unique(resampled)):
            surviving.append((candidate, resampled, affine))
        if len(surviving) >= (1 if exact else 5):
            break
    if not surviving:
        raise ValueError("no archive mask retains requested anatomy after target spacing and crop")
    random.Random(seed).shuffle(surviving)
    return surviving[0]


def resample_mask_geometry(
    labels: HostArray,
    affine: Float64Array,
    *,
    target: MaskGeometry,
) -> tuple[IntArray, Float64Array]:
    """Resample a label volume to ``target`` spacing, then center-crop or pad to its shape.

    Reimplements MONAI's nearest-neighbour ``Spacing`` followed by a centered
    ``ResizeWithPadOrCrop`` without Torch. ``labels`` is a 3D array with a 4x4
    voxel-to-world ``affine``; returns ``(int16 labels of shape target.shape, new
    affine)``. Nearest-neighbour samples outside the source use the nearest edge
    label; padding after resampling uses label 0. The rounded resampling shape
    need not extend to every input voxel center.

    Raises:
        ValueError: If ``labels`` is not 3D or ``affine`` is not 4x4.
    """

    if labels.ndim != 3 or affine.shape != (4, 4):
        raise ValueError("mask resampling requires 3D labels and a 4x4 affine")
    rotation_scale = affine[:3, :3]
    zoom_shear = np.linalg.cholesky(rotation_scale.T @ rotation_scale).T
    output_affine = np.eye(4, dtype=np.float64)
    original_spacing = np.linalg.norm(rotation_scale, axis=0)
    spacing = np.where(
        np.abs(original_spacing - target.spacing) <= 1e-3, original_spacing, target.spacing
    )
    output_affine[:3, :3] = rotation_scale @ np.linalg.inv(zoom_shear) @ np.diag(spacing)
    coordinates = np.asarray(
        np.meshgrid(*((0, n - 1) for n in labels.shape), indexing="ij")
    ).reshape(3, -1)
    corners = np.concatenate((coordinates, np.ones((1, 8))))
    output_corners = np.linalg.solve(output_affine, affine) @ corners
    output_shape = np.round(np.ptp(output_corners[:3], axis=1) + 1).astype(int)
    world_corners = affine @ corners
    offset = None
    for index in range(8):
        if np.allclose(
            np.min(output_corners[:3] - output_corners[:3, index : index + 1], axis=1), 0, rtol=1e-3
        ):
            offset = world_corners[:3, index]
            break
    if offset is None:
        offset = (
            rotation_scale @ (np.asarray(labels.shape) / 2)
            + affine[:3, 3]
            - output_affine[:3, :3] @ (output_shape / 2)
        )
    output_affine[:3, 3] = offset
    transform = np.asarray(np.linalg.solve(affine, output_affine), dtype=np.float64)
    resampled = _resample_nearest(labels, transform, tuple(int(v) for v in output_shape))
    result = np.zeros(target.shape, dtype=np.int16)
    start = np.maximum(output_shape // 2 - np.asarray(target.shape) // 2, 0)
    padding = np.maximum((np.asarray(target.shape) - output_shape) // 2, 0)
    extent = np.minimum(output_shape, target.shape)
    source_slice = tuple(slice(int(s), int(s + n)) for s, n in zip(start, extent, strict=True))
    target_slice = tuple(slice(int(s), int(s + n)) for s, n in zip(padding, extent, strict=True))
    result[target_slice] = resampled[source_slice]
    output_affine[:3, 3] += output_affine[:3, :3] @ (start - padding)
    return result, output_affine


def _resample_nearest(
    labels: HostArray,
    transform: Float64Array,
    shape: tuple[int, ...],
) -> IntArray:
    """Torch grid_sample nearest uses round-to-even; stream host coordinate slabs."""

    output = np.empty(shape, dtype=np.int16)
    x = np.arange(shape[0], dtype=np.float64)[:, None, None]
    y = np.arange(shape[1], dtype=np.float64)[None, :, None]
    for start in range(0, shape[2], 16):
        stop = min(start + 16, shape[2])
        z = np.arange(start, stop, dtype=np.float64)[None, None, :]
        indices = tuple(
            np.clip(
                np.rint(
                    transform[axis, 0] * x
                    + transform[axis, 1] * y
                    + transform[axis, 2] * z
                    + transform[axis, 3]
                ),
                0,
                labels.shape[axis] - 1,
            ).astype(np.int64)
            for axis in range(3)
        )
        output[:, :, start:stop] = labels[indices]
    return output
