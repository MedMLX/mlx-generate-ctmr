"""Structural QC gates for canonical MAISI label masks.

Tumor labels and their host organs are resolved from FDG PET/CT lesion and
organ profiles that are not part of this package; the resolved table is embedded
here as ``TUMOR_HOST_LABELS``. The gates are plausibility screens on the label
map (connectivity, containment, left/right placement); they do not establish
anatomical correctness.

There is deliberately no body-envelope containment gate. NVIDIA's own
candidate conditioning masks (``all_masks_flexible_size_and_spacing_4000``)
routinely place anatomy against background or the in-plane field-of-view edge,
and the upstream pipeline applies no such check, so the gate rejected masks
inside the distribution the MAISI ControlNet is conditioned on.
There is also no whole-foreground connectedness gate: reference conditioning
masks can contain detached components, and upstream does not require a single
component. Requested anatomy is still checked for connectedness.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
from medmlx_core.errors import InvalidInputError
from medmlx_core.maisi import (
    MaisiLabel,
    list_maisi_labels,
    maisi_label,
    validate_maisi_geometry,
)
from numpy.typing import NDArray
from scipy import ndimage as _NDIMAGE

from mlx_generate_ctmr._arrays import HostArray
from mlx_generate_ctmr._records import QcCheckPayload, QcReceiptPayload

BODY_LABEL = 200
_VALID_LABELS = frozenset(label.value for label in list_maisi_labels(include_internal=True))

_CONNECTED_3D = _NDIMAGE.generate_binary_structure(3, 3)
_RIGHT_OF_MIDLINE = frozenset({"liver", "gallbladder"})
_LEFT_OF_MIDLINE = frozenset({"spleen"})
_SUPERIOR_TO_MIDLINE = frozenset({"brain"})
_INFERIOR_TO_MIDLINE = frozenset({"bladder"})
# Tumor label -> host-organ labels, resolved from the pinned FDG lesion
# profile (``host_label_ids``). Never includes body label 200.
TUMOR_HOST_LABELS: Mapping[int, tuple[int, ...]] = MappingProxyType(
    {
        23: (28, 29, 30, 31, 32),
        24: (4,),
        26: (1,),
        27: (62,),
        128: (
            *range(33, 57),
            *range(63, 98),
            114,
            120,
            122,
            127,
        ),
    }
)


class MaisiMaskQcVerdict(StrEnum):
    """One anatomy QC outcome."""

    PASS = "pass"
    FAIL = "fail"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class MaisiMaskQcCheck:
    """One structural MAISI mask gate."""

    check_id: str
    verdict: MaisiMaskQcVerdict
    reason: str
    details: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Freeze a copy of the check details mapping."""
        payload = {} if self.details is None else dict(self.details)
        object.__setattr__(self, "details", MappingProxyType(payload))

    def to_payload(self) -> QcCheckPayload:
        """Serialize a structural mask check and its details."""
        return {
            "check_id": self.check_id,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "details": dict(self.details or {}),
        }


@dataclass(frozen=True, slots=True)
class MaisiMaskQcReceipt:
    """Machine-readable structural QC receipt for one canonical MAISI label array."""

    valid: bool
    requested_anatomy: tuple[str, ...]
    present_labels: tuple[int, ...]
    label_counts: Mapping[int, int]
    checks: tuple[MaisiMaskQcCheck, ...]
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze a copy of the per-label voxel counts."""
        object.__setattr__(self, "label_counts", MappingProxyType(dict(self.label_counts)))
        object.__setattr__(self, "limitations", tuple(self.limitations))

    def to_payload(self) -> QcReceiptPayload:
        """Serialize mask quality decisions, checks, and limitations."""
        return {
            "valid": self.valid,
            "requested_anatomy": list(self.requested_anatomy),
            "present_labels": list(self.present_labels),
            "label_counts": {str(key): value for key, value in self.label_counts.items()},
            "passed": [
                item.check_id for item in self.checks if item.verdict is MaisiMaskQcVerdict.PASS
            ],
            "failed": [
                item.check_id for item in self.checks if item.verdict is MaisiMaskQcVerdict.FAIL
            ],
            "unresolved": [
                item.check_id
                for item in self.checks
                if item.verdict is MaisiMaskQcVerdict.UNRESOLVED
            ],
            "checks": [item.to_payload() for item in self.checks],
            "limitations": list(self.limitations),
        }


def validate_maisi_mask(
    labels: object,
    *,
    spacing: tuple[float, float, float],
    orientation: tuple[str, str, str],
    anatomy: Sequence[str] = (),
) -> MaisiMaskQcReceipt:
    """Run the structural QC gates on a label volume and return a receipt.

    ``labels`` is a 3D array of MAISI ids with axes ``(X, Y, Z)`` in canonical RAS
    orientation; ``spacing`` is the voxel size in mm along those axes and
    ``orientation`` their axis codes, which must be ``("R", "A", "S")``. ``anatomy``
    names the requested labels. Problems are reported as failed checks in the
    receipt rather than raised; ``receipt.valid`` is true only if every check passes.
    Array conversion itself may still raise. Vocabulary membership is checked
    before narrowing label values to int32. The location check interprets axes as
    RAS even if the separate geometry check fails.
    """

    requested_anatomy, anatomy_errors = _resolve_requested_anatomy(anatomy)
    host = np.asarray(labels)
    vocabulary = _vocabulary_check(host)
    geometry = _geometry_check(host, spacing, orientation)
    requested = _requested_anatomy_check(
        vocabulary=vocabulary,
        requested_anatomy=requested_anatomy,
        anatomy_errors=anatomy_errors,
    )
    location = _location_check(
        vocabulary=vocabulary,
        requested_anatomy=requested_anatomy,
    )
    anatomy_structure = _anatomy_structure_check(
        vocabulary=vocabulary,
        requested_anatomy=requested_anatomy,
    )
    containment = _tumor_host_containment_check(vocabulary=vocabulary)
    checks = (
        geometry,
        vocabulary.check,
        requested,
        anatomy_structure,
        location,
        containment,
    )
    valid = all(check.verdict is MaisiMaskQcVerdict.PASS for check in checks)
    limitations = (
        "No cohort-validated organ minimum-volume, shape, or smoothness "
        "thresholds are claimed by this structural receipt.",
        "Whole-foreground connectedness and body-envelope containment of "
        "non-tumor anatomy are not checked. "
        "Tumor labels must remain bounded on all six faces.",
    )
    return MaisiMaskQcReceipt(
        valid=valid,
        requested_anatomy=tuple(label.name for label in requested_anatomy),
        present_labels=vocabulary.present_labels,
        label_counts=vocabulary.label_counts,
        checks=checks,
        limitations=limitations,
    )


def require_valid_maisi_mask(
    labels: object,
    *,
    spacing: tuple[float, float, float],
    orientation: tuple[str, str, str],
    anatomy: Sequence[str] = (),
) -> MaisiMaskQcReceipt:
    """Raise ``InvalidInputError`` when the label array fails structural QC."""

    receipt = validate_maisi_mask(
        labels,
        spacing=spacing,
        orientation=orientation,
        anatomy=anatomy,
    )
    failed = next(
        (check for check in receipt.checks if check.verdict is not MaisiMaskQcVerdict.PASS),
        None,
    )
    if failed is None:
        return receipt
    raise InvalidInputError(f"MAISI anatomy QC failed at {failed.check_id}: {failed.reason}")


def validate_write_and_require_maisi_mask(
    path: Path,
    labels: object,
    *,
    spacing: tuple[float, float, float],
    orientation: tuple[str, str, str],
    anatomy: Sequence[str] = (),
) -> MaisiMaskQcReceipt:
    """Write a structural receipt before returning or raising on its verdict."""

    receipt = validate_maisi_mask(
        labels,
        spacing=spacing,
        orientation=orientation,
        anatomy=anatomy,
    )
    path.write_text(
        json.dumps(receipt.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failed = next(
        (check for check in receipt.checks if check.verdict is not MaisiMaskQcVerdict.PASS),
        None,
    )
    if failed is not None:
        raise InvalidInputError(f"MAISI anatomy QC failed at {failed.check_id}: {failed.reason}")
    return receipt


@dataclass(frozen=True, slots=True)
class _VocabularyFacts:
    check: MaisiMaskQcCheck
    labels: HostArray | None
    present_labels: tuple[int, ...]
    label_counts: Mapping[int, int]


def _resolve_requested_anatomy(
    anatomy: Sequence[str],
) -> tuple[tuple[MaisiLabel, ...], tuple[str, ...]]:
    """Resolve names to MAISI labels; unknown and duplicate names are returned as error strings, not
    raised.
    """
    resolved: list[MaisiLabel] = []
    errors: list[str] = []
    seen: set[str] = set()
    for raw_name in anatomy:
        try:
            label = maisi_label(raw_name)
        except ValueError as error:
            errors.append(str(error))
            continue
        if label.name in seen:
            errors.append(f"Duplicate requested anatomy: {label.name}")
            continue
        seen.add(label.name)
        resolved.append(label)
    return tuple(resolved), tuple(errors)


def _geometry_check(
    labels: HostArray,
    spacing: tuple[float, float, float],
    orientation: tuple[str, str, str],
) -> MaisiMaskQcCheck:
    """Check the mask is 3D, satisfies the MAISI shape/spacing contract, and is canonical RAS."""
    if labels.ndim != 3:
        return _fail(
            "geometry",
            f"MAISI mask must be 3D; got shape {tuple(int(dim) for dim in labels.shape)}",
        )
    try:
        validate_maisi_geometry(
            (int(labels.shape[0]), int(labels.shape[1]), int(labels.shape[2])),
            spacing,
        )
    except (TypeError, ValueError) as error:
        return _fail("geometry", str(error))
    if orientation != ("R", "A", "S"):
        return _fail(
            "geometry",
            f"MAISI structural location QC requires canonical RAS orientation; got {orientation}",
        )
    return _pass(
        "geometry",
        "shape and spacing match the MAISI mask contract",
        shape=[int(dim) for dim in labels.shape],
        spacing=[float(value) for value in spacing],
        orientation=list(orientation),
    )


def _vocabulary_check(labels: HostArray) -> _VocabularyFacts:
    """Check that voxels are finite, integer-valued labels from 0..132 plus body label 200.

    Returns the check plus int32 canonical labels (None when the values cannot be
    interpreted as labels) and per-label voxel counts.
    """
    if labels.ndim != 3:
        return _VocabularyFacts(
            check=_fail("vocabulary", "blocked by invalid geometry"),
            labels=None,
            present_labels=(),
            label_counts={},
        )
    if not np.issubdtype(labels.dtype, np.number):
        return _VocabularyFacts(
            check=_fail(
                "vocabulary",
                f"MAISI mask must contain numeric labels; got {labels.dtype}",
            ),
            labels=None,
            present_labels=(),
            label_counts={},
        )
    if not np.all(np.isfinite(labels)):
        return _VocabularyFacts(
            check=_fail("vocabulary", "MAISI mask contains non-finite label values"),
            labels=None,
            present_labels=(),
            label_counts={},
        )
    if np.issubdtype(labels.dtype, np.integer):
        checked = labels
    else:
        rounded = np.rint(labels)
        if not np.array_equal(labels, rounded):
            return _VocabularyFacts(
                check=_fail(
                    "vocabulary",
                    "MAISI mask floating-point voxels must be exactly integer-valued",
                ),
                labels=None,
                present_labels=(),
                label_counts={},
            )
        checked = rounded
    # Check original values first: narrowing can wrap large integers into valid IDs.
    values, counts = np.unique(checked, return_counts=True)
    present_labels = tuple(int(value) for value in values.tolist())
    label_counts = {
        int(value): int(count)
        for value, count in zip(values.tolist(), counts.tolist(), strict=True)
    }
    unknown = sorted(value for value in present_labels if value not in _VALID_LABELS)
    if unknown:
        limits = np.iinfo(np.int32)
        canonical = (
            checked.astype(np.int32, copy=False)
            if all(limits.min <= value <= limits.max for value in present_labels)
            else None
        )
        return _VocabularyFacts(
            check=_fail(
                "vocabulary",
                "MAISI mask contains labels outside 0..132 plus body label 200",
                unknown_labels=unknown,
            ),
            labels=canonical,
            present_labels=present_labels,
            label_counts=label_counts,
        )
    canonical = checked.astype(np.int32, copy=False)
    if BODY_LABEL not in label_counts:
        return _VocabularyFacts(
            check=_fail(
                "vocabulary",
                "MAISI mask is missing body-envelope label 200; add it before generation",
            ),
            labels=canonical,
            present_labels=present_labels,
            label_counts=label_counts,
        )
    return _VocabularyFacts(
        check=_pass(
            "vocabulary",
            "labels are finite, integer-valued, and drawn from the canonical MAISI vocabulary",
            labels=present_labels,
        ),
        labels=canonical,
        present_labels=present_labels,
        label_counts=label_counts,
    )


def _anatomy_structure_check(
    *,
    vocabulary: _VocabularyFacts,
    requested_anatomy: tuple[MaisiLabel, ...],
) -> MaisiMaskQcCheck:
    """Require each requested anatomy label that is present to be one 26-connected component.

    A label covering the entire foreground also fails. Absent labels are skipped
    here and reported by `_requested_anatomy_check`.
    """
    canonical = vocabulary.labels
    if canonical is None:
        return _fail("anatomy_structure", "blocked by invalid vocabulary")
    foreground_count = int(np.count_nonzero(canonical))
    failures: list[dict[str, object]] = []
    checked = 0
    for label in requested_anatomy:
        if label.kind != "anatomy":
            continue
        mask = canonical == label.value
        voxel_count = int(np.count_nonzero(mask))
        if voxel_count == 0:
            continue
        checked += 1
        _components, component_count = cast(
            tuple[HostArray, int],
            _NDIMAGE.label(mask, structure=_CONNECTED_3D),
        )
        if int(component_count) != 1 or voxel_count >= foreground_count:
            failures.append(
                {
                    "name": label.name,
                    "value": label.value,
                    "voxels": voxel_count,
                    "connected_components": int(component_count),
                }
            )
    if failures:
        return _fail(
            "anatomy_structure",
            "requested anatomy must have nonzero bounded size and one 26-connected component",
            failures=failures,
        )
    return _pass(
        "anatomy_structure",
        "requested anatomy has nonzero bounded size and one 26-connected component",
        checked_labels=checked,
    )


def _requested_anatomy_check(
    *,
    vocabulary: _VocabularyFacts,
    requested_anatomy: tuple[MaisiLabel, ...],
    anatomy_errors: tuple[str, ...],
) -> MaisiMaskQcCheck:
    """Require every requested label to resolve and to be present in the mask."""
    if anatomy_errors:
        return _fail(
            "requested_anatomy",
            "requested anatomy contains invalid or duplicate labels",
            errors=list(anatomy_errors),
        )
    canonical = vocabulary.labels
    if canonical is None:
        return _fail("requested_anatomy", "blocked by invalid vocabulary")
    missing = [
        {"name": label.name, "value": label.value}
        for label in requested_anatomy
        if label.value not in vocabulary.label_counts
    ]
    if missing:
        return _fail(
            "requested_anatomy",
            "requested anatomy is missing from the generated label map",
            missing=missing,
        )
    present = [
        {
            "name": label.name,
            "value": label.value,
            "voxels": vocabulary.label_counts[label.value],
        }
        for label in requested_anatomy
    ]
    del canonical
    return _pass(
        "requested_anatomy",
        "every requested anatomy label is present with nonzero voxels",
        present=present,
    )


def _location_check(
    *,
    vocabulary: _VocabularyFacts,
    requested_anatomy: tuple[MaisiLabel, ...],
) -> MaisiMaskQcCheck:
    """Compare requested anatomy centroids with the foreground bounding-box midpoint.

    Only labels with a left/right/superior/inferior rule (see `_location_expectation`)
    are tested. Centroids and midpoint are in voxel indices of the RAS array.
    """
    canonical = vocabulary.labels
    if canonical is None:
        return _fail("anatomy_location", "blocked by invalid vocabulary")
    foreground = canonical != 0
    if not np.any(foreground):
        return _fail("anatomy_location", "blocked by empty foreground")
    # RAS axes: index 0 grows toward the patient's right, index 2 toward superior,
    # so "left" labels must have a smaller axis-0 centroid than the body midpoint.
    body_bounds = _bounds(foreground)
    body_midpoint = (body_bounds[0] + body_bounds[1]) / 2.0
    failures: list[dict[str, object]] = []
    applied = 0
    for label in requested_anatomy:
        expectation = _location_expectation(label)
        if expectation is None or label.kind != "anatomy":
            continue
        voxels = canonical == label.value
        if not np.any(voxels):
            continue
        centroid = np.argwhere(voxels).mean(axis=0)
        applied += 1
        if not _location_matches(expectation, centroid, body_midpoint):
            failures.append(
                {
                    "name": label.name,
                    "value": label.value,
                    "expectation": expectation,
                    "centroid": [float(value) for value in centroid.tolist()],
                }
            )
    if failures:
        return _fail(
            "anatomy_location",
            "requested anatomy violates a stable location invariant",
            body_midpoint=[float(value) for value in body_midpoint.tolist()],
            failures=failures,
        )
    if applied == 0:
        return _pass(
            "anatomy_location",
            "no requested anatomy labels require a hard-coded location invariant",
        )
    return _pass(
        "anatomy_location",
        "requested anatomy satisfies every applied location invariant",
        applied_rules=applied,
    )


def _tumor_host_containment_check(*, vocabulary: _VocabularyFacts) -> MaisiMaskQcCheck:
    """Require each tumor label present to be enclosed by its pinned host organ.

    The host organ must exist, the tumor must not touch the volume edge, and host
    voxels must be the majority of the tumor's 26-connected dilation ring.
    """
    canonical = vocabulary.labels
    if canonical is None:
        return _fail("tumor_host_containment", "blocked by invalid vocabulary")
    present_tumors = sorted(
        label for label in vocabulary.present_labels if label in TUMOR_HOST_LABELS
    )
    if not present_tumors:
        return _pass(
            "tumor_host_containment",
            "no pinned tumor labels are present in the mask",
        )
    failures: list[dict[str, object]] = []
    for tumor_label in present_tumors:
        host_labels = TUMOR_HOST_LABELS[tumor_label]
        tumor_mask = canonical == tumor_label
        host_mask = np.isin(canonical, host_labels)
        if not np.any(host_mask):
            failures.append(
                {
                    "tumor_label": tumor_label,
                    "host_labels": list(host_labels),
                    "reason": "host organ is absent",
                }
            )
            continue
        if _touches_volume_edge(tumor_mask):
            failures.append(
                {
                    "tumor_label": tumor_label,
                    "host_labels": list(host_labels),
                    "reason": "tumor touches the volume boundary",
                }
            )
            continue
        ring = _NDIMAGE.binary_dilation(tumor_mask, structure=_CONNECTED_3D) & ~tumor_mask
        ring_size = int(np.count_nonzero(ring))
        host_ring_voxels = int(np.count_nonzero(ring & host_mask))
        host_contact_fraction = host_ring_voxels / ring_size if ring_size else 0.0
        leaked = sorted(
            int(value)
            for value in np.unique(canonical[ring])
            if int(value) not in host_labels and int(value) != tumor_label
        )
        if host_ring_voxels <= ring_size - host_ring_voxels:
            failures.append(
                {
                    "tumor_label": tumor_label,
                    "host_labels": list(host_labels),
                    "leaked_labels": leaked,
                    "host_contact_fraction": host_contact_fraction,
                    "reason": "tumor border has no majority contact with the pinned host organ",
                }
            )
    if failures:
        return _fail(
            "tumor_host_containment",
            "pathology labels must stay structurally enclosed by their pinned host organs",
            failures=failures,
        )
    return _pass(
        "tumor_host_containment",
        "every pinned tumor label has majority boundary contact with its host organ",
        tumors=present_tumors,
    )


def _location_expectation(label: MaisiLabel) -> str | None:
    """Return the midline rule for a label (by name prefix or known-name sets), or None if it has
    none.
    """
    if label.name.startswith("left "):
        return "left_of_midline"
    if label.name.startswith("right "):
        return "right_of_midline"
    if label.name in _RIGHT_OF_MIDLINE:
        return "right_of_midline"
    if label.name in _LEFT_OF_MIDLINE:
        return "left_of_midline"
    if label.name in _SUPERIOR_TO_MIDLINE:
        return "superior_to_midline"
    if label.name in _INFERIOR_TO_MIDLINE:
        return "inferior_to_midline"
    return None


def _location_matches(
    expectation: str,
    centroid: HostArray,
    body_midpoint: HostArray,
) -> bool:
    """Test a centroid against a rule; axis 0 is left-right (increasing toward right), axis 2
    inferior-superior.
    """
    if expectation == "left_of_midline":
        return bool(float(centroid[0]) < float(body_midpoint[0]))
    if expectation == "right_of_midline":
        return bool(float(centroid[0]) > float(body_midpoint[0]))
    if expectation == "superior_to_midline":
        return bool(float(centroid[2]) > float(body_midpoint[2]))
    if expectation == "inferior_to_midline":
        return bool(float(centroid[2]) < float(body_midpoint[2]))
    return False


def _bounds(mask: HostArray) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return the minimum and maximum occupied voxel indices."""
    points = np.argwhere(mask)
    return points.min(axis=0), points.max(axis=0)


def _touches_volume_edge(mask: HostArray) -> bool:
    """Return whether a binary mask touches any volume face."""
    return bool(
        np.any(mask[0, :, :])
        or np.any(mask[-1, :, :])
        or np.any(mask[:, 0, :])
        or np.any(mask[:, -1, :])
        or np.any(mask[:, :, 0])
        or np.any(mask[:, :, -1])
    )


def _pass(check_id: str, reason: str, **details: object) -> MaisiMaskQcCheck:
    """Construct a passing mask quality check."""
    return MaisiMaskQcCheck(
        check_id=check_id,
        verdict=MaisiMaskQcVerdict.PASS,
        reason=reason,
        details=details,
    )


def _fail(check_id: str, reason: str, **details: object) -> MaisiMaskQcCheck:
    """Construct a failing mask quality check."""
    return MaisiMaskQcCheck(
        check_id=check_id,
        verdict=MaisiMaskQcVerdict.FAIL,
        reason=reason,
        details=details,
    )


__all__ = [
    "BODY_LABEL",
    "TUMOR_HOST_LABELS",
    "MaisiMaskQcCheck",
    "MaisiMaskQcReceipt",
    "MaisiMaskQcVerdict",
    "require_valid_maisi_mask",
    "validate_maisi_mask",
    "validate_write_and_require_maisi_mask",
]
