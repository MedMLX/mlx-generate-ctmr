"""Per-region median-HU check for generated CT, with bounded regeneration.

For each configured region (a set of MAISI label ids) with usable samples, the
median HU is compared against ``[low, high]`` from upstream median statistics;
absent regions do not reject. This is a coarse outlier screen on seeded output,
not a measure of image quality or clinical realism.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import numpy as np
from medmlx_core.errors import ModelExecutionError

from mlx_generate_ctmr._arrays import HostArray, RealArray
from mlx_generate_ctmr._nifti import nib
from mlx_generate_ctmr._records import JsonObject, RunResult


class CtQcVerdict(StrEnum):
    """Per-region or overall CT intensity outcome; ``ABSENT`` regions do not reject."""

    ACCEPT = "accept"
    REJECT = "reject"
    ABSENT = "absent"


@dataclass(frozen=True, slots=True)
class CtIntensityRegion:
    """One QC region: label ids and the inclusive HU interval their median must lie in."""

    name: str
    labels: tuple[int, ...]
    low: float
    high: float

    def __post_init__(self) -> None:
        """Validate region labels and finite intensity bounds."""
        if (
            not self.name
            or not math.isfinite(self.low)
            or not math.isfinite(self.high)
            or self.low > self.high
        ):
            raise ValueError("CT intensity region requires a name and ordered finite thresholds")


@dataclass(frozen=True, slots=True)
class CtIntensityStatistics:
    """The regions to check plus where they were loaded from and the upstream revision."""

    regions: tuple[CtIntensityRegion, ...]
    source: str
    source_revision: str


@dataclass(frozen=True, slots=True)
class CtRegionDecision:
    """Outcome for one region; ``median`` is None and ``voxels`` 0 when the region is absent."""

    region: CtIntensityRegion
    verdict: CtQcVerdict
    median: float | None
    voxels: int


@dataclass(frozen=True, slots=True)
class CtIntensityDecision:
    """Overall verdict (reject if any region is rejected) and the per-region decisions."""

    verdict: CtQcVerdict
    regions: tuple[CtRegionDecision, ...]
    statistics: CtIntensityStatistics

    def to_payload(self) -> JsonObject:
        """Serialize region-level CT intensity decisions."""
        return {
            "verdict": self.verdict.value,
            "statistics": {
                "source": self.statistics.source,
                "source_revision": self.statistics.source_revision,
            },
            "regions": [
                {
                    "name": d.region.name,
                    "labels": list(d.region.labels),
                    "verdict": d.verdict.value,
                    "median": d.median,
                    "voxels": d.voxels,
                    "low": d.region.low,
                    "high": d.region.high,
                }
                for d in self.regions
            ],
        }


def _number(row: Mapping[str, object], key: str) -> float:
    """Parse a finite numeric value from a CT statistics row."""
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"CT statistics {key} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"CT statistics {key} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"CT statistics {key} must be a finite number")
    return number


def load_ct_statistics(path: Path | None = None) -> CtIntensityStatistics:
    """Load the region statistics from ``path`` (default: the packaged JSON).

    Each region's bounds are widened to ``min(sigma_6_low, percentile_0_5)`` and
    ``max(sigma_6_high, percentile_99_5)``, except bone, whose upper bound is fixed
    at 1000 HU. The returned ``source`` is the resolved file path.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file is malformed or has no unique, named regions.
    """

    resource = path or Path(__file__).parent / "data" / "ct_intensity_qc.json"
    contents = resource.read_bytes()
    raw: object = json.loads(contents)
    if not isinstance(raw, Mapping):
        raise ValueError("CT statistics must be an object")
    payload = cast(Mapping[str, object], raw)
    rows = payload.get("regions")
    revision = payload.get("source_revision")
    if not isinstance(rows, list) or not isinstance(revision, str):
        raise ValueError("CT statistics require regions and source_revision")
    regions = tuple(_parse_region(row) for row in cast(list[object], rows))
    if not regions or len({r.name for r in regions}) != len(regions):
        raise ValueError("CT statistics require nonempty unique regions")
    return CtIntensityStatistics(regions, str(resource.resolve()), revision)


def _parse_region(raw: object) -> CtIntensityRegion:
    """Parse a CT region and its widened HU acceptance bounds."""
    if not isinstance(raw, Mapping):
        raise ValueError("CT intensity region must be an object")
    row = cast(Mapping[str, object], raw)
    name, labels_raw = row.get("name"), row.get("labels")
    if not isinstance(name, str) or not isinstance(labels_raw, list):
        raise ValueError("CT intensity region requires name and labels")
    labels = cast(list[object], labels_raw)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in labels):
        raise ValueError("CT intensity labels must be integers")
    low = min(_number(row, "sigma_6_low"), _number(row, "percentile_0_5"))
    high = max(_number(row, "sigma_6_high"), _number(row, "percentile_99_5"))
    return CtIntensityRegion(
        name, tuple(cast(int, v) for v in labels), low, 1000.0 if name == "bone" else high
    )


def adjudicate_ct_intensity(
    image: HostArray, labels: HostArray, statistics: CtIntensityStatistics
) -> CtIntensityDecision:
    """Compare each region's median HU with its bounds, following upstream ``is_outlier``.

    ``image`` (HU) and ``labels`` have the same shape. NaN voxels are excluded; a
    region with no remaining voxels is ``ABSENT`` and does not cause rejection;
    bounds are inclusive. Infinite values are not removed: an infinite median is
    rejected, but a NaN median produced by mixing negative and positive infinity
    passes the bound comparisons. Callers that require finite data must validate it
    separately; the public CT runner does so before this QC step. The overall
    verdict is ``REJECT`` if any region is rejected.

    Raises:
        ValueError: If ``image`` and ``labels`` differ in shape.
    """

    if image.shape != labels.shape:
        raise ValueError("CT intensity QC image and label shapes must match")
    decisions: list[CtRegionDecision] = []
    for region in statistics.regions:
        pixels = cast(RealArray, image[np.isin(labels, region.labels)])
        pixels = pixels[~np.isnan(pixels)]
        median = None if pixels.size == 0 else float(np.median(pixels))
        verdict = (
            CtQcVerdict.ABSENT
            if median is None
            else (
                CtQcVerdict.REJECT
                if median < region.low or median > region.high
                else CtQcVerdict.ACCEPT
            )
        )
        decisions.append(CtRegionDecision(region, verdict, median, int(pixels.size)))
    verdict = (
        CtQcVerdict.REJECT
        if any(d.verdict is CtQcVerdict.REJECT for d in decisions)
        else CtQcVerdict.ACCEPT
    )
    return CtIntensityDecision(verdict, tuple(decisions), statistics)


def attempt_directory(output: Path, index: int) -> Path:
    """Return the output folder of zero-based QC attempt ``index``: ``stages/ct_qc/attempt-N``."""
    return output / "stages" / "ct_qc" / f"attempt-{index + 1}"


def generate_with_ct_qc(
    generate: Callable[[Path, int], RunResult[str]],
    *,
    output: Path,
    seed: int,
    statistics: CtIntensityStatistics,
    max_attempts: int,
) -> RunResult[str]:
    """Generate until a result passes the HU check, saving each completed QC decision.

    ``generate(directory, seed)`` runs one attempt and must return a result whose
    outputs include ``synthetic_image`` (HU NIfTI) and ``prepared_mask`` (label
    NIfTI on the same grid). Array shapes are checked, but this helper does not
    compare their affines or check CT finiteness. The first attempt uses ``seed``;
    later attempts use seeds derived from it (31-bit, from a SHA-256 of a fixed
    salt, ``seed`` and the attempt index). Receipts go to
    ``output/stages/ct_qc/attempt-N/ct_intensity_qc.json``.
    The first accepted result is returned, with the QC record added to its details.

    Raises:
        ValueError: If ``max_attempts`` is boolean or less than 1.
        TypeError: If ``max_attempts`` cannot be used as a ``range`` bound.
        ModelExecutionError: If every attempt is rejected.
    """

    if isinstance(max_attempts, bool) or max_attempts < 1:
        raise ValueError("CT QC max_attempts must be positive")
    attempts: list[JsonObject] = []
    for index in range(max_attempts):
        # The salt keeps the package's pre-rename name so attempt seeds stay stable.
        attempt_seed = (
            seed
            if index == 0
            else int.from_bytes(
                hashlib.sha256(f"mlx-generate.ct-qc.v1:{seed}:{index}".encode()).digest()[:4], "big"
            )
            & 0x7FFFFFFF
        )
        directory = attempt_directory(output, index)
        result = generate(directory, attempt_seed)
        image = nib.load(result["outputs"]["synthetic_image"])
        label_path = result["outputs"]["prepared_mask"]
        labels = nib.load(label_path)
        decision = adjudicate_ct_intensity(
            np.asanyarray(image.dataobj), np.asanyarray(labels.dataobj), statistics
        )
        payload = decision.to_payload()
        payload.update({"attempt": index + 1, "seed": attempt_seed})
        receipt = directory / "ct_intensity_qc.json"
        receipt.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        attempts.append(payload)
        if decision.verdict is CtQcVerdict.ACCEPT:
            result["details"]["generation_seed"] = attempt_seed
            result["details"]["requested_seed"] = seed
            result["details"]["seed"] = seed
            result["outputs"]["ct_intensity_qc"] = str(receipt.resolve())
            result["details"]["ct_intensity_qc"] = {
                "verdict": "accept",
                "attempts": attempts,
                "exhaustion_policy": "raise",
            }
            return result
    raise ModelExecutionError(
        f"CT intensity QC rejected all {max_attempts} attempts; "
        f"decisions are in {output / 'stages' / 'ct_qc'}"
    )
