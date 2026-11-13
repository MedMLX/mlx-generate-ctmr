"""Pinned-source CT intensity QC decisions on HCC-TACE CT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from ct_qc_cases import ct_qc_case

from mlx_generate_ctmr.ct_qc import CtQcVerdict, adjudicate_ct_intensity, load_ct_statistics


@pytest.mark.parametrize(
    ("case", "liver_verdict"),
    [
        ("accept", CtQcVerdict.ACCEPT),
        ("reject", CtQcVerdict.REJECT),
        ("boundary", CtQcVerdict.ACCEPT),
        ("nan", CtQcVerdict.ABSENT),
    ],
)
def test_intensity_adjudication_matches_pinned_source(
    case: str, liver_verdict: CtQcVerdict
) -> None:
    """Compare CT intensity decisions with recorded source values."""
    path = Path(__file__).parent / "fixtures" / "ct_qc_reference.json"
    reference = json.loads(path.read_text())
    assert reference["source_revision"] == "5cb04e82fed71f2fe64a2617ab695be1f5a37fea"
    expected_regions = cast(dict[str, dict[str, object]], reference["cases"][case])
    image, labels = ct_qc_case(case)
    decision = adjudicate_ct_intensity(image, labels, load_ct_statistics())
    liver = next(region for region in decision.regions if region.region.name == "liver")
    assert liver.verdict is liver_verdict
    if case == "boundary":
        assert liver.median == liver.region.high
    for region in decision.regions:
        expected = expected_regions[region.region.name]
        assert (region.verdict is CtQcVerdict.REJECT) == expected["is_outlier"]
        assert region.median == expected["median_value"]
        assert (region.region.low, region.region.high) == (
            expected["low_thresh"],
            expected["high_thresh"],
        )
