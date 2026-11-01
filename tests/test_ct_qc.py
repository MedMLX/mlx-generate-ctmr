"""Direct pinned-source CT intensity QC median fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from mlx_generate_ctmr.ct_qc import CtQcVerdict, adjudicate_ct_intensity, load_ct_statistics


@pytest.mark.parametrize("case", ["accept", "reject", "boundary", "nan"])
def test_intensity_adjudication_matches_pinned_source(case: str) -> None:
    """Compare CT intensity decisions with recorded source values."""
    path = Path(__file__).parent / "fixtures" / f"ct_qc_{case}.npz"
    with np.load(path, allow_pickle=False) as fixture:
        assert fixture["source_revision"][0] == "5cb04e82fed71f2fe64a2617ab695be1f5a37fea"
        decision = adjudicate_ct_intensity(
            fixture["image"], fixture["labels"], load_ct_statistics()
        )
        reference = cast(dict[str, dict[str, object]], json.loads(str(fixture["expected"][0])))
    for region in decision.regions:
        expected = reference[region.region.name]
        assert (region.verdict is CtQcVerdict.REJECT) == expected["is_outlier"]
        assert region.median == expected["median_value"]
        assert (region.region.low, region.region.high) == (
            expected["low_thresh"],
            expected["high_thresh"],
        )
