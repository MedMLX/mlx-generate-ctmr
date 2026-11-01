"""Metal model numerics against independent MONAI references recorded on darwin-arm64."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Mlx

import json
from pathlib import Path

import numpy as np
import pytest
from fixture_cases import run_case
from fixture_parameters import load_fixture
from numerical import assert_ct_hu_parity, assert_ct_hu_transform

from mlx_generate_ctmr._arrays import NumericArray

FIXTURES = Path(__file__).parent / "fixtures" / "darwin-arm64"
CASES = (
    "vae",
    "mask_unet",
    "mask_encode",
    "rflow",
    "mask_ddpm",
    "unet",
    "controlnet",
    "mask_decode",
    "from_mask",
    "source_fp32",
)


@pytest.mark.parametrize("kind", CASES)
def test_models_match_monai(kind: str, metal: Mlx):
    """Compare native model fixtures with recorded MONAI outputs."""
    with load_fixture(FIXTURES / f"{kind}.npz") as fixture:
        metadata = json.loads(str(fixture["metadata"]))
        assert metadata["recorded_platform"] == "darwin-arm64"
        assert metadata["monai_version"] == "1.6.0" and metadata["precision"] == "fp32"
        inputs: dict[str, NumericArray] = {
            key.removeprefix("input."): cast(NumericArray, fixture[key])
            for key in fixture.files
            if key.startswith("input.")
        }
        weights: dict[str, NumericArray] = {
            key.removeprefix("weight."): cast(NumericArray, fixture[key])
            for key in fixture.files
            if key.startswith("weight.")
        }
        expected: dict[str, NumericArray] = {
            key.removeprefix("output."): cast(NumericArray, fixture[key])
            for key in fixture.files
            if key.startswith("output.")
        }
    actual = run_case(kind, inputs, weights, metadata["constructor"], metal)
    for key, reference in expected.items():
        if key == "labels":
            _assert_stable_labels(actual, expected, inputs["lookup"])
        elif key == "hu":
            assert kind in {"vae", "from_mask"}
            label = (
                inputs["label"] if kind == "from_mask" else np.ones_like(reference, dtype=np.uint8)
            )
            assert_ct_hu_parity(actual[key], expected["decoded"], label)
            assert_ct_hu_transform(actual[key], actual["decoded"], label)
        elif reference.dtype.kind in "iu":
            np.testing.assert_array_equal(actual[key], reference, err_msg=f"{kind}.{key}")
        else:
            # The port's existing FP32 model budget is unchanged. Schedule
            # coefficients need a tighter absolute bound because they approach zero.
            coefficient = key in {"betas", "alphas", "alphas_cumprod"}
            np.testing.assert_allclose(
                actual[key],
                reference,
                rtol=1e-5 if coefficient else 1e-4,
                atol=1e-8 if coefficient else 1e-5,
                err_msg=f"{kind}.{key}",
            )


def _assert_stable_labels(
    actual: dict[str, NumericArray], expected: dict[str, NumericArray], lookup: NumericArray
):
    """Check label decisions away from uncertain decoder boundaries."""
    reference = expected["tiled_logits"]
    error = 1e-5 + 1e-4 * np.abs(reference)
    winner = np.argmax(reference, axis=1, keepdims=True)
    lower = np.take_along_axis(reference - error, winner, axis=1)
    alternatives = reference + error
    np.put_along_axis(alternatives, winner, -np.inf, axis=1)
    stable = lower > alternatives.max(axis=1, keepdims=True)
    assert stable.any(), "fixture must contain labels with a certified argmax margin"
    np.testing.assert_array_equal(actual["labels"][stable], expected["labels"][stable])
    assert set(np.unique(actual["labels"])) <= set(lookup)

