"""CT runner: request rejection before weight download and Metal inference end to end."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from _contracts import WeightContract
from _native_fixture import write_native
from fixture_parameters import load_fixture
from medmlx_core import maisi
from medmlx_core.errors import InvalidInputError

from mlx_generate_ctmr import from_mask, maisi_mlx_weights, medmlx
from mlx_generate_ctmr._arrays import FloatArray, HostArray
from mlx_generate_ctmr._nifti import nib
from mlx_generate_ctmr._records import Catalog, TensorSpec

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Mlx

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "darwin-arm64"


@pytest.mark.parametrize(
    ("runner", "controls", "existing"),
    [
        ("run_ct", {"mask_path": "absent.nii.gz"}, None),
        ("run_ct", {"anatomy": ["liver"], "num_inference_steps": 0}, None),
        ("run_mr", {"modality": "ct"}, None),
        ("run_mr_brain", {"overwrite": "false"}, None),
        ("run_ct", {"anatomy": ["liver"], "ct_qc_statistics_path": "absent.json"}, None),
        ("run_ct", {"mask_path": "mask.nii.gz"}, "stages/ct_qc/attempt-2/synthetic_ct.nii.gz"),
    ],
    ids=["path", "range", "choice", "boolean", "qc-statistics", "output-conflict"],
)
def test_invalid_arguments_fail_before_downloading_weights(
    runner: str,
    controls: dict[str, object],
    existing: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject every invalid argument and output conflict before any weight download."""
    import medmlx_core.weights as managed_weights

    def download(model_id: str) -> Path:
        """Fail the test: weights must not download for an invalid request."""
        raise AssertionError(f"downloaded {model_id} before checking the arguments")

    monkeypatch.setattr(managed_weights, "weights_dir", download)
    (tmp_path / "mask.nii.gz").touch()
    output = tmp_path / "out"
    if existing is not None:
        (output / existing).parent.mkdir(parents=True)
        (output / existing).touch()
    arguments = {
        key: tmp_path / str(value) if key.endswith("_path") else value
        for key, value in controls.items()
    }
    with pytest.raises(InvalidInputError):
        cast("Callable[..., object]", getattr(medmlx, runner))(output_dir=output, **arguments)


def _tiny_catalog(
    module: WeightContract, networks: dict[str, dict[str, HostArray]], flag: str
) -> Catalog:
    """Build a small network tensor catalog for runner fixtures."""
    catalog: Catalog = {"destination_dtype": "float32", "networks": {}}
    for network, tensors in networks.items():
        items = [
            cast(
                TensorSpec,
                {
                    "name": name,
                    "torch_shape": list(array.shape),
                    "layout": module.classify_torch_layout(name, array.shape),
                    flag: network != "vae" or not name.startswith(("encoder.", "quant_conv")),
                },
            )
            for name, array in tensors.items()
        ]
        catalog["networks"][network] = {
            "tensors": items,
            "tensor_count": len(items),
            "parameter_count": int(sum(array.size for array in tensors.values())),
        }
    return catalog


@pytest.fixture
def tiny_ct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide tiny native CT network weights and admit a 16-cubed production grid."""
    with load_fixture(FIXTURES / "from_mask.npz") as fixture:
        weights = {
            key.removeprefix("weight."): fixture[key]
            for key in fixture.files
            if key.startswith("weight.")
        }
        constructor = json.loads(str(fixture["metadata"]))["constructor"]
    networks = {
        name: {
            key.removeprefix(f"{name}."): value
            for key, value in weights.items()
            if key.startswith(f"{name}.")
        }
        for name in ("vae", "unet", "controlnet")
    }
    catalog = _tiny_catalog(maisi_mlx_weights, networks, "used_at_from_mask_infer")
    directory = tmp_path / "tiny_ct_weights"
    write_native(directory, networks)
    monkeypatch.setattr(medmlx, "load_catalog", lambda: catalog)
    monkeypatch.setattr(medmlx, "PRODUCTION_VAE_CONSTRUCTOR", constructor["vae"])
    monkeypatch.setattr(medmlx, "PRODUCTION_UNET_CONSTRUCTOR", constructor["unet"])
    monkeypatch.setattr(medmlx, "PRODUCTION_CONTROLNET_CONSTRUCTOR", constructor["controlnet"])
    # Admit a tiny grid; every other gate and the graph stay production code.
    monkeypatch.setattr(maisi, "MAISI_VALID_XY", frozenset({*maisi.MAISI_VALID_XY, 16}))
    monkeypatch.setattr(maisi, "MAISI_VALID_Z", frozenset({*maisi.MAISI_VALID_Z, 16}))
    monkeypatch.setattr(
        from_mask,
        "SUPPORTED_PRODUCTION_SHAPES",
        (*from_mask.SUPPORTED_PRODUCTION_SHAPES, (16, 16, 16)),
    )
    return directory


def test_from_mask_end_to_end_on_metal(tiny_ct: Path, tmp_path: Path, metal: Mlx) -> None:
    """Exercise a small seeded from-mask pipeline on Metal and reproduce it from the seed."""
    label = np.full((16, 16, 16), 200, np.int16)
    label[4:12, 4:12, 4:12] = 1
    affine = np.diag((1.5, 1.5, 1.5, 1.0))
    affine[:3, 3] = (-12.0, -12.0, 30.0)
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(label, affine), mask)
    first, second = (
        medmlx.run_ct(
            ct_intensity_qc=False, output_dir=tmp_path / name, weights_path=tiny_ct,
            mask_path=mask, seed=3,
        )
        for name in ("a", "b")
    )
    ct = nib.load(first["outputs"]["synthetic_image"])
    data = cast(FloatArray, np.asanyarray(ct.dataobj))
    assert ct.shape == (16, 16, 16) and data.dtype == np.float32
    np.testing.assert_array_equal(ct.affine, affine)
    assert np.isfinite(data).all() and data.min() >= -1000.0 and data.max() <= 1000.0
    written = nib.load(first["outputs"]["target_mask"])
    np.testing.assert_array_equal(np.asanyarray(written.dataobj), label)
    repeat = np.asanyarray(nib.load(second["outputs"]["synthetic_image"]).dataobj)
    np.testing.assert_array_equal(data, repeat, err_msg="the same seed must reproduce the same CT")
