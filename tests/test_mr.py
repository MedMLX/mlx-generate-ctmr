"""MR weights, runner and independent Torch fixture parity."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from _mr_contracts import Metadata
from _native_fixture import write_native
from fixture_parameters import load_fixture
from medmlx_core.errors import InvalidInputError
from numerical import assert_mr_intensity_parity
from safetensors.numpy import load_file, save_file

from mlx_generate_ctmr import condition, image, image_runner, medmlx, rflow, weights
from mlx_generate_ctmr._arrays import NumericArray, RealArray, as_numpy
from mlx_generate_ctmr._nifti import nib
from mlx_generate_ctmr._records import Catalog, NetworkInventory, TensorSpec
from mlx_generate_ctmr.maisi_mlx_common import PINNED_MONAI_REVISION, PINNED_NV_GENERATE_REVISION
from mlx_generate_ctmr.maisi_mlx_feasibility import UNET_ATOL, UNET_RTOL, VAE_ATOL, VAE_RTOL
from mlx_generate_ctmr.maisi_mlx_weights import classify_torch_layout, load_catalog

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, ArrayConstructor, ArrayFactory, Dtype, Mlx

FIXTURES = Path(__file__).parent / "fixtures" / "darwin-arm64"
CONFIGURATIONS = ("mr", "mr-brain")


def recorded(configuration: str) -> tuple[dict[str, NumericArray], Metadata]:
    """Load recorded MR fixture arrays and metadata for a configuration."""
    with load_fixture(FIXTURES / f"{configuration}.npz") as archive:
        payload = {name: cast(NumericArray, archive[name]) for name in archive.files}
    metadata = cast(Metadata, json.loads(str(payload["metadata"])))
    assert metadata["source_revision"] == PINNED_NV_GENERATE_REVISION
    assert metadata["monai_revision"] == PINNED_MONAI_REVISION
    return payload, metadata


type Networks = dict[str, dict[str, NumericArray]]


def native_fixture(payload: dict[str, NumericArray]) -> tuple[Catalog, Networks]:
    """Return a catalog and source-layout network tensors from recorded fixture weights."""
    raw = {
        network: {
            name.removeprefix(f"weight.{network}."): array
            for name, array in payload.items()
            if name.startswith(f"weight.{network}.")
        }
        for network in ("vae", "unet")
    }
    networks: dict[str, NetworkInventory] = {}
    for network, tensors in raw.items():
        specs: list[TensorSpec] = [
            {
                "name": name,
                "torch_shape": list(array.shape),
                "layout": classify_torch_layout(name, array.shape),
                "used_at_from_mask_infer": not name.startswith(
                    ("encoder.", "quant_conv_mu.", "quant_conv_log_sigma.")
                ),
            }
            for name, array in tensors.items()
        ]
        networks[network] = {
            "tensors": specs,
            "tensor_count": len(specs),
            "parameter_count": sum(np.prod(t["torch_shape"]).item() for t in specs),
        }
    return {"destination_dtype": "float32", "networks": networks}, raw


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_fixture_inventory_and_native_weights(configuration: str, tmp_path: Path, metal: Mlx):
    """Round-trip recorded tensors through the native checkpoint layout and loader."""
    payload, _ = recorded(configuration)
    catalog, raw = native_fixture(payload)
    for network in raw:
        assert raw[network].keys() == {
            spec["name"] for spec in load_catalog(configuration)["networks"][network]["tensors"]
        }
    write_native(tmp_path, raw, configuration=configuration)
    for network in raw:
        loaded, factor = weights.load_network(
            tmp_path / "model.safetensors", network=network, configuration=configuration,
            catalog=catalog, mx=cast("ArrayFactory[Array, Dtype]", metal),
        )
        assert factor == (1.25 if network == "unet" else None)
        assert loaded.keys() == raw[network].keys()
        for name in loaded:
            np.testing.assert_array_equal(
                as_numpy(loaded[name]), raw[network][name], err_msg=name
            )


@pytest.mark.parametrize("problem", ["missing", "shape", "dtype", "nonfinite", "scale"])
def test_strict_loading(problem: str, tmp_path: Path, metal: Mlx):
    """Reject malformed native checkpoints at the weight boundary."""
    catalog, raw = native_fixture(recorded("mr")[0])
    write_native(tmp_path, raw, configuration="mr")
    path = tmp_path / "model.safetensors"
    arrays = load_file(path)
    key = next(name for name in arrays if name.startswith("unet.") and name != "unet.scale_factor")
    if problem == "missing":
        del arrays[key]
    elif problem == "shape":
        arrays[key] = arrays[key].reshape(-1)[:1]
    elif problem == "dtype":
        arrays[key] = arrays[key].astype(np.float64)
    elif problem == "scale":
        arrays["unet.scale_factor"] = np.array(np.nan)
    else:
        arrays[key] = np.full_like(arrays[key], np.inf)
    save_file(arrays, path)
    with pytest.raises(InvalidInputError):
        weights.load_network(
            path, network="unet", catalog=catalog, configuration="mr",
            mx=cast("ArrayFactory[Array, Dtype]", metal),
        )


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_conditions_and_intensity_mapping_match_source(configuration: str, metal: Mlx):
    """Compare MR modality, spacing, schedule and output intensities with the recordings."""
    payload, meta = recorded(configuration)
    classes = condition.modality_class_labels(
        mx=metal, configuration=configuration, modality=meta["modality"]
    )
    np.testing.assert_array_equal(as_numpy(classes), payload["output.classes"])
    for source, key in (("spacing_mm", "spacing"), ("spacing_probe", "spacing_probe")):
        millimeters = cast(RealArray, payload[f"input.{source}"])
        spacing = image.image_spacing_tensor(
            millimeters, mx=cast("ArrayFactory[Array, Dtype]", metal)
        )
        np.testing.assert_array_equal(as_numpy(spacing), payload[f"output.{key}"])
    timesteps, _ = rflow.set_timesteps_rflow(num_inference_steps=30, input_img_size_numel=8**3)
    np.testing.assert_array_equal(timesteps, payload["output.timesteps"])
    intensities = image.output_intensities(payload["output.decoded"], configuration=configuration)
    np.testing.assert_array_equal(intensities, payload["output.image"])


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_seeded_mr_unet_and_pipeline_parity(configuration: str, metal: Mlx):
    """Compare a seeded MR UNet and pipeline with the reference."""
    payload, meta = recorded(configuration)
    assert meta["monai_version"] == "1.6.0" and meta["precision"] == "fp32"
    array = cast("ArrayConstructor", metal.array)
    params = {
        network: {
            name.removeprefix(f"weight.{network}."): array(value)
            for name, value in payload.items()
            if name.startswith(f"weight.{network}.")
        }
        for network in ("unet", "vae")
    }
    ctor = meta["constructor"]
    x = array(payload["input.latent"])
    _, actual = image.denoise_unet_ncdhw(
        metal.concatenate([x, x]),
        array([1000, 1000], dtype=metal.float32),
        array([meta["modality"], 0], dtype=metal.int32),
        array(np.repeat(payload["output.spacing"], 2, axis=0)),
        params["unet"],
        constructor=ctor["unet"],
        mx=metal,
    )
    np.testing.assert_allclose(
        as_numpy(actual), payload["output.velocity_batch"], rtol=UNET_RTOL, atol=UNET_ATOL
    )
    result = image.sample_image_ncdhw(
        (32, 32, 32),
        cast(RealArray, payload["input.spacing_mm"]),
        configuration=configuration,
        modality=meta["modality"],
        latent=x,
        scale_factor=1.25,
        unet_weights=params["unet"],
        vae_weights=params["vae"],
        unet_constructor=ctor["unet"],
        vae_constructor=ctor["vae"],
        mx=metal,
    )
    np.testing.assert_allclose(
        as_numpy(result.latent), payload["output.latent"], rtol=UNET_RTOL, atol=UNET_ATOL
    )
    np.testing.assert_allclose(
        as_numpy(result.decoded), payload["output.decoded"], rtol=VAE_RTOL, atol=VAE_ATOL
    )
    assert_mr_intensity_parity(result.image, payload["output.decoded"])


@pytest.mark.parametrize(
    ("configuration", "runner"), [("mr", "run_mr"), ("mr-brain", "run_mr_brain")]
)
def test_seeded_public_runner_on_metal(
    configuration: str, runner: str, metal: Mlx, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Run the registered MR runner on Metal from the recorded latent and match the source."""
    payload, meta = recorded(configuration)
    catalog, raw = native_fixture(payload)
    directory = tmp_path / "weights"
    write_native(directory, raw, configuration=configuration)

    def provide(*args: object) -> Catalog:
        """Provide the recorded network catalog to the runner."""
        return catalog

    monkeypatch.setattr(image_runner, "load_catalog", provide)
    monkeypatch.setattr(image_runner, "PRODUCTION_UNET_CONSTRUCTOR", meta["constructor"]["unet"])
    monkeypatch.setattr(image_runner, "PRODUCTION_VAE_CONSTRUCTOR", meta["constructor"]["vae"])
    latent = payload["input.latent"]

    def fixed_latent(shape: Sequence[int], *, seed: int, mx: Mlx) -> Array:
        """Supply the recorded initial latent for the native run."""
        assert seed == meta["seed"] and tuple(shape) == latent.shape
        return cast("ArrayConstructor", mx.array)(latent)

    monkeypatch.setattr(image, "init_latent_ncdhw", fixed_latent)
    spacing = cast(RealArray, payload["input.spacing_mm"])
    result = cast("Callable[..., medmlx.RunResult[str]]", getattr(medmlx, runner))(
        weights_path=directory,
        output_dir=tmp_path / "out",
        output_size=(32, 32, 32),
        spacing_mm=spacing.tolist(),
        modality=meta["modality"],
        seed=meta["seed"],
    )
    volume = nib.load(result["outputs"]["synthetic_image"])
    stored = cast(NumericArray, np.asanyarray(volume.dataobj))
    assert_mr_intensity_parity(stored[None, None], payload["output.decoded"])
    np.testing.assert_allclose(volume.affine, np.diag((*spacing, 1)))
    assert result["details"]["modality"] == meta["modality"]
