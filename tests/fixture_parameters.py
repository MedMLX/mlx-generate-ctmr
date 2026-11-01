"""Restore seeded reference parameters without storing model tensors in fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Self, cast

import numpy as np

from mlx_generate_ctmr._arrays import HostArray


def seeded_parameters(recipe: dict[str, object]) -> dict[str, HostArray]:
    """Reproduce the source recorder's parameter stream and normalization offsets."""
    rng = np.random.default_rng(cast(int, recipe["seed"]))
    rng.normal(size=cast(int, recipe["skip"]))
    result: dict[str, HostArray] = {}
    for name, shape in cast(list[tuple[str, list[int]]], recipe["shapes"]):
        value = rng.normal(0, 0.02, tuple(shape)).astype(np.float32)
        if value.ndim == 1 and name.endswith("weight"):
            value += np.float32(1)
        result[name] = value
    return result


class FixtureArchive:
    """Expose recorded arrays and their regenerated parameters as a context manager."""

    def __init__(self, arrays: dict[str, HostArray]) -> None:
        self._arrays = arrays
        self.files = list(arrays)

    def __getitem__(self, name: str) -> HostArray:
        return self._arrays[name]

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


def load_fixture(path: Path) -> FixtureArchive:
    """Read a numeric fixture and regenerate parameters when its metadata has a recipe."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: cast(HostArray, np.array(archive[name])) for name in archive.files}
    if "metadata" in arrays:
        metadata = cast(dict[str, object], json.loads(str(arrays["metadata"])))
        recipe = metadata.get("weight_recipe")
        if recipe is not None:
            arrays.update(seeded_parameters(cast(dict[str, object], recipe)))
    return FixtureArchive(arrays)


def boundary_arrays(directory: Path, boundary: str) -> dict[str, HostArray]:
    """Load a MAISI boundary and restore its recorded reference parameters."""
    manifest = cast(dict[str, object], json.loads((directory / f"{boundary}.json").read_text()))
    specs = cast(dict[str, dict[str, object]], manifest["arrays"])
    with np.load(directory / f"{boundary}.npz", allow_pickle=False) as archive:
        arrays = {name: cast(HostArray, np.array(archive[name])) for name in archive.files}
    source = {"vae": "vae", "controlnet": "controlnet", "diffusion": "unet"}.get(boundary)
    if source is not None:
        with load_fixture(directory.parent / f"{source}.npz") as fixture:
            arrays.update({name: fixture[name] for name in specs if name.startswith("weight.")})
    elif "weight_recipe" in manifest:
        arrays.update(seeded_parameters(cast(dict[str, object], manifest["weight_recipe"])))
    return arrays
