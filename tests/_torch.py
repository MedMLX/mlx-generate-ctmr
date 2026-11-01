"""Narrow reference-only adapters for incomplete Torch NumPy/save annotations."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import BinaryIO, Protocol, cast

import torch

from mlx_generate_ctmr._arrays import HostArray


# These are the original Torch functions, with their documented arguments and
# results. They never enter the package's inference or NumPy-only conversion path.
class TorchSerialization(Protocol):
    def from_numpy(self, array: HostArray) -> torch.Tensor:
        """Create a reference tensor from a host array."""
        ...
    def save(self, obj: object, f: Path | str | BinaryIO) -> None:
        """Serialize a reference object to a path or stream."""
        ...


_torch = cast(TorchSerialization, torch)
from_numpy: Callable[[HostArray], torch.Tensor] = _torch.from_numpy
save: Callable[[object, Path | str | BinaryIO], None] = _torch.save


class RFlowStep(Protocol):
    # The pinned source converts next_timestep to int and uses float(timestep - next).
    def __call__(
        self,
        model_output: torch.Tensor,
        timestep: int | float | torch.Tensor,
        sample: torch.Tensor,
        next_timestep: int | float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one reference rectified-flow scheduler step."""
        ...


def construct[Model](factory: Callable[..., Model], constructor: Mapping[str, object]) -> Model:
    """Call a real reference model with its recorded source constructor values."""
    return factory(**constructor)


class DDPMSchedule(Protocol):
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
