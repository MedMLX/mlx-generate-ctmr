"""Request-scoped choice between fast and source-ordered FP32 arithmetic, without importing MLX.

``"fast-fp32"`` (the default) runs the image UNet, ControlNet and VAE decoder on
``fast_fp32``: implicit-GEMM convolutions on M5 neural accelerators when available
(ordinary MLX GPU convolutions otherwise), fused group norm/SiLU/residual epilogues
and flash attention, all accumulating in FP32
but not in the CUDA operation order. ``"source-fp32"`` keeps the CUDA-matched
``source_fp32``/``source_vae`` kernels that the seeded MONAI references and the
retained pretrained MR-Brain CUDA comparison were recorded against.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Final, Literal

from mlx_generate_ctmr import fast_fp32, source_fp32, source_vae

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import UnetArithmetic, VaeArithmetic

type Precision = Literal["fast-fp32", "source-fp32"]

PRECISIONS: Final[tuple[Precision, ...]] = ("fast-fp32", "source-fp32")

_precision: ContextVar[Precision] = ContextVar("mlx_generate_ctmr_precision", default="fast-fp32")


@contextmanager
def precision_mode(precision: Precision) -> Generator[None]:
    """Select the image-network precision for calls made inside the block.

    Nested calls restore the previous precision, including on exceptions. Like
    ``attention_memory_budget`` the setting is isolated between threads/tasks
    and does not propagate into newly spawned processes.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}; got {precision!r}")
    token = _precision.set(precision)
    try:
        yield
    finally:
        _precision.reset(token)


def active_precision() -> Precision:
    """Return the precision selected by the innermost ``precision_mode`` block."""
    return _precision.get()


def unet_arithmetic() -> UnetArithmetic:
    """Image UNet/ControlNet arithmetic module for the active precision."""
    return source_fp32 if _precision.get() == "source-fp32" else fast_fp32


def vae_arithmetic() -> VaeArithmetic:
    """Image VAE decoder arithmetic module for the active precision."""
    return source_vae if _precision.get() == "source-fp32" else fast_fp32
