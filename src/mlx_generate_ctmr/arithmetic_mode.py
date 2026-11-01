"""Request-scoped choice between fast and source-ordered FP32 arithmetic, without importing MLX.

``"fast"`` (the default) runs the image UNet, ControlNet and VAE decoder on
``fast_fp32``: implicit-GEMM convolutions on M5 neural accelerators when available
(ordinary MLX GPU convolutions otherwise), fused group norm/SiLU/residual epilogues
and flash attention, all accumulating in FP32
but not in the CUDA operation order. ``"source"`` keeps the CUDA-matched
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

type ArithmeticMode = Literal["fast", "source"]

ARITHMETIC_MODES: Final[tuple[ArithmeticMode, ...]] = ("fast", "source")

_mode: ContextVar[ArithmeticMode] = ContextVar("mlx_generate_ctmr_arithmetic", default="fast")


@contextmanager
def arithmetic_mode(mode: ArithmeticMode) -> Generator[None]:
    """Select the image-network arithmetic for calls made inside the block.

    Nested calls restore the previous mode, including on exceptions. Like
    ``attention_memory_budget`` the setting is isolated between threads/tasks
    and does not propagate into newly spawned processes.
    """
    if mode not in ARITHMETIC_MODES:
        raise ValueError(f"arithmetic must be one of {ARITHMETIC_MODES}; got {mode!r}")
    token = _mode.set(mode)
    try:
        yield
    finally:
        _mode.reset(token)


def active_arithmetic_mode() -> ArithmeticMode:
    """Return the mode selected by the innermost ``arithmetic_mode`` block."""
    return _mode.get()


def unet_arithmetic() -> UnetArithmetic:
    """Image UNet/ControlNet arithmetic module for the active mode."""
    return source_fp32 if _mode.get() == "source" else fast_fp32


def vae_arithmetic() -> VaeArithmetic:
    """Image VAE decoder arithmetic module for the active mode."""
    return source_vae if _mode.get() == "source" else fast_fp32
