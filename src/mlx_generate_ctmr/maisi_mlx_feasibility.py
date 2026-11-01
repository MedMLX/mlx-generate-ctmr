"""Relative/absolute FP32 tolerances used when comparing MLX outputs with MONAI references.

These are declared test budgets, not measured error bounds. Each pair is passed as
``rtol``/``atol`` to ``numpy.allclose``-style comparisons.
"""

from __future__ import annotations

from typing import Final

# Residual uses the same math. Cross-backend drift comes from reduction order.
RESIDUAL_RTOL: Final = 1.0e-5
RESIDUAL_ATOL: Final = 1.0e-6

# Attention drift comes from softmax reduction order. MLX SDPA uses fp32 softmax.
ATTENTION_RTOL: Final = 1.0e-4
ATTENTION_ATOL: Final = 1.0e-5

# VAE decode uses the same math. Cross-backend drift comes from reduction order.
VAE_RTOL: Final = 1.0e-4
VAE_ATOL: Final = 1.0e-5

# UNet drift follows the same fp32 tolerance budget as attention and VAE.
UNET_RTOL: Final = 1.0e-4
UNET_ATOL: Final = 1.0e-5

# ControlNet uses the UNet path plus 1x1 residual heads.
CONTROLNET_RTOL: Final = 1.0e-4
CONTROLNET_ATOL: Final = 1.0e-5
