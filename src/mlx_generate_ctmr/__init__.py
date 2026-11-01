"""Native MLX implementation of NVIDIA NV-Generate (MAISI) for synthetic CT and MR.

Outputs are synthetic research data, not for clinical use. Importing the package
does not import MLX, Torch or MONAI: the model functions take the MLX runtime as
an argument, and the runners import it through ``medmlx_core.runtime.import_mlx``.
"""

__version__ = "0.5.2"

from medmlx_core.maisi import NvGenerateAnatomySize

from mlx_generate_ctmr.arithmetic_mode import arithmetic_mode
from mlx_generate_ctmr.attention import spatial_attention_ncdhw
from mlx_generate_ctmr.attention_budget import attention_memory_budget
from mlx_generate_ctmr.residual import residual_block_ncdhw

__all__ = [
    "NvGenerateAnatomySize",
    "arithmetic_mode",
    "attention_memory_budget",
    "residual_block_ncdhw",
    "spatial_attention_ncdhw",
]
