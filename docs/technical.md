# Technical notes

## Peak memory

The README's peak MLX memory figures are **12.7 GiB** for `mlx-generate-ctmr-ct` (256³,
from-mask, ControlNet + four-window decode) and **11.6 GiB** for
`mlx-generate-ctmr-mr-brain` (256×256×128, guided batch 2). There is no recorded peak for
`mlx-generate-ctmr-mr`.

The memory figures are the runs' recorded `details.peak_memory_bytes` with MLX
0.32.4 on the released weights: default `fast` arithmetic with FP32 accumulation
and split-FP16 convolution operands, seed 0 and 30 steps, CT from a 256³ MAISI
label mask with intensity QC off. Peaks include VAE decoding. These figures
cover the stated pipelines and grids; they are MLX allocations, excluding host
preprocessing and other applications.
