<div align="center">

# mlx-generate-ctmr

Generate synthetic 3D CT and MRI on Apple Silicon with NVIDIA NV-Generate and native MLX.

[![PyPI](https://img.shields.io/pypi/v/mlx-generate-ctmr)](https://pypi.org/project/mlx-generate-ctmr/)
[![Python](https://img.shields.io/pypi/pyversions/mlx-generate-ctmr)](https://pypi.org/project/mlx-generate-ctmr/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://github.com/MedMLX/mlx-generate-ctmr/blob/v0.5.2.post2/LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MLX-black?logo=apple)](https://github.com/ml-explore/mlx)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-weights-yellow)](https://huggingface.co/MedMLX/mlx-generate-ctmr)
[![MedMLX](https://img.shields.io/badge/MedMLX-runtime-A51C30)](https://github.com/MedMLX/medmlx)

[Model weights](https://huggingface.co/MedMLX/mlx-generate-ctmr) · [MedMLX](https://github.com/MedMLX/medmlx)

<img src="https://raw.githubusercontent.com/MedMLX/mlx-generate-ctmr/v0.5.2.post2/docs/example.png" alt="mlx-generate-ctmr example output" width="360">

*A synthetic T1 brain MR from the MR-Brain configuration; no patient data.*

</div>

| | |
| :--- | :--- |
| **Input** | Noise, or a CT anatomy mask |
| **Output** | Synthetic 3D CT or MRI |
| **Hardware** | Apple Silicon with Metal. Peak MLX memory: **27.9 GiB** for `mlx-generate-ctmr-ct` (256³, from-mask, ControlNet + four-window decode); **23.8 GiB** for `mlx-generate-ctmr-mr-brain` (256×256×128, guided batch 2). No recorded peak for `mlx-generate-ctmr-mr`. |

The v0.5.1 memory measurements used MLX 0.32.4 and `MLX_ENABLE_TF32=0`: default `fast` arithmetic with FP32 accumulation and
split-FP16 convolution operands, production constructors with random fan-in-scaled
weights, seed 0 and 30 steps. Peaks include VAE decoding and exclude weight
loading. These figures cover the stated pipelines and grids; they are MLX
allocations, excluding host preprocessing and other applications.

## Use with MedMLX

The [MedMLX MCP server](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md) is the lowest-friction way to use this package. It installs this runtime, downloads its weights on first use, and lets an AI agent describe, check and run `mlx-generate-ctmr-ct`, `mlx-generate-ctmr-mr` and `mlx-generate-ctmr-mr-brain` for you.

```sh
uv tool install --python 3.13 "medmlx[generate,mcp]"
MEDMLX="$(uv tool dir --bin --color never)/medmlx"
mkdir -p "$HOME/medmlx-work"
claude mcp add --scope user medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"
```

For Codex, use `codex mcp add medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"`. Other MCP clients take the same command and arguments. The same install provides the `medmlx` CLI, for example `medmlx describe mlx-generate-ctmr-ct`.

> **For AI agents:** prefer the MedMLX MCP tools (`list_models`, `describe_model`, `model_readiness`, `run_model`, `get_job`) over writing Python against this package.

## Install

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "mlx-generate-ctmr>=0.5.2.post2"
```

## Run

```python
from mlx_generate_ctmr.medmlx import run

result = run(
    configuration="mr-brain",
    modality="mri_t1",
    output_dir="out",
    seed=0,
)
print(result["outputs"])
```

Omitting `weights_path` downloads the selected configuration from
`MedMLX/mlx-generate-ctmr` into the shared MedMLX cache on first use; interrupted
downloads resume and later runs work offline. With MedMLX installed, `medmlx setup
mlx-generate-ctmr-mr-brain` downloads it ahead of time. To use your own copy, pass one
configuration folder, as shown below as `weights_path`. The CT selection also includes
the mask network. `details.weights_revision` records the downloaded commit, or `null`
for a folder without a MedMLX download record.

| Model ID | `configuration` | Generates |
| --- | --- | --- |
| `mlx-generate-ctmr-ct` | `ct` (default) | CT from a MAISI label mask, or paired CT and labels |
| `mlx-generate-ctmr-mr` | `mr` | Image-only MR |
| `mlx-generate-ctmr-mr-brain` | `mr-brain` | Image-only brain MR |

`run_ct`, `run_mr` and `run_mr_brain` fix the configuration; `run` takes it as an
argument. Generation needs Apple Silicon with Metal and has no CPU fallback.

### Local weights

For an explicit local folder, download the native MR-Brain bundle and pass
`weights_path`. No checkpoint conversion is needed:

```python
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_generate_ctmr.medmlx import run

weights_root = Path(snapshot_download(
    "MedMLX/mlx-generate-ctmr",
    revision="63018cdac46394b3c12619bc9fd1cbe096c23b20",
    allow_patterns=["mr-brain/model.safetensors", "mr-brain/conversion.json"],
    local_dir="ctmr-weights",
))

result = run(
    configuration="mr-brain",
    modality="mri_t1",
    weights_path=weights_root / "mr-brain",
    output_dir="out",
    seed=0,
)
print(result["outputs"])
```

The [model repository](https://huggingface.co/MedMLX/mlx-generate-ctmr/tree/63018cdac46394b3c12619bc9fd1cbe096c23b20) has one folder per use:

| Folder | Pass as |
| --- | --- |
| `ct/` | `weights_path` for CT |
| `mask/` | `mask_weights_path` for learned paired CT |
| `mr/` | `weights_path` for general MR, under NVIDIA's noncommercial terms |
| `mr-brain/` | `weights_path` for MR-Brain |

Each folder holds `model.safetensors`, with tensors under `vae.`, `unet.` and, for
CT, `controlnet.` prefixes. A local `weights_path` folder may also contain `mask/`.

### MR

`modality` is a contrast name from the upstream mapping or its index; the default
is T1 (9). General MR accepts 8 to 11. MR-Brain accepts T1, T2, FLAIR, MRA and SWI
(9, 10, 11, 16, 20) and their skull-stripped variants (29, 30, 31, 33, 32). Other
contrasts are rejected. `guidance_scale` defaults to 15 for MR and 2 for MR-Brain,
as in the pinned inference configurations.

Default grids are 128×256×256 at 1.25×1×1 mm for MR and 256×256×128 at
0.94×0.94×1.36 mm for MR-Brain. `output_size` axes must be positive multiples of 32,
at most 512 each and 256³ voxels in total.

MR writes `synthetic_mr.nii.gz` or `synthetic_mr_brain.nii.gz`: int16, with a
diagonal RAS affine from `spacing_mm`. As upstream, decoded values are multiplied
by 1000, clipped below zero only and truncated, so intensities are arbitrary scaled
units. Skull stripping is a modality embedding, not a brain mask. MR needs no input
image.

### CT

`generation_mode` is `from_mask` when `mask_path` is given and `paired` otherwise.

- **From a mask:** `mask_path` is a MAISI label NIfTI with labels 0 to 132 and the
  required body label 200, 256×256×128 or 256×256×256, with equal X and Y spacing.
- **Paired, learned mask:** `anatomy` lists the MAISI anatomy to keep, and
  `anatomy_size` gives size controls such as `{"pancreas": 0.3}` (0 to 1, or -1;
  one controlled tumor). It needs the `mask/` weights,
  `configs/label_dict_124_to_132.json` from NV-Generate-CTMR
  (`mask_label_dict_path`) and `datasets/all_anatomy_size_conditions.json` from
  `nvidia/NV-Generate-CT` (`anatomy_size_conditions_path`), found in the weights
  folder or passed explicitly. Unrequested slots, tumors included, come from the
  database row the upstream code selects; request every tumor slot to control it.
- **Paired, archive mask:** without sizes, `mask_database_path` and
  `mask_directory` point at the pinned candidate-mask database and an extracted
  mask archive, and `body_regions` filters the candidates. Selected masks go through
  the upstream augmentation routes.

Paired grids are 256×256×256 (the default) or 256×256×128, at 1.5 mm by default.
Mask generation takes 1000 DDPM steps by default
(`mask_generation_num_inference_steps`).

CT intensity QC is on by default. Each image is checked against the pinned
per-region median HU statistics and regenerated with a derived seed, up to
`ct_qc_max_attempts` (2) attempts; if all fail, the run raises
`ModelExecutionError`. `ct_intensity_qc=False` records an opt-out, and
`ct_qc_statistics_path` replaces the statistics. With QC on, outputs are under
`stages/ct_qc/attempt-N/`.

| Role | File |
| --- | --- |
| `synthetic_image` | `synthetic_ct.nii.gz`, float32 HU in [-1000, 1000] |
| `target_mask` | `synthetic_ct_label.nii.gz`, the conditioning labels |
| `prepared_mask` | `input_mask_ras.nii.gz`, the conditioning mask in the closest canonical RAS |
| `mask_qc` | `mask_qc.json`, the structural mask QC record |
| `generated_mask`, `conditioning_mask` | Paired only: int16 label maps under `stages/mask/` |

The CT and its labels share the prepared mask's grid.

### Other arguments

| Argument | Meaning |
| --- | --- |
| `output_dir` | Directory for the outputs; created when missing |
| `seed` | Non-negative integer, default 0; paired CT derives the mask and CT seeds from it |
| `num_inference_steps` | 1 to 1000; default 30 |
| `sliding_window_size`, `sliding_window_overlap`, `autoencoder_tp_num_splits` | VAE decode windows in latent voxels (multiples of 16), overlap, and split count (1, 2, 4, 8 or 16) |
| `overwrite` | `False` by default, which refuses to replace existing outputs |
| `arithmetic` | `fast` (default) or `source`, which keeps the CUDA operation order |
| `attention_scratch_bytes` | Optional byte budget for attention score matrices; not a total memory limit |

`run` returns `{"outputs": {role: path}, "details": {...}}`. `details` records the
seeds, label table, shapes, decode settings, timings, peak MLX memory and source and
weight pins. Invalid inputs raise `InvalidInputError`, missing or mismatched weights
`AssetNotReadyError`, and a model failure, including a generated mask that fails QC,
`ModelExecutionError`.

### Python API

The module functions port the NV-Generate, MAISI and MONAI components. They take
caller-supplied weights and `mx`, and tensors use NCDHW.

```python
from mlx_generate_ctmr import arithmetic_mode, attention_memory_budget
from mlx_generate_ctmr.mr import decode_mr_ncdhw, export_mr_int16, sample_mr_latent_ncdhw
from mlx_generate_ctmr.weights import load_network

with arithmetic_mode("source"), attention_memory_budget(512 * 1024**2):
    latent = sample_mr_latent_ncdhw(...)
```

Other entry points include `unet.denoise_unet_ncdhw`,
`controlnet.controlnet_residuals_ncdhw`, `vae.decode_vae_ncdhw`,
`from_mask.sample_ct_from_mask_ncdhw`, `mask_ddpm.sample_mask_labels_ncdhw` and
`mask_postprocess.postprocess_mask`. `load_network` takes `configuration="ct"`,
`"mr"` or `"mr-brain"` and checks parameter keys, shapes, finite float32 values and
the UNet scale factor. The two context managers apply to a complete call in the
current thread or task, not to a new process; use the runner arguments in worker
requests.

Numerical parity is checked against MONAI on small seeded networks. That does not
qualify pretrained image quality. Synthetic output is not ground truth.

## License

[Apache-2.0](https://github.com/MedMLX/mlx-generate-ctmr/blob/v0.5.2.post2/LICENSE). See [third-party notices](https://github.com/MedMLX/mlx-generate-ctmr/blob/v0.5.2.post2/THIRD_PARTY_NOTICES.md). Model weights retain their publisher terms. General-MR weights have noncommercial terms; CT component terms require publisher clarification.

For research and education. Outputs require human review.
