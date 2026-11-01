# Third-party notices

MedMLX-authored additions are licensed under [Apache-2.0](LICENSE). The upstream
source components below retain their attribution and Apache-2.0 terms.

## NV-Generate and MONAI

The native MLX CT and MR generation workflows follow NVIDIA's
[NV-Generate-CTMR](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR), pinned at
`5cb04e82fed71f2fe64a2617ab695be1f5a37fea`, and its MONAI/MAISI components.
The MONAI numerical reference is revision
`eccefc57550b111ed781d82249dfe77872a0e918` (MONAI 1.6.0).
MONAI portions retain Copyright (c) MONAI Consortium. Architecture, scheduler,
preprocessing and sampling operations are adapted to native MLX execution.
The full Apache-2.0 terms are included in [LICENSE](LICENSE).

## Separately supplied assets

CT, MR and MR-Brain checkpoints have separate publisher terms. Their repositories
and revisions are listed in [the configuration data](src/mlx_generate_ctmr/data/configurations.json);
restrictions are summarized in [the README](README.md#license). The source
license does not replace those asset terms. Dependencies declared in
[pyproject.toml](pyproject.toml) retain their respective licenses.
