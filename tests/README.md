The image and annotation tests use the HCC-TACE-Seg HCC_063 contrast CT, liver
mask, and tumor mask at `~/.cache/medmlx/test-data/hcc_063_ct.nii.gz`,
`hcc_063_liver.nii.gz`, and `hcc_063_tumor.nii.gz`. Image-derived reference
arrays are written to `work/real-reference`, outside Git.

Run `scripts/record_real_model_references.py` with MONAI 1.6.0, Torch 2.14,
and the project test dependencies. Run `scripts/record_ct_qc_reference.py` and
`scripts/record_archive_augmentation_reference.py` with `--source` pointing to
the NV-Generate-CTMR checkout at revision
`5cb04e82fed71f2fe64a2617ab695be1f5a37fea`; the augmentation recorder also
requires the pinned MONAI environment. `MEDMLX_TEST_REFERENCES` overrides the
local reference directory for the archive augmentation test.
