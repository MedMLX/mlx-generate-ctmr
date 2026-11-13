"""Load a small HCC-TACE CT crop with its paired liver and tumor annotations."""

from pathlib import Path

import numpy as np

from mlx_generate_ctmr._arrays import FloatArray, IntArray
from mlx_generate_ctmr._nifti import nib


def real_ct_image(size: int) -> FloatArray:
    """Return an HCC-TACE CT cube in the MAISI encoder's clipped intensity units."""
    root = Path.home() / ".cache/medmlx/test-data"
    source = nib.load(root / "hcc_063_ct.nii.gz")
    crop = np.asarray(source.dataobj, dtype=np.float32)[
        100:100 + size, 210:210 + size, 55:55 + size,
    ]
    if crop.shape != (size,) * 3:
        raise ValueError("The HCC-TACE CT does not cover the requested encoder cube")
    return np.clip(crop / np.float32(1000), -1, 1)[None, None].copy()


def real_liver_mask() -> IntArray:
    """Map real liver and lesion annotations into the MAISI label convention.

    A CT threshold supplies the required body envelope. Label 1 denotes liver,
    label 26 denotes liver tumor, and label 200 denotes other body tissue.
    """
    root = Path.home() / ".cache/medmlx/test-data"
    region = (slice(100, 116), slice(210, 226), slice(55, 71))
    ct = np.asarray(nib.load(root / "hcc_063_ct.nii.gz").dataobj, dtype=np.float32)[region]
    liver = np.asarray(nib.load(root / "hcc_063_liver.nii.gz").dataobj, dtype=np.int16)[region] > 0
    tumor = np.asarray(nib.load(root / "hcc_063_tumor.nii.gz").dataobj, dtype=np.int16)[region] > 0
    labels = np.where(ct > -500, 200, 0).astype(np.int16)
    labels[liver] = 1
    labels[tumor] = 26
    return labels
