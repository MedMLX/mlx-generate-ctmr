"""HCC-TACE CT and source-aligned labels for intensity QC cases."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from mlx_generate_ctmr._nifti import nib


def ct_qc_case(case: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int16]]:
    """Load a CT crop in HU and map its liver and tumor masks to MAISI labels.

    The threshold case shifts the real CT intensities to place the liver median
    at 162 HU; the reject case shifts them beyond that bound. The NaN case
    removes liver samples to exercise the upstream missing-region behavior.
    """
    root = Path.home() / ".cache/medmlx/test-data"
    ct, liver, tumor = [
        nib.load(root / f"hcc_063_{name}.nii.gz") for name in ("ct", "liver", "tumor")
    ]
    if any(
        image.shape != ct.shape or not np.allclose(image.affine, ct.affine)
        for image in (liver, tumor)
    ):
        raise ValueError("HCC-TACE CT and mask grids must match")
    crop = (slice(100, 148), slice(210, 258), slice(45, 93))
    image = np.asarray(ct.dataobj, dtype=np.float64)[crop].copy()
    liver_mask = np.asarray(liver.dataobj, dtype=np.int16)[crop] > 0
    tumor_mask = np.asarray(tumor.dataobj, dtype=np.int16)[crop] > 0
    labels = np.zeros(image.shape, dtype=np.int16)
    labels[liver_mask] = 1
    labels[tumor_mask] = 26
    if not np.any(labels == 1) or not np.any(labels == 26):
        raise ValueError("HCC-TACE crop must contain liver and tumor")
    if case == "reject":
        image += 500.0
    elif case == "boundary":
        image += 162.0 - float(np.median(image[labels == 1]))
    elif case == "nan":
        image[labels == 1] = np.nan
    elif case != "accept":
        raise ValueError(f"Unknown CT QC case: {case}")
    return image, labels
