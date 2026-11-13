"""Record pinned upstream archive augmentation on HCC-TACE annotations."""

from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as functional
from monai.transforms import Rand3DElastic, RandZoom
from monai.utils import ensure_tuple_rep

REVISION = "5cb04e82fed71f2fe64a2617ab695be1f5a37fea"


def main() -> None:
    """Execute the source liver and body paths on CPU using paired scan masks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path.home() / ".cache/medmlx/test-data")
    parser.add_argument("--output", type=Path, default=Path("work/real-reference"))
    args = parser.parse_args()
    head = subprocess.run(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != REVISION:
        raise ValueError(f"Expected upstream {REVISION}, got {head}")
    subprocess.run(
        [
            "git",
            "-C",
            str(args.source),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "scripts/augmentation.py",
        ],
        check=True,
        capture_output=True,
    )
    path = args.source / "scripts/augmentation.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {"erode3d", "dilate3d", "augmentation_tumor_liver", "augmentation_body"}
    tree.body = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in tree.body} != names:
        raise ValueError("Pinned augmentation definitions are incomplete")
    scope = {
        "np": np,
        "torch": torch,
        "F": functional,
        "Rand3DElastic": Rand3DElastic,
        "RandZoom": RandZoom,
        "ensure_tuple_rep": ensure_tuple_rep,
    }
    exec(compile(tree, str(path), "exec"), scope)
    region = (slice(100, 148), slice(210, 258), slice(45, 93))
    ct = np.asarray(nib.load(args.data / "hcc_063_ct.nii.gz").dataobj, dtype=np.float32)[region]
    liver = np.asarray(nib.load(args.data / "hcc_063_liver.nii.gz").dataobj)[region] > 0
    tumor = np.asarray(nib.load(args.data / "hcc_063_tumor.nii.gz").dataobj)[region] > 0
    labels = np.where(ct > -500, 200, 0).astype(np.int16)
    labels[liver] = 1
    labels[tumor] = 26
    outputs = {"source_revision": np.asarray([REVISION])}
    torch.set_num_threads(1)
    for case in (0, 26):
        mask = labels.copy()
        if case == 0:
            mask[mask == 26] = 1
        tensor = torch.from_numpy(mask.copy())[None, None]
        # Upstream hard-codes CUDA for its spatial transform; CPU executes the same
        # pinned MONAI reference when CUDA is unavailable on the recording host.
        with patch.object(torch.Tensor, "cuda", lambda self: self), torch.no_grad():
            if case == 26:
                result = scope["augmentation_tumor_liver"](tensor, mask.shape, random_seed=4)
            else:
                result = scope["augmentation_body"](tensor, random_seed=4)
        outputs[f"input_{case}"] = mask
        outputs[f"output_{case}"] = result.cpu().numpy().squeeze().astype(np.int16)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "archive_augmentation_source.npz", **outputs)


if __name__ == "__main__":
    main()
