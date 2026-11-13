"""Record pinned upstream CT intensity decisions on HCC-TACE CT crops."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REVISION = "5cb04e82fed71f2fe64a2617ab695be1f5a37fea"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from ct_qc_cases import ct_qc_case  # noqa: E402


def main() -> None:
    """Run the unmodified pinned source and save scalar results, not images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.run(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != REVISION:
        raise ValueError(f"Expected upstream {REVISION}, got {revision}")
    subprocess.run(
        [
            "git",
            "-C",
            str(args.source),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "scripts/quality_check.py",
            "configs/image_median_statistics_ct.json",
        ],
        check=True,
        capture_output=True,
    )
    spec = importlib.util.spec_from_file_location(
        "upstream_qc", args.source / "scripts/quality_check.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load upstream QC implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    statistics = json.loads((args.source / "configs/image_median_statistics_ct.json").read_text())
    # Only source-annotated liver and tumor occur in these masks; other regions
    # are absent and must retain the upstream no-samples behavior.
    label_ids = {"liver": [1], "hepatic tumor": [26]}
    cases = {}
    for name in ("accept", "reject", "boundary", "nan"):
        image, labels = ct_qc_case(name)
        cases[name] = module.is_outlier(statistics, image, labels, label_ids)
    payload = {
        "source_revision": REVISION,
        "source": "HCC-TACE-Seg HCC_063 contrast CT, liver and tumor",
        "cases": cases,
    }
    destination = ROOT / "tests/fixtures/ct_qc_reference.json"
    destination.write_text(
        json.dumps(
            payload,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else x,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
