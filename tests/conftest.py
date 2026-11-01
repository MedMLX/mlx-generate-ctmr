"""Metal is the only inference backend; host contract tests import MLX lazily."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Mlx


import subprocess
import sys

import pytest


@pytest.fixture(scope="session")
def metal() -> Mlx:
    # Isolate native crashes while keeping host-only tests independently runnable.
    """Require a working native Metal device for marked tests."""
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\nimport mlx.core as mx\n"
                "if not mx.metal.is_available():\n"
                "    sys.exit('MLX Metal GPU is unavailable; native tests require '"
                "'an available Metal GPU on macOS Apple Silicon')",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(
            f"Native MLX probe timed out after {error.timeout} seconds; "
            f"stdout={error.stdout!r}; stderr={error.stderr!r}",
            pytrace=False,
        )
    if probe.returncode != 0:
        pytest.fail(
            f"Native MLX probe failed (exit {probe.returncode}); "
            f"stdout={probe.stdout!r}; stderr={probe.stderr!r}",
            pytrace=False,
        )
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    return cast("Mlx", mx)


@pytest.fixture
def mx(metal: Mlx) -> Mlx:
    """Alias the admitted Metal runtime for numerical tests."""
    return metal
