#!/usr/bin/env python
"""Script wrapper for ``python -m tensorsph.verification`` (campaign P482-N04).

    python scripts/run_sph_free_surface_verification.py --tier full
    python scripts/run_sph_free_surface_verification.py --tier smoke --out /tmp/sph

The runner itself lives in :mod:`tensorsph.verification.cli` so that the same
code is reachable from the package, from this script and from the tests.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorsph.verification.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
