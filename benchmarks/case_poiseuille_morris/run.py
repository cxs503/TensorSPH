#!/usr/bin/env python3
"""CASE-SPH-POISEUILLE-MORRIS -- benchmark driver.

The case lives in :mod:`tensorsph.case_poiseuille_morris` (reference solution,
channel geometry, WCSPH + mirror-wall integration, metrics).  This script is the
benchmark-pool entry point: it runs that module against this directory's
``out/`` so the artifact sits next to the case, and repeats the reproduction
command in ``out/summary.md``.

    PYTHONPATH=src:. python benchmarks/case_poiseuille_morris/run.py \
        --resolutions 8 12 16 24
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(REPO / "src"), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

from tensorsph.case_poiseuille_morris import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--out" not in argv:
        argv = ["--out", str(HERE / "out"), *argv]
    raise SystemExit(main(argv))
