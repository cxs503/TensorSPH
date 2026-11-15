#!/usr/bin/env python3
"""CASE-DUCT-POISEUILLE-3D -- benchmark entry point.

    PYTHONPATH=src:. python benchmarks/case_poiseuille_duct/run.py --reference
    PYTHONPATH=src:. python benchmarks/case_poiseuille_duct/run.py --only square-6 --steps 900
    PYTHONPATH=src:. python benchmarks/case_poiseuille_duct/run.py --merge

The case lives in :mod:`tensorsph.case_poiseuille_duct`; this script only pins
the artifact directory next to the case.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (str(REPO / "src"), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

from tensorsph.case_poiseuille_duct import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--out" not in argv:
        argv = ["--out", str(HERE / "out"), *argv]
    raise SystemExit(main(argv))
