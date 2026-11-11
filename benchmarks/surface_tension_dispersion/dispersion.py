"""Deprecated entry point of the surface-tension dispersion benchmark (thin shim).

The original harness in this directory (m13895) measured a **closed** fluid block:
four free surfaces (wave on top, free bottom, two free sides) and a non-periodic
``x``.  Every one of those interfaces carries its own ``-sigma kappa delta_s n``
force, and their projection onto ``cos(k x)`` was larger than the wave signal, so
the reported stiffness had the wrong sign and the reported ``k = 1, 2, 3`` rows
were three copies of the same problem (``dx = lambda / ppw`` makes the setup
scale invariant).  The measurement now lives in the library:

    tensorsph.surface_tension_dispersion      (implementation + audits)
    scripts/run_surface_tension_dispersion.py (entry point, writes runs/)

This file is kept so old command lines keep working; it forwards to the library.
The defective geometry is still reproducible, but only through the explicit
``--defective-control`` switch of :func:`tensorsph.surface_tension_dispersion.main`.

    PYTHONPATH=src python benchmarks/surface_tension_dispersion/dispersion.py --quick
"""
from __future__ import annotations

import sys

from tensorsph.surface_tension_dispersion import main as _main


def main(argv=None) -> dict:
    return _main(argv)


if __name__ == "__main__":
    out = main(sys.argv[1:])
    raise SystemExit(0 if out["status"] in {"PASS", "FAIL"} else 1)