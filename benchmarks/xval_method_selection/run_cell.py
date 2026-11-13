"""Run one cell of the method-selection repair matrix.

The original matrix (``grade_matrix.MATRIX``) asks for seven cells of the
2026-11-12 method A.  The audit of mission 17501 found that none of them had
been executed and that the only surviving record diverged, so the repair run is
*decision-focused* rather than a full grid: three cells, each answering one
question of the pre-registered protocol (``docs/xval_method_selection_m17501/``).

===================  =========================================================  ===============
index  label          question                                                   protocol ref
===================  =========================================================  ===============
0      D20_h025_A     does method A reach the 15 % gate when the record is long    H1, G1
                      enough to grade (t_end = 5.5 s, windows 4.0/5.0 s)?
1      D13_h025_A     same at the coarse end of the ladder (H/dx = 2.08) -         H1, ladder
                      is the failure resolution-driven?
2      D20_h025_xsph  is the growth a tensile/clumping instability (does XSPH     H3 (mechanism
                      velocity smoothing in the propagation phase remove it)?      probe, post-hoc)
3      D20_h025_comp  does SPHinXsys-style compression regularization fix the     H4 (mechanism
                      divergence that δ-SPH cannot? (δ-SPH off, compression on)   probe, post-hoc)
===================  =========================================================  ===============

Usage (one cell per process, so the cluster can shard them)::

    PYTHONPATH=src python benchmarks/xval_method_selection/run_cell.py 0 --out runs/xval/<label>.json

Cell 2 and 3 are marked *post-hoc*: the pre-registered protocol did not include
a propagation-phase XSPH coefficient or compression regularization, so their
outcomes are diagnostic evidence, not a pre-registered hypothesis test.  They
are recorded as such in the report.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_PY = REPO / "benchmarks" / "wave_flume" / "run.py"

# Common case parameters: the pre-registered geometry of the anchor
# (docs/xval_method_selection_m17501/protocol.md §4): h = 0.25 m, L = 3.2 m,
# T = 1.0 s, H/h = 0.16, c0 = 30 m/s, nu = 1e-6, settle window 0.8 s.
COMMON = [
    "--case", "regular",
    "--depth", "0.25",
    "--height-ratio", "0.16",
    "--wave-period", "1.0",
    "--flume-length", "3.2",
    "--sound-speed", "30",
    "--viscosity", "1e-6",
    "--density-diffusion", "0.1",
    "--sponge-strength", "4.0",
    "--settle", "0.8",
    "--viscosity-settle", "0.02",
    "--xsph-settle", "0.1",
    "--ramp-periods", "2.0",
    "--simulate-time", "5.5",
    "--threads", "8",
]

CELLS = {
    0: ("D20_h025_A", ["--depth-cells", "20"]),
    1: ("D13_h025_A", ["--depth-cells", "13"]),
    2: ("D20_h025_xsph", ["--depth-cells", "20", "--xsph", "0.05"]),
    # Cell 3: SPHinXsys-style compression regularization instead of δ-SPH.
    # This tests whether the divergence is caused by the density diffusion
    # formulation or by something more fundamental in the WCSPH solver.
    3: ("D20_h025_comp", [
        "--depth-cells", "20",
        "--density-diffusion", "0",
        "--density-regularization", "compression",
        "--compression-rule", "free_surface",
    ]),
}


def _run_main():
    spec = importlib.util.spec_from_file_location("wave_flume_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int, choices=sorted(CELLS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    label, extra = CELLS[args.index]
    print(f"cell {args.index}: {label} -> {args.out}", flush=True)
    return _run_main()(COMMON + extra + ["--out", args.out])


if __name__ == "__main__":
    raise SystemExit(main())
