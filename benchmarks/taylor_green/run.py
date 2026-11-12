#!/usr/bin/env python3
"""CASE-TAYLOR-GREEN-SPH benchmark driver (thin wrapper).

The case itself -- reference, parameter table, ref-vs-got rows, sensitivity
studies, figures, ``results.json`` / ``run.log`` / ``environment.json`` -- lives
in :mod:`tensorsph.taylor_green`, so that the benchmark, the package CLI
(``python -m tensorsph --case taylor_green``) and the verification ladder
(``tensorsph.vv.ladder``) all gate the *same* numbers.  This script only keeps
the historical command line of the standalone driver:

    PYTHONPATH=src python benchmarks/taylor_green/run.py --out runs/taylor_green

Full study (3 resolutions + 5 Mach numbers + 4 Reynolds numbers + figures) takes
roughly three minutes on four threads; ``--quick`` is the CI-sized subset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.taylor_green import CaseSpec, run_benchmark  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "runs" / "taylor_green"))
    parser.add_argument("--dx", default=None,
                        help="comma-separated resolutions for the convergence study "
                             "(default: 0.05,0.025,0.0125)")
    parser.add_argument("--mach", default="0.002,0.005,0.01,0.02,0.05")
    parser.add_argument("--reynolds", default="2,5,10,20")
    parser.add_argument("--main-dx", type=float, default=0.025)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="CI-sized run (two resolutions, three Mach numbers, "
                             "two Reynolds numbers)")
    args = parser.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)

    spec = CaseSpec()
    if args.t_end is not None:
        spec.t_end = args.t_end

    if args.quick:
        dxs, machs, reynolds, main_dx = (0.05, 0.025), (0.005, 0.02, 0.05), (2.0, 10.0), 0.05
    else:
        dxs = (tuple(float(v) for v in args.dx.split(",")) if args.dx else spec.dxs)
        machs = tuple(float(v) for v in args.mach.split(","))
        reynolds = tuple(float(v) for v in args.reynolds.split(","))
        main_dx = args.main_dx

    report = run_benchmark(Path(args.out), spec, dxs=dxs, machs=machs,
                           reynolds=reynolds, figures=not args.no_figures,
                           main_dx=main_dx)
    print("\nwrote %s" % (Path(args.out) / "results.json"))
    print("main case: dx=%g m, %d particles, decay-rate error %.2f %% "
          "(tolerance %.0f %%) -> %s"
          % (report["main_case"]["dx_m"], report["main_case"]["particles"],
             report["main_case"]["error_pct"], report["tolerance"]["limit_pct"],
             "PASS" if report["tolerance"]["pass"] else "FAIL"))
    return 0 if report["tolerance"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())