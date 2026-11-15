#!/usr/bin/env python3
"""Time-to-steady probe for the Navier-slip Poiseuille rung.

The steady-state gates of ``benchmarks/poiseuille_slip/run.py`` are only
meaningful once the transient has died out.  The slowest decaying mode of the
channel keeps the *shape* of the profile and only scales its amplitude, so an
under-converged run shows up as a near-uniform relative deficit -- easy to
misread as a boundary-condition error.  This probe integrates further than the
production ``t_end`` and prints the error against the closed form at several
horizons, plus the drift per 0.25 s window, so the production ``t_end`` is
chosen from evidence rather than by hope.

Usage
-----
    PYTHONPATH=src python benchmarks/poiseuille_slip/probe_convergence.py \
        --resolution 12 --slip-ratio 0.25 --t-ends 1 2 4 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from benchmarks.poiseuille.run import make_config
from benchmarks.poiseuille_slip.run import evaluate, run_case_slip

REPO = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolution", type=int, default=12)
    p.add_argument("--slip-ratio", type=float, default=0.25)
    p.add_argument("--t-ends", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    p.add_argument("--height", type=float, default=0.1)
    p.add_argument("--viscosity", type=float, default=0.01)
    p.add_argument("--body-force", type=float, default=9.81)
    p.add_argument("--columns", type=int, default=12)
    p.add_argument("--t-average", type=float, default=0.2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path,
                   default=REPO / "benchmarks/poiseuille_slip/out/probe.json")
    args = p.parse_args(argv)

    torch.set_num_threads(args.threads)
    dx = args.height / args.resolution
    cfg = make_config(dx, args.viscosity, 1.0, "cpu")
    lam = args.slip_ratio * args.height

    out = []
    for t_end in args.t_ends:
        case = run_case_slip(cfg, args.height, args.columns * dx,
                             args.body_force, t_end, args.t_average, "cpu", lam)
        res = evaluate(case, args.height, args.body_force, args.viscosity,
                       1000.0, args.resolution, lam)
        row = {
            "t_end": t_end,
            "u_max_error_pct": res["u_max_error_pct"],
            "profile_rms_interior_pct": res["profile_rms_interior_pct"],
            "wall_shear_error_pct": res["wall_shear_bottom_error_pct"],
            "u_slip_error_pct": res["u_slip_error_pct"],
            "flux_enhancement_error_pct": res["flux_enhancement_error_pct"],
            "steady_drift_last_quarter": res["steady_drift"],
            "max_mach": res["max_mach"],
            "steps": res["steps"],
            "wall_clock_s": res["wall_clock_s"],
            "trace": case["trace"],
        }
        out.append(row)
        print(f"t_end={t_end:5.2f}  u_max {row['u_max_error_pct']:+7.2f}%  "
              f"rms {row['profile_rms_interior_pct']:6.2f}%  "
              f"tau {row['wall_shear_error_pct']:+7.2f}%  "
              f"slip {row['u_slip_error_pct']:+7.2f}%  "
              f"flux {row['flux_enhancement_error_pct']:+7.2f}%  "
              f"drift {row['steady_drift_last_quarter']:.4f}  "
              f"({row['wall_clock_s']:.0f}s)", flush=True)

    payload = {
        "case": "navier_slip_poiseuille_time_to_steady",
        "resolution_H_over_dx": args.resolution,
        "slip_ratio_lam_over_H": args.slip_ratio,
        "viscosity": args.viscosity,
        "body_force": args.body_force,
        "height": args.height,
        "runs": out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())