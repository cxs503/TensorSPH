#!/usr/bin/env python3
"""Diagnostic: why is the moving-wall shear reading biased low (~0.82) while the
fixed-wall reading converges to ~0.99?

Prints, per resolution:
  * the measured rows near both walls (u/U vs y/H),
  * secant slopes from several row pairs,
  * the interior line-fit slope,
  * the per-row error.
Run: PYTHONPATH=src python benchmarks/couette/diag_wall_asym.py --res 16 --vt 1.2
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
RUN = Path(__file__).resolve().parent / "run.py"
spec = importlib.util.spec_from_file_location("couette_run", RUN)
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=16)
    ap.add_argument("--vt", type=float, default=1.2)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device("cpu")

    res = args.res
    dx = cr.HEIGHT / res
    length = cr.COLUMNS * dx
    cfg = cr.make_config(dx, cr.VISCOSITY, cr.FRICTION, dev)
    t_end = args.vt * cr.HEIGHT ** 2 / cr.VISCOSITY
    t_avg = cr.AVERAGE_FRACTION * t_end
    case = cr.run_case(cfg, cr.HEIGHT, length, cr.WALL_SPEED, t_end, t_avg, dev)
    prof = case["profile"]
    m = cr.profile_metrics(prof, cr.HEIGHT, cr.WALL_SPEED, cr.VISCOSITY)
    U = cr.WALL_SPEED
    print(f"res={res} vt={args.vt} n={case['n_fluid']} steps={case['steps']} "
          f"drift={case['steady_drift']:.4f}")
    print("row  y/H      u/U      u_ref/U   rel_err%")
    for p, r in zip(prof, m["rows"]):
        print(f"{p['row'] if 'row' in p else '':>3} {p['y_over_H']:6.4f} "
              f"{p['u_mean']/U:9.5f} {r['u_ref']/U:9.5f} {100*r['rel_err']:8.3f}")
    s = cr.analytic_scalars(cr.HEIGHT, U, cr.VISCOSITY)["slope"]
    def sec(k0, k1):
        return (prof[k1]["u_mean"] - prof[k0]["u_mean"]) / (prof[k1]["y"] - prof[k0]["y"]) / s
    n = len(prof)
    print("secant ratios (slope/analytic):")
    for k in range(0, min(4, n - 1)):
        print(f"  bottom rows {k},{k+1}: {sec(k, k+1):.4f}")
    for k in range(max(0, n - 4), n - 1):
        print(f"  top rows {k},{k+1}: {sec(k, k+1):.4f}")
    print(f"interior line fit slope ratio: {m['line_fit_slope_ratio']:.4f}")
    print(f"flux ratio: {m['flux_ratio']:.4f}  interior_l2: {m['interior_l2']:.4f}")
    print(f"wall_shear_ratio bottom={m['wall_shear_ratio']:.4f} top={m['wall_shear_ratio_top']:.4f}")
    print(f"slip/dx bottom={m['slip_length_bottom_over_dx']:.3f} top={m['slip_length_top_over_dx']:.3f}")


if __name__ == "__main__":
    main()