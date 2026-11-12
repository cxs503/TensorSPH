#!/usr/bin/env python
"""CASE-SPH-SOD-SHOCK-COMPRESSIBLE — pool entry (TensorSPH).

Runs the compressible SPH Sod shock tube at t=0.2 and prints the ref-vs-got
table with the 5% acceptance verdict.

    python benchmarks/case_sod_shock_compressible/run.py [--n 800] [--h-factor 2.5]

Writes ``result.json`` next to this file (metrics + profiles) unless --no-write.
Reference: Sod (1978) J. Comput. Phys. 27(1):1-31, doi:10.1016/0021-9991(78)90023-2.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from tensorsph.compressible import sod as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--h-factor", type=float, default=2.5)
    ap.add_argument("--t-end", type=float, default=0.2)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    cfg = S.SodSPHConfig(n_particles=args.n, h_factor=args.h_factor, t_end=args.t_end)
    res = S.run_sod_sph(cfg, output_times=[args.t_end])
    met = S.evaluate_metrics(res["snapshots"][-1], cfg)

    print(f"{'metric':22s} {'got':>10s} {'exact':>10s} {'rel.err':>9s}")
    for key in ("rho_L", "rho_star_L", "rho_star_R", "rho_R", "p_L",
                "p_star_L", "p_star_R", "p_R", "u_star", "x_contact", "x_shock"):
        ex_key = {"p_star_L": "p_star", "p_star_R": "p_star"}.get(key, key)
        print(f"{key:22s} {met['got'][key]:10.4f} {met['exact'][ex_key]:10.5f} "
              f"{met['rel_err'][key]*100:8.2f}%")
    print(f"max |rel err| = {met['max_abs_rel_err']*100:.2f}%  "
          f"<5%: {met['pass_5pct']}  <3%: {met['pass_3pct']}  "
          f"L1(rho)={met['l1']['rho']*100:.2f}%  steps={res['steps']}")

    if not args.no_write:
        out = {"case": "CASE-SPH-SOD-SHOCK-COMPRESSIBLE",
               "reference": "Sod 1978 JCP 27(1):1-31 doi:10.1016/0021-9991(78)90023-2",
               "config": res["config"], "steps": res["steps"],
               "metrics": {k: v for k, v in met.items() if k != "windows"},
               "profiles": {k: np.asarray(v).tolist()
                            for k, v in res["snapshots"][-1].items()}}
        path = os.path.join(HERE, "result.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        print("wrote", path)


if __name__ == "__main__":
    main()
