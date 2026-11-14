#!/usr/bin/env python3
"""Diagnostic 2: is the near-wall shear deficit attached to the MOVING wall, or to
the top of the domain?  Mirrors the case: moving wall at the bottom, fixed wall
at the top.  Same solver, same lattice, only ``y -> H - y``.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
RUN = Path(__file__).resolve().parent / "run.py"
spec = importlib.util.spec_from_file_location("couette_run", RUN)
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def make_walls(shifted: bool):
    """shifted=True -> moving wall at y=0, fixed wall at y=H (mirror image)."""
    def build(cfg, height, length, layers, wall_speed, device):
        pos, vel = _orig(cfg, height, length, layers, wall_speed, device)
        if not shifted:
            return pos, vel
        y = pos[:, 1].clone()
        # flip geometry; each particle keeps its own velocity, so the band that
        # used to slide at the top is now the bottom wall, still sliding at U.
        return torch.stack((pos[:, 0], height - y), dim=-1), vel
    return build


_orig = cr.build_walls


def run(res, vt, shifted, threads=1):
    torch.set_num_threads(threads)
    dev = torch.device("cpu")
    dx = cr.HEIGHT / res
    length = cr.COLUMNS * dx
    cfg = cr.make_config(dx, cr.VISCOSITY, cr.FRICTION, dev)
    t_end = vt * cr.HEIGHT ** 2 / cr.VISCOSITY
    cr.build_walls = make_walls(shifted)
    try:
        case = cr.run_case(cfg, cr.HEIGHT, length, cr.WALL_SPEED, t_end,
                           cr.AVERAGE_FRACTION * t_end, dev)
    finally:
        cr.build_walls = _orig
    prof = case["profile"]
    m = cr.profile_metrics(prof, cr.HEIGHT, cr.WALL_SPEED, cr.VISCOSITY)
    U = cr.WALL_SPEED
    inc = [prof[k + 1]["u_mean"] / U - prof[k]["u_mean"] / U
           for k in range(len(prof) - 1)]
    return {
        "shifted": shifted, "res": res, "vt": vt,
        "drift": case["steady_drift"],
        "wall_shear_ratio_bottom": m["wall_shear_ratio"],
        "wall_shear_ratio_top": m["wall_shear_ratio_top"],
        "first_increments": inc[:3], "last_increments": inc[-3:],
        "analytic_increment": 1.0 / res,
        "row0": prof[0]["u_mean"] / U, "rowN": prof[-1]["u_mean"] / U,
        "row0_ref": 0.5 / res, "rowN_ref": 1 - 0.5 / res,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=8)
    ap.add_argument("--vt", type=float, default=1.2)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    for shifted in (False, True):
        r = run(args.res, args.vt, shifted, args.threads)
        print(f"shifted={r['shifted']} res={r['res']} vt={r['vt']} drift={r['drift']:.4f} "
              f"tau_bottom={r['wall_shear_ratio_bottom']:.4f} tau_top={r['wall_shear_ratio_top']:.4f}")
        print(f"  increments analytic={r['analytic_increment']:.5f} "
              f"first3={[round(x,5) for x in r['first_increments']]} "
              f"last3={[round(x,5) for x in r['last_increments']]}")
        print(f"  row0={r['row0']:.5f} (ref {r['row0_ref']:.5f})  "
              f"rowN={r['rowN']:.5f} (ref {r['rowN_ref']:.5f})")


if __name__ == "__main__":
    main()