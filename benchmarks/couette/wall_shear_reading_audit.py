#!/usr/bin/env python3
"""Audit of the Couette wall-shear *reading* behind the method-selection record.

Anchor: ``XVAL-SPH-METHOD-SELECTION-2026-11-12``.

The registry banks, for ``wcsph_wall_no_slip``, the *sliding*-wall reading

    H/dx = 8  : 14.20 %   (``x_couette_viscous_time_top_h8``)
    H/dx = 16 : 17.08 %   (``x_couette_viscous_time_top_h16``)   -> fails the
                                                            owner 15 % standard

while the *fixed*-wall reading on the same runs is 9.55 % -> 1.81 %.  A reading that
gets worse when the mesh is refined is not a discretisation error, so three
questions follow, and this script answers them from the same runs that produced the
records:

1. **Reproduction** -- re-run the exact code path (``run.py:run_case``), do the
   banked percentages come back?
2. **Attribution** -- is the deficit a property of the *sliding* wall or of the top
   of the domain?  Answered by a mirror arm: same lattice flipped in ``y``
   (``y -> H-y``, each particle keeps its own velocity), so the sliding band sits at
   ``y = 0``.
3. **Estimator sensitivity** -- the banked reading is a one-spacing secant
   ``(u[N-1]-u[N-2])/dy``.  The same profile is re-read with a 4-row and an 8-row
   least-squares slope and with the interior-band slope, all normalised by ``U/H``.
   If the one-spacing secant is dominated by the last-row lag, the four numbers
   separate: that is what decides whether the 17 % is a wall-model defect or an
   estimator artefact.

Only the solver's own ``run_case`` is used, so a disagreement cannot come from a
re-implementation of the time loop.

Run::

    PYTHONPATH=src python benchmarks/couette/wall_shear_reading_audit.py --res 8
    PYTHONPATH=src python benchmarks/couette/wall_shear_reading_audit.py --res 16 --only normal
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
_RUN = Path(__file__).resolve().parent / "run.py"
_spec = importlib.util.spec_from_file_location("couette_run", _RUN)
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)

OUT = REPO / "benchmarks" / "couette" / "out_readings"
_ORIG_BUILD_WALLS = cr.build_walls


def _mirrored_walls(cfg, height, length, layers, wall_speed, device):
    """Wall bands flipped in ``y``; each particle keeps its velocity, so the sliding
    band ends up at ``y = 0``."""
    pos, vel = _ORIG_BUILD_WALLS(cfg, height, length, layers, wall_speed, device)
    return torch.stack((pos[:, 0], height - pos[:, 1]), dim=-1), vel


def _slope_ratio(rows, slope_ex: float) -> float:
    ys = torch.tensor([r["y"] for r in rows], dtype=torch.float64)
    us = torch.tensor([r["u_mean"] for r in rows], dtype=torch.float64)
    yc = ys - ys.mean()
    slope = float((yc * (us - us.mean())).sum() / (yc * yc).sum())
    return abs(slope) / slope_ex


def run_arm(res: int, viscous_times: float, mirrored: bool, threads: int = 1) -> dict:
    torch.set_num_threads(threads)
    dev = torch.device("cpu")
    dx = cr.HEIGHT / res
    length = cr.COLUMNS * dx
    cfg = cr.make_config(dx, cr.VISCOSITY, cr.FRICTION, dev)
    t_end = viscous_times * cr.HEIGHT ** 2 / cr.VISCOSITY
    builder = _mirrored_walls if mirrored else cr.build_walls
    original = _ORIG_BUILD_WALLS
    cr.build_walls = builder
    try:
        case = cr.run_case(cfg, cr.HEIGHT, length, cr.WALL_SPEED, t_end,
                           cr.AVERAGE_FRACTION * t_end, dev)
    finally:
        cr.build_walls = original
    profile = case["profile"]
    m = cr.profile_metrics(profile, cr.HEIGHT, cr.WALL_SPEED, cr.VISCOSITY)
    slope_ex = cr.WALL_SPEED / cr.HEIGHT
    U = cr.WALL_SPEED

    near = list(reversed(profile[:8])) if mirrored else list(profile[-8:])
    interior = [r for r in profile if 0.15 <= r["y_over_H"] <= 0.85]
    secant_slide = abs(m["wall_shear_ratio"] if mirrored else m["wall_shear_ratio_top"])
    secant_fixed = abs(m["wall_shear_ratio_top"] if mirrored else m["wall_shear_ratio"])
    increments = [
        float(profile[k + 1]["u_mean"] - profile[k]["u_mean"]) / U * res
        for k in range(len(profile) - 1)
    ]
    return {
        "resolution_H_over_dx": res,
        "mirrored": mirrored,
        "sliding_wall_at": "y=0" if mirrored else "y=H",
        "steps": case["steps"],
        "particles": case["n_fluid"],
        "dt": case["dt"],
        "samples": case["n_averaged_samples"],
        "steady_drift": case["steady_drift"],
        "interior_l2_pct": 100.0 * m["interior_l2"],
        "flux_ratio": m["flux_ratio"],
        "secant1_sliding_pct": 100.0 * (secant_slide - 1.0),
        "secant1_fixed_pct": 100.0 * (secant_fixed - 1.0),
        "fit4_sliding_pct": 100.0 * (_slope_ratio(near[:4], slope_ex) - 1.0),
        "fit8_sliding_pct": 100.0 * (_slope_ratio(near[:8], slope_ex) - 1.0),
        "interior_slope_pct": 100.0 * (_slope_ratio(interior, slope_ex) - 1.0),
        "gradient_over_analytic_per_spacing": increments,
        "profile": [{"y_over_H": r["y_over_H"], "u_over_U": r["u_mean"] / U}
                    for r in profile],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs="+", default=[8])
    ap.add_argument("--viscous-times", type=float, default=1.2)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--only", choices=["both", "normal", "mirror"], default="both")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    arms = []
    wanted = {"both": (False, True), "normal": (False,), "mirror": (True,)}[args.only]
    for res in args.res:
        for mirrored in wanted:
            t0 = time.time()
            rec = run_arm(res, args.viscous_times, mirrored, args.threads)
            rec["wall_clock_s"] = time.time() - t0
            arms.append(rec)
            print(f"res={res:2d} mirror={str(mirrored):5s} "
                  f"slide: secant1={rec['secant1_sliding_pct']:7.2f} "
                  f"fit4={rec['fit4_sliding_pct']:7.2f} "
                  f"fit8={rec['fit8_sliding_pct']:7.2f} "
                  f"interior={rec['interior_slope_pct']:7.2f} % | "
                  f"fixed secant1={rec['secant1_fixed_pct']:6.2f} % | "
                  f"l2={rec['interior_l2_pct']:5.2f} % Q={rec['flux_ratio']:.4f} "
                  f"drift={rec['steady_drift']:.4f}", flush=True)

    out = args.out or (OUT / f"wall_shear_reading_audit_res{args.res[0]}_{args.only}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # merge with any arms already recorded for the same case so the JSON accumulates
    existing = []
    merged = OUT / "wall_shear_reading_audit.json"
    if merged.exists():
        try:
            existing = [a for a in json.loads(merged.read_text())["arms"]
                        if (a["resolution_H_over_dx"], a["mirrored"]) not in
                        {(r["resolution_H_over_dx"], r["mirrored"]) for r in arms}]
        except Exception:
            existing = []
    doc = {
        "case": "plane_couette_channel",
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "generated_by": "benchmarks/couette/wall_shear_reading_audit.py",
        "quantity": "wall-shear reading |1 - slope_measured/slope_analytic| [%] at the "
                    "sliding wall of the plane Couette channel, four estimators on the "
                    "SAME profile, plus a mirror arm that moves the sliding wall from "
                    "y=H to y=0 at identical lattice",
        "setup": {
            "height_m": cr.HEIGHT, "viscosity_m2_s": cr.VISCOSITY,
            "wall_speed_m_s": cr.WALL_SPEED, "sound_speed_m_s": cr.SOUND_SPEED,
            "wall_friction": cr.FRICTION, "columns": cr.COLUMNS,
            "wall_layers": cr.WALL_LAYERS, "viscous_times": args.viscous_times,
            "averaging_fraction": cr.AVERAGE_FRACTION,
            "banked_records": {"H_over_dx_8": 14.203286170959451,
                               "H_over_dx_16": 17.076468467712292},
        },
        "estimators": {
            "secant1": "banked estimator (run.py profile_metrics): "
                       "(u[N-1]-u[N-2])/(y[N-1]-y[N-2])",
            "fit4": "least squares over the 4 rows nearest the sliding wall",
            "fit8": "least squares over the 8 rows nearest the sliding wall",
            "interior": "least squares over 0.15 <= y/H <= 0.85",
        },
        "gate_pct": 5.0,
        "owner_tolerance_pct": 15.0,
        "arms": sorted(existing + arms,
                       key=lambda r: (r["resolution_H_over_dx"], r["mirrored"])),
    }
    merged.write_text(json.dumps(doc, indent=1) + "\n")
    if out != merged:
        out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {merged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())