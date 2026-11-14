#!/usr/bin/env python3
"""Start-up of plane Poiseuille flow from rest: the Boussinesq series.

XVAL-SPH-ROUND2-METHOD-SELECTION-STARTUP-SERIES

Physics / reference
-------------------
Fluid at rest in a 2-D channel of height ``H``, walls at ``y = 0, H``, no-slip,
periodic in ``x``, uniform body force ``g_x`` switched on at ``t = 0``.
The 1-D problem is ``u_t = nu u_yy + g`` with ``u(y,0) = 0``, ``u(0,t) =
u(H,t) = 0``.  Its exact solution (Boussinesq 1868; Batchelor, *An Introduction
to Fluid Dynamics*, §4.2; Morris, Fox & Zhu, *J. Comput. Phys.* 136 (1997) 214) is

    u_s(y)  = g (H - y) y / (2 nu),              u_max,inf = g H^2 / (8 nu)
    u(y,t)  = u_s(y) - (32 u_max,inf / pi^3) sum_{k>=0}
                 sin(n pi y / H) / n^3 * exp(- n^2 pi^2 nu t / H^2),   n = 2k+1.

Usage
-----
    PYTHONPATH=src:. python benchmarks/poiseuille_startup/run.py \
        --resolutions 8 12 16 --out benchmarks/poiseuille_startup/out
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.poiseuille.run import (
    bin_rows,
    build_fluid,
    fit_profile,
    make_config,
    periodic_images,
)
from tensorsph.droplet_impact import resolve_device, rhs


def boussinesq_series(y, t, height: float, body_force: float,
                      viscosity: float, n_terms: int = 200):
    """Exact start-up from rest of the plane channel under a constant body force."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    steady = body_force * y * (height - y) / (2.0 * viscosity)

    k = np.arange(1, 2 * n_terms, 2, dtype=np.float64).reshape(-1, 1)
    yy = y.reshape(1, -1)
    decay = np.exp(-viscosity * k ** 2 * math.pi ** 2 * t / height ** 2)
    series = (np.sin(k * math.pi * yy / height) * decay / k ** 3).sum(axis=0)
    return steady - 4.0 * body_force * height ** 2 / (viscosity * math.pi ** 3) * series


def run_startup_case(cfg, height: float, length: float, body_force: float,
                     t_end: float, sample_times, device):
    """Integrate from rest and sample the velocity profile at target times."""
    dx = cfg.dx
    pos, mass = build_fluid(cfg, height, length, device)
    n = pos.shape[0]
    wall = build_mirror_walls(cfg, height, length, device)

    vel = torch.zeros_like(pos)
    h = cfg.h_factor * dx
    dt_acoustic = 0.3 * h / cfg.sound_speed
    dt_viscous = 0.125 * h * h / cfg.viscosity
    dt = min(dt_acoustic, dt_viscous)
    steps = int(round(t_end / dt))

    sample_window = 0.005
    samples_at_time = {t: {"u_sum": torch.zeros(n, device=device), "count": 0}
                       for t in sample_times}

    trace = []
    wall_clock = time.time()

    for step in range(steps):
        pos_ext = periodic_images(pos, length)
        vel_ext = torch.cat((vel, vel, vel), dim=0)
        mass_ext = torch.cat((mass, mass, mass), dim=0)
        _, _, acc = rhs(pos_ext, vel_ext, mass_ext, cfg, wall=wall)
        acc = acc[:n].clone()
        acc[:, 0] += body_force

        t = step * dt

        for st in sample_times:
            if abs(t - st) <= sample_window:
                samples_at_time[st]["u_sum"] += vel
                samples_at_time[st]["count"] += 1

        vel = vel + dt * acc
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], length)

        if step % max(1, steps // 12) == 0 or step == steps - 1:
            ux = vel[:, 0]
            trace.append({
                "t": t,
                "u_max": float(ux.max()),
                "u_mean": float(ux.mean()),
            })

    profiles = {}
    for st in sample_times:
        if samples_at_time[st]["count"] > 0:
            u_mean = samples_at_time[st]["u_sum"] / samples_at_time[st]["count"]
            rows, profile = bin_rows(pos, u_mean, height, dx)
            profiles[st] = {"rows": rows, "profile": profile}
        else:
            profiles[st] = None

    return {
        "n_fluid": int(n),
        "dt": dt,
        "steps": steps,
        "profiles": profiles,
        "trace": trace,
        "wall_clock_s": time.time() - wall_clock,
    }


def build_mirror_walls(cfg, height: float, length: float, device):
    """Mirror (ghost) walls: velocity negated, no static particles."""
    return torch.empty((0, 2), device=device)


def compare_profile(profile, t: float, height: float, body_force: float,
                    viscosity: float, lo: float = 0.1, hi: float = 0.9):
    """Compare a binned SPH profile against the Boussinesq series."""
    rows = profile["rows"]
    u_vals = profile["profile"]

    interior = [(r, u) for r, u in zip(rows, u_vals) if lo <= r / height <= hi]
    if not interior:
        return {"rms_rel": None, "rms_abs": None, "centre_rel": None}

    y_int = np.array([r for r, _ in interior])
    u_int = np.array([u for _, u in interior])

    u_series = boussinesq_series(y_int, t, height, body_force, viscosity)
    u_max_inf = body_force * height ** 2 / (8.0 * viscosity)

    diff = u_int - u_series
    rms = float(np.sqrt(np.mean(diff ** 2)))
    max_series = float(np.max(np.abs(u_series)))

    rms_rel = rms / max_series if max_series > 0 else None
    rms_abs = rms / u_max_inf

    centre_idx = int(np.argmin(np.abs(y_int - 0.5 * height)))
    centre_measured = float(u_int[centre_idx])
    centre_exact = float(u_series[centre_idx])
    centre_rel = abs(centre_measured - centre_exact) / abs(centre_exact) if abs(centre_exact) > 0 else None

    return {
        "rms_rel": rms_rel,
        "rms_abs": rms_abs,
        "centre_rel": centre_rel,
        "centre_measured": centre_measured,
        "centre_exact": centre_exact,
        "max_series": max_series,
        "u_max_inf": u_max_inf,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--viscosity", type=float, default=0.01)
    ap.add_argument("--body-force", type=float, default=1.0)
    ap.add_argument("--sample-times", type=float, nargs="+",
                    default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ap.add_argument("--out", type=Path, default=Path("benchmarks/poiseuille_startup/out"))
    args = ap.parse_args(argv)

    device = resolve_device("cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    height = 0.1
    u_max_inf = args.body_force * height ** 2 / (8.0 * args.viscosity)

    results = []
    for res in args.resolutions:
        print(f"\nRunning resolution H/dx = {res} ...")
        dx = height / res
        cfg = make_config(dx, args.viscosity, friction=0.0, device=device)

        result = run_startup_case(cfg, height, 0.4, args.body_force,
                                  t_end=args.sample_times[-1] + 0.01,
                                  sample_times=args.sample_times,
                                  device=device)

        comparisons = {}
        for t in args.sample_times:
            if result["profiles"][t] is not None:
                comp = compare_profile(result["profiles"][t], t, height,
                                       args.body_force, args.viscosity)
                comparisons[t] = comp
                print(f"  t={t:.1f}s: rms_rel={comp.get('rms_rel', 0)*100:.2f}%, "
                      f"centre_rel={comp.get('centre_rel', 0)*100:.2f}%")

        results.append({
            "resolution": res,
            "dx": dx,
            "n_fluid": result["n_fluid"],
            "dt": result["dt"],
            "steps": result["steps"],
            "comparisons": comparisons,
            "wall_clock_s": result["wall_clock_s"],
        })

    # Verdict
    criterion_pct = 15.0
    all_pass = True
    for r in results:
        for t, comp in r["comparisons"].items():
            if comp.get("rms_rel") is not None and comp["rms_rel"] > criterion_pct / 100:
                all_pass = False
            if comp.get("centre_rel") is not None and comp["centre_rel"] > criterion_pct / 100:
                all_pass = False

    verdict = {
        "criterion_pct": criterion_pct,
        "pass": all_pass,
        "max_rms_rel_pct": max(
            (c["rms_rel"] * 100 for r in results for c in r["comparisons"].values() if c.get("rms_rel") is not None),
            default=0
        ),
        "max_centre_rel_pct": max(
            (c["centre_rel"] * 100 for r in results for c in r["comparisons"].values() if c.get("centre_rel") is not None),
            default=0
        ),
    }

    payload = {
        "case": "poiseuille_startup",
        "title": "Plane Poiseuille start-up vs the Boussinesq series",
        "config": {
            "height_m": height,
            "viscosity_m2_s": args.viscosity,
            "body_force_m_s2": args.body_force,
            "resolutions": args.resolutions,
            "sample_times": args.sample_times,
        },
        "u_max_inf": u_max_inf,
        "verdict": verdict,
        "results": results,
    }

    out_path = args.out / "results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Verdict: pass={all_pass}, max_rms_rel={verdict['max_rms_rel_pct']:.2f}%, "
          f"max_centre_rel={verdict['max_centre_rel_pct']:.2f}%")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
