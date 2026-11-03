"""Resolution study of the initial-transient density error (dx = 0.006 failure).

``docs/benchmarks/gpu-scaling.md`` records that the scaling benchmark's
``dx = 0.006`` case (N ~ 10,556) fails on both CPU and CUDA with a density
compression of 17 % against the 5 % gate, while ``dx = 0.020`` and
``dx = 0.009`` pass, and ``backlog_update_m281.md`` files it as a "resolution
problem, not a device bug" and asks for follow-up. This script does the
follow-up: it reproduces the case with the *same* configuration the scaling
benchmark builds, records the whole per-step history instead of the 50-step
summary, and separates the candidate mechanisms.

Why the summary in the scaling report cannot answer the question
----------------------------------------------------------------
``benchmarks/scaling/run.py`` integrates 50 steps at ``dt = 0.8 * suggested_dt``,
i.e. ``dt ~ dx``, so the simulated physical time also scales with ``dx``. At
``dx = 0.006`` those 50 steps cover 4.7 ms, which is before the drop has fallen
the 0.04 m gap to the pool (it needs ~90 ms). The 17 % compression therefore
cannot be a splash: it is produced by the release of a lattice-initialised
pool whose masses are uniform ``rho0 dx^2`` while the discrete equilibrium
density is not uniform near the free surface and the wall. The candidates are

  A. the initial condition (uniform mass vs. the free-surface truncation),
  B. the explicit time step (a Courant problem that only shows up at fine dx),
  C. a genuine instability of the free-surface/wall treatment that appears
     once the pool has more than a few tens of layers.

Each is testable by one override, and this script runs them as a matrix:
``--hydrostatic-init`` addresses A, ``--courant-scale 0.4`` addresses B, and the
``dx`` ladder itself addresses C.

Usage::

    python benchmarks/stability/run.py --steps 50
    python benchmarks/stability/run.py --steps 200 --sizes 0.020,0.009,0.006
    python benchmarks/stability/run.py --steps 50 --hydrostatic-init
    python benchmarks/stability/run.py --steps 50 --courant-scale 0.4

Output is a JSON report plus a one-line summary per case. Exit status is
non-zero when any case exceeds the density gate, so the failure is reproducible
as a test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tensorsph.droplet_impact import Config, run, suggested_dt

# The benchmark is single-core work. Under a loaded machine the default thread
# pool thrashes and a 50-step run can take minutes wall-clock for a few seconds
# of CPU; pinning it keeps the numbers reproducible.
torch.set_num_threads(1)


def scaling_config(dx: float, steps: int, courant_scale: float,
                   density_diffusion: float, hydrostatic_init: bool) -> Config:
    """The configuration ``benchmarks/scaling/run.py`` builds for one size."""
    cfg = Config(
        dx=dx,
        device="cpu",
        steps=steps,
        output_every=1,                     # full history, not the 50-step row
        fail_fast=False,                    # keep integrating past the gate
        density_diffusion=density_diffusion,
        hydrostatic_init=hydrostatic_init,
    )
    cfg.dt = suggested_dt(cfg) * courant_scale
    return cfg


def summarise(rows: list[dict], cfg: Config) -> dict:
    # The solver gate is on *compression*: ``rho.max()/rho0 - 1``. The density
    # excursion of a released lattice is dominated by the opposite sign -- a
    # free-surface particle has a truncated kernel support and sums to ~0.63
    # rho0 -- so the two are reported separately. Mixing them makes a steady
    # rarefaction look like a blow-up.
    comp = max(float(r["rho_max"]) / cfg.rho0 - 1.0 for r in rows)
    rare = min(float(r["rho_min"]) / cfg.rho0 - 1.0 for r in rows)
    peak = max(rows, key=lambda r: float(r["rho_max"]))
    peak_speed = max(rows, key=lambda r: r["max_speed"])
    peak_accel = max(rows, key=lambda r: r["max_acceleration"])

    return {
        "dx_m": cfg.dx,
        "particle_count": rows[0]["particle_count"],
        "r_over_dx": cfg.drop_radius / cfg.dx,
        "pool_layers": cfg.pool_depth / cfg.dx,
        "dt_s": cfg.dt,
        "acoustic_courant": cfg.dt * cfg.sound_speed / (cfg.h_factor * cfg.dx),
        "steps": len(rows),
        "t_end_s": rows[-1]["time"],
        "peak_compression": comp,
        "peak_compression_step": peak["step"],
        "peak_rarefaction": rare,
        "first_breach_over_5pct_step": next(
            (r["step"] for r in rows if float(r["rho_max"]) / cfg.rho0 - 1.0 > 0.05),
            None),
        "rho_min_over_rho0": min(r["rho_min"] for r in rows) / cfg.rho0,
        "rho_max_over_rho0": max(r["rho_max"] for r in rows) / cfg.rho0,
        "rho_mean_drift": max(r["rho_mean"] for r in rows) / cfg.rho0 - 1.0,
        "peak_speed_m_per_s": peak_speed["max_speed"],
        "peak_mach": peak_speed["max_speed"] / cfg.sound_speed,
        "peak_acceleration_m_per_s2": peak_accel["max_acceleration"],
        "density_diffusion": cfg.density_diffusion,
        "hydrostatic_init": cfg.hydrostatic_init,
        "gate_5pct": "PASS" if comp <= 0.05 else "FAIL",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="0.020,0.009,0.006")
    ap.add_argument("--steps", type=int, default=50,
                    help="integration steps (scaling benchmark default: 50)")
    ap.add_argument("--courant-scale", type=float, default=0.8,
                    help="dt = courant_scale * suggested_dt (scaling default: 0.8)")
    ap.add_argument("--density-diffusion", type=float, default=0.0)
    ap.add_argument("--hydrostatic-init", action="store_true")
    ap.add_argument("--out", default="artifacts/stability/report.json")
    ap.add_argument("--artifacts", default="artifacts/stability")
    args = ap.parse_args(argv)

    cases = []
    for dx in sorted((float(x) for x in args.sizes.split(",")), reverse=True):
        cfg = scaling_config(dx, args.steps, args.courant_scale,
                             args.density_diffusion, args.hydrostatic_init)
        out = Path(args.artifacts) / f"dx_{dx:g}"
        rows = run(cfg, out, plot=False)
        item = summarise(rows, cfg)
        item["artifact_dir"] = str(out)
        cases.append(item)
        print(f"dx={dx:g}  N={item['particle_count']:6d}  dt={item['dt_s']:.3e}  "
              f"t_end={item['t_end_s']*1e3:.1f} ms  "
              f"compression={item['peak_compression']*100:6.2f} %  "
              f"rarefaction={item['peak_rarefaction']*100:7.2f} %  "
              f"Mach={item['peak_mach']:.4f}  -> {item['gate_5pct']}", flush=True)

    scores = {c["particle_count"]: c for c in cases}
    report = {
        "benchmark": "resolution_stability",
        "question": "does the dx=0.006 density-compression failure persist "
                    "beyond the 50-step initial transient, and why",
        "steps": args.steps,
        "courant_scale": args.courant_scale,
        "density_diffusion": args.density_diffusion,
        "hydrostatic_init": args.hydrostatic_init,
        "cases": cases,
        "failures": [c["dx_m"] for c in cases if c["gate_5pct"] == "FAIL"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}  ({len(cases)} cases)")
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())