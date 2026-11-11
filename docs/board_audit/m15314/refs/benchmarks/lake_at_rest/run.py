"""Lake at rest (CASE-LAKE-AT-REST-SPH): exact hydrostatic equilibrium under gravity.

Reference (exact, no fitted constant): a motionless body of water with a flat
free surface is a steady solution of the incompressible Euler equations

    u = 0,   eta = H,   p(y) = rho0 g (H - y)      (0 <= y <= H)

i.e. the hydrostatic pressure. It is the canonical "lake at rest" steady state of
the shallow-water equations (LeVeque 1998, J. Comput. Phys. 146:346-365, section
2.1) and the state every WCSPH free-surface code is initialised with (Adami, Hu
& Adams 2012, J. Comput. Phys. 231:7057-7075, section 3.1 hydrostatic test).

Acceptance criterion (case pool): pressure drift < 0.5 % of rho0 g H.

Two modes are measured, on the *production* RHS (``droplet_impact.rhs``):

  shipped : Config(hydrostatic_init=True, density_calibration=True) -- the
            repository default way of starting a still pool.
  wb      : the same plus ``equilibrium_level=H`` -- the well-balanced
            (equilibrium-preserving) pressure force, see
            ``Config.equilibrium_level`` and docs/verification_lake_at_rest.md.

Run:  python3 benchmarks/lake_at_rest/run.py [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph import droplet_impact as di  # noqa: E402

DRIFT_TOL = 0.005          # 0.5 % of rho0 g H, the pool acceptance criterion
WINDOW_FRACTION = 1.0      # all samples count as drift


def source_revision() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        return {"commit": head.stdout.strip(), "dirty": bool(
            subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                           capture_output=True, text=True, timeout=10).stdout.strip())}
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}


def build_state(cfg, device):
    """Still pool + walls, masses from the analytic hydrostatic profile."""
    pos = di.pool_lattice(cfg, device)
    wall = di.wall_lattice(cfg, device)
    unit = di.lattice_mass_unit(cfg)
    mass = di.hydrostatic_density(pos[:, 1], cfg) * (unit / cfg.rho0)
    return pos, mass, wall


def interior_mask(pos, cfg, h):
    level = cfg.pool_depth
    return ((pos[:, 0] > 3 * h) & (pos[:, 0] < cfg.pool_width - 3 * h)
            & (pos[:, 1] > 3 * h) & (pos[:, 1] < level - 3 * h))


def run_case(dx=0.02, h_factor=1.3, seconds=0.05, sound_speed=20.0, mode="shipped",
             viscosity=0.01, sample_every=None):
    cfg = di.Config(dx=dx, h_factor=h_factor, rho0=1000.0, gravity=9.81,
                    sound_speed=sound_speed, viscosity=viscosity,
                    hydrostatic_init=True, density_calibration=True)
    cfg.dt = di.suggested_dt(cfg)
    steps = max(1, int(round(seconds / cfg.dt)))
    if mode == "wb":
        cfg.equilibrium_level = cfg.pool_depth
    device = torch.device("cpu")
    pos, mass, wall = build_state(cfg, device)
    h = cfg.h_factor * cfg.dx
    level = cfg.pool_depth
    gH = cfg.gravity * level
    interior = interior_mask(pos, cfg, h)
    vel = torch.zeros_like(pos)
    every = sample_every or max(1, steps // 16)
    samples = []
    t0 = time.time()
    for step in range(steps + 1):
        rho, p, acc = di.rhs(pos, vel, mass, cfg, wall=wall, dt=cfg.dt)
        dev = (p - cfg.rho0 * cfg.gravity * (level - pos[:, 1])).abs()[interior] / (cfg.rho0 * gH)
        if step % every == 0 or step == steps:
            samples.append({
                "t": round(step * cfg.dt, 6),
                "pressure_drift_max": float(dev.max()),
                "pressure_drift_mean": float(dev.mean()),
                "v_max": float(vel.abs().max()),
                "acc_max_at_rest": float(acc.abs().max()) if step == 0 else None,
                "rho_rel_error_max": float((rho[interior] / cfg.rho0 - 1.0).abs().max()),
            })
        if step == steps:
            break
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel
    wall_seconds = time.time() - t0
    drift = max(s["pressure_drift_max"] for s in samples)
    return {
        "mode": mode,
        "viscosity": cfg.viscosity,
        "dx_m": cfg.dx,
        "h_over_dx": cfg.h_factor,
        "h_m": h,
        "sound_speed_m_s": cfg.sound_speed,
        "mach_at_rest": 0.0,
        "c0_over_sqrt_gH": cfg.sound_speed / math.sqrt(gH),
        "dt_s": cfg.dt,
        "acoustic_courant": di.acoustic_courant(cfg.dt, cfg),
        "steps": steps,
        "seconds": steps * cfg.dt,
        "gravity_dt_ratio": cfg.dt * math.sqrt(cfg.gravity / level),
        "particles": int(pos.shape[0]),
        "wall_particles": int(wall.shape[0]),
        "interior_particles": int(interior.sum()),
        "pressure_drift_max": drift,
        "pressure_drift_mean_peak": max(s["pressure_drift_mean"] for s in samples),
        "v_max": max(s["v_max"] for s in samples),
        "v_max_over_sqrt_gH": max(s["v_max"] for s in samples) / math.sqrt(gH),
        "pass_0p5pct": bool(drift < DRIFT_TOL),
        "wall_seconds": wall_seconds,
        "samples": samples,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("report.json")))
    args = ap.parse_args(argv)
    torch.set_num_threads(1)

    cases = []
    for dx in (0.025, 0.02):
        for mode in ("shipped", "wb"):
            cases.append(dict(dx=dx, mode=mode))
    # parameter sensitivity: two Mach numbers at the finer resolution
    for c0 in (40.0,):
        cases.append(dict(dx=0.02, mode="wb", sound_speed=c0))
        cases.append(dict(dx=0.02, mode="shipped", sound_speed=c0))
    # viscosity sensitivity: the inviscid limit must not be what carries the result
    cases.append(dict(dx=0.02, mode="wb", viscosity=0.0))

    results = []
    for kw in cases:
        r = run_case(seconds=0.05, **kw)
        results.append(r)
        print(f"{r['mode']:8s} dx={r['dx_m']:.3f} c0={r['sound_speed_m_s']:4.1f} "
              f"nu={kw.get('viscosity', 0.01):5.3f} drift_max={100*r['pressure_drift_max']:.3f}% "
              f"v_max={r['v_max']:.4f} -> {'PASS' if r['pass_0p5pct'] else 'FAIL'}", flush=True)

    shipped = [r for r in results if r["mode"] == "shipped" and r["sound_speed_m_s"] == 20.0]
    wb = [r for r in results if r["mode"] == "wb" and r["sound_speed_m_s"] == 20.0]
    report = {
        "benchmark": "lake_at_rest",
        "case_key": "CASE-LAKE-AT-REST-SPH",
        "question": "does a WCSPH pool stay at the exact hydrostatic equilibrium p = rho0 g (H - y), u = 0, and how far does it drift?",
        "reference": {
            "solution": "u = 0, eta = H, p(y) = rho0 g (H - y)",
            "kind": "analytic exact steady state",
            "citation": "LeVeque 1998, J. Comput. Phys. 146:346-365, sec. 2.1 (lake at rest); Adami, Hu & Adams 2012, J. Comput. Phys. 231:7057-7075, sec. 3.1 (SPH hydrostatic test)",
            "acceptance": "pressure drift < 0.5 % of rho0 g H",
            "drift_definition": "max over interior particles (>= 3h from every wall and from the free surface) and over samples of |p_i(t) - rho0 g (H - y_i)| / (rho0 g H)",
        },
        "parameters": {
            "pool_width_m": 1.0,
            "pool_depth_m": 0.35,
            "rho0_kg_m3": 1000.0,
            "gravity_m_s2": 9.81,
            "gamma": 7.0,
            "kinematic_viscosity_m2_s": 0.01,
            "kernel": "cubic spline, h = 1.3 dx",
            "dt": "acoustic Courant 0.3 (Config.max_courant)",
            "steps": results[0]["steps"],
            "simulated_seconds": results[0]["seconds"],
        },
        "acceptance_tolerance": DRIFT_TOL,
        "cases": results,
        "summary": {
            "shipped_drift_max_over_resolutions": max(r["pressure_drift_max"] for r in shipped),
            "wb_drift_max_over_resolutions": max(r["pressure_drift_max"] for r in wb),
            "improvement_factor_at_dx_0p02": (
                next(r["pressure_drift_max"] for r in shipped if r["dx_m"] == 0.02)
                / next(r["pressure_drift_max"] for r in wb if r["dx_m"] == 0.02)),
            "criterion_met": bool(all(r["pass_0p5pct"] for r in results[:-1])),
            "wb_criterion_met": bool(all(r["pass_0p5pct"] for r in wb)),
        },
        "provenance": {
            "source": source_revision(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "repro_cmd": "PYTHONPATH=src python3 benchmarks/lake_at_rest/run.py",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("wrote", args.out)
    return report


if __name__ == "__main__":
    main()
