"""Round 6 of ``CASE-SPH-OSC-DROP-FREQ``: localise the dynamic-arm instability.

Round 5 reported an honest FAIL-to-instrument: the free drop diverged in *every*
configuration, **including the surface-tension free control**, so the defect was
attributed to the density/pressure time evolution rather than to the CSF
operator.  This driver does what round 5 §5 named as the next step -- an audit of
where the energy comes from -- but at the level of the *initialisation*, because
a term audit at ``t = 0`` shows the injection happens before the first step is
taken.

Diagnosis at ``t = 0`` (``sigma = 0``, cloud at rest, ``R/dx = 7``)
------------------------------------------------------------------
``|a| = 105`` on a cloud that should be in equilibrium.  The cause is the
per-particle mass calibration of
:func:`tensorsph.oscillating_drop_dynamic.run_dynamic_drop`::

    mass_i <- mass_i * rho0 / rho_summation(pos_i)

is applied to *every* particle, including the surface rows where the kernel
support is truncated and ``rho_summation`` is short by 10-25 %.  Scaling those
masses **up** to make their own density exactly ``rho0`` over-compensates: the
first interior shell ends up at ``rho/rho0 = 1.04`` (measured: 15 particles at
``r/R in [0.80, 0.94]``), which with ``c0 = 20`` and ``gamma = 7`` is a pressure
``p = 18.1`` where the physical value is ``0`` (or the Laplace value ``2 sigma/R
= 2``).  That is a 9x over-pressure shell -- an acoustic blast whose velocity
scale ``dp/(rho0 c0) ~ 0.9`` exceeds the mode velocity ``omega eps R = 0.14`` by
6x, and the WCSPH stiffness turns it into the runaway round 5 measured.

The audit below is a `2 x 2` matrix over the two knobs that control it

* ``calibrate_mass`` -- per-particle mass calibration on/off,
* ``delta_sph`` -- Antuono density diffusion on/off (round 5 showed it inflates
  the free surface, because the surface deficit is a *truncation* deficit, not a
  physical gradient, and plain diffusion cannot tell the difference)

plus the two static readings that decide whether the initial state is an
equilibrium at all:

* ``p_max / p_lap`` -- peak pressure over the Laplace pressure,
* ``rho_profile`` -- density vs radius (the over-dense shell shows as a bump).

Run (a few seconds at ``R/dx = 7``)::

    PYTHONPATH=src python benchmarks/oscillating_drop_freq/round6_diagnose.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.oscillating_drop_dynamic import (DynamicConfig, half_plane_lattice,
                                                mirror_cloud, summation_density,
                                                initial_density, _accelerations,
                                                _dt_limit)

OUT = Path(__file__).with_name("out")


def static_state(cfg: DynamicConfig, *, calibrate_mass: bool = True,
                 passes: int = 2):
    """Build the initial state exactly as ``run_dynamic_drop`` does."""
    torch.set_default_dtype(torch.float64)
    geo = half_plane_lattice(cfg)
    pos = geo["pos"].clone()
    mass = geo["mass"].clone()
    n = pos.shape[0]
    if calibrate_mass:
        for _ in range(passes):
            pm, _, mm = mirror_cloud(pos, torch.zeros_like(pos), mass)
            rho_m = summation_density(pm, mm, cfg.h, cfg.kernel)
            mass = mass * (cfg.rho0 / rho_m[:n])
    pm, _, mm = mirror_cloud(pos, torch.zeros_like(pos), mass)
    rho = initial_density(pm, mm, cfg.h, cfg.rho0, sigma=cfg.sigma,
                          radius=cfg.radius, c0=cfg.c0,
                          gamma=cfg.gamma)[:n]
    return pos, mass, rho.clamp(min=0.5 * cfg.rho0)


def pressure(rho: torch.Tensor, cfg: DynamicConfig) -> torch.Tensor:
    return torch.clamp(cfg.rho0 * cfg.c0 ** 2 / cfg.gamma *
                       ((rho / cfg.rho0).pow(cfg.gamma) - 1.0), min=0.0)


def kinetic(mass, vel) -> float:
    return float((0.5 * mass * vel.square().sum(dim=1)).sum())


def marching(cfg, pos, mass, rho, steps, sample=(40, 120, 240)):
    """Plain KDK loop with the shipped right-hand side (no fixes applied)."""
    vel = torch.zeros_like(pos)
    _, acc, _, _, _ = _accelerations(pos, vel, rho, mass, cfg)
    a0 = float(torch.linalg.vector_norm(acc, dim=1).max())
    log = []
    for s in range(1, steps + 1):
        dt = _dt_limit(vel, cfg)
        vel_half = vel + 0.5 * dt * acc
        pos = pos + dt * vel_half
        drho, acc, _, _, _ = _accelerations(pos, vel_half, rho, mass, cfg)
        rho = torch.clamp(rho + dt * drho, min=0.2 * cfg.rho0)
        neg = pos[:, 0] < 0.0
        if bool(neg.any()):
            pos[neg, 0] = -pos[neg, 0]
        vel = vel_half + 0.5 * dt * acc
        if s in sample:
            log.append({"step": s, "t": None, "vmax": float(torch.linalg.vector_norm(vel, dim=1).max()),
                        "ek": kinetic(mass, vel), "rho_max": float(rho.max()),
                        "rho_min": float(rho.min())})
    return a0, log


def density_profile(pos, rho, radius, nbins=5):
    r = torch.linalg.vector_norm(pos, dim=1) / radius
    edges = torch.linspace(0.0, 1.0, nbins + 1)
    rows = []
    for k in range(nbins):
        m = (r >= edges[k]) & (r < edges[k + 1])
        if int(m.sum()) == 0:
            rows.append({"r_in": float(edges[k]), "r_out": float(edges[k + 1]),
                         "n": 0, "rho_mean": None, "rho_max": None})
            continue
        rows.append({"r_in": float(edges[k]), "r_out": float(edges[k + 1]),
                     "n": int(m.sum()), "rho_mean": float(rho[m].mean()),
                     "rho_max": float(rho[m].max())})
    return rows


def run_case(*, calibrate_mass: bool, delta_sph: float, sigma: float = 0.0,
             rdx: float = 7.0, steps: int = 240) -> dict:
    cfg = DynamicConfig(dx=1.0 / rdx, sigma=sigma, eps=1e-3 if sigma == 0 else 0.05,
                        delta_sph=delta_sph, c0=20.0)
    cfg.validate()
    pos, mass, rho = static_state(cfg, calibrate_mass=calibrate_mass)
    p = pressure(rho, cfg)
    p_lap = 2.0 * sigma / cfg.radius
    a0, log = marching(cfg, pos.clone(), mass.clone(), rho.clone(), steps)
    analytic_mass = cfg.rho0 * math.pi * cfg.radius ** 2
    return {
        "calibrate_mass": calibrate_mass,
        "delta_sph": delta_sph,
        "sigma": sigma,
        "radius_over_dx": rdx,
        "n_particles": int(pos.shape[0]),
        "p_max": float(p.max()),
        "p_lap": p_lap,
        "p_max_over_lap": (float(p.max()) / p_lap) if p_lap > 0 else None,
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "mass_over_analytic": float(mass.sum() / analytic_mass),
        "a_max_t0": a0,
        "rho_profile_t0": density_profile(pos, rho, cfg.radius),
        "march": log,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--rdx", type=float, default=7.0)
    ap.add_argument("--out", default="round6_diagnose.json")
    args = ap.parse_args()
    rows = []
    for ds in (0.0, 0.1):
        for calib in (True, False):
            rows.append(run_case(calibrate_mass=calib, delta_sph=ds,
                                 rdx=args.rdx, steps=args.steps))
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"radius_over_dx": args.rdx, "steps": args.steps, "rows": rows}
    (OUT / args.out).write_text(json.dumps(payload, indent=1))
    hdr = f"{'ds':>4} {'calib':>6} {'p_max':>7} {'rmax':>6} {'a0':>8} " + \
          " ".join(f"s{r['step']}:|v|" for r in rows[0]["march"])
    print(hdr)
    for r in rows:
        print(f"{r['delta_sph']:>4} {str(r['calibrate_mass']):>6} {r['p_max']:>7.2f} "
              f"{r['rho_max']:>6.3f} {r['a_max_t0']:>8.1f} " +
              " ".join(f"{m['vmax']:9.3f}" for m in r["march"]))
    print(f"wrote {OUT / args.out}")


if __name__ == "__main__":
    main()