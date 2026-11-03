"""Where does a free capillary oscillation survive in WCSPH?

The oscillating-droplet benchmark (:mod:`benchmarks.oscillating_droplet.run`)
needs a run that lasts several oscillation periods, and an undamped resonator has
nothing to absorb the interface noise that summation-density WCSPH produces.  This
tool measures the *validity envelope*: for a grid of (dx, viscosity) it advances
the same elliptical drop and records how long the solution survives before the
maximum speed exceeds ``blowup_mach * c0``.

It is deliberately cheap (a few hundred steps per cell) and it is the evidence
behind the "required viscosity" statement in
``docs/verification_oscillating_droplet.md``: the frequencies reported there are
only claimed for cells this map marks as surviving.

    PYTHONPATH=src python benchmarks/oscillating_droplet/stability_map.py \
        --dx 0.002 --nu-list 5e-4,1e-3,2e-3,3e-3 --steps 900
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import Config, next_dt, rhs, stable_dt
from tensorsph.neighbor_search import cell_linked_pairs

import importlib.util

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("osc_run", HERE / "run.py")
_osc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_osc)


def survival(dx: float, nu: float, steps: int = 900, radius: float = 0.02,
             sigma: float = 7.2, rho0: float = 1000.0, sound_speed: float = 20.0,
             eps: float = 0.08, blowup_mach: float = 0.5) -> dict:
    """Advance the elliptical drop and report how long it stays bounded."""
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, sound_speed=sound_speed,
                 gamma=7.0, gravity=0.0, viscosity=nu,
                 surface_tension_coefficient=sigma, csf_scheme="consistent")
    h = cfg.h_factor * cfg.dx
    dt = float(stable_dt(cfg))
    pos, mass = _osc.elliptical_droplet(radius * (1 + eps), radius * (1 - eps),
                                        dx, rho0)
    vel = torch.zeros_like(pos)
    t = 0.0
    vmax_hist = []
    for step in range(steps):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, _p, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None, dt=dt)
        vel = vel + dt * acc
        pos = pos + dt * vel
        t += dt
        vmax = float(torch.linalg.vector_norm(vel, dim=1).max())
        vmax_hist.append(vmax)
        if not math.isfinite(vmax) or vmax > blowup_mach * sound_speed:
            omega = _osc.rayleigh_lamb_omega(2, sigma, rho0, radius)
            return {"dx_m": dx, "nu_m2_per_s": nu, "R_over_h": radius / h,
                    "n_particles": int(pos.shape[0]), "survived": False,
                    "t_survived_s": t, "periods_survived": t / (2 * math.pi / omega),
                    "step_of_failure": step + 1, "vmax_m_per_s": vmax,
                    "vmax_over_vref": vmax / math.sqrt(sigma / (rho0 * radius)),
                    "steps_requested": steps}
        dt = next_dt(cfg, dt, vmax)
    omega = _osc.rayleigh_lamb_omega(2, sigma, rho0, radius)
    return {"dx_m": dx, "nu_m2_per_s": nu, "R_over_h": radius / h,
            "n_particles": int(pos.shape[0]), "survived": True,
            "t_survived_s": t, "periods_survived": t / (2 * math.pi / omega),
            "step_of_failure": None, "vmax_m_per_s": vmax_hist[-1],
            "vmax_over_vref": vmax_hist[-1] / math.sqrt(sigma / (rho0 * radius)),
            "steps_requested": steps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=str, default="0.002")
    ap.add_argument("--nu-list", type=str, default="5e-4,1e-3,2e-3,3e-3")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--eps", type=float, default=0.08)
    a = ap.parse_args()

    cells = []
    for dx in (float(x) for x in a.dx.split(",")):
        for nu in (float(x) for x in a.nu_list.split(",")):
            cell = survival(dx=dx, nu=nu, steps=a.steps, eps=a.eps)
            cells.append(cell)
            print(f"dx={dx*1e3:.2f}mm nu={nu:g} N={cell['n_particles']:5d} "
                  f"survived={cell['survived']} "
                  f"periods={cell['periods_survived']:.2f} "
                  f"vmax/vref={cell['vmax_over_vref']:.2f}", flush=True)

    out = {"steps_per_cell": a.steps, "amplitude_eps": a.eps,
           "criteria": "failure = max|v| > 0.5 c0 (blowup_mach)",
           "cells": cells}
    (HERE / "stability_map.json").write_text(json.dumps(out, indent=2))
    print("->", HERE / "stability_map.json")


if __name__ == "__main__":
    main()