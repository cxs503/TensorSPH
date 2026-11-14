#!/usr/bin/env python3
"""XVAL-SPH-PROJECTION-TAYLOR-GREEN: periodic ISPH projection against the exact vortex.

Why this benchmark exists
-------------------------
``XVAL-SPH-METHOD-SELECTION-2026-11-12`` refuses to route a smooth inviscid
vortex case to any scheme: the registry carries a projection (ISPH) record from
a **wall-bounded channel** only, and the selector is (correctly) not allowed to
extrapolate channel evidence to the periodic vortex class.  This driver produces
the missing measurement: the 2D Taylor-Green vortex (exact unsteady
Navier-Stokes solution, Taylor & Green 1937) integrated with the periodic
projection operators of :mod:`tensorsph.isph_periodic`.

Reference and metric
--------------------
    u = U0 exp(-2 nu k^2 t) sin(kx) cos(ky)
    v = -U0 exp(-2 nu k^2 t) cos(kx) sin(ky)
    p = (rho U0^2/4) exp(-4 nu k^2 t) (cos 2kx + cos 2ky)

with ``k = 2 pi / L``, ``nu = U0/(Re k)``.  The gate quantity is the relative L2
error of the velocity field sampled at the *Lagrangian* particle positions
against the exact field at the same time, i.e. truncation + projection error of
the scheme, nothing else.  Secondary diagnostics: kinetic-energy ratio
``E/E_exact`` (numerical dissipation) and the discrete divergence ratio before /
after each projection (how solenoidal the result is).

Everything is deterministic (lattice initial condition, no randomness); the
script writes ``results.json`` next to the ``--out`` path it is given.

    PYTHONPATH=src python benchmarks/isph_taylor_green/run.py --out runs/isph_taylor_green
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from tensorsph.isph_periodic import (periodic_pairs, project,  # noqa: E402
                                     viscous_laplacian)

TOLERANCE_PCT = 15.0   # owner precision standard


def lattice(L: float, dx: float) -> torch.Tensor:
    n = int(round(L / dx))
    if abs(n * dx - L) > 1e-12 * L:
        raise ValueError(f"dx={dx:g} does not close the periodic box L={L:g}")
    line = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    gx, gy = torch.meshgrid(line, line, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def exact_field(pos, t, *, L=1.0, U0=1.0, Re=10.0):
    k = 2.0 * math.pi / L
    nu = U0 / (Re * k)
    amp = U0 * math.exp(-2.0 * nu * k * k * t)
    return torch.stack((amp * torch.sin(k * pos[:, 0]) * torch.cos(k * pos[:, 1]),
                        -amp * torch.cos(k * pos[:, 0]) * torch.sin(k * pos[:, 1])), dim=-1)


def run_config(dx: float, courant: float, *, L=1.0, U0=1.0, Re=10.0,
               t_end=0.5, h_factor=1.3, rho0=1000.0) -> dict:
    """One Taylor-Green run with Heun time integration + a projection per stage."""
    k = 2.0 * math.pi / L
    nu = U0 / (Re * k)
    h = h_factor * dx
    dt = courant * h / U0
    pos = lattice(L, dx)
    vol = torch.full((pos.shape[0],), (L * L) / pos.shape[0], dtype=torch.float64)
    u = exact_field(pos, 0.0, L=L, U0=U0, Re=Re)
    e0 = float(u.square().sum())
    n_steps = int(round(t_end / dt))
    ratios, its = [], []
    t0 = time.time()
    for _ in range(n_steps):
        pairs = periodic_pairs(pos, 2.0 * h, L)
        u_star = u + dt * viscous_laplacian(pos, u, vol, pairs, h, L, nu)
        u_a, _p, _info = project(pos, u_star, vol, pairs, h, L, rho0, dt)
        x_mid = pos + 0.5 * dt * u_a
        x_mid = x_mid - L * torch.round(x_mid / L)
        pairs_mid = periodic_pairs(x_mid, 2.0 * h, L)
        acc = 0.5 * viscous_laplacian(pos, u, vol, pairs, h, L, nu) \
            + 0.5 * viscous_laplacian(x_mid, u_a, vol, pairs_mid, h, L, nu)
        u_new, _p, info = project(x_mid, u + dt * acc, vol, pairs_mid, h, L, rho0, dt)
        ratios.append(info["div_after_over_div_before"])
        its.append(info["cg_iterations"])
        u = u_new
        pos = pos + dt * u
        pos = pos - L * torch.round(pos / L)
    exact = exact_field(pos, t_end, L=L, U0=U0, Re=Re)
    err_pct = float((u - exact).norm() / exact.norm() * 100.0)
    e_ratio = float(u.square().sum()) / float(exact.square().sum())
    return {
        "dx": dx, "h": h, "courant": courant, "dt": dt, "steps": n_steps,
        "particles": int(pos.shape[0]), "reynolds": Re, "mach_note": "ISPH: no EOS, no acoustic Mach",
        "t_end": t_end, "nu": nu, "k": k,
        "velocity_rel_l2_pct": err_pct,
        "energy_over_exact": e_ratio,
        "energy_over_initial": float(u.square().sum()) / e0,
        "div_ratio_mean": sum(ratios) / len(ratios),
        "cg_iterations_mean": sum(its) / len(its),
        "tolerance_pct": TOLERANCE_PCT,
        "pass": err_pct <= TOLERANCE_PCT,
        "wall_seconds": time.time() - t0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "runs" / "isph_taylor_green"))
    ap.add_argument("--configs", default="0.05:0.1,0.05:0.05,0.0333333:0.1",
                    help="dx:courant list")
    ap.add_argument("--t-end", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)

    rows = []
    for spec in args.configs.split(","):
        dx_s, c_s = spec.split(":")
        rows.append(run_config(float(dx_s), float(c_s), t_end=args.t_end))
        last = rows[-1]
        print(f"dx={last['dx']:.4g} Co={last['courant']:g} N={last['particles']:5d} "
              f"err={last['velocity_rel_l2_pct']:7.3f}%  E/Eexact={last['energy_over_exact']:.3f} "
              f"div_ratio={last['div_ratio_mean']:.3f}  {'PASS' if last['pass'] else 'FAIL'}",
              flush=True)

    doc = {
        "case": "CASE-SPH-PROJECTION-TAYLOR-GREEN",
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "reference": "Taylor & Green (1937) Proc. R. Soc. Lond. A 158:499; "
                     "exact unsteady Navier-Stokes solution",
        "scheme": "periodic ISPH projection (Cummins & Rudman 1999) + pair (graph) "
                  "Laplacian viscosity, Heun time integration",
        "tolerance_pct": TOLERANCE_PCT,
        "pass": all(r["pass"] for r in rows),
        "rows": rows,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "platform": platform.platform(), "threads": args.threads},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out if out.suffix == ".json" else out / "results.json"
    target.write_text(json.dumps(doc, indent=2))
    print(f"wrote {target}")
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())