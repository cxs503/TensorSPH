#!/usr/bin/env python3
"""Error budget and blow-up anatomy of the periodic-projection Taylor-Green run.

``docs/xval_isph_projection_taylor_green.md`` (m20500) reports the gate case as an
honest FAIL (23.613 % velocity L2 error) and a *second* unexplained fact: at
``dx = 0.04`` the same configuration diverges (velocity error 2.1e5 %).  Neither
"which term carries the 23.6 %" nor "what diverges at dx = 0.04" is answered by
the stored artefacts, so this driver measures both, step by step:

* **error budget** -- the final velocity error is split into an *amplitude*
  component (the scheme damps or amplifies the vortex as a whole) and a *shape*
  component (phase / high-wavenumber distortion), by least-squares fitting
  ``u ~= c * u_exact``:

      ||u - u_e||^2 = ||u - c u_e||^2 + (c-1)^2 ||u_e||^2 ,  c = (u . u_e)/||u_e||^2

* **energy trajectory** -- ``E/E_exact`` and the discrete-divergence rms at every
  logging step, so over-dissipation can be located in time instead of blamed on
  the whole run.
* **blow-up anatomy** -- per step: ``max|u|``, the CG iteration count and the
  projection residual ratio, plus the lattice-mode content of the velocity at the
  last stable step.  That distinguishes an acoustic/CFL blow-up (max|u| explodes
  while div_ratio stays sane) from a pressure-solver breakdown (div_ratio > 1 or
  CG hitting the iteration cap).

Deterministic; writes ``results.json`` (and ``error_budget.json`` for the ladder)
next to ``--out``.

    PYTHONPATH=src python benchmarks/isph_taylor_green/diagnose_error_budget.py \
        --out outputs/isph_taylor_green --configs "0.05:0.1,0.04:0.1"
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

from tensorsph.isph_periodic import (divergence, kernel_moments,  # noqa: E402
                                     pair_geometry, periodic_pairs, project,
                                     viscous_laplacian)


def lattice(L: float, dx: float) -> torch.Tensor:
    n = int(round(L / dx))
    line = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
    gx, gy = torch.meshgrid(line, line, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def exact_field(pos, t, *, L=1.0, U0=1.0, Re=10.0):
    k = 2.0 * math.pi / L
    nu = U0 / (Re * k)
    amp = U0 * math.exp(-2.0 * nu * k * k * t)
    return torch.stack((amp * torch.sin(k * pos[:, 0]) * torch.cos(k * pos[:, 1]),
                        -amp * torch.cos(k * pos[:, 0]) * torch.sin(k * pos[:, 1])), dim=-1)


def decompose(u, u_exact):
    """Split the L2 error into amplitude (c-1) and shape (u - c u_e) parts."""
    num = float((u * u_exact).sum())
    den = float((u_exact * u_exact).sum())
    c = num / den
    shape = float((u - c * u_exact).norm())
    total = float((u - u_exact).norm())
    return {
        "amplitude_ratio_c": c,
        "total_rel_l2_pct": 100.0 * total / math.sqrt(den),
        "amplitude_component_pct": 100.0 * abs(c - 1.0),
        "shape_component_pct": 100.0 * shape / math.sqrt(den),
        # the two parts are orthogonal in L2, so the squares add up
        "orthogonality_check_pct": 100.0 * math.sqrt(
            (shape * shape + (c - 1.0) ** 2 * den)) / math.sqrt(den),
    }


def run_config(dx, courant, *, L=1.0, U0=1.0, Re=10.0, t_end=0.5, h_factor=1.3,
               rho0=1000.0, log_every=10):
    k = 2.0 * math.pi / L
    nu = U0 / (Re * k)
    h = h_factor * dx
    dt = courant * h / U0
    pos = lattice(L, dx)
    vol = torch.full((pos.shape[0],), (L * L) / pos.shape[0], dtype=torch.float64)
    u = exact_field(pos, 0.0, L=L, U0=U0, Re=Re)
    n_steps = int(round(t_end / dt))
    t0 = time.time()
    trace, ratios, its, rmin = [], [], [], []
    blew_up = None
    for step in range(1, n_steps + 1):
        pairs = periodic_pairs(pos, 2.0 * h, L)
        moments = kernel_moments(pos, vol, pairs, h, L)
        u_star = u + dt * viscous_laplacian(pos, u, vol, pairs, h, L, nu)
        u_a, _p, info = project(pos, u_star, vol, pairs, h, L, rho0, dt)
        x_mid = pos + 0.5 * dt * u_a
        x_mid = x_mid - L * torch.round(x_mid / L)
        pairs_mid = periodic_pairs(x_mid, 2.0 * h, L)
        acc = 0.5 * viscous_laplacian(pos, u, vol, pairs, h, L, nu) \
            + 0.5 * viscous_laplacian(x_mid, u_a, vol, pairs_mid, h, L, nu)
        u_new, _p2, info2 = project(x_mid, u + dt * acc, vol, pairs_mid, h, L, rho0, dt)
        _i, _j, _d, _g = pair_geometry(x_mid, pairs_mid, h, L)
        rmin.append(float(torch.linalg.vector_norm(_d, dim=-1).min()))
        ratios.append(info2["div_after_over_div_before"])
        its.append(info2["cg_iterations"])
        u = u_new
        pos = pos + dt * u
        pos = pos - L * torch.round(pos / L)
        if not torch.isfinite(u).all() or float(u.abs().max()) > 1e3 * U0:
            blew_up = {"step": step, "max_abs_u": float(u.abs().max()) if torch.isfinite(u).all() else float("inf"),
                       "div_ratio": info2["div_after_over_div_before"]}
        if step % log_every == 0 or step == n_steps or blew_up:
            t = step * dt
            exact = exact_field(pos, t, L=L, U0=U0, Re=Re)
            moments_mid = kernel_moments(x_mid, vol, pairs_mid, h, L)
            trace.append({
                "step": step, "t": t,
                "rel_l2_pct": 100.0 * float((u - exact).norm()) / float(exact.norm()),
                "energy_over_exact": float((u * u).sum()) / float((exact * exact).sum()),
                "div_rms": float(torch.sqrt((divergence(x_mid, u, vol, pairs_mid, h, L,
                                                         moments_mid) ** 2).mean())),
                "max_abs_u": float(u.abs().max()),
                "min_pair_distance_over_dx": rmin[-1] / dx if rmin else None,
                "cg_iterations": info2["cg_iterations"],
                "div_ratio": info2["div_after_over_div_before"],
            })
            if blew_up:
                break
    exact = exact_field(pos, n_steps * dt, L=L, U0=U0, Re=Re)
    budget = decompose(u, exact)
    return {
        "dx": dx, "h": h, "courant": courant, "dt": dt, "steps": n_steps,
        "particles": int(pos.shape[0]), "reynolds": Re, "nu": nu, "t_end": t_end,
        "budget": budget,
        "div_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
        "cg_iterations_mean": sum(its) / len(its) if its else None,
        "blew_up": blew_up,
        "trace": trace,
        "wall_seconds": time.time() - t0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "outputs" / "isph_taylor_green"))
    ap.add_argument("--configs", default="0.05:0.1,0.05:0.05,0.04:0.1")
    ap.add_argument("--t-end", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)

    rows = []
    for spec in args.configs.split(","):
        dx_s, co_s = spec.split(":")
        row = run_config(float(dx_s), float(co_s), t_end=args.t_end)
        rows.append(row)
        b = row["budget"]
        print(f"dx={row['dx']:g} Co={row['courant']:g} N={row['particles']:4d} "
              f"err={b['total_rel_l2_pct']:8.3f}%  amp={b['amplitude_component_pct']:6.2f}% "
              f"shape={b['shape_component_pct']:7.2f}%  c={b['amplitude_ratio_c']:.3f}  "
              f"div_ratio={row['div_ratio_mean']:.3f} "
              f"{'BLEW UP@' + str(row['blew_up']['step']) if row['blew_up'] else 'stable'}",
              flush=True)
    doc = {
        "case": "CASE-SPH-PROJECTION-TAYLOR-GREEN-ERROR-BUDGET",
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "rows": rows,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "platform": platform.platform()},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "error_budget.json"
    target.write_text(json.dumps(doc, indent=2))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())