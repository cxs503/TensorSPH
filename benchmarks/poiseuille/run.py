#!/usr/bin/env python3
"""Plane Poiseuille channel flow: steady-state verification of the viscous
operator and of the no-slip wall condition.

Physics
-------
An incompressible fluid of kinematic viscosity ``nu`` fills a 2-D channel of
height ``H`` with no-slip walls at ``y = 0`` and ``y = H`` and is driven by a
uniform body force ``g_x`` per unit mass along ``x``.  The flow is periodic in
``x``, so the problem is one-dimensional and the steady solution of
``0 = -dp/dx + mu d2u/dy2 + rho g_x`` with mass conservation is the parabola

    u(y) = g_x (y - a)(b - y) / (2 nu),   a = 0, b = H,

with peak velocity ``u_max = g_x H^2 / (8 nu)`` on the centre line and wall
shear stress ``tau_w = rho g_x H / 2``.

Why this case exists
--------------------
The repository verifies the viscous term only in its *transient* form: the
quarter-wave shear decay ``exp(-nu (pi/2H)^2 t)`` (``test_boundary.py``), whose
measured rate is 0.84 of the analytic value at ``dx = 0.02``.  A transient
energy-decay rate and a steady momentum balance are not the same test: the
first measures how fast a mode dies, the second measures whether the discrete
viscous stress can hold a shear against a constant load.  This benchmark
measures the second, and it separates the two error mechanisms that the
transient test conflates:

* ``nu_eff / nu``  -- the effective viscosity of the viscous operator, read
  from the *curvature* of the measured profile (``u'' = -g_x/nu_eff``), which
  is independent of where the wall is;
* ``delta / dx``   -- the effective wall offset, read from the *level* of the
  profile, i.e. where the no-slip plane actually sits relative to the nominal
  boundary.

Fitting
-------
The measured profile is binned on the lattice rows and fitted with a quadratic

    u(y) = c0 + c1 y + c2 y^2,

which is linear least squares and needs no optimiser.  For the analytic family
``u = C (y - a)(b - y)`` we have ``c2 = -C`` and ``c1 = C (a + b)``, hence

    nu_eff = g_x / (2 |c2|),      a + b = c1 / C,      a b = -c0 / C,

and with a symmetric offset ``a = -delta``, ``b = H + delta`` the offset is the
root of ``delta^2 + H delta - c0 / C = 0``.  The ``c1 / C`` consistency check
must return ``H`` to the extent that the profile is a genuine parabola.

Usage
-----
    PYTHONPATH=src python benchmarks/poiseuille/run.py --out benchmarks/poiseuille/out
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from tensorsph.droplet_impact import Config, resolve_device, rhs


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def build_fluid(cfg: Config, height: float, length: float, device):
    """Fluid lattice on rows ``y = dx/2, 3dx/2, ... < height``, ``x`` in [0, L)."""
    dx = cfg.dx
    xs = torch.arange(dx * 0.5, length, dx, device=device)
    ys = torch.arange(dx * 0.5, height, dx, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), cfg.rho0 * dx * dx, device=device)
    return pos, mass


def build_walls(cfg: Config, height: float, length: float, layers: int, device):
    """Static wall bands below ``y = 0`` and above ``y = H``.

    The first layer sits at ``y = -dx/2`` (bottom) and ``y = H + dx/2`` (top),
    i.e. one spacing away from the first fluid row, which is the arrangement the
    rest of the solver uses.  The bands are extended by three spacings past both
    ``x`` ends so the periodic images of the fluid always see full wall support.
    """
    dx = cfg.dx
    margin = 3.0 * dx
    xs = torch.arange(-margin, length + margin, dx, device=device)
    parts = []
    for layer in range(1, layers + 1):
        yb = torch.full_like(xs, -dx * 0.5 - (layer - 1) * dx)
        parts.append(torch.stack((xs, yb), dim=-1))
        yt = torch.full_like(xs, height + dx * 0.5 + (layer - 1) * dx)
        parts.append(torch.stack((xs, yt), dim=-1))
    return torch.cat(parts, dim=0)


# --------------------------------------------------------------------------- #
# Time integration
# --------------------------------------------------------------------------- #

def periodic_images(pos, length: float):
    """Three copies of the fluid cloud: real, +L in x, -L in x.

    The support radius ``2h = 2.6 dx`` is far smaller than ``length`` for every
    resolution used here, so a single pair of images is exact.
    """
    shift = torch.tensor([length, 0.0], dtype=pos.dtype, device=pos.device)
    return torch.cat((pos, pos + shift, pos - shift), dim=0)


def run_case(cfg: Config, height: float, length: float, body_force: float,
             t_end: float, t_average: float, device, t_steady_check: float = 0.25):
    """Integrate to a statistically steady state and return the binned profile."""
    dx = cfg.dx
    pos, mass = build_fluid(cfg, height, length, device)
    n = pos.shape[0]
    wall = build_walls(cfg, height, length, cfg.wall_layers, device)
    vel = torch.zeros_like(pos)

    h = cfg.h_factor * dx
    # Fixed step: the acoustic limit with the usual 30 % Courant margin, checked
    # against the viscous limit as well.  A fixed step keeps the run bit-for-bit
    # reproducible, which a published table needs.
    dt_acoustic = 0.3 * h / cfg.sound_speed
    dt_viscous = 0.125 * h * h / cfg.viscosity
    dt = min(dt_acoustic, dt_viscous)
    steps = int(round(t_end / dt))
    avg_from = t_end - t_average

    acc_sum = torch.zeros_like(pos)
    samples = 0
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
        if t >= avg_from:
            acc_sum += vel
            samples += 1

        vel = vel + dt * acc
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], length)

        if step % max(1, steps // 12) == 0 or step == steps - 1:
            ux = vel[:, 0]
            trace.append({
                "t": t,
                "u_max": float(ux.max()),
                "u_mean": float(ux.mean()),
                "max_mach": float(ux.abs().max()) / cfg.sound_speed,
                "wall_clock_s": time.time() - wall_clock,
            })

    u_mean = acc_sum / max(samples, 1)
    rows, profile = bin_rows(pos, u_mean, height, dx)

    # Downsampled final field, for the figures.  The flow is x-invariant, so the
    # snapshot is also the check that it stayed two-dimensional.
    stride = max(1, n // 1500)
    field = {
        "x": pos[::stride, 0].tolist(),
        "y": pos[::stride, 1].tolist(),
        "u": u_mean[::stride, 0].tolist(),
        "u_y": u_mean[::stride, 1].tolist(),
    }

    # Steady-state criterion: the centre-line velocity must have stopped moving
    # over the last ``t_steady_check`` of the run.
    tail = [p for p in trace if p["t"] >= t_end - t_steady_check]
    drift = 0.0
    if len(tail) >= 2 and abs(tail[0]["u_max"]) > 0:
        drift = abs(tail[-1]["u_max"] - tail[0]["u_max"]) / abs(tail[0]["u_max"])

    return {
        "n_fluid": int(n),
        "n_wall": int(wall.shape[0]),
        "dt": dt,
        "steps": steps,
        "n_averaged_samples": samples,
        "rows": rows,
        "profile": profile,
        "field": field,
        "trace": trace,
        "steady_drift": drift,
        "wall_clock_s": time.time() - wall_clock,
    }


def bin_rows(pos, u_mean, height: float, dx: float):
    """Average ``u_x`` over the particles of each lattice row.

    The flow is invariant in ``x``, so the scatter inside a row is a direct
    measure of how two-dimensional the solution has stayed.
    """
    row = torch.round((pos[:, 1] - dx * 0.5) / dx).to(torch.long)
    n_rows = int(round(height / dx))
    rows, profile = [], []
    for k in range(n_rows):
        sel = row == k
        if int(sel.sum()) == 0:
            continue
        u = u_mean[sel, 0]
        rows.append(k)
        profile.append({
            "row": k,
            "y": float(dx * 0.5 + k * dx),
            "y_over_H": float((dx * 0.5 + k * dx) / height),
            "u_mean": float(u.mean()),
            "u_std_x": float(u.std(unbiased=False)) if u.numel() > 1 else 0.0,
            "count": int(u.numel()),
        })
    return rows, profile


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def fit_profile(profile, height: float, body_force: float, viscosity: float,
                lo: float = 0.0, hi: float = 1.0):
    """Quadratic least-squares fit of ``u(y)``; returns the effective constants.

    ``lo``/``hi`` restrict the fit to a band of ``y/H``.  The band is how the
    bulk viscous operator is separated from the wall layer: a fit over every row
    mixes both, a fit over the middle half sees essentially only the bulk.
    """
    sel = [p for p in profile if lo <= p["y_over_H"] <= hi]
    ys = [p["y"] for p in sel]
    us = [p["u_mean"] for p in sel]
    k = len(ys)
    if k < 4:
        raise ValueError("need at least four rows to fit a parabola")
    # Normal equations for the basis (1, y, y^2), solved in closed form via
    # torch.linalg.lstsq on a 3-column Vandermonde matrix.
    a = torch.tensor([[1.0, y, y * y] for y in ys], dtype=torch.float64)
    b = torch.tensor(us, dtype=torch.float64)
    coef = torch.linalg.lstsq(a, b).solution
    c0, c1, c2 = (float(v) for v in coef)

    curvature = -c2
    nu_eff = body_force / (2.0 * curvature) if curvature > 0 else float("nan")
    # Analytic family u = C (y - a)(b - y) with C = curvature.
    sum_ab = c1 / curvature if curvature > 0 else float("nan")
    prod_ab = -c0 / curvature if curvature > 0 else float("nan")
    # Symmetric offset: a = -delta, b = H + delta  =>  a b = -delta H - delta^2,
    # so delta solves delta^2 + H delta + (a b) = 0, i.e. disc = H^2 - 4 (a b).
    disc = height * height - 4.0 * prod_ab
    delta = (-height + math.sqrt(disc)) / 2.0 if disc >= 0 else float("nan")

    centre = [p for p in sel if abs(p["y_over_H"] - 0.5) < 0.5 / max(len(sel), 1) + 1e-9]
    u_centre = centre[0]["u_mean"] if centre else max(us)
    return {
        "c0": c0,
        "c1": c1,
        "c2": c2,
        "nu_eff": nu_eff,
        "nu_eff_over_nu": nu_eff / viscosity if nu_eff == nu_eff else float("nan"),
        "a_plus_b": sum_ab,
        "a_times_b": prod_ab,
        "delta": delta,
        "u_centre_measured": u_centre,
        "u_centre_analytic_nominal": body_force * height * height / (8.0 * viscosity),
        "fit_residual_rms": float(
            torch.sqrt(torch.mean((a @ coef - b) ** 2))
        ),
    }


def analytic(y: float, height: float, body_force: float, viscosity: float,
             delta: float = 0.0) -> float:
    """Parabola between walls at ``-delta`` and ``height + delta``."""
    a, b = -delta, height + delta
    return body_force * (y - a) * (b - y) / (2.0 * viscosity)


def compare(profile, height: float, body_force: float, viscosity: float,
            delta: float):
    """Per-row relative errors against the nominal and the effective channel."""
    out = []
    for p in profile:
        u_num = p["u_mean"]
        u_nom = analytic(p["y"], height, body_force, viscosity, 0.0)
        u_eff = analytic(p["y"], height, body_force, viscosity, delta)
        out.append({
            **p,
            "u_analytic_nominal": u_nom,
            "u_analytic_effective": u_eff,
            "rel_error_nominal": (u_num - u_nom) / u_nom if u_nom else float("nan"),
            "rel_error_effective": (u_num - u_eff) / u_eff if u_eff else float("nan"),
        })
    return out


def interior_error(rows, height: float, lo: float = 0.15, hi: float = 0.85,
                   key: str = "rel_error_nominal"):
    sel = [r for r in rows if lo <= r["y_over_H"] <= hi]
    if not sel:
        return float("nan")
    return math.sqrt(sum(r[key] ** 2 for r in sel) / len(sel))


# --------------------------------------------------------------------------- #
# Case driver
# --------------------------------------------------------------------------- #

def make_config(dx: float, viscosity: float, friction: float, device) -> Config:
    return Config(
        dx=dx,
        h_factor=1.3,
        rho0=1000.0,
        sound_speed=20.0,
        gamma=7.0,
        gravity=0.0,          # the driving force is applied along x by hand
        viscosity=viscosity,
        dt=1.0e-4,            # replaced by the computed limit inside run_case
        steps=0,
        output_every=1000,
        pool_width=0.2,
        pool_depth=0.1,
        boundary="wall",
        wall_layers=3,
        wall_friction=friction,
        fail_fast=False,
        device=str(device),
    )


def run_one(resolution: int, viscosity: float, body_force: float, friction: float,
            length: float, height: float, t_end: float, t_average: float,
            device) -> dict:
    dx = height / resolution
    cfg = make_config(dx, viscosity, friction, device)
    case = run_case(cfg, height, length, body_force, t_end, t_average, device)
    fit = fit_profile(case["profile"], height, body_force, viscosity)
    fit_interior = fit_profile(case["profile"], height, body_force, viscosity,
                               lo=0.25, hi=0.75)
    rows = compare(case["profile"], height, body_force, viscosity, fit["delta"])
    # Wall shear from the measured near-wall slope, against the momentum-balance
    # value tau_w = rho g H / 2 that a steady channel must satisfy.
    slope_free = (rows[1]["u_mean"] - rows[0]["u_mean"]) / dx
    tau_measured = cfg.rho0 * viscosity * slope_free
    tau_analytic = cfg.rho0 * body_force * height / 2.0
    first = rows[0]
    last = rows[-1]
    u_max_meas = max(r["u_mean"] for r in rows)
    return {
        "resolution_H_over_dx": resolution,
        "dx": dx,
        "h": cfg.h_factor * dx,
        "wall_friction": friction,
        "viscosity": viscosity,
        "body_force": body_force,
        "reynolds_number": u_max_meas * height / viscosity,
        "fit": fit,
        "fit_interior": fit_interior,
        "wall_shear_measured": tau_measured,
        "wall_shear_analytic": tau_analytic,
        "wall_shear_ratio": tau_measured / tau_analytic,
        "steady_drift": case["steady_drift"],
        "dt": case["dt"],
        "steps": case["steps"],
        "n_fluid": case["n_fluid"],
        "n_wall": case["n_wall"],
        "n_averaged_samples": case["n_averaged_samples"],
        "wall_clock_s": case["wall_clock_s"],
        "trace": case["trace"],
        "field": case["field"],
        "rows": rows,
        "u_max_measured": u_max_meas,
        "u_max_analytic": body_force * height * height / (8.0 * viscosity),
        "u_max_ratio": u_max_meas / (body_force * height * height / (8.0 * viscosity)),
        "wall_row_slip": first["u_mean"],
        "wall_row_slip_over_umax": first["u_mean"] / u_max_meas if u_max_meas else float("nan"),
        "asymmetry": (
            abs(first["u_mean"] - last["u_mean"]) / u_max_meas
            if u_max_meas
            else float("nan")
        ),
        "max_row_std_over_umax": max(r["u_std_x"] for r in rows) / u_max_meas,
        "max_mach": max(p["max_mach"] for p in case["trace"]),
        "interior_rms_error_nominal": interior_error(rows, height, key="rel_error_nominal"),
        "interior_rms_error_effective": interior_error(rows, height, key="rel_error_effective"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/poiseuille/out"))
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--viscosity", type=float, default=0.01)
    parser.add_argument("--body-force", type=float, default=9.81)
    parser.add_argument("--height", type=float, default=0.1)
    parser.add_argument("--columns", type=int, default=12,
                        help="fluid columns in x; length = columns * dx")
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--t-average", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch CPU threads; 1 is 13x faster here because "
                             "the tensors are small enough that thread overhead "
                             "dominates")
    parser.add_argument("--quick", action="store_true",
                        help="single coarse resolution, short run")
    args = parser.parse_args(argv)

    if args.device in ("auto", "cpu"):
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    if args.quick:
        args.resolutions = [8]
        args.t_end = 0.4

    results = []
    for r in args.resolutions:
        length = args.columns * args.height / r
        print(f"[poiseuille] H/dx={r} ...", flush=True)
        res = run_one(r, args.viscosity, args.body_force, 1.0, length,
                      args.height, args.t_end, args.t_average, device)
        res["length"] = length
        print(f"[poiseuille]   nu_eff/nu={res['fit']['nu_eff_over_nu']:.4f} "
              f"(bulk {res['fit_interior']['nu_eff_over_nu']:.4f}) "
              f"delta/dx={res['fit']['delta'] / res['dx']:.3f} "
              f"u_max_ratio={res['u_max_ratio']:.4f} "
              f"rms(nominal)={100 * res['interior_rms_error_nominal']:.2f}% "
              f"({res['wall_clock_s']:.1f}s)", flush=True)
        results.append(res)

    # Negative control: free slip.  Without tangential wall stress the body
    # force is never balanced, so the flow keeps accelerating and no steady
    # parabola exists.  Measuring that is what makes the no-slip result
    # meaningful instead of tautological.
    print("[poiseuille] free-slip control ...", flush=True)
    control_length = args.columns * args.height / 8
    control = run_one(8, args.viscosity, args.body_force, 0.0, control_length,
                      args.height, args.t_end, args.t_average, device)
    control["length"] = control_length

    payload = {
        "case": "plane_poiseuille_channel",
        "description": "body-force-driven 2-D channel, no-slip walls, periodic in x",
        "height": args.height,
        "columns": args.columns,
        "viscosity": args.viscosity,
        "body_force": args.body_force,
        "analytic": "u(y) = g (y-a)(b-y)/(2 nu), a=0, b=H; u_max = g H^2/(8 nu)",
        "t_end": args.t_end,
        "t_average": args.t_average,
        "device": str(device),
        "torch": torch.__version__,
        "results": results,
        "free_slip_control": control,
        "wall_clock_s": sum(r["wall_clock_s"] for r in results)
                        + control["wall_clock_s"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"[poiseuille] wrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
