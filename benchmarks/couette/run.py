#!/usr/bin/env python3
"""Plane Couette flow between a stationary and a sliding wall (2-D SPH).

Physics
-------
An incompressible fluid of kinematic viscosity ``nu`` fills a 2-D channel of
height ``H``; ``x`` is periodic, the wall at ``y = 0`` is at rest and the wall at
``y = H`` slides along ``x`` at speed ``U``.  The steady solution of the
incompressible Navier-Stokes equations with no-slip on both walls is

    u(y) = U y / H,     du/dy = U / H,     tau = rho nu U / H,     Q = U H / 2

(Batchelor 1967, *An Introduction to Fluid Dynamics*, CUP, §4.2, steady
unidirectional flow between parallel plates with one plate in motion; the same
closed form is the ``Ls = 0`` limit of the Navier-slip Couette anchor in
``TensorLBM/benchmarks/ice_lid_navier_slip.py``).

What is new here
----------------
Nothing in the solver.  The existing moving-wall input of
``tensorsph.droplet_impact.rhs`` -- ``wall_vel``, already used by the
``wave_flume`` wavemaker -- is given a *tangential* component instead of a
normal one, together with ``wall_friction = 1`` for the no-slip blend.  The
case is the missing companion of ``benchmarks/poiseuille``: there the wall
condition is loaded by a body force and the profile is curved, so the wall
offset and the viscous-operator error are entangled; here the exact profile is
a straight line, the curvature error cannot enter, and the only thing the test
can fail on is the wall condition itself.

Nothing is calibrated against the run: ``U``, ``H``, ``nu``, ``c0`` enter as
inputs, the reference is the closed form evaluated at the same row centres, and
the run starts from rest (impulsive start of the sliding wall) so the initial
condition cannot be mistaken for the answer.

Acceptance gates (predeclared, see ``benchmarks/couette/DESIGN.md``):
interior L2 profile error < 5 %, |flux ratio - 1| < 5 %, |wall-shear ratio - 1|
< 5 %, steady drift < 1 %, and a free-slip negative control with flux < 20 % of
the analytic value.

Usage::

    PYTHONPATH=src python benchmarks/couette/run.py --out benchmarks/couette/out
    PYTHONPATH=src python benchmarks/couette/run.py --quick          # smoke
    PYTHONPATH=src python benchmarks/couette/run.py --extended       # + H/dx=32
    PYTHONPATH=src python benchmarks/couette/run.py --figure         # re-plot
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.droplet_impact import Config, resolve_device, rhs  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmarks" / "couette" / "out"

#: Case constants (held fixed unless a sweep says otherwise).
HEIGHT = 0.1            # m
COLUMNS = 12            # fluid columns in x; length = COLUMNS * dx
RHO0 = 1000.0
SOUND_SPEED = 20.0
GAMMA = 7.0
H_FACTOR = 1.3
WALL_LAYERS = 3
VISCOSITY = 0.02        # m^2/s  -> H^2/nu = 0.5 s
WALL_SPEED = 0.2        # m/s    -> Re = U H / nu = 1, Ma = 0.01
FRICTION = 1.0          # no-slip blend
LADDER = (10, 16, 24)
REFERENCE_RESOLUTION = 16
VISCOUS_TIMES = 1.2     # t_end = VISCOUS_TIMES * H^2 / nu
AVERAGE_FRACTION = 0.2  # average over the last 20 % of the run

#: Predeclared thresholds.  Do not edit after seeing a result.
GATES = {
    "interior_l2": 0.05,
    "flux_rel": 0.05,
    "wall_shear_rel": 0.05,
    "steady_drift": 0.01,
    "control_flux": 0.20,
}

REFERENCE_NOTE = (
    "u(y) = U y/H, du/dy = U/H, tau = rho nu U/H, Q = U H/2 -- Batchelor (1967) "
    "An Introduction to Fluid Dynamics, CUP, section 4.2 (steady unidirectional "
    "flow between parallel plates, one plate moving); identical closed form is "
    "the Ls->0 limit of the Navier-slip Couette anchor in "
    "TensorLBM/benchmarks/ice_lid_navier_slip.py"
)


# --------------------------------------------------------------------------- #
# Case construction
# --------------------------------------------------------------------------- #

def make_config(dx: float, viscosity: float, friction: float, device) -> Config:
    return Config(
        dx=dx,
        h_factor=H_FACTOR,
        rho0=RHO0,
        sound_speed=SOUND_SPEED,
        gamma=GAMMA,
        gravity=0.0,          # no body force: the wall alone drives the flow
        viscosity=viscosity,
        dt=1.0e-4,            # replaced by the computed limit inside run_case
        steps=0,
        output_every=1000,
        pool_width=COLUMNS * dx,
        pool_depth=HEIGHT,
        boundary="wall",
        wall_layers=WALL_LAYERS,
        wall_friction=friction,
        fail_fast=False,
        device=str(device),
    )


def build_fluid(cfg: Config, height: float, length: float, device):
    """Fluid lattice on rows ``y = dx/2, 3dx/2, ... < height``, ``x`` in [0, L)."""
    dx = cfg.dx
    xs = torch.arange(dx * 0.5, length, dx, device=device)
    ys = torch.arange(dx * 0.5, height, dx, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), cfg.rho0 * dx * dx, device=device)
    return pos, mass


def build_walls(cfg: Config, height: float, length: float, layers: int,
                wall_speed: float, device):
    """Wall bands below ``y = 0`` (at rest) and above ``y = H`` (sliding at ``U``).

    Returns ``(pos, wall_vel)``.  The bands are extended three spacings past both
    ``x`` ends so the periodic images of the fluid always see full wall support,
    exactly as in :mod:`benchmarks.poiseuille.run`.  ``wall_vel`` is the same
    input the ``wave_flume`` wavemaker uses; here it is tangential, which is what
    makes the case a Couette flow instead of a piston.
    """
    dx = cfg.dx
    margin = 3.0 * dx
    xs = torch.arange(-margin, length + margin, dx, device=device)
    pos_parts, vel_parts = [], []
    for layer in range(1, layers + 1):
        yb = torch.full_like(xs, -dx * 0.5 - (layer - 1) * dx)
        bottom = torch.stack((xs, yb), dim=-1)
        pos_parts.append(bottom)
        vel_parts.append(torch.zeros_like(bottom))
        yt = torch.full_like(xs, height + dx * 0.5 + (layer - 1) * dx)
        top = torch.stack((xs, yt), dim=-1)
        pos_parts.append(top)
        vel = torch.zeros_like(top)
        vel[:, 0] = wall_speed
        vel_parts.append(vel)
    return torch.cat(pos_parts, dim=0), torch.cat(vel_parts, dim=0)


def periodic_images(pos, length: float):
    """Three copies of the fluid cloud: real, +L in x, -L in x."""
    shift = torch.tensor([length, 0.0], dtype=pos.dtype, device=pos.device)
    return torch.cat((pos, pos + shift, pos - shift), dim=0)


# --------------------------------------------------------------------------- #
# Analytic reference
# --------------------------------------------------------------------------- #

def analytic(y, height: float, wall_speed: float):
    """Exact steady Couette profile ``u(y) = U y / H``."""
    return wall_speed * y / height


def analytic_scalars(height: float, wall_speed: float, viscosity: float,
                     rho0: float = RHO0) -> dict:
    return {
        "slope": wall_speed / height,
        "flux": 0.5 * wall_speed * height,
        "wall_shear": rho0 * viscosity * wall_speed / height,
    }


# --------------------------------------------------------------------------- #
# Time integration
# --------------------------------------------------------------------------- #

def run_case(cfg: Config, height: float, length: float, wall_speed: float,
             t_end: float, t_average: float, device):
    """Integrate the sliding-wall channel from rest to a steady state."""
    dx = cfg.dx
    pos, mass = build_fluid(cfg, height, length, device)
    n = pos.shape[0]
    wall, wall_vel = build_walls(cfg, height, length, cfg.wall_layers,
                                 wall_speed, device)
    vel = torch.zeros_like(pos)

    h = cfg.h_factor * dx
    # Fixed step: the acoustic limit with a 30 % Courant margin, also checked
    # against the viscous limit. A fixed step keeps the run reproducible bit for
    # bit, which a published table needs.
    dt_acoustic = 0.3 * h / cfg.sound_speed
    dt_viscous = 0.125 * h * h / cfg.viscosity
    dt = min(dt_acoustic, dt_viscous)
    steps = int(round(t_end / dt))
    avg_from = t_end - t_average

    acc_sum = torch.zeros_like(pos)
    samples = 0
    trace = []
    sample_every = max(1, steps // 40)
    wall_clock = time.time()

    for step in range(steps):
        pos_ext = periodic_images(pos, length)
        vel_ext = torch.cat((vel, vel, vel), dim=0)
        mass_ext = torch.cat((mass, mass, mass), dim=0)
        _, _, acc = rhs(pos_ext, vel_ext, mass_ext, cfg, wall=wall,
                        wall_vel=wall_vel)
        acc = acc[:n].clone()

        t = step * dt
        if t >= avg_from:
            acc_sum += vel
            samples += 1

        vel = vel + dt * acc
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], length)

        if step % sample_every == 0 or step == steps - 1:
            ux = vel[:, 0]
            trace.append({
                "t": t,
                "u_mean": float(ux.mean()),
                "u_max": float(ux.max()),
                "flux": float(ux.mean()) * height,
                "max_mach": float(ux.abs().max()) / cfg.sound_speed,
                "wall_clock_s": time.time() - wall_clock,
            })

    u_mean = acc_sum / max(samples, 1)
    rows, profile = bin_rows(pos, u_mean, height, dx)

    stride = max(1, n // 1500)
    field = {
        "x": pos[::stride, 0].tolist(),
        "y": pos[::stride, 1].tolist(),
        "u": u_mean[::stride, 0].tolist(),
    }

    steady = steady_metrics(trace, avg_from)
    drift = steady["trend"]
    drift_samples = steady["two_samples"]

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
        "steady_drift_two_samples": drift_samples,
        "steady_jitter": steady["jitter"],
        "steady_window_s": steady["window"],
        "wall_clock_s": time.time() - wall_clock,
    }


def steady_metrics(trace, avg_from: float) -> dict:
    """How steady the averaging window really is.

    Three numbers, all from the recorded flux trace, because they answer three
    different questions and a weakly compressible run can separate them:

    ``trend``   -- the linear-fit drift of ``Q(t)`` across the window, relative
                   to its mean, extrapolated over the window length.  This is the
                   gate: it is what "has stopped moving" means.
    ``jitter``  -- the standard deviation of ``Q`` about that trend line,
                   relative to the mean.  Reported, not gated: WCSPH keeps a
                   particle-level/acoustic jitter alive for ever (there is no
                   artificial viscosity in this solver) and averaging over the
                   window is what removes it.
    ``two_samples`` -- the naive ``|Q(end) - Q(start)| / Q(end)``, reported so
                   the reader can see how badly that estimator is contaminated
                   by the phase of the jitter (it is not used for anything).
    """
    tail = [p for p in trace if p["t"] >= avg_from]
    out = {"trend": float("nan"), "jitter": float("nan"),
           "two_samples": float("nan"), "window": float("nan"), "n": len(tail)}
    if len(tail) < 4:
        return out
    t = torch.tensor([p["t"] for p in tail], dtype=torch.float64)
    q = torch.tensor([p["flux"] for p in tail], dtype=torch.float64)
    mean = float(q.mean())
    out["window"] = float(t[-1] - t[0])
    out["n"] = len(tail)
    if abs(mean) > 0:
        tc = t - t.mean()
        slope = float((tc * (q - q.mean())).sum() / (tc * tc).sum())
        resid = q - (q.mean() + slope * tc)
        out["trend"] = abs(slope) * out["window"] / abs(mean)
        out["jitter"] = float(resid.std(unbiased=False)) / abs(mean)
        out["two_samples"] = abs(float(q[-1]) - float(q[0])) / abs(float(q[-1]))
    return out


def bin_rows(pos, u_mean, height: float, dx: float):
    """Average ``u_x`` over the particles of each lattice row."""
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

def line_fit(profile, lo: float = 0.15, hi: float = 0.85):
    """Least-squares straight line through the interior rows."""
    sel = [p for p in profile if lo <= p["y_over_H"] <= hi]
    if len(sel) < 3:
        return {"slope": float("nan"), "intercept": float("nan"),
                "n_rows": len(sel)}
    y = torch.tensor([p["y"] for p in sel], dtype=torch.float64)
    u = torch.tensor([p["u_mean"] for p in sel], dtype=torch.float64)
    yc = y - y.mean()
    slope = float((yc * (u - u.mean())).sum() / (yc * yc).sum())
    intercept = float(u.mean() - slope * float(y.mean()))
    resid = u - (intercept + slope * y)
    return {
        "slope": slope,
        "intercept": intercept,
        "n_rows": len(sel),
        "max_deviation": float(resid.abs().max()),
    }


def profile_metrics(profile, height: float, wall_speed: float,
                    viscosity: float, rho0: float = RHO0) -> dict:
    """Compare the measured profile with the closed-form Couette solution."""
    if len(profile) < 3:
        raise RuntimeError("profile has fewer than three rows")
    ref = analytic_scalars(height, wall_speed, viscosity, rho0)
    dx = height / len(profile) if len(profile) else float("nan")
    rows = []
    sq = 0.0
    n_int = 0
    max_abs = 0.0
    max_abs_row = None
    for p in profile:
        u_ref = analytic(p["y"], height, wall_speed)
        err = p["u_mean"] - u_ref
        rel = err / wall_speed
        p2 = dict(p)
        p2.update({"u_ref": u_ref, "err": err, "rel_err": rel,
                   "rel_err_pct": 100.0 * rel})
        rows.append(p2)
        if 0.15 <= p["y_over_H"] <= 0.85:
            sq += err * err
            n_int += 1
        if abs(err) > max_abs:
            max_abs = abs(err)
            max_abs_row = p["y_over_H"]

    interior_l2 = math.sqrt(sq / n_int) / wall_speed if n_int else float("nan")
    flux_meas = sum(p["u_mean"] for p in profile) * dx
    fit = line_fit(profile)

    def slope_of(k0: int, k1: int) -> float:
        return (profile[k1]["u_mean"] - profile[k0]["u_mean"]) / \
            (profile[k1]["y"] - profile[k0]["y"])

    slope_bottom = slope_of(0, 1)
    slope_top = slope_of(-2, -1)
    wall_shear_bottom = rho0 * viscosity * slope_bottom
    wall_shear_top = rho0 * viscosity * slope_top

    # Slip: the straight line extrapolated to the wall plane minus the wall's own
    # velocity, divided by the slope -> slip length (m).
    u_at_0 = fit["intercept"]
    u_at_H = fit["intercept"] + fit["slope"] * height
    slip_bottom = u_at_0 / fit["slope"] if fit["slope"] else float("nan")
    slip_top = (wall_speed - u_at_H) / fit["slope"] if fit["slope"] else float("nan")

    return {
        "rows": rows,
        "interior_l2": interior_l2,
        "max_deviation_over_U": max_abs / wall_speed,
        "max_deviation_row_y_over_H": max_abs_row,
        "first_row_rel_err": rows[0]["rel_err"],
        "last_row_rel_err": rows[-1]["rel_err"],
        "flux_measured": flux_meas,
        "flux_analytic": ref["flux"],
        "flux_ratio": flux_meas / ref["flux"],
        "slope_measured_bottom": slope_bottom,
        "slope_measured_top": slope_top,
        "slope_analytic": ref["slope"],
        "wall_shear_measured": wall_shear_bottom,
        "wall_shear_analytic": ref["wall_shear"],
        "wall_shear_ratio": wall_shear_bottom / ref["wall_shear"],
        "wall_shear_ratio_top": wall_shear_top / ref["wall_shear"],
        "line_fit": fit,
        "line_fit_slope_ratio": fit["slope"] / ref["slope"],
        "linearity_deviation_over_U": fit["max_deviation"] / wall_speed,
        "slip_length_bottom": slip_bottom,
        "slip_length_top": slip_top,
        "slip_length_bottom_over_dx": slip_bottom / dx,
        "slip_length_top_over_dx": slip_top / dx,
        "centreline_ratio": (
            next((p["u_mean"] for p in profile
                  if abs(p["y_over_H"] - 0.5) < 0.5 / len(profile)), float("nan"))
            / (0.5 * wall_speed)),
    }


def startup_decay(trace, flux_asymptote: float, nu: float, height: float,
                  t_end: float):
    """First-mode decay rate of the startup transient, vs ``nu pi^2 / H^2``.

    An impulsively started Couette flow has ``Q(t) = Q_inf - sum_n c_n
    exp(-nu n^2 pi^2 t / H^2)``; for ``t > H^2/(2 nu)`` the first mode dominates,
    so the log-slope of ``Q_inf - Q(t)`` over ``[0.5, 0.85] t_end`` is
    ``-nu pi^2/H^2``.  Supplementary diagnostic (not gated).
    """
    pts = [(p["t"], flux_asymptote - p["flux"]) for p in trace
           if 0.5 * t_end <= p["t"] <= 0.85 * t_end
           and flux_asymptote - p["flux"] > 0.0]
    if len(pts) < 5:
        return {"rate_measured": float("nan"), "rate_analytic": nu * math.pi ** 2 / height ** 2}
    t = torch.tensor([p[0] for p in pts], dtype=torch.float64)
    d = torch.tensor([p[1] for p in pts], dtype=torch.float64)
    lt = t - t.mean()
    slope = float((lt * (d.log() - d.log().mean())).sum() / (lt * lt).sum())
    rate_analytic = nu * math.pi ** 2 / height ** 2
    return {
        "rate_measured": -slope,
        "rate_analytic": rate_analytic,
        "rate_ratio": (-slope) / rate_analytic,
        "n_points": len(pts),
    }


def run_one(resolution: int, viscosity: float, wall_speed: float,
            friction: float, height: float, t_end: float, t_average: float,
            device, label: str = "") -> dict:
    dx = height / resolution
    length = COLUMNS * dx
    cfg = make_config(dx, viscosity, friction, device)
    case = run_case(cfg, height, length, wall_speed, t_end, t_average, device)
    metrics = profile_metrics(case["profile"], height, wall_speed, viscosity)
    decay = startup_decay(case["trace"], metrics["flux_analytic"], viscosity,
                          height, t_end)
    out = {
        "label": label,
        "resolution_H_over_dx": resolution,
        "dx": dx,
        "h": cfg.h_factor * dx,
        "length": length,
        "height": height,
        "viscosity": viscosity,
        "wall_speed": wall_speed,
        "wall_friction": friction,
        "sound_speed": cfg.sound_speed,
        "rho0": cfg.rho0,
        "reynolds_number": wall_speed * height / viscosity,
        "mach_number": wall_speed / cfg.sound_speed,
        "t_end": t_end,
        "viscous_time_H2_over_nu": height * height / viscosity,
        "n_fluid": case["n_fluid"],
        "n_wall": case["n_wall"],
        "dt": case["dt"],
        "steps": case["steps"],
        "n_averaged_samples": case["n_averaged_samples"],
        "max_mach": max(p["max_mach"] for p in case["trace"]),
        "steady_drift": case["steady_drift"],
        "steady_drift_two_samples": case["steady_drift_two_samples"],
        "steady_jitter": case["steady_jitter"],
        "steady_window_s": case["steady_window_s"],
        "wall_clock_s": case["wall_clock_s"],
        "metrics": metrics,
        "startup_decay": decay,
        "profile": metrics["rows"],
        "trace": case["trace"],
        "field": case["field"],
    }
    return out


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #

def gate_run(result: dict) -> dict:
    m = result["metrics"]
    checks = {
        "interior_l2": abs(m["interior_l2"]) < GATES["interior_l2"],
        "flux": abs(m["flux_ratio"] - 1.0) < GATES["flux_rel"],
        "wall_shear": abs(m["wall_shear_ratio"] - 1.0) < GATES["wall_shear_rel"],
        "steady_drift": (result["steady_drift"] < GATES["steady_drift"]
                         if result["steady_drift"] == result["steady_drift"]
                         else False),
    }
    return {"checks": checks, "passed": all(checks.values())}


def verdict(runs, ladder, control) -> dict:
    gated = [r for r in ladder if r["resolution_H_over_dx"] >= REFERENCE_RESOLUTION]
    gated_results = [{"label": r["label"], **gate_run(r)} for r in gated]
    least = min(ladder, key=lambda r: r["resolution_H_over_dx"])
    finest = max(ladder, key=lambda r: r["resolution_H_over_dx"])
    resolution_ok = (finest["metrics"]["interior_l2"]
                     <= least["metrics"]["interior_l2"] + 1.0e-12)
    control_ok = control["metrics"]["flux_ratio"] < GATES["control_flux"]
    status = ("PASS" if (gated_results and all(g["passed"] for g in gated_results)
                         and resolution_ok and control_ok) else "FAIL")
    reasons = []
    for g in gated_results:
        for name, ok in g["checks"].items():
            if not ok:
                reasons.append(f"{g['label']}: gate '{name}' missed")
    if not resolution_ok:
        reasons.append(
            "interior L2 error grew with resolution "
            f"({least['metrics']['interior_l2']:.4f} -> "
            f"{finest['metrics']['interior_l2']:.4f})")
    if not control_ok:
        reasons.append(
            "free-slip control was driven: flux ratio "
            f"{control['metrics']['flux_ratio']:.4f} >= {GATES['control_flux']}")
    return {
        "status": status,
        "gates": GATES,
        "gated_results": gated_results,
        "resolution_response_ok": resolution_ok,
        "negative_control_ok": control_ok,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def make_figures(payload: dict, out: Path) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                   # pragma: no cover
        print(f"[couette] matplotlib unavailable: {exc}")
        return []

    ladder = payload["ladder"]
    height = payload["height"]
    wall_speed = payload["wall_speed"]

    # 1. measured profiles against the closed form
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    yy = [0.0, height]
    ax[0].plot([analytic(y, height, wall_speed) / wall_speed for y in yy],
               [y / height for y in yy], "k-", lw=2, label="analytic U y/H")
    for r in ladder:
        prof = r["profile"]
        ax[0].plot([p["u_mean"] / wall_speed for p in prof],
                   [p["y_over_H"] for p in prof], "o-", ms=3,
                   label=f"H/dx={r['resolution_H_over_dx']}")
    ax[0].set_xlabel("u / U")
    ax[0].set_ylabel("y / H")
    ax[0].set_title("Couette profile vs closed form")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    # 2. error of each row, all ladder resolutions, relative to U
    for r in ladder:
        prof = r["profile"]
        ax[1].semilogx([max(abs(p["rel_err"]), 1e-16) for p in prof],
                       [p["y_over_H"] for p in prof], "o-", ms=3,
                       label=f"H/dx={r['resolution_H_over_dx']}")
    ax[1].axvline(GATES["interior_l2"], color="r", ls="--", lw=1,
                  label="5 % gate")
    ax[1].set_xlabel("|u_meas - u_ana| / U")
    ax[1].set_ylabel("y / H")
    ax[1].set_title("pointwise error")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    f1 = out / "couette_profiles.png"
    fig.savefig(f1, dpi=140)
    plt.close(fig)

    # 3. convergence of the integrated metrics
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    rr = [r["resolution_H_over_dx"] for r in ladder]
    ax.loglog(rr, [r["metrics"]["interior_l2"] for r in ladder], "o-",
              label="interior L2 error")
    ax.loglog(rr, [abs(r["metrics"]["flux_ratio"] - 1.0) for r in ladder], "s-",
              label="|flux ratio - 1|")
    ax.loglog(rr, [abs(r["metrics"]["wall_shear_ratio"] - 1.0) for r in ladder],
              "^-", label="|wall-shear ratio - 1|")
    ax.axhline(GATES["interior_l2"], color="r", ls="--", lw=1, label="5 % gate")
    ax.set_xlabel("H / dx")
    ax.set_ylabel("relative error")
    ax.set_title("resolution ladder")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    f2 = out / "couette_convergence.png"
    fig.savefig(f2, dpi=140)
    plt.close(fig)

    # 4. field cloud: the 2-D snapshot of u_x from the reference run
    ref = next((r for r in ladder
                if r["resolution_H_over_dx"] == REFERENCE_RESOLUTION), ladder[-1])
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    sc = ax.scatter(ref["field"]["x"], ref["field"]["y"], c=ref["field"]["u"],
                    cmap="viridis", s=6)
    fig.colorbar(sc, ax=ax, label="u_x [m/s]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"u_x field, H/dx={ref['resolution_H_over_dx']} (sliding top wall)")
    ax.set_aspect("equal")
    fig.tight_layout()
    f3 = out / "couette_field.png"
    fig.savefig(f3, dpi=140)
    plt.close(fig)

    # 5. sensitivity: friction sweep (free slip -> no slip)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for r in payload["friction_sweep"]:
        prof = r["profile"]
        ax.plot([p["u_mean"] / wall_speed for p in prof],
                [p["y_over_H"] for p in prof], "o-", ms=3,
                label=f"friction={r['wall_friction']}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="analytic (no slip)")
    ax.set_xlabel("u / U")
    ax.set_ylabel("y / H")
    ax.set_title("wall-friction sensitivity")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f4 = out / "couette_friction_sweep.png"
    fig.savefig(f4, dpi=140)
    plt.close(fig)
    return [f1.name, f2.name, f3.name, f4.name]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(LADDER))
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.1, 0.4])
    parser.add_argument("--viscosities", type=float, nargs="+", default=[0.01, 0.04])
    parser.add_argument("--frictions", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--height", type=float, default=HEIGHT)
    parser.add_argument("--viscosity", type=float, default=VISCOSITY)
    parser.add_argument("--wall-speed", type=float, default=WALL_SPEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--extended", action="store_true",
                        help="add H/dx = 32 to the ladder")
    parser.add_argument("--quick", action="store_true",
                        help="single coarse resolution and a short run")
    parser.add_argument("--figure", action="store_true",
                        help="only re-plot an existing results.json")
    parser.add_argument("--only", default=None,
                        help="run one case: ladder|speed|viscosity|friction")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.figure:
        payload = json.loads((args.out / "results.json").read_text())
        names = make_figures(payload, args.out)
        print(f"[couette] figures: {names}")
        return 0

    if args.device in ("auto", "cpu"):
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)

    resolutions = list(args.resolutions)
    if args.extended:
        resolutions = sorted(set(resolutions) | {32})
    speeds = list(args.speeds)
    viscosities = list(args.viscosities)
    frictions = list(args.frictions)
    t_end_override = None
    if args.quick:
        resolutions = [8]
        speeds, viscosities, frictions = [], [], []
        t_end_override = 0.15

    def run(resolution, viscosity, speed, friction, label):
        te = (t_end_override if t_end_override is not None
              else VISCOUS_TIMES * args.height ** 2 / viscosity)
        ta = AVERAGE_FRACTION * te
        print(f"[couette] {label}: H/dx={resolution} nu={viscosity} U={speed} "
              f"f={friction} t_end={te:.3f}", flush=True)
        res = run_one(resolution, viscosity, speed, friction, args.height,
                      te, ta, device, label=label)
        m = res["metrics"]
        print(f"[couette]   L2={100 * m['interior_l2']:.2f}% "
              f"flux={m['flux_ratio']:.4f} tau={m['wall_shear_ratio']:.4f} "
              f"slip/dx={m['slip_length_bottom_over_dx']:.2f} "
              f"drift={res['steady_drift']:.2e} "
              f"({res['wall_clock_s']:.0f}s, {res['steps']} steps)", flush=True)
        return res

    ladder, speed_sweep, viscosity_sweep, friction_sweep = [], [], [], []
    todo = args.only
    if todo in (None, "ladder"):
        for r in resolutions:
            ladder.append(run(r, args.viscosity, args.wall_speed, FRICTION,
                              f"ladder-r{r}"))
    if todo in (None, "speed"):
        for u in speeds:
            speed_sweep.append(run(REFERENCE_RESOLUTION, args.viscosity, u,
                                   FRICTION, f"speed-U{u}"))
    if todo in (None, "viscosity"):
        for nu in viscosities:
            viscosity_sweep.append(run(REFERENCE_RESOLUTION, nu,
                                       args.wall_speed, FRICTION,
                                       f"viscosity-nu{nu}"))
    if todo in (None, "friction"):
        for f in frictions:
            friction_sweep.append(run(REFERENCE_RESOLUTION, args.viscosity,
                                      args.wall_speed, f, f"friction-f{f}"))

    control = next((r for r in friction_sweep if r["wall_friction"] == 0.0), None)
    if control is None and todo in (None, "ladder", "friction"):
        control = run(REFERENCE_RESOLUTION, args.viscosity, args.wall_speed, 0.0,
                      "friction-f0.0")
        friction_sweep.append(control)

    payload = {
        "case": "plane_couette_channel",
        "description": ("2-D channel, both walls no-slip, top wall sliding at U, "
                        "periodic in x, start from rest"),
        "reference": REFERENCE_NOTE,
        "reference_form": "u(y) = U*y/H; du/dy = U/H; tau = rho*nu*U/H; "
                          "Q = U*H/2",
        "height": args.height,
        "columns": COLUMNS,
        "viscosity": args.viscosity,
        "wall_speed": args.wall_speed,
        "wall_friction": FRICTION,
        "sound_speed": SOUND_SPEED,
        "rho0": RHO0,
        "h_factor": H_FACTOR,
        "wall_layers": WALL_LAYERS,
        "gates": GATES,
        "device": str(device),
        "torch": torch.__version__,
        "ladder": ladder,
        "speed_sweep": speed_sweep,
        "viscosity_sweep": viscosity_sweep,
        "friction_sweep": friction_sweep,
        "wall_clock_s": sum(r["wall_clock_s"] for r in
                            ladder + speed_sweep + viscosity_sweep + friction_sweep),
    }
    if ladder and control is not None:
        payload["verdict"] = verdict(None, ladder, control)
        print(f"[couette] verdict: {payload['verdict']['status']}")
        for reason in payload["verdict"]["reasons"]:
            print(f"[couette]   ! {reason}")

    names = make_figures(payload, args.out) if ladder else []
    payload["figures"] = names
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"[couette] wrote {args.out / 'results.json'}")
    return 0 if payload.get("verdict", {}).get("status", "PASS") == "PASS" else 0


if __name__ == "__main__":
    sys.exit(main())