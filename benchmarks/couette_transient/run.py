#!/usr/bin/env python3
"""CASE-SPH-COUETTE-TRANSIENT: impulsive start-up of plane Couette flow.

Physics
-------
Fluid at rest between a stationary lower wall and an upper wall that is set into
tangential motion at ``t = 0`` with speed ``U``; channel height ``H``, periodic
in ``x``, no-slip on both walls, no body force.  The 1-D problem is

    u_t = nu u_yy,   u(y,0) = 0,   u(0,t) = 0,   u(H,t) = U,

whose exact solution (separation of variables, Fourier series of ``-U y/H``) is

    u(y,t) = U y/H + (2U/pi) sum_{n>=1} (-1)^n/n sin(n pi y/H) e^{-n^2 pi^2 nu t/H^2}

with ``t* = nu t / H^2``.  Companion exact scalars used as gates:

    tau_w(t)/(rho nu U/H) = 1 + 2 sum_{n>=1} (-1)^n e^{-n^2 pi^2 t*}
    Q(t)/(U H/2)          = 1 - (8/pi^2) sum_{k>=0} e^{-(2k+1)^2 pi^2 t*}/(2k+1)^2

The case (impulsive plane-Couette start-up compared against this series) is the
one used by Morris, Fox & Zhu, *J. Comput. Phys.* 136 (1997) 214-226, Sec. V.

Reference values are **evaluated in code** from the closed forms above (no
hand-copied numbers); the series itself is checked against a Crank-Nicolson
finite-difference solution of the same PDE in
``tests/test_couette_transient_series.py``.

Usage::

    PYTHONPATH=src:. python benchmarks/couette_transient/run.py --quick
    PYTHONPATH=src:. python benchmarks/couette_transient/run.py \
        --out benchmarks/couette_transient/out
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmarks.couette.run import (  # noqa: E402
    bin_rows,
    build_fluid,
    build_walls,
    make_config,
    periodic_images,
)
from tensorsph.droplet_impact import resolve_device, rhs  # noqa: E402

OUT = REPO / "benchmarks" / "couette_transient" / "out"

#: Case constants, identical to benchmarks/couette/run.py so the two cases can be
#: read side by side.
HEIGHT = 0.1            # m
COLUMNS = 12            # fluid columns in x
RHO0 = 1000.0
SOUND_SPEED = 20.0
GAMMA = 7.0
H_FACTOR = 1.3
WALL_LAYERS = 3
VISCOSITY = 0.02        # m^2/s
WALL_SPEED = 0.2        # m/s  -> Re = U H / nu = 1, Ma = U/c0 = 0.01
FRICTION = 1.0          # no-slip blend
DT_FACTOR = 0.3         # dt = DT_FACTOR * h / c0 (acoustic Courant margin)

LADDER = (10, 16, 24)
REFERENCE_RESOLUTION = 16
T_STAR_SAMPLES = (0.05, 0.1, 0.2, 0.3, 0.5)
T_STAR_GATED = 0.1      # first gated instant (boundary layer resolved)
T_STAR_END = 0.6
SAMPLE_WINDOW_TSTAR = 0.005   # averaging half-window in viscous time units
INTERIOR_BAND = (0.10, 0.90)

#: Predeclared thresholds (see benchmarks/couette_transient/DESIGN.md).
GATES = {
    "l2_rel": 0.08,
    "flux_rel": 0.08,
    "shear_rel": 0.08,
    "control_flux": 0.20,
    "max_mach": 0.10,
}

REFERENCE_NOTE = (
    "Morris, Fox & Zhu, J. Comput. Phys. 136 (1997) 214-226, Sec. V (plane "
    "Couette start-up from rest); exact series evaluated in code from "
    "u_t = nu u_yy with u(y,0)=0, u(0,t)=0, u(H,t)=U. No number is transcribed "
    "from the paper: the paper PDF is paywalled in this work tree and the "
    "reference is the closed form the paper itself compares against."
)


# --------------------------------------------------------------------------- #
# Exact reference (evaluated, never transcribed)
# --------------------------------------------------------------------------- #

def couette_series(y, t: float, height: float, wall_speed: float,
                   viscosity: float, rel_tol: float = 1e-15,
                   n_max: int = 20000):
    """Exact start-up profile ``u(y,t)`` (Rayleigh/Stokes series for Couette)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if t <= 0.0:
        return np.zeros_like(y)
    steady = wall_speed * y / height
    n = np.arange(1, n_max + 1, dtype=np.float64)
    decay = np.exp(-viscosity * (n * math.pi / height) ** 2 * t)
    keep = decay > rel_tol
    n, decay = n[keep], decay[keep]
    coef = ((-1.0) ** n) / n * decay
    series = (coef[:, None] * np.sin(np.outer(n * math.pi / height, y))).sum(axis=0)
    return steady + (2.0 * wall_speed / math.pi) * series


def exact_scalars(t: float, height: float, wall_speed: float, viscosity: float,
                  rho0: float = RHO0) -> dict:
    """Exact flux and wall shear of the start-up problem at time ``t``."""
    t_star = viscosity * t / height ** 2
    return {
        "t_star": t_star,
        "flux": flux_exact(t_star, height, wall_speed),
        "wall_shear": rho0 * viscosity * wall_speed / height
        * shear_factor_exact(t_star),
    }


def flux_exact(t_star: float, height: float, wall_speed: float) -> float:
    """``Q(t) = int_0^H u dy`` of the start-up profile.

    ``Q/(U H/2) = 1 - (8/pi^2) sum_{k>=0} e^{-(2k+1)^2 pi^2 t*}/(2k+1)^2``.  The
    mode count is set by the decay of the neglected tail (``~1/M`` for the
    ``1/n^2`` series), not by a fixed constant, so the small-``t*`` values are
    converged rather than truncated.
    """
    if t_star <= 0.0:
        return 0.0                      # exact: the fluid starts at rest
    m = int(math.ceil(4.0 / (math.pi * math.sqrt(t_star)))) + 10
    m = max(m, 2000)
    odd = 2.0 * np.arange(0, m, dtype=np.float64) + 1.0
    s = float(np.sum(np.exp(-odd ** 2 * math.pi ** 2 * t_star) / odd ** 2))
    return 0.5 * wall_speed * height * (1.0 - 8.0 / math.pi ** 2 * s)


def shear_factor_exact(t_star: float) -> float:
    """``tau_w / (rho nu U/H)``; 0 at ``t*=0`` (impulsive start), 1 as ``t*->inf``."""
    if t_star <= 0.0:
        return 0.0
    n = np.arange(1, 400, dtype=np.float64)
    return float(1.0 + 2.0 * np.sum(((-1.0) ** n)
                                   * np.exp(-n ** 2 * math.pi ** 2 * t_star)))


# --------------------------------------------------------------------------- #
# Case run
# --------------------------------------------------------------------------- #

def run_case(cfg, height: float, length: float, wall_speed: float,
             t_end: float, sample_times, dt_factor: float,
             snap_times=(), device=None):
    """Integrate the impulsively started channel from rest, sampling profiles."""
    dx = cfg.dx
    pos, mass = build_fluid(cfg, height, length, device)
    n = pos.shape[0]
    wall, wall_vel = build_walls(cfg, height, length, cfg.wall_layers,
                                 wall_speed, device)
    vel = torch.zeros_like(pos)

    h = cfg.h_factor * dx
    dt_acoustic = dt_factor * h / cfg.sound_speed
    dt_viscous = 0.125 * h * h / cfg.viscosity
    dt = min(dt_acoustic, dt_viscous)
    steps = int(round(t_end / dt))
    half_window = 0.5 * SAMPLE_WINDOW_TSTAR * height ** 2 / cfg.viscosity

    samples = {t: {"u_sum": torch.zeros_like(pos), "count": 0, "t_mean": []}
               for t in sample_times}
    snaps = {t: None for t in snap_times}
    trace = []
    sample_every = max(1, steps // 200)
    wall_clock = time.time()

    for step in range(steps):
        pos_ext = periodic_images(pos, length)
        vel_ext = torch.cat((vel, vel, vel), dim=0)
        mass_ext = torch.cat((mass, mass, mass), dim=0)
        _, _, acc = rhs(pos_ext, vel_ext, mass_ext, cfg, wall=wall,
                        wall_vel=wall_vel)
        acc = acc[:n].clone()

        t = step * dt
        for st in sample_times:
            if abs(t - st) <= half_window:
                samples[st]["u_sum"] += vel
                samples[st]["count"] += 1
                samples[st]["t_mean"].append(t)
        for st in snap_times:
            if snaps[st] is None and t >= st:
                stride = max(1, n // 1200)
                snaps[st] = {
                    "t": t,
                    "x": pos[::stride, 0].tolist(),
                    "y": pos[::stride, 1].tolist(),
                    "u": vel[::stride, 0].tolist(),
                }

        vel = vel + dt * acc
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], length)

        if step % sample_every == 0 or step == steps - 1:
            ux = vel[:, 0]
            trace.append({
                "t": t,
                "t_star": cfg.viscosity * t / height ** 2,
                "u_max": float(ux.max()),
                "flux": float(ux.mean()) * height,
                "max_mach": float(ux.abs().max()) / cfg.sound_speed,
            })

    profiles = {}
    for st in sample_times:
        rec = samples[st]
        if rec["count"] == 0:
            profiles[st] = None
            continue
        u_mean = rec["u_sum"] / rec["count"]
        rows, profile = bin_rows(pos, u_mean, height, dx)
        profiles[st] = {
            "target_t": st,
            "t_mean": float(np.mean(rec["t_mean"])),
            "n_frames": rec["count"],
            "rows": rows,
            "profile": profile,
            "y": [p["y"] for p in profile],
            "u": [p["u_mean"] for p in profile],
        }

    return {
        "n_fluid": int(n),
        "n_wall": int(wall.shape[0]),
        "dt": dt,
        "dt_acoustic": dt_acoustic,
        "dt_viscous": dt_viscous,
        "dt_viscous_number": dt * cfg.viscosity / (h * h),
        "steps": steps,
        "profiles": profiles,
        "snapshots": snaps,
        "trace": trace,
        "max_mach": max(p["max_mach"] for p in trace),
        "wall_clock_s": time.time() - wall_clock,
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def sample_metrics(sample: dict, height: float, wall_speed: float,
                   viscosity: float, rho0: float = RHO0) -> dict:
    """Compare one sampled profile against the exact start-up solution."""
    if sample is None:
        return {"missing": True}
    y = np.asarray(sample["y"], dtype=np.float64)
    u = np.asarray(sample["u"], dtype=np.float64)
    t = sample["t_mean"]

    u_ex = couette_series(y, t, height, wall_speed, viscosity)
    band = (y / height >= INTERIOR_BAND[0]) & (y / height <= INTERIOR_BAND[1])
    diff = u[band] - u_ex[band]
    denom = float(np.linalg.norm(u_ex[band]))
    l2_rel = float(np.linalg.norm(diff) / denom) if denom > 0 else float("nan")
    linf_rel = float(np.max(np.abs(diff)) / wall_speed) if diff.size else float("nan")

    q_ex = flux_exact(viscosity * t / height ** 2, height, wall_speed)
    # Trapezoid over the binned rows plus the two half cells to the walls, so the
    # measured flux is an estimate of int_0^H u dy and not of int_{dx/2}^{H-dx/2}.
    if len(y) > 1:
        q_num = float(np.trapezoid(u, y)
                      + 0.5 * y[0] * u[0]
                      + 0.5 * (height - y[-1]) * (u[-1] + wall_speed))
    else:
        q_num = float("nan")

    # wall shear from the two lowest bins; the reference is the *same* secant
    # taken on the exact profile, so discretisation is not charged to the solver.
    tau_num = rho0 * viscosity * (u[1] - u[0]) / (y[1] - y[0]) if len(y) > 1 else float("nan")
    u_ex_two = couette_series(y[:2], t, height, wall_speed, viscosity)
    tau_ref_secant = rho0 * viscosity * (u_ex_two[1] - u_ex_two[0]) / (y[1] - y[0])
    tau_ref = rho0 * viscosity * wall_speed / height * shear_factor_exact(
        viscosity * t / height ** 2)

    steady_err = float(np.max(np.abs(u - wall_speed * y / height)) / wall_speed)
    centre = int(np.argmin(np.abs(y - 0.5 * height)))

    return {
        "t": t,
        "t_star": viscosity * t / height ** 2,
        "l2_rel": l2_rel,
        "linf_rel": linf_rel,
        "centre_u": float(u[centre]),
        "centre_exact": float(couette_series(np.array([y[centre]]), t, height,
                                            wall_speed, viscosity)[0]),
        "flux_num": q_num,
        "flux_exact": q_ex,
        "flux_rel": (q_num / q_ex - 1.0) if q_ex else float("nan"),
        "wall_shear_num": tau_num,
        "wall_shear_exact": tau_ref,
        "wall_shear_exact_secant": tau_ref_secant,
        "shear_rel": (tau_num / tau_ref_secant - 1.0) if tau_ref_secant else float("nan"),
        "steady_profile_err": steady_err,
        "n_rows": int(len(y)),
    }


def gate_run(result: dict, metrics: dict, label: str) -> dict:
    """Apply the predeclared gates to one run."""
    reasons = []
    checked = 0
    for t, m in metrics.items():
        if m.get("missing"):
            reasons.append(f"{label}: no sample at t*={t}")
            continue
        if m["t_star"] < T_STAR_GATED - 1e-12:
            continue
        checked += 1
        if not (m["l2_rel"] < GATES["l2_rel"]):
            reasons.append(f"{label}: t*={m['t_star']:.3f} L2={100*m['l2_rel']:.2f}% "
                           f">= {100*GATES['l2_rel']:.0f}%")
        if abs(m["flux_rel"]) >= GATES["flux_rel"]:
            reasons.append(f"{label}: t*={m['t_star']:.3f} flux off by "
                           f"{100*m['flux_rel']:+.2f}%")
    # The wall-shear gate is declared at t* = 0.1 (DESIGN.md, gate M4): the run is
    # gated on the sampled instant closest to it that is not earlier than it.
    later = [m for m in metrics.values()
             if not m.get("missing") and m["t_star"] >= T_STAR_GATED - 1e-12]
    if later:
        first = min(later, key=lambda m: m["t_star"])
        if abs(first["shear_rel"]) >= GATES["shear_rel"]:
            reasons.append(f"{label}: t*={first['t_star']:.3f} wall shear off by "
                           f"{100*first['shear_rel']:+.2f}%")
    if not math.isfinite(result["max_mach"]):
        reasons.append(f"{label}: non-finite state (max Mach is not finite)")
    if result["max_mach"] > GATES["max_mach"]:
        reasons.append(f"{label}: max Mach {result['max_mach']:.3f} exceeds "
                       f"{GATES['max_mach']}")
    return {"label": label, "n_gated_samples": checked,
            "status": "PASS" if not reasons else "FAIL", "reasons": reasons}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_one(resolution: int, viscosity: float, wall_speed: float, friction: float,
            sound_speed: float = SOUND_SPEED, dt_factor: float = DT_FACTOR,
            height: float = HEIGHT, label: str = "", device=None,
            t_star_end: float = T_STAR_END, samples=T_STAR_SAMPLES):
    dx = height / resolution
    cfg = make_config(dx, viscosity, friction, device)
    cfg = dataclasses.replace(cfg, sound_speed=sound_speed)
    length = COLUMNS * dx
    t_end = t_star_end * height ** 2 / viscosity
    snap_targets = tuple(s * height ** 2 / viscosity for s in (0.1, 0.5))

    print(f"[couette-t] {label}: H/dx={resolution} nu={viscosity} U={wall_speed} "
          f"f={friction} c0={sound_speed} kappa={dt_factor} t_end={t_end:.3f}s",
          flush=True)
    # sample times are declared in viscous units t* = nu t / H^2; the integrator
    # works in seconds, so convert once here (no hand-written times anywhere).
    sample_phys = {s: s * height ** 2 / viscosity for s in samples}
    res = run_case(cfg, height, length, wall_speed, t_end,
                   [sample_phys[s] for s in samples], dt_factor,
                   snap_times=snap_targets, device=device)

    metrics = {}
    for t in samples:
        metrics[t] = sample_metrics(res["profiles"][sample_phys[t]], height,
                                    wall_speed, viscosity)
        m = metrics[t]
        if not m.get("missing"):
            print(f"[couette-t]   t*={m['t_star']:.3f} L2={100*m['l2_rel']:.2f}% "
                  f"Linf={100*m['linf_rel']:.2f}% flux={100*m['flux_rel']:+.2f}% "
                  f"shear={100*m['shear_rel']:+.2f}%", flush=True)

    out = {
        "label": label,
        "resolution": resolution,
        "dx": dx,
        "viscosity": viscosity,
        "wall_speed": wall_speed,
        "wall_friction": friction,
        "sound_speed": sound_speed,
        "dt_factor": dt_factor,
        "reynolds": wall_speed * height / viscosity,
        "mach": wall_speed / sound_speed,
        "dt": res["dt"],
        "dt_viscous_number": res["dt_viscous_number"],
        "steps": res["steps"],
        "n_fluid": res["n_fluid"],
        "n_wall": res["n_wall"],
        "max_mach": res["max_mach"],
        "metrics": {f"{k}": v for k, v in metrics.items()},
        "profiles": {f"{k}": (None if res["profiles"][sample_phys[k]] is None else {
            "t_mean": res["profiles"][sample_phys[k]]["t_mean"],
            "n_frames": res["profiles"][sample_phys[k]]["n_frames"],
            "y": res["profiles"][sample_phys[k]]["y"],
            "u": res["profiles"][sample_phys[k]]["u"],
        }) for k in samples},
        "trace": res["trace"],
        "snapshots": [v for v in res["snapshots"].values() if v is not None],
        "wall_clock_s": res["wall_clock_s"],
    }
    if friction > 0:
        out["gate"] = gate_run(res, metrics, label)
    return out


def verdict(runs: list, control: dict) -> dict:
    """Decision from the predeclared gates.

    The hypothesis in ``DESIGN.md`` is stated *at H/dx >= 16*, so the case
    verdict is carried by the levels at or above the reference resolution; a
    coarse level that misses 8 % is reported (``coarse_level_failures``) as
    convergence evidence, it is not silently dropped and it does not decide the
    case.
    """
    reasons = []
    coarse = []
    for r in runs:
        g = r.get("gate")
        if g and g["status"] != "PASS":
            if r["resolution"] >= REFERENCE_RESOLUTION:
                reasons.extend(g["reasons"])
            else:
                coarse.extend(g["reasons"])

    if not any(r["resolution"] >= REFERENCE_RESOLUTION and r["wall_friction"] > 0
               for r in runs):
        reasons.append(f"no run at the gating resolution H/dx >= "
                       f"{REFERENCE_RESOLUTION}; the case cannot be decided")

    ladder = sorted([r for r in runs if r["wall_friction"] > 0
                     and r["sound_speed"] == SOUND_SPEED
                     and r["dt_factor"] == DT_FACTOR], key=lambda r: r["resolution"])
    if len(ladder) >= 2:
        worst = [max(m["l2_rel"] for m in r["metrics"].values()
                     if not m.get("missing") and m["t_star"] >= T_STAR_GATED - 1e-12)
                 for r in ladder]
        if worst[-1] > worst[0]:
            reasons.append(f"L2 error does not decrease with resolution: "
                           f"{['%.4f' % w for w in worst]}")
    if control is not None:
        m = max(control["metrics"].values(),
                key=lambda m: m["t_star"] if not m.get("missing") else -1)
        achieved = 1.0 + m["flux_rel"]          # control flux / exact flux
        if abs(achieved) > GATES["control_flux"]:
            reasons.append(f"free-slip control still reaches {100*achieved:.1f}% of the "
                           f"exact flux at t*={m['t_star']:.2f} (> "
                           f"{100*GATES['control_flux']:.0f}%) -- the flow is driven "
                           "by something other than the wall shear")
    return {"status": "PASS" if not reasons else "FAIL", "reasons": reasons,
            "coarse_level_failures": coarse,
            "gating_policy": ("gates applied to every run at H/dx >= "
                              f"{REFERENCE_RESOLUTION} (predeclared H1); coarse "
                              "levels are reported as convergence evidence")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--resolutions", type=int, nargs="+", default=list(LADDER))
    ap.add_argument("--sweep", action="store_true",
                    help="also run the compressibility and dt sweeps + control")
    ap.add_argument("--quick", action="store_true",
                    help="smoke: one coarse resolution, short window")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.figures:
        payload = json.loads((args.out / "results.json").read_text())
        return 0 if make_figures(payload, args.out) else 1

    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)

    samples = T_STAR_SAMPLES
    t_star_end = T_STAR_END
    resolutions = list(args.resolutions)
    if args.quick:
        resolutions = [10]
        samples = (0.1,)
        t_star_end = 0.15

    runs = []
    for r in resolutions:
        runs.append(run_one(r, VISCOSITY, WALL_SPEED, FRICTION,
                            label=f"ladder-r{r}", device=device,
                            t_star_end=t_star_end, samples=samples))

    sweeps = {"sound_speed": [], "dt_factor": []}
    control = None
    if args.sweep:
        for c0 in (10.0, 40.0):
            sweeps["sound_speed"].append(
                run_one(REFERENCE_RESOLUTION, VISCOSITY, WALL_SPEED, FRICTION,
                        sound_speed=c0, label=f"c0-{c0:g}", device=device,
                        t_star_end=t_star_end, samples=samples))
        for kappa in (0.15,):
            sweeps["dt_factor"].append(
                run_one(REFERENCE_RESOLUTION, VISCOSITY, WALL_SPEED, FRICTION,
                        dt_factor=kappa, label=f"kappa-{kappa:g}", device=device,
                        t_star_end=t_star_end, samples=samples))
        control = run_one(REFERENCE_RESOLUTION, VISCOSITY, WALL_SPEED, 0.0,
                          label="control-freeslip", device=device,
                          t_star_end=t_star_end, samples=samples)

    payload = {
        "case": "couette_transient_startup",
        "description": ("2-D channel at rest, lower wall no-slip at rest, upper "
                        "wall impulsively set to U at t=0, periodic in x"),
        "reference": REFERENCE_NOTE,
        "reference_form": ("u = U y/H + (2U/pi) sum (-1)^n/n sin(n pi y/H) "
                           "exp(-n^2 pi^2 nu t/H^2); tau_w/(rho nu U/H) = "
                           "1 + 2 sum (-1)^n exp(-n^2 pi^2 t*)"),
        "height": HEIGHT,
        "viscosity": VISCOSITY,
        "wall_speed": WALL_SPEED,
        "rho0": RHO0,
        "sound_speed": SOUND_SPEED,
        "gamma": GAMMA,
        "h_factor": H_FACTOR,
        "wall_layers": WALL_LAYERS,
        "reynolds": WALL_SPEED * HEIGHT / VISCOSITY,
        "mach": WALL_SPEED / SOUND_SPEED,
        "t_star_samples": list(samples),
        "t_star_gated": T_STAR_GATED,
        "interior_band": list(INTERIOR_BAND),
        "sample_window_tstar": SAMPLE_WINDOW_TSTAR,
        "gates": GATES,
        "device": str(device),
        "torch": torch.__version__,
        "ladder": runs,
        "sweeps": sweeps,
        "control": control,
        "wall_clock_s": sum(r["wall_clock_s"] for r in
                            runs + sweeps["sound_speed"] + sweeps["dt_factor"]
                            + ([control] if control else [])),
    }
    payload["verdict"] = verdict(runs, control)
    print(f"[couette-t] verdict: {payload['verdict']['status']}")
    for reason in payload["verdict"]["reasons"]:
        print(f"[couette-t]   ! {reason}")

    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"[couette-t] wrote {args.out / 'results.json'}")
    make_figures(payload, args.out)
    return 0


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def make_figures(payload: dict, out: Path) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[couette-t] figures unavailable: {exc}")
        return []

    height = payload["height"]
    nu = payload["viscosity"]
    U = payload["wall_speed"]
    names = []
    ref = next((r for r in payload["ladder"]
                if r["resolution"] == REFERENCE_RESOLUTION), payload["ladder"][-1])

    # 1) profile overlays at the sampled instants
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for key, prof in sorted(ref["profiles"].items(), key=lambda kv: float(kv[0])):
        if prof is None:
            continue
        y = np.asarray(prof["y"], dtype=float)
        u = np.asarray(prof["u"], dtype=float)
        t = prof["t_mean"]
        ax.plot(u / U, y / height, "o", ms=3.0, alpha=0.75,
                label=f"SPH t*={nu*t/height**2:.2f}")
        yy = np.linspace(0, height, 200)
        ax.plot(couette_series(yy, t, height, U, nu) / U, yy / height, "-",
                lw=1.2, color=ax.lines[-1].get_color())
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="steady U y/H")
    ax.set_xlabel("u / U")
    ax.set_ylabel("y / H")
    ax.set_title(f"Couette impulsive start-up (H/dx={ref['resolution']})")
    ax.legend(fontsize=7, loc="upper left")
    name = "couette_transient_profiles.png"
    fig.tight_layout()
    fig.savefig(out / name, dpi=140)
    plt.close(fig)
    names.append(name)

    # 2) transient scalar histories: flux and wall shear
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    tr = ref["trace"]
    ts = np.array([p["t_star"] for p in tr])
    q = np.array([p["flux"] for p in tr])
    tgrid = np.linspace(1e-4, ts.max(), 400)
    axes[0].plot(tgrid, [flux_exact(s, height, U) for s in tgrid], "k-",
                 label="exact")
    axes[0].plot(ts, q, "o", ms=2.0, label=f"SPH H/dx={ref['resolution']}")
    axes[0].set_xlabel("t* = nu t / H^2")
    axes[0].set_ylabel("Q  [m^2/s]")
    axes[0].set_title("mass flux")
    axes[0].legend(fontsize=8)
    axes[1].plot(tgrid, [shear_factor_exact(s) for s in tgrid], "k-", label="exact")
    for r in payload["ladder"]:
        ms = [m for m in r["metrics"].values() if not m.get("missing")]
        axes[1].plot([m["t_star"] for m in ms],
                     [m["wall_shear_num"] / (payload["rho0"] * nu * U / height)
                      for m in ms], "o-", ms=4.0, label=f"H/dx={r['resolution']}")
    axes[1].axhline(1.0, color="grey", ls=":", lw=1.0)
    axes[1].set_xlabel("t*")
    axes[1].set_ylabel("tau_w / (rho nu U/H)")
    axes[1].set_title("wall shear factor")
    axes[1].legend(fontsize=7)
    name = "couette_transient_scalars.png"
    fig.tight_layout()
    fig.savefig(out / name, dpi=140)
    plt.close(fig)
    names.append(name)

    # 3) error vs resolution and field cloud
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for key in payload["t_star_samples"]:
        xs, ys = [], []
        for r in payload["ladder"]:
            m = r["metrics"].get(f"{key}")
            if m and not m.get("missing"):
                xs.append(r["resolution"])
                ys.append(100 * m["l2_rel"])
        if xs:
            axes[0].plot(xs, ys, "o-", label=f"t*={float(key):.2f}")
    axes[0].axhline(100 * payload["gates"]["l2_rel"], color="r", ls="--",
                    lw=1.0, label="gate 8%")
    axes[0].set_xlabel("H/dx")
    axes[0].set_ylabel("interior L2 error [%]")
    axes[0].set_title("resolution convergence")
    axes[0].legend(fontsize=7)

    if ref["snapshots"]:
        snap = ref["snapshots"][0]
        sc = axes[1].scatter(snap["x"], snap["y"],
                             c=np.asarray(snap["u"], dtype=float) / U, s=8,
                             cmap="viridis")
        fig.colorbar(sc, ax=axes[1], label="u / U")
        axes[1].set_xlabel("x [m]")
        axes[1].set_ylabel("y [m]")
        axes[1].set_title(f"u/U field at t*={nu*snap['t']/height**2:.2f}")
    name = "couette_transient_convergence_field.png"
    fig.tight_layout()
    fig.savefig(out / name, dpi=140)
    plt.close(fig)
    names.append(name)
    print(f"[couette-t] figures: {names}")
    return names


if __name__ == "__main__":
    sys.exit(main())