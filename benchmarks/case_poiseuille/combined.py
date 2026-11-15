#!/usr/bin/env python3
"""CASE-POISEUILLE-SPH, R3: combined Couette-Poiseuille with an interior reversal.

What was open
-------------
The anchor ``CASE-POISEUILLE-SPH`` is graded on **pressure-driven** plane
Poiseuille flow (``u_t = nu u_yy + g`` with two static no-slip walls), and the
neighbouring ``benchmarks/couette`` is graded on **wall-driven** flow (one
moving wall, no body force).  Neither run, and no run in the family, has ever
put the two drivers in the *same* channel, and none has ever held a
sign-changing (reversed) velocity against a no-slip wall.  Three claims that the
two separate certificates quietly assume are therefore untested:

1. **Superposition.** The governing equation is linear, so
   ``u_CP = u_Poiseuille + u_Couette`` must hold *exactly* in the continuum.  The
   discrete WCSPH system is not linear (Tait EOS in the pressure, density
   summation, the wall-friction blend), so this is a falsifiable statement about
   the code, not a tautology.  In particular a wall operator that leaks or
   over-drags shows up here as a residual that neither single-driver case can
   see.
2. **Reversal.** With an adverse pressure gradient (``g < 0``) and the top wall
   moving in ``+x``, the profile crosses zero inside the channel: the fluid
   above the crossing is dragged forward by the wall while the fluid below it is
   pushed backward by the pressure gradient.  The no-slip wall then carries
   shear of the opposite sign to the pressure-forced wall, and the near-wall
   layer of the *backflow* side has never been exercised.
3. **The moving-wall path of this case.** ``run.drive`` used to hardwire two
   static planes; every number quoted from this case went through ``plane_vel =
   None``.  The moving-wall code path (``wall_mirror.plane_walls`` with a
   per-plane velocity) is graded here for the first time *inside this case*.

The case
--------
Same geometry, same solver, same row binning as R1/R2; the only differences are
one Dirichlet datum (top wall moving at ``U_w``) and a reversed body force::

    u_t = nu u_yy + g,   u(0,t) = 0,   u(H,t) = U_w,   u(y,0) = 0

with the steady solution

    u(y) = g y (H - y) / (2 nu) + U_w y / H ,
    S(y) = mu du/dy = rho g (H - 2 y) / 2 + rho nu U_w / H ,
    tau_bottom = S(0) ,   tau_top = S(H) ,   S(0) - S(H) = rho g H .

Both wall shears, the zero crossing and the net momentum budget are therefore
known in closed form; the profile stays a quadratic, so one least-squares
quadratic through the interior band recovers *both* wall shears without a
near-wall stencil (the estimator family that
``docs/verification_poiseuille_wall_shear_estimator.md`` found to be the only
unbiased one at this resolution).

Operating point
---------------
``H = 0.1 m``, ``nu = 0.01 m^2/s``, ``rho0 = 1000``, ``c0 = 20 m/s`` (the case's
declared values), ``g = -4.905 m/s^2``, ``U_w = +1.470 m/s``.  The profile then
runs from ``u_min = -0.0984 m/s`` (backflow) to ``U_w``, so ``max|u|/c0 = 0.074``
-- inside the WCSPH Mach envelope that
``docs/verification_poiseuille_ma_envelope.md`` measured (``Ma <= 0.1``), which
is why the amplitude could not simply be raised to make the reversal deeper.
Reynolds number ``U_w H / nu = 14.7``: laminar, as the anchor.

Predeclared gates (thresholds fixed before the ladder was run)
--------------------------------------------------------------
C1 combined profile RMS error, all rows, normalised by ``U_w``       <= 5 %
C2 zero-crossing position error, ``|y* - y*_analytic| / H``          <= 5 %
C3 superposition residual ``u_CP - (u_P + u_C)``, RMS over rows
   normalised by ``U_w``                                            <= 5 %
C4a bottom-wall shear from the interior quadratic fit               <= 5 %
C4b top-wall shear from the interior quadratic fit                  <= 5 %
C5 net momentum budget ``rho H (g - d<u>/dt)`` vs ``rho g H``        <= 2 %
C6 backflow peak ``u_min`` magnitude, relative error                <= 5 %
   and the sign must be negative
C7 moving-wall FD reference vs closed form (amplitude)              <= 1 %
C8 static-wall regression: ``drive`` with no ``planes`` and with
   two explicitly static planes is bit-identical                     exact

C1/C2/C4/C6 are also reported per rung on ``H/dx = 8/12/16`` so the direction of
the residual with resolution is visible.

Usage
-----
    PYTHONPATH=.:src python3 benchmarks/case_poiseuille/combined.py --quick
    PYTHONPATH=.:src python3 benchmarks/case_poiseuille/combined.py \
        --out benchmarks/case_poiseuille/out

Exit status is non-zero when any predeclared gate fails.  ``out/combined.json``
carries every number quoted in the note; nothing is transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmarks.case_poiseuille import ref as refmod  # noqa: E402
from benchmarks.case_poiseuille import run as caserun  # noqa: E402

TAG = "CASE-POISEUILLE-SPH/R3"
CASE_KEY = "poiseuille-combined-couette"

HEIGHT = caserun.HEIGHT
VISCOSITY = caserun.VISCOSITY
RHO0 = caserun.RHO0
SOUND_SPEED = 20.0
MU = RHO0 * VISCOSITY

G_ADVERSE = -4.905          # adverse pressure gradient (half the R1 magnitude)
WALL_SPEED = 1.47           # top wall, +x
RESOLUTIONS = (8, 12, 16)
ANCHOR_RESOLUTION = 12      # the R1 anchor rung, reported for comparison

T_END = 1.0
STEADY_TIMES = [round(0.80 + 0.02 * i, 10) for i in range(11)]
DRIFT_FROM = 0.80           # momentum-budget window (same as the R1 anchor)

GATES = {
    "C1_profile_rms_pct": 5.0,
    "C2_zero_crossing_pct_of_H": 5.0,
    "C3_superposition_rms_pct": 5.0,
    "C4a_tau_bottom_pct": 5.0,
    "C4b_tau_top_pct": 5.0,
    "C5_momentum_budget_pct": 2.0,
    "C6_backflow_peak_pct": 5.0,
    "C7_reference_fd_vs_analytic_pct": 1.0,
}


# --------------------------------------------------------------------------- #
# Closed form
# --------------------------------------------------------------------------- #

def analytic_profile(y, height: float = HEIGHT, g: float = G_ADVERSE,
                     nu: float = VISCOSITY,
                     wall_speed: float = WALL_SPEED) -> np.ndarray:
    """Steady Couette-Poiseuille profile ``g y (H-y) / (2 nu) + U_w y / H``."""
    y = np.asarray(y, dtype=float)
    return g * y * (height - y) / (2.0 * nu) + wall_speed * y / height


def analytic_wall_shear(height: float = HEIGHT, g: float = G_ADVERSE,
                        nu: float = VISCOSITY, rho: float = RHO0,
                        wall_speed: float = WALL_SPEED) -> dict:
    """Both wall shears ``S(0)``, ``S(H)`` of the same closed form."""
    mu = rho * nu
    s0 = rho * g * height / 2.0 + mu * wall_speed / height
    sh = -rho * g * height / 2.0 + mu * wall_speed / height
    return {"tau_bottom": float(s0), "tau_top": float(sh),
            "net_bottom_minus_top": float(s0 - sh),
            "body_force_per_area": float(rho * g * height)}


def analytic_zero_crossing(height: float = HEIGHT, g: float = G_ADVERSE,
                           nu: float = VISCOSITY,
                           wall_speed: float = WALL_SPEED) -> float | None:
    """``y*`` where the profile changes sign, or ``None`` if it does not."""
    a = -g / (2.0 * nu)                       # coefficient of y^2
    b = g * height / (2.0 * nu) + wall_speed / height
    disc = b * b                               # c = 0: y = 0 is always a root
    roots = [0.0, -b / a] if a != 0.0 else []
    inner = [r for r in roots if 0.0 < r < height + 1e-15 and math.isfinite(r)]
    return float(inner[0]) if inner else None


def analytic_extrema(height: float = HEIGHT, g: float = G_ADVERSE,
                     nu: float = VISCOSITY,
                     wall_speed: float = WALL_SPEED) -> dict:
    """Interior extremum (the backflow peak) of the quadratic profile."""
    a = -g / (2.0 * nu)
    b = g * height / (2.0 * nu) + wall_speed / height
    y_ext = -b / (2.0 * a) if a != 0.0 else float("nan")
    if not (0.0 < y_ext < height):
        return {"y": float("nan"), "u": float("nan")}
    return {"y": float(y_ext), "u": float(analytic_profile([y_ext], height, g, nu,
                                                           wall_speed)[0])}


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #

def quadratic_wall_shear(y, u, height: float, rho: float = RHO0,
                         nu: float = VISCOSITY) -> dict:
    """Recover both wall shears from one least-squares quadratic ``u = a y^2 + b y + c``.

    The steady Couette-Poiseuille profile *is* quadratic, so a fit over the
    interior band contains the whole solution: the curvature gives the body
    force, the slope at ``y = 0`` and ``y = H`` give the two wall shears.  No
    near-wall row is differenced, which is what makes it usable at ``H/dx <= 16``
    (see ``docs/verification_poiseuille_wall_shear_estimator.md``).
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    a, b, c = np.polyfit(y, u, 2)
    mu = rho * nu
    return {
        "a": float(a), "b": float(b), "c": float(c),
        "tau_bottom": float(mu * b),
        "tau_top": float(mu * (2.0 * a * height + b)),
        "wall_speed_implied": float(a * height ** 2 + b * height + c),
        "body_force_implied": float(-2.0 * nu * a),
    }


def measured_zero_crossing(y, u) -> float | None:
    """First interior sign change of a sampled profile (linear interpolation).

    The Dirichlet walls are themselves zeros of the profile, so a zero sitting
    on ``y = 0`` or ``y = H`` is not a crossing and is skipped: only a sign
    change *between* two interior rows locates the reversal.
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    on_wall = np.isclose(y, y[0]) | np.isclose(y, y[-1])
    for k in range(u.size - 1):
        if u[k] * u[k + 1] < 0.0:
            w = u[k] / (u[k] - u[k + 1])
            return float(y[k] + w * (y[k + 1] - y[k]))
        if u[k + 1] == 0.0 and not on_wall[k + 1]:
            return float(y[k + 1])
    return None


def _rms(values) -> float:
    values = list(values)
    return float(np.sqrt(np.mean(np.square(values)))) if values else float("nan")


def _window_profiles(case: dict) -> np.ndarray:
    """Mean profile over the steady window (columns follow ``case['y_rows']``)."""
    return np.mean(case["u_samples"], axis=0)


def _drift(case: dict, t_from: float = DRIFT_FROM) -> dict:
    """Least-squares slope of the channel-mean velocity over the steady window.

    Read from the sampled window profiles rather than the sparse progress trace
    (which only keeps ~10 points over the whole run), so the budget closure is
    estimated from the same data the profile gates use.
    """
    times = np.asarray(case["sample_times"], dtype=float)
    u_samples = np.asarray(case["u_samples"], dtype=float)
    sel = times >= t_from - 0.5 * case["dt"]
    t = times[sel]
    u = u_samples[sel].mean(axis=1)           # mean over rows (uniform lattice)
    if t.size < 3:
        return {"slope": float("nan"), "stderr": float("nan"), "n": int(t.size)}
    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    coef, res, *_ = np.linalg.lstsq(A, u, rcond=None)
    resid = u - A @ coef
    dof = max(1, len(t) - 2)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return {"slope": float(coef[0]), "stderr": float(math.sqrt(cov[0, 0])),
            "n": int(t.size)}


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #

def run_channel(resolution: int, device, g: float = G_ADVERSE,
                wall_speed: float = WALL_SPEED, t_end: float = T_END,
                samples=None, plane_vel=True) -> dict:
    """One steady run at one resolution; ``plane_vel=False`` freezes the top wall."""
    dx = HEIGHT / resolution
    length = caserun.COLUMNS * dx
    cfg = caserun.make_case_config(dx, VISCOSITY, device,
                                   sound_speed=SOUND_SPEED)
    times = STEADY_TIMES if samples is None else samples
    if plane_vel:
        planes = [(0.0, +1), (HEIGHT, -1, (wall_speed, 0.0))]
    else:
        planes = [(0.0, +1), (HEIGHT, -1)]
    case = caserun.drive(cfg, HEIGHT, length, lambda t: g, t_end, times, device,
                         planes=planes)
    y = np.asarray(case["y_rows"], dtype=float)
    u = _window_profiles(case)
    case.update({"resolution_H_over_dx": resolution, "y": y, "u_profile": u,
                 "u_analytic": analytic_profile(y), "length": length,
                 "dx": dx, "g": g, "wall_speed": wall_speed})
    return case


def grade_run(case: dict, label: str) -> dict:
    """C1, C2, C4, C5, C6 for one run against the closed form."""
    y, u = case["y"], case["u_profile"]
    # each run is graded against *its own* closed form: the superposition legs
    # (one driver at a time) are different problems, not perturbed versions of
    # the combined one
    u_a = analytic_profile(y, g=case["g"], wall_speed=case["wall_speed"])
    scale = float(np.max(np.abs(u_a)))
    ana = analytic_wall_shear(g=case["g"], wall_speed=case["wall_speed"])
    ext = analytic_extrema()
    ystar_ana = analytic_zero_crossing()

    interior = (y / HEIGHT >= 0.15) & (y / HEIGHT <= 0.85)
    fit = quadratic_wall_shear(y[interior], u[interior], HEIGHT)

    ystar = measured_zero_crossing(y, u)
    ystar_err_pct = (100.0 * abs(ystar - ystar_ana) / HEIGHT
                     if ystar is not None and ystar_ana is not None
                     else float("nan"))
    drift = _drift(case)
    net = RHO0 * HEIGHT * (case["g"] - drift["slope"])
    budget = ana["body_force_per_area"]
    u_min = float(np.min(u))
    # the extreme of the *sampled* profile, not the analytic y: the row grid is
    # fixed, so grading u_min against the analytic value at its own extremum
    # would charge the row position to the solver
    y_min = float(y[int(np.argmin(u))])
    u_min_ref = float(analytic_profile([y_min], g=case["g"],
                                       wall_speed=case["wall_speed"])[0])

    res = {
        "label": label,
        "resolution_H_over_dx": case["resolution_H_over_dx"],
        "g": case["g"],
        "wall_speed": case["wall_speed"],
        "dx": case["dx"],
        "dt": case["dt"],
        "steps": case["steps"],
        "n_fluid": case["n_fluid"],
        "n_ghost": case["n_ghost"],
        "max_mach": max(r["max_mach"] for r in case["trace"]),
        "rows": [{"y": float(yy), "y_over_H": float(yy / HEIGHT),
                  "u": float(uu), "u_analytic": float(aa),
                  "u_residual": float(uu - aa)}
                 for yy, uu, aa in zip(y, u, u_a)],
        "profile_rms_pct": 100.0 * _rms((u - u_a) / scale),
        "profile_worst_row_pct": 100.0 * float(np.max(np.abs(u - u_a)) / scale),
        "zero_crossing_y": ystar,
        "zero_crossing_y_analytic": ystar_ana,
        "zero_crossing_pct_of_H": ystar_err_pct,
        "wall_shear": fit,
        "tau_bottom_analytic": ana["tau_bottom"],
        "tau_top_analytic": ana["tau_top"],
        "tau_bottom_pct": 100.0 * (fit["tau_bottom"] - ana["tau_bottom"]) / ana["tau_bottom"],
        "tau_top_pct": 100.0 * (fit["tau_top"] - ana["tau_top"]) / ana["tau_top"],
        "momentum_budget": {
            "net_drag_per_area": float(net),
            "body_force_per_area": float(budget),
            "ratio": float(net / budget),
            "pct": 100.0 * (net / budget - 1.0),
            "drift_slope": drift["slope"],
            "drift_stderr": drift["stderr"],
            "window": [DRIFT_FROM, case["sample_times"][-1]],
            "n_samples": drift["n"],
        },
        "backflow": {
            "u_min": u_min,
            "y_at_min": y_min,
            "u_min_analytic_at_same_row": u_min_ref,
            "extreme_analytic": ext,
            "peak_pct": 100.0 * (u_min - u_min_ref) / abs(u_min_ref),
            "sign_ok": bool(u_min < 0.0 and u_min_ref < 0.0),
        },
        "wall_clock_s": case["wall_clock_s"],
    }
    return res


# --------------------------------------------------------------------------- #
# Reference (independent finite differences, moving wall allowed)
# --------------------------------------------------------------------------- #

def reference_check(nx: int = 401, n_steps: int = 12000) -> dict:
    """CN-FD with ``u(H) = U_w`` against the closed form; the C7 gate."""
    t_end = STEADY_TIMES[-1]
    sol = refmod.cn_solve(lambda t: G_ADVERSE, HEIGHT, VISCOSITY, t_end,
                          [t_end], nx=nx, n_steps=n_steps,
                          wall_values=(0.0, WALL_SPEED))
    y_fd = sol["y"]
    u_fd = sol["samples"][-1]
    u_ana = analytic_profile(y_fd)
    scale = abs(WALL_SPEED)
    return {
        "scheme": sol["scheme"],
        "nx": nx,
        "n_steps": n_steps,
        "dy": sol["dy"],
        "dt": sol["dt"],
        "wall_values": [0.0, WALL_SPEED],
        "profile_rms_pct": 100.0 * _rms((u_fd - u_ana) / scale),
        "profile_worst_pct": 100.0 * float(np.max(np.abs(u_fd - u_ana)) / scale),
        "zero_crossing_pct_of_H": 100.0 * abs(
            (measured_zero_crossing(y_fd, u_fd) or 0.0)
            - (analytic_zero_crossing() or 0.0)) / HEIGHT,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_suite(device, resolutions=RESOLUTIONS) -> dict:
    wall_clock = time.time()
    runs, raw = [], {}
    for r in resolutions:
        print(f"[{TAG}] combined H/dx={r} ...", flush=True)
        case = run_channel(r, device)
        raw[r] = case
        res = grade_run(case, f"combined_Hdx{r}")
        print(f"[{TAG}]   C1 profile {res['profile_rms_pct']:.2f}%  "
              f"C2 crossing {res['zero_crossing_pct_of_H']:.2f}%H  "
              f"C4a {res['tau_bottom_pct']:+.1f}%  C4b {res['tau_top_pct']:+.1f}%  "
              f"({res['wall_clock_s']:.0f}s)", flush=True)
        runs.append(res)

    # superposition: the two single drivers at the anchor rung
    r = ANCHOR_RESOLUTION if ANCHOR_RESOLUTION in raw else resolutions[-1]
    print(f"[{TAG}] superposition legs at H/dx={r} ...", flush=True)
    pois_case = run_channel(r, device, wall_speed=0.0, plane_vel=False)
    cout_case = run_channel(r, device, g=0.0)
    poiseuille = grade_run(pois_case, f"poiseuille_only_Hdx{r}")
    couette = grade_run(cout_case, f"couette_only_Hdx{r}")
    comb_case = raw[r]
    comb = next(x for x in runs if x["resolution_H_over_dx"] == r)
    y = comb_case["y"]
    u_comb = comb_case["u_profile"]
    u_pois = np.interp(y, pois_case["y"], pois_case["u_profile"])
    u_cout = np.interp(y, cout_case["y"], cout_case["u_profile"])
    residual = u_comb - (u_pois + u_cout)
    superposition = {
        "resolution_H_over_dx": r,
        "rows": [{"y": float(yy), "y_over_H": float(yy / HEIGHT),
                  "u_combined": float(uc), "u_poiseuille_leg": float(up),
                  "u_couette_leg": float(uw), "residual": float(rr)}
                 for yy, uc, up, uw, rr in zip(y, u_comb, u_pois, u_cout, residual)],
        "rms_pct": 100.0 * _rms(residual / abs(WALL_SPEED)),
        "worst_row_pct": 100.0 * float(np.max(np.abs(residual)) / abs(WALL_SPEED)),
        "worst_row_y_over_H": float(y[int(np.argmax(np.abs(residual)))] / HEIGHT),
        "legs": {"poiseuille_profile_rms_pct": poiseuille["profile_rms_pct"],
                 "couette_profile_rms_pct": couette["profile_rms_pct"],
                 "combined_profile_rms_pct": comb["profile_rms_pct"],
                 "poiseuille_tau_bottom_pct": poiseuille["tau_bottom_pct"],
                 "couette_tau_top_pct": couette["tau_top_pct"]},
        "note": ("u(combined) vs u(poiseuille) + u(couette), all three runs at "
                 "the same resolution and from the same at-rest initial state; "
                 "the PDE is linear, the discrete WCSPH system is not"),
    }
    legs = {"poiseuille_only": poiseuille, "couette_only": couette}

    print(f"[{TAG}] moving-wall CN reference ...", flush=True)
    reference = reference_check()

    print(f"[{TAG}] static-wall control (no planes vs static planes) ...",
          flush=True)
    static_control = static_equivalence(device)

    gates = {
        "C1_profile_rms_5pct": all(x["profile_rms_pct"] <= GATES["C1_profile_rms_pct"]
                                   for x in runs),
        "C2_zero_crossing_5pct_H": all(
            x["zero_crossing_pct_of_H"] <= GATES["C2_zero_crossing_pct_of_H"]
            for x in runs),
        "C3_superposition_5pct": superposition["rms_pct"] <= GATES["C3_superposition_rms_pct"],
        "C4a_tau_bottom_5pct": all(abs(x["tau_bottom_pct"]) <= GATES["C4a_tau_bottom_pct"]
                                   for x in runs),
        "C4b_tau_top_5pct": all(abs(x["tau_top_pct"]) <= GATES["C4b_tau_top_pct"]
                                for x in runs),
        "C5_momentum_budget_2pct": all(
            abs(x["momentum_budget"]["pct"]) <= GATES["C5_momentum_budget_pct"]
            for x in runs),
        "C6_backflow_peak_5pct": all(
            x["backflow"]["sign_ok"]
            and abs(x["backflow"]["peak_pct"]) <= GATES["C6_backflow_peak_pct"]
            for x in runs),
        "C7_reference_fd_1pct": reference["profile_rms_pct"] <= GATES["C7_reference_fd_vs_analytic_pct"],
        "C8_static_wall_equivalence": static_control["bit_identical"],
    }
    return {
        "case": CASE_KEY,
        "anchor": "CASE-POISEUILLE-SPH",
        "note": "R3 combined Couette-Poiseuille: superposition, reversal, moving wall",
        "parameters": {
            "height": HEIGHT, "viscosity": VISCOSITY, "rho0": RHO0,
            "sound_speed": SOUND_SPEED, "g": G_ADVERSE, "wall_speed": WALL_SPEED,
            "columns": caserun.COLUMNS, "t_end": T_END,
            "steady_window": [STEADY_TIMES[0], STEADY_TIMES[-1]],
            "reynolds": WALL_SPEED * HEIGHT / VISCOSITY,
            "max_mach_analytic": float(np.max(np.abs(analytic_profile(
                np.linspace(0.0, HEIGHT, 2001))))) / SOUND_SPEED,
        },
        "analytic": {
            "profile": "g y (H-y) / (2 nu) + U_w y / H",
            "wall_shear": analytic_wall_shear(),
            "zero_crossing_y": analytic_zero_crossing(),
            "zero_crossing_over_H": (analytic_zero_crossing() or 0.0) / HEIGHT,
            "extrema": analytic_extrema(),
            "references": [
                "White, Viscous Fluid Flow 3rd ed. sec. 3-2.2 (plane Poiseuille)",
                "White, Viscous Fluid Flow 3rd ed. sec. 3-2.1 (Couette)",
                "Batchelor 1967 sec. 4.2 (superposition of the two)",
            ],
        },
        "thresholds": GATES,
        "runs": runs,
        "superposition": superposition,
        "legs": legs,
        "reference_check": reference,
        "static_wall_control": static_control,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "wall_clock_s": time.time() - wall_clock,
        "generated_by": "benchmarks/case_poiseuille/combined.py",
    }


def static_equivalence(device, resolution: int = 8, t_end: float = 0.02) -> dict:
    """C8: the ``planes`` hook must not change the static-wall path.

    ``drive`` with no ``planes`` argument and ``drive`` handed two explicitly
    static planes differ only in that the second passes ``plane_vel = (0, 0)``
    down the mirror.  A moving-wall feature that perturbs the static path would
    silently restate every number this case has ever published, so the
    equivalence is asserted instead of assumed.
    """
    dx = HEIGHT / resolution
    cfg = caserun.make_case_config(dx, VISCOSITY, device, sound_speed=SOUND_SPEED)
    length = caserun.COLUMNS * dx
    kw = dict(t_end=t_end, sample_times=[t_end * 0.5, t_end], device=device)
    a = caserun.drive(cfg, HEIGHT, length, lambda t: 9.81, **kw)
    b = caserun.drive(cfg, HEIGHT, length, lambda t: 9.81,
                      planes=[(0.0, +1), (HEIGHT, -1, (0.0, 0.0))], **kw)
    da = np.abs(np.asarray(a["u_samples"]) - np.asarray(b["u_samples"]))
    return {
        "resolution_H_over_dx": resolution,
        "t_end": t_end,
        "max_abs_diff": float(da.max()),
        "bit_identical": bool(da.max() == 0.0),
        "note": "drive(planes=None) vs drive(planes=[[0,+1],[H,-1,(0,0)]])",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="one rung (H/dx=8) at reduced cost, gates graded anyway")
    ap.add_argument("--out", type=Path, default=HERE / "out")
    args = ap.parse_args(argv)
    device = torch.device("cpu")
    resolutions = (8,) if args.quick else RESOLUTIONS
    payload = run_suite(device, resolutions)
    payload["quick"] = bool(args.quick)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / ("combined_quick.json" if args.quick else "combined.json")
    target.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"[{TAG}] wrote {target}")
    for k, v in payload["gates"].items():
        print(f"[{TAG}] {k:32s} {'PASS' if v else 'FAIL'}")
    return 0 if payload["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())