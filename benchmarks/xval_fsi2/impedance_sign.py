"""Controlled experiment: does the *sign* of the acoustic wall impedance matter?

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Inherited state: ``docs/xval_fsi2/STRONG_COUPLING_AND_ROD_STABILITY.md`` item 2 of
its "next round" list -- "diagnose the sign of the Riemann wall-pressure term
(``boundary_pressure_from_fluid``): it should damp a moving wall and instead
destabilises every configuration tested so far".

Predeclared design (written before the numbers were produced)
-------------------------------------------------------------
Decision
    Does the acoustic (Riemann star-state) wall pressure have to enter the load
    transfer with the heavy-wall sign ``p* = p_f + rho_f c_f (u_w - u_f).n``, i.e.
    is the pre-fix ``-`` convention an anti-damping defect rather than a
    modelling choice?
Hypothesis
    H1 (defect): the pre-fix term injects energy; the derived sign removes it.
    H0: the sign is immaterial to the coupled response (the term is dominated by
    something else), in which case the destabilisation reported earlier has a
    different cause.
Alternatives / arms
    ``off``      -- ``wall_impedance = False`` (no acoustic term at all).
    ``pre_fix``  -- the shipped-minus convention, ``wall_impedance_sign = -1``.
    ``physical`` -- ``wall_impedance_sign = +1``, the heavy-wall star state
                    (pinned analytically by ``tests/test_fsi2_impedance_sign.py``).
Manipulated variable: only the sign/switch of the acoustic term.  Same particle
set, ``dx``, ``dt``, kernel, inlet, rod sub-cycling, ``rho_s/rho_f`` (1, the
benchmark point), and the same loose coupling driver for every arm.
Outcome metrics (no fitted constants)
    ``sigma`` = least-squares slope of ``ln(max_i |v_beam_i|)`` vs ``t`` over the
    measurement window [0.25, 1] of the run [1/s] -- positive = energy injected;
    ``v_end`` = ``max_i |v_beam_i|`` at the last step [m/s];
    ``KE_beam`` [J/m], ``KE_fluid`` [J/m] at the last step.
Predeclared thresholds and stopping rule
    * divergent: any non-finite state value, or ``v_end > 0.3 m/s`` (the
      drag-driven response scale at this resolution is ~3e-2 m/s, carried over
      from the previous round's predeclared gate);
    * the prediction ``sigma(physical) < sigma(off) < sigma(pre_fix)`` is either
      met or the hypothesis is reported as falsified -- no threshold is moved
      after the run;
    * a run stops early only on a non-finite state.
Environment / reproduction
    torch float64 CPU, deterministic lattice, ``dx = 20 mm`` (2559 fluid
    particles, ~43 ms per fluid solve at the start of the run), ``c0 = 5 m/s``,
    ``U_mean = 0.2 m/s`` similarity-scaled case (``Re = 100`` held), 200 steps at
    ``dt = 1.73e-4 s``.  The long production run of the surviving arm is a
    separate script (``run_fsi2_benchmark.py``) so that this probe stays cheap.
Output
    ``docs/xval_fsi2/assets/impedance_sign.json`` (written by this script; every
    number in the report is read from it).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensorsph.fsi2_similarity import similarity_corrected_case
from tensorsph.fsi2_turek_hron import initial_state, step_fsi2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

ARMS = {"off": None, "pre_fix": -1.0, "physical": +1.0}


def _case(dx, ratio, c0=5.0, u_mean=0.2, sign=None):
    case = similarity_corrected_case(U_mean=u_mean)
    case.dx = dx
    case.sound_speed = c0
    case.beam_rho = 1000.0 * ratio
    if sign is None:
        case.wall_impedance = False
    else:
        case.wall_impedance = True
        case.wall_impedance_sign = sign
    return case


def _beam_ke(case, state):
    """Transverse kinetic energy of the rod per unit thickness [J/m]."""
    m_node = case.beam_rho * case.beam_thickness * case.beam_length / case.n_beam_nodes
    return 0.5 * m_node * float((state.beam_vel ** 2).sum())


def _fluid_ke(state):
    m = getattr(state, "fluid_mass", None)
    if m is None:
        return float("nan")
    return 0.5 * float((m * state.fluid_vel.pow(2).sum(dim=1)).sum())


def run_arm(arm, dx, ratio, nst, window=(0.25, 1.0)):
    case = _case(dx, ratio, sign=ARMS[arm])
    state = initial_state(case)
    dt = case.dt()
    row = {}
    t0 = time.time()
    speeds, times = [], []
    for k in range(nst):
        row = step_fsi2(state, case, dt)
        v = float(state.beam_vel.norm(dim=1).max())
        if not (math.isfinite(v) and math.isfinite(row["max_speed"])):
            return {"arm": arm, "finite": False, "step": k,
                    "wall_clock_s": time.time() - t0}
        speeds.append(v)
        times.append(state.time)
    speeds = np.asarray(speeds, dtype=float)
    times = np.asarray(times, dtype=float)
    lo = int(window[0] * (len(speeds) - 1))
    hi = int(window[1] * (len(speeds) - 1))
    tt = times[lo:hi + 1]
    ss = np.log(np.maximum(speeds[lo:hi + 1], 1e-300))
    slope = float(np.polyfit(tt, ss, 1)[0]) if len(tt) > 2 else float("nan")
    return {
        "arm": arm,
        "finite": True,
        "n_steps": nst,
        "dt": dt,
        "dx": dx,
        "ratio_solid_fluid": ratio,
        "sound_speed": case.sound_speed,
        "wall_impedance": case.wall_impedance,
        "wall_impedance_sign": case.wall_impedance_sign,
        "sigma_per_s": slope,
        "v_end": float(speeds[-1]),
        "v_max": float(speeds.max()),
        "max_fluid_speed_end": float(row["max_speed"]),
        "ke_beam_end": _beam_ke(case, state),
        "ke_fluid_end": _fluid_ke(state),
        "n_fluid": int(state.fluid_pos.shape[0]),
        "tip_y_end": float(row["tip_y"]),
        "drag_end": float(row["drag"]),
        "lift_end": float(row["lift"]),
        "divergent": bool(speeds[-1] > 0.3),
        "wall_clock_s": time.time() - t0,
        "speeds_sample": [float(s) for s in speeds[:: max(nst // 20, 1)]],
        "times_sample": [float(t) for t in times[:: max(nst // 20, 1)]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/xval_fsi2/assets/impedance_sign.json")
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--nst", type=int, default=200)
    ap.add_argument("--ratio", type=float, default=1.0)
    args = ap.parse_args()

    results = {}
    for arm in ("off", "pre_fix", "physical"):
        results[arm] = run_arm(arm, args.dx, args.ratio, args.nst)
        r = results[arm]
        print(json.dumps({k: r.get(k) for k in
                          ("arm", "finite", "sigma_per_s", "v_end",
                           "max_fluid_speed_end", "divergent")}))

    pred = (results["physical"].get("sigma_per_s", math.nan)
            < results["off"].get("sigma_per_s", math.nan)
            < results["pre_fix"].get("sigma_per_s", math.nan))
    out = {
        "date": time.strftime("%Y-%m-%d"),
        "anchor": "XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID",
        "prediction": "sigma(physical) < sigma(off) < sigma(pre_fix)",
        "prediction_met": bool(pred),
        "thresholds": {"drag_accel_scale_m_per_s": 0.03,
                       "divergent_beam_speed_m_per_s": 0.3,
                       "sigma_window": [0.25, 1.0]},
        "arms": results,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("prediction_met:", pred, "->", out_path)


if __name__ == "__main__":
    main()