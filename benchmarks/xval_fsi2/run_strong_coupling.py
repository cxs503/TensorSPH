"""Controlled experiment: does interface sub-iteration cure the FSI2 added-mass blow-up?

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Inherited state (``docs/xval_fsi2/COUPLED_RUN_DEFECTS.md``): the *loosely*
coupled driver diverges at the benchmark density ratio ``rho_s/rho_f = 1``
(beam speed 3.5e50 m/s inside 100 steps) and no load-side filter helps, because
there is no coupling iteration to converge.

Decision
--------
Can a per-step interface sub-iteration (Aitken / IQN-ILS, see
``src/tensorsph/fsi2_strong_coupling.py``) make the FSI2 point
``rho_s/rho_f = 1`` stable, so that the coupled run -- not just the V1-V4
static operators -- can be compared with the benchmark?

Alternatives (identical particle set, dx, dt, kernel, inlet, rod sub-cycling)
--------------------------------------------------------------------------
* ``loose``   -- shipped driver, one fluid solve per step (baseline, known FAIL).
* ``strong-none``  -- interface iteration without acceleration (plain fixed point).
* ``aitken``  -- interface iteration + dynamic secant relaxation.
* ``iqn-ils`` -- interface iteration + interface quasi-Newton (least squares).

Manipulated variable: the coupling scheme, at fixed ``rho_s/rho_f``.
Controls: ``rho_s/rho_f = 10`` (the regime the loose driver already survives)
for every scheme, so a scheme that "stabilises" ratio 1 by damping the physics
is detected by the control column.

Predeclared outcome metric
--------------------------
``sigma = ln(|v_beam|max(t_end)/|v_beam|max(t_20)) / (t_end - t_20)`` [1/s],
the growth rate of the beam speed, plus ``|v_beam|max`` at the last step.
Thresholds (fixed before the runs, identical to the previous round):
  * divergent  : non-finite state, or ``sigma > 0`` **and** ``v_end > 0.3 m/s``
                 (the drag-driven response at this scale is ~3e-2 m/s after
                 0.03 s, so a merely responding run stays far below 0.3 m/s);
  * stable     : everything finite and not divergent.
Stopping rule: a run stops as soon as a state value becomes non-finite, the
interface iteration aborts, or the predeclared step budget is spent.

Resources: dx = 20 mm coarse probe (2559 fluid particles, 43 ms per fluid
solve), < 8 cores locally for the probe; the long production run is submitted
to the cluster queue.

Output: ``docs/xval_fsi2/assets/strong_coupling.json`` (nothing hand-typed).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from tensorsph.fsi2_similarity import similarity_corrected_case
from tensorsph.fsi2_turek_hron import initial_state, step_fsi2
from tensorsph.fsi2_strong_coupling import run_fsi2_strong, step_fsi2_strong

DRAG_ACCEL_SCALE = 0.03  # m/s scale of a load-driven beam response, see module doc


def _case(dx, ratio, c0=5.0, u_mean=0.2):
    case = similarity_corrected_case(U_mean=u_mean)
    case.dx = dx
    case.sound_speed = c0
    case.beam_rho = 1000.0 * ratio
    return case


def run_loose(dx, ratio, nst):
    case = _case(dx, ratio)
    state = initial_state(case)
    dt = case.dt()
    v20 = None
    v_end = 0.0
    finite = True
    k = 0
    for k in range(nst):
        row = step_fsi2(state, case, dt)
        v = float(state.beam_vel.norm(dim=1).max())
        if not (math.isfinite(v) and math.isfinite(row["max_speed"])):
            finite = False
            break
        if k == 20:
            v20 = v
        v_end = v
    return _verdict_row(dx, ratio, "loose", nst, dt, v20, v_end, finite, k + 1,
                        iters=[])


def run_strong(dx, ratio, nst, solver, max_iter=20, tol=1.0e-3):
    case = _case(dx, ratio)
    state = initial_state(case)
    dt = case.dt()
    v20 = None
    v_end = 0.0
    finite = True
    iters = []
    k = 0
    aborted = False
    for k in range(nst):
        row, stats = step_fsi2_strong(state, case, dt, interface_solver=solver,
                                      max_iter=max_iter, tol=tol)
        iters.append(stats.iterations)
        v = float(state.beam_vel.norm(dim=1).max())
        if not (math.isfinite(v) and math.isfinite(row["max_speed"])) or stats.aborted:
            finite = False
            aborted = True
            break
        if k == 20:
            v20 = v
        v_end = v
    out = _verdict_row(dx, ratio, f"strong-{solver}", nst, dt, v20, v_end, finite,
                       k + 1, iters)
    out["aborted"] = aborted
    return out


def _verdict_row(dx, ratio, variant, nst, dt, v20, v_end, finite, steps_done, iters):
    sigma = float("inf")
    if finite and v20 and v20 > 0.0 and v_end > 0.0:
        n_span = max(steps_done - 20, 1)
        sigma = math.log(v_end / v20) / (n_span * dt)
    divergent = (not finite) or (sigma > 0.0 and v_end > 10 * DRAG_ACCEL_SCALE)
    return {
        "dx": dx, "ratio": ratio, "variant": variant, "n_steps_budget": nst,
        "steps_done": steps_done, "dt": dt,
        "v_at_step20": v20, "v_end": v_end, "sigma_per_s": sigma,
        "finite": finite, "verdict": "divergent" if divergent else "stable",
        "mean_iters": (sum(iters) / len(iters)) if iters else None,
        "max_iters": max(iters) if iters else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/xval_fsi2/assets/strong_coupling.json")
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--nst", type=int, default=200)
    ap.add_argument("--ratios", default="1,10")
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    rows = []
    t0 = time.time()
    for ratio in ratios:
        for variant in ("loose", "strong-none", "strong-aitken", "strong-iqn-ils"):
            if variant == "loose":
                r = run_loose(args.dx, ratio, args.nst)
            else:
                solver = variant.split("-", 1)[1]
                r = run_strong(args.dx, ratio, args.nst, solver)
            rows.append(r)
            print(f"{variant:14s} ratio={ratio:5.1f} {r['verdict']:10s} "
                  f"sigma={r['sigma_per_s']:+11.3e} v_end={r['v_end']:.3e} "
                  f"iters={r['mean_iters']} steps={r['steps_done']}", flush=True)

    payload = {
        "decision": ("does per-step interface sub-iteration make the Turek-Hron "
                     "FSI2 density ratio rho_s/rho_f = 1 stable?"),
        "metric": "sigma = d ln|v_beam|max/dt [1/s]; divergent iff sigma>0 and v_end>0.3 m/s",
        "drag_accel_scale_m_per_s": DRAG_ACCEL_SCALE,
        "u_mean": 0.2, "sound_speed": 5.0, "dx": args.dx,
        "n_steps_budget": args.nst, "rows": rows,
        "wall_clock_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("wrote", args.out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()