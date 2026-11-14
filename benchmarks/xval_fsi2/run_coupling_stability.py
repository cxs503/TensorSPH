"""Controlled experiment: can the loosely-coupled explicit FSI2 driver run FSI2?

Decision
--------
Whether the shipped coupled Turek-Hron driver (one fluid solve + one explicit
rod step per time step) can be used to produce the benchmark numbers, or whether
the ledger anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`` needs an
implicit / added-mass-aware coupling first.

Competing alternatives (all on the same particle set, same dx, same dt, same
cubic-spline kernel, same inlet)
-----------------------------------------------------------------------------
* ``loose``       -- force from fluid -> explicit rod step, no modification.
* ``impedance``   -- acoustic (Riemann star-state) wall pressure in the load
                     transfer, ``p* = p_f + rho0 c0 (v_wall - v_f).n``.
* ``relaxed``     -- under-relaxation of the nodal load, alpha = 0.05.

Manipulated variable
--------------------
The solid/fluid density ratio.  Turek-Hron FSI2 is posed at
``rho_s/rho_f = 1``; the SPHinXsys regression point is at 10.  The ratio is the
parameter that controls added mass, so it is the control knob.

Outcome metric (predeclared)
----------------------------
``sigma = ln(|v_beam|max(t_end) / |v_beam|max(t_20)) / (t_end - t_20)`` [1/s],
the exponential growth rate of the beam speed.  A run that is merely responding
to the drag load has ``sigma -> 0`` once the load saturates (the drag
acceleration scale is ``F/m = 6.4 N/m / 7 kg ~ 0.9 m/s2``, i.e. ~0.03 m/s after
0.035 s).  A loosely coupled scheme hit by added-mass instability has
``sigma > 0`` and grows without bound.  Thresholds, fixed before the runs:

* ``sigma > 0`` **and** ``max|v_beam| > 10 x 0.03 m/s``  -> divergent (FAIL),
* ``0 < sigma <= 0`` bound (|v_beam| <= 0.3 m/s at the end) -> stable.

Stopping rule: a run is abandoned as soon as a non-finite value appears.
Resources: dx = 20 mm coarse probe (fast), one dx = 12.5 mm confirmation of the
ratio-1 verdict, all local, < 8 cores.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

from tensorsph.fsi2_similarity import similarity_corrected_case
from tensorsph.fsi2_turek_hron import initial_state, step_fsi2

DRAG_ACCEL_SCALE = 0.03  # m/s after 0.035 s, see module docstring


def run_case(dx, ratio, variant, nst, u_mean=0.2, c0=5.0, stores_series=False):
    case = similarity_corrected_case(U_mean=u_mean)
    case.dx = dx
    case.sound_speed = c0
    case.beam_rho = 1000.0 * ratio
    if variant == "impedance":
        case.wall_impedance = True
    elif variant == "relaxed":
        case.wall_impedance = False
        case.coupling_relaxation = 0.05
    else:
        case.wall_impedance = False
    state = initial_state(case)
    dt = case.dt()
    v20 = None
    v_end = 0.0
    series = []
    finite = True
    for k in range(nst):
        row = step_fsi2(state, case, dt)
        v = float(state.beam_vel.norm(dim=1).max())
        if stores_series:
            series.append((row["t"], v, row["tip_y"]))
        if not math.isfinite(v) or not math.isfinite(row["max_speed"]):
            finite = False
            break
        if k == 20:
            v20 = v
        v_end = v
    sigma = float("inf")
    if finite and v20 and v20 > 0.0 and v_end > 0.0:
        sigma = math.log(v_end / v20) / ((nst - 20) * dt)
    verdict = "divergent" if (not finite or (sigma > 0.0 and v_end > 10 * DRAG_ACCEL_SCALE)) else "stable"
    return {
        "dx": dx, "ratio": ratio, "variant": variant, "n_steps": nst, "dt": dt,
        "v_at_step20": v20, "v_end": v_end, "sigma_per_s": sigma,
        "finite": finite, "verdict": verdict, "series": series,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/xval_fsi2/assets/coupling_stability.json")
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--nst", type=int, default=600)
    args = ap.parse_args()

    rows = []
    for variant in ("loose", "impedance", "relaxed"):
        for ratio in (1.0, 2.0, 5.0, 10.0):
            r = run_case(args.dx, ratio, variant, args.nst)
            rows.append(r)
            print(f"{variant:10s} ratio={ratio:5.1f} sigma={r['sigma_per_s']:+11.3e} "
                  f"v20={r['v_at_step20']:.3e} vend={r['v_end']:.3e} -> {r['verdict']}", flush=True)

    # dx-independence of the ratio-1 verdict (coarser probe could be a resolution artefact)
    confirm = run_case(0.0125, 1.0, "loose", 250)
    print(f"confirm dx=0.0125 ratio=1.0 sigma={confirm['sigma_per_s']:+11.3e} "
          f"vend={confirm['v_end']:.3e} -> {confirm['verdict']}", flush=True)

    # time series for the figure: the divergent and the stable control, same variant
    series = {
        "ratio_1": run_case(args.dx, 1.0, "loose", 120, stores_series=True)["series"],
        "ratio_10": run_case(args.dx, 10.0, "loose", args.nst, stores_series=True)["series"],
    }

    payload = {
        "decision": "does the loosely-coupled explicit FSI2 driver run the Turek-Hron point (rho_s/rho_f = 1)?",
        "metric": "sigma = d ln|v_beam|max/dt [1/s]; divergent iff sigma > 0 and v_end > 0.3 m/s",
        "drag_accel_scale_m_per_s": DRAG_ACCEL_SCALE,
        "u_mean": 0.2, "sound_speed": 5.0,
        "rows": rows, "resolution_check": confirm, "series": series,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
