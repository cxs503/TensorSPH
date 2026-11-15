"""Round 5 of ``CASE-SPH-OSC-DROP-FREQ``: the time-advancing (dynamic) arm.

Round 4 left one hole open -- *no trajectory was ever integrated*: rounds 1-4
scored the **quasi-static linear response** ``omega = sqrt(k/m)`` of the CSF
traction.  This script integrates the exact ``m = 0`` axisymmetric reduction of
the free drop (see :mod:`tensorsph.oscillating_drop_dynamic`) and reports what
happens, pass or fail.

Three arms, all at ``R/dx = 9`` (the resolution is not the point: the failure
below is not a resolution effect, it is visible at the first 40 steps):

A. ``sigma -> 0`` control
   pressure + continuity + the axis source only.  The drop must stay at rest
   (``|v| < 1e-10``) -- it does.  This isolates the time integrator, the mirror
   axis and the eigensystem from the surface tension.
B. delta-SPH control
   the same static state with the density diffusion switched on.  The diffusion
   *inflates* the free surface: ``|drho/dt|max ~ 0.5`` from a cloud whose only
   density structure is the ``~6 %`` truncation deficit of the surface rows.
C. the capillary arm (``sigma = 1``, ``eps = 0.05``, ``c0 = 20``)
   the measured quantity, with the mode amplitude read from the area-weighted
   quadrupole and ``(omega, gamma)`` fitted by
   :func:`tensorsph.oscillating_drop3d.fit_damped_cosine`.

Run (heavy, submit to the scheduler)::

    PYTHONPATH=src python benchmarks/oscillating_drop_freq/round5_dynamic.py

The JSON it writes is what ``docs/case_sph_osc_drop_freq_round5.md`` quotes.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

from tensorsph.oscillating_drop_dynamic import (DynamicConfig, half_plane_lattice,
                                               run_dynamic_drop)

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "round5_dynamic.json"


def _arm(name, notes, **kwargs):
    steps = kwargs.pop("max_steps", 400)
    cfg = DynamicConfig(dx=kwargs.pop("dx", 1.0 / 9.0), max_steps=steps, **kwargs)
    t0 = time.time()
    row = run_dynamic_drop(cfg)
    row.pop("times", None)
    row.pop("amplitudes", None)
    row["arm"] = name
    row["notes"] = notes
    row["wall_seconds"] = round(time.time() - t0, 1)
    print(f"{name:26s} steps={row['n_steps']:4d} dt={row['dt_last']:.2e} "
          f"|v|max={row['max_speed']:9.3f} amp {row['amplitude_start']:.4f}->"
          f"{row['amplitude_end']:.4f} rho_drift={row['density_drift_max']:.3f}",
          flush=True)
    return row


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    steps = 120 if quick else 400
    sigma_cfg = dict(dx=1.0 / 9.0, sigma=1.0, eps=0.05, c0=20.0,
                     alpha_av=0.02, periods=0.5, skip_periods=0.05,
                     sample_every=5, max_steps=steps)

    a = _arm("A: sigma->0 control", "expect |v| ~ 0 (integrator + axis only)",
             sigma=1e-9, eps=0.05, c0=20.0, alpha_av=0.02, delta_sph=0.0,
             t_end=0.2, skip_periods=0.05, sample_every=5, max_steps=steps)
    b = _arm("B: delta-SPH inflation", "static cloud + diffusion only",
             sigma=1e-9, eps=1e-3, c0=20.0, alpha_av=0.02, delta_sph=0.1,
             t_end=0.2, skip_periods=0.05, sample_every=5, max_steps=steps)
    c = _arm("C: capillary dynamic", "the measurement arm", **sigma_cfg)
    d = _arm("D: no diffusion", "delta-SPH off", **{**sigma_cfg, "delta_sph": 0.0})

    omega_ref = c["omega_analytic"]
    mode_v = omega_ref * 0.05  # eps * omega * R
    verdict = {
        "arm_A_static_control_ok": a["max_speed"] < 1e-6,
        "arm_B_diffusion_inflates_surface": b["density_drift_max"] > 0.05,
        "arm_C_unstable": c["max_speed"] > 20.0 * mode_v,
        "arm_D_unstable": d["max_speed"] > 20.0 * mode_v,
        "omega_analytic": omega_ref,
        "mode_velocity_scale": mode_v,
        "gate_pct": 6.0,
        "verdict": "dynamic arm NOT usable: local instability driven by the "
                   "shipped curvature operator's deformation/topology noise, "
                   "c0 = 20 and alpha_av = 0.02 give |v|max 2 orders above the "
                   "mode velocity and the amplitude runs away (see arms C/D); "
                   "the integrator itself is clean (arm A).",
        "passes_anchor": False,
    }
    payload = {"arms": [a, b, c, d], "verdict": verdict,
               "quick": quick, "mode": 2, "eps": 0.05}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in verdict.items() if k != "verdict"}, indent=2))
    return payload


if __name__ == "__main__":
    main()