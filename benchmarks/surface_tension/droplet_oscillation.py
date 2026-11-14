"""Dynamic (time-integrated) verification of the surface-tension force.

Anchor: ``XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION``.

Why this script exists
----------------------
Every previous round of this anchor produced a *static* reading: the colour
gradient, the stress tensor and the force field were integrated across a fixed
interface and compared with the Young-Laplace jump (``laplace.py``,
``sphinxys_xval.py``, ``css_flux_audit.py``, ``mixture_laplace.py``).  A static
reading cannot tell whether the surface tension that is actually *integrated in
time* converts surface energy into kinetic energy at the right rate -- a force
field can have the correct normal integral and still give the wrong dynamics.
Upstream SPHinXsys' own acceptance for its surface-tension model is a shape
regression of a relaxing square droplet, which is insensitive to the absolute
value of sigma; so the dynamic reading is missing on both sides.

Reference (exact, no adjustable parameter)
------------------------------------------
Inviscid oscillation of a 2-D drop about its circular equilibrium
(Rayleigh 1879; Lamb, *Hydrodynamics* 6th ed., arts. 313-315)::

    omega_m^2 = sigma * m * (m^2 - 1) / (rho * R^3)

``m = 1`` is the neutral translation mode, which anchors the relation at zero
independently of any measurement.  ``m = 2`` is used here: it preserves the drop
area to first order, so the perturbation is pure surface energy and the
reference contains no bulk compression.

Readings (same run, both against an exact reference)
----------------------------------------------------
1. **period** of the mode-2 order parameter ``q2(t)`` (zero-crossing spacing,
   independent of the damping envelope) vs ``T_2 = 2 pi / omega_2``.
2. **dynamic Laplace jump**: EOS-resolved mean interior pressure (``r < R/2``)
   vs ``sigma / R``.  This cannot be passed by a force field whose *magnitude*
   is wrong, unlike the static integral.
3. **falsification control**: the identical run at ``sigma = 0`` -- ``q2`` must
   stay frozen, otherwise the measured period would be a numerical artefact.

Acceptance criterion of the ledger: ``|error| < 15 %``.

Usage::

    PYTHONPATH=src python benchmarks/surface_tension/droplet_oscillation.py
    PYTHONPATH=src python benchmarks/surface_tension/droplet_oscillation.py --pilot
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

from tensorsph.capillary_oscillation import (
    MODE_DEFAULT, RADIUS_DEFAULT, RHO_WATER, SIGMA_WATER_AIR, rayleigh_2d,
    run_point,
)

GATE_PCT = 15.0


def reference_period(sigma=SIGMA_WATER_AIR, rho=RHO_WATER, radius=RADIUS_DEFAULT,
                     mode=MODE_DEFAULT) -> float:
    return 2.0 * math.pi / rayleigh_2d(sigma, rho, radius, mode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out", default="benchmarks/surface_tension/out_droplet_oscillation")
    ap.add_argument("--periods", type=float, default=2.5)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)

    sample_every = 1 if args.pilot else 20
    if args.pilot:
        matrix = [("baseline", 10, {}, 1.0)]
    else:
        matrix = [
            ("baseline", 10, {}, args.periods),
            ("baseline", 16, {}, args.periods),
            ("baseline", 20, {}, args.periods),
            ("viscous_5e-4", 10, dict(viscosity=5.0e-4), 1.0),
            ("delta_diffusion_0.1", 10, dict(density_diffusion=0.1), 1.0),
        ]

    out = {
        "reference": {
            "law": "omega_m^2 = sigma m (m^2-1) / (rho R^3)  (2-D drop, Rayleigh/Lamb)",
            "sigma_n_m": SIGMA_WATER_AIR,
            "rho0_kg_m3": RHO_WATER,
            "radius_m": RADIUS_DEFAULT,
            "mode": MODE_DEFAULT,
            "amp_eps": 0.05,
            "gate_pct": GATE_PCT,
            "period_exact_s": reference_period(),
            "laplace_exact_pa": SIGMA_WATER_AIR / RADIUS_DEFAULT,
            "model": "production CSF (cfg.csf_scheme='consistent'), free surface, "
                     "WCSPH Tait EOS c0=0.5 m/s, gravity off, nu=1e-6 m^2/s, "
                     "adaptive dt (CFL 0.25), density_calibration on",
        },
        "containment_definition": "escaped = any particle beyond 1.6 R; "
                                  "max_r_over_R = largest particle radius / R",
        "ladder": [],
        "control": None,
    }

    for name, n, kw, periods_n in matrix:
        r = run_point(n, SIGMA_WATER_AIR, periods_n, sample_every=sample_every, **kw)
        r["config"] = name
        finite = r["period_err_pct"] == r["period_err_pct"]
        r["period_pass"] = bool(finite and abs(r["period_err_pct"]) < GATE_PCT)
        r["laplace_pass"] = bool(abs(r["dp_err_pct"]) < GATE_PCT)
        r["contained"] = bool(r["escaped_particles"] == 0)
        out["ladder"].append(r)
        print(f"{name:22s} R/dx={n:3d} N={r['particles']:5d} steps={r['steps']:5d} "
              f"T={r['period_measured_s']:.4f}s (exact {r['period_exact_s']:.4f}) "
              f"err={r['period_err_pct']:+.2f}%  "
              f"dp={r['p_interior_mean_pa']:.4f}Pa err={r['dp_err_pct']:+.2f}%  "
              f"max_r/R={r['containment_max_r_over_R']:.3f} "
              f"escaped={r['escaped_particles']} t_esc={r['t_first_escape_s']:.3f}s "
              f"wall={r['wall_s']}s", flush=True)

    if not args.pilot:
        ctrl = run_point(10, 0.0, 0.0, sample_every=sample_every,
                         t_end_s=reference_period())
        out["control"] = ctrl
        print(f"control sigma=0 N={ctrl['particles']} q_initial={ctrl['q_initial']:.5f} "
              f"q_final={ctrl['q_final']:.5f} half_range={ctrl['q_half_range']:.3e} "
              f"max_r/R={ctrl['containment_max_r_over_R']:.3f} "
              f"escaped={ctrl['escaped_particles']}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "oscillation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
