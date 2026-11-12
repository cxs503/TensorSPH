"""CASE-HOLLOW-SPHERE-3D: ``Delta p = 2 sigma (1/R_in + 1/R_out)`` on a 3-D projection-SPH shell.

What this harness scores
------------------------
The analytic reference is the Young-Laplace jump across a *spherical* surface,

    Delta p = sigma (1/R1 + 1/R2) = 2 sigma / R          (one interface, a drop),

and for the hollow sphere -- liquid between ``R_in`` and ``R_out``, gas on both
sides -- the two surfaces add:

    Delta p = 2 sigma (1/R_in + 1/R_out)                 (= 4 sigma / R for a film)

This is the case that makes the verification ladder dimension-complete: the 2-D
anchor (``CASE-LAPLACE-SPH``, ``benchmarks/laplace_sph/``) uses a *circle*, where
the second principal curvature is identically zero, so a solver that silently
computes ``sigma/R`` instead of ``2 sigma/R`` passes it.  Nothing in the 2-D
ladder can catch that; a sphere can.

Two curvature routes are scored, and the split is the point:

``analytic``   ``kappa = 2 (n . r_hat) / R_face`` imposed.  Isolates the rest of
               the chain (colour -> delta_s -> traction -> Poisson -> jump) and
               must meet the 5 % gate.
``mls``        ``kappa = div n`` from the particle field, i.e. the *shipped*
               self-consistent CSF curvature.  Scored against the same closed
               form; whatever it measures is reported, not tuned.

Gates (thresholds fixed before the run, ``platform_tolerance = 0.05``):

    laplace_analytic_finest   |dp/dp_theory - 1| <= 5 %  (analytic route, finest)
    laplace_analytic_all      every level of the analytic route <= 5 %
    laplace_mls_finest        the same for the self-consistent route
    flux_both_faces           each interface's independent traction-flux integral
                              (``-int rho a_r dV / 4 pi R^2``) <= 8 % on the
                              finest shell level -- it never touches the Poisson
                              solve, so it is the cross-check of the jump and it
                              verifies ``R_in`` and ``R_out`` as *separate*
                              curvature radii
    kappa_two_radii           |kappa_face - 2/R_face| <= 5 % on both faces
    delta_normalisation       |int delta_s dn - 1| <= 5 % on the finest level
    negative_control          ``sigma = 0`` must give |dp| < 1e-10 Pa
    spurious_acceleration     measured and reported, NOT gated: the 2-D anchor
                              records the same lattice artefact as an honest FAIL

Run::

    PYTHONPATH=src python benchmarks/hollow_sphere_3d/run.py            # full
    PYTHONPATH=src python benchmarks/hollow_sphere_3d/run.py --quick    # 2 levels

Output: ``benchmarks/hollow_sphere_3d/report.json`` (+ ``figures/`` via
``make_figure.py``).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch

from tensorsph.threed.hollow_sphere import hollow_sphere_case

HERE = pathlib.Path(__file__).resolve().parent
PLATFORM_TOL = 0.05
FLUX_TOL = 0.08
SIGMA = 0.072          # water/air at 25 C, N/m
RHO0 = 1000.0          # kg/m^3
DX = 1.0               # lattice spacing; every length below is in dx

DROPLET_LEVELS = [4.0, 6.0, 8.0, 10.0]        # R/dx
SHELL_LEVELS = [(4.0, 10.0), (6.0, 14.0), (8.0, 18.0)]   # (R_in/dx, R_out/dx)


def _run(radius_in, radius_out, curvature, **kw):
    return hollow_sphere_case(radius_in=radius_in * DX, radius_out=radius_out * DX,
                              dx=DX, sigma=SIGMA, rho0=RHO0, curvature=curvature,
                              **kw)


def _pct(row, key):
    return abs(row[key]) / 100.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="two levels, no controls")
    ap.add_argument("--out", default=str(HERE / "report.json"))
    args = ap.parse_args(argv)

    droplet_levels = DROPLET_LEVELS[:2] if args.quick else DROPLET_LEVELS
    shell_levels = SHELL_LEVELS[:1] if args.quick else SHELL_LEVELS

    rows = []
    for r in droplet_levels:
        for cur in ("analytic", "mls"):
            row = _run(0.0, r, cur)
            row["geometry"] = "drop"
            rows.append(row)
            print(f"[drop  R/dx={r:4.1f} {cur:8s}] dp={row['dp_measured_Pa']:.6g} "
                  f"theory={row['dp_theory_Pa']:.6g} err={row['dp_error_pct']:+.2f}% "
                  f"kappa_out={row['kappa_outer_error_pct']:+.2f}%")
    for r_in, r_out in shell_levels:
        for cur in ("analytic", "mls"):
            row = _run(r_in, r_out, cur)
            row["geometry"] = "shell"
            rows.append(row)
            print(f"[shell {r_in:4.1f}/{r_out:4.1f} {cur:8s}] dp={row['dp_measured_Pa']:.6g} "
                  f"theory={row['dp_theory_Pa']:.6g} err={row['dp_error_pct']:+.2f}% "
                  f"flux_in={row['dp_flux_inner_over_theory']:.4f} "
                  f"flux_out={row['dp_flux_outer_over_theory']:.4f}")

    control = None
    if not args.quick:
        control = hollow_sphere_case(radius_in=6.0 * DX, radius_out=14.0 * DX, dx=DX,
                                     sigma=0.0, rho0=RHO0, curvature="analytic")
        control["geometry"] = "shell"
        print(f"[negative control sigma=0] dp={control['dp_measured_Pa']:.6g} Pa")

    def pick(geometry, curvature):
        return [r for r in rows if r["geometry"] == geometry and r["curvature"] == curvature]

    drop_a = pick("drop", "analytic")
    drop_m = pick("drop", "mls")
    shell_a = pick("shell", "analytic")
    shell_m = pick("shell", "mls")
    finest_a = max(shell_a, key=lambda r: r["R_out_over_dx"])
    finest_m = max(shell_m, key=lambda r: r["R_out_over_dx"])
    finest = max(shell_a, key=lambda r: r["R_out_over_dx"])

    gates = {
        "laplace_analytic_finest": {
            "measured": _pct(finest_a, "dp_error_pct"), "limit": PLATFORM_TOL,
            "detail": f"shell {finest_a['radius_in_m']:.0f}/{finest_a['radius_out_m']:.0f} dx",
        },
        "laplace_analytic_all_levels": {
            "measured": max(_pct(r, "dp_error_pct") for r in drop_a + shell_a),
            "limit": PLATFORM_TOL,
            "detail": "drops R/dx in " + ", ".join(f"{r['R_out_over_dx']:.0f}" for r in drop_a)
                      + "; shells " + ", ".join(f"{r['radius_in_m']:.0f}/{r['radius_out_m']:.0f}"
                                                for r in shell_a),
        },
        "laplace_mls_finest": {
            "measured": _pct(finest_m, "dp_error_pct"), "limit": PLATFORM_TOL,
            "detail": "self-consistent div-n curvature, shipped CSF route",
        },
        "flux_both_faces": {
            "measured": max(abs(finest["dp_flux_inner_over_theory"] - 1.0),
                            abs(finest["dp_flux_outer_over_theory"] - 1.0)),
            "limit": FLUX_TOL,
            "detail": f"inner {finest['dp_flux_inner_over_theory']:.4f} x 2 sigma/R_in, "
                      f"outer {finest['dp_flux_outer_over_theory']:.4f} x 2 sigma/R_out",
        },
        "kappa_two_radii": {
            "measured": max(_pct(finest, "kappa_inner_error_pct"),
                            _pct(finest, "kappa_outer_error_pct")),
            "limit": PLATFORM_TOL,
            "detail": "analytic route; mls numbers reported per row",
        },
        "delta_normalisation": {
            "measured": abs(finest["delta_s_integral_dn"] - 1.0), "limit": PLATFORM_TOL,
            "detail": f"int delta_s dn = {finest['delta_s_integral_dn']:.4f}",
        },
        "negative_control_sigma0": {
            "measured": abs(control["dp_measured_Pa"]) if control else None,
            "limit": 1e-10,
            "detail": "sigma = 0 shell must carry no rejection pressure",
        },
    }
    for name, gate in gates.items():
        if gate["measured"] is None:
            gate["result"] = "not_run"
        else:
            gate["result"] = "pass" if gate["measured"] <= gate["limit"] else "fail"

    report = {
        "case": "CASE-HOLLOW-SPHERE-3D",
        "reference": "Young-Laplace on a sphere: dp = 2 sigma (1/R_in + 1/R_out); "
                     "dp = 2 sigma / R for a drop",
        "sources": ["Brackbill, Kothe & Zemach (1992), J. Comput. Phys. 100:335",
                    "Cummins & Rudman (1999), J. Comput. Phys. 152:584",
                    "Batchelor (1967), An Introduction to Fluid Dynamics, sec. 3.9"],
        "sigma_N_per_m": SIGMA, "rho0_kg_per_m3": RHO0, "dx_m": DX,
        "platform_tolerance": PLATFORM_TOL, "flux_tolerance": FLUX_TOL,
        "torch": torch.__version__,
        "gates": gates,
        "levels": rows,
    }
    if control:
        report["negative_control"] = control
    path = pathlib.Path(args.out)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {path}")
    for name, gate in gates.items():
        print(f"  {name:28s} {gate['result']:7s} measured="
              f"{gate['measured'] if gate['measured'] is None else round(gate['measured'], 5)} "
              f"limit={gate['limit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())