"""3-D error budget of the projection-SPH Young-Laplace route (XVAL round 5).

The 2-D anchor ``docs/xval_projection_laplace_robustness.md`` measured the
Laplace jump at *one* set-up and then varied, one knob at a time, everything the
physics cannot see (interface position, lattice orientation, gauge pad, readout
band, kernel support, resolution).  That audit stops at 2-D, and it cannot be
transplanted: a circle has one non-zero principal curvature, so the 2-D
instrument never exercises the ``1/R1 + 1/R2`` structure, and the 2-D triangular
lattice symmetry (60 deg period) has no cubic counterpart.

This script runs the same question in 3-D on the spherical interface
``Delta p = 2 sigma / R`` (drop) and ``2 sigma (1/R_in + 1/R_out)`` (shell),
using the shipped operators of :mod:`tensorsph.threed.hollow_sphere`.  Every
number written here is produced by this program; nothing is typed by hand.

Knobs (one at a time away from the operating point
``R/dx = 8, pad = 4 dx, h/dx = 1.3, cubic lattice, centre at origin``):

    translation  interface offset inside the lattice cell (7 offsets)
    lattice      cubic vs fcc (two interpenetrating copies)
    pad          Dirichlet-gauge distance 2.5 .. 8 dx
    h_over_dx    kernel support 1.2 .. 1.5
    curvature    imposed ("analytic") vs self-consistent (div n)
    resolution   R/dx = 4 .. 12, both curvature routes
    shell        hollow sphere at three thicknesses (two interfaces)
    control      sigma = 0 must carry no pressure

Usage
-----
    PYTHONPATH=src python3 benchmarks/hollow_sphere_3d/audit_knobs.py
    PYTHONPATH=src python3 benchmarks/hollow_sphere_3d/audit_knobs.py --quick
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorsph.threed.hollow_sphere import (
    hollow_sphere_case,
    shell_laplace_analytic,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_knobs"

SIGMA = 0.072
RHO0 = 1000.0
BASE = dict(radius_in=0.0, radius_out=8.0, dx=1.0, pad=4.0, h_factor=1.3,
            lattice="cubic", curvature="analytic")

# Gates, declared before the numbers were inspected (owner acceptance 15 % on
# the Young-Laplace jump; 3 % is the claim line used by the 2-D rounds).
GATES = {
    "single_config_abs_err_pct": 15.0,
    "translation_spread_pct": 5.0,
    "lattice_spread_pct": 5.0,
    "pad_spread_pct_at_3dx_and_above": 5.0,
    "h_over_dx_spread_pct": 5.0,
    "resolution_abs_err_pct_at_12": 3.0,
    "fcc_abs_err_pct": 3.0,
    "shell_abs_err_pct": 5.0,
    "control_abs_pa": 1.0e-6,
}


def _run(label, **over):
    kw = dict(BASE)
    kw.update(over)
    t0 = time.time()
    rep = hollow_sphere_case(sigma=over.pop("sigma", SIGMA), rho0=RHO0, **kw)
    return {
        "label": label,
        "radius_in_over_dx": rep["radius_in_m"] / rep["dx_m"],
        "radius_out_over_dx": rep["R_out_over_dx"],
        "R_out_over_h": rep["R_out_over_h"],
        "pad_over_dx": kw["pad"] / kw["dx"],
        "h_over_dx": kw["h_factor"],
        "lattice": rep["lattice"],
        "curvature": rep["curvature"],
        "center_over_dx": (None if rep["center_m"] is None
                           else [c / rep["dx_m"] for c in rep["center_m"]]),
        "n_particles": rep["n_particles"],
        "cg_iterations": rep["cg_iterations"],
        "dp_theory_Pa": rep["dp_theory_Pa"],
        "dp_measured_Pa": rep["dp_measured_Pa"],
        "dp_error_pct": rep["dp_error_pct"],
        "dp_flux_Pa": rep["dp_flux_Pa"],
        "dp_flux_error_pct": rep["dp_flux_error_pct"],
        "kappa_outer_error_pct": rep["kappa_outer_error_pct"],
        "delta_s_integral_dn": rep["delta_s_integral_dn"],
        "residual_acc_over_mean": rep["residual_acc_over_mean_csf_acc"],
        "wall_s": time.time() - t0,
    }


def _spread(rows, key="dp_error_pct"):
    vals = [r[key] for r in rows]
    return max(vals) - min(vals)


def axis_translation(quick=False):
    offsets = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
               [0.0, 0.0, 0.5], [0.5, 0.5, 0.0], [1.0 / 3.0] * 3]
    if not quick:
        offsets.append([0.5, 0.5, 0.5])
    rows = [_run(f"offset={o}", center=o) for o in offsets]
    return {
        "axis": "translation",
        "question": "does the jump depend on where the interface falls in a cell?",
        "rows": rows,
        "spread_pct": _spread(rows),
        "kappa_spread_pct": _spread(rows, "kappa_outer_error_pct"),
    }


def axis_lattice():
    rows = [_run(f"lattice={lat}", lattice=lat) for lat in ("cubic", "fcc")]
    return {
        "axis": "lattice",
        "question": "cubic vs the denser fcc packing (defect D1 regression)",
        "rows": rows,
        "spread_pct": _spread(rows),
    }


def axis_pad(quick=False):
    pads = [3.0, 4.0, 6.0] if quick else [2.5, 3.0, 4.0, 6.0, 8.0]
    rows = [_run(f"pad={p}dx", pad=p) for p in pads]
    declared = [r for r in rows if r["pad_over_dx"] >= 3.0]
    return {
        "axis": "pad",
        "question": "Dirichlet-gauge distance (a numerical, not physical, knob)",
        "rows": rows,
        "spread_pct": max(r["dp_error_pct"] for r in rows)
        - min(r["dp_error_pct"] for r in rows),
        "spread_pct_at_3dx_and_above": max(r["dp_error_pct"] for r in declared)
        - min(r["dp_error_pct"] for r in declared),
    }


def axis_h(quick=False):
    hs = [1.2, 1.3, 1.4] if quick else [1.2, 1.3, 1.4, 1.5]
    rows = [_run(f"h={hf}dx", h_factor=hf) for hf in hs]
    return {
        "axis": "h_over_dx",
        "question": "kernel support relative to the lattice spacing",
        "rows": rows,
        "spread_pct": _spread(rows),
    }


def axis_curvature():
    rows = [_run(f"curvature={c}", curvature=c) for c in ("analytic", "mls")]
    return {
        "axis": "curvature",
        "question": "imposed curvature vs the shipped self-consistent div n route",
        "rows": rows,
        "spread_pct": _spread(rows),
    }


def axis_resolution(quick=False):
    levels = [6.0, 8.0] if quick else [4.0, 6.0, 8.0, 10.0, 12.0]
    rows = []
    for c in ("analytic", "mls"):
        for rdx in levels:
            rows.append(_run(f"R/dx={rdx:g} {c}", radius_out=rdx, curvature=c))
    return {
        "axis": "resolution",
        "question": "does the jump converge, and to what?",
        "rows": rows,
        "spread_pct": None,
    }


def axis_shell(quick=False):
    shells = [(4.0, 10.0), (6.0, 14.0)] if quick else [(4.0, 10.0), (6.0, 14.0),
                                                       (8.0, 18.0)]
    rows = []
    for r_in, r_out in shells:
        for c in ("analytic", "mls"):
            rows.append(_run(f"shell {r_in:g}/{r_out:g} {c}", radius_in=r_in,
                             radius_out=r_out, curvature=c))
    for r in rows:
        r["dp_theory_Pa"] = shell_laplace_analytic(
            SIGMA, r["radius_in_over_dx"], r["radius_out_over_dx"])
    return {
        "axis": "shell",
        "question": "two curved interfaces: dp = 2 sigma (1/R_in + 1/R_out)",
        "rows": rows,
        "spread_pct": None,
    }


def control_sigma_zero():
    rep = hollow_sphere_case(sigma=0.0, rho0=RHO0, radius_in=0.0, radius_out=8.0,
                             dx=1.0, pad=4.0, h_factor=1.3, curvature="analytic")
    return {
        "axis": "control",
        "question": "sigma = 0 must carry no pressure jump",
        "dp_measured_Pa": rep["dp_measured_Pa"],
        "dp_theory_Pa": rep["dp_theory_Pa"],
        "n_particles": rep["n_particles"],
    }


def summarise(res):
    axes = res["axes"]
    worst = max(r["dp_error_pct"] for a in axes.values()
                if isinstance(a, dict) and a.get("rows")
                for r in a["rows"])
    best = min(r["dp_error_pct"] for a in axes.values()
               if isinstance(a, dict) and a.get("rows")
               for r in a["rows"])
    contrib = []
    for key, gate in (("translation", "translation_spread_pct"),
                      ("lattice", "lattice_spread_pct"),
                      ("pad", "pad_spread_pct_at_3dx_and_above"),
                      ("h_over_dx", "h_over_dx_spread_pct")):
        a = axes[key]
        val = a.get("spread_pct_at_3dx_and_above", a["spread_pct"])
        contrib.append({"knob": key, "spread_pct": val, "gate_pct": GATES[gate],
                        "pass": abs(val) <= GATES[gate]})
    fcc = [r for r in axes["lattice"]["rows"] if r["lattice"] == "fcc"][0]
    res12 = [r for r in axes["resolution"]["rows"]
             if abs(r["radius_out_over_dx"] - 12.0) < 1e-9]
    shell_worst = max(abs(r["dp_error_pct"]) for r in axes["shell"]["rows"])
    checks = {
        "worst_single_config_pct": worst,
        "best_single_config_pct": best,
        "worst_inside_15pct": abs(worst) <= GATES["single_config_abs_err_pct"],
        "fcc_pct": fcc["dp_error_pct"],
        "fcc_inside_3pct": abs(fcc["dp_error_pct"]) <= GATES["fcc_abs_err_pct"],
        "resolution_12_pct": {r["curvature"]: r["dp_error_pct"] for r in res12},
        "resolution_12_inside_3pct": all(
            abs(r["dp_error_pct"]) <= GATES["resolution_abs_err_pct_at_12"]
            for r in res12),
        "shell_worst_pct": shell_worst,
        "shell_inside_5pct": shell_worst <= GATES["shell_abs_err_pct"],
        "control_dp_Pa": axes["control"]["dp_measured_Pa"],
        "control_inside_gate": abs(axes["control"]["dp_measured_Pa"])
        <= GATES["control_abs_pa"],
        "knob_contributions": contrib,
    }
    checks["all_pass"] = bool(
        checks["worst_inside_15pct"] and checks["fcc_inside_3pct"]
        and checks["resolution_12_inside_3pct"] and checks["shell_inside_5pct"]
        and checks["control_inside_gate"] and all(c["pass"] for c in contrib))
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(2)

    res = {
        "case": "XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE round 5 (3-D error budget)",
        "reference": ("2 sigma / R (drop); 2 sigma (1/R_in + 1/R_out) (shell)"),
        "params": {"sigma_N_per_m": SIGMA, "rho0_kg_per_m3": RHO0,
                   "base": {k: (v if not isinstance(v, float) else v)
                            for k, v in BASE.items()}},
        "gates": GATES,
        "torch": torch.__version__,
        "axes": {},
    }
    res["axes"]["translation"] = axis_translation(args.quick)
    res["axes"]["lattice"] = axis_lattice()
    res["axes"]["pad"] = axis_pad(args.quick)
    res["axes"]["h_over_dx"] = axis_h(args.quick)
    res["axes"]["curvature"] = axis_curvature()
    res["axes"]["resolution"] = axis_resolution(args.quick)
    res["axes"]["shell"] = axis_shell(args.quick)
    res["axes"]["control"] = control_sigma_zero()
    res["checks"] = summarise(res)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else (
        RESULTS / ("results_knobs_quick.json" if args.quick else "results_knobs.json"))
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["checks"], indent=1))
    print("written", out)
    return 0 if res["checks"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())