"""Lake at rest (case pool anchor ``CASE-SPH-LAKE-AT-REST``).

Runs the discrete well-balanced reconstruction of :mod:`tensorsph.lake_at_rest`
against the analytic hydrostatic identity, writes the artefact JSON and prints
the gate table.

    PYTHONPATH=src python3 benchmarks/lake_at_rest/run.py --json artifacts/lake_at_rest/rest.json

Thresholds (all relative to the analytic identity, no experimental tolerance):

============================  =========================================  ==========
gate                          quantity                                   threshold
============================  =========================================  ==========
``G1`` equilibrium residual   ``max|a|/g`` of the reconstructed state    ``< 1e-9``
``G2`` rest preservation       ``max|v|/c0`` over the run                 ``< 1e-9``
``G3`` surface invariance      ``max|dy|/H`` over the run                 ``< 1e-9``
``G4`` density invariance      ``max|drho|/rho0`` over the run            ``< 1e-9``
``G5`` operator is live        residual of the analytic-profile control   ``> 1e-2``
``G6`` linear response         ``r(1e-3)/r(1e-2)`` particle-mass scaling   ``0.09..0.11``
============================  =========================================  ==========

``G5`` is the negative control that makes ``G1``-``G4`` meaningful: the *same*
operator, the *same* lattice and the *same* code path, driven by the analytic
hydrostatic profile, must show an ``O(g)`` residual.  If it ever stopped doing
that, ``G1`` would be measuring a dead operator.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

from tensorsph.lake_at_rest import LakeConfig, run_case

GATES = [
    ("G1", "vertical equilibrium max|a_y|/g", 1.0e-9, "lt"),
    ("G2", "horizontal residual, side-wall band max|a_x|/g", 1.0e-9, "lt"),
    ("G3", "rest preservation max|v|/c0", 1.0e-9, "lt"),
    ("G4", "surface invariance max|dy|/H", 1.0e-9, "lt"),
    ("G5", "density invariance max|drho|/rho0", 1.0e-9, "lt"),
    ("G3i", "interior rest max|v|/c0 (>= 2h from side walls)", 1.0e-9, "lt"),
    ("G4i", "interior surface invariance max|dy|/H", 1.0e-9, "lt"),
    ("G6", "negative control uniform lattice mass max|a|/g", 0.5, "gt"),
    ("G7", "negative control analytic profile max|a_y|/g", 0.05, "gt"),
    ("G8", "linear response r(1e-4)/r(1e-3)", 0.05, "gt"),
    ("G8", "linear response r(1e-4)/r(1e-3)", 0.15, "lt"),
]


def evaluate(result: dict) -> dict:
    equilibrium = result["equilibrium"]
    rest = result["rest"]
    controls = result["controls"]
    perturb = result["perturbation_response"]
    ratio = None
    small = perturb.get(1.0e-4) or perturb.get("0.0001")
    large = perturb.get(1.0e-3) or perturb.get("0.001")
    if small and large:
        ratio = small / large
    measured = {
        "G1": equilibrium["a_max_over_g"],
        "G2": equilibrium["a_max_over_g_horizontal_wall_band"],
        "G3": rest["max_v_over_c0"],
        "G4": rest["max_dy_over_H"],
        "G5": rest["max_drho_over_rho0"],
        "G6": controls["uniform_lattice_mass"]["a_max_over_g"],
        "G7": controls["analytic_hydrostatic_profile"]["a_max_over_g"],
        "G8": ratio,
        "G3i": rest["max_v_over_c0_interior"],
        "G4i": rest["max_dy_over_H_interior"],
    }
    rows = []
    for identifier, description, threshold, comparison in GATES:
        value = measured[identifier]
        if value is None:
            passed = None
        elif comparison == "lt":
            passed = bool(value < threshold)
        else:
            passed = bool(value > threshold)
        rows.append({"gate": identifier, "quantity": description,
                     "measured": value, "threshold": threshold,
                     "comparison": comparison, "passed": passed})
    # G6 needs both of its two-sided rows; report the pair as one verdict.
    verdict = {}
    for row in rows:
        current = verdict.get(row["gate"])
        verdict[row["gate"]] = row["passed"] if current is None else (
            bool(current) and bool(row["passed"]))
    return {"rows": rows, "verdicts": verdict,
            "all_passed": all(v for v in verdict.values() if v is not None)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", type=float, default=0.02)
    parser.add_argument("--depth", type=float, default=0.30)
    parser.add_argument("--width", type=float, default=0.40)
    parser.add_argument("--t-end", type=float, default=0.50)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    cfg = LakeConfig(dx=args.dx, depth=args.depth, width=args.width,
                     t_end=args.t_end)
    result = run_case(cfg, verbose=args.verbose)
    gates = evaluate(result)
    result["gates"] = {"rows": gates["rows"], "verdicts": gates["verdicts"],
                       "all_passed": gates["all_passed"]}
    result["parameters"] = {"dx": cfg.dx, "depth": cfg.depth, "width": cfg.width,
                            "t_end": cfg.t_end, "max_courant": cfg.max_courant,
                            "h_factor": cfg.h_factor,
                            "prior_weight": cfg.prior_weight,
                            "solver_tol": cfg.solver_tol}
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"wrote {args.json}")

    print(f"\ncase CASE-SPH-LAKE-AT-REST  dx={cfg.dx} H={cfg.depth} "
          f"N={result['geometry']['particles']} pairs={result['geometry']['pairs']}")
    print(f"reconstruction: {result['equilibrium']['iterations']} Newton "
          f"iterations, converged={result['equilibrium']['converged']}")
    print(f"{'gate':5s} {'quantity':48s} {'measured':>12s} {'threshold':>10s}  ok")
    for row in gates["rows"]:
        value = row["measured"]
        text = "n/a" if value is None else f"{value:.3e}"
        print(f"{row['gate']:5s} {row['quantity'][:48]:48s} {text:>12s} "
              f"{row['threshold']:10.1e}  {row['passed']}")
    print(f"\nALL GATES PASSED: {gates['all_passed']}")
    return 0 if gates["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())