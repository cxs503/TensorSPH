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

# The gate table and the grader live in the library
# (``tensorsph.lake_at_rest.GATES_WALL`` / ``GATES_PERIODIC`` /
# ``evaluate_gates``) so that this runner, ``freeze.py`` and the evidence tests
# grade the *same* thresholds.  ``evaluate`` keeps this module's public surface.
from tensorsph.lake_at_rest import (GATES_PERIODIC, GATES_WALL,  # noqa: E402
                                    LakeConfig, evaluate_gates, run_case)

evaluate = evaluate_gates


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", type=float, default=0.02)
    parser.add_argument("--depth", type=float, default=0.30)
    parser.add_argument("--width", type=float, default=0.40)
    parser.add_argument("--t-end", type=float, default=0.50)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--periodic-x", action="store_true",
                        help="lateral periodicity instead of side walls: the "
                             "configuration in which the anchor tolerance is met")
    args = parser.parse_args(argv)

    cfg = LakeConfig(dx=args.dx, depth=args.depth, width=args.width,
                     t_end=args.t_end, periodic_x=args.periodic_x)
    result = run_case(cfg, verbose=args.verbose)
    gates = evaluate(result)
    result["gates"] = {"rows": gates["rows"], "verdicts": gates["verdicts"],
                       "all_passed": gates["all_passed"]}
    result["parameters"] = {"periodic_x": bool(cfg.periodic_x),
                            "dx": cfg.dx, "depth": cfg.depth, "width": cfg.width,
                            "t_end": cfg.t_end, "max_courant": cfg.max_courant,
                            "h_factor": cfg.h_factor,
                            "prior_weight": cfg.prior_weight,
                            "solver_tol": cfg.solver_tol}
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"wrote {args.json}")

    print(f"\ncase CASE-SPH-LAKE-AT-REST  dx={cfg.dx} H={cfg.depth} "
          f"periodic_x={cfg.periodic_x} "
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