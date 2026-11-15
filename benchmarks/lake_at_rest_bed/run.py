"""Lake at rest over a wavy bed: the flat-bottom certification, transported.

Case-pool anchor ``CASE-SPH-LAKE-AT-REST`` is certified on a **flat** bottom
(:mod:`tensorsph.lake_at_rest`).  This runner asks the question the
certification does not answer: *is that well-balancedness a property of the
operator, or of the geometry?*

    PYTHONPATH=src python3 benchmarks/lake_at_rest_bed/run.py \\
        --out artifacts/lake_at_rest_bed/sweep.json

What is measured, and why it cannot be tuned
--------------------------------------------
Three mass fields are driven through the *shipped* operator
(:func:`tensorsph.droplet_impact.rhs`) on a periodic channel with a cosine
ripple bed of amplitude ``A`` (max slope ``A pi / W``):

``uniform_lattice_mass``
    negative control -- ``rho0 dx^2`` everywhere; free fall in still water,
    ``max|a_y|/g = 1`` by construction.
``analytic_hydrostatic_profile``
    **the fitting-free measurement.**  The mass field is solved once so that the
    summation density equals the exact EOS hydrostatic profile; nothing is fitted
    to the equilibrium.  Its residual is the spurious acceleration a still lake
    actually suffers on a curved bed, and its 0.1 s forward run is the spurious
    current that follows from it.
``row`` / ``particle``
    the flat-bottom reconstruction (:func:`tensorsph.lake_at_rest.reconstruct`)
    with two unknown maps.  ``row`` is the translation-invariant mass field that
    certified the flat periodic tank; ``particle`` leaves every particle free and
    is the strongest thing that map can buy.  Both are *fits*: reaching roundoff
    here is not well-balancedness, it is a price, and the price (mass modulation,
    horizontal residual) is reported next to the residual.

Gate table (thresholds declared in ``tensorsph.lake_at_rest_bed.GATES``):

============================  ============================================  =============
gate                          quantity                                      threshold
============================  ============================================  =============
``B1`` flat identity           ``row`` tie, ``A = 0``: ``max|a|/g``           ``< 1e-9``
``B2`` transfer                ``row`` tie, ``A = 0.01``: ``max|a|/g``        ``> 1e-3``
``B3`` price of the fit        ``a_x/g`` after ``particle`` fit / prior      ``> 5``
``B4`` operator is live        uniform mass control ``max|a_y|/g``           ``> 0.5``
``B5`` the fit is a fit        ``particle`` fit mass modulation, ``A=0.005``  ``> 0.05``
============================  ============================================  =============

``B2`` and ``B3`` are gates that *fail* the anchor's tolerance on purpose: they
are the measurement, not a defect of the runner.  ``B1`` and ``B4`` are what keep
them honest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from tensorsph.lake_at_rest_bed import (BedConfig, evaluate_gates, run_case,
                                        sweep)

AMPLITUDES = (0.0, 0.0025, 0.005, 0.01, 0.02)
RESOLUTIONS = (0.04, 0.02)


def build_cases(dx_list, amplitudes, t_end, max_iter):
    return [BedConfig(dx=dx, bed_amplitude=amp, t_end=t_end,
                      sample_every=10, solver_max_iter=max_iter,
                      solver_tol=1.0e-11)
            for dx in dx_list for amp in amplitudes]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dx", type=float, nargs="+", default=list(RESOLUTIONS))
    parser.add_argument("--amplitude", type=float, nargs="+",
                        default=list(AMPLITUDES))
    parser.add_argument("--t-end", type=float, default=0.10)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--ties", nargs="+", default=["row", "particle"])
    parser.add_argument("--fields-dx", type=float, default=0.04,
                        help="resolution whose per-particle fields are stored")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    torch.set_default_dtype(torch.float64)
    payload = {"case": "CASE-SPH-LAKE-AT-REST/bed", "runs": []}
    fields = {}
    for dx in args.dx:
        for amp in args.amplitude:
            cfg = BedConfig(dx=dx, bed_amplitude=amp, t_end=args.t_end,
                            sample_every=10, solver_max_iter=args.max_iter)
            print(f"[bed] dx={dx:.3f} A={amp:.4f} slope={cfg.bed_amplitude and 0.5*cfg.bed_amplitude*cfg.wave_number():.4f}",
                  flush=True)
            result = run_case(cfg, ties=tuple(args.ties), t_end=args.t_end,
                              verbose=args.verbose)
            if abs(dx - args.fields_dx) < 1e-12 and len(fields) < 3 and \
                    amp in (0.0, 0.01):
                # keep the field dumps small: two amplitudes at one resolution
                fields[f"dx{dx:.3f}_A{amp:.4f}"] = result.pop("fields")
            else:
                result.pop("fields", None)
            payload["runs"].append({"dx": dx, "bed_amplitude": amp,
                                    "result": result})

    grades = evaluate_gates(payload)
    payload["gates"] = grades
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float))
    if fields:
        torch.save(fields, out.with_suffix(".fields.pt"))

    print()
    header = f"{'dx':>6} {'A':>7} {'slope':>6} {'prior a_y/g':>11} " \
             f"{'prior a_x/g':>11} {'row a_y/g':>10} {'fit a_y/g':>10} " \
             f"{'fit a_x/g':>10} {'fit dM/M':>9} {'v/c0 0.1s':>9}"
    print(header)
    print("-" * len(header))
    for run in payload["runs"]:
        res, geo = run["result"], run["result"]["geometry"]
        prior = res["controls"]["analytic_hydrostatic_profile"]
        row = res["solutions"].get("row", {})
        fit = res["solutions"].get("particle", {})
        print(f"{run['dx']:6.3f} {run['bed_amplitude']:7.4f} "
              f"{geo['max_bed_slope']:6.3f} {prior['a_y_over_g']:11.3e} "
              f"{prior['a_x_over_g']:11.3e} {row.get('a_y_over_g', 0.0):10.3e} "
              f"{fit.get('a_y_over_g', 0.0):10.3e} "
              f"{fit.get('a_x_over_g', 0.0):10.3e} "
              f"{fit.get('mass_modulation_max_abs', 0.0):9.3f} "
              f"{prior['rest']['max_v_over_c0']:9.3e}")
    print()
    for gate in grades["gates"]:
        print(f"{gate['gate']:>3} {gate['name']:<26} "
              f"measured {gate['measured']:.6e} {gate['comparison']} "
              f"{gate['threshold']:.1e}  -> {gate['verdict']}")
    print(f"\nALL GATES PASSED: {grades['passed']}")
    return 0 if grades["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())