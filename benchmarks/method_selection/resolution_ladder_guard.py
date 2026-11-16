"""Resolution-ladder guard for the ``resolution_attributable`` adjudication.

Why this module exists
----------------------
m22787 adjudicated every registry record above the owner 15 % standard and
handed out the class ``resolution_attributable`` when *one* finer rung of the
same benchmark brought the same quantity inside the standard.  One pair is not
convergence.  For the transport-velocity record the repository already owns a
four-level ladder (``dx = 0.04 / 0.03 / 0.02 / 0.01 m``,
``docs/xval_sphinx_tvc/data/ladder_dense.json``) whose error is **not monotone
in the resolution and whose finest rung is back above 15 %**, so the label as
written -- "the over-tolerance value is a discretisation symptom that the
registry's missing resolution column hides" -- is not supported by the data the
repository already has.  A reader is entitled to infer "refine and it goes
away"; for this record refine and it comes *back*.

This module replaces the label with a measurement, in three outcomes:

``resolution_attributable``
    The committed ladder has >= 3 rungs, the error does not increase anywhere
    under refinement, and the finest rung is inside the owner standard.  Only
    here may the over-tolerance value be read as a discretisation symptom.
``resolution_window``
    The error passes the standard on a **bounded** window of the rung
    parameter, and at least one rung outside that window (coarser *or finer*)
    fails, or the error rises under refinement.  The pass is a property of the
    calibrated window (grid spacing *and* the coefficient calibrated for it),
    not of convergence: the claim must be scoped, not extrapolated.
``resolution_pair_only``
    The committed artefacts have fewer than three rungs: the pair the
    adjudication used is consistent but monotonicity cannot be tested from the
    repository's own data (an honest downgrade, not a pass).

Everything is read from committed artefacts, stdlib only, no numbers typed in.

Model case, current state (recomputed from the artefacts, see
``docs/reports/method_selection_resolution_ladder_guard_m22957.json``):
``transport_velocity_correction`` -- 17.06 / 10.65 / 12.48 / 16.39 % at
``dx = 0.04 / 0.03 / 0.02 / 0.01 m``: a window, not convergence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parents[2]

#: Owner precision standard [%]; the same threshold the adjudication and the
#: router use.
OWNER_TOLERANCE_PCT = 15.0

#: Classes this guard may hand out, in report order.
GUARD_CLASSES = ("resolution_attributable", "resolution_window",
                 "resolution_pair_only")

#: Variant whose error the adjudication grades (the round-1 calibration).
TVC_VARIANT = "tvc_a010_all"

#: Bars are a *measurement*; a ladder shorter than this cannot show a trend.
MIN_RUNGS = 3


# --------------------------------------------------------------------------- #
# Artefact accessors (each returns {rung value: error [%]})
# --------------------------------------------------------------------------- #
def _load(path: Path) -> dict:
    return json.loads((ROOT / path).read_text())


def tvc_front_ladder(paths: list[str]) -> dict[float, float]:
    """Front-position mean relative error [%] at every committed ``dx`` rung."""
    out: dict[float, float] = {}
    for path in paths:
        if not (ROOT / path).exists():
            continue
        for record in _load(path)["records"]:
            if record["variant"] != TVC_VARIANT:
                continue
            rel = [abs(float(row["rel_error"])) for row in record["score"]["rows"]
                   if row["rel_error"] is not None]
            out[float(record["dx"])] = sum(rel) / len(rel) * 100.0
    return out


def couette_shear_ladder(paths: list[str]) -> dict[float, float]:
    """Bottom-wall shear error [%] at every committed ``H/dx`` rung."""
    out: dict[float, float] = {}
    for path in paths:
        if not (ROOT / path).exists():
            continue
        for row in _load(path)["ladder"]:
            metrics = row["metrics"]
            measured = float(metrics["wall_shear_measured"])
            analytic = float(metrics["wall_shear_analytic"])
            out[float(row["resolution_H_over_dx"])] = abs(
                measured - analytic) / abs(analytic) * 100.0
    return out


#: One entry per method the adjudication can call ``resolution_attributable``.
#: ``refinement`` says which way the rung value moves when the grid is refined.
LADDERS: tuple[dict, ...] = (
    {
        "method": "transport_velocity_correction",
        "parameter": "dx [m]",
        "refinement": "decreasing",
        "quantity": "front-position mean relative error Z/a vs Martin & Moyce "
                    "(1952) digitised points [%]",
        "details": "the pass at dx = 0.02/0.03 m is a calibrated window: the "
                   "coefficient alpha = 0.10 is not resolution independent "
                   "(docs/xval_sphinx_tvc/RESOLUTION_LADDER.md sections 2-3)",
        "accessor": tvc_front_ladder,
        "paths": [
            "docs/xval_sphinx_tvc/data/ladder_dense.json",
            # fifth rung, run after the guard was written; picked up when present
            "docs/xval_sphinx_tvc/data/ladder_dx0005.json",
        ],
    },
    {
        "method": "wcsph_wall_no_slip",
        "parameter": "H/dx",
        "refinement": "increasing",
        "quantity": "bottom-wall shear error vs the analytic Couette value [%]",
        "details": "the only committed multi-rung artefact is the design-time "
                   "run (H/dx = 8, 16); the over-tolerance record comes from a "
                   "different artefact (benchmarks/couette/out/results.json) "
                   "that has a single rung at H/dx = 8, and the two artefacts "
                   "stop at different terminal times, so their shared rung is "
                   "deliberately not merged. An H/dx = 32 rung is the missing "
                   "evidence that would make this a trend",
        "accessor": couette_shear_ladder,
        "paths": [
            "benchmarks/couette/out_design/results.json",
        ],
    },
)

#: ``refinement`` -> key that sorts rungs coarse -> fine.
_SORT_KEY: Mapping[str, Callable[[float], float]] = {
    "decreasing": lambda value: -value,   # dx: large dx is coarse
    "increasing": lambda value: value,    # H/dx: small H/dx is coarse
}


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classify(rungs: Mapping[float, float], refinement: str,
             tolerance: float = OWNER_TOLERANCE_PCT) -> dict:
    """Grade a measured ladder: attributable / window / pair-only.

    ``rungs`` maps the resolution parameter to the error [%] of the same
    quantity.  Nothing is extrapolated: only the measured rungs are graded.
    """
    if refinement not in _SORT_KEY:
        raise ValueError(f"unknown refinement direction {refinement!r}")
    ordered = sorted(rungs.items(), key=lambda kv: _SORT_KEY[refinement](kv[0]))
    values = [value for _, value in ordered]
    reversals = [
        {"coarse_rung": ordered[i][0], "fine_rung": ordered[i + 1][0],
         "coarse_pct": values[i], "fine_pct": values[i + 1],
         "increase_pp": values[i + 1] - values[i]}
        for i in range(len(ordered) - 1) if values[i + 1] > values[i]
    ]
    passing = [rung for rung, value in ordered if value <= tolerance]
    finest_rung, finest_pct = ordered[-1]
    monotone = not reversals
    finest_within = finest_pct <= tolerance
    if len(ordered) < MIN_RUNGS:
        klass = "resolution_pair_only"
        reason = (f"only {len(ordered)} rung(s) committed: a pair is consistent "
                  f"with convergence but cannot demonstrate it")
    elif monotone and finest_within:
        klass = "resolution_attributable"
        reason = (f"{len(ordered)} rungs, error non-increasing under refinement, "
                  f"finest rung {finest_rung:g} at {finest_pct:.2f}% <= "
                  f"{tolerance:g}%")
    else:
        klass = "resolution_window"
        failures = []
        if reversals:
            failures.append(
                "error rises under refinement ("
                + ", ".join(f"{r['coarse_rung']:g}->{r['fine_rung']:g}: "
                            f"+{r['increase_pp']:.2f} pp"
                            for r in reversals) + ")")
        if not finest_within:
            failures.append(f"finest rung {finest_rung:g} measures "
                            f"{finest_pct:.2f}% > {tolerance:g}%")
        reason = (f"{len(ordered)} rungs; the pass covers only the measured "
                  f"window, not the refinement sequence: " + "; ".join(failures))
    return {
        "class": klass,
        "reason": reason,
        "tolerance_pct": tolerance,
        "n_rungs": len(ordered),
        "monotone": monotone,
        "reversals": reversals,
        "finest_rung": finest_rung,
        "finest_pct": finest_pct,
        "finest_within_standard": finest_within,
        "passing_rungs": passing,
        "window": ({"min": min(passing), "max": max(passing)} if passing else None),
        "rungs": {f"{rung:g}": value for rung, value in ordered},
    }


def ladder_for(method: str) -> dict | None:
    """Measured ladder of ``method`` from the committed artefacts, or ``None``."""
    spec = next((item for item in LADDERS if item["method"] == method), None)
    if spec is None:
        return None
    rungs = spec["accessor"](list(spec["paths"]))
    if not rungs:
        return None
    verdict = classify(rungs, spec["refinement"])
    return {
        "method": method,
        "parameter": spec["parameter"],
        "refinement": spec["refinement"],
        "quantity": spec["quantity"],
        "details": spec["details"],
        "sources": [p for p in spec["paths"] if (ROOT / p).exists()],
        **verdict,
    }


def classify_method(method: str) -> dict | None:
    """Guard verdict for one method, or ``None`` when no ladder is declared."""
    return ladder_for(method)


# --------------------------------------------------------------------------- #
# CLI (inspection only; the adjudication imports the functions above)
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable dump")
    args = parser.parse_args(argv)
    summary = {spec["method"]: ladder_for(spec["method"]) for spec in LADDERS}
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    for method, verdict in summary.items():
        if verdict is None:
            print(f"{method}: no committed ladder")
            continue
        print(f"{method}: {verdict['class']} ({verdict['n_rungs']} rungs)")
        for rung, value in verdict["rungs"].items():
            print(f"    {verdict['parameter'].split()[0]} = {rung:>7} -> {value:6.2f} %")
        print(f"    {verdict['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())