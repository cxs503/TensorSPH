"""Transient-reading guard: was the over-tolerance value taken before the flow settled?

Why this module exists
----------------------
``adjudicate_over_tolerance.py`` grades an over-tolerance registry record on the
*resolution* axis: refine the grid and see whether the error comes back inside
the owner 15 % standard.  For ``x_couette_wall_shear`` (32.6 %, method
``wcsph_wall_no_slip``) that axis was never the only candidate: the record comes
from ``benchmarks/couette/out/results.json``, which ``run.py --quick`` produced
at ``t_end = 0.3 H^2/nu`` -- a quarter of the ``DESIGN.md`` protocol
(``t_end = 1.2 H^2/nu``).  A startup reading and a discretisation symptom are
different claims with different consequences: the first says *the artefact is
not a steady measurement*, the second says *refine and it goes away*.

This module tests the transient claim with the same instruments the record used
(one estimator, one lattice, one solver, one gate set) and **only** along the
terminal-time axis, holding ``H/dx`` at the record's own grid.  Two committed
artefacts supply the levels, and a shared level must agree across them.

Classes (predeclared):

``transient_reading``
    The record's own value is over the standard, every longer terminal time at
    the same grid measures *less* error, and at least one of those levels is
    inside the standard.  The over-tolerance value is then a pre-steady reading;
    it is not evidence about the grid.
``reading_window_time``
    Longer terminal times do not bring the value inside the standard: the
    transient claim is refuted and the resolution/accuracy label stands.
``insufficient_time_ladder``
    Fewer than ``MIN_LEVELS`` distinct terminal times exist at the record's
    grid: nothing is claimed.

Two extra fields keep the honest part visible instead of folding it into the
class:

``passing_levels_steady``
    Whether any *passing* level also cleared the repository's own steadiness
    gate (``steady_drift <= 1 %``, the gate ``viscous_time_convergence.py``
    uses).  A pass at a level that is still drifting is a pass at a
    non-admissible measurement; it refutes the *resolution* reading of the
    record without pretending the reading is a finished steady value.
``matched_protocol_resolution``
    The resolution verdict at the record's *own* protocol (the new
    ``benchmarks/couette/protocol_matrix.py`` matrix: ``H/dx`` in 8/16/32 at
    ``t_end = 0.3 H^2/nu``).  This says whether refining would have rescued the
    record anyway.  It is banked, never used to hide the time axis.

Everything is read from committed artefacts, stdlib only, no numbers typed in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]

#: Owner precision standard [%], the same threshold the adjudication uses.
OWNER_TOLERANCE_PCT = 15.0

#: Terminal-time levels needed before the time axis can be graded at all.
MIN_LEVELS = 3

#: Classes this guard may hand out, in report order (``None`` = no claim).
GUARD_CLASSES = ("transient_reading", "reading_window_time",
                 "insufficient_time_ladder")

#: Records this guard knows how to grade, declared up front.
RECORDS: tuple[dict, ...] = (
    {
        "extractor": "x_couette_wall_shear",
        "method": "wcsph_wall_no_slip",
        "record_artifact": "benchmarks/couette/out/results.json",
        "quantity": "bottom-wall shear |1 - tau_SPH/tau_exact| [%]",
        "time_sources": [
            "benchmarks/couette/out_protocol_matrix/matrix.json",
            "benchmarks/couette/out_design/viscous_time_convergence.json",
        ],
        "details": "the --quick artefact stops at t_end = 0.3 H^2/nu, a quarter "
                   "of the DESIGN.md protocol; the time axis is swept at the "
                   "record's own H/dx with the same estimator and gates",
    },
)


# --------------------------------------------------------------------------- #
# Artefact readers
# --------------------------------------------------------------------------- #
def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def record_level(record_artifact: str) -> dict:
    """Grid, terminal time and error of the single rung the record banks."""
    doc = _load(record_artifact)
    if len(doc["ladder"]) != 1:
        raise ValueError(
            f"{record_artifact} carries {len(doc['ladder'])} rungs; the record "
            "level must be the artefact's single rung")
    row = doc["ladder"][0]
    metrics = row["metrics"]
    measured = float(metrics["wall_shear_measured"])
    analytic = float(metrics["wall_shear_analytic"])
    viscous_time = float(doc["height"]) ** 2 / float(doc["viscosity"])
    return {
        "resolution_H_over_dx": float(row["resolution_H_over_dx"]),
        "t_end": float(row["t_end"]),
        "viscous_times": float(row["t_end"]) / viscous_time,
        "error_pct": abs(measured - analytic) / abs(analytic) * 100.0,
        "steady_drift": float(row.get("steady_drift", float("nan"))),
        "artifact": record_artifact,
    }


def time_levels(source: str, resolution: float) -> dict[float, dict]:
    """``viscous_times -> level`` at one grid, from either artefact schema."""
    doc = _load(source)
    out: dict[float, dict] = {}
    if "cells" in doc:                       # protocol_matrix.py
        rows = [
            {
                "viscous_times": cell["viscous_times"],
                "resolution_H_over_dx": cell["resolution_H_over_dx"],
                "error_pct": cell["wall_shear_error_pct_bottom"],
                "admissible": bool(cell["admissible"]),
                "steady_drift": cell["steady_drift"],
                "protocol": cell["protocol"],
            }
            for cell in doc["cells"]
        ]
    elif "points" in doc:                    # viscous_time_convergence.py
        rows = [
            {
                "viscous_times": point["viscous_times"],
                "resolution_H_over_dx": point["resolution_H_over_dx"],
                "error_pct": point["wall_shear_error_pct_bottom"],
                "admissible": bool(point["admissible"]),
                "steady_drift": point.get("steady_drift"),
                "protocol": f"{point['viscous_times']:g}vt",
            }
            for point in doc["points"]
        ]
    else:
        raise ValueError(f"{source} carries neither 'cells' nor 'points'")
    for row in rows:
        if float(row["resolution_H_over_dx"]) != resolution:
            continue
        level = float(row["viscous_times"])
        entry = {"error_pct": float(row["error_pct"]),
                 "admissible": row["admissible"],
                 "steady_drift": row["steady_drift"],
                 "protocol": row["protocol"],
                 "source": source}
        if level in out and abs(out[level]["error_pct"] - entry["error_pct"]) > 1e-6:
            raise ValueError(
                f"shared terminal time {level:g} H^2/nu disagrees across "
                f"artefacts at H/dx={resolution:g}: "
                f"{out[level]['source']} {out[level]['error_pct']:.6f}% vs "
                f"{source} {entry['error_pct']:.6f}%")
        out[level] = entry
    return out


def protocol_resolution_ladder(source: str, protocol: str) -> dict[float, float]:
    """``H/dx -> error [%]`` for one terminal-time protocol (protocol matrix)."""
    doc = _load(source)
    out: dict[float, float] = {}
    for cell in doc.get("cells", []):
        if cell["protocol"] != protocol:
            continue
        out[float(cell["resolution_H_over_dx"])] = float(
            cell["wall_shear_error_pct_bottom"])
    return out


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def _grade_resolution(rungs: Mapping[float, float],
                      tolerance: float = OWNER_TOLERANCE_PCT) -> dict:
    """Same shape as ``resolution_ladder_guard.classify`` (kept local, small)."""
    ordered = sorted(rungs.items())
    values = [value for _, value in ordered]
    reversals = [{"coarse_rung": ordered[i][0], "fine_rung": ordered[i + 1][0],
                  "coarse_pct": values[i], "fine_pct": values[i + 1],
                  "increase_pp": values[i + 1] - values[i]}
                 for i in range(len(ordered) - 1) if values[i + 1] > values[i]]
    finest_rung, finest_pct = ordered[-1]
    monotone = not reversals
    finest_within = finest_pct <= tolerance
    if len(ordered) < 3:
        klass = "resolution_pair_only"
        reason = f"only {len(ordered)} rung(s) at this protocol"
    elif monotone and finest_within:
        klass = "resolution_attributable"
        reason = (f"{len(ordered)} rungs, error non-increasing under refinement, "
                  f"finest {finest_rung:g} at {finest_pct:.2f}%")
    else:
        klass = "resolution_window"
        bits = []
        if reversals:
            bits.append("error rises under refinement")
        if not finest_within:
            bits.append(f"finest {finest_rung:g} at {finest_pct:.2f}% > {tolerance:g}%")
        reason = f"{len(ordered)} rungs; " + "; ".join(bits)
    return {"class": klass, "reason": reason, "monotone": monotone,
            "finest_rung": finest_rung, "finest_pct": finest_pct,
            "finest_within_standard": finest_within,
            "reversals": reversals,
            "rungs": {f"{rung:g}": value for rung, value in ordered}}


def classify(levels: Mapping[float, dict], record_pct: float,
             record_level_vt: float,
             tolerance: float = OWNER_TOLERANCE_PCT) -> dict:
    """Grade the terminal-time ladder of one record (see the module docstring)."""
    ordered = sorted(levels.items())
    if len(ordered) < MIN_LEVELS:
        return {
            "class": "insufficient_time_ladder",
            "reason": (f"only {len(ordered)} terminal time(s) committed at this "
                       f"grid; {MIN_LEVELS} are needed before the axis is graded"),
            "tolerance_pct": tolerance,
            "n_levels": len(ordered),
            "levels": {f"{vt:g}": lv["error_pct"] for vt, lv in ordered},
        }
    longer = [(vt, lv) for vt, lv in ordered if vt > record_level_vt]
    below = [(vt, lv) for vt, lv in longer if lv["error_pct"] < record_pct]
    passing = [(vt, lv) for vt, lv in longer
               if lv["error_pct"] <= tolerance]
    all_below = len(below) == len(longer) and bool(longer)
    passing_steady = [vt for vt, lv in passing if lv["admissible"]]
    if passing and all_below:
        klass = "transient_reading"
        first = min(passing)
        reason = (f"at the record's own grid every longer terminal time measures "
                  f"less error and {len(passing)} of {len(longer)} are inside "
                  f"{tolerance:g}% (first pass at {first[0]:g} H^2/nu: "
                  f"{first[1]['error_pct']:.2f}%); the over-tolerance value is a "
                  f"pre-steady reading, not a grid effect")
        if not passing_steady:
            reason += "; no passing level cleared the steadiness gate"
    else:
        klass = "reading_window_time"
        bits = []
        if not all_below:
            rising = [vt for vt, lv in longer if lv["error_pct"] >= record_pct]
            bits.append("longer terminal time does not reduce the error at "
                        + ", ".join(f"{vt:g}" for vt in rising) + " H^2/nu")
        if not passing:
            bits.append(f"no longer level is inside {tolerance:g}%")
        reason = ("the transient claim is refuted: " + "; ".join(bits))
    return {
        "class": klass,
        "reason": reason,
        "tolerance_pct": tolerance,
        "n_levels": len(ordered),
        "record_level_viscous_times": record_level_vt,
        "record_pct": record_pct,
        "levels": {f"{vt:g}": lv["error_pct"] for vt, lv in ordered},
        "admissible_levels": [f"{vt:g}" for vt, lv in ordered if lv["admissible"]],
        "passing_levels": [f"{vt:g}" for vt, _ in passing],
        "passing_levels_steady": passing_steady,
        "every_longer_level_below_record": all_below,
    }


def classify_record(extractor: str,
                    tolerance: float = OWNER_TOLERANCE_PCT) -> dict | None:
    """Guard verdict for one registry record, or ``None`` when not declared."""
    spec = next((item for item in RECORDS if item["extractor"] == extractor), None)
    if spec is None:
        return None
    record = record_level(spec["record_artifact"])
    levels: dict[float, dict] = {}
    for source in spec["time_sources"]:
        if not (ROOT / source).exists():
            continue
        for vt, level in time_levels(source, record["resolution_H_over_dx"]).items():
            levels.setdefault(vt, level)
    verdict = classify(levels, record["error_pct"], record["viscous_times"],
                       tolerance)
    protocol_ladder: dict[str, dict] = {}
    for source in spec["time_sources"]:
        if not (ROOT / source).exists():
            continue
        doc = _load(source)
        for protocol in sorted({c["protocol"] for c in doc.get("cells", [])}):
            rungs = protocol_resolution_ladder(source, protocol)
            if rungs:
                protocol_ladder[protocol] = {
                    "source": source,
                    **_grade_resolution(rungs, tolerance),
                }
    return {
        "extractor": extractor,
        "method": spec["method"],
        "quantity": spec["quantity"],
        "details": spec["details"],
        "resolution_H_over_dx": record["resolution_H_over_dx"],
        "record": record,
        "sources": [s for s in spec["time_sources"] if (ROOT / s).exists()],
        **verdict,
        "matched_protocol_resolution": protocol_ladder,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable dump")
    args = parser.parse_args(argv)
    summary = {spec["extractor"]: classify_record(spec["extractor"])
               for spec in RECORDS}
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    for extractor, verdict in summary.items():
        if verdict is None:
            continue
        print(f"{extractor}: {verdict['class']} "
              f"(H/dx={verdict['resolution_H_over_dx']:g}, "
              f"{verdict['n_levels']} terminal times)")
        for vt, value in verdict["levels"].items():
            print(f"    t_end = {vt:>4} H^2/nu -> {value:6.2f} %")
        print(f"    {verdict['reason']}")
        for protocol, res in verdict["matched_protocol_resolution"].items():
            print(f"    resolution at the {protocol!r} protocol: {res['class']} "
                  f"({res['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())