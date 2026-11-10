"""Acceptance gates of the free-surface verification surface.

One place for the thresholds, so the code, the tests, the report table and the
eventual honest FAIL all quote the same numbers. A gate is never "assumed
passed": it is ``PASS``, ``FAIL`` or ``UNMEASURED``, and ``UNMEASURED`` (a NaN,
a missing run, a criterion that the case cannot answer) counts as not passed for
the maturity decision.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

__all__ = ["Criterion", "CRITERIA", "evaluate", "maturity"]


@dataclass(frozen=True)
class Criterion:
    identifier: str
    description: str
    threshold: float
    unit: str = "-"
    kind: str = "max"          # "max": measured <= threshold; "target": |m-target| <= tol
    tolerance: float = 0.0
    source: str = ""           # what produces the number


CRITERIA: tuple = (
    Criterion(
        "C1",
        "Ritter dry-bed dam break: front-position error, resolution-matched",
        0.10,
        unit="relative to front travel",
        kind="max",
        source="tensorsph.freesurface.ritter_anchor (max over snapshots)",
    ),
    Criterion(
        "C2",
        "Solitary wave: crest-amplitude decay over 10 characteristic wavelengths",
        0.05,
        unit="relative amplitude loss",
        kind="max",
        source="tensorsph.freesurface.solitary_anchor (extrapolated from the "
               "measured attenuation per wavelength)",
    ),
    Criterion(
        "C3",
        "Sloshing first-mode frequency error",
        0.10,
        unit="relative to sqrt(pi g/L tanh(pi H/L))",
        kind="max",
        source="tensorsph.freesurface.sloshing_mode (mean-crossing period)",
    ),
    Criterion(
        "C4",
        "Resolution convergence: slope of the depth L2 error against particle spacing",
        1.0,
        unit="order of convergence",
        kind="target",
        tolerance=0.2,
        source="tensorsph.freesurface.profiles.loglog_slope (Ritter ladder)",
    ),
    Criterion(
        "C5",
        "Mass conservation: total particle mass drift (and escaped particles)",
        0.01,
        unit="relative",
        kind="max",
        source="tensorsph.freesurface.ritter_anchor / sloshing_mode / "
               "solitary_anchor mass_drift (max over the three cases)",
    ),
)


def evaluate(measured: dict, criteria: tuple = CRITERIA,
             notes: dict | None = None) -> dict:
    """Score every criterion against the measured numbers.

    ``measured`` maps criterion identifiers to floats. Missing keys and NaNs
    give ``UNMEASURED``; nothing is silently treated as passing.
    """
    notes = notes or {}
    rows = []
    for criterion in criteria:
        value = measured.get(criterion.identifier, float("nan"))
        value = float(value) if value is not None else float("nan")
        if not math.isfinite(value):
            status = "UNMEASURED"
        elif criterion.kind == "max":
            status = "PASS" if value <= criterion.threshold else "FAIL"
        elif criterion.kind == "target":
            status = ("PASS"
                      if abs(value - criterion.threshold) <= criterion.tolerance
                      else "FAIL")
        else:
            raise ValueError(f"unknown criterion kind {criterion.kind!r}")
        rows.append({
            "id": criterion.identifier,
            "description": criterion.description,
            "measured": value,
            "threshold": criterion.threshold,
            "tolerance": criterion.tolerance,
            "kind": criterion.kind,
            "unit": criterion.unit,
            "status": status,
            "source": criterion.source,
            "note": notes.get(criterion.identifier, ""),
        })
    return {"gates": rows}


def maturity(report: dict) -> dict:
    """Overall verdict: every gate must be PASS."""
    rows = report["gates"]
    failed = [r["id"] for r in rows if r["status"] != "PASS"]
    return {
        "gate_count": len(rows),
        "passed": [r["id"] for r in rows if r["status"] == "PASS"],
        "failed": failed,
        "verdict": "PASS" if not failed else "HONEST_FAIL_OPEN",
        "open_items": [f"{r['id']}: {r['status']} (measured {r['measured']:g}, "
                       f"target {r['threshold']:g})"
                       for r in rows if r["status"] != "PASS"],
    }
