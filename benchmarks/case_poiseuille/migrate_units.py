"""One-off, idempotent unit migration of an existing ``out/result.json``.

The SPH runs behind ``out/result.json`` are expensive (minutes of solver time),
and the *only* thing wrong with the committed artifact was labelling: per-row
phases are radians, but the summary emitted them into ``*_deg`` fields.  This
script rewrites the derived fields from the stored radian fits -- the raw
samples, profiles and gate verdicts are not touched -- so the artifact, the
generated note and the figures agree on degrees before the next full
re-simulation (``run.py`` now emits the unit-explicit schema directly).

Run:  PYTHONPATH=.:src python3 benchmarks/case_poiseuille/migrate_units.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out" / "result.json"
NOTE = ("unit-explicit-v2: phase_*_rad / phase_*_deg written from the stored "
        "radian fits; raw samples, profiles and gate verdicts unchanged")


def _run_units(run: dict) -> dict:
    rows = run["rows"]
    centre = min(rows, key=lambda r: abs(r["y_over_H"] - 0.5))
    run["phase_centre_rad"] = centre["phase"]
    run["phase_centre_deg"] = math.degrees(centre["phase"])
    run["phase_centre_analytic_rad"] = centre["phase_analytic"]
    run["phase_centre_analytic_deg"] = math.degrees(centre["phase_analytic"])
    for row in rows:
        row["phase_deg"] = math.degrees(row["phase"])
        row["phase_analytic_deg"] = math.degrees(row["phase_analytic"])
    return run


def main() -> int:
    if not OUT.exists():
        raise SystemExit(f"missing {OUT}")
    data = json.loads(OUT.read_text())
    runs = list(data["harmonic_runs"])
    if data.get("harmonic_fast"):
        runs.append(data["harmonic_fast"])
    for run in runs:
        _run_units(run)

    freq = data.get("frequency_discrimination")
    if freq:
        base = next(r for r in data["harmonic_runs"] if r["resolution_H_over_dx"] == 12)
        fast = data["harmonic_fast"]
        m = fast["phase_centre_rad"] - base["phase_centre_rad"]
        a = fast["phase_centre_analytic_rad"] - base["phase_centre_analytic_rad"]
        freq["phase_difference_measured_rad"] = m
        freq["phase_difference_measured_deg"] = math.degrees(m)
        freq["phase_difference_analytic_rad"] = a
        freq["phase_difference_analytic_deg"] = math.degrees(a)

    data["units_schema"] = NOTE
    OUT.write_text(json.dumps(data, indent=2))
    print(f"[migrate] rewrote phase units in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
