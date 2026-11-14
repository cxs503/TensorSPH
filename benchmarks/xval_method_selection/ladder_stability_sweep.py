"""Resolution ladder of method A's *stability onset* (anchor XVAL-SPH-METHOD-SELECTION-2026-11-12).

Why this exists
---------------
The pre-registered decision matrix (``run_cell.py``) wants four 5.5 s cells.  The
fine cell (``D20_h025_A``, H/Δx = 3.20) is the only one that ever wrote a record,
and that record is a *failure*: ``non-finite acceleration at step 3890``.  The
three remaining cells have never produced a record (the cluster nodes abort with
``free(): corrupted unsorted chunks`` before the record is written), so the
matrix cannot be graded and the anchor's physics verdict is WITHHELD.

This driver attacks the part of the question that does **not** need the full
5.5 s window and therefore does not need the cluster: *is the blow-up
resolution-driven, and at which H/Δx does it stop happening?*  The fine cell
died at t ≈ 0.53 s, inside the ramp, so a 0.60 s horizon is long enough to see
the same instability at every resolution — and it is cheap enough to run locally
(~10 min for the whole sweep instead of ~40 min per 5.5 s cell).

Two mechanism probes ride along at the fine resolution (both are the same
post-hoc probes as cells 2 and 3 of ``run_cell.py``): propagation-phase XSPH and
SPHinXsys-style compression regularization.  They answer "does this mechanism
change the *onset*, not just the amplitude".

What this does NOT claim
------------------------
A 0.60 s horizon never reaches the wave gauges (they sit 1.03–1.49 m from the
paddle; celerity ≈ 1.3 m/s), so **no wave height is measured here** and the
owner's 15 % / 3 % accuracy gates are neither proven nor disproven by this
sweep.  The sweep only grades *stability* (max|v|/c₀ < 1, compression < 10 %),
which is the necessary precondition for any gradeable cell.

Usage::

    PYTHONPATH=src python benchmarks/xval_method_selection/ladder_stability_sweep.py \
        --threads 8 --summary runs/xval/ladder_stability.json

Every number in the summary is parsed out of the records written by the shipped
driver ``benchmarks/wave_flume/run.py`` — nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_PY = REPO / "benchmarks" / "wave_flume" / "run.py"

# Case constants of the pre-registered geometry (docs/xval_method_selection_m17501/protocol.md §4).
DEPTH_M = 0.25
HEIGHT_RATIO = 0.16
SOUND_SPEED = 30.0
H_FACTOR = 1.3          # wave_flume: particle spacing h = 1.3 dx
DT_REQUESTED = 4.0e-4   # run.py --dt default
MAX_COURANT = 0.25      # run.py --max-courant default

SPEED_RATIO_MAX = 1.0    # stability gate: max|v|/c0 < 1
COMPRESSION_MAX = 0.10   # stability gate: max density compression < 10 %

# (label, depth cells through the depth, horizon s, extra CLI args, group)
SWEEP = [
    ("ladder_d10", 10, 0.60, [], "ladder"),
    ("ladder_d13", 13, 0.60, [], "ladder"),
    ("ladder_d16", 16, 0.60, [], "ladder"),
    ("ladder_d20", 20, 0.60, [], "ladder"),
    ("ladder_d26", 26, 0.60, [], "ladder"),
    ("mech_d20_xsph", 20, 0.60, ["--xsph", "0.05"], "mechanism"),
    ("mech_d20_comp", 20, 0.60,
     ["--density-diffusion", "0", "--density-regularization", "compression",
      "--compression-rule", "free_surface"], "mechanism"),
]

# run_cell.COMMON minus the resolution/thread flags, so a sweep point is the same
# physical case as the corresponding decision cell.
COMMON = [
    "--case", "regular",
    "--depth", str(DEPTH_M),
    "--height-ratio", str(HEIGHT_RATIO),
    "--wave-period", "1.0",
    "--flume-length", "3.2",
    "--sound-speed", str(SOUND_SPEED),
    "--viscosity", "1e-6",
    "--density-diffusion", "0.1",
    "--sponge-strength", "4.0",
    "--settle", "0.8",
    "--viscosity-settle", "0.02",
    "--xsph-settle", "0.1",
    "--ramp-periods", "2.0",
]

_STEP_RE = re.compile(r"at step (\d+)")


def step_size(depth_cells: int) -> float:
    """Acoustic step used by the driver (same formula as ``run.py::_step``)."""
    h = H_FACTOR * DEPTH_M / depth_cells
    return min(DT_REQUESTED, MAX_COURANT * h / SOUND_SPEED)


def h_over_dx(depth_cells: int) -> float:
    """Wave height over particle spacing — the resolution knob of the ladder."""
    return HEIGHT_RATIO * depth_cells


def parse_failure(error: str | None) -> dict:
    """Pull the blow-up step out of the driver's error string.

    ``run.py`` writes ``"FloatingPointError: non-finite acceleration at step 3890"``
    into the failed record.  The step is the only quantitative handle on *when*
    the integrator died, so a record that has none is reported as unknown rather
    than guessed.
    """
    if not error:
        return {"step": None, "onset_time_s": None}
    match = _STEP_RE.search(error)
    if match is None:
        return {"step": None, "onset_time_s": None}
    return {"step": int(match.group(1)), "onset_time_s": None}


def summarise_point(label: str, depth_cells: int, horizon: float, record: dict | None,
                    returncode: int, stdout: str) -> dict:
    """Turn one driver invocation into a comparable row."""
    dt = step_size(depth_cells)
    row = {
        "label": label,
        "depth_cells": depth_cells,
        "h_over_dx": round(h_over_dx(depth_cells), 4),
        "horizon_s": horizon,
        "dt_s": dt,
        "steps_requested": int(math.ceil(horizon / dt)),
        "returncode": returncode,
        "record_present": record is not None,
        "record_failed": None,
        "error": None,
        "onset_step": None,
        "onset_time_s": None,
        "simulated_time_s": None,
        "max_speed_m_s": None,
        "speed_ratio": None,
        "max_compression": None,
        "escaped": None,
        "verdict": "no_record",
        "stable": False,
    }
    if record is None:
        row["error"] = (stdout.strip().splitlines() or ["<no output>"])[-1][:400]
        return row
    if record.get("failed"):
        row["record_failed"] = True
        row["error"] = record.get("error")
        kind = str(record.get("error") or "").split(":", 1)[0]
        if kind != "FloatingPointError":
            # A rejected configuration (e.g. the crest is not resolved by two
            # particles) never integrates: it is out of the ladder's domain, not
            # evidence of an unstable method.
            row["verdict"] = "rejected"
            return row
        parsed = parse_failure(record.get("error"))
        row["onset_step"] = parsed["step"]
        if parsed["step"] is not None:
            row["onset_time_s"] = parsed["step"] * dt
        row["verdict"] = "blew_up"
        return row

    diag = record.get("diagnostics", {}) or {}
    row["record_failed"] = False
    row["simulated_time_s"] = diag.get("simulated_time")
    row["max_speed_m_s"] = diag.get("max_speed")
    row["max_compression"] = diag.get("max_density_compression")
    row["escaped"] = diag.get("escaped")
    speed = diag.get("max_speed")
    compression = diag.get("max_density_compression")
    if speed is not None:
        row["speed_ratio"] = speed / SOUND_SPEED
    ok = (row["speed_ratio"] is not None and compression is not None
          and row["speed_ratio"] < SPEED_RATIO_MAX and compression < COMPRESSION_MAX)
    row["stable"] = bool(ok)
    row["verdict"] = "survived_stable" if ok else "survived_unstable"
    return row


def _run_point(label: str, depth_cells: int, horizon: float, extra: list[str],
               threads: int, out_dir: Path, timeout: float) -> tuple:
    out = out_dir / f"{label}.json"
    cmd = [sys.executable, str(RUN_PY), *COMMON,
           "--depth-cells", str(depth_cells),
           "--simulate-time", str(horizon),
           "--threads", str(threads),
           "--out", str(out), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          cwd=str(REPO))
    record = None
    if out.exists():
        try:
            record = json.loads(out.read_text())
        except json.JSONDecodeError:
            record = None
    return record, proc.returncode, proc.stdout + proc.stderr


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--points", default="",
                        help="comma separated labels; default = all")
    parser.add_argument("--out-dir", default="runs/xval/ladder")
    parser.add_argument("--summary", default="runs/xval/ladder_stability.json")
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args(argv)

    wanted = {p.strip() for p in args.points.split(",") if p.strip()}
    points = [p for p in SWEEP if not wanted or p[0] in wanted]
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, depth_cells, horizon, extra, group in points:
        record, rc, stdout = _run_point(label, depth_cells, horizon, extra,
                                        args.threads, out_dir, args.timeout)
        row = summarise_point(label, depth_cells, horizon, record, rc, stdout)
        row["group"] = group
        row["extra_args"] = extra
        rows.append(row)
        print(f"{label:16s} H/dx={row['h_over_dx']:5.2f}  {row['verdict']:18s}"
              f" onset={row['onset_time_s']} s  max|v|/c0={row['speed_ratio']}"
              f"  compression={row['max_compression']}", flush=True)

    summary = {
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "case": {
            "depth_m": DEPTH_M,
            "height_ratio": HEIGHT_RATIO,
            "sound_speed_m_s": SOUND_SPEED,
            "horizon_s": sorted({p[2] for p in points}),
            "comment": ("0.60 s horizon: reaches the ramp instability of the fine cell "
                        "(t = 0.53 s) but never the wave gauges, so no wave height and "
                        "no 15 % accuracy claim is made here"),
        },
        "gates": {"speed_ratio_max": SPEED_RATIO_MAX, "compression_max": COMPRESSION_MAX},
        "rows": rows,
        "counts": {
            "points": len(rows),
            "blew_up": sum(r["verdict"] == "blew_up" for r in rows),
            "survived_stable": sum(r["verdict"] == "survived_stable" for r in rows),
            "survived_unstable": sum(r["verdict"] == "survived_unstable" for r in rows),
            "rejected": sum(r["verdict"] == "rejected" for r in rows),
            "no_record": sum(r["verdict"] == "no_record" for r in rows),
        },
    }
    summary_path = REPO / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["counts"], indent=2))
    print(f"wrote {summary_path}")
    return 0 if summary["counts"]["no_record"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())