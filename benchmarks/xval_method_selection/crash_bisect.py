"""Bisect *when* a decision cell of method A dies on the loong64 nodes.

Why this exists
---------------
``run_cell.py`` runs one cell of the pre-registered matrix (``t_end = 5.5 s``).
On the cluster nodes that long cell aborts with ``free(): corrupted unsorted
chunks`` (rc=255) *without* writing a record, while the same solver finishes a
0.05 s and a 0.60 s horizon with rc=0 on the same nodes (mission 20783 / 21050).
"Long runs crash, short runs do not" is the whole of the current knowledge, i.e.
the onset is somewhere in ``(0.60 s, 5.5 s]``.

This driver narrows that interval without touching the case: for a list of
horizons it launches ``run_cell.py <cell> --simulate-time <t>`` **as a child
process**, so a SIGABRT takes down only that one child and the bisect keeps
going.  For each horizon it records

* the process return code (a crash shows up as the child's negative rc / 255),
* whether the record file exists (a record means the solver reached the end),
* the diagnostics of a surviving record (steps, simulated time, escaped
  particles, max speed, compression),

and writes them to a summary JSON.  It never invents a value: a horizon with no
record is reported as ``record=False`` and nothing else.

This is a *diagnostic* tool.  It answers "how long can method A run", not "is
method A accurate"; no wave-height number is produced here.

Usage::

    PYTHONPATH=src python benchmarks/xval_method_selection/crash_bisect.py \\
        --cell 1 --times 0.6 1.6 2.6 3.6 4.4 5.5 \\
        --outdir runs/xval/bisect \\
        --summary runs/xval/crash_bisect.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_CELL = HERE / "run_cell.py"

# Long enough for the 5.5 s protocol horizon on a quiet node, short enough that a
# hung child does not eat the whole job window.
DEFAULT_CHILD_TIMEOUT = 3600.0


def record_path(outdir: Path, horizon: float) -> Path:
    """Deterministic record name per horizon (``0.60`` -> ``t0p60.json``)."""
    tag = f"{float(horizon):.2f}".replace(".", "p")
    return outdir / f"t{tag}.json"


def describe(rc: int, path: Path) -> dict:
    """One row of the bisect table: return code plus what the record says."""
    row = {"horizon": None, "rc": rc, "record": path.exists(), "path": str(path)}
    if not row["record"]:
        return row
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as exc:  # a half-written file is not a result
        row["record"] = False
        row["error"] = f"unreadable record: {type(exc).__name__}: {exc}"
        return row
    if record.get("failed"):
        row["solver_failed"] = True
        row["error"] = record.get("error", "run failed")
        return row
    diag = record.get("diagnostics", {})
    row["solver_failed"] = False
    row["simulated_time"] = diag.get("simulated_time")
    row["steps_taken"] = diag.get("steps_taken")
    row["escaped"] = diag.get("escaped")
    row["max_speed"] = diag.get("max_speed")
    row["max_density_compression"] = diag.get("max_density_compression")
    row["gauges"] = len(record.get("probe_x", []))
    # Wave data only becomes gradeable once the incident wave has reached the
    # first gauge; that arrival time is derived from the record's own spec.
    spec = record.get("spec", {})
    probe_x = record.get("probe_x") or []
    celerity = spec.get("celerity")
    if probe_x and celerity:
        row["first_arrival_s"] = round(float(probe_x[0]) / float(celerity), 4)
    return row


def bisect(cell: int, times, outdir: Path, timeout: float = DEFAULT_CHILD_TIMEOUT,
           runner=subprocess.run, python: str | None = None) -> dict:
    """Run every horizon in its own child process and tabulate the outcomes."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    python = python or sys.executable
    rows = []
    for horizon in times:
        path = record_path(outdir, horizon)
        if path.exists():
            path.unlink()  # never grade a stale record from an earlier run
        cmd = [python, str(RUN_CELL), str(cell),
               "--simulate-time", f"{float(horizon):g}",
               "--out", str(path)]
        row = {"horizon": float(horizon)}
        try:
            proc = runner(cmd, cwd=str(HERE.parents[1]), timeout=timeout,
                          capture_output=True, text=True)
            rc = proc.returncode
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:]
            row["stderr_tail"] = tail[0] if tail else ""
        except subprocess.TimeoutExpired:
            rc = None
            row["stderr_tail"] = f"child timed out after {timeout:g} s"
        row.update(describe(rc, path))
        row["horizon"] = float(horizon)
        rows.append(row)
        verdict = ("survived" if row.get("record") and row.get("rc") == 0
                   else "died")
        print(f"bisect cell {cell} t_end={horizon:g}s  {verdict}  rc={rc} "
              f"record={row.get('record')}", flush=True)
    survived = [r for r in rows if r.get("record") and r.get("rc") == 0]
    died = [r for r in rows if not (r.get("record") and r.get("rc") == 0)]
    summary = {
        "cell": cell,
        "horizons": [float(t) for t in times],
        "rows": rows,
        "counts": {"horizons": len(rows), "survived": len(survived),
                   "died": len(died)},
        "longest_survived": max((r["horizon"] for r in survived), default=None),
        "first_death": min((r["horizon"] for r in died), default=None),
        "onset_bracket": [max((r["horizon"] for r in survived), default=None),
                          min((r["horizon"] for r in died), default=None)],
    }
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=int, default=1,
                        help="cell index of run_cell.CELLS (default 1 = D13_h025_A)")
    parser.add_argument("--times", type=float, nargs="+",
                        default=[0.6, 1.6, 2.6, 3.6, 4.4, 5.5],
                        help="horizons to try, seconds (ascending recommended)")
    parser.add_argument("--outdir", default="runs/xval/bisect")
    parser.add_argument("--summary", default="runs/xval/crash_bisect.json")
    parser.add_argument("--child-timeout", type=float, default=DEFAULT_CHILD_TIMEOUT)
    args = parser.parse_args(argv)

    summary = bisect(args.cell, args.times, Path(args.outdir),
                     timeout=args.child_timeout)
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}: {summary['counts']} onset_bracket={summary['onset_bracket']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())