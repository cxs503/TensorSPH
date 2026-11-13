#!/usr/bin/env python3
"""Assemble the Stokes' second-problem resolution ladder from probe shards.

Why this script exists
----------------------
The ladder ``delta/dx = 8, 12, 18`` costs 132 s + 271 s + 853 s serially (three
rungs, one wall clock hour on a busy core), so it is normally produced as
independent shards — one ``run.py --probe <rung>`` per node/queue slot — and the
published ``results_ladder.json`` is then *assembled* from the shard outputs.

The first delivery of this case was assembled by a script that waited at most
two seconds for a shard (a ``while`` guard of 2 s around a ``sleep(5)`` body) and
then published a ladder with ``resolutions = [8]`` while still writing
``G6_not_growing: true, all_gates: true``.  The convergence gate was therefore
vacuous, not passed.  This replacement is **fail-closed**:

* every predeclared rung must appear exactly once, otherwise *nothing* is
  published (``results_ladder.json`` / ``benchmark.json`` / ``MANIFEST.json`` are
  left untouched) and the process exits non-zero with
  ``ladder_assembly_refused.json`` as the record of what was missing;
* every rung must carry the same case fingerprint (``nu, omega, U0, c0,
  periods, measure_periods, H/delta, columns, gamma, h/dx, wall_friction``) — a
  ladder assembled from differently-configured shards is not a convergence
  study;
* a single-rung "ladder" is refused outright: one mesh cannot test convergence;
* a shard file that is still being written is *waited for*, never skipped; the
  wait is bounded by ``--timeout`` (default 3600 s) and the poll sleeps
  ``min(--poll, remaining)``, so a short timeout can never truncate the ladder.

The convergence block, the slimming and the sha256 manifest are **the same code**
``run.py --ladder`` uses (``ladder_convergence`` / ``_slim_payload`` /
``_write_bundle``), so an assembled ladder and a serially-run ladder are
byte-comparable apart from the ``assembly`` provenance block.

Usage
-----
    # three shards (e.g. three cluster jobs), each writing results_probe.json
    PYTHONPATH=.:src python scripts/finalize_stokes_second.py \
        --shard runs/ladder/dx8 runs/ladder/dx12 runs/ladder/dx18 \
        --out benchmarks/stokes_second/out \
        --compare benchmarks/stokes_second/out/results_ladder.json

Exit status: 0 = published, 2 = refused (nothing published), 3 = bad usage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from benchmarks.stokes_second import run as bench  # noqa: E402

PROBE_FILE = "results_probe.json"

#: provenance.case keys that define the configuration.  Two rungs whose values
#: differ are not on the same ladder; ``ladder`` itself is excluded (it records
#: the module constant, not which rung this file is).
FINGERPRINT_KEYS = (
    "nu", "omega", "U0", "c0", "periods", "measure_periods",
    "H_over_delta", "columns", "gamma", "h_factor", "wall_friction",
)

PUBLISHED = ("results_ladder.json", "benchmark.json", "MANIFEST.json")
REFUSAL_FILE = "ladder_assembly_refused.json"


class AssemblyRefused(RuntimeError):
    """Raised when the shards cannot form a complete, consistent ladder."""


# --------------------------------------------------------------------------- #
# Discovery and waiting
# --------------------------------------------------------------------------- #

def _read_probe(path: Path):
    """Parse a shard file; ``None`` while it is still being written."""
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        return None
    if not isinstance(runs[0], dict) or "resolution_delta_over_dx" not in runs[0]:
        return None
    return payload


def discover(shard_dirs, probe_file: str = PROBE_FILE) -> dict:
    """Map ``resolution -> (path, payload)`` from the shard directories."""
    found: dict = {}
    duplicates = []
    for root in shard_dirs:
        path = Path(root) / probe_file
        if not path.is_file():
            continue
        payload = _read_probe(path)
        if payload is None:
            continue
        res = int(round(float(payload["runs"][0]["resolution_delta_over_dx"])))
        if res in found:
            duplicates.append({"resolution": res,
                               "paths": [str(found[res][0]), str(path)]})
            continue
        found[res] = (path, payload)
    return {"rungs": found, "duplicates": duplicates}


def wait_for_ladder(shard_dirs, required, timeout: float, poll: float,
                    probe_file: str = PROBE_FILE, verbose: bool = True) -> dict:
    """Wait until every required rung is readable; fail-closed on timeout.

    The poll sleeps ``min(poll, remaining)`` — the bug that produced the vacuous
    one-rung ladder was a 5 s sleep inside a 2 s guard, i.e. the guard expired
    before the sleep it was wrapping could ever finish.
    """
    t0 = time.monotonic()
    last_report = 0.0
    while True:
        state = discover(shard_dirs, probe_file)
        missing = [r for r in required if r not in state["rungs"]]
        if not missing:
            return state
        elapsed = time.monotonic() - t0
        remaining = timeout - elapsed
        if remaining <= 0:
            raise AssemblyRefused(
                "timed out after %.0f s waiting for rung(s) %s in %s "
                "(found %s)" % (timeout, missing, [str(d) for d in shard_dirs],
                                sorted(state["rungs"])))
        if verbose and elapsed - last_report >= max(poll, 5.0):
            last_report = elapsed
            print("  waiting: missing rungs %s (found %s, %.0f s elapsed)"
                  % (missing, sorted(state["rungs"]), elapsed), flush=True)
        time.sleep(min(poll, remaining))


# --------------------------------------------------------------------------- #
# Consistency checks
# --------------------------------------------------------------------------- #

def case_fingerprint(payload: dict) -> dict:
    case = (payload.get("provenance") or {}).get("case") or {}
    return {k: case.get(k) for k in FINGERPRINT_KEYS}


def check_fingerprints(rungs: dict) -> dict:
    """Every rung must describe the same case; return the agreed fingerprint."""
    prints = {res: case_fingerprint(p) for res, (_, p) in rungs.items()}
    reference_res = min(prints)
    reference = prints[reference_res]
    disagreements = []
    for res, fp in prints.items():
        for key in FINGERPRINT_KEYS:
            if fp.get(key) != reference.get(key):
                disagreements.append({"resolution": res, "key": key,
                                      "value": fp.get(key),
                                      "reference_resolution": reference_res,
                                      "reference_value": reference.get(key)})
    if disagreements:
        raise AssemblyRefused(
            "shards disagree on the case configuration: "
            + "; ".join("dx%s.%s=%r != %r"
                        % (d["resolution"], d["key"], d["value"],
                           d["reference_value"]) for d in disagreements))
    return reference


def validate_rungs(rungs: dict, required) -> None:
    if sorted(rungs) != sorted(required):
        raise AssemblyRefused(
            "assembled rung set %s != predeclared %s"
            % (sorted(rungs), sorted(required)))
    if len(required) < 2:
        raise AssemblyRefused(
            "a %d-rung ladder cannot test convergence (G6 would be vacuous)"
            % len(required))
    for res, (path, payload) in sorted(rungs.items()):
        run = payload["runs"][0]
        gates = run.get("gates")
        if not isinstance(gates, dict) or not gates:
            raise AssemblyRefused("rung %s has no gate block (%s)" % (res, path))
        if "amplitude_l2_half_space_pct" not in run.get("metrics", {}):
            raise AssemblyRefused(
                "rung %s has no graded metric amplitude_l2_half_space_pct (%s)"
                % (res, path))


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def compare_with(reference: dict, runs: list) -> dict:
    """Per-rung reproduction check against an already-published ladder."""
    by_res = {int(round(float(r["resolution_delta_over_dx"]))): r
              for r in reference.get("runs", [])}
    table = []
    for run in runs:
        res = int(round(float(run["resolution_delta_over_dx"])))
        old = by_res.get(res)
        if old is None:
            table.append({"resolution": res, "present_in_reference": False})
            continue
        new_l2 = abs(run["metrics"]["amplitude_l2_half_space_pct"])
        old_l2 = abs(old["metrics"]["amplitude_l2_half_space_pct"])
        table.append({
            "resolution": res,
            "l2_pct": new_l2,
            "reference_l2_pct": old_l2,
            "delta_pp": new_l2 - old_l2,
            "relative_delta": ((new_l2 - old_l2) / old_l2) if old_l2 else None,
            "steps_match": run.get("steps") == old.get("steps"),
            "dt_match": run.get("dt") == old.get("dt"),
        })
    return {"reference": str(reference.get("_path", "")), "per_rung": table}


def assemble(rungs: dict, required, repro_cmd: str, compare_to=None) -> dict:
    validate_rungs(rungs, required)
    fingerprint = check_fingerprints(rungs)

    runs = [rungs[res][1]["runs"][0] for res in sorted(rungs)]
    provenance = dict(rungs[sorted(rungs)[0]][1]["provenance"])
    provenance["case"] = dict(provenance.get("case") or {}, **fingerprint)

    conv = bench.ladder_convergence(runs, list(required))
    payload = {"mode": "ladder", "provenance": provenance, "runs": runs,
               "convergence": conv}

    sources = []
    for res in sorted(rungs):
        path, _ = rungs[res]
        sources.append({"resolution": res, "path": str(path),
                        "sha256": bench._sha256(path),
                        "bytes": path.stat().st_size})

    assembly = {
        "assembler": "scripts/finalize_stokes_second.py",
        "sources": sources,
        "fingerprint": fingerprint,
        "rungs_required": list(required),
        "rungs_assembled": [int(round(float(r["resolution_delta_over_dx"])))
                            for r in runs],
        "case_consistent": True,
    }
    if compare_to is not None:
        assembly["reproduction_check"] = compare_with(compare_to, runs)
    payload["assembly"] = assembly
    payload["repro_cmd"] = repro_cmd
    payload["elapsed_s"] = sum(float(r.get("wall_clock_s") or 0.0) for r in runs)
    return payload


def publish(payload: dict, out_dir: Path, repro_cmd: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results_ladder.json"
    path.write_text(json.dumps(bench._slim_payload(payload), indent=1,
                               default=bench._jsonable))
    bench._write_bundle(out_dir, payload, repro_cmd)
    return path


def _record_refusal(reason: str, out_dir: Path, detail=None) -> Path:
    record = {"refused": True, "reason": reason,
              "published": False,
              "untouched": [str(out_dir / name) for name in PUBLISHED],
              "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if detail is not None:
        record["detail"] = detail
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REFUSAL_FILE
    path.write_text(json.dumps(record, indent=1, default=bench._jsonable))
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Fail-closed assembly of the Stokes' second-problem ladder "
                    "from run.py --probe shard directories.")
    ap.add_argument("--shard", nargs="+", required=True,
                    help="shard directories, each holding a results_probe.json")
    ap.add_argument("--probe-file", default=PROBE_FILE)
    ap.add_argument("--out", type=Path, default=bench.OUT)
    ap.add_argument("--expected-rungs", nargs="+", type=int, default=None,
                    help="predeclared rungs (default: run.LADDER)")
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="max seconds to wait for missing shards (default 3600)")
    ap.add_argument("--poll", type=float, default=5.0,
                    help="poll interval in seconds, capped by the remaining time")
    ap.add_argument("--compare", type=Path, default=None,
                    help="existing results_ladder.json to compare against")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.poll <= 0 or args.timeout <= 0:
        print("--poll and --timeout must be positive", file=sys.stderr)
        return 3
    required = list(args.expected_rungs) if args.expected_rungs else list(bench.LADDER)
    repro_cmd = ("PYTHONPATH=.:src python scripts/finalize_stokes_second.py "
                 + " ".join(sys.argv[1:]))

    try:
        state = wait_for_ladder(args.shard, required, args.timeout, args.poll,
                                args.probe_file, verbose=not args.quiet)
        if state["duplicates"]:
            raise AssemblyRefused(
                "two shards produced the same rung: %s" % (state["duplicates"],))
        compare_to = None
        if args.compare is not None:
            compare_to = json.loads(args.compare.read_text())
            compare_to["_path"] = str(args.compare)
        payload = assemble(state["rungs"], required, repro_cmd, compare_to)
    except AssemblyRefused as exc:
        path = _record_refusal(str(exc), args.out)
        print("REFUSED: %s" % exc, file=sys.stderr)
        print("nothing published; record written to %s" % path, file=sys.stderr)
        return 2

    path = publish(payload, args.out, repro_cmd)
    conv = payload["convergence"]
    if not args.quiet:
        print("wrote %s" % path)
        print("  resolutions   %s" % conv["resolutions"])
        print("  L2 %% (half)    %s" % [round(v, 4) for v in conv["l2_by_resolution"]])
        print("  complete      %s" % conv["complete"])
        print("  G6_not_growing %s" % conv["G6_not_growing"])
        print("  all_gates     %s" % conv["all_gates"])
        check = payload["assembly"].get("reproduction_check")
        if check:
            for row in check["per_rung"]:
                if row.get("present_in_reference") is False:
                    print("  dx%s        not in reference" % row["resolution"])
                else:
                    print("  dx%-3s L2 %.4f%% vs %.4f%% (%.2e pp)"
                          % (row["resolution"], row["l2_pct"],
                             row["reference_l2_pct"], row["delta_pp"]))
    return 0 if conv["all_gates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())