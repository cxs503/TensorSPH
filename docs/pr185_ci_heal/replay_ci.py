"""Replay every step of ``.github/workflows/tests.yml`` in this container.

The workflow job that is reported as "pytest (3.10/3.11/3.12) = failure" runs,
in order: ruff lint -> pytest -> a tiny droplet case + manifest contract ->
benchmark comparisons -> hydrostatic/dam-break gates -> two negative controls.
Because a red step short-circuits the job, a local replay of the *whole* list is
the only way to show that nothing behind the (fixed) lint step is broken.

Usage (from the repository root):

    PYTHONPATH=src python docs/pr185_ci_heal/replay_ci.py [--skip pytest]

Writes ``docs/pr185_ci_heal/ci_replay.json`` (tracked evidence: return codes,
durations, last lines) and prints one line per step.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = sys.executable

STEPS = [
    ("lint", [PY, "-m", "ruff", "check", "."], 120),
    ("pytest", [PY, "-m", "pytest", "-q"], 1200),
    ("droplet_e2e", [PY, "examples/droplet_impact_2d/run.py", "--steps", "3",
                     "--output", "artifacts/ci-droplet"], 300),
    ("manifest_contract", [PY, "docs/pr185_ci_heal/steps/manifest_contract.py"], 120),
    ("neighbor_compare_2d", [PY, "benchmarks/neighbor_search/compare.py",
                             "--particles", "400", "--dimension", "2"], 300),
    ("neighbor_compare_3d", [PY, "benchmarks/neighbor_search/compare.py",
                             "--particles", "400", "--dimension", "3"], 300),
    ("rhs_compare", [PY, "benchmarks/neighbor_search/compare_rhs.py",
                     "--sizes", "128,256", "--repeats", "1"], 600),
    ("free_surface_compare", [PY, "benchmarks/free_surface/compare.py",
                              "--deltas", "0,0.1", "--bands", "6",
                              "--hydro-seconds", "0.5", "--dam-seconds", "0.3",
                              "--impact-seconds", "0.05",
                              "--json-out", "artifacts/free_surface.json"], 900),
    ("pair_reuse", [PY, "docs/pr185_ci_heal/steps/pair_reuse.py"], 300),
    ("hydrostatic_gate", [PY, "benchmarks/hydrostatic/run.py", "--seconds", "4",
                          "--boundary", "wall",
                          "--json-out", "artifacts/ci-hydrostatic.json"], 1200),
    ("fluctuation_negative_control",
     [PY, "docs/pr185_ci_heal/steps/fluctuation_gate.py"], 600),
    ("dam_break_gates", [PY, "benchmarks/dam_break/run.py", "--sizes", "0.04,0.02",
                         "--seconds", "0.4",
                         "--json-out", "artifacts/ci-dam-break.json"], 1200),
    ("fail_fast_negative_control", [PY, "docs/pr185_ci_heal/steps/fail_fast.py"], 300),
]


def tail(text: str, n: int = 3) -> list[str]:
    lines = [ln.rstrip() for ln in (text or "").strip().splitlines() if ln.strip()]
    return lines[-n:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[],
                    help="step ids to skip (e.g. pytest when it was run separately)")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    results = []
    for step_id, cmd, timeout in STEPS:
        if step_id in args.skip or (args.only and step_id not in args.only):
            results.append({"id": step_id, "skipped": True})
            print(f"{step_id:32s} SKIP")
            continue
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                  timeout=timeout)
            rc, out = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired:
            rc, out = 124, f"timeout after {timeout}s"
        dt = round(time.time() - t0, 1)
        results.append({"id": step_id, "rc": rc, "seconds": dt,
                        "last_lines": tail(out)})
        print(f"{step_id:32s} rc={rc:<3d} {dt:7.1f}s   {tail(out, 1)[0][:90] if out else ''}")
        if rc != 0:
            print("  --- failing step output tail ---")
            for ln in tail(out, 25):
                print("  " + ln)
            break

    failed = [r for r in results if not r.get("skipped") and r.get("rc") not in (0,)]
    payload = {"python": sys.version.split()[0],
               "arch": subprocess.run(["uname", "-m"], capture_output=True,
                                      text=True).stdout.strip(),
               "steps": results,
               "failed": [r["id"] for r in failed],
               "green": not failed and not any(r.get("skipped") for r in results)}
    out_f = ROOT / "docs" / "pr185_ci_heal" / "ci_replay.json"
    out_f.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n{len(results) - len(failed)}/{len(results)} steps run; failed={payload['failed']}")
    print("wrote", out_f.relative_to(ROOT))
    return 0 if payload["green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
