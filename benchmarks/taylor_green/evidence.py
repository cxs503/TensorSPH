#!/usr/bin/env python3
"""Build the machine-checkable evidence bundle for CASE-TAYLOR-GREEN-SPH.

The maturity gate does not accept a self-declaration ("the case is implemented
and verified"); it wants the artefacts to exist at the declared paths, the
implementation files to be non-placeholder, the callers to really call the code,
and the benchmark numbers to be re-checkable.  This script produces exactly that
and **fails loudly** when any claim does not hold::

    PYTHONPATH=src python benchmarks/taylor_green/evidence.py \
        --run-dir runs/taylor_green/m15860

It writes ``<run-dir>/evidence.json`` with the blocks the gate reads

``impl_files``         implementation files + a non-placeholder audit of each
                       (line count, public functions, no ``NotImplementedError``
                       / ``TODO`` / bare ``pass`` body)
``entry``              executable entry points (module CLI, package CLI,
                       benchmark driver) with the exact command
``run_artifacts``      files produced by a *real* run (results JSON, log,
                       environment, figures), each with size and sha256 so the
                       numbers can be traced to the bytes that produced them
``integration_callers`` files outside the module that import and call it, each
                       verified by grepping the symbol out of the caller source
``benchmark``          ref/got/rel_error/tolerance/pass rows copied from the run
                       (this is the L4 numeric comparison)
``tests``              the pytest node ids that assert the anchors
``repro_cmd``          one command that regenerates the artefacts
``external_anchor``    the citation note + the generated case report

Exit code is 0 only if every path exists, every caller is verified, every
implementation file passes the non-placeholder audit, and every benchmark row
passes its declared tolerance.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CASE = "CASE-TAYLOR-GREEN-SPH"

IMPL_FILES = [
    "src/tensorsph/taylor_green.py",
    "src/tensorsph/vv/tgv.py",
    "src/tensorsph/vv/ladder.py",
]

ENTRIES = [
    {"path": "src/tensorsph/__main__.py", "kind": "package_cli",
     "command": "PYTHONPATH=src python -m tensorsph --case taylor_green "
                "--output <run-dir>"},
    {"path": "src/tensorsph/taylor_green.py", "kind": "module_cli",
     "command": "PYTHONPATH=src python -m tensorsph.taylor_green --out <run-dir>"},
    {"path": "benchmarks/taylor_green/run.py", "kind": "benchmark_driver",
     "command": "PYTHONPATH=src python benchmarks/taylor_green/run.py "
                "--out <run-dir>"},
]

INTEGRATION_CALLERS = [
    {"file": "src/tensorsph/vv/ladder.py", "symbol": "taylor_green",
     "kind": "main_solver",
     "note": "verification ladder rung C2 (_tgv_anchor) calls "
             "taylor_green.run_case, so the ladder, the CLI and the benchmark "
             "gate the same case layer"},
    {"file": "src/tensorsph/__main__.py", "symbol": "taylor_green",
     "kind": "package_entry",
     "note": "python -m tensorsph --case taylor_green dispatches to the case"},
    {"file": "benchmarks/taylor_green/run.py", "symbol": "tensorsph.taylor_green",
     "kind": "benchmark",
     "note": "the standalone driver delegates to the case module"},
    {"file": "tests/test_taylor_green_benchmark.py", "symbol": "taylor_green",
     "kind": "test",
     "note": "the pytest suite asserts the reference checks and the tolerance"},
]

RUN_ARTIFACTS = [
    "results.json",
    "run.log",
    "environment.json",
    "fig1_tgv_velocity_field.png",
    "fig2_tgv_energy_decay.png",
    "fig3_tgv_convergence.png",
    "fig4_tgv_sensitivity.png",
]

TESTS = [
    "tests/test_taylor_green_benchmark.py::test_reference_satisfies_the_navier_stokes_equations",
    "tests/test_taylor_green_benchmark.py::test_reference_decay_rate_is_recovered_from_the_field",
    "tests/test_taylor_green_benchmark.py::test_parameter_table_is_self_consistent",
    "tests/test_taylor_green_benchmark.py::test_case_run_meets_the_eight_percent_tolerance",
    "tests/test_taylor_green_benchmark.py::test_mach_window_breaks_above_0_05",
    "tests/test_taylor_green_benchmark.py::test_committed_run_artifacts_reproduce_the_reference",
    "tests/test_taylor_green_benchmark.py::test_package_cli_writes_results",
]

PLACEHOLDER_MARKERS = ("NotImplementedError", "TODO", "FIXME", "raise NotImplemented")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_impl(path: Path) -> dict:
    """AST audit: substantive implementation, not a placeholder shell."""
    source = path.read_text()
    tree = ast.parse(source)
    public_defs = [n.name for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   and not n.name.startswith("_")]
    empty_bodies = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [b for b in node.body
                    if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
            if not body or all(isinstance(b, ast.Pass) for b in body):
                empty_bodies.append(node.name)
    code_lines = [ln for ln in source.splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    markers = [m for m in PLACEHOLDER_MARKERS if m in source]
    problems = []
    if len(code_lines) <= 40:
        problems.append(f"only {len(code_lines)} code lines")
    if not public_defs:
        problems.append("no public function/class")
    if empty_bodies:
        problems.append(f"empty function bodies: {empty_bodies}")
    if markers:
        problems.append(f"placeholder markers: {markers}")
    return {"path": path.relative_to(REPO).as_posix(),
            "code_lines": len(code_lines),
            "public_defs": len(public_defs),
            "sha256": sha256(path),
            "non_placeholder": not problems,
            "problems": problems}


def check_callers(callers: list) -> tuple:
    verified, problems = [], []
    for row in callers:
        target = REPO / row["file"]
        if not target.exists():
            problems.append(f"{row['file']}: caller file missing")
            continue
        text = target.read_text()
        if row["symbol"] not in text:
            problems.append(f"{row['file']}: symbol {row['symbol']!r} not found")
            continue
        verified.append(dict(row, verified=True))
    return verified, problems


def pytest_node_ids_exist(tests: list) -> list:
    problems = []
    for node in tests:
        path, _, name = node.partition("::")
        target = REPO / path
        if not target.exists():
            problems.append(f"{path}: test file missing")
        elif name and f"def {name}(" not in target.read_text():
            problems.append(f"{node}: test function not found")
    return problems


def build(run_dir: Path) -> dict:
    problems: list = []
    impl = [audit_impl(REPO / p) for p in IMPL_FILES]
    for row in impl:
        if not row["non_placeholder"]:
            problems.append(f"{row['path']}: {row['problems']}")

    entries = []
    for row in ENTRIES:
        exists = (REPO / row["path"]).exists()
        if not exists:
            problems.append(f"entry missing: {row['path']}")
        entries.append(dict(row, exists=exists))

    artifacts = []
    for name in RUN_ARTIFACTS:
        path = run_dir / name
        if not path.exists() or path.stat().st_size == 0:
            problems.append(f"run artifact missing/empty: {path}")
            artifacts.append({"path": str(path), "exists": False})
            continue
        artifacts.append({"path": path.relative_to(REPO).as_posix(),
                          "bytes": path.stat().st_size,
                          "sha256": sha256(path), "exists": True})

    callers, caller_problems = check_callers(INTEGRATION_CALLERS)
    problems += caller_problems
    problems += pytest_node_ids_exist(TESTS)

    results_path = run_dir / "results.json"
    benchmark_rows = []
    if results_path.exists():
        results = json.loads(results_path.read_text())
        benchmark_rows = results["benchmark"]["rows"]
        for row in benchmark_rows:
            for key in ("quantity", "ref", "got", "rel_error_pct", "tolerance_pct"):
                if key not in row:
                    problems.append(f"benchmark row {row.get('quantity')}: no {key}")
            if not row.get("pass", False):
                problems.append(
                    "benchmark row %s fails its tolerance: ref=%r got=%r err=%.3f%% > %.1f%%"
                    % (row["quantity"], row["ref"], row["got"],
                       row["rel_error_pct"], row["tolerance_pct"]))
        if not results["tolerance"]["pass"]:
            problems.append("main case tolerance gate is FAIL")
    else:
        problems.append(f"no results.json in {run_dir}")

    evidence = {
        "case": CASE,
        "repo": "TensorSPH",
        "level_claim": 5,
        "level_claim_note": ("L5 = L4 + external_anchor: the reference is the "
                             "Taylor-Green (1937) closed form, and its "
                             "Navier-Stokes residual is machine-verified "
                             "(round-off level) in the run artifacts"),
        "impl_files": [row["path"] for row in impl],
        "impl_audit": impl,
        "entry": [row["path"] for row in entries],
        "entry_detail": entries,
        "run_artifacts": [row["path"] for row in artifacts],
        "run_artifacts_detail": artifacts,
        "integration_callers": [row["file"] for row in callers],
        "integration_callers_detail": callers,
        "benchmark": {
            "case": CASE,
            "reference": (json.loads(results_path.read_text())["reference"]["citation"]
                          if results_path.exists() else None),
            "rows": benchmark_rows,
            "all_pass": all(r.get("pass", False) for r in benchmark_rows),
        },
        "repro_cmd": ("PYTHONPATH=src python -m tensorsph --case taylor_green "
                      f"--output {run_dir.as_posix()}"),
        "tests": TESTS,
        "external_anchor": ["docs/benchmarks/taylor-green.md",
                            "docs/references/taylor-green-reference.md"],
        "problems": problems,
        "verdict": "PASS" if not problems else "FAIL",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evidence.json").write_text(json.dumps(evidence, indent=2))
    return evidence


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/taylor_green/m15860")
    parser.add_argument("--pytest", action="store_true",
                        help="also run the pytest node ids (slow) and record the "
                             "outcome in the evidence bundle")
    args = parser.parse_args(argv)

    run_dir = REPO / args.run_dir
    evidence = build(run_dir)

    if args.pytest:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS]
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
        evidence["pytest"] = {"command": " ".join(cmd), "returncode": proc.returncode,
                              "stdout_tail": proc.stdout.strip().splitlines()[-12:]}
        if proc.returncode != 0:
            evidence["problems"].append(f"pytest returned {proc.returncode}")
            evidence["verdict"] = "FAIL"
        (run_dir / "evidence.json").write_text(json.dumps(evidence, indent=2))

    print(json.dumps({"verdict": evidence["verdict"],
                      "problems": evidence["problems"],
                      "benchmark_all_pass": evidence["benchmark"]["all_pass"],
                      "artifacts": len(evidence["run_artifacts"]),
                      "callers": len(evidence["integration_callers"]),
                      "evidence_json": str((run_dir / "evidence.json").relative_to(REPO))},
                     indent=2))
    return 0 if evidence["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())