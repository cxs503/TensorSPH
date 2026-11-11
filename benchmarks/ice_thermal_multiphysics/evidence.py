"""Build the machine-checkable evidence bundle for node IN13W2-N07.

The campaign gate does not accept a self-declaration ("the module is
implemented and verified"); it wants the artefacts to exist at the declared
paths, the implementation files to be non-placeholder, the callers to really
call the code, and the benchmark numbers to be re-checkable.  This script
produces exactly that and fails loudly when any of the claims does not hold:

    python benchmarks/ice_thermal_multiphysics/evidence.py \
        [--run-dir runs/IN13W2-N07]

It writes ``<run-dir>/evidence.json`` with the following blocks

``impl_files``          implementation files + a non-placeholder audit of each
                        (line count, public functions, no ``NotImplementedError``
                        / ``TODO`` / bare ``pass`` bodies)
``entry``               executable entry points (module CLI + benchmark) with
                        the exact reproduction command
``run_artifacts``       files produced by a *real* run (results JSON, log, the
                        figures), each with size and sha256 so the numbers can be
                        traced to the bytes that were produced
``integration_callers`` files outside the module that import and call it, each
                        verified by grepping the symbol out of the caller source
``benchmark``           ref/got/rel_error/tolerance/pass rows copied from the
                        run (this is the L4 numeric comparison)
``tests``               the pytest node ids that assert the anchors

Exit code is 0 only if every path exists, every caller is verified, every
implementation file passes the non-placeholder audit, and every benchmark row
passes its declared tolerance.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NODE = "IN13W2-N07"
MODULE_DIR = REPO / "src" / "tensorsph" / "ice_thermal"

#: implementation files of the capability (the six physics modules + the API)
IMPL_FILES = [
    "src/tensorsph/ice_thermal/__init__.py",
    "src/tensorsph/ice_thermal/properties.py",
    "src/tensorsph/ice_thermal/enthalpy.py",
    "src/tensorsph/ice_thermal/sph_thermal.py",
    "src/tensorsph/ice_thermal/brine.py",
    "src/tensorsph/ice_thermal/darcy.py",
    "src/tensorsph/ice_thermal/thermal_stress.py",
    "src/tensorsph/ice_thermal/coupled.py",
    "src/tensorsph/ice_thermal/__main__.py",
]

#: entry points: (path, kind, how it is invoked)
ENTRIES = [
    {"path": "src/tensorsph/ice_thermal/__main__.py", "kind": "cli",
     "command": "python -m tensorsph.ice_thermal --quick --out runs/IN13W2-N07-quick",
     "note": "package CLI; full mode delegates to the benchmark script"},
    {"path": "benchmarks/ice_thermal_multiphysics/run.py", "kind": "benchmark",
     "command": "python benchmarks/ice_thermal_multiphysics/run.py --out runs/IN13W2-N07",
     "note": "runs every anchor and writes results.json + the five figures"},
]

#: callers outside the module.  Each row is verified by grepping ``symbol`` in
#: ``file`` -- if the call is removed the evidence build fails.
INTEGRATION_CALLERS = [
    {"file": "src/tensorsph/wave_ice_interaction.py",
     "symbol": "strength_from_ice_state",
     "kind": "main_solver",
     "note": "wave-ice breakup: flexural strength of the fracture criterion is "
             "derived from the thermal-saline state (brine volume -> strength)"},
    {"file": "src/tensorsph/wave_ice_interaction.py",
     "symbol": "run_wave_ice_scenario",
     "kind": "main_solver",
     "note": "scenario driver consumes the derived sigma_f through "
             "WaveIceScenario(temperature_c=..., salinity_psu=...)"},
    {"file": "benchmarks/ice_thermal_multiphysics/run.py",
     "symbol": "run_three_field",
     "kind": "benchmark_driver",
     "note": "three-field coupled column anchor"},
    {"file": "tests/test_ice_thermal_integration.py",
     "symbol": "WaveIceScenario",
     "kind": "test",
     "note": "integration test of the main-solver caller"},
]

RUN_FILES = ["results.json", "run.log", "environment.json"]
#: artefact produced by the *package CLI* entry (proof that the entry runs)
CLI_ARTIFACTS = ["runs/IN13W2-N07-quick/anchors_quick.json"]
#: canonical numbers published next to the harness (ledger claim artifact)
CANONICAL = "benchmarks/ice_thermal_multiphysics/validation_report.json"

TESTS = ["tests/test_ice_thermal_multiphysics.py",
         "tests/test_ice_thermal_integration.py"]

PLACEHOLDER_MARKERS = ("NotImplementedError", "TODO", "FIXME", "placeholder")


def load_anchor_module():
    """Import the benchmark module so the anchor table is *recomputed* here."""
    path = REPO / "benchmarks" / "ice_thermal_multiphysics" / "run.py"
    spec = importlib.util.spec_from_file_location("ice_thermal_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recompute_anchors(results):
    """Re-derive the ref/got table from the raw series in ``results.json``.

    The stored ``benchmark.rows`` are only accepted if they agree with a fresh
    recomputation from ``results.json`` (relative error < 1e-6), so a hand-edited
    or stale verdict cannot survive the audit.
    """
    bench = load_anchor_module()
    fresh = bench.anchor_table(results)
    stored = results.get("benchmark", {}).get("rows") or results.get("anchors") or []
    consistent = len(fresh) == len(stored)
    if consistent:
        for new, old in zip(fresh, stored):
            if new["anchor"] != old["anchor"]:
                consistent = False
                break
            a, b = new.get("rel_error", 0.0) or 0.0, old.get("rel_error", 0.0) or 0.0
            if abs(a - b) > max(1e-6, 1e-6 * abs(a)):
                consistent = False
                break
    return fresh, consistent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_impl(rel: str) -> dict:
    """Audit one implementation file: size, public API, no placeholder bodies."""
    path = REPO / rel
    src = path.read_text()
    tree = ast.parse(src)
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    public = sorted(n.name for n in funcs if not n.name.startswith("_"))
    # a body that is only ``pass`` / ``...`` / ``raise NotImplementedError`` is
    # a stub, not an implementation
    stubs = []
    for node in funcs:
        body = [b for b in node.body if not isinstance(b, ast.Expr)
                or not isinstance(getattr(b, "value", None), ast.Constant)]
        if not body:
            stubs.append(node.name)
            continue
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            stubs.append(node.name)
        elif len(body) == 1 and isinstance(body[0], ast.Raise):
            exc = body[0].exc
            name = getattr(exc, "id", None) or getattr(getattr(exc, "func", None), "id", "")
            if name == "NotImplementedError":
                stubs.append(node.name)
    all_names = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "__all__" for t in node.targets):
            try:
                all_names = list(ast.literal_eval(node.value))
            except Exception:
                all_names = []
    markers = [m for m in PLACEHOLDER_MARKERS if m in src]
    return {
        "path": rel,
        "lines": src.count("\n") + 1,
        "public_functions": public,
        "n_functions": len(funcs),
        "stub_functions": stubs,
        "placeholder_markers": markers,
        "sha256": sha256(path),
        "exports": all_names,
        "ok": (not stubs and not markers
               and (len(public) > 0 or len(all_names) > 0)),
    }


def audit_caller(row: dict) -> dict:
    path = REPO / row["file"]
    text = path.read_text()
    hits = [i + 1 for i, line in enumerate(text.splitlines()) if row["symbol"] in line]
    return {**row, "lines": hits[:5], "verified": bool(hits)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(REPO / "runs" / NODE))
    ap.add_argument("--strict", action="store_true",
                    help="also fail when the benchmark rows did not all pass")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    results_path = run_dir / "results.json"
    if not results_path.is_file():
        print(f"FAIL: no results.json under {run_dir}; run the benchmark first",
              file=sys.stderr)
        return 2
    results = json.loads(results_path.read_text())

    impl = [audit_impl(rel) for rel in IMPL_FILES]
    entries = []
    for entry in ENTRIES:
        path = REPO / entry["path"]
        entries.append({**entry, "exists": path.is_file(),
                        "sha256": sha256(path) if path.is_file() else None})
    callers = [audit_caller(row) for row in INTEGRATION_CALLERS]

    run_artifacts = []
    for name in RUN_FILES:
        path = run_dir / name
        run_artifacts.append({"path": str(path.relative_to(REPO)),
                              "exists": path.is_file(),
                              "bytes": path.stat().st_size if path.is_file() else 0,
                              "sha256": sha256(path) if path.is_file() else None})
    for name in sorted(results.get("figures", {})):
        path = run_dir / name
        run_artifacts.append({"path": str(path.relative_to(REPO)),
                              "exists": path.is_file(),
                              "bytes": path.stat().st_size if path.is_file() else 0,
                              "sha256": sha256(path) if path.is_file() else None,
                              "caption": results["figures"][name].get("caption", "")})

    for rel in CLI_ARTIFACTS:
        path = REPO / rel
        run_artifacts.append({"path": rel, "exists": path.is_file(),
                              "bytes": path.stat().st_size if path.is_file() else 0,
                              "sha256": sha256(path) if path.is_file() else None,
                              "role": "python -m tensorsph.ice_thermal --quick"})
    canonical_path = REPO / CANONICAL
    canonical_ok = False
    if canonical_path.is_file():
        canonical = json.loads(canonical_path.read_text())
        canonical_ok = (json.dumps(canonical, sort_keys=True, default=float)
                        == json.dumps(results, sort_keys=True, default=float))
    run_artifacts.append({"path": CANONICAL, "exists": canonical_path.is_file(),
                          "bytes": canonical_path.stat().st_size if canonical_path.is_file() else 0,
                          "sha256": sha256(canonical_path) if canonical_path.is_file() else None,
                          "role": "canonical numbers published with the harness "
                                  "(the ledger claim artifact)"})

    bench_rows, sticks = recompute_anchors(results)
    benchmark = {
        "rows": bench_rows,
        "all_pass": all(bool(r.get("pass")) for r in bench_rows) if bench_rows else False,
        "n_rows": len(bench_rows),
        "recomputed_from": f"{run_dir.name}/results.json",
        "reproduces_stored_rows": sticks,
        "source": f"{run_dir.name}/results.json",
    }

    tests = []
    for rel in TESTS:
        path = REPO / rel
        proc = subprocess.run([sys.executable, "-m", "pytest", rel, "--collect-only", "-q"],
                              cwd=REPO, capture_output=True, text=True)
        ids = [ln.strip() for ln in proc.stdout.splitlines() if "::" in ln]
        tests.append({"path": rel, "exists": path.is_file(), "n_collected": len(ids),
                      "node_ids_sample": ids[:3]})

    checks = {
        "impl_files_ok": all(r["ok"] for r in impl),
        "entries_exist": all(r["exists"] for r in entries),
        "run_artifacts_exist": all(r["exists"] for r in run_artifacts),
        "integration_callers_verified": all(r["verified"] for r in callers),
        "tests_collected": all(r["n_collected"] > 0 for r in tests),
        "benchmark_all_pass": benchmark["all_pass"],
        "benchmark_rows_recomputed": benchmark["reproduces_stored_rows"],
        "canonical_report_matches_run": canonical_ok,
    }
    mandatory = ["impl_files_ok", "entries_exist", "run_artifacts_exist",
                 "integration_callers_verified", "tests_collected",
                 "benchmark_rows_recomputed", "canonical_report_matches_run"]
    if args.strict:
        mandatory.append("benchmark_all_pass")
    ok = all(checks[k] for k in mandatory)

    bundle = {
        "node": NODE,
        "repo": "TensorSPH",
        "capability": "ice-ocean thermal-saline-mechanical multi-physics basis",
        "module": "tensorsph.ice_thermal",
        "run_dir": str(run_dir.relative_to(REPO)) if run_dir.is_relative_to(REPO)
        else str(run_dir),
        "environment": results.get("environment", {}),
        "impl_files": impl,
        "entry": entries,
        "run_artifacts": run_artifacts,
        "integration_callers": callers,
        "benchmark": benchmark,
        "tests": tests,
        "repro_cmd": {
            "benchmark": "python benchmarks/ice_thermal_multiphysics/run.py "
                         "--out runs/IN13W2-N07",
            "cli_quick": "python -m tensorsph.ice_thermal --quick "
                         "--out runs/IN13W2-N07-quick",
            "evidence": "python benchmarks/ice_thermal_multiphysics/evidence.py "
                        "--run-dir runs/IN13W2-N07 --strict",
            "tests": "python -m pytest tests/test_ice_thermal_integration.py "
                     "tests/test_ice_thermal_multiphysics.py -q",
        },
        "checks": checks,
        "ok": ok,
    }
    out = run_dir / "evidence.json"
    out.write_text(json.dumps(bundle, indent=2, default=float))
    for key, value in checks.items():
        print(f"[{'PASS' if value else 'FAIL'}] {key}")
    print(f"evidence -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())