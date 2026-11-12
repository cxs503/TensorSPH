"""m16257: normalise the audit evidence into the gate's contract and emit the
tensor_maturity_verdicts row.

The gate (maturity_gate.py JUDGE_SCHEME) machine-checks plain path pointers, so the
pointer keys are reduced to path-only values here and the prose moved to *_note keys.
Nothing is hidden: the notes keep the delivery state (branch-only, absent from main)
so the repo-owned path pointers are the honest ones.

Usage: python emit_verdict.py [--apply]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
OUT = WT / "docs/board_audit/m16257"
SCOPE = "m16214 CASE-SPH-FSI-ELASTIC-GATE"
LEVEL = 2
REPO = "TensorSPH"

ev0 = json.loads((OUT / "evidence.json").read_text())
SHA = [ln.split()[0] for ln in (HERE / "artifact_sha256.txt").read_text().splitlines() if ln.strip()]
deliv = ev0["delivery_state"]
bench = ev0["benchmark"]
find = ev0["findings"]

rework = (
    "REWORK (5 items, in order): "
    "1) V1's -20.37% moving-wall load-transfer bias must be fixed at source in "
    "gate_load()/interpolation (missing interior half of the kernel support), then the production "
    "run must use load_scale=1.0; the delivered result is silently multiplied by 1.255762 derived "
    "from that failed check. "
    "2) V2 must be rebuilt as a real pytest unit test: gravity OFF (rod_forces currently adds full "
    "gravity to a 1e-5-scaled load, and the 4 mm gate carries 348x its own Euler load), static "
    "equilibrium by energy minimisation, assert the Green's-function tip inside 10%. "
    "3) The Antoci 2007 gate-tip curve (DOI 10.1016/j.compstruc.2007.01.002, closed access) must be "
    "obtained/digitised and registered in tensor_benchmark_cases for anchor "
    "CASE-SPH-FSI-ELASTIC-GATE with the 10% tolerance; then report ref-vs-got peak AND phase. "
    "4) branch agent/m16214 @9fd55b0 is NOT on GitHub and NOT in main - the whole module exists "
    "only in the local worktree; merge it and keep run artifacts outside .gitignore'd artifacts/. "
    "5) the mandated metric is geometrically saturated (peak 354-365 mm on a 360 mm gate; +25.6% "
    "load moves it -2.9% and the phase not at all) - pick a discriminating metric (pre-floor tip "
    "displacement or gate load history) before claiming any tolerance."
)

ev = {
    "rework_directive": rework,
    "impl_files": ["src/tensorsph/fsi_elastic_gate.py"],
    "impl_files_note": (
        "repo-relative pointer is the honest one: the file is present ONLY in worktree m16214 as "
        "local commit 9fd55b0 on agent/m16214 (git ls-remote origin refs/heads/agent/m16214 -> "
        "empty; git cat-file main:src/tensorsph/fsi_elastic_gate.py -> missing). Byte-identical "
        "audit copy: docs/board_audit/m16257/probe/probe/src/tensorsph/fsi_elastic_gate.py sha256="
        + SHA[1] + " (518 lines, stretch+bending rod via autograd, Adami wall load transfer, "
          "WCSPH reuse; not a placeholder, not a stub)"),
    "entry": "benchmarks/fsi_elastic_gate/run_antoci.py",
    "entry_note": (
        "same branch-only status as impl_files; runs end-to-end (audited: 253 particles/1642 steps "
        "111 s at dx=12.5 mm; 406/2052 steps at dx=10 mm) and is deterministic (this audit's re-run "
        "reproduces peak_tip_x=0.3543333617266935 bit-for-bit); no CLI/API/registry hook, only "
        "sys.path + direct import"),
    "run_artifacts": [
        "/root/tp-agent/work/m16214/artifacts/fsi_elastic_gate/antoci_elastic_gate.json",
        "/root/tp-agent/work/m16214/artifacts/fsi_elastic_gate/antoci_elastic_gate.png",
        "docs/board_audit/m16257/probe/antoci_elastic_gate.json",
        "docs/board_audit/m16257/probe/probe_verify_results.json",
        "docs/board_audit/m16257/probe/probe_load_scale.json",
        "docs/board_audit/m16257/fsi_elastic_gate.png",
        "docs/board_audit/m16257/report.md",
    ],
    "run_artifacts_note": (
        "the delivered json/png live in a .gitignore'd artifacts/ dir (never enters git) and their "
        "mtime 11:26 is 6 min AFTER the mission's finished_at 11:20:15Z; the audit copies are "
        "byte-identical (sha256 in docs/board_audit/m16257/probe/artifact_sha256.txt)"),
    "tests": [],
    "tests_note": (
        "ZERO tests: worktree m16214 tests/ has no test_*fsi* file, no test imports "
        "fsi_elastic_gate, the shipped script asserts nothing (V1/V2 print FAIL and execution "
        "continues) - the L1 rung of the ladder is not just failing, it is unguarded"),
    "integration_callers": [],
    "integration_callers_note": (
        "single call site = its own harness benchmarks/fsi_elastic_gate/run_antoci.py; no other "
        "TensorSPH module, CLI entry or platform tool imports fsi_elastic_gate; nothing is in main, "
        "so integration is structurally impossible to demonstrate at L3"),
    "benchmark": {
        "ref": None,
        "ref_status": "WITHHELD",
        "got_peak_tip_m": bench["got_peak_tip_x_m"],
        "got_peak_time_s": bench["got_peak_time_s"],
        "got_peak_tip_m_fine": bench["got_peak_tip_x_m_fine"],
        "got_peak_time_s_fine": bench["got_peak_time_s_fine"],
        "error_vs_ref_pct": None,
        "tolerance_pct": 10.0,
        "pass": False,
        "registered_in_tensor_benchmark_cases": False,
        "registry_rows_for_anchor": bench["registry_rows_for_anchor"],
        "registry_note": bench["registry_note"],
        "ref_note": bench["ref_note"],
        "self_convergence_pct": bench["self_convergence_pct"],
        "delivered_internal_verification": bench["delivered_internal_verification"],
        "audit_reproduction": bench["audit_reproduction"],
        "load_scale_sensitivity": bench["load_scale_sensitivity"],
        "physics_diagnostics": bench["physics_diagnostics"],
    },
    "external_anchor": None,
    "external_anchor_note": ev0["external_anchor_note"],
    "repro_cmd": ev0["repro_cmd"],
    "delivery_state": deliv,
    "findings": find,
    "verdict_note": ev0["verdict_note"],
}

(OUT / "evidence_gate_contract.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False))


def sql_lit(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


stmt = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
    f"VALUES ({sql_lit(REPO)}, {sql_lit(SCOPE)}, 16257, {LEVEL}, "
    f"{sql_lit(json.dumps(ev, ensure_ascii=False))}::jsonb, 'board-audit') "
    "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET "
    "mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, "
    "evidence = EXCLUDED.evidence, gate_state = 'pending', gate_level = NULL, "
    "gate_note = NULL, created_at = now() RETURNING id, repo, scope, level, verdict_by;"
)
(OUT / "insert_verdict.sql").write_text(stmt + "\n")

if "--apply" in sys.argv:
    r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
                        "-c", stmt], capture_output=True, text=True)
    print(r.stdout or r.stderr)
else:
    print("dry run; SQL at", OUT / "insert_verdict.sql")
mirror = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-tA",
                         "-c", f"SELECT id,scope,level,verdict_by,coalesce(gate_state,'-') "
                               f"FROM tensor_maturity_verdicts WHERE scope={sql_lit(SCOPE)}"],
                        capture_output=True, text=True)
print("rows for scope:", mirror.stdout.strip())