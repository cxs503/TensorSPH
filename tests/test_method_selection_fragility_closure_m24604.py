"""Freshness guard for the m24604 closure re-verification of PR#1026 (m21797).

The closure JSON is a snapshot of the evidence chain taken on a specific main.
These tests fail loudly if the m21797 deliverables drift out of the tree again
(a file deleted, or a file changed without re-running the closure driver).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLOSURE_JSON = ROOT / "docs/reports/method_selection_fragility_closure_m24604.json"
FIGURE = ROOT / "docs/figures/method_selection_fragility/closure_m24604.png"
REPORT = ROOT / "docs/dev_report_method_selection_fragility_closure_m24604.md"
DRIVER = ROOT / "benchmarks/method_selection/closure_reverify_m24604.py"


def _payload() -> dict:
    return json.loads(CLOSURE_JSON.read_text())


def test_closure_evidence_exists_and_verdict_all_intact():
    p = _payload()
    assert p["mission"] == 24604
    assert p["verdict"] == "ALL-INTACT"
    assert p["n_missing"] == 0
    assert p["anti_drift_ok"] is True
    assert p["tests_ok"] is True
    assert p["fragility_n_depth_mismatches"] == 0
    assert p["fragility_invariants_ok"] is True


def test_all_eleven_pr1026_files_present_and_hashed():
    p = _payload()
    assert len(p["per_file"]) == 11
    for row in p["per_file"]:
        path = ROOT / row["file"]
        assert path.exists(), f"m21797 deliverable missing from tree: {row['file']}"
        assert row["sha256_main"] == hashlib.sha256(path.read_bytes()).hexdigest(), (
            f"{row['file']} changed since the closure snapshot -- re-run "
            "benchmarks/method_selection/closure_reverify_m24604.py")


def test_registry_and_fragility_record_counts_agree():
    p = _payload()
    reg = json.loads(
        (ROOT / "docs/reports/method_selection_registry.json").read_text())
    assert p["registry_n_records"] == len(reg["records"])
    frag = json.loads(
        (ROOT / "docs/reports/method_selection_fragility_m21797.json").read_text())
    assert p["fragility_n_records"] == frag["n_records"]


def test_closure_artifacts_present():
    assert FIGURE.exists() and FIGURE.stat().st_size > 0
    assert REPORT.exists() and "ALL-INTACT" in REPORT.read_text()
    assert DRIVER.exists()


def test_anti_drift_check_still_green():
    r = subprocess.run(
        [sys.executable, "benchmarks/method_selection/audit_decision_fragility.py",
         "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "anti-drift check OK" in r.stdout
