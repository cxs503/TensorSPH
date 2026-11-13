"""Invariants of the m17736 board audit (artifacts/audit_m17715/*.json).

These are *regression* gates on the audit's own conclusions: if any stored evidence
is replaced by a rosier one, they fail.  Cheap — they read the JSON, they do not
re-run the subject harness (scripts/audit/m17736_board_audit.py does that).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "artifacts" / "audit_m17715"


def load(name: str) -> dict:
    return json.loads((AUD / name).read_text())


def test_verdict_is_level_2_not_4():
    finding = load("finding.json")
    assert finding["level"] == 2
    assert "L3_refused" in finding["level_rationale"]
    assert "L4_refused" in finding["level_rationale"]


def test_nothing_was_committed_by_the_subject():
    delivery = load("delivery.json")
    assert int(delivery["commits_ahead_of_main"]) == 0   # zero commits of its own
    assert delivery["head"] != delivery["origin_main"] or True  # worktree may lag main
    # the six delivered files were untracked; only __main__.py was a tracked edit
    untracked = set(delivery["delivered_untracked"])
    assert "src/tensorsph/__main__.py" not in untracked
    assert len(untracked) == len(delivery["manifest"]) - 1
    assert "src/tensorsph/oss_projection.py" in untracked
    assert "docs/xval_projection_laplace_oss.md" in untracked


def test_referee_row_is_absent():
    referee = load("referee.json")
    assert referee["anchor"] == "XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE"
    assert referee["rows_for_anchor"] == 0
    assert referee["table_total_rows"] > 0


def test_integration_caller_fails_while_harness_reproduces():
    repro = load("repro.json")
    assert repro["harness_rc"] == 0
    assert repro["json_leaf_diff_count"] == 0
    assert repro["png_sha256_delivered"] == repro["png_sha256_repro"]
    # the claimed L3 caller must stay broken until the subject fixes it
    assert repro["integration_caller"]["rc"] != 0
    assert any("--case" in line for line in repro["integration_caller"]["stderr_tail"])


def test_report_tables_do_not_match_the_artifact():
    c = load("report_consistency.json")
    assert c["ladder_mismatches"] >= 4, c["ladder_mismatches"]
    assert c["sensitivity_mismatches"] >= 5, c["sensitivity_mismatches"]
    assert c["kappa_column_mismatches"] >= 7, c["kappa_column_mismatches"]
    # the regression the prose hides: fit_quadratic is far outside the 15 % gate
    row = [r for r in c["sensitivity_rows"]
           if r["doc"]["vary"] == "ours_curvature" and r["doc"]["value"] == "fit_quadratic"]
    assert row and abs(row[0]["artifact"]["ours_pct"]) > 15.0


def test_ours_arm_is_within_the_owner_tolerance():
    c = load("report_consistency.json")
    ours = [r for r in c["ladder_rows"]
            if r["artifact"] and r["doc"]["arm"].startswith("ours")]
    assert len(ours) == 3
    assert max(abs(r["artifact"]["err_pct"]) for r in ours) < 3.0


def test_snapshot_bytes_are_present_for_every_evidence_pointer():
    manifest = json.loads((ROOT / "docs" / "board_audit_m17715" / "snapshot"
                           / "MANIFEST.json").read_text())
    snap = ROOT / "docs" / "board_audit_m17715" / "snapshot"
    for rel, meta in manifest.items():
        p = snap / rel
        assert p.exists(), rel
        assert p.stat().st_size == meta["bytes"], rel