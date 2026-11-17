"""Closure tests for the re-delivered m21677 footprint deliverables (m24140).

The original PR #1020 was ruled closed (owner 2026-11-05); its 13 files reached
main through other merges.  These tests pin the "see-it-through" closure:

* all 13 PR #1020 deliverables exist in the worktree;
* the committed footprint ledger still matches a fresh recomputation from the
  raw ensemble artefacts (verdict + per-grid obs/synth/residual/judgeable);
* the generated closure document exists, carries the committed ledger's
  sha256 digest, and states the recomputed verdict.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"
ART = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_footprint"
DOC = ROOT / "docs" / "case_sph_wedge_entry_2d_footprint_closure_m24140.md"

sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT / "src"))

import make_closure_doc as mcd  # noqa: E402
import footprint_ledger as fl  # noqa: E402


def test_all_thirteen_pr1020_deliverables_are_present():
    missing = [f for f in mcd.PR1020_FILES if not (ROOT / f).exists()]
    assert len(mcd.PR1020_FILES) == 13
    assert not missing, "missing PR#1020 deliverables: %s" % missing


def test_committed_ledger_matches_fresh_recomputation():
    committed = json.loads((ART / "footprint_ledger.json").read_text())
    fresh = fl.build(ART)
    mcd.check_ledger(committed, fresh)
    # the verdict itself must be the honest one, not a hand-tuned pass
    assert fresh["verdict"]["verdict"] == committed["verdict"]["verdict"]
    assert fresh["verdict"]["verdict"] in (
        "PASS", "FAIL_above", "FAIL_below", "NOT_DECIDABLE")


def test_closure_doc_is_generated_and_consistent():
    assert DOC.exists(), "run make_closure_doc.py to generate the closure doc"
    text = DOC.read_text()
    ledger_sha = hashlib.sha256(
        (ART / "footprint_ledger.json").read_bytes()).hexdigest()
    assert ledger_sha in text, "closure doc digest does not match the ledger"
    fresh = fl.build(ART)
    assert fresh["verdict"]["verdict"] in text
    # the doc must not claim a 3 % agreement it does not have
    assert "no key quantity is within 3 % of its reference" in text
    # and the figure referenced by the doc must exist
    import re
    m = re.search(r"!\[closure figure\]\(([^)]+)\)", text)
    assert m, "closure figure reference missing"
    assert (ROOT / m.group(1)).exists()


def test_closure_figure_exists():
    assert mcd.FIG.exists(), "closure figure missing -- run make_closure_doc.py"
    assert mcd.FIG.stat().st_size > 10_000, "closure figure suspiciously small"
