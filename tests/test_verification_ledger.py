"""Guards for ``docs/verification_ledger.md`` (the evidence ledger).

The ledger exists to stop stale numbers from being cited as current precision.
These tests are what makes it trustworthy: the committed document must be
byte-identical to a fresh render, every number in it must come from a committed
JSON artifact, every historical quote must still be present in the document that
makes it, and the set of drifted quotes is pinned so a silent fix (or a silent
new drift) fails the suite.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_LEDGER_PATH = ROOT / "benchmarks" / "evidence_ledger" / "ledger.py"


def _load_ledger():
    spec = importlib.util.spec_from_file_location("evidence_ledger", _LEDGER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = _load_ledger()


@pytest.fixture(scope="module")
def built():
    return L.build()


def test_committed_document_is_in_sync():
    """Regenerating must reproduce the committed file exactly (no hand edits)."""
    assert L.DOC.exists(), "docs/verification_ledger.md is missing"
    assert L.DOC.read_text() == L.render(), (
        "docs/verification_ledger.md is out of date; run "
        "`PYTHONPATH=src python benchmarks/evidence_ledger/ledger.py`")


def test_every_artifact_row_reads_a_committed_json():
    for entry in L.ARTIFACTS:
        path = ROOT / entry["json"]
        assert path.is_file(), f"{entry['json']} missing"
        doc = ROOT / entry["doc"]
        assert doc.is_file(), f"{entry['doc']} missing"


def test_error_column_is_derived_not_typed(built):
    """Where both values exist, the quoted error follows from measured/reference."""
    for row in built["artifacts"]:
        if row["measured"] is None or row["reference"] in (None, 0):
            continue
        recomputed = (row["measured"] - row["reference"]) / row["reference"] * 100.0
        assert row["error_pct"] == pytest.approx(recomputed, abs=1e-6), row["key"]
        assert row["grade"] == L.grade(row["error_pct"]), row["key"]


def test_grade_thresholds_match_the_platform_standard():
    assert L.grade(0.0) == "REAL (<3 %)"
    assert L.grade(-3.0) == "REAL (<3 %)"
    assert L.grade(3.01) == "REAL (<15 %)"
    assert L.grade(15.0) == "REAL (<15 %)"
    assert L.grade(-15.01) == "WITHHELD (>15 %)"
    assert L.grade(220.0) == "WITHHELD (>15 %)"


def test_historical_quotes_still_exist_in_their_documents(built):
    """A quote the ledger talks about must really be in the document."""
    for claim in built["claims"]:
        assert claim["present_in_doc"], (claim["doc"], claim["quote"])


def test_band_readout_is_never_graded_as_current_precision(built):
    """Every band-definition row is WITHHELD: the definition costs ~55 points."""
    for row in built["artifacts"]:
        if "band read-out" not in row["definition"].lower():
            continue
        assert row["grade"].startswith("WITHHELD"), row["key"]
    cost = L.readout_cost()
    assert cost["n_runs"] >= 4
    assert cost["cost_mean_pct"] < -30.0, cost


def test_drift_register_is_exactly_what_the_artifacts_say(built):
    """Pinned drift set: a repair (or a new drift) must update this test."""
    drifted = sorted(claim["doc"] for claim in built["claims"] if claim["drifted"])
    assert drifted == ["docs/verification_surface_tension_consistent.md"], drifted


def test_unmapped_quote_is_reported_not_guessed(built):
    unmapped = [claim for claim in built["claims"] if not claim["mapped"]]
    assert len(unmapped) == 1
    assert unmapped[0]["quote"] == "-93.6%"
    assert unmapped[0]["artifact"] is None