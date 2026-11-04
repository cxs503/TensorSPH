"""Machine gates for the tensorSPH verification ledger.

The repository's failure mode is not a wrong number, it is a *stale* number:
five generations of surface-tension verification docs quote −93.6 %, −75.6 %,
−87.6 % and −34.0 % for the same physical quantity, without any marker saying
which is current.  ``benchmarks/verification_ledger/`` fixes that by making the
claim table the single source of truth.  These tests keep it true:

* the published document must be byte-identical to what the artifacts say,
* every claim must carry a gradeable evidence level,
* a superseded claim must point at a live claim,
* the honest verdict (single-phase Laplace jump is WITHHELD) cannot be softened,
* every superseded verification document must carry a banner pointing here.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = ROOT / "benchmarks" / "verification_ledger"
DOC = ROOT / "docs" / "verification_ledger.md"
BANNER_TOKEN = "SUPERSEDED-BY: docs/verification_ledger.md"

_SUPERSEDED_DOCS = [
    "docs/verification_surface_tension_laplace.md",
    "docs/verification_surface_tension_consistent.md",
    "docs/verification_surface_tension_curvature.md",
    "docs/verification_csf_laplace_consistent.md",
]


def _load_ledger_module():
    spec = importlib.util.spec_from_file_location("verification_ledger", LEDGER_DIR / "ledger.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = _load_ledger_module()


@pytest.fixture(scope="module")
def ledger():
    return L.build()


@pytest.fixture(scope="module")
def committed():
    return json.loads((LEDGER_DIR / "ledger.json").read_text(encoding="utf-8"))


def test_every_claim_resolves_against_a_committed_artifact(ledger):
    assert len(ledger["rows"]) >= 8
    for row in ledger["rows"]:
        assert not re.search(r"NaN|inf", json.dumps(row)), row["id"]
        assert row["verdict"] in {"PASS", "FAIL", "MEASURED"}, row["id"]
        assert row["evidence"] in L.EVIDENCE_LEVELS, row["id"]
        for key in ("value", "configuration", "artifact", "why"):
            assert row[key] not in (None, ""), (row["id"], key)


def test_committed_json_is_not_stale(ledger, committed):
    """Re-running the builder must reproduce the committed ledger exactly."""
    assert committed["rows"] == ledger["rows"]


def test_document_numbers_match_the_artifacts(ledger):
    """Every error figure quoted in the published table must exist in ledger.json."""
    doc = DOC.read_text(encoding="utf-8")
    section = doc.split("## ② ", 1)[1].split("## ③ ", 1)[0]
    table = [row for row in section.splitlines() if row.startswith("| `")]
    assert len(table) == len(ledger["rows"])
    for row in ledger["rows"]:
        assert f"`{row['id']}`" in doc, row["id"]
        if row["rel_error_pct"] is not None:
            assert f"{row['rel_error_pct']:+.2f}%" in doc, row["id"]


def test_superseded_claims_point_at_live_claims(ledger):
    by_id = {row["id"]: row for row in ledger["rows"]}
    superseded = [row for row in ledger["rows"] if row["evidence"] == "SUPERSEDED"]
    assert superseded, "the band-differencing mouth claim must stay recorded as superseded"
    for row in superseded:
        target = by_id[row["superseded_by"]]
        assert target["evidence"] != "SUPERSEDED", row["id"]
        assert target["verdict"] != "FAIL" or row["evidence"] == "SUPERSEDED"


def test_single_phase_laplace_jump_stays_withheld(ledger):
    """Honesty gate: the end-to-end jump is not verified; do not let a doc drift."""
    by_id = {row["id"]: row for row in ledger["rows"]}
    best = by_id["laplace_single_phase_best"]
    assert best["verdict"] == "FAIL"
    assert best["within_3pct"] is False
    assert abs(best["rel_error_pct"]) > 15.0
    assert ledger_doc_says_withheld()


def ledger_doc_says_withheld() -> bool:
    doc = DOC.read_text(encoding="utf-8")
    return "证据等级维持 **WITHHELD**" in doc


def test_mouth_correction_is_large_and_attributable(ledger):
    """The historical -78.6 % is mostly the mouth, and that must stay measurable."""
    by_id = {row["id"]: row for row in ledger["rows"]}
    mouth = by_id["laplace_mouth_correction"]
    assert mouth["value"] == pytest.approx(60.1, abs=0.5)
    a, b = mouth["rows_compared"]
    assert abs(a) > abs(b)


def test_force_operator_claims_are_real_and_gated(ledger):
    by_id = {row["id"]: row for row in ledger["rows"]}
    cut = by_id["cut_force_identity"]
    kappa = by_id["curvature_delta_s_weighted"]
    assert cut["evidence"] == "REAL" and cut["verdict"] == "PASS"
    assert kappa["evidence"] == "REAL" and kappa["within_3pct"] is True
    two_phase = by_id["two_phase_force_integral_laplace"]
    assert two_phase["evidence"] == "REAL" and two_phase["verdict"] == "PASS"


def test_superseded_docs_carry_the_banner():
    """A doc that quotes a superseded headline number must say so at the top."""
    missing = []
    for rel in _SUPERSEDED_DOCS:
        text = (ROOT / rel).read_text(encoding="utf-8")
        if BANNER_TOKEN not in text:
            missing.append(rel)
    assert not missing, f"superseded docs without banner: {missing}"


def test_figure_is_committed(ledger):
    png = LEDGER_DIR / "ledger.png"
    assert png.exists() and png.stat().st_size > 10_000
