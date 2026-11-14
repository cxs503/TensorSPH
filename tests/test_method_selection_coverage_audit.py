"""Tests for the evidence-coverage audit of the method-selection registry.

Anchor ``XVAL-SPH-METHOD-SELECTION-2026-11-12``, mission 20432.

What these tests lock down (each one can fail on its own):

* every quantity an underlying artefact gates is banked as a registry record
  (no metric shopping: a scheme may not be quoted on the kindest quantity of a
  benchmark while a gated quantity of the same run is 5x worse),
* every banked percentage is reproduced by an *independent* recomputation from
  the raw artefact (the audit does not import the registry builder),
* the decision card states the difference between the claimed bound and the
  scheme's worst banked record instead of presenting the claim as a bound,
* gates an artefact declares but never measures stay enumerated as data.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tensorsph.method_selection import (
    OWNER_TOLERANCE_PCT,
    CaseProfile,
    load_registry,
    render_decision,
    select_method,
)

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "benchmarks" / "method_selection" / "audit_quantity_coverage.py"
OUT_JSON = REPO / "docs" / "reports" / "method_selection_coverage_audit_m20432.json"
OUT_MD = REPO / "docs" / "reports" / "method_selection_coverage_audit_m20432.md"
REGISTRY_JSON = REPO / "docs" / "reports" / "method_selection_registry.json"

COUETTE = CaseProfile(acoustic_mach=0.01, reynolds=1.0, domain="closed")


def _audit_module():
    spec = importlib.util.spec_from_file_location("quantity_coverage_audit", AUDIT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audit_documents_match_regeneration():
    """Anti-drift: the committed audit JSON/MD are pure recomputations."""
    module = _audit_module()
    result = module.audit()
    committed = json.loads(OUT_JSON.read_text())
    assert result == committed, "audit drifted; rerun audit_quantity_coverage.py"
    assert module.render_markdown(result) == OUT_MD.read_text()
    assert module.main(["--check"]) == 0


def test_every_gated_quantity_is_banked():
    """No metric shopping: coverage of artefact-gated quantities is complete."""
    result = json.loads(OUT_JSON.read_text())
    assert result["gated_quantities"] >= 10
    assert result["coverage_complete"] is True
    assert result["unbanked_gated_quantities"] == []
    for row in result["quantities"]:
        if row["gate_pct"] is not None:
            assert row["banked"], row["extractor"]


def test_banked_numbers_equal_independent_recomputation():
    result = json.loads(OUT_JSON.read_text())
    assert result["integrity_failures"] == []
    assert result["stored_vs_recomputed_mismatches"] == []
    direct = [row for row in result["quantities"] if row["recompute_kind"] == "direct"]
    assert len(direct) >= 14, "the audit must recompute from raw pairs, not stored %"
    for row in result["quantities"]:
        if row["banked"]:
            assert row["integrity_ok"] is True, row["extractor"]


def test_wall_shear_is_banked_and_flagged_over_the_owner_standard():
    """m20432 finding: the no-slip scheme's wall shear is 32.6 % off, not 5.7 %."""
    audit = json.loads(OUT_JSON.read_text())
    row = next(r for r in audit["quantities"]
               if r["extractor"] == "x_couette_wall_shear")
    assert row["banked"] is True
    assert row["recomputed_pct"] == pytest.approx(32.65, abs=0.05)
    assert row["within_owner_tolerance"] is False
    assert row["own_gate_passed"] is False
    assert "x_couette_wall_shear" in audit["records_over_owner_tolerance"]

    registry = load_registry(REGISTRY_JSON)
    wall = [r for r in registry["wcsph_wall_no_slip"]
            if r.quantity.startswith("wall-shear error")]
    assert wall and wall[0].measured_pct > OWNER_TOLERANCE_PCT


def test_decision_card_separates_claim_from_worst_banked_record():
    registry = load_registry(REGISTRY_JSON)
    selection = select_method(COUETTE, registry=registry)
    fallback = next(d for d in selection.candidates
                    if d.method == "wcsph_wall_no_slip")
    assert fallback.worst_measured_pct is not None
    assert fallback.worst_measured_pct <= OWNER_TOLERANCE_PCT  # the claim
    assert fallback.worst_banked_pct > OWNER_TOLERANCE_PCT      # the truth
    # and the routing note must carry the failing quantity, not only the first one
    note = next(n for n in selection.notes if "wcsph_wall_no_slip" in n)
    assert "32.6" in note

    text = render_decision(
        select_method(CaseProfile(has_free_surface=True), registry={
            "wcsph_delta_sph": [
                type(fallback.evidence[0])(
                    method="wcsph_delta_sph", case="synthetic", quantity="q",
                    measured_pct=3.0, gate_pct=5.0, source="s.json"),
                type(fallback.evidence[0])(
                    method="wcsph_delta_sph", case="synthetic-bad", quantity="q2",
                    measured_pct=40.0, gate_pct=5.0, source="s.json"),
            ]
        })
    )
    assert "claimed bound: 3%" in text
    assert "worst banked record: 40%" in text


def test_declared_but_unmeasured_gates_stay_enumerated():
    audit = json.loads(OUT_JSON.read_text())
    gates = {gate["gate"]: gate for gate in audit["unbacked_gates"]}
    assert "steady_drift" in gates and "control_flux" in gates
    assert all(gate["why"] for gate in audit["unbacked_gates"])
    verdicts = {row["artifact"]: row["own_verdict"] for row in audit["artefact_verdicts"]}
    assert verdicts["benchmarks/couette/out/results.json"] == "FAIL"