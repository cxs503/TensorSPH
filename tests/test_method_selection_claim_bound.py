"""The advertised accuracy bound must be a bound, not a best case.

Mission m21115 (anchor ``XVAL-SPH-METHOD-SELECTION-2026-11-12``) found three
gaps in the evidence-gated selector and pins each of them here:

1. ``worst_measured_pct`` -- the number the decision card calls a "claimed
   bound" -- is a maximum over the records that *individually* pass 15 %, so a
   scheme with one favourable record and a 23.8 % one is still advertised as
   "5.99 %".  The public registry really contains three such schemes
   (``csf_surface_tension``, ``transport_velocity_correction``,
   ``wcsph_wall_no_slip``).  ``Selection.headline_bound_pct`` is the companion
   that cannot be undercut: it is the worst live *accuracy* record over every
   scheme the selection puts forward.
2. A conditioning record (``kind="robustness"``, e.g. how far a wall-shear
   reading moves across a viscous-time sweep) is not an error against a
   reference.  It used to be able to satisfy the owner standard -- one such
   record reads 0.0 % -- and could therefore rescue a scheme whose real error
   records fail.
3. ``build_registry.build_matrix`` rebuilt ``Evidence`` objects without
   ``status``/``precondition``/``kind``, so the *published* route matrix
   re-banked refuted records and mixed conditioning into accuracy.  The matrix
   must be the decision it claims to show.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tensorsph.method_selection import (
    EVIDENCE_ROBUSTNESS,
    OWNER_TOLERANCE_PCT,
    CaseProfile,
    Decision,
    Evidence,
    Selection,
    load_registry,
    select_method,
)

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "benchmarks" / "method_selection" / "build_registry.py"
REGISTRY_JSON = REPO / "docs" / "reports" / "method_selection_registry.json"
MATRIX_JSON = REPO / "docs" / "reports" / "method_selection_m18056.json"


def _builder():
    spec = importlib.util.spec_from_file_location("ms_build_registry", BUILDER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


def _ev(method="csf_surface_tension", measured=5.0, gate=15.0, status="banked",
        kind="accuracy", case="CASE-X", quantity="q"):
    return Evidence(method=method, case=case, quantity=quantity,
                    measured_pct=measured, gate_pct=gate, source="synthetic",
                    status=status, kind=kind)


# --------------------------------------------------------------------------- #
# 1. the headline bound cannot be undercut by a favourable record
# --------------------------------------------------------------------------- #
def test_headline_bound_covers_every_endorsed_accuracy_record():
    registry = {
        "csf_surface_tension": [_ev(measured=5.0), _ev(measured=23.8)],
        # a modifier that rides along on the same recommendation
        "density_diffusion": [_ev(method="density_diffusion", measured=1.02)],
    }
    profile = CaseProfile(acoustic_mach=0.05, has_free_surface=True,
                          capillary_dominated=True)
    selection = select_method(profile, registry=registry)
    assert selection.primary is not None
    assert selection.primary.method == "csf_surface_tension"
    # the old advertisement, unchanged and still visible for the audit trail
    assert selection.primary.worst_measured_pct == pytest.approx(5.0)
    # the bound
    assert selection.primary.worst_accuracy_pct == pytest.approx(23.8)
    assert selection.headline_bound_pct == pytest.approx(23.8)
    assert selection.is_verified is True           # a scheme was selected ...
    assert selection.within_owner_standard is False  # ... but it is not <15 %
    payload = selection.as_dict()
    assert payload["within_owner_standard"] is False
    assert payload["headline_bound_pct"] == pytest.approx(23.8)
    assert payload["headline_quantity"].startswith("csf_surface_tension:")


def test_headline_bound_includes_modifiers_not_only_the_primary():
    registry = {
        "wcsph_delta_sph": [_ev(method="wcsph_delta_sph", measured=2.97)],
        "transport_velocity_correction": [
            _ev(method="transport_velocity_correction", measured=10.81),
            _ev(method="transport_velocity_correction", measured=16.96),
        ],
    }
    profile = CaseProfile(acoustic_mach=0.04, has_free_surface=True,
                          gravity_waves=True, tensile_stretching=True)
    selection = select_method(profile, registry=registry)
    assert selection.primary is not None and selection.primary.method == "wcsph_delta_sph"
    assert selection.primary.worst_accuracy_pct == pytest.approx(2.97)
    modifiers = {d.method: d for d in selection.modifiers}
    assert "transport_velocity_correction" in modifiers
    assert selection.headline_bound_pct == pytest.approx(16.96)
    assert selection.within_owner_standard is False


def test_headline_bound_ignores_refuted_records():
    registry = {"csf_surface_tension": [
        _ev(measured=5.0), _ev(measured=40.0, status="refuted")]}
    profile = CaseProfile(acoustic_mach=0.05, has_free_surface=True,
                          capillary_dominated=True)
    selection = select_method(profile, registry=registry)
    assert selection.headline_bound_pct == pytest.approx(5.0)
    assert selection.within_owner_standard is True


def test_selection_without_selected_scheme_is_never_within_standard():
    selection = select_method(CaseProfile(thermal_transport=True), registry={})
    assert selection.primary is None
    assert selection.is_verified is False
    assert selection.within_owner_standard is False


# --------------------------------------------------------------------------- #
# 2. conditioning records are not accuracy evidence
# --------------------------------------------------------------------------- #
def test_robustness_record_cannot_satisfy_the_owner_standard():
    """A 0 % *conditioning* record used to count as a passing error record."""
    registry = {"wcsph_wall_no_slip": [
        _ev(method="wcsph_wall_no_slip", measured=20.0, gate=5.0),
        _ev(method="wcsph_wall_no_slip", measured=0.0, gate=5.0,
            kind=EVIDENCE_ROBUSTNESS, case="reading mobility"),
    ]}
    profile = CaseProfile(acoustic_mach=0.05, reynolds=50.0, domain="closed")
    selection = select_method(profile, registry=registry)
    decisions = {d.method: d for d in selection.candidates}
    wall = decisions["wcsph_wall_no_slip"]
    assert wall.status == "FAILED", (
        "a scheme whose only accuracy record is 20 % must not be selectable, "
        "however stable its conditioning record is")
    assert wall.worst_accuracy_pct == pytest.approx(20.0)
    assert wall.worst_robustness_pct == pytest.approx(0.0)


def test_scheme_with_only_conditioning_records_is_withheld():
    registry = {"wcsph_wall_no_slip": [
        _ev(method="wcsph_wall_no_slip", measured=0.0, kind=EVIDENCE_ROBUSTNESS)]}
    profile = CaseProfile(acoustic_mach=0.05, reynolds=50.0, domain="closed")
    selection = select_method(profile, registry=registry)
    wall = {d.method: d for d in selection.candidates}["wcsph_wall_no_slip"]
    assert wall.status == "WITHHELD"
    assert wall.worst_accuracy_pct is None
    assert selection.headline_bound_pct is None


def test_evidence_rejects_unknown_kind():
    with pytest.raises(ValueError, match="evidence kind"):
        _ev(kind="guess")


# --------------------------------------------------------------------------- #
# 3. the published matrix must be the decision
# --------------------------------------------------------------------------- #
def test_matrix_propagates_refuted_status_instead_of_rebanking():
    builder = _builder()
    registry = {
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "mission": 21115,
        "records": [{
            "method": "wcsph_wall_no_slip", "case": "plane Couette channel",
            "quantity": "wall-shear error", "measured_pct": 32.65,
            "gate_pct": 5.0, "source": "synthetic", "source_sha256": "",
            "status": "refuted", "precondition": "startup transient",
            "kind": "accuracy",
        }],
    }
    matrix = builder.build_matrix(registry)
    row = next(r for r in matrix["profiles"] if r["label"] == "planar Couette channel")
    wall = next(d for d in row["candidates"] if d["method"] == "wcsph_wall_no_slip")
    assert wall["status"] == "WITHHELD", (
        "the matrix dropped the record status and re-banked a refuted record")
    assert row["primary_method"] != "wcsph_wall_no_slip"


def test_matrix_propagates_kind_so_conditioning_is_not_an_error():
    builder = _builder()
    registry = {
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "mission": 21115,
        "records": [{
            "method": "wcsph_wall_no_slip", "case": "reading mobility",
            "quantity": "mobility", "measured_pct": 18.26, "gate_pct": 5.0,
            "source": "synthetic", "source_sha256": "", "status": "banked",
            "precondition": "", "kind": "robustness",
        }],
    }
    matrix = builder.build_matrix(registry)
    row = next(r for r in matrix["profiles"] if r["label"] == "planar Couette channel")
    assert row["primary_accuracy_pct"] in (None,)   # no accuracy record at all
    assert row["within_owner_standard"] is False


def test_published_registry_marks_the_couette_mobility_records_as_robustness():
    reg = load_registry(REGISTRY_JSON)
    wall = reg["wcsph_wall_no_slip"]
    robustness = [ev for ev in wall if ev.kind == EVIDENCE_ROBUSTNESS]
    assert len(robustness) == 2
    assert all("mobility" in ev.quantity for ev in robustness)
    # and they no longer decide the scheme's accuracy bound
    decision = Decision(method="wcsph_wall_no_slip", status="SELECTED",
                        role="primary", rationale="", evidence=wall)
    assert decision.worst_accuracy_pct == pytest.approx(17.076, abs=1e-3)
    assert decision.worst_robustness_pct == pytest.approx(18.255, abs=1e-3)


def test_published_matrix_is_not_stale_and_flags_routes_outside_the_standard():
    builder = _builder()
    committed = json.loads(MATRIX_JSON.read_text())
    fresh = builder.build_matrix(json.loads(REGISTRY_JSON.read_text()))
    assert committed["profiles"] == fresh["profiles"], (
        "docs/reports/method_selection_m18056.json is stale: rerun "
        "benchmarks/method_selection/build_registry.py")
    by_label = {row["label"]: row for row in committed["profiles"]}
    # primary inside 15 %, but an attached modifier is not -> the whole
    # recommendation is outside the owner standard (2 of the 6 verified routes)
    for label in ("capillary droplet (Laplace law)", "breaking wave over a slope"):
        assert by_label[label]["verified"] is True
        assert by_label[label]["within_owner_standard"] is False
        assert by_label[label]["headline_bound_pct"] > OWNER_TOLERANCE_PCT
    inside = [row for row in committed["profiles"] if row["within_owner_standard"]]
    assert len(inside) == 4