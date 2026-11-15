"""Strict owner-standard enforcement in the method selector (m21537).

The lenient selector answers "does *some* record of this scheme pass?"; the
strict mode answers "is *no* live accuracy record of the schemes put forward
outside 15 %?"  These tests pin both, because the pinned route matrix belongs to
the lenient mode and the platform's claim belongs to the strict one.
"""
from __future__ import annotations

import json
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
REPORT = REPO / "docs/reports/method_selection_strict_bound_m21537.json"
REGISTRY = REPO / "docs/reports/method_selection_registry.json"

BREAKING = dict(acoustic_mach=0.04, has_free_surface=True, gravity_waves=True,
                tensile_stretching=True)
CAPILLARY = dict(acoustic_mach=0.05, has_free_surface=True, capillary_dominated=True)


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def test_report_exists_and_is_recomputed():
    payload = json.loads(REPORT.read_text())
    assert payload["mission"] == 21537
    assert payload["owner_tolerance_pct"] == OWNER_TOLERANCE_PCT
    assert payload["counts"]["profiles"] == 7


def test_strict_mode_never_endorses_a_scheme_above_the_owner_standard(registry):
    payload = json.loads(REPORT.read_text())
    for row in payload["rows"]:
        assert row["strict"]["endorsed_above_tolerance"] == [], row["label"]
        bound = row["strict"]["headline_bound_pct"]
        if row["strict"]["within_owner_standard"]:
            assert bound is not None and bound <= OWNER_TOLERANCE_PCT


def test_lenient_routes_are_unchanged(registry):
    """strict_bound must be opt-in: the default route matrix is pinned."""
    payload = json.loads(REPORT.read_text())
    for row in payload["rows"]:
        assert row["lenient"]["primary"] == row["expected_primary"], row["label"]


def test_breaking_wave_drops_the_out_of_bound_modifier(registry):
    lenient = select_method(CaseProfile.from_mapping(BREAKING), registry=registry)
    strict = select_method(CaseProfile.from_mapping(BREAKING), registry=registry,
                           strict_bound=True)
    assert lenient.within_owner_standard is False          # 16.96 % headline
    assert strict.within_owner_standard is True
    assert strict.headline_bound_pct <= OWNER_TOLERANCE_PCT
    assert "transport_velocity_correction" in [d.method for d in lenient.modifiers]
    assert "transport_velocity_correction" not in [d.method for d in strict.modifiers]
    excluded = {d.method: d for d in strict.strict_excluded}
    assert "transport_velocity_correction" in excluded
    assert excluded["transport_velocity_correction"].worst_accuracy_pct > OWNER_TOLERANCE_PCT
    assert any("strict bound" in note and "transport_velocity_correction" in note
               for note in strict.notes)


def test_capillary_profile_is_withheld_not_substituted(registry):
    """A rule that covers the regime must not be replaced by an unrelated one."""
    strict = select_method(CaseProfile.from_mapping(CAPILLARY), registry=registry,
                           strict_bound=True)
    assert strict.primary is None
    assert strict.within_owner_standard is False
    assert any("no later rule is substituted" in note for note in strict.notes)
    assert strict.modifiers == [], "no add-on can be attached to a withheld primary"
    card = render_decision(strict)
    assert "csf_surface_tension" in card
    assert "excluded csf_surface_tension" in card


def test_strict_excluded_always_carries_its_disqualifying_record(registry):
    strict = select_method(CaseProfile.from_mapping(BREAKING), registry=registry,
                           strict_bound=True)
    for decision in strict.strict_excluded:
        assert decision.worst_accuracy_pct is not None
        assert decision.worst_accuracy_pct > OWNER_TOLERANCE_PCT
        assert decision.accuracy_within_owner_standard is not True
        record = decision.worst_accuracy_record
        assert record is not None and record.quantity


def test_as_dict_exposes_strict_bookkeeping(registry):
    strict = select_method(CaseProfile.from_mapping(BREAKING), registry=registry,
                           strict_bound=True)
    payload = strict.as_dict()
    assert payload["strict_bound"] is True
    assert payload["strict_excluded"], "excluded schemes must be machine readable"
    entry = payload["strict_excluded"][0]
    assert {"method", "role", "worst_accuracy_pct", "worst_accuracy_case"} <= set(entry)
