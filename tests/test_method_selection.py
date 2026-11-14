"""Tests for :mod:`tensorsph.method_selection` and its evidence registry.

The point of these tests is the *gate*, not the routing table: a scheme must not
be selectable unless a machine-readable benchmark record backs it, and the
registry must be a pure recomputation of committed artefacts (so a hand-edited
number is caught here instead of in review).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tensorsph.method_selection import (
    METHODS,
    MODIFIERS,
    OWNER_TOLERANCE_PCT,
    CaseProfile,
    Evidence,
    load_registry,
    render_decision,
    select_method,
)

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "benchmarks" / "method_selection" / "build_registry.py"
REGISTRY_JSON = REPO / "docs" / "reports" / "method_selection_registry.json"
MATRIX_JSON = REPO / "docs" / "reports" / "method_selection_m18056.json"


def _builder():
    spec = importlib.util.spec_from_file_location("method_selection_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve annotations through sys.modules, so register first
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _evidence(method: str, measured: float, gate: float = 5.0) -> Evidence:
    return Evidence(method=method, case="synthetic", quantity="synthetic metric",
                    measured_pct=measured, gate_pct=gate, source="synthetic.json")


WAVE = CaseProfile(acoustic_mach=0.03, has_free_surface=True, gravity_waves=True)
SHOCK = CaseProfile(acoustic_mach=0.9, shock_advection=True)
COUETTE = CaseProfile(acoustic_mach=0.01, reynolds=1.0, domain="closed")
VORTEX = CaseProfile(acoustic_mach=0.05, reynolds=1e4, domain="periodic",
                     smooth_inviscid_vortex=True)


# --------------------------------------------------------------------------- #
# Gate semantics
# --------------------------------------------------------------------------- #
def test_no_registry_means_no_selection():
    """An empty registry must yield WITHHELD, never a silent default scheme."""
    selection = select_method(WAVE, registry={})
    assert selection.primary is None
    assert selection.is_verified is False
    assert all(d.status == "WITHHELD" for d in selection.candidates)
    assert any("evidence gate" in caveat for d in selection.candidates
               for caveat in d.caveats)


def test_failing_evidence_blocks_selection():
    registry = {"wcsph_delta_sph": [_evidence("wcsph_delta_sph", measured=40.0)]}
    selection = select_method(WAVE, registry=registry)
    assert selection.primary is None
    decision = selection.candidates[0]
    assert decision.status == "FAILED"
    assert decision.worst_measured_pct == pytest.approx(40.0)
    assert any("40" in caveat for caveat in decision.caveats)


def test_owner_tolerance_boundary_is_inclusive():
    tol = OWNER_TOLERANCE_PCT
    ok = select_method(WAVE, registry={"wcsph_delta_sph": [_evidence("wcsph_delta_sph", tol)]})
    bad = select_method(
        WAVE, registry={"wcsph_delta_sph": [_evidence("wcsph_delta_sph", tol + 1e-6)]}
    )
    assert ok.primary is not None and ok.primary.status == "SELECTED"
    assert bad.primary is None
    assert bad.candidates[0].status == "FAILED"


def test_selected_decision_never_claims_more_than_owner_tolerance():
    registry = load_registry(REGISTRY_JSON)
    for profile in (WAVE, SHOCK, COUETTE,
                    CaseProfile(capillary_dominated=True, has_free_surface=True),
                    CaseProfile(thermal_transport=True)):
        selection = select_method(profile, registry=registry)
        assert selection.primary is not None, profile
        assert selection.primary.worst_measured_pct is not None
        assert selection.primary.worst_measured_pct <= OWNER_TOLERANCE_PCT


def test_passing_and_failing_records_coexist_cleanly():
    """A method with one bad record is still selectable, but the bad record shows."""
    registry = {
        "csf_surface_tension": [
            _evidence("csf_surface_tension", measured=5.0, gate=15.0),
            _evidence("csf_surface_tension", measured=23.8, gate=15.0),
        ]
    }
    selection = select_method(
        CaseProfile(capillary_dominated=True, has_free_surface=True), registry=registry
    )
    primary = selection.primary
    assert primary is not None and primary.status == "SELECTED"
    # the accuracy claim is bounded by the passing record, not by the failing one
    assert primary.worst_measured_pct == pytest.approx(5.0)
    assert any("23.8" in caveat for caveat in primary.caveats)


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def test_projection_preference_is_backed_by_the_isph_channel_record():
    """m18278: ISPH carries a channel record (0.87 %), so it wins, not the fallback."""
    selection = select_method(COUETTE, registry=load_registry(REGISTRY_JSON))
    assert selection.primary is not None
    assert selection.primary.method == "isph_projection"
    assert selection.primary.status == "SELECTED"
    assert selection.primary.worst_measured_pct <= 15.0
    assert not any("fell back" in note for note in selection.notes)


def test_smooth_inviscid_vortex_is_refused_not_silently_downgraded():
    """Gresho-class flows: WCSPH fallback is known to lose ~37 % -> refuse."""
    selection = select_method(VORTEX, registry=load_registry(REGISTRY_JSON))
    assert selection.primary is None
    assert selection.is_verified is False
    assert any("outside the validated regime" in note for note in selection.notes)


def test_shock_profile_routes_to_compressible_scheme():
    selection = select_method(SHOCK, registry=load_registry(REGISTRY_JSON))
    assert selection.primary is not None
    assert selection.primary.method == "compressible_riemann"


def test_free_surface_profile_selects_wcsph_with_density_diffusion():
    selection = select_method(WAVE, registry=load_registry(REGISTRY_JSON))
    assert selection.primary is not None
    assert selection.primary.method == "wcsph_delta_sph"
    modifiers = {d.method: d.status for d in selection.modifiers}
    assert modifiers["density_diffusion"] == "SELECTED"


def test_tensile_stretching_modifier_is_backed_by_the_tvc_record():
    profile = WAVE.replace(tensile_stretching=True)
    selection = select_method(profile, registry=load_registry(REGISTRY_JSON))
    modifiers = {d.method: d for d in selection.modifiers}
    # m18278 banked particle shifting (8.83 % <= 10 % gate); the transport
    # velocity correction was the last WITHHELD hole and now carries the
    # Martin & Moyce dam-break record (10.81 % at dx=0.03 m).
    assert modifiers["particle_shifting"].status == "SELECTED"
    tvc = modifiers["transport_velocity_correction"]
    assert tvc.status == "SELECTED"
    assert tvc.worst_measured_pct is not None
    assert tvc.worst_measured_pct <= OWNER_TOLERANCE_PCT
    # the coarser rung is inside the registry on purpose; it must surface as a
    # caveat instead of being dropped, so resolution sensitivity stays visible.
    assert any("dx=0.04 m" in caveat for caveat in tvc.caveats)


def test_capillary_rule_outranks_free_surface_rule():
    profile = CaseProfile(acoustic_mach=0.05, has_free_surface=True,
                          capillary_dominated=True)
    selection = select_method(profile, registry=load_registry(REGISTRY_JSON))
    assert selection.primary is not None
    assert selection.primary.method == "csf_surface_tension"


# --------------------------------------------------------------------------- #
# Contract / robustness
# --------------------------------------------------------------------------- #
def test_profile_rejects_bad_input():
    with pytest.raises(ValueError):
        CaseProfile(domain="wobbly")
    with pytest.raises(ValueError):
        CaseProfile(acoustic_mach=-1.0)
    with pytest.raises(ValueError):
        CaseProfile.from_mapping({"acoustic_mach": 0.1, "colour": "blue"})


def test_decision_is_deterministic_and_json_serialisable():
    registry = load_registry(REGISTRY_JSON)
    first = select_method(WAVE, registry=registry).as_dict()
    second = select_method(WAVE, registry=registry).as_dict()
    assert first == second
    assert json.loads(json.dumps(first)) == first
    assert WAVE.has_free_surface is True  # profile not mutated


def test_every_catalogued_module_is_importable():
    for key, spec in {**METHODS, **MODIFIERS}.items():
        for module in spec.modules:
            importlib.import_module(module), key


def test_render_decision_is_human_readable():
    text = render_decision(select_method(WAVE, registry=load_registry(REGISTRY_JSON)))
    assert "primary scheme" in text
    assert "wcsph_delta_sph" in text
    assert "owner standard 15%" in text


# --------------------------------------------------------------------------- #
# Registry provenance / anti-drift
# --------------------------------------------------------------------------- #
def test_registry_matches_committed_artifacts():
    builder = _builder()
    rebuilt = builder.build_registry()
    committed = json.loads(REGISTRY_JSON.read_text())
    assert rebuilt == committed, "registry drifted from its source artefacts; rerun the builder"
    assert rebuilt["missing_sources"] == []
    assert len(rebuilt["records"]) >= 6
    for record in rebuilt["records"]:
        source = REPO / record["source"]
        assert source.exists(), record["source"]
        assert len(record["source_sha256"]) == 64
        assert record["source_sha256"] == builder.sha256_of(source)


def test_matrix_matches_committed_document():
    builder = _builder()
    registry = json.loads(REGISTRY_JSON.read_text())
    matrix = builder.build_matrix(registry)
    committed = json.loads(MATRIX_JSON.read_text())
    assert matrix == committed, "routing matrix drifted; rerun the builder"
    assert matrix["unexpected_routes"] == []


def test_build_check_flag_is_green():
    builder = _builder()
    assert builder.main(["--check"]) == 0


def test_m18278_withheld_methods_are_closed_by_records():
    """Anchor increment: the three previously WITHHELD schemes carry records."""
    registry = load_registry(REGISTRY_JSON)
    for method, gate in (
        ("isph_projection", 15.0),
        ("compression_regularization", 15.0),
        ("particle_shifting", 15.0),
    ):
        records = registry[method]
        assert records, f"{method} still has no evidence record"
        assert min(r.measured_pct for r in records) <= gate
        assert records[0].source_sha256


def test_transport_velocity_record_is_reproducible_from_raw_curves():
    """The banked percentage must be recomputable from the raw Z/a curves.

    Anti-drift for the number itself, not just for the file hash: the stored
    ``score`` block of the artefact is ignored here and the error is recomputed
    against the in-repo Martin & Moyce digitisation.
    """
    artefact = json.loads((REPO / "docs/xval_sphinx_tvc/data/grid_dx003.json").read_text())
    builder = _builder()
    record = next(r for r in json.loads(REGISTRY_JSON.read_text())["records"]
                  if r["extractor"] == "x_transport_velocity")
    exp = json.loads((REPO / "benchmarks/dam_break/martin_moyce_1952.json").read_text())
    arm = next(r for r in artefact["records"]
               if r["variant"] == builder.TVC_VARIANT and abs(r["dx"] - 0.03) < 1e-12)
    tau, za = arm["tau"], arm["Z_over_a"]
    rel = []
    for point in exp["data"]:
        t_ref, z_ref = point["tau"], point["Z_over_a"]
        if not z_ref or t_ref < tau[0] or t_ref > tau[-1]:
            continue
        for k in range(len(tau) - 1):
            if tau[k] <= t_ref <= tau[k + 1]:
                w = 0.0 if tau[k + 1] == tau[k] else (t_ref - tau[k]) / (tau[k + 1] - tau[k])
                got = za[k] + w * (za[k + 1] - za[k])
                rel.append(abs(got - z_ref) / z_ref)
                break
    recomputed = sum(rel) / len(rel) * 100.0
    assert recomputed == pytest.approx(record["measured_pct"], rel=1e-9)
    assert record["within_owner_tolerance"] is True


def test_every_catalogued_scheme_has_a_verification_record():
    """Evidence-completeness gate: the selector must have nothing left to withhold.

    Adding a scheme to ``METHODS``/``MODIFIERS`` without banking a machine-readable
    record trips this test -- that is the point of the gate: an unverified scheme
    would otherwise be reported as WITHHELD while looking shipped.
    """
    doc = json.loads(REGISTRY_JSON.read_text())
    assert doc["methods_without_evidence"] == []
    assert doc["missing_sources"] == []
    evidenced = set(doc["methods_with_evidence"])
    assert set(METHODS) | set(MODIFIERS) <= evidenced
    for record in doc["records"]:
        assert record["source_sha256"], record["extractor"]


def test_vortex_refusal_survives_generic_projection_evidence():
    """A channel record must not be extrapolated to a periodic vortex case."""
    registry = load_registry(REGISTRY_JSON)
    assert registry["isph_projection"], "precondition: generic record exists"
    selection = select_method(VORTEX, registry=registry)
    assert selection.primary is None
    assert any("isph_projection=WITHHELD" in note for note in selection.notes)
