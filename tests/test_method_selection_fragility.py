"""Evidence depth and leave-one-record-out fragility of the selector (m21797).

Anchor ``XVAL-SPH-METHOD-SELECTION-2026-11-12``.

Two layers are pinned here:

* the router's own ``Decision.evidence_depth`` / ``Selection.single_sourced`` --
  a bound measured on one benchmark family must be labelled as such, and
  variants of one benchmark must not count as independent reproduction;
* the independent audit driver ``benchmarks/method_selection/audit_decision_fragility.py``
  -- its committed artefacts must be a fresh build (``--check``) and its
  load-bearing-record claim must be reproducible by hand.

A third test guards the unit defect fixed in the same mission: the m21537 report
multiplied already-percent registry values by 100 (23.80 % printed as 2379.93 %).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from tensorsph.method_selection import (
    OWNER_TOLERANCE_PCT,
    CaseProfile,
    Decision,
    Evidence,
    case_family,
    load_registry,
    render_decision,
    select_method,
)

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "benchmarks/method_selection/audit_decision_fragility.py"
AUDIT_JSON = REPO / "docs/reports/method_selection_fragility_m21797.json"
AUDIT_MD = REPO / "docs/xval_method_selection_fragility_m21797.md"
STRICT_JSON = REPO / "docs/reports/method_selection_strict_bound_m21537.json"
STRICT_MD = REPO / "docs/xval_method_selection_strict_bound_m21537.md"
REGISTRY = REPO / "docs/reports/method_selection_registry.json"

SHOCK = dict(acoustic_mach=0.9, shock_advection=True, domain="closed")


def _record(case: str, measured: float, *, status: str = "banked",
            kind: str = "accuracy", method: str = "probe") -> Evidence:
    return Evidence(method=method, case=case, quantity="q", measured_pct=measured,
                    gate_pct=5.0, source="tests", status=status, kind=kind)


def _decision(evidence: list[Evidence]) -> Decision:
    return Decision(method="probe", status="SELECTED", role="primary",
                    rationale="test", evidence=evidence)


# --------------------------------------------------------------------------- #
# Router layer
# --------------------------------------------------------------------------- #
def test_case_family_collapses_benchmark_variants():
    assert case_family("CASE-LAPLACE-SPH (consistent CSF)") == "case-laplace-sph"
    assert case_family(" plane Couette channel ") == "plane couette channel"
    assert case_family("CASE-COUETTE-VISCOUS-TIME (moving wall, H/dx=16)") == \
        "case-couette-viscous-time"


def test_variants_of_one_benchmark_are_not_independent_reproduction():
    decision = _decision([
        _record("CASE-LAPLACE-SPH", 5.6),
        _record("CASE-LAPLACE-SPH (consistent CSF)", 23.8),
    ])
    depth = decision.evidence_depth
    assert depth["n_live_accuracy"] == 2
    assert depth["n_benchmark_families"] == 1
    assert depth["single_source"] is True
    assert decision.is_single_source is True
    assert depth["spread_pct"] == 23.8 - 5.6


def test_a_second_family_ends_the_single_source_state():
    decision = _decision([
        _record("plane Couette channel", 2.9),
        _record("CASE-COUETTE-VISCOUS-TIME (steady-state envelope, H/dx=8)", 9.5),
    ])
    depth = decision.evidence_depth
    assert depth["n_benchmark_families"] == 2
    assert depth["single_source"] is False
    assert decision.is_single_source is False


def test_refuted_and_conditioning_records_are_not_evidence_of_reproduction():
    decision = _decision([
        _record("case-a", 2.0),
        _record("case-b", 3.0, status="refuted"),
        _record("case-c", 4.0, kind="robustness"),
    ])
    depth = decision.evidence_depth
    assert depth["n_records"] == 3
    assert depth["n_live_accuracy"] == 1
    assert depth["n_benchmark_families"] == 1
    assert decision.is_single_source is True


def test_decision_without_live_accuracy_record_is_not_single_source():
    decision = _decision([_record("case-a", 3.0, status="refuted")])
    assert decision.evidence_depth["single_source"] is False
    assert decision.is_single_source is False
    assert decision.as_dict()["evidence_depth"]["n_live_accuracy"] == 0


def test_selection_rollup_reports_single_source_schemes():
    selection = select_method(CaseProfile(acoustic_mach=0.03, has_free_surface=True,
                                          gravity_waves=True))
    dumped = selection.as_dict()
    assert "single_sourced" in dumped and "weakest_link" in dumped
    assert set(selection.single_sourced) <= {d.method for d in selection.endorsed}
    assert dumped["single_sourced"] == selection.single_sourced
    assert dumped["endorsed_depth"]["wcsph_delta_sph"]["n_live_accuracy"] >= 1


def test_card_carries_the_evidence_depth_section_and_the_caveat():
    selection = select_method(CaseProfile(acoustic_mach=0.03, has_free_surface=True,
                                          gravity_waves=True))
    card = render_decision(selection)
    assert "## evidence depth" in card
    assert "SINGLE SOURCE" in card
    assert "has not been reproduced on a second benchmark family" in card


# --------------------------------------------------------------------------- #
# Audit driver
# --------------------------------------------------------------------------- #
def test_audit_artefacts_are_a_fresh_build():
    proc = subprocess.run([sys.executable, str(AUDIT), "--check"],
                          cwd=REPO, capture_output=True, text=True,
                          env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_audit_payload_is_self_consistent():
    payload = json.loads(AUDIT_JSON.read_text())
    assert payload["anchor"] == "XVAL-SPH-METHOD-SELECTION-2026-11-12"
    assert payload["mission"] == 21797
    assert payload["owner_tolerance_pct"] == OWNER_TOLERANCE_PCT
    # the independent recomputation must agree with the router, field by field
    assert payload["depth_mismatches"] == []
    assert payload["n_depth_mismatches"] == 0
    # refuted / conditioning records may never be load-bearing
    assert payload["invariants_ok"] is True
    assert payload["inert_violations"] == []
    # every published (strict-mode) recommendation is audited
    assert len(payload["profiles"]) == 5


def test_audit_depth_matches_the_registry_by_hand():
    payload = json.loads(AUDIT_JSON.read_text())
    records = json.loads(REGISTRY.read_text())["records"]
    live = [r for r in records
            if r.get("kind", "accuracy") == "accuracy" and r.get("status") != "refuted"]
    assert payload["n_records"] == len(records)
    for method, depth in payload["depth"].items():
        mine = [r for r in live if r["method"] == method]
        assert depth["n_live_accuracy"] == len(mine), method
        if mine:
            assert abs(depth["worst_pct"] - max(r["measured_pct"] for r in mine)) < 1e-12


def test_shock_recommendation_now_stands_on_two_benchmark_families():
    """m22388 added the strong-shock/contact family: the Sod tube is no longer single-sourced.

    Before m22388 this test asserted the *opposite* (one load-bearing record,
    ``single_sourced == ["compressible_riemann"]``).  The added family
    (``CASE-SPH-STRONG-SHOCK-CONTACT``, a 1000:1 driver with a 10.3x contact jump)
    makes the recommendation reproduced on two independent datasets, so the
    invariant to guard is now the stronger one: no single family can remove it,
    while removing the *method* still flips the recommendation to unverified.
    """
    payload = json.loads(AUDIT_JSON.read_text())
    row = next(r for r in payload["profiles"] if r["label"] == "Sod shock tube")
    assert row["single_sourced"] == []
    depth = row["depth"]["compressible_riemann"]
    assert depth["n_benchmark_families"] == 2 and depth["single_source"] is False
    assert set(depth["benchmark_families"]) == {
        "case-sph-sod-shock-compressible", "case-sph-strong-shock-contact"}
    # every single-family removal keeps the recommendation (no flip)
    assert all(c["changes"] == [] for c in row["critical_removals"])

    # reproduce the flip by hand: drop the method, the recommendation is gone
    registry = load_registry()
    registry = {k: v for k, v in registry.items() if k != "compressible_riemann"}
    trial = select_method(CaseProfile(**SHOCK), registry=registry, strict_bound=True)
    assert trial.primary is None or trial.primary.status != "SELECTED"
    assert trial.is_verified is False


# --------------------------------------------------------------------------- #
# Regression guard for the percent-scaling defect in the m21537 report
# --------------------------------------------------------------------------- #
def test_strict_bound_report_does_not_scale_percent_values_twice():
    payload = json.loads(STRICT_JSON.read_text())
    expected = {row["label"]: row["strict"]["headline_bound_pct"]
                for row in payload["rows"]}
    md = STRICT_MD.read_text()
    checked = 0
    for line in md.splitlines():
        if not line.startswith("| ") or line.startswith("| 画像") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        label, printed = cells[0], cells[5]
        if label not in expected or expected[label] is None:
            continue
        value = float(printed)
        assert value < 100.0, f"{label}: printed bound {value} looks 100x too large"
        assert abs(value - expected[label]) < 0.006, (label, value, expected[label])
        checked += 1
    assert checked == 5, checked


def test_audit_markdown_matches_the_json():
    payload = json.loads(AUDIT_JSON.read_text())
    md = AUDIT_MD.read_text()
    single = sorted(m for m, d in payload["depth"].items() if d["single_source"])
    assert f"**{len(single)} 个是单一来源**" in md
    for method in single:
        assert method in md
    for row in payload["profiles"]:
        assert row["label"] in md

# --------------------------------------------------------------------------- #
# Regression guards for the substitution defect found by the audit
# --------------------------------------------------------------------------- #
CAPILLARY = dict(acoustic_mach=0.05, has_free_surface=True, capillary_dominated=True)


def test_shock_rule_never_falls_through_to_a_projection_solver():
    registry = load_registry()
    stripped = {k: v for k, v in registry.items() if k != "compressible_riemann"}
    for strict in (False, True):
        trial = select_method(CaseProfile(**SHOCK), registry=stripped,
                              strict_bound=strict)
        assert trial.primary is None, (strict, trial.primary.method)
        assert trial.is_verified is False
        assert any(note.startswith("regime locked") for note in trial.notes)
        # the negative control: the full registry still selects the Riemann solver
        control = select_method(CaseProfile(**SHOCK), registry=registry,
                                strict_bound=strict)
        assert control.primary is not None
        assert control.primary.method == "compressible_riemann"


def test_capillary_rule_never_falls_through_to_surface_tension_free_wcsph():
    registry = load_registry()
    stripped = {k: v for k, v in registry.items() if k != "csf_surface_tension"}
    for strict in (False, True):
        trial = select_method(CaseProfile(**CAPILLARY), registry=stripped,
                              strict_bound=strict)
        assert trial.primary is None, (strict, trial.primary.method)
        assert "wcsph_delta_sph" not in [d.method for d in trial.candidates
                                        if d.status == "SELECTED"]
        assert any(note.startswith("regime locked") for note in trial.notes)


def test_audit_regime_probe_reports_no_substitution():
    payload = json.loads(AUDIT_JSON.read_text())
    probes = payload["regime_lock_probe"]
    assert {p["rule"] for p in probes} == {"shock", "capillary"}
    for probe in probes:
        for mode in ("lenient", "strict"):
            row = probe[mode]
            assert row["primary"] is None, (probe["rule"], mode, row["primary"])
            assert row["verified"] is False
            assert row["regime_locked"] is True, (probe["rule"], mode)
