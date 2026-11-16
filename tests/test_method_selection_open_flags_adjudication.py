"""Guards for the adjudication of every over-15 % registry record (mission 22787).

The coverage audit (mission 20432) reports two open flags and stops:
``integrity_failures`` and ``records_over_owner_tolerance``.  Mission 22608 left
them unadjudicated.  Mission 22787 traced the single integrity failure to the
auditor's own positional row slice and classified the five over-tolerance records.
These tests pin both results so they cannot silently regress:

* the Couette interior L2 recomputation must follow the artefact's declared
  ``0.15 <= y/H <= 0.85`` band at *every* resolution (the interesting one is
  H/dx = 16, where a positional slice used to disagree with the stored value);
* the adjudication must cover exactly the audit's over-tolerance set, every
  class must be justified by data (not by a hand-kept label), no pairing check
  may fail, and no over-tolerance scheme may be endorsed inside the owner
  standard by a published strict-bound decision;
* both derived-document sets must be fresh (``--check`` green).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks" / "method_selection"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BENCH_DIR))

import audit_quantity_coverage as aqc  # noqa: E402
import adjudicate_over_tolerance as adjudicator  # noqa: E402

OWNER_TOLERANCE_PCT = aqc.OWNER_TOLERANCE_PCT
AUDIT_JSON = ROOT / "docs" / "reports" / "method_selection_coverage_audit_m20432.json"
ADJ_JSON = ROOT / "docs" / "reports" / "method_selection_open_flags_adjudication_m22787.json"
STRICT_JSON = ROOT / "docs" / "reports" / "method_selection_strict_bound_m21537.json"
REGISTRY_JSON = ROOT / "docs" / "reports" / "method_selection_registry.json"


@pytest.fixture(scope="module")
def audit() -> dict:
    return json.loads(AUDIT_JSON.read_text())


@pytest.fixture(scope="module")
def adjudication() -> dict:
    return json.loads(ADJ_JSON.read_text())


@pytest.fixture(scope="module")
def registry_by_extractor() -> dict[str, dict]:
    return {r["extractor"]: r for r in json.loads(REGISTRY_JSON.read_text())["records"]}


# --------------------------------------------------------------------------- #
# 1. The auditor's interior-L2 recomputation follows the declared band
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "artifact, expected_rows",
    [("benchmarks/couette/out/results.json", 6),
     ("benchmarks/couette/out_design/results.json", 12)],
)
def test_couette_interior_l2_uses_the_declared_band(artifact, expected_rows):
    doc = json.loads((ROOT / artifact).read_text())
    band = aqc._couette_interior_rows(doc)
    assert len(band) == expected_rows
    assert all(0.15 <= float(r["y_over_H"]) <= 0.85 for r in band)
    stored = float(aqc._couette_finest(doc)["metrics"]["interior_l2"]) * 100.0
    recomputed = aqc.r_couette_l2(doc)
    assert math.isclose(recomputed, stored, rel_tol=1e-9)


def test_positional_slice_is_the_defect_that_was_fixed():
    """Negative control: the old positional slice still disagrees at H/dx = 16.

    If this ever stops being true the fixture is not exercising the bug any more,
    and the guard above would pass for the wrong reason.
    """
    doc = json.loads((ROOT / "benchmarks/couette/out_design/results.json").read_text())
    rows = aqc._couette_finest(doc)["metrics"]["rows"]
    positional = math.sqrt(
        sum(float(r["rel_err"]) ** 2 for r in rows[1:-1]) / len(rows[1:-1])) * 100.0
    stored = float(aqc._couette_finest(doc)["metrics"]["interior_l2"]) * 100.0
    assert not math.isclose(positional, stored, rel_tol=1e-6)
    assert math.isclose(aqc.r_couette_l2(doc), stored, rel_tol=1e-9)


def test_audit_no_longer_reports_an_integrity_failure(audit):
    assert audit["integrity_failures"] == []
    assert audit["stored_vs_recomputed_mismatches"] == []


# --------------------------------------------------------------------------- #
# 2. Coverage: every over-tolerance record is adjudicated, and vice versa
# --------------------------------------------------------------------------- #
def test_adjudication_covers_every_registry_record_over_the_standard(
        audit, adjudication, registry_by_extractor):
    assert adjudication["audit_integrity_failures"] == audit["integrity_failures"]
    registry_over = sorted(e for e, r in registry_by_extractor.items()
                           if float(r["measured_pct"]) > OWNER_TOLERANCE_PCT)
    assert sorted(adjudication["over_tolerance_records"]) == registry_over
    assert (sorted(a["extractor"] for a in adjudication["adjudications"])
            == registry_over)
    # the audit's own list is a *subset*: it only re-derives catalogue quantities
    assert set(audit["records_over_owner_tolerance"]) <= set(registry_over)
    gap = sorted(set(registry_over) - set(audit["records_over_owner_tolerance"]))
    assert sorted(adjudication["audit_coverage_gap"]) == gap
    for entry in adjudication["adjudications"]:
        assert entry["audit_catalogue"] is (entry["extractor"] not in gap)
    assert adjudication["owner_tolerance_pct"] == OWNER_TOLERANCE_PCT


def test_every_record_has_a_known_class(adjudication):
    known = {"resolution_attributable", "reading_mobility",
             "genuine_accuracy_failure", "documented_trap_arm"}
    assert {a["class"] for a in adjudication["adjudications"]} <= known
    for name, members in adjudication["classes"].items():
        assert name in known
        assert sorted(members) == sorted(
            a["extractor"] for a in adjudication["adjudications"]
            if a["class"] == name)


def test_pairing_checks_do_not_report_problems(adjudication):
    for entry in adjudication["adjudications"]:
        assert entry["problems"] == [], entry["extractor"]


# --------------------------------------------------------------------------- #
# 3. Each class is justified by data, not by a label
# --------------------------------------------------------------------------- #
def test_resolution_attributable_records_pass_on_the_finer_rung(
        adjudication, registry_by_extractor):
    entries = [a for a in adjudication["adjudications"]
               if a["class"] == "resolution_attributable"]
    assert entries, "expected at least one resolution-attributable record"
    for entry in entries:
        pair = entry["pair"]
        coarse = registry_by_extractor[entry["extractor"]]
        fine = registry_by_extractor[pair["fine_extractor"]]
        assert float(coarse["measured_pct"]) > OWNER_TOLERANCE_PCT
        assert float(fine["measured_pct"]) <= OWNER_TOLERANCE_PCT
        assert coarse["method"] == fine["method"]
        # The re-derivation must land on the same two numbers as the registry.
        assert math.isclose(entry["rederived"]["coarse_pct"],
                            float(coarse["measured_pct"]), rel_tol=1e-9)
        assert math.isclose(entry["rederived"]["fine_pct"],
                            float(fine["measured_pct"]), rel_tol=1e-9)


def test_reading_mobility_records_are_robustness_kind(adjudication):
    entries = [a for a in adjudication["adjudications"]
               if a["class"] == "reading_mobility"]
    assert entries, "expected at least one robustness-kind record"
    for entry in entries:
        assert entry["registry_kind"] == "robustness"
        assert entry["rederived"]["n_points"] >= entry["rederived"]["n_admissible"]
        assert math.isclose(entry["rederived"]["bottom_spread_pct"],
                            entry["banked_pct"], rel_tol=1e-9)


def test_genuine_failures_are_reproducible_from_the_artefact(adjudication):
    entries = [a for a in adjudication["adjudications"]
               if a["class"] == "genuine_accuracy_failure"]
    assert entries, "expected at least one genuine accuracy failure"
    for entry in entries:
        assert entry["registry_kind"] == "accuracy"
        recomputed = entry["rederived"].get("recomputed_pct")
        if recomputed is None:
            # outside the audit catalogue: honestly marked "stored only"
            assert entry["audit_catalogue"] is False
            assert entry["recomputed_pct"] is None
            continue
        assert recomputed > OWNER_TOLERANCE_PCT
        assert math.isclose(recomputed, entry["banked_pct"], rel_tol=1e-6)


def test_trap_arm_record_is_a_negative_control_not_a_live_claim(adjudication):
    entry = next((a for a in adjudication["adjudications"]
                  if a["class"] == "documented_trap_arm"), None)
    assert entry is not None
    assert entry["rederived"]["shipped_arm_pct"] <= OWNER_TOLERANCE_PCT
    assert entry["banked_pct"] > 10 * OWNER_TOLERANCE_PCT
    assert entry["extractor"] in adjudication["endorsed_with_trap_arm"]
    assert entry["extractor"] not in adjudication["decision_leaks"]


def test_moving_wall_failure_is_not_a_single_rung_artefact(adjudication):
    """The exclusion of wcsph_wall_no_slip rests on the moving-wall reading.

    If that reading improved below the standard at the finer rung, the published
    exclusion would be rung-dependent and the adjudication would have to say so.
    """
    entry = next((a for a in adjudication["adjudications"]
                  if a["extractor"] == "x_couette_viscous_time_top_h16"), None)
    assert entry is not None
    response = entry["rederived"]["resolution_response"]
    assert response["16"]["top_pct_at_steady_end"] > OWNER_TOLERANCE_PCT
    assert entry["rederived"]["worsens_with_resolution"] is True


# --------------------------------------------------------------------------- #
# 4. No published decision endorses an over-tolerance scheme inside the standard
# --------------------------------------------------------------------------- #
def test_no_decision_leak(adjudication, registry_by_extractor):
    assert adjudication["decision_leaks"] == []
    assert adjudication["endorsed_with_spread"] == []
    strict = json.loads(STRICT_JSON.read_text())
    # Only live accuracy claims can leak; a spread record and a documented trap
    # arm are not accuracy claims for the shipped configuration (they are
    # reported separately as endorsed_with_spread / endorsed_with_trap_arm).
    live = {"resolution_attributable", "genuine_accuracy_failure"}
    over_methods = {registry_by_extractor[a["extractor"]]["method"]
                    for a in adjudication["adjudications"]
                    if a["class"] in live}
    for row in strict["rows"]:
        strict_row = row["strict"]
        if not strict_row["within_owner_standard"]:
            continue
        endorsed = set()
        if strict_row["primary_status"] == "SELECTED":
            endorsed.add(strict_row["primary"])
        endorsed.update(strict_row.get("modifiers", []))
        leaked = endorsed & over_methods
        assert not leaked, f"{row['label']} endorses {leaked} inside the standard"
        for entry in strict_row.get("endorsed_above_tolerance", []):
            assert entry in (None, ""), row["label"]


def test_every_over_tolerance_record_cited_by_a_decision_is_adjudicated(
        adjudication, registry_by_extractor):
    strict = json.loads(STRICT_JSON.read_text())
    cited = set()
    for row in strict["rows"]:
        for excluded in row["strict"].get("excluded", []):
            worst = excluded.get("worst_accuracy_pct")
            if worst is not None and float(worst) > OWNER_TOLERANCE_PCT:
                cited.add(float(worst))
    adjudicated = {round(float(a["banked_pct"]), 6)
                   for a in adjudication["adjudications"]}
    unadjudicated = {value for value in cited
                     if round(value, 6) not in adjudicated}
    assert not unadjudicated, (
        "a published exclusion cites an over-tolerance record that the "
        f"adjudication does not cover: {sorted(unadjudicated)}")


# --------------------------------------------------------------------------- #
# 5. Derived documents stay fresh
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ["audit_quantity_coverage.py",
                                    "adjudicate_over_tolerance.py"])
def test_producer_outputs_are_a_fresh_build(script):
    proc = subprocess.run(
        [sys.executable, str(BENCH_DIR / script), "--check"],
        cwd=ROOT, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert proc.returncode == 0, proc.stdout + proc.stderr