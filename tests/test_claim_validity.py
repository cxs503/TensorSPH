"""Tests for the claim-validity gate (``benchmarks/claim_validity``).

The gate sits *before* the tolerance ladder of ``docs/verification_ledger.md``:
it decides whether a reading was taken from a live simulation, and it checks the
arithmetic of a coupling that a note claimed.  These tests make the gate
trustworthy rather than decorative:

* anti-drift   — committed JSON / Markdown must equal a fresh regeneration,
* precedence   — a reading from an INVALID run may never be graded FAIL
                 (FAIL asserts a measured physics defect),
* arithmetic   — a quoted scaling relation is checked numerically, and the
                 predicted value follows sqrt(sigma_eff), not a linear reading,
* independence — ladder rungs sharing one ratio to machine precision count once,
* controls     — the gate can fail (a blown-up synthetic run is rejected) and
                 does not over-reject (a bounded run with one escapee passes).
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VALIDITY = REPO / "benchmarks" / "claim_validity" / "validity.py"
LEDGER = REPO / "benchmarks" / "verification_ledger" / "ledger.json"


def _load_validity():
    spec = importlib.util.spec_from_file_location("claim_validity_gate", VALIDITY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load_validity()


@pytest.fixture(scope="module")
def audit():
    return V.build()


# ---------------------------------------------------------------------------
# anti-drift
# ---------------------------------------------------------------------------
def test_committed_audit_json_is_up_to_date(audit):
    committed = json.loads((REPO / "benchmarks" / "claim_validity" / "audit.json").read_text())
    assert committed == json.loads(V._dump(audit)), (
        "benchmarks/claim_validity/audit.json is stale — run "
        "`python benchmarks/claim_validity/validity.py`"
    )


def test_committed_markdown_is_up_to_date(audit):
    committed = (REPO / "docs" / "verification_claim_validity.md").read_text()
    assert committed == V.render_markdown(audit), (
        "docs/verification_claim_validity.md is stale — regenerate it from the gate"
    )


def test_check_mode_reports_no_drift():
    assert V.main(["--check"]) == 0


# ---------------------------------------------------------------------------
# precedence: validity before tolerance
# ---------------------------------------------------------------------------
def test_the_broken_oscillation_reading_is_frozen_not_graded(audit):
    probe = audit["probes"][[p["claim_id"] for p in audit["probes"]].index(
        "oscillating_droplet.frequency.lamb")]
    assert probe["status"] == "INVALID"
    hard = [c for c in probe["checks"] if c["kind"] == "hard"]
    assert hard and not any(c["passed"] for c in hard)

    ledger = json.loads(LEDGER.read_text())
    entry = next(e for e in ledger["entries"] if e["claim_id"] == probe["claim_id"])
    assert entry["run_validity"] == "INVALID"
    assert entry["grade"] == "WITHHELD", (
        "an INVALID run must not be graded FAIL: FAIL states a measured physics defect"
    )
    assert any(
        f["rule"] == "run_validity" and f["claim_id"] == entry["claim_id"]
        for f in ledger["findings"]
    )


def test_no_invalid_run_is_graded_on_the_tolerance_ladder():
    ledger = json.loads(LEDGER.read_text())
    offenders = [
        e["claim_id"] for e in ledger["entries"]
        if e.get("run_validity") == "INVALID" and e["grade"] == "FAIL"
    ]
    assert offenders == []


def test_unprobed_is_not_a_pass(audit):
    """An artifact without telemetry is 'cannot assert', never 'validated'."""
    s = audit["summary"]
    assert s["n_probed"] + len(audit["unprobed_claims"]) == s["n_claims"]
    assert s["by_status"].get("UNPROBED", 0) == len(audit["unprobed_claims"])
    assert all(c["claim_id"] not in s["grades_frozen"] for c in audit["unprobed_claims"])
    assert all(c["reason"] for c in audit["unprobed_claims"])


# ---------------------------------------------------------------------------
# coupling arithmetic
# ---------------------------------------------------------------------------
def test_capillary_coupling_is_checked_with_the_square_root_law(audit):
    c = audit["coupling"]
    static = c["static_sigma_eff_over_sigma"]
    # the check must be sqrt(sigma_eff), not the linear reading of the deficit
    expected = 100.0 * (math.sqrt(static) - 1.0)
    assert c["predicted_rel_error_pct"] == pytest.approx(expected, abs=1e-4)
    linear = 100.0 * (static - 1.0)
    assert linear < c["predicted_rel_error_pct"] < 0.0
    assert c["predicted_rel_error_pct"] == pytest.approx(-16.10, abs=0.05)


def test_the_quoted_note_coupling_does_not_hold(audit):
    nc = audit["coupling"]["note_claim"]
    assert nc["residual_pt"] == pytest.approx(27.43, abs=0.05)
    assert nc["verdict"] == "unsupported"
    assert nc["residual_pt"] > nc["criterion_pt"]


def test_a_coherent_rung_reproduces_the_static_force_path(audit):
    """Positive control: the reading that is not blown up lands on sigma_eff/sigma."""
    rows = {r["reading"]: r for r in audit["coupling"]["readings"]}
    coarse = rows["lamb.coarse.n2.dx2.0mm"]
    assert abs(coarse["implied_minus_static_pct"]) < 3.0
    assert abs(coarse["omega_minus_predicted_pct"]) < 3.0
    # and the dead run does not, by 20x more
    quoted = rows["quoted.ledger_claim.dx2.0mm"]
    assert quoted["run_validity"] == "INVALID"
    assert abs(quoted["implied_minus_static_pct"]) > 10.0 * abs(coarse["implied_minus_static_pct"])


# ---------------------------------------------------------------------------
# ladder independence
# ---------------------------------------------------------------------------
def test_two_rungs_reporting_one_ratio_count_once(audit):
    li = audit["ladder_independence"]
    assert li["duplicate_groups"] == [["mode2", "mode3"]]
    assert li["n_independent"] < li["n_rows"]
    by_case = {r["case"]: r for r in li["rows"]}
    assert abs(by_case["mode2"]["omega_over_reference"]
               - by_case["mode3"]["omega_over_reference"]) < 1e-9
    # the reported errors then differ only through the analytic reference
    assert abs(by_case["mode2"]["reported_rel_error_pct"]
               - by_case["mode3"]["reported_rel_error_pct"]) < 1e-9


# ---------------------------------------------------------------------------
# controls: the gate must be able to fail, and must not over-reject
# ---------------------------------------------------------------------------
def test_gate_is_not_vacuous(audit):
    st = audit["self_test"]
    assert st["blown_up_run"] == "INVALID"
    assert st["healthy_run"] == "VALID"
    assert st["escaped_but_bounded"] == "VALID"
    assert st["noisy_estimator"] == "SUSPECT"
    for k, expected in st["expected"].items():
        assert st[k] == expected, k


def test_unknown_claims_are_reported_not_guessed():
    v = V.evaluate("some.claim.that.has.no.probe")
    assert v["status"] == "UNPROBED"
    assert v["checks"] == []
    assert "no validity probe" in v["reason"]


# ---------------------------------------------------------------------------
# traceability
# ---------------------------------------------------------------------------
def test_every_probe_names_artifacts_that_exist(audit):
    for p in audit["probes"]:
        for rel in p.get("artifacts", []):
            assert (REPO / rel).is_file(), (p["claim_id"], rel)
        for c in p["checks"]:
            assert c["detail"], (p["claim_id"], c["id"])


def test_probe_values_are_read_not_typed(audit):
    """The probe's numbers must equal the artifact's, recomputed here."""
    osc = REPO / "benchmarks" / "oscillating_droplet"
    vr = json.loads((osc / "validation_report.json").read_text())
    ga = json.loads((osc / "gauge_audit.json").read_text())
    probe = [p for p in audit["probes"] if p["status"] != "UNPROBED"][0]
    checks = {c["id"]: c for c in probe["checks"]}
    assert checks["containment"]["value"] == pytest.approx(
        ga["recorded"]["containment"]["final_rms_radius_over_radius"], abs=1e-6)
    assert checks["mach_number"]["value"] == pytest.approx(
        ga["recorded"]["containment"]["max_speed_m_per_s"]
        / vr["frequency"]["setup"]["sound_speed"], abs=1e-6)
    assert checks["fit_quality"]["value"] == pytest.approx(
        vr["frequency"]["frequency"]["fit"]["r2"], abs=1e-6)
