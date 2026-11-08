"""Credit qualification tests for the verification claim ledger.

The ledger grades *readings*.  An artifact grades *run sets*.  A reading can sit
inside the 3 % band while the artifact that produced it says ``verdict = fail``
(a missing ladder case, a failing interior criterion): the number is then a
best-of-N / finest-rung selection out of a run set that does not pass.  Until
this file existed the ledger had no way to see that, because the artifact's own
verdict never travelled with the entry.

These tests do two different things:

* **rule tests** — drive ``credit_findings`` with synthetic entries, so the rule
  is shown to fire on the intended shape and only on that shape, and
* **ledger invariants** — assert the committed ledger carries the declared
  verdict of every credit and that nothing credited from a failing artifact is
  left without a finding.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "benchmarks" / "verification_ledger" / "build_ledger.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("verification_ledger_builder_oc", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = _load_builder()


@pytest.fixture(scope="module")
def ledger():
    return B.build()


@pytest.fixture(scope="module")
def by_id(ledger):
    return {e["claim_id"]: e for e in ledger["entries"]}


def _entry(claim_id: str, grade: str, rel_error_pct: float, verdict: dict | None) -> dict:
    """A minimal entry, shaped like the ones ``build()`` emits."""
    return {
        "claim_id": claim_id,
        "grade": grade,
        "rel_error_pct": rel_error_pct,
        "artifact": f"benchmarks/synthetic/{claim_id}.json",
        "declared_verdict": verdict,
    }


FAILING = {
    "value": "fail",
    "json_path": "verdict",
    "failing_criteria": ["interior_pressure_within_15pct_best"],
    "missing_cases": [],
}
PASSING = {"value": "pass", "json_path": "verdict", "failing_criteria": [], "missing_cases": []}


# ---------------------------------------------------------------------------
# the rule itself
# ---------------------------------------------------------------------------
def test_rule_fires_on_a_pass_taken_from_a_failing_artifact():
    found = B.credit_findings([_entry("a.b", "PASS", -1.0, FAILING)])
    assert [f["rule"] for f in found] == ["credit_from_failing_artifact"]
    assert found[0]["claim_id"] == "a.b"
    assert found[0]["severity"] == "high"
    # the machine-checkable reason travels with the finding, it is not prose-only
    assert "interior_pressure_within_15pct_best" in found[0]["detail"]


def test_rule_accepts_a_pass_from_a_passing_or_silent_artifact():
    assert B.credit_findings([_entry("a.b", "PASS", -1.0, PASSING)]) == []
    assert B.credit_findings([_entry("a.b", "MARGINAL", -8.0, None)]) == []


def test_rule_does_not_touch_grades_that_are_not_credits():
    """A FAIL/WITHHELD from a failing artifact is consistent, not a finding."""
    for grade in ("FAIL", "WITHHELD", "WITHDRAWN"):
        assert B.credit_findings([_entry("a.b", grade, -40.0, FAILING)]) == []


def test_missing_cases_are_quoted_when_that_is_why_the_artifact_failed():
    dv = dict(FAILING, failing_criteria=[], missing_cases=["h1.8"])
    found = B.credit_findings([_entry("a.b", "PASS", -2.0, dv)])
    assert "missing cases: h1.8" in found[0]["detail"]


def test_normalise_verdict_understands_the_shapes_used_in_the_repo():
    assert B.normalise_verdict("fail") == "fail"
    assert B.normalise_verdict("PASS") == "pass"
    assert B.normalise_verdict({"status": "FAIL"}) == "fail"
    assert B.normalise_verdict({"evidence_grade": "WITHHELD"}) == "fail"
    assert B.normalise_verdict({"worst_within_15pct": False}) == "fail"
    assert B.normalise_verdict(False) == "fail"
    assert B.normalise_verdict("partial") is None


# ---------------------------------------------------------------------------
# invariants of the committed ledger
# ---------------------------------------------------------------------------
def test_every_credit_carries_its_artifact_declared_verdict(ledger):
    for e in ledger["entries"]:
        assert "declared_verdict" in e, e["claim_id"]
        if e["grade"] in {"PASS", "MARGINAL"}:
            assert "credit_qualified" in e, e["claim_id"]


def test_no_credit_from_a_failing_artifact_is_left_unflagged(ledger):
    finded = {(f["rule"], f["claim_id"]) for f in ledger["findings"]}
    for e in ledger["entries"]:
        if e["grade"] not in {"PASS", "MARGINAL"}:
            continue
        dv = e["declared_verdict"]
        if dv and dv["value"] == "fail":
            assert ("credit_from_failing_artifact", e["claim_id"]) in finded, (
                f"{e['claim_id']} is credited {e['grade']} out of an artifact that "
                "declares fail, without a finding"
            )
            assert e["credit_qualified"] is False
        else:
            assert e["credit_qualified"] is True, e["claim_id"]


def test_summary_counts_the_unqualified_credits(ledger):
    s = ledger["summary"]
    unqualified = [
        e for e in ledger["entries"]
        if e["grade"] in {"PASS", "MARGINAL"} and e["credit_qualified"] is False
    ]
    assert s["n_credits_unqualified"] == len(unqualified)
    silent = [
        e for e in ledger["entries"]
        if e["grade"] in {"PASS", "MARGINAL"} and e["declared_verdict"] is None
    ]
    assert s["n_credits_without_declared_verdict"] == len(silent)
    credits = s["by_grade"].get("PASS", 0) + s["by_grade"].get("MARGINAL", 0)
    assert len(unqualified) + len(silent) <= credits


def test_known_selections_are_recorded_as_unqualified(by_id):
    """Best-of-N / partial-ladder selections must not be quoted as verification."""
    for cid in (
        "threed_hydrostatic.bottom_pressure.adami_best",
        "threed_shear.decay_rate.finest",
    ):
        e = by_id[cid]
        assert e["grade"] == "PASS", cid
        assert e["credit_qualified"] is False, cid
        assert e["declared_verdict"]["value"] == "fail", cid


def test_unfiltered_complements_are_registered_and_bound_the_selection(by_id):
    """The worst rung / worst run must be in the ledger next to the best one."""
    pairs = [
        ("threed_shear.decay_rate.finest", "threed_shear.decay_rate.worst_of_ladder"),
        ("threed_hydrostatic.bottom_pressure.adami_best",
         "threed_hydrostatic.bottom_pressure.adami_worst"),
    ]
    for best, worst in pairs:
        assert abs(by_id[worst]["rel_error_pct"]) >= abs(by_id[best]["rel_error_pct"]), worst
        assert by_id[worst]["quantity_id"] == by_id[best]["quantity_id"], worst


def test_hydrostatic_profile_is_registered_because_the_bottom_layer_is_not_the_profile(by_id):
    mean = by_id["threed_hydrostatic.profile.mean_rel_error_core.adami_best"]
    worst = by_id["threed_hydrostatic.profile.max_rel_error_core.adami_best"]
    assert 3.0 < abs(mean["rel_error_pct"]) <= 15.0
    assert abs(worst["rel_error_pct"]) > 15.0 and worst["grade"] == "FAIL"
    # the bottom-layer PASS must not be the only registered reading of that artifact
    assert len({e["quantity_id"] for e in by_id.values()
                if e["artifact"].endswith("threed_hydrostatic/report.json")}) >= 3


def test_committed_ledger_is_still_in_sync_after_the_new_rule(ledger):
    path = REPO / "benchmarks" / "verification_ledger" / "ledger.json"
    committed = json.loads(path.read_text())
    assert committed == json.loads(B._dump(ledger))

def test_overcredit_doc_is_generated_and_up_to_date():
    """The document's numbers must come from the ledger, never from the author."""
    spec = importlib.util.spec_from_file_location(
        "overcredit_doc", REPO / "benchmarks" / "verification_ledger" / "make_overcredit_doc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    committed = mod.DOC.read_text()
    assert committed == mod.render(), (
        "docs/verification_ledger_overcredit_m6510.md is stale — run "
        "`python benchmarks/verification_ledger/make_overcredit_doc.py`"
    )
    assert "credit_from_failing_artifact" in committed
