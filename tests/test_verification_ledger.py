"""Machine gate for the TensorSPH verification-claims ledger (mission m3544).

These tests encode the accounting rules that keep the verification documents
honest.  They do not check that the physics is right (that is what the
benchmarks are for); they check that every number a reader can cite is
traceable to a live artifact, that a withdrawn PASS stays withdrawn and
labelled, and that the authoritative reading is the one the ledger names.

If physics improves, the ledger must be regenerated (``audit.py``) and the
authoritative row updated -- several tests below then re-evaluate themselves
rather than breaking, by comparing the recorded grade against the recorded
number instead of hard-coding "WITHHELD".
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LEDGER_DIR = REPO / "benchmarks" / "verification_ledger"
MANIFEST_PATH = LEDGER_DIR / "claims_manifest.json"
LEDGER_PATH = LEDGER_DIR / "ledger.json"

MARKER = "台帐 m3544"
GATE_15 = 15.0
GATE_3 = 3.0


def _load_audit():
    spec = importlib.util.spec_from_file_location("verification_ledger_audit",
                                                  LEDGER_DIR / "audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def audit():
    return _load_audit()


@pytest.fixture(scope="module")
def ledger(audit):
    """Rebuild the ledger live; the committed JSON must match it."""
    return audit.build_ledger()


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def committed():
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def by_id(ledger):
    return {c["id"]: c for c in ledger["claims"]}


# --------------------------------------------------------------------------
# 1. Every quoted number must exist in the document that quotes it
# --------------------------------------------------------------------------
def test_every_claim_quote_appears_verbatim_in_its_document(ledger):
    bad = [(c["id"], c["checks"]["quote"]["detail"])
           for c in ledger["claims"] if c["checks"]["quote"]["status"] != "OK"]
    assert not bad, f"quotes not found in their documents: {bad}"


# --------------------------------------------------------------------------
# 2. Every cited artifact and JSON pointer must still resolve
# --------------------------------------------------------------------------
def test_every_cited_artifact_pointer_resolves(ledger):
    bad = [(c["id"], c["checks"]["provenance"]["status"])
           for c in ledger["claims"]
           if c["checks"]["provenance"]["status"] in ("ARTIFACT_MISSING", "POINTER_MISSING")]
    assert not bad, f"claims whose artifact/pointer is gone: {bad}"


# --------------------------------------------------------------------------
# 3. The authoritative row must be fully traceable and arithmetically right
# --------------------------------------------------------------------------
def test_authoritative_claim_is_traceable_and_arithmetic_is_consistent(ledger):
    auth = next(c for c in ledger["claims"] if c["authoritative"])
    prov = auth["checks"]["provenance"]
    assert prov["status"] == "RESOLVED", prov
    arith = prov.get("arithmetic", {})
    assert arith.get("status") == "OK", (
        f"authoritative artifact stores {arith.get('stored_pct')} but its own raw "
        f"values give {arith.get('recomputed_pct')}"
    )
    # The stored error must agree with the value the ledger publishes.
    assert math.isclose(prov["artifact_error_pct"], auth["reported_error_pct"],
                        abs_tol=0.15)


# --------------------------------------------------------------------------
# 4. The published evidence grade must follow from the published number
# --------------------------------------------------------------------------
def test_evidence_grade_follows_from_the_authoritative_reading(ledger):
    s = ledger["summary"]
    err = abs(s["authoritative_error_pct"])
    expected = "WITHHELD" if err > GATE_15 else "REAL"
    assert s["evidence_grade"] == expected, (
        f"grade {s['evidence_grade']} does not follow from {err:.2f}% "
        f"({GATE_15}% platform bar)"
    )
    assert s["within_15pct"] is (err <= GATE_15)
    assert s["within_3pct"] is (err <= GATE_3)


def test_laplace_dp_is_not_claimed_within_3pct(ledger):
    """Owner's accuracy standard: >15% is unacceptable, 3% is the publication bar."""
    s = ledger["summary"]
    assert s["within_15pct"] is False
    assert s["within_3pct"] is False
    assert s["evidence_grade"] == "WITHHELD"


# --------------------------------------------------------------------------
# 5. No PASS may survive unsupported; withdrawing one must be recorded
# --------------------------------------------------------------------------
def test_no_pass_claim_is_left_unsupported(ledger, manifest):
    withdrawn = set(manifest["audit_allowlists"]["withdrawn_pass_claims"])
    claims = by_id(ledger)
    for cid in withdrawn:
        assert cid in claims, f"withdrawn PASS {cid} no longer exists in the ledger"
    unsupported = set(ledger["summary"]["unsupported_pass_claims"])
    undeclared = unsupported - withdrawn
    assert not undeclared, (
        f"claims assert a PASS that the ledger cannot support and nobody withdrew: "
        f"{sorted(undeclared)}"
    )
    # A withdrawn PASS must actually still be unsupported; if physics improves
    # the gate should be removed deliberately, not silently.
    for cid in withdrawn:
        assert cid in unsupported, (
            f"{cid} is listed as withdrawn but is now supported -- remove it from "
            "audit_allowlists.withdrawn_pass_claims and let the PASS stand"
        )


def test_withdrawn_pass_document_carries_the_pointer_block(ledger, manifest):
    claims = by_id(ledger)
    for cid in manifest["audit_allowlists"]["withdrawn_pass_claims"]:
        assert claims[cid]["checks"]["marker"]["status"] == "OK", (
            f"{claims[cid]['doc']} still shows an unqualified PASS")


# --------------------------------------------------------------------------
# 6. Artifacts overwritten in place are flagged, and the set is declared
# --------------------------------------------------------------------------
def test_overwritten_artifacts_match_the_declared_allowlist(ledger, manifest):
    declared = set(manifest["audit_allowlists"]["overwritten_claims"])
    found = set(ledger["summary"]["provenance_overwritten"])
    assert found == declared, (
        f"orphaned claims changed: found {sorted(found)}, declared {sorted(declared)}. "
        "Either an artifact was overwritten again (record it) or a claim was repaired."
    )


# --------------------------------------------------------------------------
# 7. Temporal consistency: a snapshot claim must not contradict its own run
# --------------------------------------------------------------------------
def test_snapshot_claims_are_flagged_when_the_run_was_not_settled(ledger):
    flagged = set(ledger["summary"]["consistency_flagged"])
    assert "C03" in flagged, (
        "the coarsest-grid absolute snapshot must be flagged: its artifact's own "
        "trajectory never gets inside the 15% bar"
    )
    c03 = next(c for c in ledger["claims"] if c["id"] == "C03")
    consistency = c03["checks"]["consistency"]
    assert consistency["settled"] is False
    assert consistency["snapshot_matches_endpoint"] is False


def test_derived_trajectories_do_not_support_the_single_pass_claim(ledger):
    """The +5.6% headline must be unreachable in the settled tail of its own run."""
    derived = {d["claim"]: d for d in ledger["summary"]["derived_trajectories"]}
    c03 = derived["C03"]
    assert c03["min_abs_error_achievable_pct"] > GATE_15, (
        "C03's trajectory now reaches the 15% bar; the withdrawal must be revisited"
    )
    assert c03["tail_mean_error_pct"] < -GATE_15


# --------------------------------------------------------------------------
# 8. The contradiction is documented, not hidden
# --------------------------------------------------------------------------
def test_cross_document_contradiction_is_documented(ledger, manifest):
    computed = ledger["summary"]["spread_pct_points"]
    declared = manifest["known_contradiction"]["spread_pct_points"]
    assert math.isclose(computed, declared, abs_tol=1.0), (
        f"documented spread {declared} pt but the claims now span {computed} pt"
    )
    assert computed > GATE_15, (
        "the claims have converged; the contradiction section should be retired"
    )
    # more than one measurement definition is in play -- that is the whole point
    classes = {c["definition_class"] for c in ledger["claims"]}
    assert len(classes) >= 3, classes


# --------------------------------------------------------------------------
# 9. Every ledger document points at the ledger
# --------------------------------------------------------------------------
def test_every_ledger_document_carries_the_pointer_block(ledger):
    missing = sorted({c["doc"] for c in ledger["claims"]
                      if c["checks"]["marker"]["status"] != "OK"})
    assert not missing, f"documents citing ledger claims but lacking '{MARKER}': {missing}"


# --------------------------------------------------------------------------
# 10. The committed ledger must be regenerable -- no hand edits
# --------------------------------------------------------------------------
def test_committed_ledger_is_reproducible(ledger, committed):
    document = ledger["summary"]
    stored = committed["summary"]
    for key in ("n_claims", "n_docs", "spread_pct_points", "provenance_resolved",
                "provenance_overwritten", "consistency_flagged",
                "unsupported_pass_claims", "authoritative_claim",
                "authoritative_error_pct", "evidence_grade",
                "within_15pct", "within_3pct"):
        assert stored[key] == document[key], (
            f"ledger.json[{key!r}] is stale ({stored[key]!r} != {document[key]!r}); "
            "re-run benchmarks/verification_ledger/audit.py"
        )
