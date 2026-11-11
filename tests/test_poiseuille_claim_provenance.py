"""Gates for the Poiseuille *claim-provenance* audit.

`docs/verification_poiseuille_wall_shear_audit.md` (mission m14711) attributed a
wall-shear deficit to a no-slip boundary condition that "was never applied",
quoting `u_max = 8.87 m/s`, slip `0.999996`, `Ma = 0.493`, `nu = 1e-4 m^2/s` and
an analytic `tau_w = 50 Pa`.  Four of those readings belong to the free-slip
*control* of the same artifact (a case that must slip and must accelerate at
`g`), one belongs to a different resolution than the one the table claims, and
`rho0 g H / 2` of the stored run is `490.5 Pa`.

These gates pin the archaeology so that the defect cannot silently reappear, and
so that the corrected reading cannot silently drift:

* every number quoted by the two *peer* records (channel, estimator) still
  resolves to the artifact path it declares -- so the binder is not simply
  flagging everything;
* the four suspect readings still resolve to the free-slip-control block, and
  the analytic constant still does not exist anywhere in the artifacts;
* the artifact's own analytic columns still equal their derivations, so a typed
  constant is detectable at all;
* the "wall condition not applied" hypothesis is still falsified by the stored
  momentum budget (no-slip saturates, control accelerates at `g`);
* the generated record/artifact are in sync with a fresh derivation, and the
  audited document still carries the correction banner.

Nothing here advances an SPH state: the whole file is post-processing of
committed JSON, so it runs in about a second.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "poiseuille"
ARTIFACT = BENCH / "out" / "provenance_audit.json"
REGISTER = BENCH / "claims_register.json"
DOC = ROOT / "docs" / "verification_poiseuille_claim_provenance.md"
AUDITED_DOC = ROOT / "docs" / "verification_poiseuille_wall_shear_audit.md"
PEER_DOCS = {
    "docs/verification_poiseuille_channel.md",
    "docs/verification_poiseuille_wall_shear_estimator.md",
}


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "provenance_audit", BENCH / "provenance_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fresh() -> dict:
    return _load_module().main()


def _claims(payload: dict) -> dict[str, dict]:
    return {claim["id"]: claim for claim in payload["claims"]}


def test_artifact_matches_a_fresh_derivation(committed, fresh):
    assert committed["claims"] == fresh["claims"]
    assert committed["analytic_identities"] == fresh["analytic_identities"]
    assert committed["hypothesis_tests"] == fresh["hypothesis_tests"]
    assert committed["summary"] == fresh["summary"]


def test_every_declared_quote_is_unique_in_its_document(fresh):
    for claim in fresh["claims"]:
        assert claim["quote_occurrences"] == 1, (
            f"{claim['id']}: quote occurs {claim['quote_occurrences']} times in {claim['doc']};"
            " the claim cannot be bound to a unique place in the record")


def test_register_paths_resolve(fresh):
    """A typo in the register would silently pass as 'not_in_artifact' otherwise."""
    module = _load_module()
    results = json.loads((BENCH / "out" / "results.json").read_text(encoding="utf-8"))
    estimators = json.loads(
        (BENCH / "out" / "wall_shear_estimators.json").read_text(encoding="utf-8"))
    index, derived = module.build_index(results, estimators)
    register = json.loads(REGISTER.read_text(encoding="utf-8"))
    for claim in register["claims"]:
        module.resolve(index, derived, claim["path"])


def test_peer_records_bind_to_their_artifact_paths(fresh):
    claims = _claims(fresh)
    peer = [claim for claim in claims.values() if claim["doc"] in PEER_DOCS]
    assert len(peer) >= 12
    defective = [claim["id"] for claim in peer if claim["status"] != "verified"]
    assert defective == [], f"peer records no longer bind to the artifact: {defective}"


def test_the_audited_record_misreads_the_free_slip_control(fresh):
    claims = _claims(fresh)
    expectations = {
        "audit.u_max.measured": "results.free_slip_control.u_max_measured",
        "audit.wall_row_slip": "results.free_slip_control.wall_row_slip_over_umax",
        "audit.mach": "results.free_slip_control.max_mach",
        "audit.tau_over_tau_w": "results.results[2].wall_shear_ratio",
    }
    for claim_id, path in expectations.items():
        claim = claims[claim_id]
        assert claim["status"] == "misattributed", claim
        assert claim["actual_path"] == path, claim


def test_the_audited_analytic_wall_shear_is_not_the_stored_constant(fresh):
    claim = _claims(fresh)["audit.tau_analytic"]
    assert claim["status"] == "not_in_artifact", claim
    # the stored constant is 490.5 Pa, so the printed 50 Pa is ~90 % away, and the
    # nearest unrelated readout is far outside the rounding tolerance
    assert abs(claim["artifact_value"] - 490.5) < 1e-6
    assert claim["rel_error"] > 0.8
    assert claim["nearest_rel_error"] > claim["tol_rel"]
    # 50 Pa can only come from g = 2 tau / (rho H) = 1 m/s^2, a third value again.
    assert claim["implied_body_force"] == pytest.approx(1.0, rel=1e-6)


def test_the_audited_parameters_do_not_describe_the_run(fresh):
    claims = _claims(fresh)
    for claim_id in ("audit.body_force", "audit.viscosity"):
        claim = claims[claim_id]
        assert claim["status"] == "not_in_artifact", claim


def test_stored_analytic_columns_equal_their_derivations(fresh):
    identities = fresh["analytic_identities"]
    assert len(identities) == 8  # 4 blocks x (tau_w, u_max)
    violations = [row for row in identities if row["status"] != "verified"]
    assert violations == [], violations


def test_the_no_slip_hypothesis_is_falsified_by_the_stored_budget(fresh):
    tests = fresh["hypothesis_tests"]
    no_slip = tests["falsification"]["no_slip_rate_over_g"]
    control = tests["falsification"]["free_slip_rate_over_g"][0]
    assert max(abs(rate) for rate in no_slip) < 0.05   # saturates
    assert control == pytest.approx(1.0, abs=0.02)     # accelerates at g
    assert tests["falsification"]["verdict"] == "falsified"
    # and the peak velocity converges towards the analytic value instead of blowing up
    ratios = [entry["u_max_ratio"] for entry in tests["no_slip_ladder"]]
    assert ratios[0] > ratios[1] > ratios[2]
    assert abs(ratios[-1] - 1.0) < 0.03
    assert tests["free_slip_control"]["u_max_ratio"] > 7.0
    # the drag the wall transmits is the momentum balance, not the secant
    drag = [entry["drag_over_weight_control_corrected"] for entry in tests["drag_ladder"]]
    assert max(abs(value - 1.0) for value in drag) < 0.025
    secant = [entry["secant_ratio"] for entry in tests["drag_ladder"]]
    assert max(secant) < 0.7


def test_audited_document_carries_the_correction_banner():
    text = AUDITED_DOC.read_text(encoding="utf-8")
    assert "校正声明" in text
    for pointer in ("provenance_audit.py", "verification_poiseuille_claim_provenance.md",
                    "490.5 Pa"):
        assert pointer in text
    # the honest self-statement of the audited record must survive the correction
    assert "本审计仅分析已有工件" in text


def test_record_doc_and_figure_are_generated_from_the_artifact(fresh):
    text = DOC.read_text(encoding="utf-8")
    assert f"共 **{fresh['summary']['n_claims']}** 条断言" in text
    for claim in fresh["claims"]:
        if claim["status"] != "verified":
            assert claim["id"] in text
    assert (ROOT / "docs" / "figures" / "poiseuille_claim_provenance.png").stat().st_size > 10_000
    assert "figures/poiseuille_claim_provenance.png" in text
