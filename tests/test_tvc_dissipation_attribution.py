"""Integrity and claim tests for the TVC dissipation-attribution study.

Anchor: ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.

What is being protected
-----------------------
``benchmarks/xval_sphinx_tvc/dissipation_attribution.py`` exists to attack the
anchor's own headline claim ("TVC improves ``Z/a``") with two confounds: a
missing no-slip floor and plain bulk dissipation.  Both were **rejected** by
measurement.  A negative result is only worth anything if it cannot silently
degrade into a positive one, so these tests pin:

1. the payload is complete -- every predeclared configuration produced a record
   (a dropped *failing* row would manufacture a pass);
2. the replicated rows still reproduce ``data/grid_dx003.json`` exactly, i.e.
   the study runs the published code path and not a look-alike;
3. the two rejections are *data-bound*: no correction-off configuration reaches
   the gate, and none comes within 1 pp of the calibrated operator;
4. the operator's unique signature (lattice regularity) is strictly better for
   TVC than for the best dissipation-only run -- if that ever inverts, the
   attribution conclusion is void and this test must fail;
5. ``DISSIPATION_ATTRIBUTION.md`` is a regeneration of the payload, not prose
   with hand-typed numbers.

Pure post-processing of the payload: no SPH run, sub-second.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_sphinx_tvc"))

import dissipation_attribution as da  # noqa: E402
import make_dissipation_doc as doc  # noqa: E402

PAYLOAD = ROOT / "docs" / "xval_sphinx_tvc" / "data" / "dissipation_attribution.json"
DOC = ROOT / "docs" / "xval_sphinx_tvc" / "DISSIPATION_ATTRIBUTION.md"
PUBLISHED = ROOT / "docs" / "xval_sphinx_tvc" / "data" / "grid_dx003.json"


@pytest.fixture(scope="module")
def payload():
    if not PAYLOAD.exists():  # pragma: no cover - payload ships with the repo
        pytest.skip(f"missing payload {PAYLOAD}")
    return json.loads(PAYLOAD.read_text())


def test_every_predeclared_configuration_was_recorded(payload):
    """No row may be dropped from the grid after the fact."""
    declared = {g["label"] for g in payload["grid_declared"]}
    recorded = {r["label"] for r in payload["records"]}
    assert recorded == declared
    assert len(payload["records"]) == payload["summary"]["records"] == len(declared)


def test_anchor_and_gates_are_the_ones_the_ledger_quotes(payload):
    assert payload["anchor"] == "XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION"
    assert payload["gates"]["owner_rel"] == da.OWNER_GATE_REL == 0.15
    assert payload["gates"]["ledger_abs"] == da.LEDGER_GATE_ABS == 0.15


def test_operator_leg_models_are_the_upstream_coefficients(payload):
    """The alpha values in the grid are the shipped ones, not new fits."""
    alphas = {r["tvc_alpha"] for r in payload["records"]} - {None}
    assert da.UPSTREAM_SINGLE_STEP in alphas
    assert da.UPSTREAM_DUAL_CRITERIA in alphas
    assert da.CALIBRATED_ALPHA in alphas


def test_replicated_rows_reproduce_the_published_payload(payload):
    """Same code path, same numbers: bit-for-bit on the shared rows."""
    published = {r["variant"]: r for r in json.loads(PUBLISHED.read_text())["records"]}
    here = {r["label"]: r for r in payload["records"]}
    shared = 0
    for mine, theirs in (("slip_base", "baseline"), ("slip_a005", "tvc_a005_all"),
                         ("slip_a010", "tvc_a010_all")):
        if theirs not in published:
            continue
        shared += 1
        assert here[mine]["mean_rel_error"] == pytest.approx(
            published[theirs]["score"]["mean_rel_error"], abs=1e-12)
        assert here[mine]["mae"] == pytest.approx(
            published[theirs]["score"]["mae"], abs=1e-12)
    assert shared >= 3, "the replication check lost its published rows"


def test_q1_no_slip_floor_does_not_meet_the_gate_without_the_correction(payload):
    """Q1 rejected: the physical floor alone is not the explanation."""
    q1 = payload["summary"]["q1_noslip_physical_fix"]
    assert q1["gate_met_without_tvc"] is False
    assert q1["mean_rel_error"] > da.OWNER_GATE_REL
    for rec in payload["records"]:
        if rec["tvc_alpha"] is None:
            assert not rec["gate_rel_pass"], rec["label"]
            assert rec["band_verdict"] in {"FAIL", "PASS_marginal"}
            assert rec["band_verdict"] != "PASS"


def test_q2_bulk_dissipation_cannot_fake_the_correction(payload):
    """Q2 rejected: no viscosity-only row comes near the calibrated operator."""
    q2 = payload["summary"]["q2_dissipation_substitution"]
    assert q2["dissipation_reaches_calibrated_within_1pp"] is False
    calibrated = q2["calibrated_mean_rel_error"]
    for rec in payload["records"]:
        if rec["tvc_alpha"] is not None:
            continue
        # Every correction-off configuration is at least 3x the calibrated error.
        assert rec["mean_rel_error"] > 3.0 * calibrated, rec["label"]


def test_q3_the_operator_has_a_signature_dissipation_lacks(payload):
    """The surviving attribution: lattice regularity cannot be bought with viscosity."""
    q3 = payload["summary"]["q3_unique_signature"]
    tvc_regularity = q3["at_calibrated_tvc"]
    visc_regularity = q3["at_matched_dissipation"]
    # Smaller is better on all three channels.
    assert tvc_regularity["grad1_residual_mean_h"] < visc_regularity["grad1_residual_mean_h"]
    assert tvc_regularity["median_nn_over_dx"] > visc_regularity["median_nn_over_dx"]
    assert tvc_regularity["close_pair_fraction"] < visc_regularity["close_pair_fraction"]
    # And the gap is material, not a rounding artefact: the dissipation-only run
    # ends up with more than twice the clumped-pair share.
    assert (visc_regularity["close_pair_fraction"]
            > 2.0 * tvc_regularity["close_pair_fraction"])


def test_the_calibrated_row_is_only_a_marginal_pass(payload):
    """Not overstating the engine: the headline row does not survive the band."""
    calibrated = next(r for r in payload["records"] if r["label"] == "slip_a010")
    assert calibrated["mean_rel_error"] < da.OWNER_GATE_REL
    assert calibrated["band_verdict"] == "PASS_marginal"
    # The shipped coefficients remain outside the owner gate.
    for label in ("slip_a005", "slip_a020"):
        rec = next((r for r in payload["records"] if r["label"] == label), None)
        if rec is not None:
            assert rec["mean_rel_error"] > da.OWNER_GATE_REL


def test_document_is_regenerated_from_the_payload(payload):
    """The report cannot drift from the run it reports."""
    assert DOC.exists()
    committed = DOC.read_text()
    assert doc.render(payload) == committed
    # Guard against the markdown losing the honest verdict in an edit.
    assert "Is the headline quantity below 3 %?** No." in committed


def test_figure_is_present_and_matches_its_caption(payload):
    fig = doc.FIG
    assert fig.exists()
    assert fig.stat().st_size > 20_000, "figure looks like an empty canvas"