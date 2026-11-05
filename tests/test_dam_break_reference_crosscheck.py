"""Guards on the B2 dam-break cross-source reference audit.

The audit's job is to answer one question: *can the bundled Martin & Moyce table
carry a validation verdict?* These tests pin the **measured** facts that answer
it, so that neither the tables nor the verdict can drift silently:

* the bundled table stays identifiable as a piecewise-linear sketch (arithmetic
  abscissa, 0.05 ordinate quantum, four slope levels, no acceleration phase);
* the independent transcription keeps the acceleration phase that a gate-release
  experiment must have;
* the cross-source spread stays far above the claimed +-0.05, and the verdict
  stays ``withheld`` while it does;
* the archived ladder really was scored against both references, and its stated
  MAE values are reproducible from the stored trajectories.

Both directions are pinned on purpose: replacing the bundled table with a real
digitisation would flip gates 1 and 3, and the tests then fail, which forces the
verdict, the document and the gate wiring to be rewritten by hand instead of
being left stale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "dam_break"
ARTIFACT = BENCH / "reference_crosscheck.json"
LADDER = BENCH / "reference_ladder.json"
DOC = ROOT / "docs" / "benchmarks" / "dam-break-reference-crosscheck.md"

sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT / "src"))

import reference_crosscheck as rc  # noqa: E402
import reference_ladder as rl  # noqa: E402

# measured on the current tables; edit only together with the artifacts
BUNDLED_QUANTUM = 0.05
BUNDLED_SLOPE_LEVELS = 4
BUNDLED_TAU_STEP = 0.226
CROSSCHECK_QUANTUM = 0.01
MIN_SPREAD = 0.5          # lower bound on the reference uncertainty, in Z/a
MAX_ACCEL_RATIO = 0.5     # first-interval slope / peak slope
PINNED_MAE_BUNDLED = {0.04: 0.5988, 0.02: 0.5770}
PINNED_MAE_CROSSCHECK = {0.04: 0.1302, 0.02: 0.0438}
MAE_TOL = 0.01


@pytest.fixture(scope="module")
def tables():
    return {"bundled": rc.load_bundled(), "crosscheck": rc.load_crosscheck()}


@pytest.fixture(scope="module")
def report(tables):
    return rc.evaluate(tables)


@pytest.fixture(scope="module")
def artifact():
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def ladder():
    return json.loads(LADDER.read_text())


# --------------------------------------------------------------------------- #
# 1. structure of the bundled table
# --------------------------------------------------------------------------- #
def test_bundled_table_is_an_exact_lattice(tables):
    g = rc.grid_stats(tables["bundled"]["tau"], tables["bundled"]["za"])
    assert g["tau_grid_is_arithmetic"] is True
    assert g["tau_step_spread"] < 1e-12
    assert g["tau_step_mean"] == pytest.approx(BUNDLED_TAU_STEP, abs=1e-12)
    assert g["ordinate_quantum"] == pytest.approx(BUNDLED_QUANTUM)
    assert g["n_slope_levels"] == BUNDLED_SLOPE_LEVELS


def test_bundled_table_has_no_acceleration_phase(tables):
    g = rc.grid_stats(tables["bundled"]["tau"], tables["bundled"]["za"])
    assert g["first_slope_to_peak"] > MAX_ACCEL_RATIO, (
        "the bundled table gained an acceleration phase -- if it was replaced by a "
        "real digitisation, gates 1-3 and the document must be rewritten")


def test_crosscheck_keeps_the_acceleration_phase(tables):
    g = rc.grid_stats(tables["crosscheck"]["tau"], tables["crosscheck"]["za"])
    assert g["first_slope_to_peak"] <= MAX_ACCEL_RATIO
    assert g["tau_grid_is_arithmetic"] is False
    assert g["ordinate_quantum"] <= CROSSCHECK_QUANTUM + 1e-12
    assert g["n_slope_levels"] > BUNDLED_SLOPE_LEVELS


def test_crosscheck_starts_at_the_column_edge(tables):
    # Z/a is measured from the initial column edge, so the first point is 0
    # and the transcription (which is wall-based) must have had +1 removed.
    assert tables["crosscheck"]["za"][0] == pytest.approx(0.0, abs=1e-12)
    assert tables["bundled"]["za"][0] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# 2. the cross-source spread and the verdict it forces
# --------------------------------------------------------------------------- #
def test_cross_source_spread_exceeds_the_claim(report):
    spread = report["cross_source_spread"]
    claimed = report["tables"]["bundled"]["claimed_uncertainty"]
    assert spread["max_abs_delta"] >= MIN_SPREAD
    assert spread["max_abs_delta"] / claimed > rc.SPREAD_LIMIT
    assert spread["rms_delta"] > claimed
    assert spread["n_samples"] > 100


def test_verdict_is_withheld_while_the_spread_is_that_large(report):
    gates = report["gates"]
    assert gates["bundled_is_pointwise"]["result"] == "fail"
    assert gates["acceleration_phase_present"]["result"] == "fail"
    assert gates["claimed_uncertainty_consistent"]["result"] == "fail"
    assert report["verdict"]["physical_validation_status"] == "withheld"


def test_verdict_would_flip_if_the_reference_were_clean():
    """Negative control: a scatter-bearing table on a non-uniform grid passes."""
    tau = [0.0, 0.31, 0.72, 1.10, 1.46, 1.88, 2.21, 2.60, 2.95]
    za = [0.0, 0.118, 0.447, 0.951, 1.462, 2.053, 2.508, 3.012, 3.447]
    g = rc.grid_stats(tau, za)
    assert g["tau_grid_is_arithmetic"] is False
    assert g["ordinate_quantum"] is None or g["ordinate_quantum"] <= rc.QUANTUM_LIMIT
    assert g["n_slope_levels"] > rc.SLOPE_LEVEL_LIMIT
    assert g["first_slope_to_peak"] <= MAX_ACCEL_RATIO


# --------------------------------------------------------------------------- #
# 3. the archived artifact must be a rendering, not a hand edit
# --------------------------------------------------------------------------- #
def test_artifact_matches_a_fresh_evaluation(artifact, report):
    assert artifact["benchmark"] == "B2_dam_break_reference_crosscheck"
    for name, gate in report["gates"].items():
        assert artifact["gates"][name]["result"] == gate["result"], name
    assert artifact["cross_source_spread"]["max_abs_delta"] == pytest.approx(
        report["cross_source_spread"]["max_abs_delta"], abs=1e-9)
    assert artifact["verdict"]["physical_validation_status"] == \
        report["verdict"]["physical_validation_status"]
    assert artifact["figures"], "re-run with --plot so the document can embed figures"


# --------------------------------------------------------------------------- #
# 4. the ladder: same simulations, three references
# --------------------------------------------------------------------------- #
def test_ladder_was_scored_against_all_three_references(ladder):
    assert ladder["benchmark"] == "B2_dam_break_reference_ladder"
    assert set(ladder["gates"]) == {"bundled", "crosscheck", "ritter"}
    assert [r["dx"] for r in ladder["results"]] == sorted(
        (r["dx"] for r in ladder["results"]), reverse=True)


def test_ladder_mae_is_reproducible_from_the_stored_trajectories(ladder, tables):
    """The archived MAE must follow from the archived curve, not from a edit."""
    refs = {"bundled": tables["bundled"], "crosscheck": tables["crosscheck"],
            "ritter": rl.ritter_reference()}
    for rec in ladder["results"]:
        for name, ref in refs.items():
            stored = rec[f"error_vs_{name}"]["mae"]
            fresh = rl.error_vs(rec["sim_tau"], rec["sim_za"], ref)["mae"]
            assert fresh == pytest.approx(stored, abs=1e-9), (rec["dx"], name)


def test_error_vs_is_consistent_with_the_reference_itself(tables):
    """Comparing a reference with an interpolant of itself must give zero."""
    ref = tables["bundled"]
    err = rl.error_vs(ref["tau"], ref["za"], ref)
    assert err["mae"] == pytest.approx(0.0, abs=1e-12)
    assert all(row["error"] == pytest.approx(0.0, abs=1e-9) for row in err["rows"])


def test_ladder_mae_values_are_pinned(ladder):
    got_b = {r["dx"]: r["error_vs_bundled"]["mae"] for r in ladder["results"]}
    got_c = {r["dx"]: r["error_vs_crosscheck"]["mae"] for r in ladder["results"]}
    for dx, value in PINNED_MAE_BUNDLED.items():
        assert got_b[dx] == pytest.approx(value, abs=MAE_TOL)
    for dx, value in PINNED_MAE_CROSSCHECK.items():
        assert got_c[dx] == pytest.approx(value, abs=MAE_TOL)


def test_reference_choice_moves_the_error_more_than_the_gate_limit(ladder):
    for rec in ladder["results"]:
        shift = rec["error_vs_bundled"]["mae"] - rec["error_vs_crosscheck"]["mae"]
        assert shift > ladder["attribution"]["gate_limit"]
    att = ladder["attribution"]
    assert att["reference_spread_dominates_gate"] is True
    assert att["reference_spread_over_gate_limit"] > 1.0


def test_ritter_reference_is_the_two_tau_line():
    ref = rl.ritter_reference(step=0.25, tau_max=2.0)
    for tau, za in zip(ref["tau"], ref["za"]):
        assert za == pytest.approx(2.0 * tau, abs=1e-12)


# --------------------------------------------------------------------------- #
# 5. the document is generated from the artifacts
# --------------------------------------------------------------------------- #
def test_document_is_generated_from_the_artifacts(artifact, ladder):
    text = DOC.read_text()
    assert "Do not edit by hand" in text
    spread = artifact["cross_source_spread"]
    assert f"{spread['max_abs_delta']:.4f}" in text
    ratio = artifact["gates"]["claimed_uncertainty_consistent"]["measured"]
    assert f"{ratio:.1f}x" in text
    for rec in ladder["results"]:
        assert f"{rec['error_vs_crosscheck']['mae']:.4f}" in text
    assert artifact["verdict"]["physical_validation_status"] in text
