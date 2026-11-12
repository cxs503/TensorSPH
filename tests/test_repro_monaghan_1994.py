"""Gates for REPRO-SPH-MONAGHAN1994.

Two layers, both cheap (no solver in CI):

* the comparison primitives are pinned on synthetic data whose answer is known by
  construction, so a regression in interpolation, non-dimensionalisation or the
  metric definitions cannot pass silently;
* the committed artifacts are re-read and re-checked: the recorded MAE must be
  reproducible from the recorded trajectory, the withheld reference must not be
  used as a pass source, and the protocol constants must be the ones the doc
  quotes.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for candidate in (str(ROOT), str(ROOT / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from benchmarks.repro_monaghan_1994.run import (                       # noqa: E402
    COMPRESSION_LIMIT, TOL_REL, WINDOW, bracket, compare, evaluation_grid,
    interp, references, trajectory)

RESULTS = ROOT / "benchmarks" / "repro_monaghan_1994" / "results"


# --------------------------------------------------------------------------- #
# layer 1: primitives
# --------------------------------------------------------------------------- #
def test_interp_hits_nodes_and_clamps():
    xs, ys = [0.0, 1.0, 3.0], [0.0, 2.0, 3.0]
    assert interp(xs, ys, 0.0) == pytest.approx(0.0)
    assert interp(xs, ys, 1.0) == pytest.approx(2.0)
    assert interp(xs, ys, 0.5) == pytest.approx(1.0)
    assert interp(xs, ys, 2.0) == pytest.approx(2.5)
    assert interp(xs, ys, -5.0) == pytest.approx(0.0)     # clamped, no crash
    assert interp(xs, ys, 99.0) == pytest.approx(3.0)
    assert interp([], [], 1.0) is None


def test_evaluation_grid_respects_window_and_overlap():
    sim = [0.0, 0.5, 1.0, 2.0, 4.0]
    ref = {"tau": [0.0, 1.5]}
    grid = evaluation_grid(sim, ref, WINDOW, n=11)
    assert grid[0] == pytest.approx(WINDOW[0])
    assert grid[-1] == pytest.approx(1.5)                 # table ends first
    assert len(grid) == 11
    assert evaluation_grid(sim, {"tau": [10.0, 11.0]}, WINDOW) == []


def test_compare_identical_curves_is_zero_error():
    ref = {"name": "synthetic", "path": "none", "tau": [0.0, 1.0, 2.0, 3.0, 4.0],
           "za": [0.0, 1.0, 2.0, 3.0, 4.0], "claimed_uncertainty": 0.05}
    sim_tau = [0.0 + 0.01 * i for i in range(401)]
    sim_za = [1.0 * t for t in sim_tau]
    result = compare(sim_tau, sim_za, ref)
    assert result["mae"] == pytest.approx(0.0, abs=1e-12)
    assert result["max_abs"] == pytest.approx(0.0, abs=1e-12)
    assert result["within_uncertainty_fraction"] == pytest.approx(1.0)
    assert result["mae_over_amplitude"] == pytest.approx(0.0, abs=1e-12)


def test_compare_constant_offset_is_measured_exactly():
    ref = {"name": "synthetic", "path": "none", "tau": [0.0, 4.0],
           "za": [0.0, 4.0], "claimed_uncertainty": 0.05}
    sim_tau = [0.0 + 0.01 * i for i in range(401)]
    sim_za = [t + 0.2 for t in sim_tau]
    result = compare(sim_tau, sim_za, ref)
    assert result["bias"] == pytest.approx(0.2, abs=1e-9)          # sign is reported
    assert result["max_abs"] == pytest.approx(0.2, abs=1e-9)
    # amplitude is the reference's own, so the metric is a relative error
    assert result["mae_over_amplitude"] == pytest.approx(0.2 / result["amplitude"], rel=1e-9)
    assert result["within_uncertainty_fraction"] == pytest.approx(0.0)


def test_compare_reports_missing_overlap_instead_of_a_number():
    ref = {"name": "synthetic", "path": "none", "tau": [10.0, 11.0], "za": [0.0, 1.0]}
    result = compare([0.0, 1.0], [0.0, 1.0], ref)
    assert result["n_samples"] == 0
    assert "reason" in result


def test_bracket_detects_inconsistent_reference_pair():
    a = {"name": "a", "tau": [0.0, 4.0], "za": [0.0, 4.0], "claimed_uncertainty": 0.05}
    b = {"name": "b", "tau": [0.0, 4.0], "za": [0.0, 2.0], "claimed_uncertainty": 0.05}
    sim_tau = [0.0 + 0.01 * i for i in range(401)]
    sim_za = [1.0 * t for t in sim_tau]      # sits between the two
    result = bracket(sim_tau, sim_za, a, b)
    assert result["reference_self_consistent"] is False
    assert result["reference_spread_over_claimed"] > 2.0
    assert result["inside_fraction"] > 0.0    # the band, not either curve
    assert 0.0 <= result["inside_fraction"] <= 1.0


def test_bracket_consistency_flag_when_pair_agrees():
    a = {"name": "a", "tau": [0.0, 4.0], "za": [0.0, 4.0], "claimed_uncertainty": 0.05}
    b = {"name": "b", "tau": [0.0, 4.0], "za": [0.01, 4.01], "claimed_uncertainty": 0.05}
    sim_tau = [0.0 + 0.01 * i for i in range(401)]
    result = bracket(sim_tau, [1.0 * t for t in sim_tau], a, b)
    assert result["reference_self_consistent"] is True


def test_trajectory_conversion_moves_origin_to_column_edge():
    class _Cfg:
        column_width = 0.2
        gravity = 9.81

    class _Run:
        times = [0.0, 0.1]
        front = [0.2, 0.3]

    tau, za = trajectory(_Cfg(), _Run())
    assert tau[0] == pytest.approx(0.0)
    assert za[0] == pytest.approx(0.0)                       # front at the column edge
    assert za[1] == pytest.approx(0.5)                       # 0.1 m advance / a
    assert tau[1] == pytest.approx(0.1 * math.sqrt(2 * 9.81 / 0.2))


def test_references_carry_the_audit_grades():
    refs = references()
    assert refs["bundled"]["usable_as_validation_target"] is False
    assert refs["crosscheck"]["usable_as_validation_target"] is True
    assert refs["crosscheck"]["tau"][0] == pytest.approx(0.0)
    assert refs["crosscheck"]["za"][0] == pytest.approx(0.0)  # X - 1 convention


# --------------------------------------------------------------------------- #
# layer 2: committed artifacts
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def summary():
    path = RESULTS / "summary.json"
    if not path.exists():
        pytest.skip("results/summary.json absent: run run.py --stage run")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def runs():
    paths = sorted(RESULTS.glob("trajectory_*.json"))
    if not paths:
        pytest.skip("no committed trajectories")
    return [json.loads(p.read_text()) for p in paths]


def test_recorded_mae_reproduces_from_recorded_trajectory(summary, runs):
    """The artifact's headline numbers must be recomputable, not fossilised."""
    refs = references()
    for run in runs:
        for name, ref in refs.items():
            recorded = summary["comparisons"][run["case"]][name]
            fresh = compare(run["tau"], run["za"], ref)
            assert fresh["n_samples"] == recorded["n_samples"]
            assert fresh["mae"] == pytest.approx(recorded["mae"], rel=1e-9, abs=1e-12)
            assert fresh["mae_over_amplitude"] == pytest.approx(
                recorded["mae_over_amplitude"], rel=1e-9, abs=1e-12)


def test_withheld_reference_is_never_a_pass_source(summary):
    gate = summary["gates"]["trajectory_vs_bundled"]
    assert gate["result"] == "withheld"
    assert summary["verdict"]["withheld_reference_used_as_pass_source"] is False
    # the graded gate must be the independent transcription
    assert summary["gates"]["trajectory_vs_independent"]["result"] in {"pass", "fail"}


def test_protocol_constants_match_the_document(summary):
    assert summary["protocol"]["window_tau"] == list(WINDOW)
    assert summary["protocol"]["tolerance_relative_to_amplitude"] == TOL_REL
    assert summary["protocol"]["compression_limit"] == COMPRESSION_LIMIT
    doc = (ROOT / "docs" / "repro_monaghan_1994.md").read_text()
    assert f"{100 * TOL_REL:.0f} %" in doc
    assert "WITHHELD" in doc


def test_verdict_is_consistent_with_its_gates(summary):
    graded = summary["gates"]["trajectory_vs_independent"]["result"]
    assert summary["verdict"]["physical_validation"] == graded
    measured = summary["gates"]["trajectory_vs_independent"]["measured"]
    assert (measured <= TOL_REL) == (graded == "pass")


def test_compression_gate_holds(summary, runs):
    finest = min(runs, key=lambda r: r["dx"])
    assert finest["max_density_compression"] <= COMPRESSION_LIMIT
    assert summary["gates"]["compressibility"]["result"] == "pass"


def test_reference_pair_spread_is_recorded_as_a_failure(summary):
    """The reference, not the solver, fails its own reproducibility test here."""
    gate = summary["gates"]["reference_self_consistency"]
    assert gate["result"] == "fail"
    assert gate["measured"] > 2.0


def test_doi_resolution_artifact_declares_its_claim():
    path = RESULTS / "refs_verified.json"
    if not path.exists():
        pytest.skip("refs_verified.json absent: run verify_refs.py")
    payload = json.loads(path.read_text())
    claim = payload["anchor_citation_claim"]
    assert claim["volume"] == "110" and claim["pages"] == "399-406"
    if payload["crossref_search"]["resolved"]:
        selected = payload["crossref_search"]["selected"]
        assert selected["match_fields"]["author"] is True
        assert selected["match_fields"]["volume"] is True
        assert selected["match_fields"]["pages"] is True