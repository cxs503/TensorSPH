"""Tests for the Kleefsman-H2 arrival-deficit attribution.

Two kinds of assertions:

* **invariants** on the pure helpers, driven by synthetic fields whose answer is
  known analytically (the fan-locus mapping must be invertible and must return
  the threshold for a front that moves at the threshold locus; Richardson must
  recover an exact quadratic; the crossing finder must interpolate; a shift
  pinned at the search bound must be reported as undecidable);
* **regressions** on the committed artifacts: the analysis must reproduce the
  arrival times of the committed runs (0.4443 / 0.4541 / 0.4841 s) and must
  keep the honest verdicts (arrival FAIL, peak not comparable, station shift
  bounded strictly inside (0, 1)).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "kleefsman_obstacle"))

import arrival_attribution as A  # noqa: E402

RES = ROOT / "benchmarks/kleefsman_obstacle/results"


# --------------------------------------------------------------------------- #
# analytic mapping
# --------------------------------------------------------------------------- #
def test_locus_speed_at_zero_depth_is_the_classical_dry_bed_front():
    assert A.locus_speed(h=0.0) == pytest.approx(2.0 * math.sqrt(A.G * A.H0), rel=1e-12)


def test_locus_speed_decreases_monotonically_with_depth():
    speeds = [A.locus_speed(h=h) for h in (0.0, 0.005, 0.02, 0.05, 0.20)]
    assert all(a > b for a, b in zip(speeds, speeds[1:])), speeds
    # a deeper locus can never move upstream
    assert A.locus_speed(h=0.02) == pytest.approx(3.317, abs=2e-3)


def test_depth_speed_mapping_round_trips():
    for h in (0.004, 0.02, 0.043, 0.09):
        assert A.depth_from_speed(A.locus_speed(h=h)) == pytest.approx(h, rel=1e-12)


def test_effective_depth_of_a_threshold_locus_is_the_threshold():
    """A front that arrives exactly on the threshold locus must map back to it."""
    t_a = A.STATION_X - A.GATE_X
    t_a /= A.locus_speed()
    assert A.effective_front_depth(t_a, A.STATION_X - A.GATE_X) == pytest.approx(
        A.THRESHOLD, rel=1e-12)
    # and a 25 % late arrival must need a thicker front (the diagnostic must bite)
    late = A.effective_front_depth(1.25 * t_a, A.STATION_X - A.GATE_X)
    assert late > A.THRESHOLD
    assert late / A.THRESHOLD == pytest.approx(2.15, abs=0.35)


def test_effective_depth_is_nan_for_missing_arrival():
    assert math.isnan(A.effective_front_depth(None, 1.198))
    assert math.isnan(A.effective_front_depth(0.0, 1.198))


# --------------------------------------------------------------------------- #
# signal helpers
# --------------------------------------------------------------------------- #
def test_first_crossing_interpolates_a_ramp():
    t = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 0.01, 0.03])
    assert A.first_crossing(t, y, 0.02) == pytest.approx(0.75)


def test_first_crossing_returns_none_when_never_reached():
    t = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 0.005, 0.01])
    assert A.first_crossing(t, y, 0.02) is None


def test_first_crossing_handles_a_signal_that_starts_above():
    t = np.array([0.0, 0.5])
    y = np.array([0.05, 0.06])
    assert A.first_crossing(t, y, 0.02) == pytest.approx(0.0)


def test_first_crossing_uses_the_first_of_several_crossings():
    t = np.array([0.0, 0.3, 0.6, 0.9])
    y = np.array([0.0, 0.05, 0.01, 0.05])
    assert A.first_crossing(t, y, 0.02) == pytest.approx(0.12)


def test_least_squares_speed_recovers_a_synthetic_front():
    t = np.linspace(0.0, 1.0, 101)
    x = 1.228 + 3.4 * t
    assert A.least_squares_speed(t, x, 1.3, 2.4) == pytest.approx(3.4, rel=1e-9)
    # samples outside the window must not be used
    x2 = np.where(t > 0.5, 1.228 + 30.0 * (t - 0.5) + 3.4 * 0.5, 1.228 + 3.4 * t)
    assert A.least_squares_speed(t, x2, 1.3, 2.0) == pytest.approx(3.4, rel=1e-6)


def test_rise_slope_recovers_a_ramp_and_rejects_a_short_window():
    t = np.linspace(0.0, 1.0, 51)
    h = np.clip(0.6 * (t - 0.2), 0.0, None)
    assert A.rise_slope(t, h, 0.3, 0.6) == pytest.approx(0.6, rel=1e-6)
    assert A.rise_slope(t, h, 0.31, 0.32) is None


# --------------------------------------------------------------------------- #
# Richardson
# --------------------------------------------------------------------------- #
def test_richardson_recovers_an_exact_second_order_limit():
    dx = np.array([0.04, 0.02, 0.01])
    f = 0.44 + 12.5 * dx ** 2
    p, exact = A.richardson(dx, f)
    assert p == pytest.approx(2.0, rel=1e-6)
    assert exact == pytest.approx(0.44, abs=1e-12)


def test_richardson_refuses_a_non_monotone_ladder():
    p, exact = A.richardson([0.04, 0.02, 0.01], [0.4255, 0.5175, 0.3773])
    assert math.isnan(p) and math.isnan(exact)


def test_richardson_rejects_a_wrong_number_of_grids():
    with pytest.raises(ValueError):
        A.richardson([0.04, 0.02], [0.5, 0.4])


# --------------------------------------------------------------------------- #
# phase-lag fit
# --------------------------------------------------------------------------- #
def test_optimal_shift_recovers_a_synthetic_lag():
    t_ref = np.linspace(0.0, 1.5, 1501)
    h_ref = 0.3 * np.exp(-((t_ref - 0.6) / 0.2) ** 2)
    t_sim = np.linspace(0.0, 1.5, 400)
    h_sim = 0.3 * np.exp(-((t_sim - 0.675) / 0.2) ** 2)   # 75 ms late
    delta, rms0, rms1 = A.optimal_shift(t_ref, h_ref, t_sim, h_sim, (0.3, 1.2))
    assert delta == pytest.approx(0.075, abs=2e-3)
    assert rms1 < 0.25 * rms0
    assert A.shift_is_decidable(delta, rms0, rms1)


def test_shift_is_undecidable_when_pinned_at_the_search_bound():
    assert not A.shift_is_decidable(-0.20, 0.1, 0.1)
    assert not A.shift_is_decidable(0.0, 0.1, 0.1)
    assert A.shift_is_decidable(0.07, 0.1, 0.02)


# --------------------------------------------------------------------------- #
# regressions on the committed artifacts
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def report():
    return A.analyse()


def test_analysis_reproduces_the_committed_arrival_times(report):
    got = {g["tag"]: g["arrival_level_s"] for g in report["grids"]}
    assert got["dx0.01"] == pytest.approx(0.4443, abs=5e-4)
    assert got["dx0.02"] == pytest.approx(0.4541, abs=5e-4)
    assert got["dx0.04"] == pytest.approx(0.4841, abs=5e-4)


def test_level_arrival_never_precedes_the_kinematic_front(report):
    for g in report["grids"]:
        assert g["arrival_front_s"] < g["arrival_level_s"]
        # the level lag is a small part of the deficit, not the whole story
        assert 0.0 < g["level_minus_front_s"] < 0.25 * (g["arrival_level_s"] - 0.3591)


def test_front_speed_is_dx_independent_while_the_arrival_is_not(report):
    speeds = [g["front_speed_m_s"] for g in report["grids"]]
    assert max(speeds) - min(speeds) < 0.10 * np.mean(speeds)
    assert all(s < 0.80 * 2.0 * math.sqrt(A.G * A.H0) for s in speeds)


def test_arrival_ladder_is_monotone_and_still_fails_after_extrapolation(report):
    res = report["resolution_test"]
    assert res["observed_order"] == pytest.approx(1.62, abs=0.05)
    assert res["continuum_arrival_s"] == pytest.approx(0.4396, abs=1e-3)
    assert res["continuum_rel_error_vs_reference"] > 0.10
    assert "converged-but-wrong" in res["verdict"]


def test_compressibility_test_keeps_its_falsifying_ratio(report):
    for grid, v in report["compressibility_test"]["by_grid"].items():
        assert v["arrival_change_rel"] < 0.01
        assert v["peak_change_rel"] > 0.10
        assert v["compression_ratio"] > 3.0


def test_station_shift_explains_only_part_of_the_deficit(report):
    budget = report["station_sensitivity"]["slab"]["deficit_budget"]
    for tag, v in budget.items():
        frac = v["fraction_explained_by_shift_to_upstream_station"]
        assert 0.0 < frac < 1.0, (tag, frac)
        # the coordinate that would explain everything is upstream of the box face
        assert v["x_needed_to_explain_deficit_m"] < A.STATION_X
        assert v["x_needed_to_explain_deficit_m"] > A.UPSTREAM_STATION_X - 0.30


def test_decision_stays_honest(report):
    dec = report["decision"]
    assert dec["passes"] is False
    assert dec["worst_rel_error"] > 0.10
    assert dec["peak_comparable"] is False
    assert 3 <= len(dec["falsified"]) <= 5
    assert len(dec["unresolved"]) == 2
    assert report["canonical_reference"]["arrival_time_s"] == pytest.approx(0.3591)


def test_reference_curves_map_back_onto_the_threshold_order(report):
    """The fan-locus gauge must be self-consistent on the reference curves."""
    for name in ("experiment", "comflow"):
        h_eff = report["reference"][name]["implied_front_depth_m"]
        assert h_eff == pytest.approx(A.THRESHOLD, rel=0.35), (name, h_eff)


def test_shape_test_is_flagged_undecidable_rather_than_claimed(report):
    for tag, v in report["shape_test"].items():
        if not v["decidable"]:
            assert v["not_decidable_reason"]
            assert "top-3" in v["not_decidable_reason"]


def test_committed_json_matches_a_fresh_run(report):
    committed = json.loads((RES / "arrival_attribution/attribution.json").read_text())
    assert committed["decision"] == report["decision"]
    assert [g["tag"] for g in committed["grids"]] == [g["tag"] for g in report["grids"]]