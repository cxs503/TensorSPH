"""Tests for the R1 steady-window stationarity check and the scale-consistent band.

Two defects motivated this file, and each gets an executable guard:

1. **Mixed units in ``U_input``** (``benchmarks/case_poiseuille/uncertainty.py``).
   The R1 functionals have truth ``0`` and declared scale ``1.0``, so the
   comparison error ``E`` is a percentage point of that scale.  The band was
   built from the same samples divided by their *mean* -- the quantity that goes
   to zero under refinement.  The result was an anti-convergent ``U_input``
   (6.02 -> 17.01 -> 44.91 % over H/dx = 8,12,16) and a gate U3b that could never
   pass.  The tests below pin (a) the identity ``U_input == 100*half_range/scale``
   for the graded rows, (b) the monotonicity that made the old estimator invalid,
   and (c) that the old number is still published for audit and is the *larger*
   (i.e. more conservative) of the two.
2. **An unverified stationarity assumption** (``steady_window.py``).  A half range
   taken over a window that is still filling is not a repeatability.  The tests
   pin the drift/noise discrimination of the trend test and the analytic fill
   bound, and check the committed extended-run artifact against its gates.

Run:  pytest tests/test_case_poiseuille_steady_window.py -v
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "case_poiseuille"))

from tensorsph import womersley as wm  # noqa: E402
from tensorsph.verification.uncertainty import repeatability  # noqa: E402
import uncertainty as case_unc  # noqa: E402
import steady_window as sw  # noqa: E402

CASE = REPO / "benchmarks" / "case_poiseuille"
RESULT = CASE / "out" / "result.json"
UNC = CASE / "out" / "uncertainty.json"
STEADY = CASE / "out" / "steady_window.json"

pytestmark = pytest.mark.skipif(not RESULT.is_file(), reason="ladder artifact absent")


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads(RESULT.read_text())


@pytest.fixture(scope="module")
def published() -> dict:
    return json.loads(UNC.read_text())


@pytest.fixture(scope="module")
def steady() -> dict:
    if not STEADY.is_file():
        pytest.skip("extended-window artifact absent (run steady_window.py)")
    return json.loads(STEADY.read_text())


# --------------------------------------------------------------------------- #
# 1. the functional is the one the case grades
# --------------------------------------------------------------------------- #
def test_functional_is_zero_on_the_analytic_parabola() -> None:
    y = np.linspace(0.0, sw.case.HEIGHT, 41)
    u_par = wm.steady_velocity(y, sw.case.HEIGHT, sw.case.BODY_FORCE,
                               sw.case.VISCOSITY)
    assert sw.steady_functional(u_par, y, sw.case.HEIGHT) == pytest.approx(0.0, abs=1e-12)


def test_functional_recovers_a_prescribed_relative_error() -> None:
    """A profile scaled by ``1 + eps*y/H`` must give ``RMS(eps*y/H)`` exactly.

    This pins the definition (relative, interior band, RMS) against a case whose
    answer is known by hand -- a functional that silently changed its
    normalisation would move every published error in the case.
    """
    height = sw.case.HEIGHT
    y = np.linspace(0.0, height, 121)
    u_par = wm.steady_velocity(y, height, sw.case.BODY_FORCE, sw.case.VISCOSITY)
    eps = 0.07
    u = u_par * (1.0 + eps * y / height)
    interior = (y / height >= sw.case.INTERIOR_LO) & (y / height <= sw.case.INTERIOR_HI)
    expect = 100.0 * math.sqrt(float(np.mean((eps * y[interior] / height) ** 2)))
    assert sw.steady_functional(u, y, height) == pytest.approx(expect, rel=1e-12)


def test_sample_grid_reproduces_the_committed_window() -> None:
    """Window A must be sampled on the same 0.02 s grid the ladder used."""
    times = sw.sample_times()
    a = [t for t in times if sw.WINDOW_A[0] - 1e-9 <= t <= sw.WINDOW_A[1] + 1e-9]
    assert len(a) == 11, "the committed ladder took 11 steady samples"
    assert a[0] == pytest.approx(0.80) and a[-1] == pytest.approx(1.00)
    assert times[-1] == pytest.approx(sw.T_END)


# --------------------------------------------------------------------------- #
# 2. drift vs noise: the stationarity test has to discriminate
# --------------------------------------------------------------------------- #
def _synthetic_rung(values, times, resolution=12):
    return {"resolution_H_over_dx": resolution, "dt": 1e-4, "steps": 1,
            "wall_clock_s": 0.0, "n_fluid": 1, "max_mach": 0.0,
            "sample_times": list(times), "functional_percent": list(values)}


def test_a_ramp_is_rejected_as_non_stationary() -> None:
    """Six e-folding times of decay left = 0.25 %, so a 5 % ramp is a transient."""
    times = sw.sample_times()
    ramp = [1.0 + 0.05 * (t - 0.8) / 1.2 for t in times]
    g = sw.analyse_rung(_synthetic_rung(ramp, times), None, 0.0)
    assert g["gates"]["S2_window_is_stationary"] is False
    assert abs(g["trend_window_A"]["t_stat"]) > 2.0


def test_flat_noise_is_accepted_as_stationary() -> None:
    """A flat mean with +/-2 % noise must pass: scatter is not a drift."""
    times = sw.sample_times()
    rng = np.random.default_rng(20261113)
    noisy = [1.0 + 0.02 * float(rng.standard_normal()) for _ in times]
    g = sw.analyse_rung(_synthetic_rung(noisy, times), None, 0.0)
    assert g["gates"]["S2_window_is_stationary"] is True
    assert abs(g["trend_window_A"]["t_stat"]) < 3.0


def test_trend_test_recovers_a_known_slope() -> None:
    times = [0.0, 0.1, 0.2, 0.3, 0.4]
    values = [1.0, 1.2, 1.4, 1.6, 1.8]
    tr = sw.trend_test(times, values)
    assert tr["slope_per_s"] == pytest.approx(2.0, rel=1e-9)
    assert tr["drift_over_window"] == pytest.approx(0.8, rel=1e-9)
    assert tr["n"] == 5 and tr["dof"] == 3


def test_fill_bound_is_the_analytic_decay_rate() -> None:
    """e-folding time of the slowest Boussinesq mode is H^2/(nu pi^2)."""
    fill = sw.fill_prediction(0.8, sw.case.HEIGHT, sw.case.VISCOSITY)
    expect_efold = sw.case.HEIGHT ** 2 / (sw.case.VISCOSITY * math.pi ** 2)
    assert fill["e_folding_s"] == pytest.approx(expect_efold, rel=1e-12)
    assert fill["predicted_fill_ratio"] == pytest.approx(math.exp(-0.8 / expect_efold),
                                                         rel=1e-12)
    assert fill["predicted_fill_ratio"] < 0.01     # gate S3's threshold


# --------------------------------------------------------------------------- #
# 3. the published stationarity artifact
# --------------------------------------------------------------------------- #
def test_extended_run_reproduces_the_committed_window(steady: dict, result: dict) -> None:
    """Same protocol, same answer: otherwise the extended run is a different case."""
    committed = {r["resolution_H_over_dx"]: r for r in result["constant_ladder"]["runs"]}
    for g in steady["rungs"]:
        r = g["resolution_H_over_dx"]
        assert g["S1_deviation_percent"] is not None, r
        assert g["S1_deviation_percent"] <= steady["thresholds"]["S1_reproduce_percent"], r
        # independent recomputation from the two artifacts, not a re-read
        ref = float(np.mean(committed[r]["steady_rms_samples_pct"]))
        assert g["window_A"]["mean_percent"] == pytest.approx(ref, rel=0.01), r


def test_all_predeclared_stationarity_gates_pass(steady: dict) -> None:
    for key, ok in steady["gates"].items():
        assert ok is True, key
    assert steady["all_gates_pass"] is True


def test_windows_A_and_B_agree_within_the_scatter(steady: dict) -> None:
    for g in steady["rungs"]:
        d = abs(g["window_B"]["mean_percent"] - g["window_A"]["mean_percent"])
        assert d <= g["between_window"]["union_half_range_percent_of_scale"] + 1e-12, \
            g["resolution_H_over_dx"]
        assert g["u_input_percent_of_scale"] <= 1.0, g["resolution_H_over_dx"]


def test_between_window_band_supports_the_acceptance(steady: dict) -> None:
    for g in steady["rungs"]:
        assert g["U_V_percent"] <= steady["thresholds"]["tolerance_percent"], \
            g["resolution_H_over_dx"]


# --------------------------------------------------------------------------- #
# 4. the scale-consistent band (the defect fix)
# --------------------------------------------------------------------------- #
def test_r1_u_input_is_the_scatter_in_percent_of_scale(result: dict,
                                                       published: dict) -> None:
    """``U_input`` must be ``100 * half_range / scale`` with scale = 1.0.

    The samples are stored in per cent of that scale already, so the per-cent
    figure is the raw half range -- no division by the sample mean.
    """
    info = published["r1_input_uncertainty"]
    assert info["scale"] == 1.0
    finest = max(result["constant_ladder"]["runs"],
                 key=lambda r: r["resolution_H_over_dx"])
    rep = repeatability(finest["steady_rms_samples_pct"])
    assert info["steady_half_range_percent_of_scale"] == pytest.approx(
        rep["half_range"], rel=1e-12)
    assert info["steady_half_range_percent_of_mean"] == pytest.approx(
        rep["half_range_percent_of_mean"], rel=1e-12)


def test_published_band_uses_the_scale_consistent_value(published: dict) -> None:
    info = published["r1_input_uncertainty"]
    q = published["r1_quantities"]["r1_steady_profile_rms"]
    v = q["validation"]
    assert v["U_input_percent"] == pytest.approx(
        info["steady_half_range_percent_of_scale"], rel=1e-12)
    assert v["U_input_percent"] != pytest.approx(
        info["steady_half_range_percent_of_mean"], rel=1e-6)
    # and U_V must then be the quadrature sum of two same-unit numbers
    assert v["U_V_percent"] == pytest.approx(
        math.hypot(v["U_num_percent"], v["U_input_percent"]), rel=1e-9)


def test_mean_normalised_band_is_anti_convergent(published: dict) -> None:
    """The evidence that the old estimator was invalid, not merely different.

    Dividing a fixed noise floor by a functional that decays under refinement
    makes the "uncertainty" grow with refinement.  Both statements are checked:
    the mean-normalised series is monotone increasing, the scale-normalised one
    stays inside a factor of two over the same ladder.
    """
    table = published["r1_input_uncertainty"]["rung_table"]
    assert [r["resolution_H_over_dx"] for r in table] == [8, 12, 16]
    by_mean = [r["half_range_percent_of_mean"] for r in table]
    by_scale = [r["half_range_percent_of_scale"] for r in table]
    assert by_mean[0] < by_mean[1] < by_mean[2], by_mean
    assert max(by_scale) / min(by_scale) < 2.0, by_scale
    # the mean-normalised value at the finest rung is the 44.91 % released earlier
    assert by_mean[-1] == pytest.approx(44.91, abs=0.05)


def test_U3b_now_passes_and_the_old_band_is_still_published(published: dict) -> None:
    gates = published["gates"]
    assert gates["U2b_R1_validation_E_leq_U_V"] is True
    assert gates["U3b_R1_band_supports_5pct"] is True
    assert gates["U4_r1_u_input_shares_the_scale_of_E"] is True
    legacy = published["r1_input_uncertainty"]["legacy_normalization"]
    assert legacy["used_for_grading"] is False
    assert legacy["value_percent"] > published["r1_input_uncertainty"][
        "steady_half_range_percent_of_scale"]


def test_corrected_band_is_not_an_underestimate_of_the_scatter(published: dict) -> None:
    """Sanity on direction: the graded band is the single-sample half range.

    A window-mean functional scatters less than a single sample, so using the
    single-sample half range over-states ``U_input`` -- the conservative side.
    If this ever flipped, the band would be optimistic and the case would be
    certifying more than it measured.
    """
    q = published["r1_quantities"]["r1_steady_profile_rms"]
    assert q["validation"]["U_input_percent"] >= published[
        "r1_input_uncertainty"]["rung_table"][-1][
        "half_range_percent_of_scale"] - 1e-12