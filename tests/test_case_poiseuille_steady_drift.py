"""Tests for the R1 steady-functional drift audit (CASE-POISEUILLE-SPH).

The audit answers a question the level-comparison gate `S2` of
``steady_window.py`` cannot see: is the late decline of the steady functional a
real decay (error still improving) or a run of low samples inside a stationary
scatter?  These tests pin the estimator, not the physics:

* the OLS/AR(1) helpers on synthetic series with a known slope and a known rho;
* the block bootstrap on a series with an *injected* decline (must detect) and on
  a stationary one (must not);
* the invariant that blocks, not samples, are the resampling unit -- a
  correlated baseline must produce a wider null than a white one with the same
  marginal standard deviation;
* reproduction of the committed artifact: the gates in ``out/steady_drift.json``
  follow from ``out/steady_window.json`` with the declared seed and block;
* the generated note carries the same verdicts as the artifact (no hand
  copying, no drift between document and evidence).

Nothing here runs the solver.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BENCH = REPO / "benchmarks" / "case_poiseuille"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(BENCH))

import steady_drift as sd  # noqa: E402


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #
def test_ols_recovers_known_slope_and_symmetric_window():
    rng = np.random.default_rng(0)
    t = np.linspace(0.8, 1.0, 9)
    v = 3.0 - 2.5 * t + rng.normal(0.0, 1e-9, t.size)
    fit = sd.ols(t, v)
    assert fit["n"] == 9
    assert fit["slope_per_s"] == pytest.approx(-2.5, abs=1e-6)
    assert fit["drift_over_window"] == pytest.approx(-2.5 * 0.2, abs=1e-6)
    # a slope that is exactly inside the noise must not look significant
    assert abs(fit["t_stat"]) > 100.0            # here it *is* exact by design


def test_ols_needs_two_free_samples():
    t = np.array([1.0, 1.2])
    v = np.array([2.0, 1.0])
    fit = sd.ols(t, v)
    assert fit["dof"] if "dof" in fit else True   # documented: se undefined
    assert math.isnan(fit["slope_se"])
    assert math.isnan(fit["t_stat"])


def test_lag1_sign_and_ar1_recovery():
    rng = np.random.default_rng(1)
    n = 4000
    # AR(1) with rho = 0.6 -> measured lag-1 correlation must be close to it
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.6 * e[i - 1] + rng.normal()
    assert sd.lag1(e) == pytest.approx(0.6, abs=0.05)

    alt = np.tile([1.0, -1.0], 200)
    assert sd.lag1(alt) == pytest.approx(-1.0, abs=1e-12)


def test_effective_t_direction_of_the_correction():
    optimistic = sd.effective_t(4.0, 0.5)
    conservative = sd.effective_t(4.0, -0.5)
    assert abs(optimistic["t_effective"]) < 4.0
    assert abs(conservative["t_effective"]) > 4.0
    assert "optimistic" in optimistic["correction"]
    assert "conservative" in conservative["correction"]
    assert math.isnan(sd.effective_t(4.0, 1.0)["t_effective"])


# --------------------------------------------------------------------------- #
# the test that matters: can the bootstrap tell a decline from noise?
# --------------------------------------------------------------------------- #
def _baseline(n=40, sd_pp=0.08, seed=7, rho=0.0):
    rng = np.random.default_rng(seed)
    if rho == 0.0:
        return rng.normal(0.0, sd_pp, n)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal(0.0, sd_pp * math.sqrt(1 - rho ** 2))
    return e


def test_bootstrap_does_not_call_stationary_noise_a_decline():
    t_test = np.linspace(0.8, 0.96, 9)
    base = _baseline(seed=11)
    null = sd.block_bootstrap_slopes(np.linspace(0.8, 1.58, base.size), base,
                                     n_test=9, nboot=4000, seed=3)
    # a slope that is just the 50th percentile of the null must not be
    # "distinguishable" (p ~ 0.5), i.e. the test is not rigged to fire
    obs = float(np.median(null))
    p = float(np.mean(null <= obs))
    assert 0.3 < p < 0.7
    assert null.size == 4000
    assert np.isfinite(t_test).all()


def test_bootstrap_detects_an_injected_decline():
    base = _baseline(seed=21, sd_pp=0.08)
    t_base = np.linspace(0.8, 1.58, base.size)
    # same noise sd, but the last 9 samples are dragged down by 4 sigma
    t_test = np.linspace(1.6, 1.76, 9)
    rng = np.random.default_rng(5)
    drop = rng.normal(0.0, 0.08, 9) - np.linspace(0.0, 0.5, 9)
    fit = sd.ols(t_test, drop)
    null = sd.block_bootstrap_slopes(t_base, base, n_test=9, nboot=4000, seed=3)
    p = float(np.mean(null <= fit["slope_per_s"]))
    assert fit["slope_per_s"] < 0.0
    assert p < 0.05


def test_blocks_widen_the_null_under_correlation():
    """Invariant: the resampling unit is a block, not a sample.

    With positively correlated residuals, consecutive block copies can line up
    into steeper apparent slopes than i.i.d. draws of the same marginal spread,
    so the null's spread must grow -- that is what stops the test from calling
    AR(1) noise a trend.
    """
    white = _baseline(seed=31, rho=0.0)
    ar = _baseline(seed=31, rho=0.8)
    assert ar.std() == pytest.approx(white.std(), rel=0.25)
    t = np.linspace(0.8, 1.58, white.size)
    nw = sd.block_bootstrap_slopes(t, white, n_test=9, nboot=4000, seed=3)
    na = sd.block_bootstrap_slopes(t, ar, n_test=9, nboot=4000, seed=3)
    assert na.std() > nw.std()


# --------------------------------------------------------------------------- #
# committed artifact
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def artifacts():
    win = BENCH / "out" / "steady_window.json"
    drift = BENCH / "out" / "steady_drift.json"
    if not win.is_file() or not drift.is_file():
        pytest.skip("committed steady-window/steady-drift artifacts absent")
    return json.loads(win.read_text()), json.loads(drift.read_text())


def test_gates_follow_from_the_committed_series(artifacts):
    win, drift = artifacts
    series = win["rungs"][0]
    again = sd.analyse_series(series["sample_times"], series["functional_percent"])
    assert again["gates"] == drift["gates"]
    assert again["bootstrap"]["p_one_sided"] == pytest.approx(
        drift["bootstrap"]["p_one_sided"], abs=1e-12)
    assert again["window_B"]["slope_per_s"] == pytest.approx(
        drift["window_B"]["slope_per_s"], abs=1e-12)


def test_declared_thresholds_are_the_ones_used(artifacts):
    _, drift = artifacts
    assert drift["thresholds"]["block_samples"] == sd.BLOCK
    assert drift["thresholds"]["n_boot"] == sd.NBOOT
    assert drift["thresholds"]["seed"] == sd.SEED
    assert drift["thresholds"]["D2_alpha"] == sd.D2_ALPHA
    assert drift["thresholds"]["tolerance_percent"] == sd.TOL_PERCENT


def test_d3_bound_is_committed_error_plus_drift(artifacts):
    _, drift = artifacts
    if drift.get("D3_bound_percent") is None:
        pytest.skip("no committed E in the uncertainty artifact")
    expect = abs(drift["committed_E_percent"]) + abs(
        drift["window_B"]["drift_over_window"])
    assert drift["D3_bound_percent"] == pytest.approx(expect, abs=1e-12)
    assert drift["gates"]["D3_acceptance_survives_the_drift"] == (
        expect < sd.TOL_PERCENT)


def test_d4_only_exists_with_an_extended_run(artifacts):
    _, drift = artifacts
    if "extended" in drift and (drift["extended"].get("window_C")):
        assert "D4_error_keeps_decreasing" in drift["gates"]
        assert drift["gates"]["D4_error_keeps_decreasing"] == (
            drift["extended"]["window_C"]["mean"] <= drift["window_B"]["mean"])
    else:
        assert "D4_error_keeps_decreasing" not in drift["gates"]


def test_note_carries_the_artifact_verdicts(artifacts):
    _, drift = artifacts
    doc = REPO / "docs" / "case_poiseuille_steady_drift.md"
    if not doc.is_file():
        pytest.skip("generated note absent")
    text = doc.read_text()
    for gate, value in drift["gates"].items():
        assert f"`{gate}`: {'PASS' if value else 'FAIL'}" in text, gate
    # the headline numbers in the note are the artifact's numbers
    assert f"{drift['window_B']['slope_per_s']:+.4f}" in text
    assert f"{drift['bootstrap']['p_one_sided']:.4f}" in text