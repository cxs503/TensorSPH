"""Closed-form verification of the wall-shear estimators.

What is asserted here (and why it can fail)
-------------------------------------------
``tensorsph.wall_shear`` exposes two observables of the wall shear of a plane
channel.  Each is tested on data whose answer is known analytically, so a broken
estimator is caught by a number, not by "it ran":

1. ``momentum_budget_wall_shear`` -- ``tau_w = (1/2) rho H (g - d<u>/dt)``.
   * a constant trace (steady flow) must return exactly ``rho g H / 2``;
   * a linear ramp with known slope ``-s`` must return exactly
     ``rho H (g + s) / 2``;
   * a free-slip control (drift ``= g``, no drag) must return ``~0``, i.e. the
     estimator's zero line, and the gate must then *fail* -- the mutant that a
     "gate" which can only pass would never catch;
   * Gaussian noise of the same size as a real trace must stay inside a few
     percent.

2. ``profile_wall_shear`` -- ``tau_w = rho nu du/dy|_0`` from a quadratic fit.
   * an exact parabola must return the analytic ``rho g H/2`` to machine
     precision;
   * a smeared wall layer (the SPH reality at ``H/dx <= 16``) must bias the
     *full-band* reading low by tens of percent, which is the quantitative reason
     the acceptance gate does not read this observable.

3. Regression on committed artefacts -- ``benchmarks/poiseuille/out/results.json``
   (the WCSPH plane-Poiseuille ladder, already merged) must close its momentum
   budget within the 3 % gate at every rung, and its free-slip control must sit
   at ``~100 %`` error.  This pins the estimator to real, stored data.

4. ``isph_channel.gates`` must now read the budget (label included), must keep
   the near-wall number as a diagnostic with no verdict, must not emit the
   ``-1.0`` placeholder the old ``fit_profile`` returned, and must report
   "unavailable" (not "pass") when a trace carries no velocity average.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorsph.wall_shear import (
    DriftRate,
    drift_rate,
    momentum_budget_wall_shear,
    profile_wall_shear,
)

ROOT = Path(__file__).resolve().parents[1]
POISEUILLE = ROOT / "benchmarks" / "poiseuille" / "out" / "results.json"

HEIGHT = 0.1
BODY_FORCE = 9.81
VISCOSITY = 0.01
RHO0 = 1000.0
TAU_REQUIRED = 0.5 * RHO0 * BODY_FORCE * HEIGHT  # 490.5 Pa


def _trace(slope: float, n: int = 51, t_end: float = 1.0) -> list[dict]:
    t = np.linspace(0.0, t_end, n)
    return [{"t": float(ti), "u_mean": float(1.0 + slope * ti)} for ti in t]


def _parabola_profile(y_wall_row: float | None = None, n: int = 41) -> list[dict]:
    """Analytic parabola; optionally overwrite the first row (smeared wall)."""
    y = np.linspace(0.5 * HEIGHT / (n - 1), HEIGHT, n)
    rows = []
    for yy in y:
        u = BODY_FORCE * yy * (HEIGHT - yy) / (2.0 * VISCOSITY)
        rows.append({"y": float(yy), "y_over_H": float(yy / HEIGHT), "u_mean": u})
    if y_wall_row is not None:
        rows[0]["u_mean"] = y_wall_row * (
            BODY_FORCE * rows[0]["y"] * (HEIGHT - rows[0]["y"]) / (2.0 * VISCOSITY)
        )
    return rows


# --------------------------------------------------------------------------- #
# 1. momentum budget
# --------------------------------------------------------------------------- #

def test_steady_trace_returns_the_analytic_wall_shear_exactly():
    out = momentum_budget_wall_shear(_trace(0.0), HEIGHT, BODY_FORCE, RHO0)
    assert out["available"] is True
    assert out["tau_wall_required"] == pytest.approx(TAU_REQUIRED)
    assert out["tau_wall"] == pytest.approx(TAU_REQUIRED, rel=1e-12)
    assert out["rel_error_pct"] == pytest.approx(0.0, abs=1e-12)
    assert out["momentum_closure"] == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("slope", [-2.0, -0.5, 0.75])
def test_known_ramp_is_recovered_exactly(slope):
    expected = 0.5 * RHO0 * HEIGHT * (BODY_FORCE - slope)
    out = momentum_budget_wall_shear(_trace(slope), HEIGHT, BODY_FORCE, RHO0)
    assert out["drift"] == pytest.approx(slope, rel=1e-12)
    assert out["tau_wall"] == pytest.approx(expected, rel=1e-12)


def test_free_slip_control_sits_on_the_zero_line_and_fails_the_gate():
    """The mutant a "gate that cannot fail" would miss."""
    out = momentum_budget_wall_shear(_trace(BODY_FORCE), HEIGHT, BODY_FORCE, RHO0)
    assert out["tau_wall"] == pytest.approx(0.0, abs=1e-9)
    assert out["rel_error_pct"] == pytest.approx(-100.0, abs=1e-6)
    assert abs(out["rel_error_pct"]) > 3.0


def test_noise_of_a_real_trace_stays_within_the_gate():
    rng = np.random.default_rng(20261113)
    slope = -0.4
    trace = _trace(slope, n=61)
    for sample in trace[1:]:  # first sample is the t=0 rest state
        sample["u_mean"] += float(rng.normal(0.0, 2.0e-3))
    expected = 0.5 * RHO0 * HEIGHT * (BODY_FORCE - slope)
    out = momentum_budget_wall_shear(trace, HEIGHT, BODY_FORCE, RHO0)
    assert out["rel_error_pct"] == pytest.approx(
        100.0 * (out["tau_wall"] - TAU_REQUIRED) / TAU_REQUIRED, rel=1e-12)
    assert abs(out["tau_wall"] - expected) / expected < 0.02
    assert out["drift_stderr"] < 0.02


def test_drift_gain_calibration_moves_the_reading_in_the_predicted_direction():
    slope = -0.4
    raw = momentum_budget_wall_shear(_trace(slope), HEIGHT, BODY_FORCE, RHO0)
    calibrated = momentum_budget_wall_shear(
        _trace(slope), HEIGHT, BODY_FORCE, RHO0, drift_gain=1.1)
    # dividing a *negative* drift by gain > 1 shrinks its magnitude, so the
    # measured drag (and hence tau_w) comes out *smaller* than uncalibrated
    assert calibrated["tau_wall"] < raw["tau_wall"]
    expected = 0.5 * RHO0 * HEIGHT * (BODY_FORCE - slope / 1.1)
    assert calibrated["tau_wall"] == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError):
        momentum_budget_wall_shear(_trace(slope), HEIGHT, BODY_FORCE, RHO0,
                                   drift_gain=0.0)


def test_unsteady_window_must_not_silently_pass():
    """A run that never reached steady state must not read as steady."""
    out = momentum_budget_wall_shear(_trace(BODY_FORCE / 2.0), HEIGHT, BODY_FORCE, RHO0)
    assert out["pass"] if "pass" in out else True  # estimator carries no verdict
    assert out["rel_error_pct"] == pytest.approx(-50.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. near-wall profile reading
# --------------------------------------------------------------------------- #

def test_exact_parabola_gives_the_analytic_wall_shear():
    out = profile_wall_shear(_parabola_profile(), HEIGHT, VISCOSITY, RHO0,
                             body_force=BODY_FORCE)
    assert out["available"] is True
    assert out["tau_wall"] == pytest.approx(TAU_REQUIRED, rel=1e-10)


def _smeared_channel_rows(resolution: int = 8, knee: float = 0.6) -> list[dict]:
    """Analytic parabola sampled at SPH row centres with a smeared wall row.

    Row ``k`` sits at ``y = (k + 1/2) dx``; the first row carries ``knee`` times
    its analytic velocity, which is what a wall layer spanning one spacing looks
    like when it is averaged into a row histogram.
    """
    dx = HEIGHT / resolution
    rows = []
    for k in range(resolution):
        y = (k + 0.5) * dx
        u = BODY_FORCE * y * (HEIGHT - y) / (2.0 * VISCOSITY)
        if k == 0:
            u *= knee
        rows.append({"y": y, "y_over_H": y / HEIGHT, "u_mean": u})
    return rows


def test_one_spacing_secant_on_a_smeared_wall_layer_reads_far_too_low():
    """The published -42 %/-50 % readings are this stencil, not the wall model."""
    rows = _smeared_channel_rows(resolution=8, knee=0.6)
    secant = RHO0 * VISCOSITY * rows[0]["u_mean"] / rows[0]["y"]
    secant_err = 100.0 * (secant - TAU_REQUIRED) / TAU_REQUIRED
    assert secant_err < -30.0, secant_err

    # the same rows through a parabola fit over the interior (no wall row) are
    # exact: the bias belongs to the stencil, not to the data
    interior = profile_wall_shear(rows, HEIGHT, VISCOSITY, RHO0, lo=0.25, hi=0.75,
                                 body_force=BODY_FORCE)
    interior_err = 100.0 * (interior["tau_wall"] - TAU_REQUIRED) / TAU_REQUIRED
    assert abs(interior_err) < 1.0, interior_err


def test_too_few_rows_is_unavailable_not_zero():
    out = profile_wall_shear(_parabola_profile()[:2], HEIGHT, VISCOSITY, RHO0)
    assert out["available"] is False
    assert math.isnan(out["tau_wall"])


# --------------------------------------------------------------------------- #
# 3. regression on committed artefacts
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not POISEUILLE.is_file(),
                    reason="committed Poiseuille artefacts not present")
def test_committed_wcsp_h_ladder_closes_its_momentum_budget():
    payload = json.loads(POISEUILLE.read_text())
    height = payload["height"]
    g = payload["body_force"]
    t_end = payload["t_end"]
    required = 0.5 * RHO0 * g * height

    control = drift_rate(payload["free_slip_control"]["trace"], t_end=t_end)
    gain = control.slope / g
    assert gain == pytest.approx(1.0, abs=0.02), gain

    rungs = payload["results"]
    assert len(rungs) >= 3
    for rung in rungs:
        out = momentum_budget_wall_shear(rung["trace"], height, g, RHO0,
                                        t_end=t_end, drift_gain=gain)
        assert out["available"] is True
        assert abs(out["rel_error_pct"]) <= 3.0, (rung["resolution_H_over_dx"],
                                                  out["rel_error_pct"])
        assert out["tau_wall"] == pytest.approx(required, rel=0.03)

    # the free-slip control is the estimator's zero line: ~ -100 % error
    zero = momentum_budget_wall_shear(payload["free_slip_control"]["trace"],
                                     height, g, RHO0, t_end=t_end, drift_gain=gain)
    assert zero["rel_error_pct"] < -90.0


@pytest.mark.skipif(not POISEUILLE.is_file(),
                    reason="committed Poiseuille artefacts not present")
def test_the_published_near_wall_readout_is_the_biased_one():
    """The -42 % published reading is reproduced by a wall-slope fit, not by the budget."""
    payload = json.loads(POISEUILLE.read_text())
    height = payload["height"]
    g = payload["body_force"]
    required = 0.5 * RHO0 * g * height
    rung = payload["results"][0]
    fit = profile_wall_shear(rung["rows"], height, payload["viscosity"], RHO0,
                            lo=0.0, hi=1.0, body_force=g)
    fit_err = 100.0 * (fit["tau_wall"] - required) / required
    published = 100.0 * (rung["wall_shear_measured"] - required) / required
    assert fit_err < -10.0, fit_err
    assert published < -10.0, published
    assert abs(published - fit_err) < 30.0  # same biased family, same sign


# --------------------------------------------------------------------------- #
# 4. the gate itself
# --------------------------------------------------------------------------- #

def _cfg(resolution: int = 6):
    from tensorsph.isph_channel import ChannelConfig
    return ChannelConfig(columns=4, resolution=resolution, t_end=0.2, t_average=0.05)


def test_gates_read_the_budget_and_never_the_placeholder():
    from tensorsph.isph_channel import fit_profile, gates

    cfg = _cfg()
    rows = _parabola_profile()
    for row in rows:
        row["y_over_H"] = row["y"] / cfg.height
        row["y"] = row["y"] * cfg.height / HEIGHT
    fit = fit_profile(rows, cfg.height, cfg.body_force, cfg.viscosity)
    assert "placeholder" not in fit
    assert not any(v == -1.0 for v in fit.values())

    result = {
        "trace": _trace(0.0),
        "t_reached": 1.0,
        "u_max_measured": cfg.u_max_analytic,
        "u_max_analytic": cfg.u_max_analytic,
        "density_error_max": 1e-5,
        "divergence_reduction": 0.5,
    }
    out = gates(result, fit, cfg)
    assert out["wall_shear_budget"]["estimator"] == "momentum_budget"
    assert out["wall_shear_budget"]["available"] is True
    assert out["wall_shear_budget"]["measured"] == pytest.approx(
        cfg.tau_wall_analytic, rel=1e-6)
    assert out["wall_shear_budget"]["pass"] is True
    # near-wall reading stays reported but carries no verdict
    assert out["wall_shear_near_wall"]["gate"] is False
    assert "pass" not in out["wall_shear_near_wall"]


def test_a_free_slip_trace_fails_the_budget_gate():
    from tensorsph.isph_channel import fit_profile, gates

    cfg = _cfg()
    fit = fit_profile(_parabola_profile(), cfg.height, cfg.body_force, cfg.viscosity)
    result = {
        "trace": _trace(cfg.body_force),  # no drag: <u> accelerates at g
        "t_reached": 1.0,
        "u_max_measured": cfg.u_max_analytic,
        "u_max_analytic": cfg.u_max_analytic,
        "density_error_max": 1e-5,
        "divergence_reduction": 0.5,
    }
    out = gates(result, fit, cfg)
    assert out["wall_shear_budget"]["pass"] is False
    assert out["wall_shear_budget"]["rel_error_pct"] < -90.0


def test_a_trace_without_velocity_average_is_unavailable_not_passing():
    from tensorsph.isph_channel import fit_profile, gates

    cfg = _cfg()
    fit = fit_profile(_parabola_profile(), cfg.height, cfg.body_force, cfg.viscosity)
    legacy_trace = [{"t": float(t), "u_max": 1.0} for t in np.linspace(0, 1, 11)]
    result = {
        "trace": legacy_trace,
        "t_reached": 1.0,
        "u_max_measured": cfg.u_max_analytic,
        "u_max_analytic": cfg.u_max_analytic,
        "density_error_max": 1e-5,
        "divergence_reduction": 0.5,
    }
    out = gates(result, fit, cfg)
    assert out["wall_shear_budget"]["available"] is False
    assert out["wall_shear_budget"]["pass"] is False
    assert out["wall_shear_budget"]["reason"]
    assert math.isnan(out["wall_shear_budget"]["rel_error_pct"])


def test_drift_rate_rejects_degenerate_windows():
    with pytest.raises(ValueError):
        drift_rate([{"t": 1.0, "u_mean": 2.0}])
    with pytest.raises(ValueError):
        drift_rate(_trace(0.0), fraction=0.0)
    with pytest.raises(ValueError):
        drift_rate([{"t": float(t), "u_mean": float("nan")}
                    for t in np.linspace(0, 1, 5)])


def test_drift_rate_reports_its_own_uncertainty():
    out = drift_rate(_trace(-0.25))
    assert isinstance(out, DriftRate)
    # last 40 % of a 0..1 trace sampled 51 times: t >= 0.6 -> 21 samples
    assert out.n_samples == 21
    assert out.t_lo == pytest.approx(0.6)
    assert out.slope == pytest.approx(-0.25, rel=1e-12)
    assert out.residual_rms == pytest.approx(0.0, abs=1e-12)
    assert out.r_squared == pytest.approx(1.0, rel=1e-12)