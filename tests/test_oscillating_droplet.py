"""Tests for the oscillating-droplet (2-D Rayleigh-Lamb) benchmark.

Two layers, deliberately separated:

* **Analysis layer** - the tools that turn a time series into a frequency are
  scored against *closed-form* inputs whose answer is known exactly (a sampled
  cosine, a decaying cosine with a prescribed ``omega``).  This is the layer that
  can silently lie about a simulation, so it gets exact tests.  See the playbook
  entry "verify the verifier first": build a解析 benchmark for the tool before
  trusting its verdict on the solver.
* **Physics layer** - the analytic reference itself (2-D vs 3-D coefficient), the
  initial-condition geometry, and a short end-to-end run that must record an
  oscillation and must *not* report success when the solution diverges.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1] / "benchmarks" / "oscillating_droplet"

_spec = importlib.util.spec_from_file_location("osc_run", HERE / "run.py")
osc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(osc)


# --------------------------------------------------------------------------- #
# analytic reference
# --------------------------------------------------------------------------- #
def test_two_dimensional_rayleigh_lamb_coefficient():
    """omega_2^2 = 6 sigma/(rho R^3) in 2-D, and the 3-D value is *different*.

    Pinning both is the point: the 3-D result is ``n(n-1)(n+2) = 8`` for n=2,
    so substituting it is a silent 15 % frequency error (sqrt(8/6) = 1.155).
    """
    n, sigma, rho, R = 2, 7.2, 1000.0, 0.02
    omega = osc.rayleigh_lamb_omega(n, sigma, rho, R)
    assert omega == pytest.approx(math.sqrt(6 * sigma / (rho * R ** 3)), rel=1e-14)

    omega_3d_coefficient = math.sqrt(8 * sigma / (rho * R ** 3))
    assert omega_3d_coefficient / omega == pytest.approx(math.sqrt(8 / 6), rel=1e-12)


def test_mode_one_is_rejected():
    """n=1 is a rigid translation and carries no restoring force."""
    with pytest.raises(ValueError):
        osc.rayleigh_lamb_omega(1, 7.2, 1000.0, 0.02)


def test_frequency_scales_as_sqrt_sigma_and_R_to_the_minus_three_halves():
    base = osc.rayleigh_lamb_omega(2, 7.2, 1000.0, 0.02)
    assert osc.rayleigh_lamb_omega(2, 4 * 7.2, 1000.0, 0.02) == pytest.approx(2 * base, rel=1e-12)
    assert osc.rayleigh_lamb_omega(2, 7.2, 1000.0, 0.04) == pytest.approx(
        base / 2 ** 1.5, rel=1e-12)


# --------------------------------------------------------------------------- #
# geometry and initial condition
# --------------------------------------------------------------------------- #
def test_ellipse_is_area_preserving_and_has_the_requested_aspect():
    """a*b = R^2(1-eps^2) and the measured particle cloud follows it."""
    R, dx, rho0, eps = 0.02, 0.002, 1000.0, 0.08
    pos, mass = osc.elliptical_droplet(R * (1 + eps), R * (1 - eps), dx, rho0)
    # total mass must equal rho0 * area, within the discretisation of the rim
    area = math.pi * R * R * (1 - eps ** 2)
    assert float(mass.sum()) == pytest.approx(rho0 * area, rel=0.03)
    # second moments of a uniform ellipse give a^2/4 and b^2/4
    sxx = float((mass * pos[:, 0] ** 2).sum())
    syy = float((mass * pos[:, 1] ** 2).sum())
    assert sxx / float(mass.sum()) == pytest.approx((R * (1 + eps)) ** 2 / 4, rel=0.04)
    assert syy / float(mass.sum()) == pytest.approx((R * (1 - eps)) ** 2 / 4, rel=0.04)


def test_shape_observable_matches_linear_theory_for_a_circle_and_an_ellipse():
    """delta = (a^2-b^2)/(a^2+b^2) ~ 2 eps, and exactly 0 for a circle."""
    R, dx, rho0 = 0.02, 0.002, 1000.0
    pos, mass = osc.elliptical_droplet(R, R, dx, rho0)
    assert abs(osc.shape_observables(pos, mass)["delta"]) < 0.03
    eps = 0.08
    pos, mass = osc.elliptical_droplet(R * (1 + eps), R * (1 - eps), dx, rho0)
    assert osc.shape_observables(pos, mass)["delta"] == pytest.approx(
        2 * eps, abs=0.02)


# --------------------------------------------------------------------------- #
# analysis layer against closed-form inputs
# --------------------------------------------------------------------------- #
def test_zero_crossing_frequency_recovers_an_exact_cosine():
    """The estimator is grid-limited, so the tolerance is set by the sampling.

    Linear interpolation between samples puts the crossing within O(dt^2), which
    on this 400-sample / 3-period record is ~0.15 %: measured 73.390 against the
    exact 73.5.  The assertion pins that accuracy rather than pretending the
    estimator is exact, which is why the benchmark quotes it as a cross-check and
    not as the primary estimator.
    """
    omega = 73.5
    t = np.linspace(0.0, 3 * 2 * math.pi / omega, 400)
    y = 0.16 * np.cos(omega * t)
    got = osc.zero_crossing_frequency(t, y)
    assert got["omega"] == pytest.approx(omega, rel=3e-3)
    assert got["n_crossings"] == 6
    # spacing of crossings is grid-limited too: 0.74 % at this sampling
    assert got["period_spread_pct"] < 1.5


def test_damped_cosine_fit_recovers_omega_and_the_decay_rate():
    """The fit must return the prescribed frequency *and* damping, not just a
    good residual: an over-damped signal is exactly the case this benchmark
    meets, and a fit that trades omega against lambda silently biases omega."""
    omega_true, lam_true = 73.5, 12.0
    t = np.linspace(0.0, 0.17, 300)
    y = 0.16 * np.exp(-lam_true * t) * np.cos(omega_true * t) + 0.01
    fit = osc.damped_cosine_fit(t, y, 0.75 * omega_true, 1.25 * omega_true)
    assert fit["omega"] == pytest.approx(omega_true, rel=2e-3)
    assert fit["decay_rate_per_s"] == pytest.approx(lam_true, rel=0.05)


def test_fit_rejects_the_second_harmonic():
    """A signal at 2*omega must not be fitted as omega: that is the failure mode
    seen on a coarse, noisy run, and it would be reported as a +100 % error."""
    omega_true = 73.5
    t = np.linspace(0.0, 0.17, 300)
    y = np.cos(2 * omega_true * t)
    fit = osc.damped_cosine_fit(t, y, 0.75 * omega_true, 1.25 * omega_true)
    # within the searched band the closest available answer is the band edge,
    # and the residual must stay poor - the fit does not pretend to match
    assert fit["rms_residual"] > 0.1


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("eps", [0.08])
def test_short_run_records_an_oscillation(eps, tmp_path):
    """A minimal run must produce a bounded, finite, oscillating record.

    Kept to a fraction of a period so it stays a unit test; the frequency claim
    lives in the benchmark, not here.
    """
    run = osc.run_oscillation(dx=0.002, eps=eps, viscosity=3.0e-3,
                              periods=0.25, samples_per_period=60)
    assert run["status"] in ("OK", "TOO_SHORT")
    hist = run["history"]
    assert len(hist["delta"]) > 5
    d = np.asarray(hist["delta"])
    assert np.all(np.isfinite(d))
    assert abs(d).max() < 1.0
    assert float(np.asarray(hist["vmax"]).max()) < 0.5 * 20.0


def test_divergent_run_is_reported_as_diverged_not_ok():
    """The gate must not turn a blow-up into a number.

    Driven at a viscosity that the stability map documents as unusable
    (nu = 1e-4), the integrator has to abort and say so.
    """
    run = osc.run_oscillation(dx=0.002, eps=0.08, viscosity=1.0e-4,
                              periods=1.0, samples_per_period=60)
    assert run["status"] in ("DIVERGED", "TOO_SHORT")
    assert run["measured"] is None
    assert run["dissipation"] is None


def test_resume_state_round_trip(tmp_path):
    """A paused run resumes to the same total simulated time."""
    state = tmp_path / "s.pt"
    first = osc.run_oscillation(dx=0.002, eps=0.08, viscosity=3.0e-3,
                                periods=0.5, samples_per_period=60,
                                state_path=state, max_wall=0.5)
    assert first["status"] == "PAUSED"
    assert state.exists()
    t1 = first["setup"]["t_end_s"]
    second = osc.run_oscillation(dx=0.002, eps=0.08, viscosity=3.0e-3,
                                 periods=0.5, samples_per_period=60,
                                 state_path=state, max_wall=None)
    assert second["status"] in ("OK", "TOO_SHORT")
    assert second["setup"]["resumed"] is True
    assert second["setup"]["t_end_s"] > t1