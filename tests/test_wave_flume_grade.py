"""Tests for :mod:`tensorsph.wave_flume_grade`.

The grading module turns a gauge record into a wave height and an attenuation
coefficient, so it is tested against signals whose answers are known exactly:

* a sinusoid of amplitude ``a`` must give ``H = 2 a`` from the crossing
  estimator, from ``2 sqrt(2) sigma`` and from the spectral peak;
* an exponentially decaying sinusoid ``a0 exp(-mu x/2)`` must return ``mu``;
* the laminar boundary-layer coefficient must reproduce its own closed form and
  the physical limits (it grows like ``sqrt(nu)`` and like ``sigma^{2.5}``);
* a record with a lost surface (``nan`` samples) must not silently report a
  smaller height.

No solver is run here; the graded runs live in ``benchmarks/wave_flume``.
"""
import math

import pytest
import torch

from tensorsph.wave_flume_grade import (
    attenuation_rate,
    airgap_check,
    grade_regular_record,
    laminar_boundary_layer_damping,
    spectral_amplitude,
    wave_height,
    wave_height_target,
)

GRAVITY = 9.81


def series(height=0.04, period=1.0, duration=8.0, dt=0.004, decay=0.0,
           phase=0.0, mean=0.0, samples=1):
    t = torch.arange(0.0, duration, dt, dtype=torch.float64)
    a = 0.5 * height
    eta = mean + a * torch.exp(-decay * t) * torch.cos(2.0 * math.pi * t / period
                                                        + phase)
    return t, eta


def test_wave_height_recovers_the_amplitude_of_a_sinusoid():
    t, eta = series(height=0.05, period=1.25, duration=6.0)
    got = wave_height(t, eta, smooth_samples=1)
    assert got["n_waves"] == pytest.approx(4, abs=1)
    assert got["height"] == pytest.approx(0.05, rel=2e-3)
    assert got["h_rms"] == pytest.approx(0.05, rel=1e-2)
    assert got["height_crossings"] == pytest.approx(0.05, rel=2e-3)


def test_wave_height_is_insensitive_to_a_shifted_mean_and_a_phase():
    t, eta = series(height=0.03, period=0.8, duration=6.0, mean=0.004,
                    phase=0.9)
    got = wave_height(t, eta, smooth_samples=1)
    assert got["height"] == pytest.approx(0.03, rel=5e-3)


def test_smoothing_removes_gauge_jitter_without_eating_the_wave():
    t, eta = series(height=0.04, period=1.0, duration=6.0)
    torch.manual_seed(7)
    noise = 0.02 * torch.randn_like(eta)      # 50 % of the half-height
    got = wave_height(t, eta + noise, smooth_samples=7, period=1.0)
    # The harmonic fit averages the jitter down (sqrt(2/N)); the crossing
    # estimator fragments on spurious crossings and is reported but not graded.
    assert abs(got["height"] - 0.04) < 0.03 * 0.04
    assert got["height_crossings"] < got["h_rms"] * 1.5


def test_attenuation_recovers_a_known_decay_coefficient():
    mu = 0.05                                  # 1/m
    xs = [0.8, 1.0, 1.2, 1.4]
    hs = [0.04 * math.exp(-mu * x) for x in xs]
    got = attenuation_rate(xs, hs)
    assert got["mu_per_m"] == pytest.approx(mu, rel=1e-9)
    assert got["loss_over_span"] == pytest.approx(1.0 - math.exp(-mu * 0.6), rel=1e-9)
    assert [p["mu_per_m"] for p in got["pairs"]] == pytest.approx([mu] * 3, rel=1e-9)


def test_attenuation_reports_growth_as_a_negative_coefficient():
    got = attenuation_rate([0.8, 1.6], [0.04, 0.05])
    assert got["mu_per_m"] < 0.0


def test_attenuation_rejects_a_single_gauge():
    with pytest.raises(ValueError):
        attenuation_rate([0.8], [0.04])


def test_laminar_damping_matches_its_closed_form_and_scalings():
    period, depth, nu = 1.0, 0.25, 1.0e-6
    got = laminar_boundary_layer_damping(period, depth, GRAVITY, nu)
    sigma = 2.0 * math.pi / period
    k = got["k"]
    # omega^2 = g k tanh(kh) must hold for the returned k.
    assert sigma ** 2 == pytest.approx(GRAVITY * k * math.tanh(k * depth), rel=1e-12)
    # sqrt(nu) scaling: four times the viscosity doubles mu.
    doubled = laminar_boundary_layer_damping(period, depth, GRAVITY, 4.0 * nu)
    assert doubled["mu_per_m"] == pytest.approx(2.0 * got["mu_per_m"], rel=1e-12)
    # The magnitude for a 1 s wave in 0.25 m of water is ~1.2e-3 1/m: a few
    # tenths of a percent over a couple of metres, far below the 10 % gate.
    assert 0.5e-3 < got["mu_per_m"] < 2.5e-3
    assert 1.0 - math.exp(-got["mu_per_m"] * 2.0) < 0.01


def test_laminar_damping_decreases_with_water_depth_and_rejects_bad_input():
    # u_b = a sigma / sinh(kh): the deeper the water, the less bed velocity and
    # the less damping, by orders of magnitude between 0.1 m and 1 m.
    shallow = laminar_boundary_layer_damping(1.0, 0.1, GRAVITY)
    deep = laminar_boundary_layer_damping(1.0, 1.0, GRAVITY)
    assert shallow["mu_per_m"] > 20.0 * deep["mu_per_m"]
    assert laminar_boundary_layer_damping(1.0, 0.25, GRAVITY)["mu_per_m"] > \
        laminar_boundary_layer_damping(0.6, 0.25, GRAVITY)["mu_per_m"]
    with pytest.raises(ValueError):
        laminar_boundary_layer_damping(0.0, 0.25, GRAVITY)


def test_airgap_check_counts_lost_samples():
    t = torch.arange(0.0, 4.0, 0.004, dtype=torch.float64)
    eta = torch.zeros_like(t)
    eta[::10] = float("nan")
    got = airgap_check(t, eta, 0.25)
    assert got["dry_fraction"] == pytest.approx(0.1, abs=1e-12)
    assert got["samples"] == t.numel()


def test_wave_height_target_reads_the_spec_mapping():
    assert wave_height_target({"height": 0.04}) == 0.04

    class Spec:
        height = 0.03

    assert wave_height_target(Spec()) == 0.03
    with pytest.raises(ValueError):
        wave_height_target({"height": -1.0})


def _synthetic_record(height=0.04, period=1.0, decay=0.0, depth=0.25,
                      gauge_x=(0.9, 1.1, 1.3, 1.5), duration=10.0):
    """A record shaped like ``benchmarks/wave_flume/run.py`` output."""
    import math
    from tensorsph.wave_flume import wavenumber_from_period

    k = wavenumber_from_period(period, depth, GRAVITY)
    omega = 2.0 * math.pi / period
    dt = 0.004
    steps = int(round(duration / dt))
    times = [i * dt for i in range(steps + 1)]
    probe_eta = []
    for x in gauge_x:
        amp = 0.5 * height * math.exp(-decay * x)
        probe_eta.append([amp * math.cos(k * x - omega * t) for t in times])
    spec = {"height": height, "depth": depth, "period": period,
            "gravity": GRAVITY, "kind": "regular", "k": k, "omega": omega,
            "celerity": omega / k, "wavelength": 2.0 * math.pi / k,
            "length_scale": 2.0 * math.pi / k, "stroke": 0.1}
    config = {"depth_cells": depth / 0.0125, "height_cells": height / 0.0125}
    return {"case": "regular", "spec": spec, "config": config,
            "probe_x": list(gauge_x), "times": times, "probe_eta": probe_eta}


def test_grade_of_a_synthetic_non_decaying_record_passes():
    got = grade_regular_record(_synthetic_record(decay=0.0), start=2.0)
    assert got["passed"] is True
    assert got["worst_error_vs_target"] == pytest.approx(0.0, abs=2e-3)
    assert got["attenuation"]["mu_per_m"] == pytest.approx(0.0, abs=2e-3)
    # the analytic wave satisfies its own dispersion relation, so the two-probe
    # wave number must come back exactly
    assert got["dispersion"]["wavelength_error"] == pytest.approx(0.0, abs=2e-3)
    assert got["reflection"]["ratio"] < 1e-3


def test_grade_of_a_synthetic_decaying_record_fails_the_height_gate():
    got = grade_regular_record(_synthetic_record(decay=0.25), start=2.0)
    assert got["passed"] is False
    errors = [g["error_vs_target"] for g in got["gauges"]]
    assert min(errors) < -0.10
    assert got["attenuation"]["mu_per_m"] == pytest.approx(0.25, rel=0.05)


def test_spectral_amplitude_reports_the_period_and_amplitude():
    t, eta = series(height=0.06, period=2.0, duration=20.0)
    got = spectral_amplitude(t, eta)
    assert got["period"] == pytest.approx(2.0, rel=2e-2)
    assert got["amplitude"] == pytest.approx(0.03, rel=2e-2)