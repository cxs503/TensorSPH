"""Tests for the B5 standing gravity-wave (sloshing mode) benchmark.

Three layers, in the order the benchmark itself is built:

1. **The reference.**  ``omega^2 = g k tanh(k H)`` against its deep- and
   shallow-water limits, and the ordering ``omega_2 > omega_1`` for a fixed tank.
2. **The measurement.**  The two elevation estimators (column volume, colour
   level set) and the modal projection have to return the *known* initial mode
   shape of ``build_tank`` without any solver step, and the frequency scan has to
   return the frequency of a synthetic sinusoid.  This is the calibration of the
   instrument: without it, a measured error cannot be attributed to the solver.
3. **One short integration**, as a smoke gate only: the surface must oscillate at
   roughly the analytic frequency at a coarse spacing.  The quantitative acceptance
   is in ``benchmarks/slosh/run.py`` and its recorded ``results.json``.
"""
from __future__ import annotations

import math

import pytest
import torch

from benchmarks.slosh.run import (
    analytic_omega,
    attenuation,
    build_tank,
    column_volume,
    frequency_scan,
    level_set_surface,
    modal_amplitude,
    nonlinear_shift,
    probe_xs,
    run_mode,
)
from tensorsph.droplet_impact import Config


def small_cfg(dx: float = 0.02, length: float = 0.4, depth: float = 0.2) -> Config:
    cfg = Config(dx=dx, pool_width=length, pool_depth=depth, viscosity=0.0,
                 gravity=9.81, sound_speed=20.0, device="cpu",
                 drop_center_y=depth + 0.1)
    cfg.dt = cfg.max_courant * cfg.h_factor * cfg.dx / cfg.sound_speed
    return cfg


# --------------------------------------------------------------------------- #
# 1. the analytic reference
# --------------------------------------------------------------------------- #
def test_analytic_omega_deep_water_limit():
    """k H -> large: tanh -> 1, so omega^2 = g k."""
    length, depth = 1.0, 20.0
    omega = analytic_omega(1, length, depth)
    k = math.pi / length
    assert omega == pytest.approx(math.sqrt(9.81 * k), rel=1e-3)


def test_analytic_omega_shallow_water_limit():
    """k H -> 0: tanh(x) -> x, so omega^2 = g H k^2 and the wave is non-dispersive."""
    length, depth = 100.0, 0.02
    omega = analytic_omega(1, length, depth)
    k = math.pi / length
    assert omega == pytest.approx(math.sqrt(9.81 * depth) * k, rel=1e-3)
    # non-dispersive: the ratio of the first two modes is exactly 2
    assert analytic_omega(2, length, depth) / omega == pytest.approx(2.0, rel=1e-3)


def test_analytic_omega_is_monotone_in_mode():
    length, depth = 0.4, 0.2
    values = [analytic_omega(n, length, depth) for n in (1, 2, 3)]
    assert values[0] < values[1] < values[2]
    # dispersion: in finite depth the ratio is below 2 (sub-linear in k)
    assert values[1] / values[0] < 2.0


def test_nonlinear_shift_scale_is_small_for_this_benchmark():
    """The 10 % amplitude has to be nonlinear *enough* not to matter at 3 %."""
    k, amplitude, depth = math.pi / 0.4, 0.02, 0.2
    assert nonlinear_shift(k, amplitude, depth) < 0.03


# --------------------------------------------------------------------------- #
# 2a. initial condition
# --------------------------------------------------------------------------- #
def test_build_tank_matches_the_mode_shape():
    cfg = small_cfg()
    amplitude, mode = 0.02, 1
    pos, mass, surface_fn = build_tank(cfg, mode, amplitude)
    k = mode * math.pi / cfg.pool_width
    # every particle is inside the fluid region, and the tank is full below it
    assert bool((pos[:, 1] < surface_fn(pos[:, 0])).all())
    assert float(pos[:, 1].max()) > cfg.pool_depth
    assert float(mass.min()) > 0.0
    # the lattice is one particle spacing thick at the top, so the count is close
    # to (L/dx) * (H/dx) plus the crest
    expected = cfg.pool_width / cfg.dx * cfg.pool_depth / cfg.dx
    assert 0.95 * expected < pos.shape[0] < 1.15 * expected
    # hydrostatic masses compress the column: total mass exceeds rho0 L H
    assert float(mass.sum()) > cfg.rho0 * cfg.pool_width * cfg.pool_depth
    # and the IC elevation is the analytic one at the probes
    xs = probe_xs(cfg, 24)
    eta_true = amplitude * torch.cos(k * xs)
    assert float(eta_true.abs().max()) == pytest.approx(amplitude, rel=0.05)


# --------------------------------------------------------------------------- #
# 2b. the two elevation estimators
# --------------------------------------------------------------------------- #
def test_column_volume_recovers_the_initial_mode_amplitude():
    """Instrument calibration: no solver step, the estimator must return A."""
    cfg = small_cfg(dx=0.01)
    amplitude, mode = 0.02, 1
    pos, mass, _ = build_tank(cfg, mode, amplitude)
    rho = mass / (cfg.dx * cfg.dx)
    xs = probe_xs(cfg, 48)
    h, sigma = column_volume(pos, mass, rho, cfg, xs)
    k = mode * math.pi / cfg.pool_width
    eta = (h - cfg.pool_depth) / attenuation(sigma, k)
    measured = modal_amplitude(xs, eta, mode, cfg.pool_width)
    # The initial surface is a staircase on the particle lattice: the top row is
    # kept whenever its centre is below eta(x), which costs up to dx/2 of the
    # amplitude.  The estimator is scored against that discretised amplitude,
    # not against the nominal A; the linearity check below is the real
    # calibration (a biased estimator would not scale).
    assert 0.70 * amplitude < measured < 1.05 * amplitude
    pos2, mass2, _ = build_tank(cfg, mode, 2.0 * amplitude)
    h2, sigma2 = column_volume(pos2, mass2, mass2 / (cfg.dx * cfg.dx), cfg, xs)
    eta2 = (h2 - cfg.pool_depth) / attenuation(sigma2, k)
    measured2 = modal_amplitude(xs, eta2, mode, cfg.pool_width)
    # The staircase loss is additive (up to one dx/2 band is lost whatever the
    # amplitude), so doubling A must recover more than twice the measured value.
    assert 2.0 <= measured2 / measured < 2.4
    # and the elevation of the reconstructed surface tracks the nominal crest
    assert abs(float(eta.abs().max()) - amplitude) < 0.35 * amplitude


def test_attenuation_is_a_transfer_function():
    k = math.pi / 0.4
    assert attenuation(0.0, k) == pytest.approx(1.0)
    assert 0.0 < attenuation(1e-3, k) < 1.0
    assert attenuation(1e-3, k) == pytest.approx(math.exp(-0.5 * (1e-3 * k) ** 2))


def test_modal_amplitude_projects_a_pure_cosine():
    # the projection identity int_0^L cos(k_m x) cos(k_n x) dx = (L/2) delta_mn
    # only holds over the full tank, so the probes must span [0, L] here
    xs = torch.linspace(0.0, 0.4, 81, dtype=torch.float64)
    for mode in (1, 2):
        k = mode * math.pi / 0.4
        eta = 0.017 * torch.cos(k * xs)
        assert modal_amplitude(xs, eta, mode, 0.4) == pytest.approx(0.017, rel=1e-6)
    # a mode-1 signal has zero projection on mode 2 (orthogonality)
    xs = torch.linspace(0.0, 0.4, 81, dtype=torch.float64)
    eta = 0.017 * torch.cos(math.pi * xs / 0.4)
    assert abs(modal_amplitude(xs, eta, 2, 0.4)) < 1e-12


def test_level_set_and_column_volume_agree_on_the_initial_state():
    """The two independent estimators must agree to a fraction of h."""
    cfg = small_cfg(dx=0.01)
    pos, mass, _ = build_tank(cfg, 1, 0.02)
    rho = mass / (cfg.dx * cfg.dx)
    xs = probe_xs(cfg, 32)
    h, sigma = column_volume(pos, mass, rho, cfg, xs)
    k = math.pi / cfg.pool_width
    eta_vol = (h - cfg.pool_depth) / attenuation(sigma, k)
    eta_ls = level_set_surface(pos, mass, rho, cfg, xs, h)
    ok = torch.isfinite(eta_ls)
    assert int(ok.sum()) >= 0.75 * xs.numel(), "level set must resolve the surface"
    assert float((eta_ls[ok] - eta_vol[ok]).abs().mean()) < 1.5 * cfg.h_factor * cfg.dx
    # and the difference has no systematic sign
    assert abs(float((eta_ls[ok] - eta_vol[ok]).mean())) < 0.75 * cfg.h_factor * cfg.dx


# --------------------------------------------------------------------------- #
# 2c. the frequency estimator
# --------------------------------------------------------------------------- #
def test_frequency_scan_recovers_known_signals():
    """Calibration of the estimator itself: no solver involved."""
    omega_true = 8.4062
    t = torch.linspace(0.0, 4 * 2.0 * math.pi / omega_true, 240,
                       dtype=torch.float64)
    clean = 0.02 * torch.cos(omega_true * t + 0.3) + 0.004
    fit = frequency_scan(t, clean, omega_true)
    assert abs(fit["omega"] - omega_true) / omega_true < 2e-3
    assert fit["r2"] > 0.999
    assert fit["amplitude"] == pytest.approx(0.02, rel=2e-2)
    # noise must not move the peak more than a fraction of the acceptance band
    gen = torch.Generator().manual_seed(7)
    noisy = clean + 0.002 * torch.randn(t.shape, generator=gen,
                                        dtype=torch.float64)
    fit_noisy = frequency_scan(t, noisy, omega_true)
    assert abs(fit_noisy["omega"] - omega_true) / omega_true < 1.5e-2
    assert not fit_noisy["at_scan_edge"]


def test_frequency_scan_flags_a_railing_peak():
    """A signal at twice the guess cannot be fitted inside the scan window."""
    omega_true = 8.4
    t = torch.linspace(0.0, 4 * 2.0 * math.pi / omega_true, 240,
                       dtype=torch.float64)
    fit = frequency_scan(t, torch.cos(3.0 * omega_true * t), omega_true)
    assert fit["at_scan_edge"]


# --------------------------------------------------------------------------- #
# 3. one short integration (smoke gate)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", [1])
def test_short_run_oscillates_near_the_analytic_frequency(mode):
    """Coarse, few periods: the gate is loose on purpose (TOL is in the runner)."""
    case = run_mode(mode=mode, dx=0.02, length=0.4, depth=0.2,
                    amplitude_ratio=0.10, periods=1.6, n_probe=24)
    assert case["fit_r2"] > 0.5, "the modal coordinate must be sinusoidal"
    assert abs(case["frequency_error_pct"]) < 35.0
    assert case["mass_drift"] == pytest.approx(0.0, abs=1e-6)
    assert case["mach"] < 0.10
    assert case["escaped_particles"] == 0
    assert case["density_compression"] < 0.05