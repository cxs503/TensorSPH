"""Capillary wave tests: surface-tension frequency validation.

The analytical capillary wave frequency for a 2-D inviscid wave is:

    omega^2 = sigma * k^3 / rho

These tests verify that the CSF surface tension model in TensorSPH
reproduces this frequency within an acceptable tolerance.

References:
- Lamb, H. (1932). Hydrodynamics, 6th ed., Art. 267.
- Brackbill et al. (1992). JCP 100(2), 335-354.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.droplet_impact import Config, hydrostatic_density, rhs, suggested_dt
from tensorsph.neighbor_search import PairListCache


def analytical_omega(sigma: float, k: float, rho: float) -> float:
    """Analytical capillary wave frequency."""
    return math.sqrt(sigma * k**3 / rho)


def _make_wave(cfg: Config, wavelength: float, amplitude: float,
               device: torch.device):
    """Create sinusoidal interface perturbation."""
    dx = cfg.dx
    xs = torch.arange(dx * 0.5, cfg.pool_width, dx, device=device)
    ys = torch.arange(dx * 0.5, cfg.pool_depth, dx, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    positions = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)

    interface_y = cfg.pool_depth / 2.0 + amplitude * torch.sin(
        2.0 * math.pi * positions[:, 0] / wavelength)
    fluid_mask = positions[:, 1] < interface_y
    positions = positions[fluid_mask]
    mass = hydrostatic_density(positions[:, 1], cfg,
                               surface=cfg.pool_depth) * dx * dx
    return positions, mass


def _run_short(cfg: Config, wavelength: float, amplitude: float, steps: int):
    """Run a few steps and return the acceleration field."""
    device = torch.device(cfg.device if cfg.device != "auto" else "cpu")
    positions, mass = _make_wave(cfg, wavelength, amplitude, device)
    velocity = torch.zeros_like(positions)

    h = cfg.h_factor * cfg.dx
    fluid_count = positions.shape[0]
    cache = PairListCache(2.0 * h, cfg.pair_skin_factor * h, fluid_count=fluid_count)

    pairs, _ = cache.get(positions)
    density, pressure, acc = rhs(positions, velocity, mass, cfg, pairs=pairs,
                                 wall=None, dt=cfg.dt)
    return positions, density, acc


class TestCapillaryWaveFrequency:
    """Tests for capillary wave frequency against analytical solution."""

    @pytest.fixture
    def wave_cfg(self):
        cfg = Config(
            dx=0.005,
            pool_width=0.2,
            pool_depth=0.3,
            sound_speed=20.0,
            gamma=7.0,
            gravity=0.0,
            viscosity=0.0,
            density_diffusion=0.0,
            surface_tension_coefficient=0.072,
            device="cpu",
            boundary="clamp",
            adaptive_dt=False,
            max_courant=0.25,
        )
        cfg.dt = suggested_dt(cfg)
        return cfg

    def test_surface_tension_produces_restoring_acceleration(self, wave_cfg):
        """With sigma > 0 and gravity = 0, the wave should have non-zero acceleration."""
        positions, density, acc = _run_short(wave_cfg, 0.2, 0.005, 10)
        # Acceleration should be non-zero (surface tension provides restoring force)
        assert float(acc.abs().max()) > 0.0, "surface tension should produce acceleration"

    def test_zero_surface_tension_gives_zero_acceleration(self, wave_cfg):
        """With sigma = 0 and gravity = 0, a flat interface should have zero acceleration."""
        wave_cfg.surface_tension_coefficient = 0.0
        wave_cfg.dt = suggested_dt(wave_cfg)
        # Flat interface: no perturbation
        positions, mass = _make_wave(wave_cfg, 0.2, 0.0, torch.device("cpu"))
        velocity = torch.zeros_like(positions)
        h = wave_cfg.h_factor * wave_cfg.dx
        fluid_count = positions.shape[0]
        cache = PairListCache(2.0 * h, wave_cfg.pair_skin_factor * h,
                              fluid_count=fluid_count)
        pairs, _ = cache.get(positions)
        density, pressure, acc = rhs(positions, velocity, mass, wave_cfg, pairs=pairs,
                                     wall=None, dt=wave_cfg.dt)
        # With no gravity and no surface tension, acceleration should be ~zero
        assert float(acc.abs().max()) < 0.1, "no forces should give near-zero acceleration"

    def test_analytical_frequency_computation(self):
        """Verify the analytical frequency formula."""
        sigma = 0.072  # water-air
        rho = 1000.0
        wavelength = 0.2
        k = 2.0 * math.pi / wavelength
        omega = analytical_omega(sigma, k, rho)
        # omega^2 = sigma * k^3 / rho
        expected = math.sqrt(sigma * k**3 / rho)
        assert omega == pytest.approx(expected, rel=1e-10)

    def test_frequency_scales_with_k_power_3_over_2(self):
        """omega ~ k^(3/2), so doubling k should increase omega by 2^(3/2) ≈ 2.828."""
        sigma = 0.072
        rho = 1000.0
        k1 = 2.0 * math.pi / 0.2
        k2 = 2.0 * math.pi / 0.1  # half wavelength -> double k
        omega1 = analytical_omega(sigma, k1, rho)
        omega2 = analytical_omega(sigma, k2, rho)
        ratio = omega2 / omega1
        assert ratio == pytest.approx(2.0 ** 1.5, rel=1e-10)

    def test_frequency_scales_with_sqrt_sigma(self):
        """omega ~ sqrt(sigma)."""
        rho = 1000.0
        k = 2.0 * math.pi / 0.2
        omega1 = analytical_omega(0.072, k, rho)
        omega2 = analytical_omega(0.288, k, rho)  # 4x sigma
        ratio = omega2 / omega1
        assert ratio == pytest.approx(2.0, rel=1e-10)  # sqrt(4) = 2

    def test_capillary_dt_limit_is_finite_when_sigma_positive(self, wave_cfg):
        """Capillary time step limit should be finite when surface tension is active."""
        from tensorsph.surface_tension import capillary_dt_limit
        limit = capillary_dt_limit(wave_cfg)
        assert 0 < limit < float("inf"), "capillary dt limit should be finite and positive"

    def test_capillary_dt_limit_is_infinite_when_sigma_zero(self, wave_cfg):
        """Capillary time step limit should be infinite when surface tension is off."""
        from tensorsph.surface_tension import capillary_dt_limit
        wave_cfg.surface_tension_coefficient = 0.0
        limit = capillary_dt_limit(wave_cfg)
        assert limit == float("inf"), "capillary dt limit should be infinite when sigma=0"

    def test_stable_dt_includes_capillary_when_sigma_positive(self, wave_cfg):
        """stable_dt should return the capillary limit when it's the most restrictive."""
        from tensorsph.droplet_impact import stable_dt
        wave_cfg.surface_tension_coefficient = 0.072
        dt_stable = stable_dt(wave_cfg, max_speed=0.0)
        from tensorsph.surface_tension import capillary_dt_limit
        cap_limit = capillary_dt_limit(wave_cfg)
        # Capillary limit should be one of the limits considered
        assert dt_stable <= cap_limit, "stable_dt should respect capillary limit"


class TestCapillaryWaveAmplitude:
    """Tests for amplitude tracking in capillary wave."""

    @pytest.fixture
    def wave_cfg(self):
        cfg = Config(
            dx=0.005,
            pool_width=0.2,
            pool_depth=0.3,
            sound_speed=20.0,
            gamma=7.0,
            gravity=0.0,
            viscosity=0.0,
            density_diffusion=0.0,
            surface_tension_coefficient=0.072,
            device="cpu",
            boundary="clamp",
            adaptive_dt=False,
            max_courant=0.25,
        )
        cfg.dt = suggested_dt(cfg)
        return cfg

    def test_amplitude_tracking_with_perturbation(self, wave_cfg):
        """A perturbed interface should have non-zero tracked amplitude."""
        from benchmarks.capillary_wave.run import track_amplitude
        positions, _ = _make_wave(wave_cfg, 0.2, 0.005, torch.device("cpu"))
        amp = track_amplitude(positions, wave_cfg, 0.2, wave_cfg.pool_depth / 2.0)
        assert amp > 0.0, "perturbed interface should have non-zero amplitude"

    def test_amplitude_tracking_flat_interface(self, wave_cfg):
        """A flat interface should have near-zero tracked amplitude."""
        from benchmarks.capillary_wave.run import track_amplitude
        positions, _ = _make_wave(wave_cfg, 0.2, 0.0, torch.device("cpu"))
        amp = track_amplitude(positions, wave_cfg, 0.2, wave_cfg.pool_depth / 2.0)
        assert amp < 3.0 * wave_cfg.dx, "flat interface should have near-zero amplitude"


class TestCapillaryWavePeriod:
    """Full time-integration verification (m3420).

    The m2153 report only checked the analytical formula against itself; the
    measured oscillation period was never produced (the amplitude tracker
    returned a non-negative envelope, so ``find_period`` could never find a
    zero crossing).  These tests close that debt: integrate the wave for
    ~2.6 analytical periods and compare the measured period with
    ``T = 2 pi / sqrt(sigma k^3 / rho)`` (Lamb 1932, Art. 267).
    """

    @pytest.fixture(scope="class")
    def measured(self):
        from benchmarks.capillary_wave import run as cap_run

        cfg = Config(
            dx=0.01,
            pool_width=0.2,
            pool_depth=0.3,
            sound_speed=20.0,
            gamma=7.0,
            gravity=0.0,
            viscosity=0.0,
            density_diffusion=0.0,
            surface_tension_coefficient=0.072,
            device="cpu",
            boundary="clamp",
            adaptive_dt=True,
            max_courant=0.25,
        )
        cfg.dt = suggested_dt(cfg)
        k = 2.0 * math.pi / 0.2
        omega = analytical_omega(0.072, k, cfg.rho0)
        T = 2.0 * math.pi / omega
        steps = int(round(2.6 * T / cfg.dt))
        times, amps = cap_run.run_capillary_wave(
            cfg, 0.2, 0.005, steps, max(1, steps // 60))
        return cap_run.find_period(times, amps), T, times, amps

    def test_period_is_measurable(self, measured):
        period, T, times, amps = measured
        assert period is not None, (
            "signed modal amplitude must produce zero crossings; "
            "a non-negative envelope cannot be period-resolved")

    def test_period_within_15pct_of_analytical(self, measured):
        period, T, times, amps = measured
        assert period is not None
        err = abs(period - T) / T
        assert err < 0.15, f"measured T={period:.4f} s vs analytical {T:.4f} s ({err:.1%})"

    def test_signed_amplitude_recovers_initial_mode(self, wave_cfg):
        from benchmarks.capillary_wave.run import track_amplitude
        positions, _ = _make_wave(wave_cfg, 0.2, 0.005, torch.device("cpu"))
        amp = track_amplitude(positions, wave_cfg, 0.2, wave_cfg.pool_depth / 2.0)
        assert 0.0 < amp < 2.0 * 0.005, f"modal amplitude {amp} should recover A0=0.005"
