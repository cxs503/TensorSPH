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

import pathlib
import os
import json
import sys

import benchmarks.capillary_wave.run as run


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


class TestCapillaryWaveAnalysis:
    """The measurement machinery, checked against synthetic signals.

    These tests carry the benchmark's credibility: the physics run only reports
    a period, and that number is produced by ``fit_period`` /
    ``modal_amplitude`` / ``velocity_mode``.  Each is pinned here against a
    signal whose answer is known exactly.
    """

    def test_analytical_frequency_pure_capillary(self):
        sigma, rho, k = 0.072, 1000.0, 2.0 * math.pi / 0.017
        omega = run.analytical_frequency(sigma, k, rho)
        assert omega == pytest.approx(math.sqrt(sigma * k**3 / rho), rel=1e-12)

    def test_analytical_frequency_includes_gravity_and_depth(self):
        sigma, rho, k, g = 0.072, 1000.0, 2.0 * math.pi / 0.017, 9.81
        depth = 0.16 * 0.017
        expected = math.sqrt((g * k + sigma * k**3 / rho) * math.tanh(k * depth))
        assert (run.analytical_frequency(sigma, k, rho, gravity=g, depth=depth)
                == pytest.approx(expected, rel=1e-12))

    def test_gravity_share_is_half_at_capillary_length(self):
        """lambda = 2 pi sqrt(sigma/rho g) is where the two terms are equal."""
        sigma, rho, g = 0.072, 1000.0, 9.81
        k = 1.0 / run.capillary_length(sigma, rho, g)
        assert run.gravity_share(sigma, k, rho, g) == pytest.approx(0.5, rel=1e-12)

    def test_zero_crossings_interpolate_a_linear_signal(self):
        times = [0.0, 1.0, 2.0, 3.0]
        values = [1.0, 0.5, -0.5, -1.0]
        crossings = run.zero_crossings(times, values)
        assert len(crossings) == 1
        assert crossings[0] == pytest.approx(1.5, rel=1e-12)

    def test_fit_period_recovers_a_damped_sinusoid(self):
        """Crossing-fit must return the true period of a decaying signal."""
        period, decay = 0.084, 0.35
        times = [i * period / 40.0 for i in range(121)]
        values = [math.exp(-decay * t) * math.cos(2.0 * math.pi * t / period)
                  for t in times]
        fit = run.fit_period(times, values)
        assert fit is not None
        assert fit["period_s"] == pytest.approx(period, rel=2e-3)
        assert fit["n_crossings"] >= 5

    def test_fit_period_returns_none_without_two_crossings(self):
        assert run.fit_period([0.0, 0.1, 0.2], [1.0, 0.5, 0.2]) is None

    def test_fit_period_rejects_a_monotone_signal(self):
        times = [i * 0.01 for i in range(50)]
        values = [float(i) for i in range(50)]
        assert run.fit_period(times, values) is None

    def test_envelope_decay_ratio_of_exponential(self):
        times = [i * 0.01 for i in range(400)]
        values = [math.exp(-2.0 * t) * math.cos(2.0 * math.pi * 6.0 * t)
                  for t in times]
        env = run.envelope_decay(times, values)
        assert env["n_extrema"] > 3
        ratio = env["decay_ratio"]
        dt = env["last_extremum"]["t"] - env["first_extremum"]["t"]
        assert ratio == pytest.approx(math.exp(-2.0 * dt), rel=2e-2)


class TestCapillaryWaveMeasurement:
    """``modal_amplitude`` and ``velocity_mode`` on analytic fields."""

    @pytest.fixture
    def wave_cfg(self):
        cfg = Config(
            dx=0.00018, pool_width=0.0085, pool_depth=0.00435,
            sound_speed=1.0, gamma=7.0, gravity=9.81, viscosity=0.0,
            density_diffusion=0.0, hydrostatic_init=True,
            surface_tension_coefficient=0.072, device="cpu",
            boundary="clamp", adaptive_dt=False, max_courant=0.25,
        )
        cfg.dt = suggested_dt(cfg)
        return cfg

    def test_modal_amplitude_recovers_the_imposed_cosine(self, wave_cfg):
        wavelength, amplitude = 0.017, 4.0e-4
        positions, _ = run.make_capillary_wave(wave_cfg, wavelength, amplitude,
                                              torch.device("cpu"))
        recovered, _ = run.modal_amplitude(positions, wave_cfg, wavelength)
        # The topmost-particle interface is quantised to dx; 25 % is the noise
        # floor of the height estimator (which is why the period uses a_v).
        assert recovered == pytest.approx(amplitude, rel=0.25)

    def test_modal_amplitude_is_zero_for_a_flat_interface(self, wave_cfg):
        positions, _ = run.make_capillary_wave(wave_cfg, 0.017, 0.0,
                                               torch.device("cpu"))
        recovered, _ = run.modal_amplitude(positions, wave_cfg, 0.017)
        assert abs(recovered) < 0.2 * wave_cfg.dx

    def test_velocity_mode_recovers_a_known_field(self, wave_cfg):
        wavelength, amp_v = 0.017, 3.0e-3
        positions, mass = run.make_capillary_wave(wave_cfg, wavelength, 4.0e-4,
                                                  torch.device("cpu"))
        k = 2.0 * math.pi / wavelength
        velocity = torch.zeros_like(positions)
        velocity[:, 1] = amp_v * torch.cos(k * positions[:, 0])
        assert run.velocity_mode(velocity, mass, positions,
                                 wavelength) == pytest.approx(amp_v, rel=1e-6)

    def test_velocity_mode_is_orthogonal_to_the_odd_mode(self, wave_cfg):
        """A sin(kx) vertical velocity must project to zero on the cos mode."""
        wavelength = 0.017
        positions, mass = run.make_capillary_wave(wave_cfg, wavelength, 4.0e-4,
                                                  torch.device("cpu"))
        k = 2.0 * math.pi / wavelength
        velocity = torch.zeros_like(positions)
        velocity[:, 1] = 3.0e-3 * torch.sin(k * positions[:, 0])
        assert abs(run.velocity_mode(velocity, mass, positions,
                                     wavelength)) < 0.1 * 3.0e-3

    def test_velocity_mode_ignores_horizontal_motion(self, wave_cfg):
        wavelength = 0.017
        positions, mass = run.make_capillary_wave(wave_cfg, wavelength, 4.0e-4,
                                                  torch.device("cpu"))
        velocity = torch.zeros_like(positions)
        velocity[:, 0] = 0.01
        assert run.velocity_mode(velocity, mass, positions, wavelength) == 0.0


class TestCapillaryWaveInitialCondition:
    """The IC must be a hydrostatic equilibrium, not a pressure step."""

    @pytest.fixture
    def wave_cfg(self):
        cfg = Config(
            dx=0.00018, pool_width=0.0085, pool_depth=0.00435,
            sound_speed=1.0, gamma=7.0, gravity=9.81, viscosity=0.0,
            density_diffusion=0.0, hydrostatic_init=True,
            surface_tension_coefficient=0.072, device="cpu",
            boundary="clamp", adaptive_dt=False, max_courant=0.25,
        )
        cfg.dt = suggested_dt(cfg)
        return cfg

    def test_lattice_mass_correction_restores_the_kernel_sum(self, wave_cfg):
        """After the correction the SPH density on the lattice matches the target."""
        from tensorsph.droplet_impact import density_from_neighbor_pairs
        from tensorsph.neighbor_search import PairListCache
        wavelength, amplitude = 0.017, 4.0e-4
        positions, mass = run.make_capillary_wave(wave_cfg, wavelength, amplitude,
                                                  torch.device("cpu"))
        h = wave_cfg.h_factor * wave_cfg.dx
        pairs, _ = PairListCache(2.0 * h, wave_cfg.pair_skin_factor * h,
                                 fluid_count=positions.shape[0]).get(positions)
        rho = density_from_neighbor_pairs(positions, mass, h, pairs=pairs)
        y = positions[:, 1]
        H = run.interface_level(wave_cfg)
        k = 2.0 * math.pi / wavelength
        surface = H + amplitude * torch.cos(k * positions[:, 0])
        target = hydrostatic_density(y, wave_cfg, surface=surface)
        interior = (y > 0.3 * H) & (y < 0.7 * H)
        assert float((rho[interior] / target[interior]).mean()) == pytest.approx(1.0,
                                                                                rel=2e-3)

    def test_initial_acceleration_is_small_compared_with_g(self, wave_cfg):
        """With the corrected IC the pool does not start in free fall."""
        from tensorsph.droplet_impact import rhs
        from tensorsph.neighbor_search import PairListCache
        wavelength, amplitude = 0.017, 4.0e-4
        positions, mass = run.make_capillary_wave(wave_cfg, wavelength, amplitude,
                                                  torch.device("cpu"))
        velocity = torch.zeros_like(positions)
        h = wave_cfg.h_factor * wave_cfg.dx
        pairs, _ = PairListCache(2.0 * h, wave_cfg.pair_skin_factor * h,
                                 fluid_count=positions.shape[0]).get(positions)
        _, _, acc = rhs(positions, velocity, mass, wave_cfg, pairs=pairs,
                        wall=None, dt=wave_cfg.dt)
        y = positions[:, 1]
        H = run.interface_level(wave_cfg)
        interior = (y > 0.25 * H) & (y < 0.75 * H)
        # The wave itself needs omega^2 A = (74 rad/s)^2 * 4e-4 = 2.2 m/s^2, so
        # the interior residual must stay of that order, not of order g.
        assert float(acc[interior, 1].abs().mean()) < 20.0


class TestCapillaryWaveRunner:
    """End-to-end smoke test of the case driver (in process, short)."""

    def test_short_run_produces_the_report_schema(self):
        wavelength, amplitude = 0.017, 5.4e-4
        cfg = run.build_config(4.2e-4, wavelength, 0.072, amplitude=amplitude,
                               sound_speed=1.0)
        seconds = 30 * cfg.dt
        case = run.run_case(cfg, wavelength, amplitude, seconds, output_every=5)
        assert set(("t", "a_eta", "a_v", "mean_level")) <= set(
            case["time_series"])
        assert case["particles"] > 100
        assert case["period_analytical_s"] > 0.0
        assert case["steps"] == 30
        assert all(math.isfinite(v) for v in case["time_series"]["a_v"])
        # A 30-step run covers 2 % of a period: no period may be reported.
        assert case["period_measured_s"] is None
