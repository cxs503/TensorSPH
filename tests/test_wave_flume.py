"""Tests for :mod:`tensorsph.wave_flume`.

Two kinds of test, both cheap:

* **analytic anchors** -- the wave theory the flume is graded against must itself
  be right, so the dispersion solver, the piston transfer function, the
  first-order solitary wave and Goring's paddle kinematics are pinned against
  independent limits and against each other (closed form vs numerical quadrature
  vs finite differences).
* **measurement chain** -- the estimators that turn a gauge record into a number
  (spectral peak, zero crossings, two-probe wave number, incident/reflected
  separation, free-surface extraction, XSPH fixed point) are pinned on synthetic
  signals with known answers.  Without that, a "1 % dispersion error" would
  measure the estimator, not the solver.

Nothing here runs the flume for a physically interesting time; the graded runs
live in ``benchmarks/wave_flume`` and ``docs/verification_wave_flume.md``.
"""
import math

import pytest
import torch

from tensorsph.wave_flume import (
    WaveFlumeConfig,
    airy_elevation,
    default_probe_positions,
    dominant_omega,
    flume_particle_mass,
    flume_wave_spec,
    flume_water_lattice,
    goring_max_stroke,
    goring_paddle_acceleration,
    goring_paddle_velocity,
    goring_stroke,
    linear_dispersion_omega,
    paddle_lattice,
    paddle_motion,
    paddle_stroke_for_height,
    phase_speed,
    piston_stroke_amplitude,
    probe_elevation,
    separate_incident_reflected,
    solitary_alpha,
    solitary_wave_celerity,
    solitary_wave_elevation,
    solitary_wave_length,
    sponge_rate,
    static_wall_lattice,
    surface_profile,
    validate_flume,
    wavenumber_from_omega,
    wavenumber_from_period,
    wavenumber_from_two_probes,
    wavelength,
    xsph_correction,
    zero_crossing_period,
)

GRAVITY = 9.81
DEPTH = 0.25
PERIOD = 1.0
HEIGHT = 0.04


def base_config(**overrides) -> WaveFlumeConfig:
    """A valid case: 12.5 particles through the depth is not allowed, 10 is."""
    values = dict(flume_length=2.0, depth=0.2, wave_height=0.04, wave_period=0.8,
                  dx=0.02, sponge_length=0.46)
    values.update(overrides)
    return WaveFlumeConfig(**values)


# --------------------------------------------------------------------------- #
# Linear wave theory
# --------------------------------------------------------------------------- #

def test_dispersion_reduces_to_the_deep_and_shallow_water_limits():
    k = 5.0
    gravity = 9.81
    deep = linear_dispersion_omega(k, depth=1.0e6)
    shallow = linear_dispersion_omega(k, depth=1.0e-6)
    assert deep == pytest.approx(math.sqrt(gravity * k), rel=1.0e-10)
    # tanh(kh) -> kh for kh -> 0, so omega -> k sqrt(g h); with h = 1e-6 this is
    # a very small frequency, hence the sqrt-based comparison.
    assert shallow == pytest.approx(k * math.sqrt(gravity * 1.0e-6), rel=1.0e-9)


def test_deep_water_wavelength_matches_the_textbook_value():
    # lambda = g T^2 / 2 pi = 1.56 T^2 in deep water (Dean & Dalrymple eq. 3.36).
    period = 1.0
    assert wavelength(period, depth=1.0e4) == pytest.approx(9.81 / (2.0 * math.pi),
                                                            rel=1.0e-6)


def test_wavenumber_inverts_the_dispersion_relation_at_every_depth_regime():
    for depth, period in ((0.25, 1.0), (2.0, 1.0), (0.05, 6.0), (50.0, 0.5)):
        k = wavenumber_from_period(period, depth)
        omega = 2.0 * math.pi / period
        assert linear_dispersion_omega(k, depth) == pytest.approx(omega, rel=1.0e-11)
        assert wavenumber_from_omega(omega, depth) == pytest.approx(k, rel=1.0e-11)


def test_phase_speed_is_between_the_shallow_and_deep_limits():
    # A 6 s wave in 0.25 m of water is shallow (kh = 0.17): its celerity is just
    # above sqrt(g h) and far below the deep-water value g T / 2 pi.
    period, depth = 6.0, 0.25
    c = phase_speed(period, depth)
    # kh = 0.17, so the celerity approaches sqrt(g h) from *below*.
    assert c < math.sqrt(GRAVITY * depth)
    assert c == pytest.approx(math.sqrt(GRAVITY * depth), rel=1.0e-2)
    # and a 0.2 s wave in 3 m of water is deep: c -> g T / 2 pi
    assert phase_speed(0.2, 3.0) == pytest.approx(GRAVITY * 0.2 / (2.0 * math.pi),
                                                 rel=1.0e-2)


def test_airy_elevation_is_a_travelling_wave_of_the_requested_height():
    k = wavenumber_from_period(PERIOD, DEPTH)
    omega = linear_dispersion_omega(k, DEPTH)
    crest = max(airy_elevation(x, 0.0, HEIGHT, k, omega)
                for x in [i * 0.001 for i in range(3000)])
    assert crest == pytest.approx(HEIGHT / 2.0, rel=1.0e-3)
    # eta(x, t) = eta(x - c t, 0) for a right-going wave
    celerity = omega / k
    shifted = airy_elevation(0.4, 0.0, HEIGHT, k, omega)
    assert airy_elevation(0.4 + celerity * 0.25, 0.25, HEIGHT, k, omega) == \
        pytest.approx(shifted, rel=1.0e-9)


# --------------------------------------------------------------------------- #
# Piston wavemaker transfer function
# --------------------------------------------------------------------------- #

def test_piston_transfer_tends_to_two_in_deep_water():
    # H/S -> 2 as kh -> infinity (a deep-water piston needs half the wave height
    # in stroke) and H/S -> kh as kh -> 0.
    stroke = piston_stroke_amplitude(HEIGHT, k=40.0, depth=1.0)
    assert stroke == pytest.approx(HEIGHT / 2.0, rel=2.0e-2)


def test_piston_transfer_tends_to_kh_for_long_waves():
    k, depth = 0.02, 1.0          # kh = 0.02
    assert piston_stroke_amplitude(HEIGHT, k, depth) == pytest.approx(
        HEIGHT / (k * depth), rel=2.0e-2)


def test_piston_transfer_series_branch_joins_the_closed_form():
    # kh = 1e-3 is where the series takes over: the series and the (well
    # conditioned) closed form must agree to far below the 1e-3 switch-over.
    from tensorsph.wave_flume import _piston_transfer
    depth = 0.25
    for kh in (9.9e-4, 1.01e-3):

        k = kh / depth
        closed = 2.0 * (math.cosh(2.0 * kh) - 1.0) / (math.sinh(2.0 * kh) + 2.0 * kh)
        assert _piston_transfer(k, depth) == pytest.approx(closed, rel=1.0e-6)


def test_stroke_and_height_are_exact_inverses():
    k = wavenumber_from_period(PERIOD, DEPTH)
    stroke = piston_stroke_amplitude(HEIGHT, k, DEPTH)
    assert paddle_stroke_for_height(stroke, k, DEPTH) == pytest.approx(HEIGHT,
                                                                     rel=1.0e-12)


# --------------------------------------------------------------------------- #
# First-order solitary wave
# --------------------------------------------------------------------------- #

def test_solitary_profile_has_the_requested_height_and_half_width():
    alpha = solitary_alpha(HEIGHT, DEPTH)
    assert solitary_wave_elevation(0.0, 0.0, HEIGHT, DEPTH) == pytest.approx(HEIGHT)
    # sech^2 = 1/2 at x - c t = +/- arccosh(sqrt 2) / alpha
    half_width = math.acosh(math.sqrt(2.0)) / alpha
    assert solitary_wave_elevation(half_width, 0.0, HEIGHT, DEPTH) == \
        pytest.approx(HEIGHT / 2.0, rel=1.0e-12)
    assert solitary_wave_length(HEIGHT, DEPTH) == pytest.approx(1.0 / alpha)


def test_solitary_profile_travels_at_the_first_order_celerity():
    celerity = solitary_wave_celerity(HEIGHT, DEPTH)
    assert celerity == pytest.approx(math.sqrt(GRAVITY * (DEPTH + HEIGHT)))
    eta = solitary_wave_elevation(0.6, 0.25, HEIGHT, DEPTH)
    # eta(x, t) = f(x - c t): the profile at (0.6, 0.25) is the profile at
    # (0.6 - c 0.25, 0).
    assert solitary_wave_elevation(0.6 - celerity * 0.25, 0.0, HEIGHT, DEPTH) == \
        pytest.approx(eta, rel=1.0e-12)


def test_solitary_volume_matches_the_analytic_integral():
    # int H sech^2(alpha x) dx = 2 H / alpha
    alpha = solitary_alpha(HEIGHT, DEPTH)
    step = 0.0005
    xs = torch.arange(-20.0 / alpha, 20.0 / alpha, step, dtype=torch.float64)
    eta = torch.stack([torch.tensor(solitary_wave_elevation(float(x), 0.0, HEIGHT,
                                                           DEPTH),
                                   dtype=torch.float64) for x in xs])
    volume = float(eta.sum()) * step
    assert volume == pytest.approx(2.0 * HEIGHT / alpha, rel=1.0e-6)


# --------------------------------------------------------------------------- #
# Goring's paddle kinematics
# --------------------------------------------------------------------------- #

def test_goring_stroke_time_derivative_is_the_paddle_velocity():
    alpha = solitary_alpha(HEIGHT, DEPTH)
    celerity = solitary_wave_celerity(HEIGHT, DEPTH)
    for tau in (-1.5, -0.6, 0.0, 0.4, 1.2):
        step = 1.0e-6 / (alpha * celerity)
        derivative = (goring_stroke(tau + step, HEIGHT, DEPTH)
                      - goring_stroke(tau - step, HEIGHT, DEPTH)) / (2.0 * step)
        assert derivative == pytest.approx(
            goring_paddle_velocity(tau, HEIGHT, DEPTH), rel=1.0e-7)


def test_goring_velocity_derivative_is_the_paddle_acceleration():
    for tau in (-1.0, -0.2, 0.0, 0.7):
        step = 1.0e-7
        derivative = (goring_paddle_velocity(tau + step, HEIGHT, DEPTH)
                      - goring_paddle_velocity(tau - step, HEIGHT, DEPTH)) / (2.0 * step)
        assert derivative == pytest.approx(
            goring_paddle_acceleration(tau, HEIGHT, DEPTH), rel=1.0e-6, abs=1.0e-9)


def test_goring_max_stroke_equals_the_integral_of_the_paddle_velocity():
    # Independent quadrature of int U dt over the whole generator pulse.
    alpha = solitary_alpha(HEIGHT, DEPTH)
    celerity = solitary_wave_celerity(HEIGHT, DEPTH)
    half = 25.0 / (alpha * celerity)
    step = 2.0 * half / 400000
    total = 0.0
    for index in range(400000):
        tau = -half + (index + 0.5) * step
        total += goring_paddle_velocity(tau, HEIGHT, DEPTH) * step
    assert total == pytest.approx(goring_max_stroke(HEIGHT, DEPTH), rel=1.0e-3)


def test_goring_stroke_is_zero_when_the_crest_leaves_the_paddle():
    assert goring_stroke(0.0, HEIGHT, DEPTH) == pytest.approx(0.0, abs=1.0e-15)
    assert goring_stroke(1.0, HEIGHT, DEPTH) == pytest.approx(
        -goring_stroke(-1.0, HEIGHT, DEPTH), rel=1.0e-12)


# --------------------------------------------------------------------------- #
# Paddle motion and sponge
# --------------------------------------------------------------------------- #

def test_regular_paddle_motion_is_the_ramped_sinusoid_and_its_derivatives():
    cfg = base_config()
    spec = flume_wave_spec(cfg)
    ramp_end = cfg.ramp_periods * spec.period
    time = ramp_end + 0.3
    offset, speed, accel = paddle_motion(spec, cfg, time)
    assert offset == pytest.approx(0.5 * spec.stroke * math.sin(spec.omega * time),
                                   rel=1.0e-12)
    assert speed == pytest.approx(0.5 * spec.stroke * spec.omega
                                  * math.cos(spec.omega * time), rel=1.0e-12)
    assert accel == pytest.approx(-0.5 * spec.stroke * spec.omega ** 2
                                  * math.sin(spec.omega * time), rel=1.0e-12)
    # and the three agree with each other by finite differences
    step = 1.0e-7
    a = paddle_motion(spec, cfg, time + step)[0]
    b = paddle_motion(spec, cfg, time - step)[0]
    assert (a - b) / (2.0 * step) == pytest.approx(speed, rel=1.0e-5)
    a = paddle_motion(spec, cfg, time + step)[1]
    b = paddle_motion(spec, cfg, time - step)[1]
    assert (a - b) / (2.0 * step) == pytest.approx(accel, rel=1.0e-5)


def test_the_ramp_starts_the_paddle_from_rest():
    cfg = base_config()
    spec = flume_wave_spec(cfg)
    offset, speed, accel = paddle_motion(spec, cfg, 0.0)
    assert (offset, speed, accel) == (0.0, 0.0, 0.0)


def test_solitary_paddle_starts_from_rest_and_ends_at_the_full_stroke():
    cfg = base_config(wave_type="solitary", wave_height=0.04, sponge_length=0.9)
    spec = flume_wave_spec(cfg)
    _, start_speed, _ = paddle_motion(spec, cfg, 0.0)
    peak = max(goring_paddle_velocity(0.01 * i / 100.0 * 30.0, spec.height, spec.depth)
               for i in range(-300, 300))
    assert abs(start_speed) < 0.05 * peak
    end_offset, _, _ = paddle_motion(spec, cfg, 20.0)
    # Within the 3-characteristic-time lead the start is not exactly at -S/2.
    assert end_offset == pytest.approx(goring_max_stroke(spec.height, spec.depth),
                                       rel=1.0e-2)


def test_sponge_rate_is_zero_upstream_monotone_and_peaks_at_the_wall():
    cfg = base_config()
    spec = flume_wave_spec(cfg)
    start = cfg.sponge_start()
    assert sponge_rate(0.0, cfg, spec) == 0.0
    assert sponge_rate(start - 1.0e-9, cfg, spec) == 0.0
    values = [sponge_rate(start + f * cfg.sponge_length, cfg, spec)
              for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert values[-1] == pytest.approx(cfg.sponge_strength * spec.celerity
                                       / cfg.sponge_length)
    # one wavelength per sponge length is the time a wave needs to cross it
    assert values[-1] * 0.1 / spec.celerity < 1.0


def test_sponge_rate_accepts_a_tensor_and_matches_the_scalar_path():
    cfg = base_config()
    spec = flume_wave_spec(cfg)
    xs = torch.linspace(0.0, cfg.flume_length, 17, dtype=torch.float64)
    values = sponge_rate(xs, cfg, spec)
    for x, value in zip(xs.tolist(), values.tolist()):
        assert value == pytest.approx(sponge_rate(x, cfg, spec), rel=1.0e-12)


# --------------------------------------------------------------------------- #
# Lattices and case validation
# --------------------------------------------------------------------------- #

def test_water_lattice_fills_exactly_the_requested_depth():
    cfg = base_config()
    device = torch.device("cpu")
    positions = flume_water_lattice(cfg, device)
    rows = int(round(cfg.depth / cfg.dx))
    assert positions[:, 1].min().item() == pytest.approx(0.5 * cfg.dx)
    assert positions[:, 1].max().item() == pytest.approx(cfg.depth - 0.5 * cfg.dx)
    assert positions[:, 0].max().item() < cfg.flume_length
    assert positions.shape[0] == int(round(cfg.flume_length / cfg.dx)) * rows


def test_walls_and_paddle_keep_one_spacing_clearance_from_the_water():
    cfg = base_config()
    device = torch.device("cpu")
    water = flume_water_lattice(cfg, device)
    fixed = static_wall_lattice(cfg, device)
    paddle = paddle_lattice(cfg, device)
    # first water column at dx/2, paddle face particles at -dx/2: one spacing.
    assert water[:, 0].min().item() - paddle[:, 0].max().item() == \
        pytest.approx(cfg.dx, rel=1.0e-6)
    # The last water column is within one spacing of the downstream wall layer
    # (the floor layer extends past the wall, so select the wall column first).
    wall_column = fixed[fixed[:, 0] > cfg.flume_length]
    clearance = wall_column[:, 0].min().item() - water[:, 0].max().item()
    assert 0.0 < clearance <= 1.5 * cfg.dx


def test_masses_follow_the_hydrostatic_profile_and_the_lattice_calibration():
    cfg = base_config()
    device = torch.device("cpu")
    positions = flume_water_lattice(cfg, device)
    mass = flume_particle_mass(cfg, positions, device)
    bottom = mass[positions[:, 1] <= 0.5 * cfg.dx].mean()
    top = mass[positions[:, 1] >= cfg.depth - 0.5 * cfg.dx].mean()
    # a resting column is densest at the bottom; the difference is rho0 g h / c0^2
    assert bottom > top
    # rho_bottom / rho_top - 1 = g (depth - dx) / c0^2 for the Tait profile.
    assert (bottom / top - 1.0) == pytest.approx(
        cfg.gravity * (cfg.depth - cfg.dx) / cfg.sound_speed ** 2, rel=5.0e-2)


@pytest.mark.parametrize("overrides", [
    {"dx": 0.021},                                   # depth is not a whole cell count
    {"flume_length": 1.99},                          # nor is the flume length
    {"sponge_length": 0.2},                          # sponge shorter than half a wave
    {"wave_height": 0.02, "dx": 0.02},               # crest not resolved
    {"depth": 0.2, "dx": 0.05, "wave_height": 0.15},  # too few rows through depth
    {"boundary": "open"},                            # only kinematic walls exist
])
def test_validate_flume_rejects_configurations_that_would_be_silently_wrong(overrides):
    cfg = base_config(**overrides)
    with pytest.raises(ValueError):
        validate_flume(cfg)


def test_default_probes_sit_in_the_propagation_zone_and_span_under_half_a_wave():
    cfg = base_config()
    spec = flume_wave_spec(cfg)
    probes = default_probe_positions(cfg, spec)
    assert all(0.0 < p < cfg.sponge_start() for p in probes)
    assert probes[1] - probes[0] < 0.5 * spec.wavelength


# --------------------------------------------------------------------------- #
# Measurement chain on synthetic signals
# --------------------------------------------------------------------------- #

def synthetic_regular(cfg, omega, k, height, seconds=4.0, rate=400.0, phase=0.0):
    spec = flume_wave_spec(cfg)
    count = int(seconds * rate)
    times = [i / rate for i in range(count)]
    series = []
    for x in default_probe_positions(cfg, spec):
        series.append([airy_elevation(x, t, height, k, omega, phase) for t in times])
    return times, series


def test_dominant_omega_recovers_a_known_frequency():
    k = wavenumber_from_period(PERIOD, DEPTH)
    omega = linear_dispersion_omega(k, DEPTH)
    times, series = synthetic_regular(base_config(), omega, k, 0.04)
    measured = dominant_omega(times, series[1])
    assert measured == pytest.approx(omega, rel=1.0e-3)


def test_zero_crossing_period_recovers_a_known_period():
    k = wavenumber_from_period(PERIOD, DEPTH)
    omega = linear_dispersion_omega(k, DEPTH)
    times, series = synthetic_regular(base_config(), omega, k, 0.04)
    period, crossings = zero_crossing_period(times, series[1])
    assert period == pytest.approx(2.0 * math.pi / omega, rel=1.0e-3)
    assert crossings >= 3


def test_two_probe_phase_recovers_a_known_wave_number():
    k = wavenumber_from_period(PERIOD, DEPTH)
    omega = linear_dispersion_omega(k, DEPTH)
    cfg = base_config()
    spec = flume_wave_spec(cfg)
    probes = default_probe_positions(cfg, spec)
    times, series = synthetic_regular(cfg, omega, k, 0.04)
    measured = wavenumber_from_two_probes(times, series[0], series[1],
                                         probes[1] - probes[0])
    assert measured == pytest.approx(k, rel=1.0e-3)


def test_incident_reflected_separation_recovers_a_known_reflection_coefficient():
    k = wavenumber_from_period(PERIOD, DEPTH)
    omega = linear_dispersion_omega(k, DEPTH)
    cfg = base_config()
    spec = flume_wave_spec(cfg)
    probes = default_probe_positions(cfg, spec)
    incident, reflected_ratio = 0.02, 0.07
    rate, seconds = 400.0, 4.0
    times = [i / rate for i in range(int(seconds * rate))]
    series = []
    for x in probes:
        series.append([incident * math.cos(k * x - omega * t)
                       + incident * reflected_ratio * math.cos(k * x + omega * t)
                       for t in times])
    result = separate_incident_reflected(times, series, probes, omega, k)
    assert result["incident_amplitude"] == pytest.approx(incident, rel=5.0e-3)
    assert result["ratio"] == pytest.approx(reflected_ratio, rel=5.0e-2)
    assert result["relative_residual"] < 1.0e-6


def test_separation_needs_enough_gauges():
    k, omega = 4.0, 6.0
    times = [0.001 * i for i in range(500)]
    series = [[0.01 * math.cos(k * x - omega * t) for t in times] for x in (0.1, 0.2)]
    with pytest.raises(ValueError):
        separate_incident_reflected(times, series, (0.1, 0.2), omega, k)


def test_probe_elevation_and_profile_are_unbiased_on_a_flat_lattice():
    cfg = base_config()
    device = torch.device("cpu")
    positions = flume_water_lattice(cfg, device)
    offset = cfg.surface_offset()
    eta = probe_elevation(positions, 0.5, cfg.depth, cfg.dx, surface_offset=offset)
    assert eta == pytest.approx(0.0, abs=1.0e-6)
    x, profile = surface_profile(positions, cfg.depth, cfg.dx, surface_offset=offset)
    assert float(profile.abs().max()) == pytest.approx(0.0, abs=1.0e-6)
    assert x.numel() > 10


def test_probe_elevation_reports_a_lifted_patch_and_nan_outside_the_water():
    cfg = base_config()
    device = torch.device("cpu")
    positions = flume_water_lattice(cfg, device)
    lifted = positions.clone()
    near = (lifted[:, 0] > 0.48) & (lifted[:, 0] < 0.52)
    lifted[near, 1] += 0.003
    eta = probe_elevation(lifted, 0.50, cfg.depth, cfg.dx,
                          half_width=1.5 * cfg.dx, surface_offset=cfg.surface_offset())
    assert eta == pytest.approx(0.003, abs=1.0e-6)
    assert math.isnan(probe_elevation(positions, cfg.flume_length + 1.0, cfg.depth,
                                      cfg.dx))


# --------------------------------------------------------------------------- #
# XSPH filter
# --------------------------------------------------------------------------- #

def test_xsph_leaves_a_uniform_velocity_field_untouched():
    cfg = base_config()
    device = torch.device("cpu")
    positions = flume_water_lattice(cfg, device)
    n = positions.shape[0]
    velocity = torch.zeros_like(positions)
    velocity[:, 0] = 0.25
    density = torch.full((n,), cfg.rho0)
    mass = flume_particle_mass(cfg, positions, device)
    from tensorsph.neighbor_search import cell_linked_pairs
    pairs = cell_linked_pairs(positions, 2.0 * cfg.h_factor * cfg.dx)
    smoothed = xsph_correction(positions, velocity, mass, density, pairs,
                               cfg.h_factor * cfg.dx, 0.5)
    assert torch.allclose(smoothed, velocity, atol=1.0e-6)


def test_xsph_reduces_a_particle_scale_perturbation_and_conserves_momentum():
    cfg = base_config()
    device = torch.device("cpu")
    positions = flume_water_lattice(cfg, device)
    n = positions.shape[0]
    density = torch.full((n,), cfg.rho0)
    mass = flume_particle_mass(cfg, positions, device)
    from tensorsph.neighbor_search import cell_linked_pairs
    pairs = cell_linked_pairs(positions, 2.0 * cfg.h_factor * cfg.dx)
    velocity = torch.zeros_like(positions)
    # A deterministic particle-scale perturbation (period 7 rows in index space).
    index = torch.arange(n, dtype=positions.dtype)
    velocity[:, 0] = 0.01 * torch.sin(37.0 * index)
    smoothed = xsph_correction(positions, velocity, mass, density, pairs,
                               cfg.h_factor * cfg.dx, 0.5)
    rms_before = float((velocity[:, 0] ** 2).mean().sqrt())
    rms_after = float((smoothed[:, 0] ** 2).mean().sqrt())
    assert rms_after < 0.9 * rms_before
    # Momentum: the pair contributions are antisymmetric, but each particle is
    # then divided by its own kernel weight, so conservation is approximate --
    # to a few per cent, not to machine precision.
    before = float((mass * velocity[:, 0]).sum())
    after = float((mass * smoothed[:, 0]).sum())
    scale = float((mass * velocity[:, 0].abs()).sum())
    assert abs(after - before) < 0.05 * scale


@pytest.mark.slow
def test_a_short_flume_run_stays_bounded_and_keeps_its_mass(target_dt=2.0e-4):
    """A few hundred steps of a small flume: no blow-up, no mass loss, no escape."""
    from tensorsph.wave_flume import run_case
    cfg = base_config(flume_length=1.4, depth=0.2, wave_height=0.04, dx=0.02,
                      dt=target_dt, steps=300, output_every=50,
                      sponge_length=0.46, ramp_periods=1.0, settle_time=0.02,
                      xsph_settle=0.3, density_diffusion=0.0)
    run = run_case(cfg, probe_positions=(0.2, 0.3, 0.4))
    assert run.steps_taken > 0
    assert run.escaped == 0
    assert run.max_density_compression < 0.05
    assert run.mass_drift == 0.0
    assert all(math.isfinite(value) for value in run.speed_history)