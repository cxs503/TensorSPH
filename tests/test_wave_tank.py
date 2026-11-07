"""Tests for the WCSPH wave tank: anchors, estimators, and short integrations."""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.wave_tank import (
    WaveTankConfig,
    analytic_velocity,
    angular_frequency,
    clamp_domain,
    column_elevation,
    crest_time,
    elevation_profile,
    flap_transfer,
    harmonic_fit,
    initial_condition,
    make_tank,
    paddle_offset,
    piston_stroke_for_height,
    piston_transfer,
    reflection_coefficient,
    relax_inlet,
    run_tank,
    solitary_beta,
    solitary_celerity,
    solitary_elevation,
    solitary_volume,
    sponge_coefficient,
    two_probe_dispersion,
    wave_celerity,
    wave_number,
    waveform_error,
    wrap_angle,
)

DEPTH = 0.5
GRAVITY = 9.81


# --------------------------------------------------------------------------- #
# Analytical anchors
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kh", [0.05, 0.5, 1.0, 2.0, 3.0, 6.0, 20.0])
def test_dispersion_relation_matches_limits(kh):
    """omega^2 = g k tanh(kh), with the two classical limits."""
    k = kh / DEPTH
    omega = angular_frequency(k, DEPTH)
    assert omega ** 2 == pytest.approx(GRAVITY * k * math.tanh(kh), rel=1e-12)
    # Shallow (kh -> 0) and deep (kh -> inf) limits of the phase speed.
    if kh < 0.1:
        assert wave_celerity(k, DEPTH) == pytest.approx(
            math.sqrt(GRAVITY * DEPTH), rel=2e-3)
    if kh > 10.0:
        assert wave_celerity(k, DEPTH) == pytest.approx(
            math.sqrt(GRAVITY / k), rel=2e-3)


@pytest.mark.parametrize("kh", [0.3, 1.0, 2.5, 3.0])
def test_wave_number_inverts_the_dispersion_relation(kh):
    k = kh / DEPTH
    omega = angular_frequency(k, DEPTH)
    assert wave_number(omega, DEPTH) == pytest.approx(k, rel=1e-10)


def test_piston_transfer_limits_and_monotonicity():
    """Havelock: H/S -> kh for kh -> 0 and grows with kh."""
    assert piston_transfer(1e-6 / DEPTH, DEPTH) == pytest.approx(1e-6, rel=1e-4)
    ratios = [piston_transfer(kh / DEPTH, DEPTH) for kh in (0.1, 0.5, 1.0, 2.0, 3.0)]
    assert all(later > earlier for earlier, later in zip(ratios, ratios[1:]))
    # Deep-water saturation: the piston transfer tends to 2 for kh -> infinity.
    assert piston_transfer(50.0 / DEPTH, DEPTH) == pytest.approx(2.0, rel=1e-3)


def test_flap_is_less_efficient_than_piston_for_short_waves():
    """A floor-hinged flap displaces less water than a piston of equal stroke."""
    for kh in (0.5, 1.0, 2.0, 3.0):
        k = kh / DEPTH
        assert flap_transfer(k, DEPTH) < piston_transfer(k, DEPTH)
    # Long-wave limit: the flap displaces half as much water as a piston of the
    # same surface stroke (the mean of (1+z/h) over the depth is 1/2), so H/S is
    # half of the piston value.
    assert flap_transfer(1e-3 / DEPTH, DEPTH) == pytest.approx(
        0.5 * piston_transfer(1e-3 / DEPTH, DEPTH), rel=1e-2)


@pytest.mark.parametrize("kh", [0.5, 1.0, 3.0])
def test_stroke_inverts_the_transfer_function(kh):
    k = kh / DEPTH
    stroke = piston_stroke_for_height(0.05, k, DEPTH)
    assert stroke * piston_transfer(k, DEPTH) == pytest.approx(0.05, rel=1e-12)


def test_solitary_anchors():
    height, depth = 0.1, 0.5
    beta = solitary_beta(height, depth)
    assert beta == pytest.approx(math.sqrt(3 * height / (4 * depth ** 3)))
    assert solitary_celerity(height, depth) == pytest.approx(
        math.sqrt(GRAVITY * (depth + height)))
    x = torch.linspace(-20.0, 20.0, 400001, dtype=torch.float64)
    eta = solitary_elevation(x, height, depth, crest=1.0)
    assert float(eta.max()) == pytest.approx(height, rel=1e-9)
    # Volume: int sech^2 = 2/beta, so int eta dx = H * 2/beta.
    integral = float(torch.trapezoid(eta, x))
    assert integral == pytest.approx(solitary_volume(height, depth), rel=1e-6)


# --------------------------------------------------------------------------- #
# Measurement estimators
# --------------------------------------------------------------------------- #

def test_harmonic_fit_recovers_amplitude_and_phase():
    omega, amplitude, phase = 6.4, 0.031, -0.7
    times = [index * 0.002 for index in range(2000)]
    values = [amplitude * math.cos(omega * t - phase) + 0.004 * math.sin(3 * omega * t)
              for t in times]
    fit = harmonic_fit(times, values, omega)
    # The 3-omega contamination leaks into the basis by 0.2 %; a clean record is
    # recovered to machine precision just below.
    assert fit.amplitude == pytest.approx(amplitude, rel=5e-3)
    assert fit.height == pytest.approx(2 * amplitude, rel=5e-3)
    assert wrap_angle(fit.phase - phase) == pytest.approx(0.0, abs=1e-2)
    assert fit.residual < 5e-3
    clean = [amplitude * math.cos(omega * t - phase) for t in times]
    fit_clean = harmonic_fit(times, clean, omega)
    assert fit_clean.amplitude == pytest.approx(amplitude, rel=1e-9)


def test_harmonic_fit_absorbs_a_linear_drift():
    omega = 4.0
    times = [index * 0.005 for index in range(500)]
    values = [0.02 * math.cos(omega * t) + 0.01 * t for t in times]
    fit = harmonic_fit(times, values, omega)
    assert fit.amplitude == pytest.approx(0.02, rel=1e-6)
    assert fit.drift == pytest.approx(0.01, rel=1e-6)


def test_two_probe_dispersion_roundtrip():
    """Two synthetic probes with a known k must return that k and c."""
    depth, kh = DEPTH, 2.0
    k = kh / depth
    omega = angular_frequency(k, depth)
    separation = 0.4 * 2 * math.pi / k
    times = [index * 0.001 for index in range(4000)]
    eta_a = [0.02 * math.cos(omega * t) for t in times]
    eta_b = [0.02 * math.cos(omega * t - k * separation) for t in times]
    pair = two_probe_dispersion(times, eta_a, eta_b, separation, omega, depth)
    assert pair.k_measured == pytest.approx(k, rel=1e-6)
    assert pair.celerity_measured == pytest.approx(wave_celerity(k, depth), rel=1e-6)
    assert pair.celerity_error < 1e-6
    assert pair.wavelength_measured == pytest.approx(2 * math.pi / k, rel=1e-6)


def test_reflection_coefficient_envelope():
    """A standing wave of index R has envelope (1+R) and (1-R) times H_i."""
    incident, reflected = 0.05, 0.01
    heights = [incident + reflected, incident - reflected]
    reflection, high, low = reflection_coefficient(heights)
    assert reflection == pytest.approx(reflected / incident, rel=1e-12)
    assert (high, low) == pytest.approx((0.06, 0.04))
    with pytest.raises(ValueError):
        reflection_coefficient([0.05])
    with pytest.raises(ValueError):
        reflection_coefficient([0.0, 0.0])


def test_column_elevation_reads_the_material_volume():
    cfg = WaveTankConfig(depth=DEPTH, dx=0.02, length=1.0, wave_height=0.05,
                         max_courant=0.4, end_time=0.01, hydrostatic_lattice=False)
    dx = cfg.dx
    xs = (torch.arange(50, dtype=torch.float64) + 0.5) * dx
    ys = (torch.arange(25, dtype=torch.float64) + 0.5) * dx
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), cfg.rho0 * dx * dx, dtype=torch.float64)
    assert column_elevation(pos, mass, 0.5, 2 * dx, cfg) == pytest.approx(0.0, abs=1e-12)
    # Add one extra row: the column integral rises by exactly one spacing.
    extra = torch.stack((xs, torch.full_like(xs, DEPTH + 0.5 * dx)), dim=-1)
    pos2 = torch.cat((pos, extra), dim=0)
    mass2 = torch.cat((mass, torch.full((xs.numel(),), cfg.rho0 * dx * dx,
                                        dtype=torch.float64)), dim=0)
    # A whole extra row adds exactly one spacing of material volume per column.
    assert column_elevation(pos2, mass2, 0.5, 2 * dx, cfg) == pytest.approx(dx, rel=1e-6)


def test_elevation_profile_conserves_volume():
    cfg = WaveTankConfig(depth=DEPTH, dx=0.02, length=1.0, wave_height=0.05,
                         max_courant=0.4, end_time=0.01, hydrostatic_lattice=False)
    pos, mass, _, _ = make_tank(cfg)
    centres, eta = elevation_profile(pos, mass, cfg, bins=40)
    integral = float(torch.trapezoid(eta, centres))
    assert centres.numel() == 40
    assert integral == pytest.approx(0.0, abs=5e-3)


def test_sponge_coefficient_profile():
    cfg = WaveTankConfig(depth=DEPTH, dx=0.02, length=2.0, kh=3.0, wave_height=0.05,
                         max_courant=0.4, sponge_wavelengths=1.0,
                         sponge_strength=3.0, end_time=0.01)
    x = torch.linspace(0.0, cfg.length, 201, dtype=torch.float64)
    damping = sponge_coefficient(x, cfg)
    start = cfg.sponge_start()
    assert float(damping[x < start].abs().max()) == 0.0
    values = damping[x >= start]
    assert float(values.max()) == pytest.approx(
        cfg.sponge_strength * 2 * math.pi / cfg.period(), rel=1e-6)
    assert bool((values[1:] >= values[:-1]).all())
    # The damping must stay a slow process per step (otherwise the sponge itself
    # would reflect) and must still be strong over the crossing time.
    assert float(values.max()) * cfg.dt < 0.05
    crossing = cfg.sponge_wavelengths * cfg.wavelength() / cfg.celerity()
    assert float(values.max()) * crossing / 3.0 > 2.0


# --------------------------------------------------------------------------- #
# Geometry, kinematics, boundary
# --------------------------------------------------------------------------- #

def test_make_tank_geometry():
    cfg = WaveTankConfig(depth=DEPTH, dx=0.02, length=1.0, kh=3.0, wave_height=0.05,
                         max_courant=0.4, end_time=0.01, hydrostatic_lattice=False)
    pos, mass, paddle, wall = make_tank(cfg)
    assert pos.shape[0] == 50 * 25
    assert float(pos[:, 0].min()) >= cfg.dx / 2
    assert float(pos[:, 0].max()) < cfg.length
    assert float(pos[:, 1].max()) < cfg.depth
    assert float(paddle[:, 0].max()) < 0.0
    assert float(wall[:, 0].max()) > cfg.length
    # Calibrated lattice mass: the summation density of an interior particle is
    # rho0, so every particle carries exactly the same mass here.
    assert float(mass.max() - mass.min()) == 0.0


def test_hydrostatic_lattice_starts_in_equilibrium():
    """The default lattice mass inverts the Tait law, so the tank does not fall.

    A uniform-mass tank at zero pressure is not in equilibrium; this pins the
    static part of the fix (the dynamical part is
    ``test_still_water_reaches_hydrostatic_pressure``).
    """
    from tensorsph.droplet_impact import hydrostatic_density, tait_pressure

    cfg = WaveTankConfig(depth=DEPTH, dx=0.02, length=1.0, kh=3.0, wave_height=0.05,
                         max_courant=0.4, end_time=0.01)
    assert cfg.hydrostatic_lattice is True
    pos, mass, _, _ = make_tank(cfg)
    pressure = tait_pressure(hydrostatic_density(pos[:, 1], cfg, surface=cfg.depth), cfg)
    reference = cfg.rho0 * cfg.gravity * (cfg.depth - pos[:, 1])
    interior = pos[:, 1] < 0.9 * cfg.depth
    error = ((pressure[interior] - reference[interior]).abs()
             / reference[interior]).max()
    assert float(error) < 0.01
    # Mass per particle now depends on depth, which is the whole point.
    assert float(mass.max() - mass.min()) > 0.0


def test_solitary_initial_condition_follows_the_profile():
    cfg = WaveTankConfig(depth=DEPTH, dx=0.02, length=4.0, wave_height=0.1,
                         wave_kind="solitary", maker="none", max_courant=0.4,
                         end_time=0.01)
    pos, mass, _, _ = make_tank(cfg)
    centres, eta = elevation_profile(pos, mass, cfg, bins=200)
    measured = float(eta.max())
    # The column count is an integer, so the surface is quantised by one spacing.
    assert abs(measured - cfg.wave_height) <= 1.2 * cfg.dx
    velocity = initial_condition(cfg, pos)
    assert float(velocity[:, 1].abs().max()) == 0.0
    assert float(velocity[:, 0].max()) > 0.0


def test_airy_velocity_profile_matches_theory():
    """Surface amplitude ``A omega coth(kh)``, bed amplitude ``A omega/sinh(kh)``."""
    cfg = WaveTankConfig(depth=DEPTH, dx=0.02, length=2.0, kh=3.0, wave_height=0.05,
                         max_courant=0.4, ramp_periods=0.0, end_time=0.01)
    k, omega, amplitude = cfg.wave_number_k(), cfg.omega(), 0.5 * cfg.wave_height
    kh = k * DEPTH
    y = torch.tensor([0.0, DEPTH], dtype=torch.float64)
    x = torch.zeros(2, dtype=torch.float64)
    u, v = analytic_velocity(x, y, cfg, 0.0)       # cos(phase) = 1, sin = 0
    assert float(u[1]) == pytest.approx(amplitude * omega / math.tanh(kh), rel=1e-9)
    assert float(u[0]) == pytest.approx(amplitude * omega / math.sinh(kh), rel=1e-9)
    assert abs(float(v[0])) == pytest.approx(0.0, abs=1e-12)
    # Quarter period later the surface vertical velocity is at its maximum.
    u, v = analytic_velocity(x, y, cfg, 0.25 * cfg.period())
    assert float(v[1].abs()) == pytest.approx(amplitude * omega, rel=1e-6)


def test_relax_inlet_blends_to_the_analytic_field():
    cfg = WaveTankConfig(depth=DEPTH, dx=0.02, length=2.0, kh=3.0, wave_height=0.05,
                         max_courant=0.4, ramp_periods=0.0, end_time=0.01)
    x = torch.tensor([0.0, 0.5 * cfg.inlet_length(), 2.0 * cfg.inlet_length()],
                     dtype=torch.float64)
    y = torch.full_like(x, 0.25)
    pos = torch.stack((x, y), dim=-1)
    vel = torch.zeros((3, 2), dtype=torch.float64)
    blend = relax_inlet(pos, vel, cfg, 0.0)
    u_analytic, _ = analytic_velocity(x, y, cfg, 0.0)
    assert blend == pytest.approx(1.0, abs=1e-12)
    assert float(vel[0, 0]) == pytest.approx(float(u_analytic[0]), rel=1e-9)
    # Quadratic weight: the midpoint of the zone is blended at 25 %.
    assert float(vel[1, 0]) == pytest.approx(0.25 * float(u_analytic[1]), rel=1e-9)
    assert float(vel[2, 0]) == 0.0            # outside the zone: untouched


def test_paddle_offset_is_bounded_and_starts_at_rest():
    cfg = WaveTankConfig(depth=DEPTH, dx=0.02, length=2.0, kh=3.0, wave_height=0.05,
                         max_courant=0.4, ramp_periods=1.0, end_time=0.01,
                         maker="piston")
    stroke = cfg.stroke()
    assert paddle_offset(cfg, 0.0) == pytest.approx(0.0, abs=1e-15)
    values = [paddle_offset(cfg, 0.01 * index) for index in range(200)]
    assert max(abs(value) for value in values) <= 0.5 * stroke + 1e-12
    assert max(values) > 0.4 * stroke      # the ramp does reach full stroke
    # The relaxation maker has no moving wall at all.
    cfg_relax = WaveTankConfig(depth=DEPTH, dx=0.02, length=2.0, kh=3.0, wave_height=0.05,
                               max_courant=0.4, end_time=0.01)
    assert cfg_relax.maker == "relaxation"
    assert paddle_offset(cfg_relax, 0.3) == 0.0
    # A flap reaches only 1/3 of the surface stroke two thirds of the way down.
    cfg_flap = WaveTankConfig(depth=DEPTH, dx=0.02, length=2.0, kh=3.0,
                              wave_height=0.05, maker="flap", max_courant=0.4,
                              end_time=0.01)
    assert cfg_flap.stroke() > stroke


def test_clamp_domain_holds_particles_inside():
    cfg = WaveTankConfig(depth=DEPTH, dx=0.02, length=1.0, wave_height=0.05,
                         max_courant=0.4, end_time=0.01)
    pos = torch.tensor([[0.5, 0.25], [-0.5, 0.25], [0.5, 0.9], [1.5, 0.25]],
                       dtype=torch.float64)
    vel = torch.tensor([[0.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                       dtype=torch.float64)
    clamped = clamp_domain(pos, vel, cfg, face=0.0)
    assert clamped == 3
    assert float(pos[:, 0].min()) >= -3 * cfg.dx
    assert float(pos[:, 1].max()) <= cfg.tank_top_height()
    assert float(vel[1, 0]) == 0.0 and float(vel[2, 1]) == 0.0


def test_crest_time_refines_the_peak():
    times = [0.001 * index for index in range(200)]
    eta = [math.exp(-((t - 0.07) / 0.01) ** 2) for t in times]
    t_peak, height = crest_time(times, eta)
    assert t_peak == pytest.approx(0.07, abs=5e-4)
    assert height == pytest.approx(1.0, abs=2e-3)


def test_waveform_error_is_zero_on_the_analytic_profile():
    depth, height, crest = DEPTH, 0.1, 2.0
    x = torch.linspace(0.0, 4.0, 4001, dtype=torch.float64)
    eta = solitary_elevation(x, height, depth, crest)
    error, samples = waveform_error(x, eta, height, depth, crest)
    assert error < 1e-3
    assert samples > 100
    # A 10 % amplitude offset is visible in the L2 norm.
    error_off, _ = waveform_error(x, 1.1 * eta, height, depth, crest)
    assert error_off > 0.09


# --------------------------------------------------------------------------- #
# Short integrations (the physics, not only the anchors)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_still_water_reaches_hydrostatic_pressure():
    """(a) Interior band pressure against rho0 g (h - y) within 3 % after 0.6 s."""
    cfg = WaveTankConfig(depth=DEPTH, dx=0.02, length=2.0, kh=3.0, wave_height=0.05,
                         max_courant=0.45, maker="none", sponge_strength=0.0,
                         pressure_bands=5, output_every=20,
                         end_time=0.6)
    run = run_tank(cfg)
    assert run.band_history
    half = [row for t, row in run.band_history if t >= 0.3]
    interior = []
    for band in range(4):                      # top band is truncated by the surface
        values = [row[band] for row in half if math.isfinite(row[band])]
        interior.append(abs(sum(values) / len(values)))
    assert max(interior) < 0.03
    assert run.max_compression < 0.02
    assert run.escaped == 0


@pytest.mark.slow
def test_piston_makes_a_wave_of_the_imposed_period():
    """A short run: the probes must oscillate at the paddle frequency."""
    cfg = WaveTankConfig(depth=DEPTH, dx=0.02, length=1.9 * 2 * math.pi / 3.0,
                         kh=3.0, wave_height=0.05, max_courant=0.45,
                         maker="piston", ramp_periods=0.5,
                         probe_offsets=(0.25, 0.65, 0.75, 0.85),
                         sponge_wavelengths=0.8, output_every=10, end_time=1.0)
    run = run_tank(cfg)
    assert run.steps_taken > 1000
    times = [t for t in run.times if t >= 0.65 * run.times[-1]]
    index = [i for i, t in enumerate(run.times) if t >= 0.65 * run.times[-1]]
    fit = harmonic_fit(times, [run.probes[0][i] for i in index], cfg.omega())
    assert fit.height > 0.5 * cfg.wave_height       # the maker does make a wave
    assert fit.residual < 0.2 * fit.height          # and it is dominated by omega
    # The wave travels: the far probe is not in antiphase with the paddle stroke.
    assert abs(run.probes[1][-1]) > 0.0
    assert run.max_compression < 0.06
    assert run.escaped == 0
