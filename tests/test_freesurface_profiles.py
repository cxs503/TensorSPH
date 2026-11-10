"""Tests for the free-surface measurement operators.

Everything in ``tensorsph.freesurface.profiles`` is what turns particles into the
field quantities the analytic solutions are stated in, so a mistake there would
be invisible in the residuals (both sides would be wrong the same way). Each
operator is therefore tested on a synthetic field whose answer is known exactly.
"""
import math

import pytest
import torch

from tensorsph.freesurface import profiles


def uniform_layer(depth: float, length: float = 1.0, dx: float = 0.01,
                  rho0: float = 1000.0):
    """A rectangular lattice of water on the floor, uniform mass ``rho0 dx^2``."""
    xs = torch.arange(0.5 * dx, length, dx, dtype=torch.float64)
    ys = torch.arange(0.5 * dx, depth, dx, dtype=torch.float64)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    positions = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((positions.shape[0],), rho0 * dx * dx, dtype=torch.float64)
    return positions, mass


def test_slab_centres_tile_the_window():
    centres = profiles.slab_centres(0.0, 0.5, 0.1)
    assert centres.numel() == 5
    assert torch.allclose(centres, torch.tensor([0.05, 0.15, 0.25, 0.35, 0.45],
                                                dtype=torch.float64))
    assert profiles.slab_centres(0.0, 0.55, 0.1).numel() == 6


def test_binned_depth_of_a_uniform_layer_is_the_layer_height():
    """Binning is mass based, so a uniform lattice must read back its own depth."""
    dx, depth = 0.01, 0.12
    positions, mass = uniform_layer(depth, dx=dx)
    centres, h = profiles.binned_depth(positions, mass, 1000.0, 2 * dx,
                                       x_min=0.0, x_max=1.0)
    interior = (centres > 3 * dx) & (centres < 1.0 - 3 * dx)
    assert torch.allclose(h[interior], torch.full_like(h[interior], depth),
                          atol=1e-9)


def test_binned_depth_conserves_volume():
    dx = 0.01
    positions, mass = uniform_layer(0.12, dx=dx)
    slab = 2 * dx
    centres, h = profiles.binned_depth(positions, mass, 1000.0, slab,
                                       x_min=0.0, x_max=1.0)
    assert float((h * slab).sum()) == pytest.approx(1.0 * 0.12, rel=1e-12)


def test_binned_velocity_is_mass_weighted_and_marks_dry_strips():
    dx = 0.01
    positions, mass = uniform_layer(0.10, dx=dx)
    velocity = torch.zeros_like(positions)
    # two columns of water moving at different speeds, everything else still
    velocity[positions[:, 0] < 0.20, 0] = 1.0
    velocity[(positions[:, 0] >= 0.20) & (positions[:, 0] < 0.40), 0] = 3.0
    centres, u = profiles.binned_velocity(positions, velocity, mass, 0.10,
                                          x_min=0.0, x_max=1.2)
    assert u[0] == pytest.approx(1.0)
    assert u[2] == pytest.approx(3.0)
    assert math.isnan(float(u[-1]))          # dry strip has no water


def test_front_from_profile_interpolates_the_threshold_crossing():
    """A linear ramp from 0.10 m to 0 m over 10 cm, threshold 0.02 m."""
    slab = 0.01
    centres = profiles.slab_centres(0.0, 0.12, slab)
    depth = torch.clamp(0.10 - 0.10 * (centres - 0.05) / 0.10, min=0.0)
    front = profiles.front_from_profile(centres, depth, 0.02, slab)
    # depth = 0.02 exactly at x = 0.13, i.e. half way between the strip centres
    # 0.125 (h = 0.025) and 0.135 (h = 0.015): the crossing must land on 0.13.
    centres2 = profiles.slab_centres(0.0, 0.20, slab)
    depth2 = torch.clamp(0.10 - 0.10 * (centres2 - 0.05) / 0.10, min=0.0)
    front2 = profiles.front_from_profile(centres2, depth2, 0.02, slab)
    assert front2 == pytest.approx(0.13, abs=1e-9)
    assert front > 0.0  # the crossing is past the sampled window


def test_front_from_profile_on_a_dry_bed_returns_the_upstream_edge():
    centres = profiles.slab_centres(0.0, 0.10, 0.01)
    depth = torch.zeros_like(centres)
    assert profiles.front_from_profile(centres, depth, 1e-3, 0.01) == pytest.approx(0.0)


def test_relative_l2_norm_is_zero_on_the_reference_and_one_on_a_doubled_error():
    ref = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float64)
    assert profiles.relative_l2_norm(ref, ref) == 0.0
    assert profiles.relative_l2_norm(ref + ref, ref) == pytest.approx(1.0)


def test_relative_l2_rms_is_normalised_by_the_scale_not_the_signal():
    sim = torch.tensor([0.0, 2.0], dtype=torch.float64)
    ref = torch.tensor([1.0, 1.0], dtype=torch.float64)
    # RMS deviation is 1.0, scale is 4.0
    assert profiles.relative_l2(sim, ref, 4.0) == pytest.approx(0.25)


def test_loglog_slope_recovers_a_known_order():
    spacings = [0.04, 0.02, 0.01, 0.005]
    errors = [3.0 * h ** 1.0 for h in spacings]
    assert profiles.loglog_slope(spacings, errors) == pytest.approx(1.0, abs=1e-12)
    errors2 = [2.0 * h ** 2.0 for h in spacings]
    assert profiles.loglog_slope(spacings, errors2) == pytest.approx(2.0, abs=1e-12)


def test_loglog_slope_rejects_a_degenerate_ladder():
    with pytest.raises(ValueError):
        profiles.loglog_slope([0.01], [1.0])
    with pytest.raises(ValueError):
        profiles.loglog_slope([0.01, 0.01], [1.0, 0.5])


def test_oscillation_frequency_recovers_a_known_omega():
    omega = 6.5
    times = [i * 0.005 for i in range(1200)]
    signal = [math.cos(omega * t) for t in times]
    assert profiles.oscillation_frequency(times, signal) == pytest.approx(omega, rel=2e-3)


def test_oscillation_frequency_rejects_a_monotone_record():
    with pytest.raises(ValueError):
        profiles.oscillation_frequency([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1, 2, 3, 4])


def test_peak_decay_rate_recovers_an_exponential_envelope():
    rate, omega = 0.35, 5.0
    times = [i * 0.004 for i in range(2500)]
    signal = [math.exp(-rate * t) * math.cos(omega * t) for t in times]
    measured = profiles.peak_decay_rate(times, signal)
    assert measured == pytest.approx(rate, rel=0.05)


def test_peak_decay_rate_needs_a_decaying_record():
    assert math.isnan(profiles.peak_decay_rate([0.0, 1.0], [1.0, 2.0]))


def test_first_mode_amplitude_recovers_the_released_amplitude():
    slab, length, amplitude = 0.01, 0.6, 0.02
    centres = profiles.slab_centres(0.0, length, slab)
    eta = amplitude * torch.cos(math.pi * centres / length)
    assert profiles.first_mode_amplitude(centres, eta, length, slab) == pytest.approx(
        amplitude, rel=1e-9)


def test_first_mode_projection_is_orthogonal_to_the_second_mode():
    slab, length = 0.005, 0.6
    centres = profiles.slab_centres(0.0, length, slab)
    eta = 0.01 * torch.cos(2.0 * math.pi * centres / length)
    assert abs(profiles.first_mode_amplitude(centres, eta, length, slab)) < 1e-6
