"""Tests for the Ritter reference solution, the depth measure, and one smoke run.

The analytic content is verified *before* the solution is used as a benchmark
reference: three exact identities (dam-section depth, released volume, Riemann
invariant) plus the characteristic speeds. A wrong reference is worse than no
reference, so these are pinned numerically, not by inspection.
"""
import math

import pytest
import torch

from tensorsph.dam_break import DamBreakConfig, run_case
from tensorsph.ritter import (
    depth,
    front_x,
    particle_depth,
    profile_l1_error,
    profile_mass,
    released_volume,
    tail_x,
    validity_window,
    velocity,
    wave_speed,
)


def test_dam_section_depth_is_four_ninths():
    """``h(x_dam, t) = 4 h0 / 9`` -- the classical Ritter value, any ``h0``."""
    for h0 in (0.05, 0.1, 0.4, 1.0):
        for g in (9.81, 1.0, 30.0):
            h = depth(torch.tensor([0.0], dtype=torch.float64), 0.3, h0,
                      dam_x=0.0, gravity=g)
            assert float(h) == pytest.approx(4.0 * h0 / 9.0, rel=1.0e-12)


def test_depth_is_undisturbed_behind_the_tail_and_dry_ahead_of_the_front():
    h0, t, g = 0.1, 0.7, 9.81
    wave_speed(h0, g)
    x = torch.tensor([tail_x(t, h0, dam_x=1.0, gravity=g) - 0.5,
                      front_x(t, h0, dam_x=1.0, gravity=g) + 0.5], dtype=torch.float64)
    h = depth(x, t, h0, dam_x=1.0, gravity=g)
    assert float(h[0]) == pytest.approx(h0, rel=1.0e-15)
    assert float(h[1]) == 0.0


def test_released_volume_matches_the_integral_of_the_profile():
    """The closed form ``h0 c0 t`` and the profile integral must agree.

    This is a check on the *shape* of the solution, not on its normalisation:
    the two sides are computed by unrelated routes.
    """
    for h0 in (0.05, 0.1, 0.2):
        for t in (0.05, 0.2, 0.5):
            exact = released_volume(t, h0)
            numeric = profile_mass(t, h0, points=200001)
            assert numeric == pytest.approx(exact, rel=2.0e-6)


def test_riemann_invariant_is_constant_across_the_fan():
    """``u + 2 sqrt(g h) = 2 c0`` for every point of the rarefaction fan."""
    h0, t, g = 0.1, 0.4, 9.81
    c0 = wave_speed(h0, g)
    x = torch.linspace(-c0 * t + 1.0e-6, 2.0 * c0 * t - 1.0e-6, 64,
                       dtype=torch.float64)
    h = depth(x, t, h0, dam_x=0.0, gravity=g)
    u = velocity(x, t, h0, dam_x=0.0, gravity=g)
    invariant = u + 2.0 * torch.sqrt(g * h)
    assert torch.allclose(invariant, torch.full_like(invariant, 2.0 * c0), rtol=1.0e-9)


def test_front_speed_is_twice_the_wave_speed():
    h0, g = 0.1, 9.81
    t = 0.25
    c0 = wave_speed(h0, g)
    z0, z1 = front_x(t, h0, gravity=g), front_x(t + 1.0e-6, h0, gravity=g)
    assert (z1 - z0) / 1.0e-6 == pytest.approx(2.0 * c0, rel=1.0e-6)
    # and the analytic velocity field just inside the front equals that speed
    # (the point x >= z_front is dry, so it is approached from the wet side)
    u_front = velocity(torch.tensor([z0 - 1.0e-9], dtype=torch.float64), t, h0,
                       gravity=g)
    assert float(u_front) == pytest.approx(2.0 * c0, rel=1.0e-6)


def test_depth_decreases_monotonically_through_the_fan():
    h0, t, g = 0.1, 0.3, 9.81
    c0 = wave_speed(h0, g)
    x = torch.linspace(-c0 * t, 2.0 * c0 * t, 200, dtype=torch.float64)
    h = depth(x, t, h0, dam_x=0.0, gravity=g)
    assert bool((h.diff() <= 1.0e-15).all())


def test_validity_window_is_the_time_for_the_tail_to_reach_the_wall():
    h0, a, g = 0.1, 1.0, 9.81
    t_valid = validity_window(h0, a, gravity=g)
    assert tail_x(t_valid, h0, dam_x=a, gravity=g) == pytest.approx(0.0, abs=1.0e-12)


def test_particle_depth_measure_recovers_a_known_depth():
    """A lattice filling ``[0, L] x [0, H]`` must measure depth ``H`` everywhere."""
    dx, a, h0 = 0.01, 0.5, 0.10
    xs = torch.arange(dx * 0.5, a, dx, dtype=torch.float64)
    ys = torch.arange(dx * 0.5, h0, dx, dtype=torch.float64)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    centres, measured = particle_depth(pos, dx, bin_width=2.0 * dx, lo=0.0, hi=a)
    wet = measured > 0.5 * h0
    assert bool(wet.any())
    assert torch.allclose(measured[wet], torch.full_like(measured[wet], h0), rtol=0.05,
                          atol=1.5 * dx)
    # the counting measure is the mass distribution, so it cannot lose particles
    total = float((measured * 2.0 * dx).sum())
    assert total == pytest.approx(pos.shape[0] * dx * dx, rel=1.0e-12)


def test_profile_l1_error_is_zero_for_identical_profiles():
    x = torch.linspace(0.0, 1.0, 51, dtype=torch.float64)
    h = depth(x, 0.3, 0.1, dam_x=0.5)
    assert profile_l1_error(x, h, h) == 0.0
    with pytest.raises(ValueError):
        profile_l1_error(x, h, torch.zeros_like(h))


def test_solver_reproduces_the_ritter_front_in_the_shallow_limit():
    """Fast regression guard: one coarse shallow run reproduces the Ritter front.

    The tolerance is the *coarse* case's, not a claim of accuracy: ``dx = 0.02``
    with five layers under-predicts the front by ~17 % of ``Z`` at this window
    (measured: 0.4 a/sqrt(gH)). What this test pins is that the front is on the
    right side of the analytic value, advances with the analytic speed to within
    the discretisation error of the configuration, and that the run stays
    conservative. The error ladder and the observed order live in
    ``docs/benchmarks/ritter-shallow-limit.md``.
    """
    a, h0, dx = 1.0, 0.1, 0.02
    c0 = wave_speed(h0)
    span = 0.4 * a / c0
    cfg = DamBreakConfig(dx=dx, column_width=a, column_height=h0,
                         pool_width=a + 2.0 * c0 * span + 6 * dx,
                         tank_height=1.5 * h0, sound_speed=20.0 * c0,
                         viscosity=0.0, hydrostatic_init=False, device="cpu",
                         output_every=20)
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(math.ceil(span / cfg.dt)) + 1
    run = run_case(cfg)
    assert run.mass_drift == 0.0
    assert run.escaped == 0
    front = run.front[-1]
    reference = front_x(run.simulated_time, h0, dam_x=a)
    relative = (front - reference) / reference
    assert -0.25 < relative < 0.05, relative          # measured: about -0.17
    # the discretisation error must not be an instability: the front still moves
    speed = (run.front[-1] - run.front[0]) / (run.times[-1] - run.times[0])
    assert speed == pytest.approx(2.0 * c0, rel=0.25)
