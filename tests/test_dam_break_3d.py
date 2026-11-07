"""Tests for the 3-D dam-break benchmark (IN16-K6).

Coverage:
1. Column geometry & particle count (3-D lattice)
2. Hydrostatic initialisation matches analytic profile
3. Front position & Ritter reference
4. Invalid cases are rejected (negative tests)
5. Short run is bounded, conserves mass, energy drift bounded
6. Wall geometry spans the 3-D tank
7. Same-seed reproducibility
8. Particle overlap / no-particle initial rejection (negative test)
9. 3-D kernel properties
10. Boundary clamping
"""
import math

import pytest
import torch

from tensorsph.dam_break_3d import (
    DamBreak3DConfig,
    DamBreak3DRun,
    apply_boundary_3d,
    cubic_spline_kernel_3d,
    cubic_spline_gradient_3d,
    front_position_3d,
    make_column_3d,
    pair_kernels_3d,
    ritter_front,
    ritter_front_over_a,
    run_case,
    validate_case,
    wall_lattice_3d,
)
from tensorsph.droplet_impact import hydrostatic_density


def small_case(**overrides):
    """Small 3-D dam-break config for fast tests (~50 particles, ~50 steps)."""
    base = dict(
        dx=0.04, pool_width=0.4, tank_depth=0.2,
        column_width=0.08, column_depth=0.08, tank_height=0.3,
        sound_speed=25.0, steps=50, output_every=25, device="cpu",
    )
    base.update(overrides)
    cfg = DamBreak3DConfig(**base)
    cfg.column_height = 2.0 * cfg.column_width
    if cfg.sound_speed > 0:
        cfg.dt = cfg.dt_from_courant()
    return cfg


# --------------------------------------------------------------------------- #
# Test 1: Column geometry & particle count
# --------------------------------------------------------------------------- #

def test_column_3d_is_inside_the_tank_and_rests_on_the_floor():
    cfg = small_case()
    positions, mass = make_column_3d(cfg, torch.device("cpu"))
    assert positions.shape[1] == 3, "3-D positions must have 3 coordinates"
    assert mass.shape == (positions.shape[0],)
    assert float(positions[:, 0].max()) < cfg.column_width
    assert float(positions[:, 0].min()) > 0.0
    assert float(positions[:, 1].min()) == pytest.approx(cfg.dx * 0.5)
    assert float(positions[:, 1].max()) < cfg.column_height
    assert float(positions[:, 2].max()) < cfg.column_depth
    assert float(positions[:, 2].min()) > 0.0
    assert (mass > 0).all()


# --------------------------------------------------------------------------- #
# Test 2: Hydrostatic initialisation
# --------------------------------------------------------------------------- #

def test_hydrostatic_initialisation_3d_matches_the_analytic_profile():
    cfg = small_case(hydrostatic_init=True)
    positions, mass = make_column_3d(cfg, torch.device("cpu"))
    expected = hydrostatic_density(positions[:, 1], cfg,
                                   surface=cfg.column_height) * cfg.dx ** 3
    torch.testing.assert_close(mass, expected)

    uniform_cfg = small_case(hydrostatic_init=False)
    _, uniform_mass = make_column_3d(uniform_cfg, torch.device("cpu"))
    assert float(mass.sum()) > float(uniform_mass.sum())


# --------------------------------------------------------------------------- #
# Test 3: Front position & Ritter reference
# --------------------------------------------------------------------------- #

def test_front_and_ritter_reference_3d():
    cfg = small_case()
    positions = torch.tensor(
        [[0.1, 0.1, 0.05], [0.3, 0.2, 0.05], [0.2, 0.15, 0.08]],
        dtype=torch.float32,
    )
    assert front_position_3d(positions, cfg) == pytest.approx(0.3)
    assert front_position_3d(torch.empty((0, 3)), cfg) == 0.0
    t = 0.2
    assert ritter_front(t, cfg) == pytest.approx(
        2.0 * math.sqrt(cfg.gravity * cfg.column_height) * t)
    # Non-dimensional Ritter: Z/a = 2*tau
    tau = t * math.sqrt(2.0 * cfg.gravity / cfg.column_width)
    assert ritter_front_over_a(tau) == pytest.approx(2.0 * tau)


# --------------------------------------------------------------------------- #
# Test 4: Invalid cases are rejected (negative tests)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("overrides,error", [
    (dict(dx=0.0), ValueError),
    (dict(column_width=0.0), ValueError),
    (dict(column_width=5.0), ValueError),       # wider than tank
    (dict(sound_speed=0.0), ValueError),
    (dict(boundary="clamp"), ValueError),        # only wall boundary supported
])
def test_invalid_cases_3d_are_rejected(overrides, error):
    cfg = small_case(**overrides)
    with pytest.raises(error):
        validate_case(cfg)


def test_time_step_above_courant_limit_is_rejected_3d():
    cfg = small_case()
    cfg.dt = 10.0 * cfg.dt_from_courant()
    with pytest.raises(ValueError, match="Courant"):
        validate_case(cfg)


def test_column_too_tall_is_rejected():
    cfg = small_case()
    cfg.column_height = cfg.tank_height + 1.0
    with pytest.raises(ValueError, match="column_height"):
        validate_case(cfg)


def test_column_too_deep_is_rejected():
    cfg = small_case()
    cfg.column_depth = cfg.tank_depth + 1.0
    with pytest.raises(ValueError, match="column_depth"):
        validate_case(cfg)


# --------------------------------------------------------------------------- #
# Test 5: Short run is bounded, conserves mass
# --------------------------------------------------------------------------- #

def test_short_run_3d_is_bounded_and_conserves_mass():
    cfg = small_case(steps=50, output_every=25)
    run = run_case(cfg)
    assert run.escaped == 0
    assert len(run.times) == len(run.front) == len(run.kinetic_energy)
    assert run.front[-1] >= run.front[0], "the front must not retreat"
    assert run.max_speed < 15.0
    assert run.max_density_compression < 0.25
    assert run.mass > 0
    # Potential energy decreases as the column collapses.
    assert run.potential_energy[-1] < run.potential_energy[0]
    # Mass conservation (particle masses are constant).
    assert run.initial_mass > 0.0
    assert run.final_mass == run.initial_mass
    assert run.mass_drift == 0.0


def test_energy_drift_bounded():
    """Total energy drift should be bounded for a short run.

    In WCSPH with density summation, mechanical energy (KE+PE) is not strictly
    conserved because pressure work is not tracked.  The drift is bounded by
    the numerical dissipation of the scheme.
    """
    cfg = small_case(steps=50, output_every=25)
    run = run_case(cfg)
    # Energy drift is bounded (not necessarily < 1 % for short runs).
    assert abs(run.energy_drift) < 0.20, (
        f"energy drift {run.energy_drift:.4e} exceeds 20 %")


# --------------------------------------------------------------------------- #
# Test 6: Wall geometry spans the 3-D tank
# --------------------------------------------------------------------------- #

def test_wall_geometry_3d_spans_the_tank():
    cfg = small_case()
    wall = wall_lattice_3d(cfg, torch.device("cpu"), top=cfg.tank_top())
    assert wall.shape[1] == 3
    assert float(wall[:, 1].min()) < 0.0
    assert float(wall[:, 0].min()) < 0.0
    assert float(wall[:, 0].max()) > cfg.pool_width
    assert float(wall[:, 1].max()) <= cfg.tank_top() + 1e-6
    assert float(wall[:, 2].min()) < 0.0
    assert float(wall[:, 2].max()) > cfg.tank_depth


# --------------------------------------------------------------------------- #
# Test 7: Same-seed reproducibility
# --------------------------------------------------------------------------- #

def test_same_seed_reproducibility_3d():
    cfg = small_case(steps=30, output_every=15)
    run1 = run_case(cfg, seed=42)
    run2 = run_case(cfg, seed=42)
    assert run1.front == run2.front
    assert run1.kinetic_energy == run2.kinetic_energy
    assert run1.potential_energy == run2.potential_energy
    torch.testing.assert_close(run1.final_positions, run2.final_positions)


def test_different_seeds_diverge_3d():
    """Different seeds should produce different trajectories (chaotic sensitivity)."""
    cfg = small_case(steps=50, output_every=25)
    run1 = run_case(cfg, seed=1)
    run2 = run_case(cfg, seed=2)
    # Front positions should differ (at least at some point)
    fronts_differ = any(abs(f1 - f2) > 1e-8 for f1, f2 in zip(run1.front, run2.front))
    assert fronts_differ, "different seeds should produce different fronts"


# --------------------------------------------------------------------------- #
# Test 8: Particle overlap / no-particle initial rejection (negative test)
# --------------------------------------------------------------------------- #

def test_no_particles_when_column_too_small():
    """A column smaller than dx in any dimension produces no particles."""
    cfg = DamBreak3DConfig(
        dx=0.1, pool_width=0.6, tank_depth=0.3,
        column_width=0.05, column_depth=0.05, column_height=0.05,
        tank_height=0.4, sound_speed=25.0, steps=10, output_every=5,
        device="cpu",
    )
    cfg.dt = cfg.dt_from_courant()
    with pytest.raises(ValueError, match="too coarse"):
        make_column_3d(cfg, torch.device("cpu"))


# --------------------------------------------------------------------------- #
# Test 9: 3-D kernel properties
# --------------------------------------------------------------------------- #

def test_3d_kernel_normalisation():
    """The 3-D kernel at r=0 should equal 1/(pi*h^3)."""
    h = 1.0
    r = torch.tensor([0.0], dtype=torch.float64)
    w = cubic_spline_kernel_3d(r, h)
    expected = 1.0 / math.pi
    assert float(w) == pytest.approx(expected, rel=1e-10)


def test_3d_kernel_gradient_at_origin():
    """The kernel gradient at r=0 should be zero (symmetry)."""
    h = 1.0
    delta = torch.zeros((1, 3), dtype=torch.float64)
    grad = cubic_spline_gradient_3d(delta, h)
    assert torch.allclose(grad, torch.zeros_like(grad), atol=1e-14)


def test_3d_kernel_support_cutoff():
    """The kernel should be zero for q >= 2."""
    h = 1.0
    r = torch.tensor([2.0, 2.5, 10.0], dtype=torch.float64)
    w = cubic_spline_kernel_3d(r, h)
    assert (w == 0.0).all()


def test_3d_pair_kernels_consistency():
    """pair_kernels_3d should return the same W and grad W as the individual functions."""
    h = 1.3 * 0.02
    delta = torch.tensor([[0.01, 0.02, 0.015]], dtype=torch.float64)
    w1 = cubic_spline_kernel_3d(torch.linalg.vector_norm(delta, dim=-1), h)
    g1 = cubic_spline_gradient_3d(delta, h)
    w2, g2 = pair_kernels_3d(delta, h)
    torch.testing.assert_close(w1, w2)
    torch.testing.assert_close(g1, g2)


# --------------------------------------------------------------------------- #
# Test 10: Boundary clamping
# --------------------------------------------------------------------------- #

def test_boundary_3d_clamps_particles():
    """apply_boundary_3d should clamp particles inside the tank."""
    cfg = small_case()
    pos = torch.tensor([
        [-0.1, 0.1, 0.05],   # outside left
        [0.5, 0.1, 0.05],    # outside right (tw=0.4)
        [0.1, -0.1, 0.05],   # below floor
        [0.1, 0.4, 0.05],    # above top (th=0.3)
        [0.1, 0.1, -0.1],    # outside front
        [0.1, 0.1, 0.3],     # outside back (td=0.2)
    ], dtype=torch.float32)
    vel = torch.ones_like(pos)
    events = apply_boundary_3d(pos, vel, cfg)
    assert events == 6
    assert float(pos[:, 0].min()) >= -1e-6
    assert float(pos[:, 0].max()) <= cfg.pool_width + 1e-6
    assert float(pos[:, 1].min()) >= -1e-6
    assert float(pos[:, 1].max()) <= cfg.tank_top() + 1e-6
    assert float(pos[:, 2].min()) >= -1e-6
    assert float(pos[:, 2].max()) <= cfg.tank_depth + 1e-6


def test_dimensionless_front():
    """Dimensionless front Z/a vs tau should be computable."""
    cfg = small_case(steps=30, output_every=15)
    run = run_case(cfg)
    tau, za = run.dimensionless(cfg)
    assert len(tau) == len(za)
    assert all(t >= 0 for t in tau)
    # Front starts at x=column_width, so Z/a >= 1.0 at t=0
    assert za[0] > 0.5, f"initial Z/a={za[0]} is unreasonably small"
    assert all(z >= 0.5 for z in za)
