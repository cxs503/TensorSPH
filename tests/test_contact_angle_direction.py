"""Analytic verification of the wall contact-angle boundary condition.

``docs/contact_angle_boundary.md`` claims that after the boundary condition the
interface normal satisfies Young's relation and that the gradient magnitude is
preserved.  The operator-level tests that shipped with mission m2090 only
checked array shapes and magnitudes, so a wrong *sign* (wetting and non-wetting
swapped) would have passed.  This file scores the operator against the analytic
statement it implements:

    n_out = -grad(c) / |grad(c)|      (outward interface normal)
    angle(n_out, n_w) = theta         (Young's relation, n_w = wall -> fluid)

and against the two physical limits: ``theta = 0`` (wetting film, interface
parallel to the wall) and ``theta = pi`` (completely non-wetting, interface
wrapped over the wall).  Everything here is a single static evaluation of
O(100) particles, so the file runs in well under a second.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.droplet_impact import Config, density_from_neighbor_pairs
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import (apply_contact_angle_boundary,
                                       compute_surface_tension_force)

THETAS = [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2,
          2 * math.pi / 3, 5 * math.pi / 6, math.pi]


def _flat_wall(n=40, dx=0.01, y=0.0):
    """One row of wall particles along y = const."""
    return torch.tensor([[(k - n // 2) * dx, y] for k in range(n)],
                        dtype=torch.float64)


def _one_particle(height=0.01):
    """A single fluid particle a fraction of h above the wall."""
    return torch.tensor([[0.0, height]], dtype=torch.float64)


def _angle_deg(a, b):
    ca = (a * b).sum() / (a.norm() * b.norm())
    return float(torch.rad2deg(torch.arccos(ca.clamp(-1.0, 1.0))))


# --------------------------------------------------------------------------- #
# analytic statement: angle(outward normal, wall normal) == theta
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("theta", THETAS)
def test_outward_normal_makes_the_contact_angle_with_the_wall(theta):
    pos = _one_particle()
    wall = _flat_wall()
    h = 0.013
    grad = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    out = apply_contact_angle_boundary(grad, pos, wall, h, theta)

    n_wall = torch.tensor([0.0, 1.0], dtype=torch.float64)
    n_out = -out[0] / out[0].norm()
    # Young's relation: the outward normal makes exactly theta with the wall
    # normal, to round-off (the operator is a pure rotation).
    assert _angle_deg(n_out, n_wall) == pytest.approx(math.degrees(theta),
                                                      abs=1e-9)


@pytest.mark.parametrize("theta", THETAS)
def test_gradient_magnitude_is_preserved_exactly(theta):
    pos = _one_particle()
    wall = _flat_wall()
    grad = torch.tensor([[0.3, -0.7]], dtype=torch.float64)
    out = apply_contact_angle_boundary(grad, pos, wall, 0.013, theta)
    assert float(out.norm()) == pytest.approx(float(grad.norm()), rel=1e-12)


def test_zero_angle_points_the_gradient_into_the_wall():
    """A perfectly wetting film puts the interface parallel to the wall, so
    grad(c) is anti-parallel to the wall normal."""
    grad = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    out = apply_contact_angle_boundary(grad, _one_particle(), _flat_wall(),
                                       0.013, 0.0)
    assert out[0].tolist() == pytest.approx([0.0, -1.0], abs=1e-12)


def test_pi_angle_points_the_gradient_away_from_the_wall():
    grad = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    out = apply_contact_angle_boundary(grad, _one_particle(), _flat_wall(),
                                       0.013, math.pi)
    assert out[0].tolist() == pytest.approx([0.0, 1.0], abs=1e-12)


def test_tilt_side_is_kept_from_the_incoming_gradient():
    """The tangential component must not flip: the BC must not turn a wetting
    interface into a non-wetting one on the other side of the contact line."""
    wall = _flat_wall()
    # gradient tilted towards +x: interface leans to the +x side of the wall
    grad = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    out = apply_contact_angle_boundary(grad, _one_particle(), wall, 0.013,
                                       math.pi / 3)
    assert float(out[0, 0]) > 0.0


def test_operator_is_idempotent():
    """Applying the same contact angle twice must not drift the direction."""
    grad = torch.tensor([[-0.4, 0.6], [0.9, 0.1]], dtype=torch.float64)
    pos = torch.tensor([[0.0, 0.01], [0.5 * 0.01, 0.012]], dtype=torch.float64)
    wall = _flat_wall()
    once = apply_contact_angle_boundary(grad, pos, wall, 0.013, math.pi / 3)
    twice = apply_contact_angle_boundary(once, pos, wall, 0.013, math.pi / 3)
    assert torch.allclose(once, twice, atol=1e-12)


# --------------------------------------------------------------------------- #
# scope of the operator: only the wall band, and no-ops stay bit-exact
# --------------------------------------------------------------------------- #
def test_far_particles_are_bit_identical():
    grad = torch.tensor([[1.0, 0.5], [-0.2, 0.3]], dtype=torch.float64)
    pos = torch.tensor([[0.0, 10.0], [0.0, 20.0]], dtype=torch.float64)
    out = apply_contact_angle_boundary(grad, pos, _flat_wall(), 0.013,
                                       math.pi / 3)
    assert torch.equal(out, grad)


def test_no_wall_is_a_noop():
    grad = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
    pos = _one_particle()
    assert torch.equal(apply_contact_angle_boundary(grad, pos, None, 0.013,
                                                    math.pi / 3), grad)
    empty = torch.empty((0, 2), dtype=torch.float64)
    assert torch.equal(apply_contact_angle_boundary(grad, pos, empty, 0.013,
                                                    math.pi / 3), grad)


def test_particle_coincident_with_a_wall_particle_is_left_alone():
    """No wall normal can be defined at zero distance: skip, do not invent."""
    grad = torch.tensor([[0.3, 0.4]], dtype=torch.float64)
    pos = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    wall = _flat_wall()
    out = apply_contact_angle_boundary(grad, pos, wall, 0.013, math.pi / 3)
    assert torch.equal(out, grad)


def test_input_tensor_is_not_mutated():
    grad = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    keep = grad.clone()
    apply_contact_angle_boundary(grad, _one_particle(), _flat_wall(), 0.013,
                                 math.pi / 3)
    assert torch.equal(grad, keep)


@pytest.mark.parametrize("theta", [-0.1, math.pi + 0.1, float("nan")])
def test_invalid_contact_angle_is_rejected(theta):
    grad = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    with pytest.raises(ValueError):
        apply_contact_angle_boundary(grad, _one_particle(), _flat_wall(),
                                     0.013, theta)


def test_dtype_and_device_are_preserved():
    for dtype in (torch.float32, torch.float64):
        grad = torch.tensor([[0.0, 1.0]], dtype=dtype)
        out = apply_contact_angle_boundary(grad, _one_particle().to(dtype),
                                           _flat_wall().to(dtype), 0.013,
                                           math.pi / 3)
        assert out.dtype is dtype


# --------------------------------------------------------------------------- #
# physics: a sessile cap must spread when wetting and bead up when not
# --------------------------------------------------------------------------- #
def _cap(radius=0.06, dx=0.01, rho0=1000.0):
    """Half disc of hex-packed particles standing on y = 0.5 dx."""
    dy = dx * math.sqrt(3.0) / 2.0
    pts = []
    j = 0
    y = 0.0
    while y <= radius:
        x = -radius + (0.5 * dx if j % 2 else 0.0)
        while x <= radius:
            if x * x + y * y <= radius * radius:
                pts.append((x, y + 0.5 * dx))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), rho0 * dx * dy, dtype=torch.float64)


def _cap_forces(theta, sigma=0.072, dx=0.01):
    pos, mass = _cap(dx=dx)
    wall = _flat_wall(n=60, dx=dx)
    base = dict(dx=dx, rho0=1000.0, sound_speed=20.0, gravity=0.0,
                viscosity=0.01, surface_tension_coefficient=sigma)
    cfg = Config(**base, contact_angle=theta)
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    with_bc = compute_surface_tension_force(pos, rho, mass, pairs, cfg, wall=wall)
    without = compute_surface_tension_force(pos, rho, mass, pairs, Config(**base),
                                            wall=wall)
    return pos, wall, h, with_bc, without


def test_wetting_spreads_the_drop_and_non_wetting_beads_it_up():
    """The tangential part of the BC-induced force must push the contact line
    outwards for theta < pi/2 and inwards for theta > pi/2."""
    pos, _, _, wet, ref = _cap_forces(math.pi / 6)
    _, _, _, dry, _ = _cap_forces(5 * math.pi / 6)
    right = pos[:, 0] > 0.02
    assert float((wet - ref)[right, 0].sum()) > 0.0     # wetting: pushes out (+x)
    assert float((dry - ref)[right, 0].sum()) < 0.0     # non-wetting: pulls in
    # and relative to the non-wetting case the wetting force is outward, which
    # is the physically meaningful antisymmetry (an absolute mirror test would
    # require a mirror-symmetric lattice, which the staggered hex rows are not).
    assert float((wet - dry)[right, 0].sum()) > 0.0


def test_boundary_condition_only_touches_the_wall_band():
    pos, wall, h, with_bc, without = _cap_forces(math.pi / 3)
    delta = (with_bc - without).norm(dim=1)
    dist = torch.cdist(pos, wall).min(dim=1).values
    assert float(delta.max()) > 0.0                       # it does something
    # locality: the BC acts inside the wall band, and the CSF divergence then
    # carries it at most one more kernel radius, so > 4h is untouched.
    far = dist > 4.0 * h
    assert int(far.sum()) > 0
    assert torch.equal(delta[far], torch.zeros_like(delta[far]))


def test_contact_angle_none_is_bit_exactly_the_old_behaviour():
    """Backward compatibility: passing a wall must not change anything while
    ``contact_angle`` is unset."""
    pos, mass = _cap()
    wall = _flat_wall(n=60)
    cfg = Config(dx=0.01, rho0=1000.0, sound_speed=20.0, gravity=0.0,
                 viscosity=0.01, surface_tension_coefficient=0.072)
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    with_wall = compute_surface_tension_force(pos, rho, mass, pairs, cfg, wall=wall)
    no_wall = compute_surface_tension_force(pos, rho, mass, pairs, cfg)
    assert torch.equal(with_wall, no_wall)