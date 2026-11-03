"""Analytic verification of the wall contact-angle boundary condition.

``apply_contact_angle_boundary`` is exact geometry, so it can be scored against
Young's relation ``n . w = cos(theta)`` (``tensorsph.wetting`` module docstring)
with no tolerance beyond floating-point round-off.  These tests do exactly
that, for every contact angle in a sweep and for walls at four orientations, so
a hard-coded ``+y`` wall normal or a swapped ``sin``/``cos`` cannot pass.

The two integration tests in ``tests/test_contact_angle.py`` that ask for a
non-zero surface-tension acceleration on a synthetic six-particle cloud are
marked ``xfail(strict=True)`` there: the CSF curvature vanishes on that cloud
for a reason that has nothing to do with wetting (see
``docs/verification_surface_tension_laplace.md``).  What is verified here is the
boundary condition itself, plus the fact that the force path accepts a wall
argument at all -- before this change ``rhs`` passed ``wall=`` to a function
that had no such parameter, so *every* wall-bounded run with
``surface_tension_coefficient > 0`` raised ``TypeError``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.droplet_impact import Config, cubic_spline_kernel
from tensorsph.surface_tension import (
    apply_contact_angle_boundary,
    compute_surface_tension_force,
)
from tensorsph.wetting import measure_contact_angle, wall_normals

DX = 0.01
H = 1.3 * DX


def flat_wall(half_length=10 * DX, spacing=DX, rotation=0.0, offset=0.0):
    """A planar lattice of wall particles, optionally rotated and shifted.

    Returns ``(wall, w)`` where ``w`` is the analytic unit normal of the plane,
    pointing from the wall into the fluid (``+y`` before rotation).
    """
    x = torch.arange(-half_length, half_length + 0.5 * spacing, spacing,
                     dtype=torch.float64)
    points = torch.stack((x, torch.zeros_like(x)), dim=1)
    points = points + torch.tensor([0.0, offset], dtype=torch.float64)
    angle = torch.tensor(rotation, dtype=torch.float64)
    rot = torch.stack((
        torch.stack((angle.cos(), -angle.sin())),
        torch.stack((angle.sin(), angle.cos())),
    ))
    wall = points @ rot.T
    w = rot @ torch.tensor([0.0, 1.0], dtype=torch.float64)
    return wall, w


def point_above_wall(distance=0.5 * H, rotation=0.0, offset=0.0):
    """A fluid particle ``distance`` above the wall plane, on the wall normal."""
    angle = torch.tensor(rotation, dtype=torch.float64)
    rot = torch.stack((
        torch.stack((angle.cos(), -angle.sin())),
        torch.stack((angle.sin(), angle.cos())),
    ))
    local = torch.tensor([0.0, distance + offset], dtype=torch.float64)
    return (rot @ local).reshape(1, 2)


ANGLES = [10.0, 20.0, 45.0, 60.0, 90.0, 120.0, 135.0, 160.0, 170.0]
ROTATIONS = [0.0, math.pi / 6, math.pi / 2, 1.7 * math.pi]


@pytest.mark.parametrize("degrees", ANGLES)
@pytest.mark.parametrize("rotation", ROTATIONS)
def test_enforced_angle_matches_youngs_relation(degrees, rotation):
    """The constructed normal satisfies ``n . w = cos(theta)``."""
    wall, w = flat_wall(rotation=rotation)
    pos = point_above_wall(rotation=rotation)
    theta = math.radians(degrees)

    # An oblique incoming normal: the tangential direction must be preserved.
    grad_c = torch.tensor([[-0.8, 0.6]], dtype=torch.float64) * 2.5
    out = apply_contact_angle_boundary(grad_c, pos, wall, H, theta)

    measured = measure_contact_angle(out, w.reshape(1, 2))
    assert float(torch.abs(measured - theta)) < 1.0e-6, (
        f"theta {degrees} deg on a wall rotated {math.degrees(rotation):.1f} deg: "
        f"measured {math.degrees(float(measured)):.6f} deg")


@pytest.mark.parametrize("degrees", ANGLES)
def test_gradient_magnitude_is_preserved_exactly(degrees):
    wall, _w = flat_wall()
    pos = point_above_wall()
    grad_c = torch.tensor([[-0.8, 0.6]], dtype=torch.float64) * 2.5
    out = apply_contact_angle_boundary(grad_c, pos, wall, H,
                                       math.radians(degrees))
    before = float(torch.linalg.vector_norm(grad_c))
    after = float(torch.linalg.vector_norm(out))
    assert after == pytest.approx(before, rel=1e-12)


def test_tangential_direction_is_kept():
    """The boundary condition rotates about the wall normal; it must not pick a
    different azimuth (that would be a fabricated direction, not a rotation)."""
    wall, _w = flat_wall()
    pos = point_above_wall()
    grad_c = torch.tensor([[-1.0, 0.5]], dtype=torch.float64)
    out = apply_contact_angle_boundary(grad_c, pos, wall, H, math.radians(70.0))

    before = -grad_c[0] / torch.linalg.vector_norm(grad_c[0])
    after = -out[0] / torch.linalg.vector_norm(out[0])
    # Remove the wall-normal component; the tangential part must be parallel.
    w = torch.tensor([0.0, 1.0], dtype=torch.float64)
    tang = lambda v: v - (v @ w) * w
    tb, ta = tang(before), tang(after)
    cos = float((tb @ ta) / (torch.linalg.vector_norm(tb)
                             * torch.linalg.vector_norm(ta)))
    assert cos == pytest.approx(1.0, abs=1e-12)


def test_wall_normal_is_recovered_not_assumed():
    """``wall_normals`` must return the geometry of the wall it is given."""
    for rotation in ROTATIONS:
        wall, w = flat_wall(rotation=rotation)
        pos = point_above_wall(rotation=rotation)
        normal, distance, near = wall_normals(pos, wall, H)
        assert bool(near.all())
        assert float(distance[0]) == pytest.approx(0.5 * H, rel=1e-12)
        assert float(torch.linalg.vector_norm(normal[0] - w)) < 1.0e-12


def test_normal_is_parallel_to_the_wall_inside_the_lattice():
    """For a particle sitting in the middle of a wall slab the weighted mean
    cancels (opposing sides), so no wall exists there and the boundary condition
    must stay inert instead of inventing a direction."""
    wall = torch.stack((
        torch.linspace(-5 * DX, 5 * DX, 11, dtype=torch.float64),
        torch.full((11,), -DX, dtype=torch.float64),
    ), dim=1)
    below = torch.stack((
        torch.linspace(-5 * DX, 5 * DX, 11, dtype=torch.float64),
        torch.full((11,), -3 * DX, dtype=torch.float64),
    ), dim=1)
    wall = torch.cat((wall, below), dim=0)
    pos = torch.tensor([[-ly, -2 * DX] for ly in (0.02, 0.0, -0.02)],
                       dtype=torch.float64)
    grad_c = torch.tensor([[0.7, 0.7]] * 3, dtype=torch.float64)
    out = apply_contact_angle_boundary(grad_c, pos, wall, H, math.radians(30.0))
    assert torch.allclose(out, grad_c)


def test_far_field_and_empty_wall_are_no_ops():
    empty = torch.empty((0, 2), dtype=torch.float64)
    grad_c = torch.tensor([[0.3, -0.4]], dtype=torch.float64)
    pos = point_above_wall()
    assert torch.allclose(apply_contact_angle_boundary(grad_c, pos, empty, H, 1.0),
                          grad_c)
    wall, _w = flat_wall()
    far = pos + torch.tensor([0.0, 50 * H], dtype=torch.float64)
    assert torch.allclose(apply_contact_angle_boundary(grad_c, far, wall, H, 1.0),
                          grad_c)


def test_force_path_accepts_a_wall_and_stays_finite():
    """Regression: ``rhs`` has always called
    ``compute_surface_tension_force(..., wall=wall)``; the definition had no
    ``wall`` parameter, so wall-bounded surface-tension runs raised TypeError.
    """
    cfg = Config(dx=DX, device="cpu", surface_tension_coefficient=0.072,
                 contact_angle=math.pi / 3)
    wall, _w = flat_wall()
    pos = torch.tensor([[0.0, 0.6 * H], [DX, 0.8 * H], [2 * DX, 0.7 * H]],
                       dtype=torch.float64)
    mass = torch.full((3,), cfg.rho0 * DX * DX, dtype=torch.float64)
    density = torch.full((3,), cfg.rho0, dtype=torch.float64)
    pairs = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.long)
    acc = compute_surface_tension_force(pos, density, mass, pairs, cfg, wall=wall)
    assert acc.shape == (3, 2)
    assert bool(torch.isfinite(acc).all())
    # With contact angle disabled the same call must still work.
    cfg.contact_angle = None
    acc_no = compute_surface_tension_force(pos, density, mass, pairs, cfg,
                                          wall=wall)
    assert bool(torch.isfinite(acc_no).all())