"""Geometry of the contact-angle boundary condition, checked against its definition.

The contact angle is defined as the angle, measured inside the fluid, between the
interface normal and the wall normal.  For the colour-gradient formulation used
in :mod:`tensorsph.surface_tension` that makes the boundary condition an exact
geometric statement rather than a modelling choice:

    angle(n_new, n_wall) = theta   and   |n_new| = |n_old|

These tests check that statement directly (no reference solution, no
discretisation error) plus the containment property the solver relies on: only
particles within ``2h`` of a wall may be touched.
"""
import math

import pytest
import torch

from tensorsph.droplet_impact import Config, wall_lattice
from tensorsph.surface_tension import (
    apply_contact_angle_boundary,
    compute_surface_tension_force,
)


def _wall_and_particles(device="cpu", n=64):
    """A wall row plus random particles inside / outside the influence band."""
    cfg = Config(dx=0.05, pool_width=0.35, pool_depth=0.15, drop_radius=0.04,
                 drop_center_y=0.25, device=device)
    h = cfg.h_factor * cfg.dx
    wall_x = torch.arange(-0.1, 0.5 + 1e-9, cfg.dx, device=device)
    wall = torch.stack((wall_x, torch.full_like(wall_x, -0.5 * cfg.dx)), dim=1)

    gen = torch.Generator().manual_seed(7)
    x = torch.rand(n, 1, generator=gen) * 0.35
    y_in = torch.rand(n, 1, generator=gen) * (1.5 * h)
    y_out = 4.0 * h + torch.rand(n, 1, generator=gen) * h
    pos = torch.cat((torch.cat((x, y_in), dim=1),
                     torch.cat((x.clone(), y_out), dim=1)), dim=0).to(device)

    phi = torch.rand(2 * n, 1, generator=gen) * 2.0 * math.pi
    mag = 0.2 + torch.rand(2 * n, 1, generator=gen)
    grad = (torch.cat((torch.cos(phi), torch.sin(phi)), dim=1) * mag).to(device)
    return cfg, pos, grad, wall, h, n


def _wall_normals(pos, wall):
    d = torch.cdist(pos, wall)
    delta = pos - wall[d.argmin(dim=1)]
    return delta / torch.linalg.vector_norm(delta, dim=1, keepdim=True)


@pytest.mark.parametrize("theta_deg", [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
def test_recovered_angle_matches_prescription(theta_deg):
    """The angle between the reconstructed normal and the wall normal is theta."""
    device = "cpu"
    _, pos, grad, wall, h, n = _wall_and_particles(device)
    theta = math.radians(theta_deg)

    out = apply_contact_angle_boundary(grad, pos, wall, h, theta)
    n_wall = _wall_normals(pos, wall)

    cos_meas = ((out[:n] * n_wall[:n]).sum(dim=1)
                / torch.linalg.vector_norm(out[:n], dim=1))
    measured = torch.rad2deg(torch.acos(cos_meas.clamp(-1.0, 1.0)))
    # 0 and 180 deg are antipodal: the tangential direction is undefined there,
    # so the angle is reached from either side (|cos| = 1).
    if theta_deg in (0.0, 180.0):
        measured = torch.rad2deg(torch.acos(cos_meas.abs().clamp(-1.0, 1.0)))
        expected = 0.0
    else:
        expected = theta_deg
    # 0.05 deg is the float32 round-off of acos near 1, not a model error.
    assert torch.allclose(measured,
                          torch.full_like(measured, expected), atol=5.0e-2)


def test_magnitude_preserved_and_input_untouched():
    """|grad c| is preserved (so the CSF delta function is unchanged)."""
    device = "cpu"
    _, pos, grad, wall, h, n = _wall_and_particles(device)
    grad_before = grad.clone()

    out = apply_contact_angle_boundary(grad, pos, wall, h, math.pi / 3)

    mag0 = torch.linalg.vector_norm(grad[:n], dim=1)
    mag1 = torch.linalg.vector_norm(out[:n], dim=1)
    assert torch.allclose(mag0, mag1, rtol=1e-5, atol=1e-7)
    assert torch.equal(grad, grad_before), "the boundary condition must not be applied in place"


def test_outside_band_is_untouched():
    """Particles further than 2h from the wall are returned unchanged."""
    device = "cpu"
    _, pos, grad, wall, h, n = _wall_and_particles(device)

    out = apply_contact_angle_boundary(grad, pos, wall, h, math.radians(45))
    assert torch.equal(out[n:], grad[n:])
    assert not torch.equal(out[:n], grad[:n])


def test_empty_wall_and_zero_gradient_are_safe():
    device = "cpu"
    _, pos, grad, wall, h, n = _wall_and_particles(device)
    empty = torch.empty((0, 2), dtype=pos.dtype, device=device)
    assert torch.equal(apply_contact_angle_boundary(grad, pos, empty, h, 0.3), grad)

    zero = torch.zeros_like(grad)
    assert torch.equal(apply_contact_angle_boundary(zero, pos, wall, h, 0.3), zero)


def test_degenerate_incoming_normal_still_reaches_theta():
    """A normal already parallel to the wall normal has no tangential part."""
    device = "cpu"
    cfg = Config(dx=0.05, device=device)
    h = cfg.h_factor * cfg.dx
    pos = torch.tensor([[0.2, 0.01]], dtype=torch.float32, device=device)
    wall = torch.tensor([[0.2, -0.025]], dtype=torch.float32, device=device)
    grad = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=device)

    out = apply_contact_angle_boundary(grad, pos, wall, h, math.pi / 2)
    n_wall = _wall_normals(pos, wall)
    cos_meas = (out[0] * n_wall[0]).sum() / torch.linalg.vector_norm(out[0])
    assert torch.isclose(torch.rad2deg(torch.acos(cos_meas.clamp(-1, 1))),
                         torch.tensor(90.0), atol=1e-3)
    assert torch.isclose(torch.linalg.vector_norm(out[0]),
                         torch.linalg.vector_norm(grad[0]))


def test_force_only_changes_inside_wall_band():
    """The CSF force is unchanged outside 2h and reoriented inside it."""
    device = "cpu"
    cfg = Config(dx=0.05, pool_width=0.35, pool_depth=0.15, drop_radius=0.04,
                 drop_center_y=0.25, device=device,
                 surface_tension_coefficient=0.072)
    h = cfg.h_factor * cfg.dx
    xs = torch.arange(0.5 * cfg.dx, cfg.pool_width, cfg.dx, device=device)
    ys = torch.arange(0.5 * cfg.dx, cfg.pool_depth, cfg.dx, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    wall = wall_lattice(cfg, torch.device(device))

    d = torch.cdist(pos, pos)
    iu = torch.triu_indices(pos.shape[0], pos.shape[0], offset=1)
    keep = d[iu[0], iu[1]] < h
    pairs = torch.stack((iu[0][keep], iu[1][keep]), dim=1).contiguous()
    rho = torch.full((pos.shape[0],), cfg.rho0, dtype=pos.dtype, device=device)
    mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx * cfg.dx,
                      dtype=pos.dtype, device=device)

    cfg_ca = Config(dx=cfg.dx, pool_width=cfg.pool_width, pool_depth=cfg.pool_depth,
                    drop_radius=cfg.drop_radius, drop_center_y=cfg.drop_center_y,
                    device=device, surface_tension_coefficient=cfg.surface_tension_coefficient,
                    contact_angle=math.pi / 3)

    acc_none = compute_surface_tension_force(pos, rho, mass, pairs, cfg, wall=wall)
    acc_ca = compute_surface_tension_force(pos, rho, mass, pairs, cfg_ca, wall=wall)

    near = torch.cdist(pos, wall).min(dim=1).values < 2.0 * h
    assert bool(near.any()) and not bool(near.all())
    assert acc_none.abs().max() > 0.0
    assert torch.allclose(acc_ca[~near], acc_none[~near], atol=1e-12)
    assert not torch.allclose(acc_ca[near], acc_none[near], atol=1e-12)