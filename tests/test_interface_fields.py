"""Invariants of the CSF interface fields and the contact-angle boundary.

These tests pin the operators that replaced the neighbour-count heuristic of
0.8.0 (see ``docs/benchmarks/surface-tension-laplace.md``). They are cheap: one
planar interface, one small disc, no time integration.
"""
import math

import torch

from tensorsph import Config
from tensorsph.droplet_impact import wall_lattice
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import (
    apply_contact_angle_boundary,
    interface_fields,
    compute_surface_tension_force,
)

RHO0 = 1000.0
SIGMA = 0.072


def _cfg(dx, **kwargs):
    return Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=20.0,
                  surface_tension_coefficient=SIGMA, **kwargs)


def _half_plane(dx, hf=2.0):
    ys = (torch.arange(120, dtype=torch.float64) + 0.5) * dx - 120 * dx
    xs = (torch.arange(120, dtype=torch.float64) + 0.5) * dx - 60 * dx
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack([X.flatten(), Y.flatten()], dim=1)
    pos = pos[pos[:, 1] <= 0.0].contiguous()
    n = pos.shape[0]
    cfg = _cfg(dx, surface_tension_h_factor=hf)
    mass = torch.full((n,), RHO0 * dx * dx, dtype=torch.float64)
    density = torch.full((n,), RHO0, dtype=torch.float64)
    pairs = cell_linked_pairs(pos, 2.0 * cfg.h_factor * dx)
    return pos, mass, density, pairs, cfg


def _disc(radius, dx):
    n = int(round(2.0 * radius / dx))
    xs = (torch.arange(n, dtype=torch.float64) + 0.5) * dx - radius
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    pos = torch.stack([X.flatten(), Y.flatten()], dim=1)
    pos = pos[torch.linalg.vector_norm(pos, dim=1) <= radius].contiguous()
    m = pos.shape[0]
    cfg = _cfg(dx)
    mass = torch.full((m,), RHO0 * dx * dx, dtype=torch.float64)
    density = torch.full((m,), RHO0, dtype=torch.float64)
    pairs = cell_linked_pairs(pos, 2.0 * cfg.h_factor * dx)
    return pos, mass, density, pairs, cfg


def test_planar_normal_points_out_of_the_fluid():
    """On a half plane y<=0 the outward normal must be +y, i.e. -grad c."""
    pos, mass, density, pairs, cfg = _half_plane(0.0025)
    f = interface_fields(pos, density, mass, pairs, cfg)
    # interface shell only: away from the lateral ends, where the finite patch
    # has its own boundary, and only where the delta function is significant
    h_c = cfg.surface_tension_h_factor * cfg.h_factor * cfg.dx
    surface = f["delta_s"] > 0.5 * f["delta_s"].max()
    surface &= pos[:, 0].abs() < 0.4 * float(pos[:, 0].abs().max())
    surface &= pos[:, 1] > -2.0 * h_c          # the interface shell itself
    n_y = f["normal"][surface][:, 1]
    assert float(n_y.mean()) > 0.98
    # and the tangential component averages out
    assert abs(float(f["normal"][surface][:, 0].mean())) < 0.02


def test_flat_interface_has_zero_curvature():
    """A plane is not curved: median kappa must be small compared with 1/h_c."""
    pos, mass, density, pairs, cfg = _half_plane(0.0025)
    f = interface_fields(pos, density, mass, pairs, cfg)
    surface = f["delta_s"] > 0.5 * f["delta_s"].max()
    surface &= pos[:, 0].abs() < 0.4 * float(pos[:, 0].abs().max())
    h_c = cfg.surface_tension_h_factor * cfg.h_factor * cfg.dx
    assert abs(float(f["curvature"][surface].median())) * h_c < 0.25


def test_disc_curvature_has_the_right_sign_and_scale():
    """kappa * R = +1 for a disc of radius R (within 25% at R/dx = 32)."""
    radius, dx = 0.1, 0.1 / 32
    pos, mass, density, pairs, cfg = _disc(radius, dx)
    f = interface_fields(pos, density, mass, pairs, cfg)
    surface = f["delta_s"] > 0.1 * f["delta_s"].max()
    assert float(f["curvature"][surface].median()) * radius == \
        pytest.approx(1.0, abs=0.25)


def test_force_pulls_a_droplet_inward_everywhere():
    """The CSF force on a disc must point inward for every interface particle."""
    radius, dx = 0.1, 0.1 / 16
    pos, mass, density, pairs, cfg = _disc(radius, dx)
    acc = compute_surface_tension_force(pos, density, mass, pairs, cfg)
    f = interface_fields(pos, density, mass, pairs, cfg)
    surface = f["delta_s"] > 0.1 * f["delta_s"].max()
    n_out = pos / torch.linalg.vector_norm(pos, dim=1, keepdim=True)
    inward = -(acc * n_out).sum(dim=1)
    assert float((inward[surface] > 0).double().mean()) == 1.0


def test_zero_sigma_gives_zero_force():
    pos, mass, density, pairs, _ = _disc(0.05, 0.005)
    off = Config(dx=0.005, h_factor=1.3, rho0=RHO0, sound_speed=20.0,
                 surface_tension_coefficient=0.0)
    acc = compute_surface_tension_force(pos, density, mass, pairs, off)
    assert torch.allclose(acc, torch.zeros_like(acc))


def _bottom_wall():
    cfg = Config(dx=0.05, pool_width=0.4, pool_depth=0.15, drop_radius=0.05,
                 drop_center_y=0.25)
    wall = wall_lattice(cfg, torch.device("cpu")).to(torch.float64)
    return wall, float(wall[:, 1].min()), 0.05


def test_contact_angle_ninety_degrees_makes_the_normal_tangential():
    """theta = pi/2: the wall-normal component of the normal is removed."""
    wall, y_bottom, dx = _bottom_wall()
    pos = torch.tensor([[0.1, y_bottom + 0.5 * dx]], dtype=torch.float64)
    grad = torch.tensor([[0.6, 0.8]], dtype=torch.float64)   # normal (-0.6,-0.8)
    out = apply_contact_angle_boundary(grad, pos, wall, 2.0 * dx, math.pi / 2)
    assert abs(float(out[0, 1])) < 0.25 * abs(float(grad[0, 1]))


def test_contact_angle_wetting_and_drying_rotate_in_opposite_senses():
    """theta < pi/2 and theta > pi/2 push the wall-normal component apart."""
    wall, y_bottom, dx = _bottom_wall()
    pos = torch.tensor([[0.1, y_bottom + 0.5 * dx]], dtype=torch.float64)
    grad = torch.tensor([[0.6, 0.8]], dtype=torch.float64)   # normal (-0.6,-0.8)
    wet = apply_contact_angle_boundary(grad, pos, wall, 2.0 * dx, math.pi / 6)
    dry = apply_contact_angle_boundary(grad, pos, wall, 2.0 * dx, 5.0 * math.pi / 6)
    assert float(wet[0, 1]) > float(grad[0, 1])      # wetting: normal turns down
    assert float(dry[0, 1]) < float(grad[0, 1])      # drying: normal turns up


def test_contact_angle_preserves_magnitude_and_ignores_far_particles():
    wall, y_bottom, dx = _bottom_wall()
    pos = torch.tensor([[0.1, y_bottom + 0.5 * dx],
                        [0.1, y_bottom + 20.0 * dx]], dtype=torch.float64)
    grad = torch.tensor([[0.3, 0.4], [0.3, 0.4]], dtype=torch.float64)
    out = apply_contact_angle_boundary(grad.clone(), pos, wall, 2.0 * dx, math.pi / 3)
    assert torch.allclose(out[1], grad[1])                       # far: untouched
    assert torch.isclose(torch.linalg.vector_norm(out[0]),
                         torch.linalg.vector_norm(grad[0]), atol=1e-6)
    assert not torch.allclose(out[0], grad[0])                   # near: rotated


import pytest  # noqa: E402  (kept last so the module reads top-down)