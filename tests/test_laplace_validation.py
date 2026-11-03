"""Analytic verification of the CSF surface-tension model (Laplace law).

The m277 surface-tension commit shipped unit tests only: shapes, zero-``sigma``
short circuits and unit normals. None of them compares the model with an
analytic surface-tension result, which is why a sign error in the colour
gradient survived review -- it made the gradient of every *smooth* field cancel
to zero, the CSF force physically inert, while all ten existing tests still
passed (a uniform colour field has a zero gradient either way).

What is pinned here
-------------------
1. ``test_color_gradient_linear_field_consistency`` -- an exact analytic
   gradient (``c = a x`` -> ``grad c = (a, 0)``). Fails on the sign error.
2. ``test_curvature_of_circular_interface`` -- ``kappa = 1/R`` on a full
   lattice patch, i.e. the Laplace operator itself.
3. ``test_interface_delta_integrates_to_colour_jump`` -- ``integral |grad c| dr
   = 1`` across the interface, which is the factor the pressure jump is linear
   in.
4. ``test_laplace_pressure_jump_from_operator`` -- ``Delta p = sigma kappa
   integral |grad c| dr`` reproduces ``sigma / R``.
5. ``test_free_surface_droplet_laplace_deficit`` (xfail, strict) -- the shipped
   *free-surface* force routine under-predicts the jump by a factor ~5 because
   the interface has no exterior particles: the delta function only integrates
   to ~0.25 of the colour jump. When somebody fixes that, this test XPASSes and
   the strict marker turns the suite red on purpose.
"""
import math

import pytest
import torch

from tensorsph import (
    Config,
    compute_color_gradient,
    compute_curvature,
    compute_normal,
    density_from_neighbor_pairs,
)
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import compute_surface_tension_force

DX = 2.5e-3
RADIUS = 4.0e-2
RHO0 = 1000.0
H_FACTOR = 1.3
SIGMA = 0.072


def _patch(half_ratio=1.4, dx=DX, dtype=torch.float64):
    """Full lattice patch spanning both sides of the interface."""
    half = half_ratio * RADIUS
    axis = torch.arange(-half, half + dx, dx, dtype=dtype)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), RHO0 * dx * dx, dtype=dtype)
    density = torch.full((pos.shape[0],), RHO0, dtype=dtype)
    return pos, mass, density, dx * dx


def _disk(dx=DX, dtype=torch.float64):
    """Hexagonal fluid lattice disk: the configuration the solver actually has."""
    dy = dx * math.sqrt(3.0) / 2.0
    radius = RADIUS
    rows = int(math.floor(radius / dy)) + 1
    cols = int(math.floor(radius / dx)) + 1
    chunks = []
    for k in range(-rows, rows + 1):
        x0 = 0.5 * dx if (k + rows) % 2 else 0.0
        xs = torch.arange(-cols, cols + 1, dtype=dtype) * dx + x0
        chunks.append(torch.stack((xs, torch.full_like(xs, k * dy)), dim=-1))
    pos = torch.cat(chunks, dim=0)
    pos = pos[torch.linalg.vector_norm(pos, dim=-1) <= radius]
    return pos - pos.mean(dim=0), dx * dy


def _cfg(sigma=SIGMA):
    return Config(dx=DX, h_factor=H_FACTOR, rho0=RHO0, gravity=0.0,
                  surface_tension_coefficient=sigma, fail_fast=False)


def _operator_laplace(w_over_dx=2.0):
    """kappa, integral |grad c| dr and the implied Delta p on a full patch."""
    pos, mass, density, area = _patch()
    h = H_FACTOR * DX
    r = torch.linalg.vector_norm(pos, dim=1)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    color = 0.5 * (1.0 - torch.tanh((r - RADIUS) / (w_over_dx * DX)))
    grad_c = compute_color_gradient(pos, color, mass, density, pairs, h)
    normal = compute_normal(-grad_c)
    kappa = compute_curvature(pos, normal, mass, density, pairs, h)
    delta_s = torch.linalg.vector_norm(grad_c, dim=1)
    ring = delta_s > 0.25 * float(delta_s.max())
    kappa_mean = float(kappa[ring].mean())
    int_delta = float((delta_s * area
                       / (2.0 * math.pi * r.clamp_min(DX))).sum())
    return kappa_mean, int_delta, SIGMA * kappa_mean * int_delta


def test_color_gradient_linear_field_consistency():
    """grad(c) of c = a*x must return (a, 0); the old sign error gave ~0."""
    pos, mass, density, area = _patch()
    h = H_FACTOR * DX
    pairs = cell_linked_pairs(pos, 2.0 * h)
    a = 3.0
    grad = compute_color_gradient(pos, a * pos[:, 0], mass, density, pairs, h)
    interior = torch.linalg.vector_norm(pos, dim=1) < 0.8 * RADIUS
    mean = grad[interior].mean(dim=0)
    assert float(mean[0]) == pytest.approx(a, rel=0.02), (
        f"linear-field gradient {float(mean[0]):.4f} vs analytic {a}")
    assert abs(float(mean[1])) < 0.02 * a


def test_curvature_of_circular_interface():
    """kappa of a smooth circular interface must equal 1/R."""
    kappa_mean, _, _ = _operator_laplace()
    expected = 1.0 / RADIUS
    assert kappa_mean == pytest.approx(expected, rel=0.05), (
        f"kappa {kappa_mean:.4f} 1/m vs analytic {expected:.4f} 1/m")


def test_interface_delta_integrates_to_colour_jump():
    """integral |grad c| dr across the interface must equal the colour jump = 1."""
    _, int_delta, _ = _operator_laplace()
    assert int_delta == pytest.approx(1.0, rel=0.05), (
        f"integral |grad c| dr = {int_delta:.4f}, expected 1")


def test_laplace_pressure_jump_from_operator():
    """sigma * kappa * integral |grad c| dr must reproduce sigma / R."""
    _, _, delta_p = _operator_laplace()
    expected = SIGMA / RADIUS
    assert delta_p == pytest.approx(expected, rel=0.05), (
        f"Laplace jump {delta_p:.4f} Pa vs analytic {expected:.4f} Pa")


def test_surface_tension_force_points_inward():
    """On a static droplet the CSF acceleration must point towards the centre."""
    pos, area = _disk()
    h = H_FACTOR * DX
    mass = torch.full((pos.shape[0],), RHO0 * area, dtype=torch.float64)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    r = torch.linalg.vector_norm(pos, dim=1)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    acc = compute_surface_tension_force(pos, rho, mass, pairs, _cfg())
    r_hat = pos / r[:, None]
    radial = (acc * r_hat).sum(dim=1)
    surface = r > 0.9 * RADIUS
    assert bool(surface.any())
    assert float(radial[surface].mean()) < 0.0, "force points outward"


def test_laplace_jump_vanishes_without_surface_tension():
    """sigma = 0 must give an exactly zero force (no spurious contribution)."""
    pos, area = _disk()
    h = H_FACTOR * DX
    mass = torch.full((pos.shape[0],), RHO0 * area, dtype=torch.float64)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    acc = compute_surface_tension_force(pos, rho, mass, pairs, _cfg(sigma=0.0))
    assert float(torch.linalg.vector_norm(acc, dim=1).max()) == 0.0


def test_laplace_jump_scales_linearly_with_sigma():
    """Delta p is linear in sigma: no hidden sigma dependence in the routine."""
    pos, area = _disk()
    h = H_FACTOR * DX
    mass = torch.full((pos.shape[0],), RHO0 * area, dtype=torch.float64)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    r_hat = pos / torch.linalg.vector_norm(pos, dim=1, keepdim=True)
    jumps = []
    for sigma in (0.036, 0.072):
        acc = compute_surface_tension_force(pos, rho, mass, pairs,
                                            _cfg(sigma=sigma))
        jumps.append(-float((rho * (acc * r_hat).sum(dim=1)).sum() * area))
    assert jumps[0] != 0.0
    assert jumps[1] / jumps[0] == pytest.approx(2.0, rel=1e-6)


@pytest.mark.xfail(strict=True, reason=(
    "known defect: on a free-surface droplet the interface has no exterior "
    "particles, so |grad c| integrates to ~0.25 of the colour jump and the "
    "Laplace jump comes out ~5x too small. Needs a normalised delta function "
    "or a two-phase colour field; see docs/benchmarks/laplace.md"))
def test_free_surface_droplet_laplace_deficit():
    """Shipped free-surface force routine must reproduce sigma / R within 15%."""
    pos, area = _disk()
    h = H_FACTOR * DX
    mass = torch.full((pos.shape[0],), RHO0 * area, dtype=torch.float64)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    acc = compute_surface_tension_force(pos, rho, mass, pairs, _cfg())
    r_hat = pos / torch.linalg.vector_norm(pos, dim=1, keepdim=True)
    delta_p = -float((rho * (acc * r_hat).sum(dim=1)).sum() * area)
    r_eq = math.sqrt(pos.shape[0] * area / math.pi)
    assert delta_p == pytest.approx(SIGMA / r_eq, rel=0.15)
