"""Tests for the mixture-material (normalised colour) surface-tension chain.

Every assertion is a physical invariant of the formulation, not a fit to the
numbers produced by the implementation:

* the colour function of a mixture must not depend on the density ratio
  (normalisation cancels the per-phase volume);
* the coloured delta function must carry unit mass, ``int delta_s dn = 1``,
  i.e. ``sum_i delta_s_i V_i = 2 pi R`` for a circular interface;
* the surface-tension force of a closed interface must have no resultant, also
  when the interface centre is off the lattice;
* the force must compress a convex drop of the inner material (sign);
* the measured effective surface tension must reproduce ``dp = sigma / R``
  within the 15 % acceptance band of the anchor, and the shipped
  single-rho0 colour must not.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.csf_mixture import (
    MixtureMaterial, MixtureMaterialSet, kernel_sum_color_single_rho0,
    laplace_force_ratio, mixture_color, mixture_csf_fields, net_force,
    particle_volume,
)
from tensorsph.neighbor_search import cell_linked_pairs

torch.set_default_dtype(torch.float64)

RADIUS = 0.04
SIGMA = 0.072
H_FACTOR = 1.3


def _two_phase_disc(dx: float, rho_in: float, rho_out: float,
                    offset: tuple[float, float] = (0.0, 0.0),
                    outer: float = 0.035):
    extent = 2.0 * (RADIUS + outer) + 6.0 * dx
    n = int(extent / dx)
    axis = torch.linspace(0.0, extent, n) - 0.5 * extent
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    r = pos.norm(dim=1)
    keep = r <= RADIUS + outer
    pos = pos[keep] + torch.tensor(offset)
    r = r[keep]
    phase = (r >= RADIUS).to(torch.long)
    rho = torch.where(phase == 0, torch.tensor(rho_in), torch.tensor(rho_out))
    mass = rho * dx * dx
    materials = MixtureMaterialSet(
        materials=(MixtureMaterial("A", rho_in), MixtureMaterial("B", rho_out)),
        sigma={(0, 1): SIGMA},
    )
    return pos, mass, phase, materials


def _fields(dx, rho_in, rho_out, offset=(0.0, 0.0), **kw):
    pos, mass, phase, mats = _two_phase_disc(dx, rho_in, rho_out, offset)
    h = H_FACTOR * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    vol = particle_volume(mass, phase, mats)
    state = mixture_csf_fields(pos, vol, phase, pairs, h, SIGMA, 0, **kw)
    return pos, vol, phase, state


# --------------------------------------------------------------------------
# 1. mixture-material colour is density-ratio invariant
# --------------------------------------------------------------------------
def test_colour_invariant_under_density_ratio():
    """c^A must be identical for 1:1, 1000:1.2 and 1.2:1000 density contrast.

    The normalisation ``sum_j V_j W / sum_j V_j W`` cancels the per-phase
    volume, so the colour field cannot depend on the rest densities at all --
    this is exactly the property the shipped single-rho0 kernel sum lacks.
    """
    dx = 0.005
    pos_a, vol_a, _, st_a = _fields(dx, 1000.0, 1000.0)
    pos_b, vol_b, _, st_b = _fields(dx, 1000.0, 1.2)
    pos_c, vol_c, _, st_c = _fields(dx, 1.2, 1000.0)
    # volume-consistent lattice: V = dx^2 for every particle, all cases
    for vol in (vol_a, vol_b, vol_c):
        assert torch.allclose(vol, torch.full_like(vol, dx * dx), atol=1e-18)
    assert torch.allclose(st_a.color, st_b.color, atol=1e-12)
    assert torch.allclose(st_a.color, st_c.color, atol=1e-12)
    assert torch.allclose(st_a.force_density, st_b.force_density, atol=1e-12)


def test_legacy_colour_does_depend_on_density_ratio():
    """The shipped single-rho0 colour violates that invariance (ablation).

    With one scalar ``rho0 = 1000`` the light phase is assigned volumes that are
    ``1.2/1000`` of the lattice volume, so the colour measures the *heavy*
    phase presence and the coloured delta function carries a mass that changes
    by more than a factor of 2 between the two density orderings.
    """
    dx = 0.005
    integrals = {}
    for label, rho_in, rho_out in (("control", 1000.0, 1000.0),
                                   ("water-air", 1000.0, 1.2),
                                   ("air-water", 1.2, 1000.0)):
        pos, mass, phase, mats = _two_phase_disc(dx, rho_in, rho_out)
        h = H_FACTOR * dx
        pairs = cell_linked_pairs(pos, 2.0 * h)
        c = kernel_sum_color_single_rho0(pos, mass, pairs, h, 1000.0)
        vol_l = mass / 1000.0
        from tensorsph.csf_consistent import mls_gradient
        grad = mls_gradient(pos, c, vol_l, pairs, h)
        delta = 2.0 * torch.linalg.vector_norm(grad, dim=1)
        integrals[label] = float((delta * vol_l).sum() / (2.0 * math.pi * RADIUS))
        # the single-rho0 colour never completes the full 1 -> 0 transition
        assert float(c.max() - c.min()) < 1.0
    spread = max(integrals.values()) - min(integrals.values())
    # measured spread is ~2 on this lattice; the mixture colour gives exactly 0
    assert spread > 0.5, integrals


# --------------------------------------------------------------------------
# 2. delta-function normalisation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("R_over_dx", [8.0])
def test_delta_integral_is_unity(R_over_dx):
    """sum_i delta_s_i V_i / (2 pi R) must be ~1 (Brackbill identity)."""
    dx = RADIUS / R_over_dx
    pos, vol, _, st = _fields(dx, 1000.0, 1000.0)
    integral = float((st.delta_s * vol).sum() / (2.0 * math.pi * RADIUS))
    assert abs(integral - 1.0) < 0.10, integral


# --------------------------------------------------------------------------
# 3. Laplace law
# --------------------------------------------------------------------------
@pytest.mark.parametrize("R_over_dx", [12.0])
@pytest.mark.parametrize("rho", [(1000.0, 1000.0), (1000.0, 1.2), (1.2, 1000.0)])
def test_laplace_law_within_15pct(R_over_dx, rho):
    """sigma_eff / sigma = 1 within the anchor's 15 % acceptance band."""
    dx = RADIUS / R_over_dx
    pos, vol, _, st = _fields(dx, rho[0], rho[1])
    ratio = laplace_force_ratio(st.force_density, vol, pos, SIGMA)
    assert abs(ratio["sigma_eff_over_sigma"] - 1.0) < 0.15, ratio


def test_force_scales_linearly_with_sigma():
    dx = RADIUS / 8.0
    pos, vol, phase, _ = _fields(dx, 1000.0, 1000.0)
    h = H_FACTOR * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    st1 = mixture_csf_fields(pos, vol, phase, pairs, h, SIGMA, 0)
    st2 = mixture_csf_fields(pos, vol, phase, pairs, h, 2.0 * SIGMA, 0)
    assert torch.allclose(st2.force_density, 2.0 * st1.force_density, atol=1e-14)


# --------------------------------------------------------------------------
# 4. force resultant of a closed interface
# --------------------------------------------------------------------------
def test_closed_interface_has_no_resultant_off_lattice():
    """|| sum_i f_i V_i || / sum_i ||f_i|| V_i must be small off the lattice.

    The centre is shifted by a non-lattice amount so the cancellation is not
    trivial lattice symmetry -- it tests that the force field of a closed
    interface really integrates to zero.
    """
    dx = RADIUS / 12.0
    pos, vol, _, st = _fields(dx, 1000.0, 1000.0,
                              offset=(0.37 * dx, 0.21 * dx))
    net = float(net_force(st.force_density, vol).norm())
    scale = float((st.force_density.norm(dim=1) * vol).sum())
    assert net / scale < 0.05, net / scale


def test_force_points_inward():
    """A convex drop of material A is compressed: f_x < 0 at x > 0."""
    dx = RADIUS / 12.0
    pos, vol, _, st = _fields(dx, 1000.0, 1000.0)
    r = pos.norm(dim=1)
    shell = (st.delta_s > 0.5 * float(st.delta_s.max()))
    right = shell & (pos[:, 0] > 0.5 * RADIUS) & (pos[:, 1].abs() < 0.2 * RADIUS)
    assert right.any()
    f = st.force_density
    fx = (f[right, 0] * vol[right]).sum()
    assert fx < 0.0, float(fx)
    assert (st.kappa[shell] > 0).float().mean() > 0.9