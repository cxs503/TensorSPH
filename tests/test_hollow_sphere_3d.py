"""CASE-HOLLOW-SPHERE-3D: the spherical Young-Laplace anchor.

The invariants asserted here are the ones the case *claims*:

1. the closed form, including both degenerations (drop ``2 sigma/R`` and thin
   film ``4 sigma/R``) -- so the reference cannot drift silently;
2. the geometry: the lattice really is a hollow sphere with a gas cavity, and
   its particle count matches the analytic volume;
3. the 3-D MLS operators on fields with a known answer (a linear field is
   reproduced exactly; the divergence of exact sphere normals is ``-2/R``) --
   these isolate operator error from *field* error, which is what makes the
   curvature number in the report interpretable;
4. the case gate itself: the projection route with imposed curvature reproduces
   ``2 sigma (1/R_in + 1/R_out)`` inside the platform tolerance, and ``sigma = 0``
   carries no pressure;
5. the honest record of the self-consistent route: its error at the test
   resolution is committed here so a future change cannot quietly regress it.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.threed import hollow_sphere as hs
from tensorsph.threed.neighbors3 import box_pairs

SIGMA = 0.072
RHO0 = 1000.0


# --------------------------------------------------------------------------- #
# 1. reference
# --------------------------------------------------------------------------- #
def test_laplace_closed_form_and_its_degenerations():
    # shell: two surfaces, 2 sigma (1/R_in + 1/R_out)
    assert hs.shell_laplace_analytic(SIGMA, 0.006, 0.014) == pytest.approx(
        2.0 * SIGMA * (1.0 / 0.006 + 1.0 / 0.014))
    # drop: R_in -> 0 leaves one surface, 2 sigma / R  (NOT sigma / R)
    assert hs.shell_laplace_analytic(SIGMA, 0.0, 0.01) == pytest.approx(2.0 * SIGMA / 0.01)
    # thin film: R_in -> R_out -> 4 sigma / R (the soap bubble)
    film = hs.shell_laplace_analytic(SIGMA, 0.01, 0.01 * (1.0 + 1e-6))
    assert film == pytest.approx(4.0 * SIGMA / 0.01, rel=1e-5)
    # the 2-D anchor value is exactly half -- this is the dimension check
    assert hs.shell_laplace_analytic(SIGMA, 0.0, 0.01) == pytest.approx(
        2.0 * (SIGMA / 0.01))


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError):
        hs.shell_laplace_analytic(-1.0, 0.0, 0.01)
    with pytest.raises(ValueError):
        hs.sphere_shell_lattice(0.014, 0.010, 0.001, 0.004)   # R_in > R_out
    with pytest.raises(ValueError):
        hs.sphere_shell_lattice(0.0, 0.010, 0.001, 0.0)       # no Dirichlet pad


# --------------------------------------------------------------------------- #
# 2. geometry
# --------------------------------------------------------------------------- #
def test_lattice_is_a_hollow_sphere():
    dx, r_in, r_out, pad = 1.0, 6.0, 14.0, 4.0
    pos, mass, liquid, r, half = hs.sphere_shell_lattice(r_in, r_out, dx, pad)
    assert pos.shape[1] == 3
    assert bool((r <= half + 1e-12).all())
    assert half == pytest.approx(r_out + pad)
    # every site is classified exactly once, and the cavity is empty
    assert bool((r[liquid] >= r_in - 1e-12).all())
    assert bool((r[liquid] <= r_out + 1e-12).all())
    assert not bool(liquid[r < r_in - 1e-12].any())
    # the liquid shell volume is right to the lattice quantisation (a few %)
    v_shell = 4.0 / 3.0 * math.pi * (r_out ** 3 - r_in ** 3)
    v_lattice = float(liquid.sum()) * dx ** 3
    assert abs(v_lattice / v_shell - 1.0) < 0.06
    assert mass.shape == liquid.shape and bool((mass > 0).all())


# --------------------------------------------------------------------------- #
# 3. operators with known answers
# --------------------------------------------------------------------------- #
def _cloud(r_in=0.0, r_out=4.0, pad=3.0, h=1.3):
    pos, mass, liquid, r, half = hs.sphere_shell_lattice(r_in, r_out, 1.0, pad)
    vol = mass / RHO0
    pairs = box_pairs(pos, 2.0 * h)
    m, delta, weight, grad, i, j = hs._mls_ops(pos, vol, pairs, h)
    m_inv, ok = hs._regularised_inverse(m)
    return pos, liquid, r, vol, pairs, m_inv, ok, grad, i, j


def test_mls_gradient_is_exact_on_a_linear_field():
    pos, *_rest = _cloud()
    vol, pairs, m_inv, ok, grad, i, j = _rest[2], _rest[3], _rest[4], _rest[5], _rest[6], _rest[7], _rest[8]
    for axis in range(3):
        g = hs._mls_gradient(m_inv, vol, grad, i, j, pos[:, axis], pos.shape[0])
        inner = torch.linalg.vector_norm(pos, dim=1) < 2.0
        expected = torch.zeros(3, dtype=pos.dtype)
        expected[axis] = 1.0
        assert torch.allclose(g[inner].mean(dim=0), expected, atol=1e-9)


def test_mls_divergence_recovers_the_sphere_curvature_of_exact_normals():
    pos, _liq, r, vol, pairs, m_inv, ok, grad, i, j = _cloud()
    rhat = pos / r.clamp_min(1e-30).unsqueeze(-1)
    # droplet convention: colour c = 1 inside, so n = -grad c / |grad c| = -rhat
    kappa = hs._mls_divergence(m_inv, vol, grad, i, j, -rhat, pos.shape[0], 3)
    band = (r > 3.5) & (r < 4.5)
    measured = float(kappa[band].mean())
    assert abs(measured / (-2.0 / 4.0) - 1.0) < 0.02      # <2 % on exact normals


# --------------------------------------------------------------------------- #
# 4. the case gate
# --------------------------------------------------------------------------- #
def test_projection_route_meets_the_laplace_gate_on_a_drop():
    row = hs.hollow_sphere_case(radius_in=0.0, radius_out=6.0, dx=1.0,
                                sigma=SIGMA, rho0=RHO0, curvature="analytic",
                                pad=4.0)
    assert row["dp_theory_Pa"] == pytest.approx(2.0 * SIGMA / 6.0)
    assert abs(row["dp_error_pct"]) < 5.0, row["dp_error_pct"]
    assert abs(row["delta_s_integral_dn"] - 1.0) < 0.05
    # the independent traction-flux route agrees with the Poisson route
    assert abs(row["dp_flux_over_theory"] - 1.0) < 0.1


def test_projection_route_meets_the_gate_on_a_hollow_sphere():
    """Two curvature radii, both verified: the increment over the 2-D anchor."""
    row = hs.hollow_sphere_case(radius_in=6.0, radius_out=14.0, dx=1.0,
                                sigma=SIGMA, rho0=RHO0, curvature="analytic",
                                pad=4.0)
    assert row["dp_theory_Pa"] == pytest.approx(2.0 * SIGMA * (1.0 / 6.0 + 1.0 / 14.0))
    assert abs(row["dp_error_pct"]) < 5.0, row["dp_error_pct"]
    # each face carries its own 2 sigma / R_jump
    assert abs(row["dp_flux_inner_over_theory"] - 1.0) < 0.08
    assert abs(row["dp_flux_outer_over_theory"] - 1.0) < 0.05
    # a hollow sphere must not look like a drop: the two differ by 2 sigma/R_in
    assert row["dp_theory_Pa"] > 1.5 * 2.0 * SIGMA / 14.0


def test_negative_control_sigma_zero_carries_no_pressure():
    row = hs.hollow_sphere_case(radius_in=0.0, radius_out=4.0, dx=1.0,
                                sigma=0.0, rho0=RHO0, curvature="analytic", pad=3.0)
    assert abs(row["dp_measured_Pa"]) < 1e-10


# --------------------------------------------------------------------------- #
# 5. honest record of the self-consistent route
# --------------------------------------------------------------------------- #
def test_self_consistent_curvature_error_is_recorded_not_hidden():
    """The shipped div-n curvature route, measured at the test resolution.

    At ``R/dx = 6`` (drop) the committed ladder records a small positive bias;
    the band below is wide enough to absorb kernel/threshold changes but not a
    silent 2x regression (a ``sigma/R`` instead of ``2 sigma/R`` bug would land
    at +100 %).
    """
    row = hs.hollow_sphere_case(radius_in=0.0, radius_out=6.0, dx=1.0,
                                sigma=SIGMA, rho0=RHO0, curvature="mls", pad=4.0)
    assert -15.0 < row["dp_error_pct"] < 25.0, row["dp_error_pct"]
    # curvature itself is the sharper diagnostic than the jump
    assert abs(row["kappa_outer_error_pct"]) < 10.0, row["kappa_outer_error_pct"]