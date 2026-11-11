"""CSF operator-level guards used by the dispersion verification.

These are the unit-level facts the campaign in
``tests/test_surface_tension_dispersion.py`` and
``docs/verification_surface_tension_dispersion.md`` builds on:

* the exact reference formulas (wave stress coefficient, Lamb curvature) and the
  modal-projection algebra (a synthetic ``A cos(kx)`` stress is recovered to
  machine precision);
* the surface-delta normalisers (``legacy`` is bit-preserved, ``local_range``
  improves the closed drop, an unknown name is rejected);
* the bulk is a bit-exact no-op.

The physics *verdict* (which gates pass, and the curvature/normal audits) lives in
``tests/test_surface_tension_dispersion.py``; this file deliberately stays at the
operator level so a change of geometry does not churn it.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.csf_consistent import colour_range, surface_tension_acceleration
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension_dispersion import (
    AMPLITUDE_FRACTION, DROP_EPS, RHO0, SIGMA, csf_state, delta_measure_audit,
    drop_geometry, drop_modal_stiffness, exact_wave_stress_coeff, hex_lattice,
    wave_geometry,
)


# --------------------------------------------------------------------------- #
# exact references + projection algebra
# --------------------------------------------------------------------------- #
def test_exact_wave_stress_coeff_reduces_to_linear_theory():
    k, amp = 2.0, 1e-7
    assert exact_wave_stress_coeff(k, amp) == pytest.approx(amp * k * k, rel=1e-9)


def test_exact_wave_stress_coeff_is_the_second_derivative():
    """For ``eta = -a cos(kx)`` the mode coefficient is ``a k^2`` (large amplitude too)."""
    k, amp = 1.0, 0.2
    ref = exact_wave_stress_coeff(k, amp)
    # the arclength factor ds/dx = sqrt(1 + eta'^2) makes the finite-amplitude
    # reference slightly *smaller* than the linear a k^2 (second order in a k)
    assert 0.95 <= ref / (amp * k * k) <= 1.001


def test_modal_projection_recovers_a_synthetic_stress():
    """``2/L sum A cos(kx) V / (rho0 V)`` must return ``A`` to machine precision."""
    k, amp, n = 3.0, 0.37, 480
    length = 2.0 * math.pi / k
    x = torch.arange(n, dtype=torch.float64) * (length / n)
    dx = length / n
    stress = amp * torch.cos(k * x)                    # the modal stress field
    dp_k = 2.0 / length * (stress * torch.cos(k * x) * dx).sum()
    assert float(dp_k) == pytest.approx(amp, rel=1e-9)


def test_lamb_reference_coefficient_is_n2m1_eps_over_R():
    """``kappa_n = (n^2-1) eps / R`` [1/m] -- the *curvature*, not divided by R again."""
    mode, eps, radius = 3, 0.05, 0.02
    geom = drop_geometry(mode, eps, radius, radius / 16)
    row = drop_modal_stiffness(geom, SIGMA, periodic_x=False)
    assert row["dp_analytic"] == pytest.approx(-SIGMA * (mode ** 2 - 1) * eps / radius)


# --------------------------------------------------------------------------- #
# surface-delta normalisers
# --------------------------------------------------------------------------- #
def test_colour_range_is_one_without_pairs_and_half_on_a_free_surface():
    g = wave_geometry(1.0, 1e-9, 0.02)                  # flat limit
    state, _, _ = csf_state(g, SIGMA)
    assert float(colour_range(state.color, torch.empty((0, 2), dtype=torch.long)).max()) == 1.0
    h = 1.3 * g["dx"]
    pairs = cell_linked_pairs(g["pos"], 2.0 * h)
    rng = colour_range(state.color, pairs)
    shell = state.delta_s > 0.2 * state.delta_s.max()
    # the void half of the 0 -> 1 transition has no particles, so the measured
    # neighbourhood range never spans the full transition on a one-sided free
    # surface (measured 0.32 at h = 1.3 dx; the exact value is lattice dependent)
    assert 0.2 < float(rng[shell].max()) < 0.65


def test_legacy_delta_function_is_bit_preserved():
    """``delta_normalization='legacy'`` must reproduce ``2|grad c|`` exactly."""
    g = wave_geometry(2.0, AMPLITUDE_FRACTION * math.pi, 0.02)
    state, _, _ = csf_state(g, SIGMA)
    mag = torch.linalg.vector_norm(state.grad_c, dim=1)
    shell = mag > 1e-3 * mag.max()
    assert torch.allclose(state.delta_s[shell], 2.0 * mag[shell], atol=0, rtol=1e-14)


def test_local_range_normalisation_improves_the_drop_measure():
    table = delta_measure_audit(ppw=16, radius_over_dx=16, shifts=2)
    legacy = table["drop_l2"]["legacy"]
    fixed = table["drop_l2"]["local_range"]
    assert abs(fixed - 1.0) < abs(legacy - 1.0)
    assert fixed > 0.85


def test_unknown_delta_normalisation_is_rejected():
    g = drop_geometry(2, DROP_EPS, 0.02, 0.02 / 8)
    with pytest.raises(ValueError):
        csf_state(g, SIGMA, delta_normalization="colour_range_typo", periodic_x=False)


def test_bulk_particles_carry_no_surface_tension():
    """Deep interior particles must get exactly zero CSF force."""
    g = drop_geometry(2, DROP_EPS, 0.02, 0.02 / 12)
    h = 1.3 * g["dx"]
    acc, _ = surface_tension_acceleration(
        g["pos"], torch.full((g["pos"].shape[0],), RHO0, dtype=torch.float64),
        g["mass"], cell_linked_pairs(g["pos"], 2.0 * h), h, SIGMA,
        rho0=RHO0, curvature_scheme="fit_quadratic", delta_normalization="local_range")
    bulk = torch.linalg.vector_norm(g["pos"], dim=1) < 0.5 * g["radius"]
    assert float(acc[bulk].abs().max()) == 0.0


def test_hex_lattice_has_no_square_rows():
    """Regression: ``torch.round`` half-to-even used to align every other row."""
    dx = 0.1
    pts = hex_lattice(dx, 0.0, 1.0, 0.0, 1.0)
    d = torch.cdist(pts, pts)
    d.fill_diagonal_(float("inf"))
    nearest = d.min(dim=1).values
    assert float(nearest.min()) == pytest.approx(dx, rel=1e-12)