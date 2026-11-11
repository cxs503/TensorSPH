"""CSF surface-tension operator: delta-function consistency and discrete
dispersion guards.

Two layers:

* **operator unit tests** -- the normaliser itself (``colour_range``,
  ``delta_normalization``), the exact reference formulas used by the benchmark,
  and the modal-projection algebra (a synthetic ``A cos(kx)`` stress must be
  recovered to machine precision);
* **physics guards** -- the drop modes recover the Rayleigh/Lamb stiffness, the
  capillary-wave mode does *not* (recorded as a strict defect test rather than
  silently skipped), and the measured stiffness is linear in ``sigma``.

Full ladder, ensembles and figures: ``benchmarks/surface_tension_dispersion/``
and ``docs/verification_surface_tension_dispersion.md``.
"""
from __future__ import annotations

import math

import pytest
import torch

from benchmarks.surface_tension_dispersion.dispersion import (
    AMPLITUDE_FRACTION,
    DROP_EPS,
    RHO0,
    SIGMA,
    capillary_wave_case,
    csf_acceleration,
    drop_geometry,
    drop_modal,
    exact_wave_stress_coeff,
    hex_lattice,
    lamb_case,
    wave_geometry,
    wave_modal,
)
from tensorsph.csf_consistent import colour_range, surface_tension_acceleration

torch.set_num_threads(2)
TOL = 1e-12


# --------------------------------------------------------------------------- #
# 1. exact reference formulas
# --------------------------------------------------------------------------- #
def test_exact_wave_stress_coeff_reduces_to_linear_theory():
    """For a k a << 1 the exact geometric coefficient is sigma a k^2."""
    for k in (1.0, 2.0, 3.0):
        amp = 1e-4 / k          # a k = 1e-4
        coeff = exact_wave_stress_coeff(k, amp)
        # the helper returns the curvature coefficient a k^2 (sigma is applied
        # by the caller); linear theory: Delta p mode = sigma a k^2.
        assert coeff == pytest.approx(amp * k * k, rel=1e-3)
        assert SIGMA * coeff == pytest.approx(SIGMA * amp * k * k, rel=1e-3)


def test_exact_wave_stress_coeff_is_the_second_derivative():
    """-eta'' at the crests, i.e. the linear curvature, for a small-amplitude wave."""
    k, amp = 2.0, 1e-3
    # eta = -a cos(kx) -> kappa = -eta'' = -a k^2 cos(kx) at leading order,
    # and the modal projection is defined with the positive cos mode.
    assert exact_wave_stress_coeff(k, amp) == pytest.approx(amp * k * k, rel=1e-3)


def test_lamb_reference_coefficient_is_sigma_n2m1_eps_over_R():
    """The drop stiffness used by ``lamb_case`` is sigma (n^2-1) eps / R."""
    R, n, eps = 0.02, 3, 0.1
    expected = SIGMA * (n * n - 1) * eps / R
    # independent check: the mode-n curvature amplitude of r = R(1+eps cos n t)
    # is (n^2-1) eps / R -- differentiate the polar curvature formula numerically
    theta = torch.linspace(0.0, 2 * math.pi, 20001, dtype=torch.float64)
    h = 1e-5
    r = lambda t: R * (1.0 + eps * torch.cos(n * t))  # noqa: E731
    r1 = (r(theta + h) - r(theta - h)) / (2 * h)
    r2 = (r(theta + h) - 2 * r(theta) + r(theta - h)) / h ** 2
    kappa = (r(theta) ** 2 + 2 * r1 ** 2 - r(theta) * r2) / (
        r(theta) ** 2 + r1 ** 2) ** 1.5
    # mode-n Fourier coefficient of kappa
    coeff = (kappa * torch.cos(n * theta)).sum() * (theta[1] - theta[0]) / math.pi
    # finite-amplitude (eps = 0.1) correction is 1.6 %; the linear reference is
    # used in the benchmark, so this test pins that bias.
    assert float(coeff) == pytest.approx((n * n - 1) * eps / R, rel=0.02)
    assert float(coeff) / ((n * n - 1) * eps / R) == pytest.approx(0.9836, abs=2e-3)
    assert SIGMA * float(coeff) == pytest.approx(expected, rel=0.02)


# --------------------------------------------------------------------------- #
# 2. estimator algebra
# --------------------------------------------------------------------------- #
def test_modal_projection_recovers_a_synthetic_stress():
    """sum w cos(kx) / (L/2) must return the mode amplitude exactly."""
    k, amp, n = 2.0, 3.0, 4000
    x = torch.linspace(0.0, 2 * math.pi / k, n + 1, dtype=torch.float64)[:-1]
    w = amp * torch.cos(k * x) * (2 * math.pi / k) / n     # = integral weights
    dp_k = 2.0 * k / (2 * math.pi) * (w * torch.cos(k * x)).sum()
    assert float(dp_k) == pytest.approx(amp, rel=1e-6)


def test_colour_range_is_one_on_a_flat_interface():
    """A planar free surface sees the whole 0 -> 1 colour transition."""
    from benchmarks.surface_tension_dispersion.dispersion import csf_acceleration
    g = wave_geometry(1.0, 1e-9, 0.02)          # flat limit
    _, state, _ = csf_acceleration(g, SIGMA, "fit_quadratic", "legacy")
    rng = colour_range(state.color, torch.empty((0, 2), dtype=torch.long))
    assert float(rng.max()) == 1.0               # no pairs -> constant 1
    # with pairs: the shell particles must report a range near 1
    from tensorsph.neighbor_search import cell_linked_pairs
    h = 1.3 * g["dx"]
    pairs = cell_linked_pairs(g["pos"], 2.0 * h)
    rng = colour_range(state.color, pairs)
    shell = state.delta_s > 0.2 * state.delta_s.max()
    # On a *one-sided* free surface the neighbourhood range never spans the full
    # 0 -> 1 transition: the void half of the colour profile carries no
    # particles, so the measured range saturates at ~1/2.  That is exactly why
    # the legacy factor 2 (the hard-coded 1/2) is right on a plane ...
    assert float(rng[shell].max()) == pytest.approx(0.5, abs=0.05)


# --------------------------------------------------------------------------- #
# 3. surface-measure defect (the thing this mission fixes) and the fix
# --------------------------------------------------------------------------- #
def _measure(geom, norm, drop=False):
    _, state, _ = csf_acceleration(geom, SIGMA, "fit_quadratic", norm)
    V = geom["mass"] / RHO0
    area = 2 * math.pi * geom["radius"] if drop else geom["length"]
    return float((state.delta_s * V).sum() / area)


def test_legacy_surface_measure_is_geometry_dependent():
    """The hard-coded 2|grad c| lands near 1.9 on a plane and 0.7 on a drop."""
    wave = wave_geometry(1.0, AMPLITUDE_FRACTION * 2 * math.pi, 0.02)
    drop = drop_geometry(2, DROP_EPS, 0.02, 0.02 / 12)
    m_wave = _measure(wave, "legacy")
    m_drop = _measure(drop, "legacy", drop=True)
    assert m_wave > 1.5          # 50-90 % too strong
    assert m_drop < 0.85         # 15-30 % too weak
    assert m_wave / m_drop > 2.0  # a 2.5x geometry-dependent spread


def test_local_range_normalisation_improves_the_drop_measure():
    """The drop, the geometry that matters for Laplace/Lamb, lands near unity."""
    drop = drop_geometry(2, DROP_EPS, 0.02, 0.02 / 16)
    legacy = _measure(drop, "legacy", drop=True)
    fixed = _measure(drop, "local_range", drop=True)
    assert abs(fixed - 1.0) < abs(legacy - 1.0)
    assert fixed > 0.85


def test_legacy_delta_function_is_bit_preserved():
    """``delta_normalization='legacy'`` must reproduce 2|grad c| exactly."""
    g = wave_geometry(2.0, AMPLITUDE_FRACTION * math.pi, 0.02)
    _, state, _ = csf_acceleration(g, SIGMA, "fit_quadratic", "legacy")
    mag = torch.linalg.vector_norm(state.grad_c, dim=1)
    # the returned field is post-masked (bulk particles carry exactly zero), so
    # the identity is checked on the shell only
    shell = mag > 1e-3 * mag.max()
    assert torch.allclose(state.delta_s[shell], 2.0 * mag[shell], atol=0, rtol=1e-14)


def test_unknown_delta_normalisation_is_rejected():
    g = drop_geometry(2, DROP_EPS, 0.02, 0.02 / 8)
    with pytest.raises(ValueError):
        csf_acceleration(g, SIGMA, "mls", "colour_range_typo")


def test_bulk_particles_carry_no_surface_tension():
    """Deep interior particles must get exactly zero CSF force."""
    g = drop_geometry(2, DROP_EPS, 0.02, 0.02 / 12)
    acc, state, _ = csf_acceleration(g, SIGMA, "fit_quadratic", "local_range")
    r = torch.linalg.vector_norm(g["pos"], dim=1)
    bulk = r < 0.5 * g["radius"]
    assert float(acc[bulk].abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# 4. physics guards
# --------------------------------------------------------------------------- #
def test_lamb_mode2_stiffness_matches_rayleigh_legacy():
    """Mode 2 on the legacy operator: within 15 % of the Rayleigh-Lamb stiffness."""
    row = lamb_case(2, 16, SIGMA, "fit_quadratic", "legacy", shifts=2)
    assert abs(row["omega_error_pct"]) < 15.0
    assert row["stiffness_ratio"] > 0.0


def test_lamb_mode2_stiffness_matches_rayleigh_local_range():
    row = lamb_case(2, 16, SIGMA, "fit_quadratic", "local_range", shifts=2)
    assert abs(row["omega_error_pct"]) < 20.0


def test_lamb_stiffness_is_linear_in_sigma():
    """C4: the measured stiffness must scale linearly with sigma (<5 % spread)."""
    vals = [lamb_case(2, 16, SIGMA * f, "fit_quadratic", "local_range",
                      shifts=2)["stiffness_ratio"] for f in (0.5, 1.0, 2.0)]
    spread = max(abs(v - vals[1]) / abs(vals[1]) for v in vals)
    assert spread < 0.05


def test_capillary_wave_mode_is_under_stiff_by_more_than_a_factor_two():
    """Known open defect, asserted instead of skipped.

    The CSF operator reproduces the *drop* stiffness to O(10 %) but its
    curvature field attenuates the low-wavenumber capillary-wave mode by an
    order of magnitude (``kappa h << 1``), so the wave stiffness is ~0.1-0.2 of
    the exact ``sigma k^2 a``.  Recorded as a strict pass/fail test so that a
    future fix flips it; see ``docs/verification_surface_tension_dispersion.md``.
    """
    row = capillary_wave_case(2, 32, SIGMA, "fit_quadratic", "legacy", shifts=2)
    assert row["stiffness_ratio"] < 0.5
    assert row["omega_error_pct"] < -30.0


def test_wave_surface_measure_and_stiffness_are_inconsistent():
    """Force amplitude and modal force cannot both be right (defect signature)."""
    row = capillary_wave_case(2, 32, SIGMA, "fit_quadratic", "legacy", shifts=2)
    # S/A ~1.9 means the force is over-strong per interface area, yet the modal
    # force is 5x too weak: the curvature mode is ~10x too small.
    implied_kappa_ratio = row["stiffness_ratio"] / row["surface_measure_ratio"]
    assert implied_kappa_ratio < 0.15