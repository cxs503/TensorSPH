"""Round-2 invariants for ``CASE-SPH-OSC-DROP-FREQ``.

These tests pin the *numbers* that round 2 of the case established, so that the
statements in ``docs/case_sph_osc_drop_freq_round2.md`` cannot drift away from
the code:

* the axisymmetric volume element closes (``sum dV / (4/3 pi R^3) = 1``);
* the curvature estimator is **not** the defect round 1 blamed -- the meridional
  and azimuthal parts each give ``1/R`` to a few percent, i.e. ``<kappa> R ~ 2``,
  and round 1's ``1.2`` was its own missing factor 2 in the coarea;
* the colour-gradient surface delta is 32 % short of the plane-curve coarea and
  the axisymmetric coarea overshoots by 25 % -- i.e. the measure is *anisotropic*
  over the sphere and one global constant cannot fix it;
* with that measure renormalised and the curvature taken as exact, the discrete
  quadrature reproduces the Rayleigh modal stiffness within the 6 % gate;
* the shipped curvature estimator does **not** resolve the ``O(eps)``
  deformation response (a known limitation, pinned here so a future "the
  curvature is fine" claim has to update this test rather than contradict it).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "benchmarks" / "oscillating_drop_freq"))

from tensorsph.oscillating_drop3d import (  # noqa: E402
    axisym_drop_geometry,
    exact_modal_stress,
    legendre_p,
    rayleigh_frequency_3d,
)
from tensorsph.surface_tension_dispersion import csf_state  # noqa: E402

from ensemble import axisym_geometry_rotated  # noqa: E402

R = 0.02
SIGMA = 7.2
RHO = 1000.0
GATE_PCT = 6.0


def _state(rdx: float, eps: float, mode: int = 2):
    geom = axisym_drop_geometry(R, R / rdx, mode=mode, eps=eps, rho0=RHO)
    state, h, _ = csf_state(geom, SIGMA, periodic_x=False)
    return geom, state


def test_volume_element_closes():
    for rdx in (8.0, 12.0, 16.0):
        geom = axisym_drop_geometry(R, R / rdx, mode=2, eps=0.02, rho0=RHO)
        ratio = float(geom["dvol"].sum()) / (4.0 / 3.0 * math.pi * R ** 3)
        assert 0.95 < ratio < 1.07, f"R/dx={rdx}: dV closes to {ratio:.3f} of the sphere"


def test_curvature_split_is_not_the_defect():
    """Round 1 reported ``<kappa> R = 1.21`` and blamed the azimuthal term.

    The estimate is actually ``<kappa> R = 2.0..2.2`` once the delta-weighted mean
    is taken with the coarea it implies; meridional and azimuthal parts each come
    out at ~``1/R``.  This test falsifies the round-1 attribution.
    """
    for rdx in (12.0, 16.0):
        geom, state = _state(rdx, 0.0)
        w = state.delta_s * geom["dvol"]
        sign = torch.where(geom["pos"][:, 0] >= 0.0, 1.0, -1.0)
        kappa_az = state.normal[:, 0] * sign / geom["rho_safe"]
        k_m = float((state.kappa * w).sum() / w.sum()) * R
        k_az = float((kappa_az * w).sum() / w.sum()) * R
        assert 0.90 < k_m < 1.15, f"R/dx={rdx}: meridional kappa R = {k_m:.3f}"
        assert 0.90 < k_az < 1.20, f"R/dx={rdx}: azimuthal kappa R = {k_az:.3f}"
        assert 1.90 < k_m + k_az < 2.30, f"R/dx={rdx}: total kappa R = {k_m + k_az:.3f}"


def test_colour_delta_is_anisotropic_over_the_sphere():
    """Plane-curve coarea 0.68 vs axisymmetric coarea 1.25: not a scale error."""
    geom, state = _state(12.0, 0.0)
    dA = geom["mass"] / RHO
    dV = geom["dvol"]
    plane = float((state.delta_s * dA).sum()) / (2.0 * math.pi * R)
    axisym = float(2.0 * (state.delta_s * dV).sum()) / (4.0 * math.pi * R ** 2)
    assert 0.60 < plane < 0.78, f"plane-curve coarea ratio {plane:.3f}"
    assert 1.15 < axisym < 1.40, f"axisymmetric coarea ratio {axisym:.3f}"
    # the two disagree by far more than either is off on its own => anisotropic
    assert axisym / plane > 1.6


def test_quadrature_reproduces_rayleigh_with_exact_curvature():
    """Measure + volume element + modal projector, with the curvature exact.

    Averaged over four lattice orientations (the problem is rotationally
    symmetric, so the answer may not depend on the lattice orientation) the
    discrete capillary quadrature lands inside the 6 % gate.
    """
    dps, dp_exact = [], None
    for k in range(4):
        psi = math.pi * k / 16.0
        geom = axisym_geometry_rotated(R, R / 12.0, mode=2, eps=0.05, rho0=RHO, psi=psi)
        state, _, _ = csf_state(geom, SIGMA, periodic_x=False)
        kappa = (2.0 / R + 2.0 * 2.0 * 0.05 * legendre_p(2, geom["cos_theta"]) / R)
        area = 4.0 * math.pi * R ** 2
        w = state.delta_s * geom["dvol"]
        scale = area / (2.0 * float(w.sum()))
        dp = float((5.0 / area * (-SIGMA * kappa * w * legendre_p(2, geom["cos_theta"]))
                    .sum()) * scale)
        dps.append(dp)
        dp_exact = exact_modal_stress(SIGMA, R, 2, 0.05)
    ratio = (sum(dps) / len(dps)) / dp_exact
    omega_ratio = math.sqrt(ratio)
    omega_exact = rayleigh_frequency_3d(SIGMA, RHO, R, 2)
    err = (omega_ratio - 1.0) * 100.0
    assert omega_ratio > 0.0
    assert abs(err) < GATE_PCT, (f"quadrature route: omega = {omega_ratio * omega_exact:.2f} "
                                 f"vs Rayleigh {omega_exact:.2f} rad/s ({err:+.2f} %)")


def test_shipped_curvature_does_not_resolve_the_deformation_known_limitation():
    """Known limitation, pinned deliberately.

    The shipped ``fit_quadratic`` curvature is accurate on a *static* circle but
    its response to an ``O(eps)`` interface deformation is not converged
    (measured response ratio -0.9..+0.2 against an exact value of 1.0 over every
    stencil/smoothing combination, see
    ``benchmarks/oscillating_drop_freq/curvature_response.json``).  If this test
    starts failing, someone made the estimator respond correctly: re-measure the
    case and update the doc instead of deleting this test.
    """
    def proj(eps: float) -> float:
        geom, state = _state(12.0, eps)
        w = state.delta_s * geom["dvol"]
        area = 4.0 * math.pi * R ** 2
        share = legendre_p(2, geom["cos_theta"])
        norm = float(5.0 / area * w.sum())
        return float(5.0 / area * (state.kappa * w * share).sum()) / norm

    exact = 2.0 * 2.0 * 0.05 / R
    ratio = (proj(0.05) - proj(0.0)) / exact
    assert abs(ratio - 1.0) > 0.20, (
        f"response ratio {ratio:+.3f} is now within 20 % of the exact 1.0 -- the "
        "known limitation documented in docs/case_sph_osc_drop_freq_round2.md has "
        "been fixed; re-run benchmarks/oscillating_drop_freq/ensemble.py and "
        "re-publish the case verdict")