"""Tests for the mode-2 transferability measurement of the well-balanced CSF split.

Anchor ``XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE``.  What is pinned here:

1. the analytic reference itself (``omega^2 = sigma n(n^2-1)(k^{2n}-1)/(2 rho R^3 k^{2n})``)
   against the closed forms of its two limits, so a wrong exponent cannot slip
   into the benchmark unnoticed;
2. the benchmark's geometry (an ellipse of *equal area* on the shipped lattice,
   so a shape sweep does not silently change the resolution);
3. the instrument: the modal projection of a pure ``cos 2theta`` radial field is
   recovered exactly, and of a radially symmetric field is zero;
4. the two claims the deliverable rests on, at a resolution cheap enough for CI:
   the shipped (full) arm produces a non-zero mode-2 response, and the
   well-balanced (irrotational) arm produces none (``R2 << 1``) -- i.e. the split
   is a fixed-point device for shape dynamics, not a dynamics-preserving
   formulation.  This is the *measurement*, not a preference.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.laplace_isph import csf_state_two_phase, hex_block

from benchmarks.laplace_sph.modal_forcing_split import (
    RHO0,
    SIGMA,
    ellipse_block,
    omega_theory,
    run_case,
)

torch.set_num_threads(2)


# --------------------------------------------------------------------------- #
# 1. the analytic reference
# --------------------------------------------------------------------------- #
def test_unconfined_limit_matches_classic_rayleigh_halved():
    """``k -> inf``: equal-density carrier halves the classic (vacuum) value."""
    R, rho, sigma, n = 0.02, 1000.0, 0.072, 2
    classic = math.sqrt(sigma * n * (n * n - 1.0) / (rho * R ** 3))
    assert omega_theory(R, mode=n, sigma=sigma, rho=rho, k=None) == pytest.approx(
        classic / math.sqrt(2.0), rel=1e-12)


def test_confined_limit_is_below_unconfined():
    R = 0.02
    w_unconf = omega_theory(R, k=None)
    for k in (1.2, 1.6, 3.0):
        w = omega_theory(R, k=k)
        assert 0.0 < w < w_unconf
        # k^{2n}/(k^{2n}-1) is the confinement amplification of omega^2 (n = 2 -> k^4)
        assert w_unconf / w == pytest.approx(math.sqrt(k ** 4 / (k ** 4 - 1.0)),
                                             rel=1e-12)
    # the two references used in the deliverable
    assert omega_theory(R, k=1.6) == pytest.approx(4.783317567664203, rel=1e-12)
    assert w_unconf == pytest.approx(5.196152422706631, rel=1e-12)


def test_mode_one_is_a_translation_and_has_no_frequency():
    """``n = 1`` is a rigid-body translation: no restoring force."""
    assert omega_theory(0.02, mode=1, k=None) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 2. geometry: equal area, fixed resolution
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("aspect", [1.0, 1.1, 1.5])
def test_ellipse_block_conserves_liquid_area(aspect):
    dx = 0.02 / 8
    _pos, mass, liquid, _half, a, b = ellipse_block(aspect, dx)
    assert math.pi * a * b == pytest.approx(math.pi * 0.02 ** 2, rel=1e-12)
    cell = dx * dx * math.sqrt(3.0) / 2.0
    area = float(liquid.sum()) * cell
    assert area == pytest.approx(math.pi * 0.02 ** 2, rel=0.02)


# --------------------------------------------------------------------------- #
# 3. the instrument: modal projection
# --------------------------------------------------------------------------- #
def test_modal_projection_recovers_a_pure_mode_two_field():
    from benchmarks.laplace_sph.modal_forcing_split import _modal_mean, _mode_weights

    # a ring of points; a radial field with amplitude 1 * cos 2theta
    n = 720
    theta = torch.arange(n, dtype=torch.float64) * (2.0 * math.pi / n)
    r = torch.full_like(theta, 0.02)
    pos = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    rhat, c2, near = _mode_weights(pos)
    vol = torch.full((n,), 1.0e-6, dtype=torch.float64)
    f_mode2 = rhat * c2[:, None] * 3.5          # 3.5 cos^2 2theta rhat
    f_radial = rhat * 2.0                       # no cos 2theta content
    assert _modal_mean(f_mode2, vol, rhat, c2, near) == pytest.approx(3.5, rel=1e-9)
    assert abs(_modal_mean(f_radial, vol, rhat, c2, near)) < 1e-9


# --------------------------------------------------------------------------- #
# 4. the two claims (cheap resolution, deterministic)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def case():
    return run_case(1.10, 8, rel_tol=1e-8, cg_max_iter=20000)


def test_split_reconstructs_the_force(case):
    assert case["reconstruction_defect"] <= 1e-12
    assert abs(case["orthogonality"]) <= 1e-6


def test_shipped_arm_has_a_mode_two_response(case):
    """The full CSF arm drives mode 2 with a restoring (negative) acceleration."""
    assert case["modal_accel_full_projected_m_s2"] < 0.0
    assert case["omega_meas_full_rad_s"] == case["omega_meas_full_rad_s"]  # not NaN
    assert 1.0 < case["omega_meas_full_rad_s"] < 10.0


def test_well_balanced_arm_annihilates_the_mode_two_response(case):
    """``u* = dt Gq`` is mapped to zero by the projector, so the split is a fixed point.

    This is the finding of ``docs/xval_projection_laplace_modal_transferability.md``:
    the well-balanced arm keeps the *static* jump and loses the *dynamics*.
    """
    assert case["R2_velocity"] < 1e-5
    assert abs(case["v2_irrotational_over_vref"]) < 1e-5
    # and it is not a small-number artefact: the full arm's response is O(1e-3)
    assert abs(case["v2_full_over_vref"]) > 1e-5


def test_balanced_arm_still_holds_the_static_circle():
    """The round-3 static result must survive: a circle is a fixed point (control)."""
    rec = run_case(1.0, 8, rel_tol=1e-8, cg_max_iter=20000)
    assert rec["umax_over_vref_irrotational"] < 1e-5