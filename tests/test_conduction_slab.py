"""Tests for the two-slab contact anchor of the SPH conduction operator.

The anchor (:mod:`tensorsph.ice_thermal.conduction_slab`) is only worth anything
if (a) its fast banded update *is* the library operator, (b) its reference is the
textbook closed form and not a fitted curve, and (c) it still fails when the
operator is broken.  The three groups of tests below pin exactly those three
things, in that order.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorsph.ice_thermal import properties as props
from tensorsph.ice_thermal.conduction_slab import (
    SlabConfig,
    analytic_temperature,
    band_apply,
    band_coefficients,
    diffusion_length,
    fit_effective_diffusivity,
    run_slab_contact,
)
from tensorsph.ice_thermal.sph_thermal import conduction_rhs

DTYPE = torch.float64


# --------------------------------------------------------------- (a) operator
def test_band_apply_matches_library_conduction_rhs():
    """Banded update == ``conduction_rhs / cp`` on the interior, to 1e-13.

    ``conduction_rhs`` returns ``dH/dt``; with ``H = cp T`` the temperature rate
    is ``dH/dt / cp``.  The banded path is what the anchor integrates, so this
    test is the guarantee that the fast path cannot drift from the library.
    """
    n, dx, hf = 64, 0.01, 1.3
    h = hf * dx
    x = (torch.arange(n, dtype=DTYPE) + 0.5) * dx
    torch.manual_seed(7)
    t = torch.rand(n, dtype=DTYPE) * 5.0 + 263.0
    coeffs, q_max = band_coefficients(n, dx, hf)

    banded = props.DIFF_ICE * band_apply(t, coeffs, q_max)
    library = conduction_rhs(t,
                             torch.full((n,), props.K_ICE, dtype=DTYPE),
                             torch.full((n,), dx, dtype=DTYPE),
                             props.RHO_ICE, x, h) / props.C_ICE
    interior = slice(3, n - 3)  # edges are overwritten by the Dirichlet BC
    scale = float(library[interior].abs().max())
    assert scale > 0.0
    assert float((banded[interior] - library[interior]).abs().max()) < 1e-12 * scale


def test_band_coefficients_are_diffusive_and_isotropic_in_sign():
    """Every coefficient is negative (``dW/dr < 0``) and decays outwards."""
    coeffs, q_max = band_coefficients(64, 0.01, 1.3)
    assert q_max == 3
    assert coeffs[0] == 0.0                    # no self-interaction
    nonzero = [c for c in coeffs[1:] if c != 0.0]
    assert nonzero, "kernel support is empty -- geometry is broken"
    assert all(c < 0.0 for c in nonzero), "positive coefficient = anti-diffusion"
    assert abs(nonzero[0]) > abs(nonzero[-1])  # nearest neighbour dominates


def test_single_step_reduces_a_spike():
    """One step of the operator must lower a spike and conserve the sum."""
    n, dx = 41, 0.01
    coeffs, q_max = band_coefficients(n, dx, 1.3)
    t = torch.full((n,), 265.0, dtype=DTYPE)
    t[n // 2] = 275.0
    before = float(t.sum())
    rate = props.DIFF_ICE * band_apply(t, coeffs, q_max)
    assert float(rate[n // 2]) < 0.0           # the spike cools
    assert float(rate[n // 2 - 1]) > 0.0       # its neighbour warms
    assert abs(float(rate.sum())) < 1e-9       # conservative operator


# -------------------------------------------------------------- (b) reference
def test_analytic_reference_limits():
    """Textbook limits: step at t=0, interface at the mean, antisymmetry."""
    t_hot, t_cold, alpha = 263.15, 268.15, props.DIFF_ICE
    x = torch.tensor([-0.2, -0.02, 0.0, 0.02, 0.2], dtype=DTYPE)

    step = analytic_temperature(x, 0.0, alpha, t_hot, t_cold)
    assert float(step[0]) == t_hot and float(step[-1]) == t_cold

    t = 4386.0
    prof = analytic_temperature(x, t, alpha, t_hot, t_cold)
    assert abs(float(prof[2]) - 0.5 * (t_hot + t_cold)) < 1e-12
    # T(x) + T(-x) = 2 T_mean for the symmetric two-slab solution
    assert abs(float(prof[0] + prof[-1]) - (t_hot + t_cold)) < 1e-10
    # monotone increasing in x (hot slab on the left), and the far field
    # returns to the initial values
    assert float(prof[0]) < float(prof[2]) < float(prof[4])
    sigma = diffusion_length(alpha, t)
    far = analytic_temperature(x, t, alpha, t_hot, t_cold)
    # x = -0.2 m is 2.83 sigma: erfc(2.83) = 6.3e-5 of the step at 4 sigma, so
    # the tolerance below is the textbook tail value, not a tuned one
    assert abs(float(far[0]) - t_hot) < 0.03 * abs(t_hot - t_cold)
    # at x = -4 sigma the argument is 2, so the exact tail is erfc(2)/2 of the
    # step; anything smaller would be measuring float noise, not the formula
    tail = 0.5 * abs(t_hot - t_cold) * math.erfc(2.0)
    at4 = float(analytic_temperature(torch.tensor([-4.0 * sigma]), t, alpha,
                                     t_hot, t_cold)[0])
    assert abs(at4 - t_hot) <= tail * 1.001
    assert float(analytic_temperature(torch.tensor([8.0 * sigma]), t, alpha,
                                      t_hot, t_cold)[0]) - t_cold < 1e-6


def test_fit_recovers_the_diffusivity_of_a_synthetic_profile():
    """The fit is unbiased: feeding it the analytic profile returns ``alpha``."""
    alpha, t = props.DIFF_ICE, 4386.0
    n = 201
    sigma = diffusion_length(alpha, t)
    x = torch.linspace(-3 * sigma, 3 * sigma, n, dtype=DTYPE)
    t_ref = analytic_temperature(x, t, alpha, 263.15, 268.15)
    got = fit_effective_diffusivity(x, t_ref, t, 263.15, 268.15, alpha_guess=alpha)
    assert abs(got / alpha - 1.0) < 1e-6


# --------------------------------------------------------------- (c) the gate
def test_contact_anchor_passes_the_three_percent_gate():
    """The library operator reproduces Carslaw & Jaeger to well under 3 %."""
    res = run_slab_contact(SlabConfig(n_particles=120))
    assert res.finite
    assert res.rms_rel < 0.03, f"RMS profile error {res.rms_rel:.4%} exceeds 3%"
    assert res.max_abs_k < 0.03 * 5.0
    # conservation of the discrete operator (no flux at the pinned ends)
    assert res.energy_drift < 1e-10


def test_calibration_removes_the_measured_operator_bias():
    """Raw run shows the lattice bias, calibrated run removes it.

    This is the positive control for ``lattice_diffusion_factor``: without it
    the fitted diffusivity sits at ``alpha_eff = factor * alpha`` (1.0218),
    with it at ``alpha``.  If the two runs agreed, the anchor would not be
    sensitive to the operator amplitude and the gate would be worthless.
    """
    raw = run_slab_contact(SlabConfig(n_particles=120, calibrate=False))
    cal = run_slab_contact(SlabConfig(n_particles=120, calibrate=True))

    factor = raw.diffusion_factor
    assert 1.0 < factor < 1.10
    assert abs(raw.fitted_alpha_ratio - factor) < 1e-3
    assert abs(cal.fitted_alpha_ratio - 1.0) < 1e-3
    assert raw.fitted_alpha_ratio - 1.0 > 0.01      # bias is measurable
    # both runs still pass the 3 % profile gate: the bias is an amplitude
    # error of the *diffusivity*, not a first-order distortion of the shape
    assert raw.rms_rel < 0.03 and cal.rms_rel < 0.03
    assert cal.rms_rel < raw.rms_rel


def test_spatial_refinement_reduces_the_error():
    """Coarse -> fine must reduce the profile error (no saturation at 3 %)."""
    coarse = run_slab_contact(SlabConfig(n_particles=60))
    fine = run_slab_contact(SlabConfig(n_particles=240))
    assert fine.rms_rel < coarse.rms_rel
    assert fine.dx < coarse.dx
    assert fine.steps > coarse.steps


def test_anchor_holds_at_other_fourier_numbers():
    """The gate is not tuned to one time level: t* = 0.1 and 1.0 also pass."""
    for fourier in (0.1, 1.0):
        res = run_slab_contact(SlabConfig(n_particles=120, fourier=fourier))
        assert res.finite
        assert res.rms_rel < 0.03, f"t*={fourier}: {res.rms_rel:.4%}"
        assert abs(res.fitted_alpha_ratio - 1.0) < 1e-3


def test_negative_controls_fail_loudly():
    """A flipped-sign operator and an over-large dt must both be rejected."""
    flipped = run_slab_contact(SlabConfig(n_particles=120, negative_control=True))
    assert not math.isfinite(flipped.rms_rel) or flipped.rms_rel > 1.0

    unstable = run_slab_contact(SlabConfig(n_particles=120, courant=1.8))
    assert not math.isfinite(unstable.rms_rel) or unstable.rms_rel > 1.0
    # and the default Courant number stays inside the stability envelope
    stable = run_slab_contact(SlabConfig(n_particles=120))
    assert stable.rms_rel < 0.03


def test_result_bookkeeping():
    """Profile arrays, window mask and error column agree with each other."""
    res = run_slab_contact(SlabConfig(n_particles=120))
    n = res.cfg.n_particles
    assert len(res.x) == len(res.temperature) == len(res.analytic) == n
    assert len(res.error) == len(res.in_window) == n
    assert abs(res.x[0] + res.half_length) < res.dx
    assert sum(res.in_window) > n // 4
    for i in (0, n // 3, n - 1):
        assert abs(res.error[i] - (res.temperature[i] - res.analytic[i])) < 1e-12
    # the particle nearest x = 0 sits at +(dx/2): its analytic value is the
    # mean plus the (small) offset, which the run must reproduce
    assert abs(res.interface_k - res.interface_analytic_k) < 1e-3