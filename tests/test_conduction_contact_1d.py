"""CASE-SPH-CONDUCTION-1D: two half-spaces in contact vs the erfc similarity solution.

Gates (all cheap enough for CI -- the full ladder/figure run lives in
``benchmarks/case_conduction_1d/run.py``):

* analytic limits of the reference: ``T -> T_hot`` for ``x -> -inf``,
  ``T -> T_cold`` for ``x -> +inf``, ``T = (T_hot + T_cold)/2`` at ``x = 0``,
  and time invariance in the similarity variable ``eta``;
* the ``O(N)`` lattice stencil used by the solver equals the ``O(N^2)``
  ``conduction_rhs`` pair sum to round-off (so the fast path is the *same*
  operator, not a second implementation of the physics);
* the SPH field is inside the 3 % platform bar at two resolutions, with the
  ``k -> 0`` calibration, and the fine/coarse pair shows second-order
  behaviour;
* the Crank-Nicolson finite-difference reference (independent code path)
  confirms the erfc profile;
* the residual diffusivity left by the repository's three-mode calibration
  equals the ``k -> 0`` gap of the lattice symbol (the physical explanation of
  the error plateau, not a fitted fudge).
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.heat_conduction import (ContactConfig, contact_temperature,
                                       fd_contact_reference, lattice_stencil,
                                       lattice_symbol_ratio, run_contact_sph,
                                       stencil_rhs, stencil_vs_dense)
from tensorsph.ice_thermal.sph_thermal import lattice_diffusion_factor

ALPHA = 1.0
T_HOT, T_COLD = 1.0, 0.0
X_CUT = 4.0


def test_reference_limits_and_symmetry():
    x = torch.tensor([-50.0, -1.0, 0.0, 1.0, 50.0], dtype=torch.float64)
    t = contact_temperature(x, 0.5, ALPHA, T_HOT, T_COLD)
    assert t[0] == pytest.approx(T_HOT, abs=1e-12)
    assert t[-1] == pytest.approx(T_COLD, abs=1e-12)
    assert t[2] == pytest.approx(0.5 * (T_HOT + T_COLD), abs=1e-15)
    # antisymmetry about the contact plane when T_hot = -T_cold
    x = torch.tensor([-2.0, -0.3, 0.3, 2.0], dtype=torch.float64)
    t = contact_temperature(x, 0.7, ALPHA, 1.0, -1.0)
    assert float(t[0]) == pytest.approx(-float(t[-1]), abs=1e-14)
    assert float(t[1]) == pytest.approx(-float(t[2]), abs=1e-14)


def test_reference_is_self_similar():
    """Same eta at different times gives the same temperature."""
    eta = torch.tensor([0.3, 0.7, 1.4], dtype=torch.float64)
    ref = None
    for t in (0.05, 0.5, 5.0):
        x = eta * 2.0 * math.sqrt(ALPHA * t)
        t_val = contact_temperature(x, t, ALPHA, T_HOT, T_COLD)
        if ref is None:
            ref = t_val
        else:
            assert torch.allclose(t_val, ref, atol=1e-13)


def test_reference_rejects_zero_time():
    with pytest.raises(ValueError):
        contact_temperature(torch.zeros(1), 0.0, ALPHA)


def test_stencil_equals_dense_operator():
    """The time loop's 5-point lattice stencil is the dense Cleary-Monaghan sum."""
    assert stencil_vs_dense() <= 1e-6


def test_stencil_coefficients_are_diffusive():
    coeffs = lattice_stencil(0.02, 1.3 * 0.02)
    assert [m for m, _ in coeffs] == [1, 2]          # support 2h = 2.6 dx
    assert all(a < 0.0 for _, a in coeffs)           # rhs = +a (2T_i - T_i-1 - T_i+1)
    # a uniform field must have zero rate; a peak must decay
    t = torch.full((20,), 3.0, dtype=torch.float64)
    assert float(stencil_rhs(t, coeffs).abs().max()) == pytest.approx(0.0, abs=1e-12)
    peak = torch.zeros(21, dtype=torch.float64)
    peak[10] = 1.0
    rate = stencil_rhs(peak, coeffs)
    assert float(rate[10]) < 0.0 < float(rate[9])


@pytest.mark.parametrize("dx, tol", [(0.04, 1.0e-4), (0.02, 2.0e-5)])
def test_contact_profile_against_erfc(dx, tol):
    res = run_contact_sph(ContactConfig(n_particles=int(round(12.0 / dx)), dx=dx,
                                        calibration="k0"))
    mx, rms = res.error(X_CUT)
    assert mx <= 3.0e-2                     # platform bar
    assert mx <= tol                        # second-order accuracy actually delivered
    assert res.interface_temperature() == pytest.approx(0.5, abs=1e-3)


def test_second_order_trend():
    coarse = run_contact_sph(ContactConfig(n_particles=150, dx=0.08, calibration="k0"))
    fine = run_contact_sph(ContactConfig(n_particles=300, dx=0.04, calibration="k0"))
    order = math.log(coarse.error(X_CUT)[0] / fine.error(X_CUT)[0]) / math.log(2.0)
    assert 1.7 <= order <= 2.3


def test_calibration_factor_is_the_k0_symbol():
    """The k->0 calibration and the three-mode mean bracket the symbol."""
    factor = lattice_diffusion_factor(h_factor=1.3)[0]
    k0 = lattice_symbol_ratio(1e-3, 1.0, 1.3)
    assert 1.0 < factor < k0 < 1.05
    # the three-mode mean sits below the k->0 limit by ~4e-4 -- that gap is the
    # residual diffusivity measured in the benchmark
    assert 1e-4 < (k0 - factor) / k0 < 1e-3


def test_residual_diffusivity_explains_the_plateau():
    """lattice calibration leaves d(alpha)/alpha ~ the 3-mode vs k0 gap."""
    res = run_contact_sph(ContactConfig(n_particles=600, dx=0.02, calibration="lattice"))
    x = torch.tensor(res.x)
    a = torch.tensor(res.temperature)
    b = torch.tensor(res.analytic)
    e = a - b
    eta = x / (2.0 * math.sqrt(ALPHA * res.time))
    tp = 0.5 * (T_COLD - T_HOT) * (2.0 / math.sqrt(math.pi)) * torch.exp(-eta ** 2) \
        / (2.0 * math.sqrt(ALPHA * res.time))
    tpp = -2.0 * eta * tp / (2.0 * math.sqrt(ALPHA * res.time))
    keep = x.abs() <= X_CUT
    lhs, rhs = e[keep], res.time * tpp[keep]
    d_alpha = float((lhs * rhs).sum() / (rhs * rhs).sum())
    gap = (lattice_symbol_ratio(1e-3, 1.0, 1.3) - lattice_diffusion_factor(h_factor=1.3)[0]) \
        / lattice_symbol_ratio(1e-3, 1.0, 1.3)
    # the model must explain the plateau, not merely correlate with it: the
    # residual left after removing it is a small fraction of the raw error
    residual = float((lhs - d_alpha * rhs).abs().max())
    assert d_alpha == pytest.approx(gap, rel=0.30)
    assert residual <= 0.2 * float(lhs.abs().max())


def test_fd_reference_matches_erfc():
    x, T = fd_contact_reference(n=601)
    exact = contact_temperature(x, 0.5, ALPHA, T_HOT, T_COLD)
    assert float((T - exact).abs().max()) <= 1.0e-2


def test_wall_condition_does_not_pollute_the_metric():
    narrow = run_contact_sph(ContactConfig(n_particles=300, dx=0.04, calibration="k0"))
    wide = run_contact_sph(ContactConfig(n_particles=600, dx=0.04, calibration="k0"))
    assert abs(narrow.error(X_CUT)[0] - wide.error(X_CUT)[0]) <= 1e-6