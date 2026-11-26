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
    k0 = lattice_symbol_ratio(0., 1.0, 1.3)
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
    gap = (lattice_symbol_ratio(0., 1.0, 1.3) - lattice_diffusion_factor(h_factor=1.3)[0]) \
        / lattice_symbol_ratio(0., 1.0, 1.3)
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

@pytest.mark.parametrize("dx,h_factor,volume", [(1., 1.3, None), (.125, 1.75, .25)])
def test_long_wave_ratio_from_quadratic_patch(dx, h_factor, volume, record_property):
    # T=x²/2 has unit curvature. Its center rate from the physical particle
    # pairs is the long-wave effective diffusivity for unit conductivity.
    from tensorsph.ice_thermal.sph_thermal import conduction_rhs

    positions = torch.arange(-5, 6, dtype=torch.float64)*dx
    ones = torch.ones_like(positions)
    rate = conduction_rhs(positions.square()/2, ones, ones*(volume or dx), 1.,
                          positions, h_factor*dx)[5].item()
    errors = []
    for k in (0., 1e-200, -1e-200, 2.**-38/dx):
        ratio = lattice_symbol_ratio(k, dx, h_factor*dx, volume=volume)
        errors.append(abs(ratio/rate-1))
    record_property("pair_reference_diffusivity_ratio", rate)
    record_property("max_relative_error", max(errors))
    assert max(errors) < 2e-15


def test_contact_solver_uses_exact_long_wave_calibration(record_property):
    dx, hf = .125, 1.3
    result = run_contact_sph(ContactConfig(n_particles=32, dx=dx, h_factor=hf,
                                           fourier=.01, calibration="k0"))
    expected = -2*sum(a*(m*dx)**2 for m,a in lattice_stencil(dx,hf*dx))
    error = abs(result.diffusion_factor/expected-1)
    record_property("relative_calibration_error", error)
    assert error < 2e-15


@pytest.mark.parametrize("alpha", [0.25, 4.0])
def test_fixed_wall_contact_diffusivity_scaling(alpha, record_property):
    # At fixed Fourier number the erfc profile and both discrete solutions
    # are independent of alpha. The wall planes are six similarity lengths out.
    options = dict(n_particles=96, dx=0.0625, fourier=0.0625, calibration="k0")
    fast = run_contact_sph(ContactConfig(alpha=alpha, **options))
    dense = run_contact_sph(ContactConfig(alpha=alpha, dense=True, **options))
    backend_error = max(abs(a-b) for a, b in zip(fast.temperature, dense.temperature))
    analytic_error = fast.error()[0]
    record_property("max_backend_difference_K", backend_error)
    record_property("max_normalized_erfc_error", analytic_error)
    assert fast.time * alpha == pytest.approx(options["fourier"])
    assert backend_error < 1e-12
    assert analytic_error < 0.002


@pytest.mark.parametrize("eta_factor", [0.125, 0.5])
def test_configured_pair_regularization_in_lattice_calibration(eta_factor, record_property):
    from tensorsph.heat_conduction_2d import ModeDecayConfig, run_mode_decay

    # The production calibration projects dense particle rates onto three modes.
    # Independently evaluate their exact eigenvalues from the stencil symbol.
    dx, h_factor = 0.0625, 1.5
    ratios = [lattice_symbol_ratio(2 * math.pi * m / (256 * dx), dx,
                                   h_factor * dx, eta_factor) for m in (1, 2, 3)]
    reference = sum(ratios) / len(ratios)
    contact = run_contact_sph(ContactConfig(n_particles=96, dx=dx, h_factor=h_factor,
                                            eta_factor=eta_factor, fourier=0.0625))
    # This comparison mode deliberately uses the 1-D calibration on a 2-D grid.
    mode = run_mode_decay(ModeDecayConfig(n=16, h_factor=h_factor, eta_factor=eta_factor,
                                          calibration="lattice_1d", amplitude_end=0.95))
    contact_error = abs(contact.diffusion_factor / reference - 1)
    mode_error = abs(mode.factor / reference - 1)
    record_property("contact_relative_factor_error", contact_error)
    record_property("comparison_mode_relative_factor_error", mode_error)
    record_property("contact_normalized_erfc_error", contact.error()[0])
    assert max(contact_error, mode_error) < 1e-12
    assert contact.diffusion_scatter == pytest.approx(max(ratios) - min(ratios), abs=1e-12)
    assert contact.error()[0] < 0.002


@pytest.mark.parametrize("alpha", [0.25, 4.0])
def test_fd_contact_at_fixed_fourier_number(alpha, record_property):
    # The contact lies midway between nodes on both grids. The fixed walls
    # are six similarity lengths away, so the infinite-domain erfc is resolved.
    fourier = 0.0625
    errors = []
    for n in (96, 192):
        x, temperature = fd_contact_reference(fourier=fourier, alpha=alpha, half_length=3., n=n)
        exact = contact_temperature(x, fourier / alpha, alpha)
        error = float((temperature - exact).abs().max())
        record_property(f"max_normalized_erfc_error_n{n}", error)
        errors.append(error)
    assert errors[1] < 3e-4
    assert errors[1] < 0.3 * errors[0]


@pytest.mark.parametrize("solver", ["sph", "fd"])
def test_node_centered_contact_symmetry_and_refinement(solver, record_property):
    hot, cold = 7., -1.
    errors = []
    for n in (97, 193):
        if solver == "fd":
            x, temperature = fd_contact_reference(fourier=0.0625, half_length=3., n=n,
                                                   t_hot=hot, t_cold=cold)
        else:
            cfg = ContactConfig(n_particles=n, dx=6./n, fourier=0.0625,
                                t_hot=hot, t_cold=cold, calibration="k0")
            result = run_contact_sph(cfg)
            x = torch.tensor(result.x, dtype=torch.float64)
            temperature = torch.tensor(result.temperature, dtype=torch.float64)
        exact = contact_temperature(x, 0.0625, 1., hot, cold)
        errors.append(float((temperature - exact).abs().max()) / (hot - cold))
        symmetry = float((temperature + temperature.flip(0) - (hot + cold)).abs().max())
        record_property(f"max_normalized_erfc_error_n{n}", errors[-1])
        record_property(f"max_reflection_defect_K_n{n}", symmetry)
    assert symmetry < 1e-12
    assert errors[1] < 0.3 * errors[0]


def test_finite_slab_contact_wall_influence_and_refinement(record_property):
    from tensorsph.heat_conduction import finite_slab_contact_temperature

    hot, cold, alpha, length = 7., -1., 0.25, 1.
    fourier = 0.03
    errors = []
    for n in (32, 64):
        cfg = ContactConfig(n_particles=n, dx=length/n, fourier=fourier,
                            alpha=alpha, t_hot=hot, t_cold=cold,
                            calibration="k0", reference="finite_slab")
        result = run_contact_sph(cfg)
        errors.append(result.error()[0])
        record_property(f"sph_max_normalized_error_n{n}", errors[-1])
    assert errors[1] < 0.3 * errors[0]
    assert errors[1] < 2e-4
    x, fd = fd_contact_reference(fourier, alpha, length/2, n=129, t_hot=hot, t_cold=cold)
    exact = finite_slab_contact_temperature(x, fourier/alpha, alpha, length, hot, cold)
    fd_error = float((exact - fd).abs().max()) / (hot-cold)
    record_property("fd_max_normalized_error", fd_error)
    assert fd_error < 2e-4
    assert exact[0] == pytest.approx(hot, abs=1e-14, rel=0)
    assert exact[-1] == pytest.approx(cold, abs=1e-14, rel=0)
    infinite = contact_temperature(x, fourier/alpha, alpha, hot, cold)
    discrepancy = float((exact - infinite).abs().max()) / (hot-cold)
    record_property("semi_infinite_max_normalized_difference", discrepancy)
    assert discrepancy > 0.01


@pytest.mark.parametrize("fourier", [0.001, 0.001000001, 4.0])
def test_finite_slab_contact_short_and_long_time_limits(fourier):
    from tensorsph.heat_conduction import finite_slab_contact_temperature

    x = torch.linspace(-1., 1., 65, dtype=torch.float64)
    time = fourier * 2**2
    exact = finite_slab_contact_temperature(x, time, 1., 2., -3., 5.)
    if fourier < 0.01:
        reference = contact_temperature(x, time, 1., -3., 5.)
    else:
        reference = 1 + 4*x
    torch.testing.assert_close(exact, reference, atol=5e-15, rtol=0)


@pytest.mark.parametrize("dense", [False, True])
def test_fixed_wall_contact_flux_refinement(dense, record_property):
    # Differentiate the finite slab Fourier series at x=-L/2. Both ends
    # have the same gradient and opposite inward normals.
    hot, cold, alpha, length, fo = 310., 302., .25, 1., .03
    exact = alpha * (hot-cold) / length * (1 + 2 * sum(
        (-1)**m * math.exp(-4*math.pi**2*m*m*fo) for m in range(1, 25)))
    errors = []
    for n in (32, 64):
        result = run_contact_sph(ContactConfig(
            n_particles=n, dx=length/n, alpha=alpha, fourier=fo*length**2,
            t_hot=hot, t_cold=cold, dense=dense, calibration="k0"))
        error = max(abs(result.wall_flux_left-exact), abs(result.wall_flux_right+exact))
        errors.append(error / exact)
        record_property(f"max_relative_flux_error_n{n}", errors[-1])
    record_property("analytic_left_inward_flux", exact)
    assert errors[1] < .005
    assert errors[1] < .3*errors[0]


@pytest.mark.parametrize("h_factor", [1.7, 14.0])
def test_contact_wall_fluxes_balance_dense_rates(h_factor, record_property):
    from tensorsph.heat_conduction import contact_wall_fluxes
    from tensorsph.ice_thermal.sph_thermal import conduction_rhs

    dx, k, n = .125, .375, 12
    h = h_factor * dx
    x = (torch.arange(n, dtype=torch.float64) + .5) * dx
    t = 2. + x**2
    walls = 1.3, 4.8
    fluxes = contact_wall_fluxes(t, lattice_stencil(dx, h), dx, k, *walls)
    ext = torch.cat((2*walls[0]-t.flip(0), t, 2*walls[1]-t.flip(0)))
    pos = torch.cat((-x.flip(0), x, 2*n*dx-x.flip(0)))
    rates = conduction_rhs(ext, torch.full_like(ext, k), torch.full_like(ext, dx),
                           1., pos, h)
    difference = abs(sum(fluxes)-float(dx*rates[n:2*n].sum()))
    record_property("absolute_heat_balance_residual", difference)
    assert fluxes[0] < 0 < fluxes[1]
    assert difference < 1e-13
