"""Tests for the 2-D conduction operator family (CASE-SPH-CONDUCTION-2D).

What is asserted (invariants, not just "the code ran"):

* the ``torch.roll`` fast path is the *same* operator as the dense pair sum;
* the 2-D code degenerates exactly to the shipped 1-D ``conduction_rhs``;
* the lattice product mode is an eigenfunction with rate ``symbol_rate`` -- this is
  what lets the symbol be used as a calibration at all;
* calibrating with the 2-D symbol makes the mode decay exact (RK4), while the 1-D
  calibration (dimensional mismatch) keeps a >100 % error: the trap is pinned;
* the 2-D-calibrated arm converges at ~2nd order and the committed artefact's main
  run is inside the 3 % gate, and the registry reads that same number;
* ``heat_conduction_operator`` now has two benchmark *families* (the verification
  debt this case was built to pay) and the refuted 1-D-lift record is present.
"""
from __future__ import annotations

import json
import cmath
import math
from pathlib import Path

import pytest
import torch

from tensorsph.heat_conduction_2d import (
    ModeDecayConfig,
    degenerate_1d_agreement,
    fd_mode_decay,
    lattice_offsets,
    mode_field,
    run_mode_decay,
    stencil_rhs_2d,
    stencil_vs_dense_2d,
    symbol_rate,
)

ROOT = Path(__file__).resolve().parents[1]
ARTEFACT = ROOT / "benchmarks" / "case_conduction_2d" / "result.json"
REGISTRY = ROOT / "docs" / "reports" / "method_selection_registry.json"


def test_stencil_is_the_dense_pair_sum():
    assert stencil_vs_dense_2d() < 1e-12


def test_two_d_code_degenerates_to_shipped_one_d_operator():
    # exact, not approximate: same kernel, same harmonic mean, same 2/rho
    assert degenerate_1d_agreement() == 0.0


def test_product_mode_is_an_eigenfunction_of_the_lattice_operator():
    n, dx = 24, 1.0 / 24
    h = 1.3 * dx
    k = 2.0 * 3.141592653589793
    field = mode_field(n, dx, k, k)
    rate = stencil_rhs_2d(field, lattice_offsets(dx, h))
    lam = symbol_rate(dx, h, k, k)
    assert float((rate * field).sum() / (field * field).sum()) == \
        _approx(-lam, rel=1e-12)
    residual = rate + lam * field
    assert float(residual.abs().max()) < 1e-10 * float(field.abs().max())


def _approx(value, rel):
    class _C:
        def __eq__(self, other):
            return abs(float(other) - value) <= rel * abs(value)
    return _C()


def test_symbol_calibration_makes_the_decay_exact():
    res = run_mode_decay(ModeDecayConfig(n=32, calibration="mode_2d",
                                         integrator="rk4"))
    assert res.max_normalised_error < 1e-7


def test_one_d_calibration_does_not_transfer_to_two_d():
    res = run_mode_decay(ModeDecayConfig(n=32, calibration="lattice_1d",
                                         integrator="euler"))
    assert res.max_normalised_error > 1.0     # >100 %: O(dx) dimensional mismatch


def test_two_d_calibrated_arm_is_second_order_and_inside_the_gate():
    coarse = run_mode_decay(ModeDecayConfig(n=32, calibration="k0_2d",
                                            integrator="rk4"))
    fine = run_mode_decay(ModeDecayConfig(n=64, calibration="k0_2d",
                                          integrator="rk4"))
    assert fine.max_normalised_error < coarse.max_normalised_error
    order = (torch.log(torch.tensor(coarse.max_normalised_error / fine.max_normalised_error))
             / torch.log(torch.tensor(2.0)))
    assert 1.5 < float(order) < 2.5
    assert fine.max_normalised_error < 0.03


def test_committed_artefact_is_inside_the_gate():
    doc = json.loads(ARTEFACT.read_text())
    assert doc["case"] == "CASE-SPH-CONDUCTION-2D"
    assert doc["main_run"]["max_normalised_error"] <= doc["gate"]
    assert doc["equivalence"]["stencil_vs_dense_max_abs"] < 1e-12
    assert doc["equivalence"]["degenerate_1d_vs_shipped_conduction_rhs_max_abs"] == 0.0
    # the independent reference validates the closed form used as the target
    assert doc["fd_reference"][0]["max_normalised_error_vs_closed_form"] < 0.02


def test_registry_reads_the_same_number_and_the_operator_has_two_families():
    from tensorsph.method_selection import case_family, load_registry

    doc = json.loads(ARTEFACT.read_text())
    registry = load_registry()
    ev = registry["heat_conduction_operator"]
    families = {case_family(e.case) for e in ev}
    assert len(families) >= 2, families
    banked = [e for e in ev if not e.is_refuted and e.case.startswith("CASE-SPH-CONDUCTION-2D")]
    assert banked, "the 2-D record must be banked"
    assert abs(banked[0].measured_pct - doc["main_run"]["max_normalised_error"] * 100.0) < 1e-9
    refuted = [e for e in ev if e.is_refuted]
    assert refuted and all(e.measured_pct > 100.0 for e in refuted)


def test_fd_reference_converges_to_the_closed_form():
    coarse = fd_mode_decay(n=32)["max_normalised_error_vs_closed_form"]
    fine = fd_mode_decay(n=64)["max_normalised_error_vs_closed_form"]
    assert fine < coarse
    assert fine < 0.01


def test_two_material_periodic_transfer_conserves_heat():
    temperature = torch.tensor([[0., 1., 0., 1.]], dtype=torch.float64)
    k = 1 + 2 * temperature
    offsets = (((0, 1), -1.), ((0, -1), -1.))
    rhs = stencil_rhs_2d(temperature, offsets, conductivity=k)
    # Each interface has harmonic conductivity 1.5, and each cell has two faces.
    torch.testing.assert_close(rhs, 6 * (1 - 2 * temperature))
    assert rhs.sum().item() == 0


def test_dirichlet_walls_preserve_affine_temperature_and_restore_wall_perturbation():
    offsets = lattice_offsets(1., 1.3)
    x = (torch.arange(12, dtype=torch.float64) + 0.5) / 12
    temperature = (10 - 7*x).expand(5, -1).clone()
    rhs = stencil_rhs_2d(temperature, offsets, x_wall_temperatures=(10., 3.))
    assert rhs.abs().max() < 1e-12
    temperature[:, 0] += 1.
    rhs = stencil_rhs_2d(temperature, offsets, x_wall_temperatures=(10., 3.))
    assert (rhs[:, 0] < 0.).all()
    assert rhs[:, -1].abs().max() < 1e-12


def test_dirichlet_transient_converges_interior_and_wall_adjacent():
    from benchmarks.case_conduction_2d.wall_decay import measure

    coarse, fine = measure(16), measure(32)
    assert fine["max_error"] < coarse["max_error"] / 3.5
    assert fine["wall_adjacent_max_error"] < coarse["wall_adjacent_max_error"] / 7
    assert fine["mean_temperature_drift"] < 1e-13


def test_spatial_wall_profiles_approach_the_harmonic_equilibrium():
    from benchmarks.case_conduction_2d.wall_profile import measure

    coarse, fine = measure(16), measure(32)
    assert fine["max_error"] < coarse["max_error"] / 3
    assert fine["wall_adjacent_max_error"] < 0.007


def test_wall_vector_uses_y_axis_on_a_rectangular_grid():
    # Tangentially independent pairs isolate reflection at each row's own wall.
    t = torch.zeros((3, 5), dtype=torch.float64)
    left = torch.tensor([1., 2., 3.], dtype=t.dtype)
    rhs = stencil_rhs_2d(t, (((0, 1), -1.), ((0, -1), -1.)),
                         x_wall_temperatures=(left, 0.))
    torch.testing.assert_close(rhs[:, 0], 4 * left)
    assert torch.count_nonzero(rhs[:, 1:]) == 0


def test_wall_power_accounts_for_material_density_area_and_transverse_pairs():
    from tensorsph.heat_conduction_2d import stencil_rhs_with_wall_power_2d

    dx, rho = .2, 2.7
    rows = torch.arange(5, dtype=torch.float64)
    field = (rows[:, None] + torch.arange(7, dtype=torch.float64)).cos()
    k = 1 + field.square()
    offsets = lattice_offsets(dx, 1.3 * dx)
    walls = (rows / 2, 1 - rows)
    rate, power = stencil_rhs_with_wall_power_2d(field, offsets, dx, walls, rho, k)
    direct = stencil_rhs_2d(field, offsets, rho, k, x_wall_temperatures=walls)
    torch.testing.assert_close(rate, direct, rtol=0, atol=0)
    torch.testing.assert_close(power.sum(), rho * dx**2 * direct.sum())


def test_wall_heat_budget_converges_to_the_two_fourier_flux_integrals():
    from benchmarks.case_conduction_2d.wall_heat_budget import measure

    coarse, fine = measure(16), measure(32)
    assert fine["wall_max_abs_error"] < coarse["wall_max_abs_error"] / 3.8
    assert fine["wall_max_abs_error"] < 5e-5
    assert max(coarse["heat_balance_residual"], fine["heat_balance_residual"]) < 1e-13


def test_oscillating_wall_matches_time_periodic_slab():
    from benchmarks.case_conduction_2d.wall_cycle import measure

    coarse, fine = measure(16), measure(32)
    assert fine["max_error"] < coarse["max_error"] / 3.8
    assert fine["max_error"] < .011


def test_slab_midpoint_samples_walls_at_stage_times():
    from benchmarks.case_conduction_2d.wall_decay import evolve_slab

    times = []

    def walls(t):
        times.append(t)
        return t, 0.

    field = torch.zeros(4, 8, dtype=torch.float64)
    _, steps, dt = evolve_slab(field, 1/8, walls, .004, .16)
    assert steps == 2
    assert times == [0., dt/2, dt, 1.5*dt]


def test_second_moment_calibrates_steady_affine_flux():
    from tensorsph.heat_conduction_2d import lattice_diffusion_factor_2d, stencil_rhs_with_wall_power_2d

    n = 24
    dx = 1 / n
    x = (torch.arange(n, dtype=torch.float64) + .5) * dx
    field = (1 + 2 * x).expand(3, -1)
    offsets = lattice_offsets(dx, 1.3 * dx)
    _, power = stencil_rhs_with_wall_power_2d(field, offsets, dx, (1., 3.), conductivity=4.)
    flux = power / (3 * dx * lattice_diffusion_factor_2d(1.3, dx, mode="moment"))
    torch.testing.assert_close(flux, x.new_tensor([-8., 8.]), rtol=1e-12, atol=1e-12)


def test_layered_path_reproduces_multilayer_series_resistance():
    from tensorsph.heat_conduction_2d import lattice_diffusion_factor_2d, stencil_rhs_with_wall_power_2d

    # A particle pair can cross more than one of these narrow layers.
    k = torch.tensor([1., 2., 7., 1., 3., 8., 3., 5.], dtype=torch.float64)
    dx = 1 / len(k)
    resistance = dx / k
    flux = 2 / resistance.sum()
    field = (2 - flux * (resistance.cumsum(0) - .5 * resistance)).expand(5, -1)
    h_factor = 2.
    offsets = lattice_offsets(dx, h_factor * dx)
    rate, power = stencil_rhs_with_wall_power_2d(
        field, offsets, dx, (2., 0.), conductivity=k, layered_x=True)
    assert rate.abs().max() < 1e-12
    factor = lattice_diffusion_factor_2d(h_factor, dx, mode="moment")
    torch.testing.assert_close(power / (5 * dx * factor), torch.stack((flux, -flux)),
                               rtol=1e-13, atol=1e-13)


def test_layered_contact_exact_across_multiple_faces_and_near_walls():
    from tensorsph.heat_conduction_2d import lattice_diffusion_factor_2d, stencil_rhs_with_wall_power_2d

    k = torch.tensor([1., 3., 2., 5., 4., 8., 7., 2.], dtype=torch.float64)
    faces = k.new_tensor([.3, .15, 0., .24, 0., .08, .15])
    dx = 1 / len(k)
    bulk = dx / k
    flux = 2 / (bulk.sum() + faces.sum())
    cumulative = bulk.cumsum(0) - .5 * bulk + torch.cat((k.new_zeros(1), faces.cumsum(0)))
    t = (3 - flux * cumulative).expand(4, -1)
    offsets = lattice_offsets(dx, 2 * dx)
    rate, power = stencil_rhs_with_wall_power_2d(t, offsets, dx, (3., 1.), conductivity=k,
                                                layered_x=True, contact_resistance_x=faces)
    assert rate.abs().max() < 1e-12
    factor = lattice_diffusion_factor_2d(2., dx, mode="moment")
    torch.testing.assert_close(power / (4 * dx * factor), torch.stack((flux, -flux)))


def test_layered_periodic_contact_at_seam():
    dx = .25
    t = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64)
    offsets = (((0, 1), -.5), ((0, -1), -.5))
    rate = stencil_rhs_2d(t, offsets, conductivity=t.new_ones(4), layered_x=True, dx=dx,
                          contact_resistance_x=t.new_tensor([0., 0., 0., 3 * dx]))
    torch.testing.assert_close(rate, t.new_tensor([[-1.25, 1., 0., .25]]))


def test_contact_decay_refines_temperature_and_jump():
    from benchmarks.case_conduction_2d.contact_decay import measure

    coarse, fine = measure(16), measure(32)
    for key in ("temperature_max_abs_error", "jump_abs_error"):
        assert fine[key] < coarse[key] / 3.5


def test_particle_heat_capacity_balances_wall_power():
    from tensorsph.heat_conduction_2d import stencil_rhs_with_wall_power_2d

    t = torch.arange(15, dtype=torch.float64).reshape(3, 5) / 4
    cp = 1 + t.square()
    dx, rho = .2, 2.
    offsets = lattice_offsets(dx, 1.3 * dx)
    rate, power = stencil_rhs_with_wall_power_2d(t, offsets, dx, (1., 3.), rho=rho,
                                                 specific_heat=cp)
    base, base_power = stencil_rhs_with_wall_power_2d(t, offsets, dx, (1., 3.), rho=rho)
    torch.testing.assert_close(rate * cp, base)
    torch.testing.assert_close(power, base_power)
    torch.testing.assert_close(power.sum(), dx**2 * rho * (cp * rate).sum())


def test_heterogeneous_contact_decay_and_stored_heat_refine():
    from benchmarks.case_conduction_2d.contact_decay import measure

    coarse, fine = (measure(n, heterogeneous=True) for n in (16, 32))
    for key in ("temperature_max_abs_error", "jump_abs_error", "stored_heat_change_abs_error"):
        assert fine[key] < coarse[key] / 3.5


def test_every_internal_cut_closes_the_left_slab_heat_balance():
    from tensorsph.heat_conduction_2d import stencil_rhs_with_wall_power_2d, stencil_x_face_power_2d

    dx, rho = .125, 1.7
    t = torch.arange(40, dtype=torch.float64).reshape(5, 8).sin()
    k = 1 + t[0].square()
    cp = 2 + t
    faces = .15 * k[:-1]
    walls = (t[:, 0] + 2, t[:, -1] - 1)
    offsets = lattice_offsets(dx, 2 * dx)
    opts = dict(conductivity=k, layered_x=True, contact_resistance_x=faces)
    rate, wall_power = stencil_rhs_with_wall_power_2d(t, offsets, dx, walls, rho=rho,
                                                     specific_heat=cp, **opts)
    for face in range(1, t.shape[1]):
        cut = stencil_x_face_power_2d(t, offsets, dx, face, walls, **opts)
        stored = (cp[:, :face] * rate[:, :face]).sum() * rho * dx**2
        torch.testing.assert_close(cut, wall_power[0] - stored, atol=2e-13, rtol=0)


def test_slab_film_particles_match_external_power_with_insulated_wall():
    from tensorsph.heat_conduction_2d import (
        lattice_diffusion_factor_2d, prepare_slab_film_rhs, stencil_rhs_with_wall_power_2d,
    )

    dx, rho = .125, 2.
    t = 300 + torch.arange(40, dtype=torch.float64).reshape(5, 8).sin()
    k = 1 + (t[0]-300).square()
    cp = 1.5 + (t-300).square()
    offsets = lattice_offsets(dx, 1.3*dx)
    opts = dict(conductivity=k, specific_heat=cp, rho=rho, layered_x=True)
    films = (0., 4.)
    rhs = prepare_slab_film_rhs(t, offsets, dx, films, **opts)
    rate, power, wall = rhs(t, (295., 306.))
    direct_rate, direct_power = stencil_rhs_with_wall_power_2d(t, offsets, dx, wall, **opts)
    factor = lattice_diffusion_factor_2d(1.3, dx, mode="moment")
    torch.testing.assert_close(rate, direct_rate/factor)
    torch.testing.assert_close(power, direct_power/factor, atol=2e-12, rtol=1e-12)
    expected = 5*dx*t.new_tensor(films)*(t.new_tensor([295., 306.])-wall)
    torch.testing.assert_close(power, expected)
    torch.testing.assert_close(power.sum(), rho*dx**2*(cp*rate).sum())


def test_diagonal_wall_transfer_is_assigned_to_ghost_wall_row():
    from tensorsph.heat_conduction_2d import stencil_rhs_with_wall_power_2d

    field = torch.zeros((5, 3), dtype=torch.float64)
    left = field.new_tensor([1., 3., 5., 1., 2.])
    offsets = (((1, -1), -1.), ((-1, 1), -1.))
    dx = .3
    rate, power = stencil_rhs_with_wall_power_2d(field, offsets, dx, (left, 0.), power_by_row=True)
    torch.testing.assert_close(rate[:, 0], 4 * torch.roll(left, -1))
    torch.testing.assert_close(power[0], 4 * dx**2 * left)
    assert torch.count_nonzero(power[1]) == 0


def test_varying_local_film_preserves_wall_law_and_refines():
    from benchmarks.case_conduction_2d.local_film import measure

    coarse, fine = (measure(n) for n in (16, 32))
    for key in ("field_max_error_K", "wall_max_error_K", "flux_max_error_W_m2"):
        assert fine[key] < .75 * coarse[key]
    assert fine["local_balance_W_m2"] < 1e-12
    assert fine["storage_balance_J"] < 1e-12


def test_mixed_wall_decay_matches_temperature_and_local_flux():
    from benchmarks.case_conduction_2d.mixed_wall_decay import measure

    coarse, fine = measure(16), measure(32)
    for metric in ("max_temperature_error", "fixed_flux_error"):
        assert fine[metric] < coarse[metric] / 3.8
    assert fine["heat_balance"] < 1e-13
    assert fine["insulated_flux"] == 0


def test_insulated_heterogeneous_slab_conserves_heat_and_respects_reflection():
    from tensorsph.heat_conduction_2d import stencil_rhs_with_wall_power_2d

    field = torch.arange(35, dtype=torch.float64).reshape(5, 7).sin()
    k, cp = 2 + field, 3 - field
    offsets = lattice_offsets(.1, .13)

    def evaluate(t, conductivity, capacity, walls):
        return stencil_rhs_with_wall_power_2d(t, offsets, .1, walls, conductivity=conductivity,
                                              specific_heat=capacity, power_by_row=True)

    rate, powers = evaluate(field, k, cp, (None, None))
    assert powers.abs().max() == 0
    assert abs(.01 * (cp * rate).sum()) < 1e-12
    rate, powers = evaluate(field, k, cp, (None, 2.))
    mirror_rate, mirror_powers = evaluate(field.flip([1]), k.flip([1]), cp.flip([1]), (2., None))
    torch.testing.assert_close(mirror_rate.flip([1]), rate)
    torch.testing.assert_close(mirror_powers.flip([0]), powers)
    torch.testing.assert_close(.01 * (cp * rate).sum(), powers.sum())


def test_insulated_step_temperature_and_internal_flux_approach_neumann_series():
    from benchmarks.case_conduction_2d.insulated_step import measure

    coarse, fine = measure(16), measure(32)
    for metric in ("max_temperature_error", "center_flux_error"):
        assert fine[metric] < coarse[metric]/3.8
    assert max(coarse["mean_drift"], fine["mean_drift"]) < 1e-13


def test_steady_slab_solves_pair_equations_with_spatial_wall_data():
    from tensorsph.heat_conduction_2d import lattice_diffusion_factor_2d, solve_steady_slab_2d

    ny, nx, dx, k = 7, 11, .07, 1.7
    phase = torch.arange(ny, dtype=torch.float64)
    wall = phase.cos()+275.
    source = (torch.arange(nx, dtype=torch.float64)[None, :] + phase[:, None]).sin()
    offsets = lattice_offsets(dx, 1.3*dx)
    factor = lattice_diffusion_factor_2d(1.3, dx, mode="moment")
    for walls in ((None, wall), (wall, None), (wall, wall.flip((0,)))):
        t = solve_steady_slab_2d(source, dx, walls, conductivity=k)
        rate = stencil_rhs_2d(t, offsets, x_wall_temperatures=walls)
        torch.testing.assert_close(k*rate/factor, -source, atol=1e-10, rtol=0)


def test_closed_steady_slab_uses_mean_and_rejects_incompatible_heating():
    import pytest
    from tensorsph.heat_conduction_2d import solve_steady_slab_2d

    previous = None
    for n in (16, 32, 64, 128, 256):
        x = (torch.arange(n, dtype=torch.float64)+.5)/n
        mode = torch.cos(2*torch.pi*x[:, None])*torch.cos(torch.pi*x)
        # Unit conductivity, two insulated x walls and periodic y.
        t = solve_steady_slab_2d(5*torch.pi**2*mode, 1/n, (None, None), mean_temperature=7.)
        error = float((t-7.-mode).abs().max())
        assert abs(float(t.mean())-7.) < 1e-13
        if previous is not None:
            assert error < previous/3.8
        previous = error
    assert error < .0004
    with pytest.raises(ValueError, match="zero net heat"):
        solve_steady_slab_2d(torch.ones_like(mode), 1/n, (None, None), mean_temperature=7.)
    with pytest.raises(ValueError, match="mean_temperature"):
        solve_steady_slab_2d(mode, 1/n, (None, None))


def test_harmonic_slab_matches_real_space_exchange_and_storage():
    from tensorsph.heat_conduction_2d import lattice_diffusion_factor_2d, solve_harmonic_slab_2d

    dx, k, omega, capacity = .1, 1.4, 2.7, 3.
    x = torch.arange(11, dtype=torch.float64)
    y = torch.arange(7, dtype=torch.float64)
    source = torch.exp(1j*(x+y[:, None]))+.4j
    wall = torch.exp(.31j*y)
    factor = lattice_diffusion_factor_2d(1.3, dx, mode="moment")
    offsets = lattice_offsets(dx, 1.3*dx)
    for walls in ((wall, wall.conj()), (None, wall), (wall, None), (None, None)):
        field = solve_harmonic_slab_2d(source, dx, walls, omega, conductivity=k,
                                       volumetric_heat_capacity=capacity)
        for component in ("real", "imag"):
            traces = tuple(None if w is None else getattr(w, component) for w in walls)
            rate = stencil_rhs_2d(getattr(field, component), offsets, x_wall_temperatures=traces)
            expected = getattr(1j*omega*capacity*field-source, component)
            torch.testing.assert_close(k*rate/factor, expected, atol=5e-12, rtol=0)
        if all(w is None for w in walls):
            torch.testing.assert_close(field.mean(), source.mean()/(1j*omega*capacity))


def test_complex_pair_powers_match_independent_quadratures_through_layered_contact():
    from tensorsph.heat_conduction_2d import stencil_rhs_with_wall_power_2d, stencil_x_face_power_2d

    index = torch.arange(40, dtype=torch.float64).reshape(5, 8)
    field = torch.cos(index)+1j*torch.sin(.3*index)
    dx = .125
    offsets = lattice_offsets(dx, 1.3*dx)
    walls = (torch.exp(.17j*index[:, 0]), None)
    material = dict(conductivity=1+index[0], layered_x=True, contact_resistance_x=.02*index[0, :-1])
    def outputs(t, w):
        rate, powers = stencil_rhs_with_wall_power_2d(t, offsets, dx, w, specific_heat=2.,
                                                      power_by_row=True, **material)
        cut = stencil_x_face_power_2d(t, offsets, dx, 3, w, **material)
        return rate, powers, cut
    for t in (field, field.real):
        actual = outputs(t, walls)
        real = outputs(t.real, (walls[0].real, None))
        imag = outputs(t.imag if t.is_complex() else torch.zeros_like(t), (walls[0].imag, None))
        for value, a, b in zip(actual, real, imag):
            torch.testing.assert_close(value, torch.complex(a, b))


def test_uniform_wall_flux_lift_and_heat_balance():
    import pytest
    from tensorsph.heat_conduction_2d import solve_steady_slab_2d
    n, k = 20, 2.
    dx = 1/n
    x = (torch.arange(n, dtype=torch.float64)+.5)*dx
    source = torch.zeros((5, n), dtype=x.dtype)
    for walls, flux, exact in (
        ((None, 7.), (3., 0.), 7.+3*(1-x)/k),
        ((7., None), (0., -3.), 7.-3*x/k),
    ):
        u = solve_steady_slab_2d(source, dx, walls, conductivity=k, x_wall_inward_flux=flux)
        torch.testing.assert_close(u, exact.expand_as(u), rtol=0, atol=1e-12)
    raw = (-x.square()/2-2*x)/k
    u = solve_steady_slab_2d(source+1, dx, (None, None), conductivity=k,
                            x_wall_inward_flux=(2., -3.), mean_temperature=4.)
    torch.testing.assert_close(u, (4.+raw-raw.mean()).expand_as(u), atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="net source"):
        solve_steady_slab_2d(source, dx, (None,None), x_wall_inward_flux=(2., -3.), mean_temperature=4.)
    with pytest.raises(ValueError, match="temperature None"):
        solve_steady_slab_2d(source, dx, (3.,None), x_wall_inward_flux=(2., 0.))


def test_uniform_flux_with_spatial_source_converges():
    from tensorsph.heat_conduction_2d import solve_steady_slab_2d
    errors = []
    for n in (12, 24, 48):
        x = (torch.arange(n, dtype=torch.float64)+.5)/n
        mode = torch.cos(torch.pi*x)[None,:]*torch.cos(2*torch.pi*x)[:,None]
        source = 5*torch.pi**2*mode
        exact = 2. - .7*(x-.5) + mode
        u = solve_steady_slab_2d(source, 1/n, (None,None), x_wall_inward_flux=(.7,-.7),
                                mean_temperature=2.)
        errors.append(float((u-exact).abs().max()))
    assert all(a/b > 3.7 for a,b in zip(errors, errors[1:])), errors


def test_harmonic_flux_storage_and_reflected_mixed_wall():
    from tensorsph.heat_conduction_2d import solve_harmonic_slab_2d
    source = torch.full((5, 17), .25-.4j, dtype=torch.complex128)
    options = dict(dx=.1, angular_frequency=2., conductivity=3., volumetric_heat_capacity=4.)
    west, east = .2+.7j, -1.+.5j
    u = solve_harmonic_slab_2d(source, x_wall_temperatures=(None,None),
                              x_wall_inward_flux=(west,east), **options)
    torch.testing.assert_close(8j*u.mean(), source.mean()+(west+east)/1.7)
    left = solve_harmonic_slab_2d(source, x_wall_temperatures=(None,2.+1j),
                                  x_wall_inward_flux=(west,0.), **options)
    right = solve_harmonic_slab_2d(source, x_wall_temperatures=(2.+1j,None),
                                   x_wall_inward_flux=(0.,west), **options)
    torch.testing.assert_close(left, right.flip((1,)), atol=1e-12, rtol=0)


def test_slab_section_flux_resolves_storage_in_each_column():
    from tensorsph.heat_conduction_2d import solve_harmonic_slab_2d, solve_steady_slab_2d

    nx, ny, dx = 17, 5, .08
    x = torch.arange(nx, dtype=torch.float64)*dx
    y = torch.arange(ny, dtype=torch.float64)[:,None]*dx
    source = (.6-.3j)*torch.cos(x+2*y)
    for walls, imposed in (((None,None), (.2+.7j, -.3+.1j)),
                           ((y[:,0]+1j,None), (0., .1-.8j)),
                           ((None,y[:,0]-1j), (.3j,0.)),
                           ((y[:,0],-y[:,0]), (0.,0.))):
        u, q = solve_harmonic_slab_2d(source, dx, walls, 2., conductivity=3.,
                      volumetric_heat_capacity=4., x_wall_inward_flux=imposed, return_x_flux=True)
        torch.testing.assert_close(q.diff(), dx*(source-8j*u).mean(dim=0), rtol=0, atol=2e-13)
        if walls[0] is None:
            assert abs(q[0]-imposed[0]) < 1e-13
        if walls[1] is None:
            assert abs(q[-1]+imposed[1]) < 1e-13
    _, q = solve_steady_slab_2d(torch.ones_like(source.real), dx, (None,None),
             mean_temperature=7., x_wall_inward_flux=(2., -2.-nx*dx), return_x_flux=True)
    torch.testing.assert_close(q, 2.+dx*torch.arange(nx+1, dtype=x.dtype), atol=1e-13, rtol=0)


def test_harmonic_layered_slab_solves_full_pair_equations():
    from tensorsph.heat_conduction_2d import (
        lattice_diffusion_factor_2d, solve_harmonic_layered_slab,
        stencil_rhs_with_wall_power_2d,
    )
    nx, dx, omega = 15, .08, 3.
    x = torch.arange(nx, dtype=torch.float64)
    k = torch.where(x<6, 2., 5.).double()
    c = 1.+x/10
    r = torch.zeros_like(x[:-1])
    r[[5,10]] = torch.tensor([.2,.4], dtype=r.dtype)
    source = torch.exp(.1j*x)
    offsets = lattice_offsets(dx, 1.3*dx)
    factor = lattice_diffusion_factor_2d(1.3, dx, mode="moment")
    for walls in ((.3+.8j,-.4j), (None, .8j), (None,None)):
        u = solve_harmonic_layered_slab(source, dx, walls, omega, conductivity=k,
                          volumetric_heat_capacity=c, contact_resistance_x=r)
        rate, powers = stencil_rhs_with_wall_power_2d(u.expand(4,-1), offsets, dx, walls,
                          conductivity=k, layered_x=True, contact_resistance_x=r)
        torch.testing.assert_close(rate/factor+source, (1j*omega*c*u).expand_as(rate),
                                   rtol=1e-11, atol=1e-11)
        balance = dx*(1j*omega*c*u-source).sum()-powers.sum()/(4*dx*factor)
        assert abs(balance) < 1e-11


def test_transverse_layered_mode_matches_resolved_particle_exchange():
    from tensorsph.heat_conduction_2d import lattice_diffusion_factor_2d, solve_harmonic_layered_slab
    nx, ny, dx = 15, 11, .065
    beta = -4*torch.pi/(ny*dx)
    x = torch.arange(nx, dtype=torch.float64)*dx
    y = torch.arange(ny, dtype=torch.float64)*dx
    phase = torch.exp(1j*beta*y)
    k = 1.+(x>.4).double()
    c = 1.+x
    contact = torch.zeros_like(x[:-1])
    contact[6] = .3
    source = torch.exp(.5j*x)
    for walls in ((.4j, -.2+1j), (None, 1.+.2j)):
        u = solve_harmonic_layered_slab(source, dx, walls, 4., conductivity=k,
                     volumetric_heat_capacity=c, contact_resistance_x=contact, transverse_wavenumber=beta)
        field = phase[:,None]*u
        traces = tuple(None if t is None else t*phase for t in walls)
        rate = stencil_rhs_2d(field, lattice_offsets(dx,1.3*dx), x_wall_temperatures=traces,
                              dx=dx, layered_x=True, contact_resistance_x=contact, conductivity=k)
        factor = lattice_diffusion_factor_2d(1.3, dx, mode="moment")
        torch.testing.assert_close(rate/factor+phase[:,None]*source, 4j*c*field, atol=2e-12, rtol=1e-11)


def test_section_power_resolves_periodic_diagonal_intersections():
    from tensorsph.heat_conduction_2d import stencil_x_face_power_2d

    # Crossing from column 1 to 3 through cut 2 occurs at 1/4 of the link.
    # Row 4 therefore deposits 3/4 on row 4 and 1/4 across the periodic seam.
    field = torch.zeros((5,6), dtype=torch.complex128)
    field[4,1] = 1.+2j
    offsets = (((1,2), -1.), ((-1,-2), -1.))
    args = (field, offsets, .5, 2, (None,None))
    rows = stencil_x_face_power_2d(*args, power_by_row=True)
    total = stencil_x_face_power_2d(*args)
    expected = torch.tensor([.25,0.,0.,0.,.75], dtype=torch.float64)*(.5+1j)
    torch.testing.assert_close(rows, expected)
    torch.testing.assert_close(rows.sum(), total)


def test_layered_slab_grid_accepts_nonseparable_complex_loads():
    from tensorsph.heat_conduction_2d import lattice_diffusion_factor_2d, solve_harmonic_layered_slab
    dx = .11
    for ny in (1, 6, 7):
        y = torch.arange(ny, dtype=torch.float64)[:,None]
        x = torch.arange(10, dtype=torch.float64)*dx
        source = torch.exp(.8j*y*x) + .1*y*x
        walls = (torch.exp(.4j*y[:,0]), .25j)
        k = 2.+x
        c = 1.+x*x
        contacts = torch.zeros_like(x[:-1])
        contacts[4] = .2
        u = solve_harmonic_layered_slab(source, dx, walls, 2., conductivity=k,
                             volumetric_heat_capacity=c, contact_resistance_x=contacts)
        rate = stencil_rhs_2d(u, lattice_offsets(dx,1.3*dx), x_wall_temperatures=walls,
                              dx=dx, layered_x=True, contact_resistance_x=contacts, conductivity=k)
        factor = lattice_diffusion_factor_2d(1.3, dx, mode="moment")
        torch.testing.assert_close(rate/factor+source, 2j*c*u, atol=1e-12, rtol=1e-11)


def test_banded_harmonic_slab_agrees_with_dense_on_wide_stencils():
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_slab
    import pytest
    pytest.importorskip("scipy.linalg")
    for nx, ny, smoothing in ((2,1,.8), (17,5,1.3), (29,8,2.1)):
        x = torch.arange(nx, dtype=torch.float64)/nx
        y = torch.arange(ny, dtype=torch.float64)[:,None]
        source = torch.exp(2j*x*y)
        opts = dict(conductivity=1.+x, volumetric_heat_capacity=3.-x, h_factor=smoothing,
                    contact_resistance_x=.1*x[:-1])
        for walls in ((None,None), (torch.exp(.3j*y[:,0]),None)):
            u = solve_harmonic_layered_slab(source, 1/nx, walls, 4., **opts)
            compact = solve_harmonic_layered_slab(source, 1/nx, walls, 4., backend="banded", **opts)
            torch.testing.assert_close(compact, u, atol=2e-12, rtol=1e-11)


def test_layered_harmonic_films_close_storage_and_reciprocity():
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_films
    x = (torch.arange(20, dtype=torch.float64)+.5)/20
    k, c = 1.+x, 2.+x*x
    faces = torch.zeros_like(x[:-1])
    faces[9] = .35
    films = torch.tensor([2., 7.], dtype=x.dtype)
    ambient = torch.tensor([1.+.5j, -.2j], dtype=torch.complex128)
    opts = dict(film_coefficients=films, conductivity=k, volumetric_heat_capacity=c,
                contact_resistance_x=faces)
    source = torch.exp(1j*x)
    u, walls, q = solve_harmonic_layered_films(source, .05, ambient, 4., **opts)
    torch.testing.assert_close(q, films*(ambient-walls), atol=2e-12, rtol=0)
    torch.testing.assert_close(q.sum(), (4j*c*u-source).mean(), atol=2e-12, rtol=0)
    zero = torch.zeros_like(source)
    _, _, a = solve_harmonic_layered_films(zero,.05,(1.,0.),4.,**opts)
    _, _, b = solve_harmonic_layered_films(zero,.05,(0.,1.),4.,**opts)
    torch.testing.assert_close(a[1], b[0], atol=2e-12, rtol=0)


def test_dynamic_films_close_wall_and_layer_energy():
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_films
    n = 24
    x = torch.arange(n, dtype=torch.float64)/n
    h, cw = torch.tensor([2.,5.], dtype=x.dtype), torch.tensor([.4,1.],dtype=x.dtype)
    a = torch.tensor([1.+.2j, .5j],dtype=torch.complex128)
    u, wall, q = solve_harmonic_layered_films(1j*x,1/n,a,3.,conductivity=1.+x,
                     volumetric_heat_capacity=2.,film_coefficients=h,wall_areal_heat_capacity=cw)
    torch.testing.assert_close(q+3j*cw*wall, h*(a-wall), atol=1e-12,rtol=0)
    torch.testing.assert_close(q.sum(),(6j*u-1j*x).mean(),atol=1e-12,rtol=0)


@pytest.mark.parametrize("walls",[(.6,-.3),(.6,None),(None,-.3),(None,None)])
@pytest.mark.parametrize("theta",[.5,1.])
def test_implicit_step_closes_heat_balance_on_every_section(walls,theta):
    from tensorsph.heat_conduction_2d import step_transient_slab_2d
    dx,dt,capacity = .12,.1,2.
    field = torch.arange(54,dtype=torch.float64).reshape(6,9).sin()
    source = field.cos()
    west,east = (.8 if walls[0] is None else 0.),(-.3 if walls[1] is None else 0.)
    updated,flux = step_transient_slab_2d(field,dt,dx,walls,theta=theta,heat_source=source,
                     conductivity=1.5,volumetric_heat_capacity=capacity,
                     x_wall_inward_flux=(west,east),return_x_flux=True)
    stored = capacity*dx*(updated-field).mean(dim=0)
    torch.testing.assert_close(stored,dt*(dx*source.mean(dim=0)-flux.diff()),atol=3e-13,rtol=0.)


@pytest.mark.parametrize("harmonic",[False, True])
def test_two_row_flux_walls_recover_transverse_polynomial_solution(harmonic):
    from tensorsph.heat_conduction_2d import solve_harmonic_slab_2d, solve_steady_slab_2d
    n,k,c,omega = 16,1.7,2.3,5.
    x = (torch.arange(n,dtype=torch.float64)+.5)/n
    beta = 2*torch.pi
    mode = torch.exp(1j*beta*x) if harmonic else torch.cos(beta*x)
    linear = .4+mode[:,None]
    quadratic = -.2+.3*mode[:,None]
    exact = 2.+linear*x+quadratic*x.square()
    west = -k*linear[:,0]
    east = k*(linear+2*quadratic)[:,0]
    source = -2*k*quadratic+k*beta**2*mode[:,None]*(x+.3*x.square())
    opts = dict(conductivity=k,x_wall_inward_flux=(west,east),return_x_flux=True)
    if harmonic:
        field,flux = solve_harmonic_slab_2d(source+1j*omega*c*exact,1/n,(None,None),omega,
                                            volumetric_heat_capacity=c,**opts)
    else:
        field,flux = solve_steady_slab_2d(source,1/n,(None,None),mean_temperature=float(exact.mean()),**opts)
    torch.testing.assert_close(field,exact,atol=2e-12,rtol=0.)
    faces = torch.arange(n+1,dtype=torch.float64)/n
    torch.testing.assert_close(flux,(-k*(.4-.4*faces)).to(flux.dtype),atol=2e-12,rtol=0.)


@pytest.mark.parametrize("damping",[False,True])
def test_wall_jumps_return_requested_fields_and_heat(damping):
    from tensorsph.heat_conduction_2d import solve_transient_slab_2d,step_transient_slab_2d
    initial = torch.arange(24,dtype=torch.float64).reshape(4,6)/24
    profile = torch.linspace(.2,1.,4,dtype=initial.dtype)
    jumps = (.015,.03,.08)
    levels = (1.,-.4,.2,1.3)
    def west(t):
        return levels[sum(t>=j for j in jumps)]*profile+t
    opts = dict(volumetric_heat_capacity=2.,x_wall_inward_flux=(0.,.2))
    fields,heat = solve_transient_slab_2d(initial,[0.,.08,.1],.2,(west,None),wall_jump_times=jumps,
                                          rannacher=damping,return_x_heat=True,**opts)
    current = initial
    total = torch.zeros_like(heat)
    for a,b,lvl in zip((0.,*jumps),(*jumps,.1),levels):
        subgrid = (a,(a+b)/2,b) if damping else (a,b)
        for lo,hi in zip(subgrid,subgrid[1:]):
            value = lvl*profile+(hi if damping else (lo+hi)/2)
            current,flux = step_transient_slab_2d(current,hi-lo,.2,(value,None),
                                      theta=1. if damping else .5,return_x_flux=True,**opts)
            total[int(a>=.08)] += (hi-lo)*flux
        if b==.08:
            torch.testing.assert_close(current,fields[1])
    torch.testing.assert_close(current,fields[-1])
    torch.testing.assert_close(heat,total)
    torch.testing.assert_close(.4*fields.diff(dim=0).mean(dim=1),-heat.diff(dim=1),atol=1e-12,rtol=0.)


@pytest.mark.parametrize("damping",[False,True])
def test_transient_load_jump_preserves_net_heat(damping):
    from tensorsph.heat_conduction_2d import solve_transient_slab_2d
    initial = torch.zeros((4,8),dtype=torch.float64)
    start, stop = .013, .046
    def pulse(t):
        return float(start<=t<stop)
    field,heat = solve_transient_slab_2d(initial,[0.,.03,.08],.125,(None,None),
                    heat_source=lambda t: 3*pulse(t), x_wall_inward_flux=(pulse,lambda t: 2*pulse(t)),
                    wall_jump_times=(start,stop),rannacher=damping,return_x_heat=True)
    durations = initial.new_tensor([.03-start,stop-.03])
    torch.testing.assert_close(heat[:,0],durations,atol=1e-12,rtol=0.)
    torch.testing.assert_close(heat[:,-1],-2*durations,atol=1e-12,rtol=0.)
    torch.testing.assert_close(field.diff(dim=0).mean(dim=(1,2)),6*durations,atol=1e-12,rtol=0.)


def test_volume_loss_equilibrates_two_insulated_walls():
    from tensorsph.heat_conduction_2d import solve_steady_slab_2d, solve_transient_slab_2d
    source = torch.full((5,8),6.,dtype=torch.float64)
    opts = dict(dx=.125,x_wall_temperatures=(None,None),heat_loss_coefficient=3.)
    steady,flux = solve_steady_slab_2d(source,return_x_flux=True,**opts)
    torch.testing.assert_close(steady,torch.full_like(steady,2.))
    torch.testing.assert_close(flux,torch.zeros_like(flux))
    times = [0.,.1,.35]
    states = solve_transient_slab_2d(source*0,times,heat_source=source,volumetric_heat_capacity=2.,**opts)
    predicted = 0.
    for i,dt in enumerate((.1,.25),1):
        predicted = ((2-.5*3*dt)*predicted+6*dt)/(2+.5*3*dt)
        torch.testing.assert_close(states[i],torch.full_like(steady,predicted))


@pytest.mark.parametrize("loss",[-1.,float("nan"),float("inf")])
def test_invalid_bulk_heat_loss(loss):
    from tensorsph.heat_conduction_2d import solve_steady_slab_2d
    with pytest.raises(ValueError,match="heat_loss_coefficient"):
        solve_steady_slab_2d(torch.ones(4,8),.125,(1.,None),heat_loss_coefficient=loss)


@pytest.mark.parametrize("backend",["banded","torch"])
def test_layered_heat_loss_in_grid_and_film_solvers(backend):
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_films, solve_harmonic_layered_slab
    n = 12
    loss = torch.linspace(0.,7.,n,dtype=torch.float64)
    target = .6+.8j
    source = target*(loss+3j)
    opts = dict(conductivity=torch.ones_like(loss),volumetric_heat_capacity=1.,heat_loss_coefficient=loss,backend=backend)
    grid = solve_harmonic_layered_slab(source.expand(5,n),1/n,(target,None),3.,**opts)
    torch.testing.assert_close(grid,torch.full_like(grid,target))
    slab,walls,flux = solve_harmonic_layered_films(source,1/n,(target,target),3.,
                                                   film_coefficients=(2.,3.),**opts)
    torch.testing.assert_close(slab,torch.full_like(slab,target))
    torch.testing.assert_close(walls,torch.full_like(walls,target))
    torch.testing.assert_close(flux,torch.zeros_like(flux),atol=1e-12,rtol=0.)


@pytest.mark.parametrize("theta",[.5,1.])
def test_implicit_layered_slab_satisfies_local_particle_balance(theta):
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab, stencil_rhs_2d, lattice_diffusion_factor_2d
    n,dx = 10,.1
    x = (torch.arange(n,dtype=torch.float64)+.5)*dx
    times = [0.,.03,.08,.12]
    capacity,k,loss = 1.+x,2.+(x>.5),.5+x
    contact = torch.zeros_like(x[:-1])
    contact[n//2-1] = .23
    opts = dict(conductivity=k,volumetric_heat_capacity=capacity,heat_loss_coefficient=loss,
                contact_resistance_x=contact,heat_source=x.square(),theta=theta)
    fields = solve_transient_layered_slab(x,times,dx,(1.3,None),**opts)
    restarted = solve_transient_layered_slab(fields[1],times[1:],dx,(1.3,None),**opts)
    torch.testing.assert_close(fields[1:],restarted)
    offsets = lattice_offsets(dx,1.3*dx)
    factor = lattice_diffusion_factor_2d(1.3,dx,mode="moment")
    for i,(start,stop) in enumerate(zip(times,times[1:])):
        stage = (1-theta)*fields[i]+theta*fields[i+1]
        rate = stencil_rhs_2d(stage[None,:],offsets,x_wall_temperatures=(1.3,None),dx=dx,
                              conductivity=k,layered_x=True,contact_resistance_x=contact)[0]/factor
        torch.testing.assert_close(capacity*(fields[i+1]-fields[i]),(stop-start)*(rate+x.square()-loss*stage),
                                   rtol=0.,atol=2e-14)


def test_layered_transient_jump_matches_manual_restart():
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab as solve
    x = torch.linspace(.1,.9,8,dtype=torch.float64)
    opts = dict(conductivity=1+x,volumetric_heat_capacity=2-x)
    jumps = [.023,.061]
    left = lambda t: 1. if t<jumps[0] else -2.
    heating = lambda t: x if t<jumps[1] else 2-x
    history = solve(x,[0.,.05,.1],.125,(left,None),heat_source=heating,load_jump_times=jumps,**opts)
    a = solve(x,[0.,.023],.125,(1.,None),heat_source=x,**opts)
    b = solve(a[-1],[.023,.05,.061],.125,(-2.,None),heat_source=x,**opts)
    c = solve(b[-1],[.061,.1],.125,(-2.,None),heat_source=2-x,**opts)
    torch.testing.assert_close(history,torch.stack([x,b[1],c[-1]]))


@pytest.mark.parametrize("jumps",[[0.],[.1],[.03,.03],[float("nan")]])
def test_layered_transient_rejects_invalid_events(jumps):
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab
    x = torch.ones(4)
    with pytest.raises(ValueError,match="load_jump_times"):
        solve_transient_layered_slab(x,[0.,.1],.25,(None,None),conductivity=x,
                                    volumetric_heat_capacity=1.,load_jump_times=jumps)


def test_layered_interval_heat_matches_interior_pair_tallies_across_jump():
    from tensorsph.heat_conduction_2d import (
        lattice_diffusion_factor_2d, solve_transient_layered_slab, stencil_x_face_power_2d,
        stencil_rhs_with_wall_power_2d,
    )
    x = torch.arange(8,dtype=torch.float64)/8
    dx,jump = .125,.021
    k = 1.+2*x
    resistance = torch.zeros_like(x[:-1])
    resistance[3] = .18
    west = lambda t: .3+t if t<jump else 1.2+t
    load = lambda t: x+t
    opts = dict(conductivity=k,contact_resistance_x=resistance,volumetric_heat_capacity=1+x,heat_source=load)
    _,heat = solve_transient_layered_slab(x,[0.,.06],dx,(west,None),return_x_heat=True,
                                         load_jump_times=[jump],**opts)
    times = [0.,jump,.06]
    states = solve_transient_layered_slab(x,times,dx,(west,None),load_jump_times=[jump],**opts)
    offsets = lattice_offsets(dx,1.3*dx)
    factor = lattice_diffusion_factor_2d(1.3,dx,mode="moment")
    expected = torch.zeros_like(heat[0])
    for i,(a,b) in enumerate(zip(times,times[1:])):
        stage = .5*(states[i]+states[i+1])[None,:]
        wall = .3+.5*(a+b) if i==0 else 1.2+.5*(a+b)
        kw = dict(conductivity=k,layered_x=True,contact_resistance_x=resistance)
        _,powers = stencil_rhs_with_wall_power_2d(stage,offsets,dx,(wall,None),**kw)
        cuts = [stencil_x_face_power_2d(stage,offsets,dx,j,(wall,None),**kw) for j in range(1,len(x))]
        expected += (b-a)*torch.stack([powers[0],*cuts,-powers[1]])/(dx*factor)
    torch.testing.assert_close(heat[0],expected,rtol=0.,atol=5e-15)


@pytest.mark.parametrize("theta",[.5,1.])
def test_layered_flux_jump_reverses_sign_and_balances_each_column(theta):
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab
    x = torch.arange(8,dtype=torch.float64)/8
    dx,jump = .125,.031
    times = torch.tensor([0.,.05,.12],dtype=x.dtype)
    west = lambda t: 1.4 if t<jump else -.6
    east = lambda t: .2 if t<jump else .7
    states,heat = solve_transient_layered_slab(x,times,dx,(None,None),volumetric_heat_capacity=1+x,
            conductivity=2+x,heat_source=.1,theta=theta,load_jump_times=[jump],
            x_wall_inward_flux=(west,east),return_x_heat=True)
    before = times.clamp(max=jump).diff()
    after = (times-jump).clamp(min=0.).diff()
    torch.testing.assert_close(heat[:,0],1.4*before-.6*after)
    torch.testing.assert_close(heat[:,-1],-.2*before-.7*after)
    torch.testing.assert_close(dx*(1+x)*states.diff(dim=0)+heat.diff(dim=1),
                              (.1*dx*times.diff()[:,None]).expand_as(states[:-1]),rtol=0.,atol=2e-14)


@pytest.mark.parametrize("flux",[(0.,),(1.,0.),(0.,complex(0,1)),(0.,lambda t:float("nan"))])
def test_layered_invalid_wall_flux(flux):
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab
    x = torch.ones(4)
    with pytest.raises(ValueError,match="flux"):
        solve_transient_layered_slab(x,[0.,.1],.25,(1.,None),conductivity=x,volumetric_heat_capacity=1.,
                                    x_wall_inward_flux=flux)


def test_layered_flux_wall_with_contacts_inside_kernel_support():
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab
    dx = .1
    k = torch.tensor([1.,4.,2.,.7,8.,1.,2.,6.,.9,3.],dtype=torch.float64)
    contacts = k[:-1]/10
    dr = dx/k
    r = dr.cumsum(0)-dr/2+torch.cat((k.new_zeros(1),contacts.cumsum(0)))
    initial = 2.-1.7*r
    times = k.new_tensor([0.,.02,.19])
    fields,heat = solve_transient_layered_slab(initial,times,dx,(None,None),conductivity=k,
                volumetric_heat_capacity=1.+k,contact_resistance_x=contacts,h_factor=2.1,
                x_wall_inward_flux=(1.7,-1.7),return_x_heat=True)
    torch.testing.assert_close(fields,initial.expand_as(fields),atol=1e-13,rtol=0.)
    torch.testing.assert_close(heat,(1.7*times.diff()[:,None]).expand_as(heat),atol=1e-13,rtol=0.)


def test_layered_rannacher_heat_matches_explicit_half_steps():
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab as solve
    k = torch.linspace(1.,3.,8,dtype=torch.float64)
    opts = dict(conductivity=k,volumetric_heat_capacity=k+1,return_x_heat=True)
    history,heat = solve(k,[0.,.08,.12],.125,(None,None),load_jump_times=[.03],rannacher=True,
                         x_wall_inward_flux=(lambda t:1. if t<.03 else -.3,0.),**opts)
    a,qa = solve(k,[0.,.015,.03],.125,(None,None),theta=1.,x_wall_inward_flux=(1.,0.),**opts)
    b,qb = solve(a[-1],[.03,.055,.08],.125,(None,None),theta=1.,x_wall_inward_flux=(-.3,0.),**opts)
    c,qc = solve(b[-1],[.08,.12],.125,(None,None),x_wall_inward_flux=(-.3,0.),**opts)
    torch.testing.assert_close(history,torch.stack([k,b[-1],c[-1]]))
    torch.testing.assert_close(heat,torch.stack([qa.sum(0)+qb.sum(0),qc[0]]))


def test_layered_steady_contact_with_two_transient_films():
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab
    dx = .125
    k = torch.tensor([1.,3.,2.,.6,1.,.7,4.,5.],dtype=torch.float64)
    contacts = .03*k[:-1]
    h0,h1 = 3.,7.
    dr = dx/k
    q = (4.-1.)/(1/h0+dr.sum()+contacts.sum()+1/h1)
    r = dr.cumsum(0)-dr/2+torch.cat((k.new_zeros(1),contacts.cumsum(0)))
    initial = 4.-q*(1/h0+r)
    times = k.new_tensor([0.,.07,.23])
    fields,heat = solve_transient_layered_slab(initial,times,dx,(None,None),conductivity=k,
            volumetric_heat_capacity=k+1,contact_resistance_x=contacts,x_wall_films=((h0,4.),(h1,1.)),
            return_x_heat=True,h_factor=2.1)
    torch.testing.assert_close(fields,initial.expand_as(fields),rtol=0.,atol=1e-13)
    torch.testing.assert_close(heat,(q*times.diff()[:,None]).expand_as(heat),rtol=0.,atol=1e-13)


@pytest.mark.parametrize("wall,flux,film",[(1.,0.,(1.,2.)),(None,float("nan"),(1.,2.)),(None,0.,(-1.,2.)),
                                        (None,0.,(2.,float("nan")))])
def test_invalid_layered_film(wall,flux,film):
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab
    k = torch.ones(5)
    with pytest.raises(ValueError):
        solve_transient_layered_slab(k,[0.,.1],.2,(wall,None),conductivity=k,volumetric_heat_capacity=1.,
                                    x_wall_inward_flux=(flux,0.),x_wall_films=(film,None))


@pytest.mark.parametrize("with_heat",[True,False])
def test_layer_wall_trace_after_ambient_and_flux_jump(with_heat):
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab
    dx,jump = .125,.04
    k = torch.arange(1,9,dtype=torch.float64)
    t = k.new_tensor([0.,jump,.09])
    west = lambda s: 1. if s<jump else -2.
    ambient = lambda s: .5 if s<jump else 3.
    result = solve_transient_layered_slab(k,t,dx,(None,None),conductivity=k,volumetric_heat_capacity=k+1,
                    x_wall_films=(None,(4.,ambient)),x_wall_inward_flux=(west,0.),
                    load_jump_times=[jump],return_x_heat=with_heat,return_wall_temperatures=True)
    u,walls = result[0],result[-1]
    assert len(result)==2+int(with_heat)
    torch.testing.assert_close(walls[:,0],u[:,0]+dx/(2*k[0])*k.new_tensor([1.,-2.,-2.]))
    inward = 4.*(k.new_tensor([.5,3.,3.])-walls[:,1])
    torch.testing.assert_close(walls[:,1]-u[:,-1],inward*dx/(2*k[-1]))
    if with_heat:
        torch.testing.assert_close(result[1][:,0],t.diff()*k.new_tensor([1.,-2.]))


def test_temperature_and_insulated_layer_wall_traces():
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab
    k = torch.ones(6,dtype=torch.float64)
    t = k.new_tensor([0.,.03,.1])
    u,walls = solve_transient_layered_slab(k,t,1/6,(lambda s:.3+s,None),conductivity=k,
                    volumetric_heat_capacity=1.,return_wall_temperatures=True)
    torch.testing.assert_close(walls[:,0],.3+t)
    torch.testing.assert_close(walls[:,1],u[:,-1])


def test_low_frequency_films_uniform_exact_storage_response():
    from tensorsph.heat_conduction_2d import solve_low_frequency_layered_films

    drive = .7+.4j
    tau = .35
    loss = torch.linspace(.8,2.,12,dtype=torch.float64)
    films = torch.tensor([2.,5.],dtype=torch.float64)
    for backend in ("torch","banded"):
        steady,lag = solve_low_frequency_layered_films(
            loss*drive,1/12,(drive,drive),film_coefficients=films,
            wall_areal_heat_capacity=tau*films,conductivity=2.+loss,
            volumetric_heat_capacity=tau*loss,heat_loss_coefficient=loss,backend=backend)
        for (field,walls,flux),value in ((steady,drive),(lag,-tau*drive)):
            assert (field-value).abs().max()<1e-12
            assert (walls-value).abs().max()<1e-12
            assert flux.abs().max()<1e-12


def test_low_frequency_slab_fixed_drive_insulated_end():
    from tensorsph.heat_conduction_2d import solve_low_frequency_layered_slab

    n = 32
    x = (torch.arange(n,dtype=torch.float64)+.5)/n
    drive = .8-.3j
    for backend in ("torch","banded"):
        steady,lag = solve_low_frequency_layered_slab(
            torch.zeros(n),1/n,(drive,None),conductivity=torch.ones(n)*2,
            volumetric_heat_capacity=3.,backend=backend)
        # k*T1'' = C*T0, with T1(0)=0 and T1'(1)=0.
        exact = 1.5*drive*(x*x/2-x)
        assert (steady-drive).abs().max()<1e-12
        assert (lag-exact).abs().max()<2/n**2


def test_steady_insulated_transverse_mode_and_aliased_constant():
    import math
    from tensorsph.heat_conduction_2d import lattice_diffusion_factor_2d, solve_low_frequency_layered_slab

    n = 24
    dx = 1/n
    wave = -2*math.pi
    x = (torch.arange(n,dtype=torch.float64)+.5)*dx
    source = (.6+.2j)*torch.cos(math.pi*x)
    rate = 2*symbol_rate(dx,1.3*dx,math.pi,wave)/lattice_diffusion_factor_2d(1.3,dx,mode="moment")
    for backend in ("torch","banded"):
        options = dict(conductivity=torch.full((n,),2.),volumetric_heat_capacity=3.,backend=backend)
        steady,lag = solve_low_frequency_layered_slab(source,dx,(None,None),transverse_wavenumber=wave,**options)
        torch.testing.assert_close(steady,source/rate,atol=1e-12,rtol=1e-10)
        torch.testing.assert_close(lag,-3*source/rate**2,atol=1e-12,rtol=1e-10)
        for alias in (0.,math.tau/dx):
            rate_dc = 2*symbol_rate(dx,1.3*dx,math.pi,0.)/lattice_diffusion_factor_2d(1.3,dx,mode="moment")
            dc,dc_lag = solve_low_frequency_layered_slab(source,dx,(None,None),transverse_wavenumber=alias,**options)
            torch.testing.assert_close(dc,source/rate_dc,atol=1e-12,rtol=1e-10)
            torch.testing.assert_close(dc_lag,-3*source/rate_dc**2,atol=1e-12,rtol=1e-10)
            with pytest.raises(ValueError,match="zero net source"):
                solve_low_frequency_layered_slab(source+1,dx,(None,None),transverse_wavenumber=alias,**options)


def test_closed_slab_nonuniform_capacity_sets_energy_mean():
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_slab, solve_low_frequency_layered_slab

    n = 16
    x = (torch.arange(n,dtype=torch.float64)+.5)/n
    capacity = 1+x
    source = (.5+.3j)*(x-.5)
    opts = dict(conductivity=2.+x,volumetric_heat_capacity=capacity,backend="banded")
    steady,lag = solve_low_frequency_layered_slab(source,1/n,(None,None),**opts)
    assert abs((capacity*steady).sum())<1e-13
    assert abs((capacity*lag).sum())<1e-13
    omega = .02
    finite = solve_harmonic_layered_slab(source,1/n,(None,None),omega,**opts)
    torch.testing.assert_close(finite,steady+1j*omega*lag,rtol=0.,atol=3e-8)


def test_sdirk2_stiff_layered_mode_and_internal_cut_balance():
    import math
    from tensorsph.heat_conduction_2d import lattice_diffusion_factor_2d, solve_transient_layered_slab

    n = 16
    dx = 1/n
    wave = 7*math.pi
    initial = torch.sin(wave*(torch.arange(n,dtype=torch.float64)+.5)*dx)
    rate = symbol_rate(dx,1.3*dx,wave,0.)/lattice_diffusion_factor_2d(1.3,dx,mode="moment")
    u,heat = solve_transient_layered_slab(initial,[0.,.1,10.],dx,(0.,0.),conductivity=torch.ones(n),
                                          volumetric_heat_capacity=1.,method="sdirk2",return_x_heat=True)
    gamma = 1-1/math.sqrt(2)
    exact = initial
    for state,dt in zip(u[1:],(.1,9.9)):
        z = rate*dt
        exact = exact*((1-(1-2*gamma)*z)/(1+gamma*z)**2)
        torch.testing.assert_close(state,exact,atol=2e-12,rtol=0.)
    assert (dx*u.diff(dim=0)+heat.diff(dim=1)).abs().max()<1e-12


def test_sdirk2_step_straddles_flux_pulse_and_closes_film_balance():
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    n = 8
    dx = 1/n
    u,heat = solve_transient_layered_slab(torch.zeros(n),[0.,.1,.2],dx,(None,None),
            conductivity=torch.ones(n),volumetric_heat_capacity=2.,method="sdirk2",return_x_heat=True,
            x_wall_films=(None,(2.,.3)),x_wall_inward_flux=(lambda t:(3+t) if .03<=t<.04 else 0.,0.),
            load_jump_times=(.03,.04))
    assert abs(heat[0,0]-.01*(3+.035))<1e-13
    assert abs(heat[1,0])<1e-13
    assert (2*dx*u.diff(dim=0)+heat.diff(dim=1)).abs().max()<1e-12


def test_sdirk2_periodic_rows_resolve_transverse_mode_and_storing_columns():
    import math
    from tensorsph.heat_conduction_2d import solve_transient_slab_2d, lattice_diffusion_factor_2d

    n = 12
    dx = 1/n
    x = (torch.arange(n,dtype=torch.float64)+.5)*dx
    # The row-constant component makes section heat appreciable. Each
    # transverse component has its own discrete Fourier decay rate.
    base = torch.sin(math.pi*x)
    row = .3*torch.cos(2*math.pi*x)[:,None]
    u,heat = solve_transient_slab_2d(base*(1+row),[0.,.02],dx,(0.,0.),method="sdirk2",return_x_heat=True)
    gamma = 1-1/math.sqrt(2)
    def gain(w):
        z = .02*symbol_rate(dx,1.3*dx,math.pi,w)/lattice_diffusion_factor_2d(1.3,dx,mode="moment")
        return (1-(1-2*gamma)*z)/(1+gamma*z)**2
    torch.testing.assert_close(u[-1],base*(gain(0.)+row*gain(2*math.pi)),rtol=0.,atol=1e-12)
    assert (dx*u.diff(dim=0).mean(1)+heat.diff(dim=1)).abs().max()<1e-12


@pytest.mark.parametrize("method,rannacher", [("theta",False),("theta",True),("sdirk2",False)])
def test_layered_bulk_reservoir_jumps_preserve_separate_exchange(method,rannacher):
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    n,dx,jump = 8,.125,.083
    times = torch.linspace(0.,.2,21,dtype=torch.float64)
    c = torch.linspace(.7,2.,n,dtype=torch.float64)
    u,section,exchange = solve_transient_layered_slab(torch.zeros_like(c),times,dx,(None,None),
        conductivity=torch.ones_like(c),volumetric_heat_capacity=c,method=method,rannacher=rannacher,
        heat_loss_coefficient=4*c,bulk_ambient_temperature=lambda t: 2. if t<jump else -.5,
        load_jump_times=[jump],return_x_heat=True,return_bulk_exchange=True)
    first = -2*torch.expm1(-4*times.clamp(max=jump))
    exact = torch.where(times<=jump,first,-.5+(first+.5)*torch.exp(-4*(times-jump)))
    torch.testing.assert_close(u,exact[:,None].expand_as(u),atol=2e-3,rtol=0.)
    assert (c*dx*u.diff(dim=0)+section.diff(dim=1)-exchange).abs().max()<1e-12


@pytest.mark.parametrize("method,rannacher", [("theta",False),("theta",True),("sdirk2",False)])
def test_storing_films_and_switched_flux_balance_both_walls(method,rannacher):
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    n,dx,jump = 10,.1,.063
    times = [0.,.03,.11,.18]
    cw = torch.tensor([.3,.7],dtype=torch.float64)
    def run(west):
        return solve_transient_layered_slab(torch.zeros(n),times,dx,(None,None),
            conductivity=torch.ones(n),volumetric_heat_capacity=2.,x_wall_films=((2.,1.),None),
            x_wall_inward_flux=(.4,lambda t: 1. if t<jump else -.5),wall_areal_heat_capacity=cw,
            initial_wall_temperatures=(west,-.2),load_jump_times=[jump],return_external_heat=True,
            return_x_heat=True,return_wall_temperatures=True,method=method,rannacher=rannacher)
    u,q,w,e = run(.3)
    torch.testing.assert_close(w[0],cw.new_tensor([.3,-.2]))
    assert (2*dx*u.diff(dim=0)+q.diff(dim=1)).abs().max()<1e-12
    assert (cw*w.diff(dim=0)+q[:,[0,-1]]*cw.new_tensor([1.,-1.])-e).abs().max()<1e-12
    assert abs(e[:,1].sum().item()-(jump-.5*(times[-1]-jump)))<1e-13


@pytest.mark.parametrize("cw", [(0.,.6),(.3,0.)])
def test_one_storing_film_preserves_layered_steady_contact_flux(cw):
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    n,dx,j = 12,1/12,.7
    k = torch.tensor([2.]*(n//2)+[.8]*(n//2),dtype=torch.float64)
    contacts = torch.zeros(n-1,dtype=k.dtype)
    contacts[n//2-1] = .15
    r = dx/k
    cells = 2.-j*(r.cumsum(0)-r/2+torch.cat((k.new_zeros(1),contacts.cumsum(0))))
    right = 2.-j*(r.sum()+contacts.sum())
    surface = (2.,right.item())
    initial = tuple(t if cap else None for t,cap in zip(surface,cw))
    times = [0.,.013,.06]
    u,q,w,external = solve_transient_layered_slab(cells,times,dx,(None,None),conductivity=k,
        volumetric_heat_capacity=1.,contact_resistance_x=contacts,method="sdirk2",h_factor=1.3,
        x_wall_films=((3.,2.+j/3),(1.4,right-j/1.4)),wall_areal_heat_capacity=cw,
        initial_wall_temperatures=initial,return_x_heat=True,return_wall_temperatures=True,return_external_heat=True)
    dt = k.new_tensor(times).diff()
    torch.testing.assert_close(u,cells.expand_as(u),atol=1e-12,rtol=0.)
    torch.testing.assert_close(w,k.new_tensor(surface).expand_as(w),atol=1e-12,rtol=0.)
    torch.testing.assert_close(q,(dt[:,None]*j).expand_as(q),atol=1e-12,rtol=0.)
    torch.testing.assert_close(external,dt[:,None]*k.new_tensor([j,-j]),atol=1e-12,rtol=0.)


def test_exponential_affine_reservoir_heat_and_zero_loss_limit():
    import numpy as np
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    c,h,dx = 1.7,4.1,.125
    times = np.array([0.,.031,.19,.42])
    initial,base,slope = -.2,.7,1.3
    for loss in (0.,h):
        u,section,bulk = solve_transient_layered_slab(np.full(8,initial),times,dx,(None,None),
            conductivity=np.ones(8),volumetric_heat_capacity=c,method="exponential",
            heat_source=lambda t: c*(.3+.2*t),heat_loss_coefficient=loss,
            bulk_ambient_temperature=lambda t: base+slope*t,return_x_heat=True,return_bulk_exchange=True)
        if loss:
            rate = loss/c
            constant,linear = rate*base+.3,rate*slope+.2
            exact = initial+(constant/rate-initial)*(-np.expm1(-rate*times))+linear/rate*(times+np.expm1(-rate*times)/rate)
        else:
            exact = initial+.3*times+.1*times**2
        supplied = c*dx*np.diff(.3*times+.1*times**2)
        np.testing.assert_allclose(u.numpy(),np.broadcast_to(exact[:,None],u.shape),atol=2e-13,rtol=0.)
        target = dx*c*np.diff(exact)-supplied
        np.testing.assert_allclose(bulk.numpy(),np.broadcast_to(target[:,None],bulk.shape),atol=2e-13,rtol=0.)
        assert section.abs().max()<1e-13


def test_exponential_quadratic_tracks_manufactured_bulk_and_storing_wall_heat():
    import numpy as np
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    # Uniform T=t(1-t) makes conduction vanish at every cut. Bulk exchange
    # supplies material storage and the external film supplies wall storage.
    c,h,dx,cw,film = 1.7,2.3,.25,.43,3.1
    temperature = lambda t: t*(1-t)
    rate = lambda t: 1-2*t
    air = lambda t: temperature(t)+cw*rate(t)/film
    for times in (np.array([0.,1.]),np.array([0.,.09,.5,.83,1.])):
        u,q,w,bulk,external = solve_transient_layered_slab(np.zeros(4),times,dx,(temperature,None),
            conductivity=np.ones(4),volumetric_heat_capacity=c,method="exponential_quadratic",
            heat_loss_coefficient=h,bulk_ambient_temperature=lambda t: temperature(t)+c*rate(t)/h,
            x_wall_films=(None,(film,air)),wall_areal_heat_capacity=(0.,cw),initial_wall_temperatures=(None,0.),
            return_x_heat=True,return_wall_temperatures=True,return_bulk_exchange=True,return_external_heat=True)
        target = temperature(times)
        for field in (u,w):
            np.testing.assert_allclose(field,np.broadcast_to(target[:,None],field.shape),atol=3e-13,rtol=0.)
        np.testing.assert_allclose(q,0.,atol=3e-13)
        np.testing.assert_allclose(bulk,np.broadcast_to(c*dx*np.diff(target)[:,None],bulk.shape),atol=3e-13)
        np.testing.assert_allclose(external,np.column_stack((np.zeros(len(times)-1),cw*np.diff(target))),atol=3e-13)


def test_interval_particle_temperature_across_internal_jump():
    import numpy as np
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    times = np.array([0.,.2,.7])
    jump = .13
    # Piecewise constant heating gives piecewise linear temperature.
    exact = lambda t: 1+2*t-3*np.maximum(t-jump,0.)
    primitive = lambda t: t+t*t-1.5*np.maximum(t-jump,0.)**2
    kwargs = dict(conductivity=np.ones(3),volumetric_heat_capacity=1.,
                  heat_source=lambda t: 2. if t<jump else -1.,load_jump_times=[jump],
                  return_interval_temperature=True)
    for method in ("theta","sdirk2","exponential","exponential_quadratic"):
        u,mean = solve_transient_layered_slab(np.ones(3),times,1/3,(None,None),method=method,**kwargs)
        np.testing.assert_allclose(u,np.broadcast_to(exact(times)[:,None],u.shape),rtol=0.,atol=2e-13)
        target = np.diff(primitive(times))/np.diff(times)
        np.testing.assert_allclose(mean,np.broadcast_to(target[:,None],mean.shape),rtol=0.,atol=2e-13)


def test_interval_wall_temperature_uses_one_sided_quadratic_loads():
    import numpy as np
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    times = np.array([0., .2, .7])
    jump = .13
    wall = lambda t: 2+t*t+(3. if t >= jump else 0.)
    primitive = lambda t: 2*t+t**3/3+3*np.maximum(t-jump, 0.)
    _, mean = solve_transient_layered_slab(
        np.ones(4), times, .25, (wall, 1.), conductivity=np.ones(4),
        volumetric_heat_capacity=1., load_jump_times=[jump],
        method="exponential_quadratic", return_interval_wall_temperatures=True)
    exact = np.diff(primitive(times))/np.diff(times)
    np.testing.assert_allclose(mean[:,0], exact, rtol=0., atol=1e-13)
    np.testing.assert_allclose(mean[:,1], 1., rtol=0., atol=1e-13)


def test_layered_slab_temperature_traces_exact_for_piecewise_quadratics():
    from tensorsph.heat_conduction_2d import layered_slab_contact_temperatures

    dx = .2
    k = torch.tensor([1.1,2.6,.9,1.4],dtype=torch.float64)
    west = k.new_tensor([.5,.2,1.2,-.4])
    flux = k.new_tensor([.7,-.1,.5,.9,-.3])
    # Integrate the prescribed linear flux from each west face to the center
    # and east face. The column offsets intentionally permit independent jumps.
    center = west-dx*(flux[:-1]+.25*flux.diff())/(2*k)
    east = west-dx*(flux[:-1]+flux[1:])/(2*k)
    phase = k.new_tensor([1.,-.5])[:,None]*(1+.4j)
    left,right = layered_slab_contact_temperatures(phase*center,phase*flux,dx,k)
    torch.testing.assert_close(left,phase*east[:-1],rtol=0.,atol=2e-15)
    torch.testing.assert_close(right,phase*west[1:],rtol=0.,atol=2e-15)
    with pytest.raises(ValueError,match="flux"):
        layered_slab_contact_temperatures(center,flux[:-1],dx,k)


@pytest.mark.parametrize("backend", ["torch","banded"])
def test_harmonic_layered_section_flux_matches_pair_tallies(backend):
    from tensorsph.heat_conduction_2d import (
        lattice_offsets, lattice_diffusion_factor_2d, solve_harmonic_layered_slab,
        stencil_x_face_power_2d, stencil_rhs_with_wall_power_2d,
    )

    nx,ny,dx = 8,4,.125
    x = torch.arange(nx,dtype=torch.float64)
    y = torch.arange(ny,dtype=torch.float64)
    source = (.3+.2j)*(x+.4*y[:,None])
    walls = ((.8-.4j)*(1+.2*y),None)
    k = torch.where(x<3,1.2,2.7).double()
    r = torch.zeros(nx-1,dtype=k.dtype)
    r[2] = .19
    opts = dict(conductivity=k,contact_resistance_x=r)
    u,q = solve_harmonic_layered_slab(source,dx,walls,3.,volumetric_heat_capacity=1.7,
                                      backend=backend,return_x_flux=True,**opts)
    offsets = lattice_offsets(dx,1.3*dx)
    scale = ny*dx*lattice_diffusion_factor_2d(1.3,dx,mode="moment")
    _,ends = stencil_rhs_with_wall_power_2d(u,offsets,dx,walls,layered_x=True,**opts)
    cuts = [stencil_x_face_power_2d(u,offsets,dx,i,walls,layered_x=True,**opts) for i in range(1,nx)]
    expected = torch.stack([ends[0],*cuts,-ends[1]])/scale
    torch.testing.assert_close(q,expected,atol=1e-13,rtol=0.)


@pytest.mark.parametrize("mode",[-2,0,1,4])
def test_modal_cut_flux_matches_explicit_periodic_rows(mode):
    from tensorsph.heat_conduction_2d import (
        lattice_offsets,lattice_diffusion_factor_2d,layered_slab_modal_cut_flux,stencil_x_face_power_2d,
        stencil_rhs_with_wall_power_2d,
    )

    n,dx = 8,.125
    beta = 2*torch.pi*mode
    x = (torch.arange(n,dtype=torch.float64)+.5)*dx
    phase = torch.exp(1j*beta*x)
    field = (.7+.2j)*torch.cos(2*x)+.3j*x
    walls = (.3-.4j,None)
    k = torch.where(x<.5,1.3,2.1).double()
    resistance = torch.zeros(n-1,dtype=x.dtype)
    resistance[3] = .27
    options = dict(conductivity=k,contact_resistance_x=resistance)
    _, ends = stencil_rhs_with_wall_power_2d(phase[:,None]*field,lattice_offsets(dx,1.3*dx),
            dx,(phase*walls[0],None),layered_x=True,power_by_row=True,**options)
    for face, power in ((0,ends[0]),(n,-ends[1])):
        flux = layered_slab_modal_cut_flux(field,dx,face,walls,transverse_wavenumber=beta,**options)
        expected = power/(dx*lattice_diffusion_factor_2d(1.3,dx,mode="moment"))
        torch.testing.assert_close(phase*flux,expected,rtol=1e-13,atol=1e-13)
    # Include cuts near ghost particles as well as the resistive material boundary.
    for cut in (1,4,7):
        modal = layered_slab_modal_cut_flux(field,dx,cut,walls,transverse_wavenumber=beta,**options)
        row = stencil_x_face_power_2d(phase[:,None]*field,lattice_offsets(dx,1.3*dx),dx,cut,
                (phase*walls[0],None),layered_x=True,power_by_row=True,**options)
        expected = row/(dx*lattice_diffusion_factor_2d(1.3,dx,mode="moment"))
        torch.testing.assert_close(phase*modal,expected,rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize("mode",[-2,1])
@pytest.mark.parametrize("backend",["torch","banded"])
def test_modal_flux_on_every_cut_and_wall_agrees_with_periodic_lattice(mode,backend):
    from tensorsph.heat_conduction_2d import (solve_harmonic_layered_slab,stencil_x_face_power_2d,
        stencil_rhs_with_wall_power_2d,lattice_offsets,lattice_diffusion_factor_2d)

    n,dx = 9,1/9
    x = (torch.arange(n,dtype=torch.float64)+.5)*dx
    phase = torch.exp(2j*torch.pi*mode*x)
    walls = (1.-.3j,.2+.5j)
    k = 1+x
    contacts = torch.zeros_like(k[:-1])
    contacts[4] = .25
    opts = dict(conductivity=k,contact_resistance_x=contacts)
    u,flux = solve_harmonic_layered_slab(.4j*x,dx,walls,2.,volumetric_heat_capacity=1.3,
            transverse_wavenumber=2*torch.pi*mode,backend=backend,return_x_flux=True,**opts)
    lattice = phase[:,None]*u
    wall_rows = tuple(phase*v for v in walls)
    offsets = lattice_offsets(dx,1.3*dx)
    _,ends = stencil_rhs_with_wall_power_2d(lattice,offsets,dx,wall_rows,power_by_row=True,layered_x=True,**opts)
    cuts = [stencil_x_face_power_2d(lattice,offsets,dx,i,wall_rows,power_by_row=True,layered_x=True,**opts)
            for i in range(1,n)]
    exact = torch.stack([ends[0],*cuts,-ends[1]],dim=1)/(dx*lattice_diffusion_factor_2d(1.3,dx,mode="moment"))
    torch.testing.assert_close(phase[:,None]*flux,exact,atol=2e-13,rtol=0.)



def test_wall_flux_recovery_differentiates_cubic_and_preserves_each_mean():
    from tensorsph.heat_conduction_2d import recover_dirichlet_wall_flux_2d

    dx, k = .125, 3.
    x = (torch.arange(8,dtype=torch.float64)+.5)*dx
    scale = torch.tensor([1+.3j,-.25j,.8+.1j],dtype=torch.complex128)
    field = scale[:,None]*(1+x+2*x**2-x**3)
    exact = torch.stack((-k*scale,2*k*scale))
    offset = torch.tensor([.7j,-.2],dtype=field.dtype)
    flux = recover_dirichlet_wall_flux_2d(field,dx,k,exact.mean(1)+offset)
    torch.testing.assert_close(flux,exact+offset[:,None],rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize("backend",["torch","banded"])
def test_low_frequency_cut_flux_predicts_small_frequency_for_traveling_walls(backend):
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_slab, solve_low_frequency_layered_slab

    source = torch.zeros(12,dtype=torch.complex128)
    options = dict(conductivity=torch.full((12,),2.),volumetric_heat_capacity=3.,transverse_wavenumber=2.5,
                   backend=backend,return_x_flux=True)
    dc,lag = solve_low_frequency_layered_slab(source,1/12,(1+.4j,.1j),**options)
    errors = []
    for omega in (.002,.001):
        direct = solve_harmonic_layered_slab(source,1/12,(1+.4j,.1j),omega,**options)
        errors.append(torch.stack([abs(actual-base-1j*omega*first).max()
                                   for actual,base,first in zip(direct,dc,lag)]))
    # Halving omega leaves one quarter of the leading truncation error.
    torch.testing.assert_close(errors[0]/errors[1],torch.full_like(errors[0],4.),rtol=.001,atol=0.)


@pytest.mark.parametrize("backend", ["torch", "banded"])
def test_traveling_film_mode_against_exponential_slab(backend, record_property):
    import cmath
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_films

    beta, omega, k, capacity = 1.4, .8, 2., 3.
    exponent = cmath.sqrt(beta**2 + 1j*omega*capacity/k)
    walls = torch.tensor([1., cmath.exp(-exponent)], dtype=torch.complex128)
    inward = k*exponent*walls*walls.new_tensor([1., -1.])
    films = torch.tensor([3., 5.], dtype=torch.float64)
    storage = torch.tensor([.2, .4], dtype=torch.float64)
    ambient = walls + (inward+1j*omega*storage*walls)/films
    errors = []
    for n in (16, 32, 64):
        x = (torch.arange(n, dtype=torch.float64)+.5)/n
        field, surface, flux = solve_harmonic_layered_films(torch.zeros(n), 1/n, ambient, omega,
            conductivity=torch.full((n,), k), volumetric_heat_capacity=capacity, film_coefficients=films,
            wall_areal_heat_capacity=storage, transverse_wavenumber=beta, backend=backend)
        errors.append(torch.stack(((field-torch.exp(-exponent*x)).abs().max(),
                                   (surface-walls).abs().max(), (flux-inward).abs().max())))
        for name, error in zip(("field_K", "surface_K", "inward_flux_W_m2"), errors[-1]):
            record_property(f"n{n}_{name}", error.item())
    for coarse, fine in zip(errors, errors[1:]):
        assert torch.all(coarse/fine > 1.8)
    assert torch.all(errors[-1] < errors[-1].new_tensor([.003, .003, .008]))


def test_traveling_film_low_frequency_response():
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_films, solve_low_frequency_layered_films

    source = torch.zeros(16)
    ambient = (.8+.3j, -.4j)
    options = dict(conductivity=torch.full((16,), 2.), volumetric_heat_capacity=3.,
                   film_coefficients=(3., 5.), wall_areal_heat_capacity=(.2, .4),
                   transverse_wavenumber=1.4)
    steady, lag = solve_low_frequency_layered_films(source, 1/16, ambient, **options)
    errors = []
    for omega in (.004, .002):
        direct = solve_harmonic_layered_films(source, 1/16, ambient, omega, **options)
        errors.append(torch.stack([(actual-base-1j*omega*first).abs().max()
                                   for actual, base, first in zip(direct, steady, lag)]))
    torch.testing.assert_close(errors[0]/errors[1], torch.full_like(errors[0], 4.), rtol=.001, atol=0.)


def test_layered_dirichlet_flux_in_thermal_resistance_coordinate(record_property):
    from tensorsph.heat_conduction_2d import recover_dirichlet_wall_flux_2d

    # Repeated alternating laminae have total resistance 5/8. A smooth mode
    # in resistance has continuous temperature and flux at every interface.
    scale = torch.tensor([1+.3j, -.5+.1j, .25j], dtype=torch.complex128)
    errors = []
    for n in (16, 32, 64):
        k = torch.tensor([1., 4.], dtype=torch.float64).repeat(n//2)
        resistance = (1/n) / k
        r = resistance.cumsum(0) - resistance/2
        field = scale[:,None] * torch.exp(r)
        exact = torch.stack((-scale, math.exp(5/8)*scale))
        flux = recover_dirichlet_wall_flux_2d(field, 1/n, k, exact.mean(dim=1))
        torch.testing.assert_close(flux.mean(dim=1), exact.mean(dim=1), atol=2e-15, rtol=0)
        errors.append(float((flux-exact).abs().max()))
        record_property(f"max_flux_error_{n}", errors[-1])
    assert min(math.log2(a/b) for a,b in zip(errors, errors[1:])) > 2.8


@pytest.mark.parametrize("layered", [False, True])
def test_wall_flux_with_resistive_contacts_and_internal_heat(layered, record_property):
    from tensorsph.heat_conduction_2d import recover_dirichlet_wall_flux_2d

    n = 12
    dx = 1/n
    k = torch.linspace(1., 3., n, dtype=torch.float64) if layered else torch.ones(n, dtype=torch.float64)
    contacts = torch.zeros(n-1, dtype=k.dtype)
    contacts[[1, 3, 8, 9]] = contacts.new_tensor([.23, .11, .45, .08])
    edges = (dx/k).cumsum(0)
    r = edges - dx/(2*k)
    # T is a cubic between contacts. The jump equals R times the
    # conductive derivative, which is continuous at each face.
    derivative = lambda t: 1 + 2*t + 6*t**2
    jumps = torch.cat((k.new_zeros(1), (contacts*derivative(edges[:-1])).cumsum(0)))
    amplitudes = torch.tensor([1+.3j, -.5+.1j, .25j], dtype=torch.complex128)
    field = amplitudes[:,None] * (r + r**2 + 2*r**3 + jumps)
    exact = torch.stack((-amplitudes, amplitudes * derivative(edges[-1])))
    conductivity = k if layered else 1.
    flux = recover_dirichlet_wall_flux_2d(field, dx, conductivity, exact.mean(1),
                                          contact_resistance_x=contacts)
    error = float((flux - exact).abs().max())
    record_property("max_flux_error", error)
    assert error < 1e-11
    total = edges[-1]
    right = amplitudes * (total + total**2 + 2*total**3 + jumps[-1])
    anchored = recover_dirichlet_wall_flux_2d(field, dx, conductivity, exact.mean(1),
        contact_resistance_x=contacts, x_wall_temperatures=(0., right))
    anchored_error = float((anchored-exact).abs().max())
    record_property("prescribed_wall_max_flux_error", anchored_error)
    assert anchored_error < 1e-11


def test_fixed_wall_flux_from_solved_contact_slab(record_property):
    from tensorsph.heat_conduction_2d import (
        layered_slab_modal_cut_flux, recover_dirichlet_wall_flux_2d, solve_harmonic_layered_slab,
    )

    beta, omega = 2*math.pi, .5
    errors = []
    wall_errors = []
    direct_errors = []
    for n in (16, 32, 64, 128):
        dx = 1/n
        k = torch.ones(n, dtype=torch.float64)
        k[n//8:7*n//8] = 2
        contacts = k.new_zeros(n-1)
        contacts[[n//8-1, 7*n//8-1]] = k.new_tensor([.12, .18])
        bulk = dx/k
        r = bulk.cumsum(0) - bulk/2 + torch.cat((k.new_zeros(1), contacts.cumsum(0)))
        right = bulk.sum() + contacts.sum()
        # T = r(x)*exp(i*beta*y+i*omega*t) has unit conductive slope,
        # with jumps equal to each face resistance. The source balances
        # transverse diffusion and storage in each layer.
        solution, cuts = solve_harmonic_layered_slab(
            (k*beta**2 + 1j*omega)*r, dx, (0., right), omega, conductivity=k,
            volumetric_heat_capacity=1., transverse_wavenumber=beta, contact_resistance_x=contacts,
            return_x_flux=True)
        phase = torch.exp(1j * beta * (torch.arange(n, dtype=k.dtype)+.5)*dx)
        # Both wall cuts use positive-x orientation. The analytic flux is -1.
        exact_cuts = torch.stack([layered_slab_modal_cut_flux(
            r, dx, face, (0., right), conductivity=k, transverse_wavenumber=beta,
            contact_resistance_x=contacts) for face in (0, n)])
        record_property(f"direct_exact_field_error_{n}", float((exact_cuts+1).abs().max()))
        direct_errors.append(float((cuts[[0,n]]+1).abs().max()))
        record_property(f"direct_solved_field_error_{n}", direct_errors[-1])
        record_property(f"particle_temperature_error_{n}", float((solution-r).abs().max()))
        reference = torch.stack((-phase, phase))
        exact_recovery = recover_dirichlet_wall_flux_2d(
            phase[:,None]*r, dx, k, [0.,0.], contact_resistance_x=contacts,
            x_wall_temperatures=(0., right*phase))
        exact_error = float((exact_recovery-reference).abs().max())
        record_property(f"recovered_exact_field_error_{n}", exact_error)
        assert exact_error < 2e-12
        field = phase[:,None]*solution
        flux = recover_dirichlet_wall_flux_2d(field, dx, k, [0.,0.], contact_resistance_x=contacts)
        errors.append(float((flux - reference).abs().max()))
        record_property(f"wall_flux_error_{n}", errors[-1])
        anchored = recover_dirichlet_wall_flux_2d(field, dx, k, [0.,0.],
            contact_resistance_x=contacts, x_wall_temperatures=(0., right*phase))
        wall_errors.append(float((anchored - reference).abs().max()))
        record_property(f"prescribed_wall_flux_error_{n}", wall_errors[-1])
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
    assert errors[-1] < 0.05
    assert wall_errors[0] < errors[0] / 30
    assert wall_errors[1] < errors[1] / 2
    assert wall_errors[-1] < 0.05
    assert all(fine < coarse for coarse, fine in zip(direct_errors, direct_errors[1:]))
    assert direct_errors[-1] < 0.022


@pytest.mark.parametrize("backend", ["torch", "banded"])
def test_harmonic_grid_row_flux_matches_pair_tallies(backend, record_property):
    from tensorsph.heat_conduction_2d import (
        lattice_diffusion_factor_2d, solve_harmonic_layered_slab,
        stencil_rhs_with_wall_power_2d, stencil_x_face_power_2d,
    )

    nx, ny, dx = 10, 8, .1
    x = (torch.arange(nx, dtype=torch.float64)+.5)*dx
    phase = torch.exp(2j*math.pi*torch.arange(ny, dtype=x.dtype)/ny)
    source = .5 + phase[:,None]*x + .3j*phase[:,None]**2*x**2
    walls = (1+.2*phase, .25j*phase**3)
    contacts = torch.zeros_like(x[:-1])
    contacts[3] = .2
    material = dict(conductivity=1+x, contact_resistance_x=contacts)
    options = dict(volumetric_heat_capacity=2., backend=backend, **material)
    field, cuts = solve_harmonic_layered_slab(source, dx, walls, .5, return_x_flux="by_row", **options)
    _, means = solve_harmonic_layered_slab(source, dx, walls, .5, return_x_flux=True, **options)
    offsets = lattice_offsets(dx, 1.3*dx)
    tally = dict(layered_x=True, power_by_row=True, **material)
    _, ends = stencil_rhs_with_wall_power_2d(field, offsets, dx, walls, **tally)
    interior = [stencil_x_face_power_2d(field, offsets, dx, i, walls, **tally) for i in range(1, nx)]
    expected = torch.stack([ends[0], *interior, -ends[1]], dim=1)
    expected /= dx*lattice_diffusion_factor_2d(1.3, dx, mode="moment")
    error = float((cuts-expected).abs().max())
    record_property("maximum_pair_tally_discrepancy", error)
    assert error < 1e-12
    torch.testing.assert_close(cuts.mean(0), means, rtol=0, atol=1e-12)


@pytest.mark.parametrize("backend", ["torch", "banded"])
def test_quadratic_low_frequency_grid_flux_against_sine_mode(backend, record_property):
    from tensorsph.heat_conduction_2d import solve_low_frequency_layered_slab

    k, capacity = 1.5, 3.
    eigenvalue = k * 5 * math.pi**2
    for n in (8, 16, 32):
        dx = 1/n
        x = (torch.arange(n, dtype=torch.float64)+.5)*dx
        faces = torch.arange(n+1, dtype=x.dtype)*dx
        phase = torch.exp(2j*math.pi*x)[:,None]
        source = phase * torch.sin(math.pi*x)
        coeffs = solve_low_frequency_layered_slab(source, dx, (0.,0.),
            conductivity=torch.full_like(x,k), volumetric_heat_capacity=capacity,
            backend=backend, return_x_flux="by_row", order=2)
        scale = capacity**2/eigenvalue**3
        field, flux = coeffs[2]
        field_error = float((field/scale-source).abs().max())
        flux_error = float((flux/(scale*k*math.pi)+phase*torch.cos(math.pi*faces)).abs().max())
        record_property(f"normalized_field_error_{n}", field_error)
        record_property(f"normalized_flux_error_{n}", flux_error)
        if n > 8:
            assert field_error < previous_field / 3
            assert flux_error < previous_flux / 3
        previous_field, previous_flux = field_error, flux_error


def test_quadratic_grid_predicts_contact_slab_frequency_response(record_property):
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_slab, solve_low_frequency_layered_slab

    n, dx = 12, 1/12
    x = (torch.arange(n, dtype=torch.float64)+.5)*dx
    phase = torch.exp(2j*math.pi*x)[:,None]
    source = .75 + phase*x
    walls = (1.+.5j*phase[:,0], .3j)
    contacts = torch.zeros_like(x[:-1])
    contacts[5] = .25
    options = dict(conductivity=1.+x, volumetric_heat_capacity=2.+x,
                   contact_resistance_x=contacts, return_x_flux="by_row")
    coefficients = solve_low_frequency_layered_slab(source, dx, walls, order=2, **options)
    errors = []
    for omega in (.1, .05):
        direct = solve_harmonic_layered_slab(source, dx, walls, omega, **options)
        discrepancy = []
        for exact, dc, linear, quadratic in zip(direct, *coefficients):
            discrepancy.append(float((exact-(dc+1j*omega*linear-omega**2*quadratic)).abs().max()))
        errors.append(discrepancy)
        record_property(f"temperature_remainder_{omega}", discrepancy[0])
        record_property(f"flux_remainder_{omega}", discrepancy[1])
    for coarse, fine in zip(*errors):
        assert 7.8 < coarse/fine < 8.2


@pytest.mark.parametrize("backend", ["torch", "banded"])
def test_storing_film_quadratic_response_against_symmetric_slab(backend, record_property):
    from tensorsph.heat_conduction_2d import solve_low_frequency_layered_films

    # A unit ambient on both sides gives steady T=1. For s=iω, integrate
    # k*T_j''=C*T_{j-1} from the symmetry plane, then apply the film law.
    k, capacity, film, surface, half = 1.7, 2.3, 3.2, .6, .5
    wall1 = -(capacity*half + surface)/film
    center1 = wall1 - capacity*half**2/(2*k)
    q2 = capacity*(center1*half + capacity*half**3/(6*k))
    wall2 = (-q2 - surface*wall1)/film
    def primitive(u):
        return capacity/k * (center1*u**2/2 + capacity*u**4/(24*k))

    errors = []
    for n in (16, 32, 64):
        x = (torch.arange(n, dtype=torch.float64)+.5)/n
        exact = wall2 + primitive(x-half) - primitive(half)
        _, _, (field, walls, flux) = solve_low_frequency_layered_films(
            torch.zeros_like(x), 1/n, (1.,1.), film_coefficients=(film,film),
            conductivity=torch.full_like(x,k), volumetric_heat_capacity=capacity,
            wall_areal_heat_capacity=(surface,surface), backend=backend, order=2)
        row = [float((field-exact).abs().max()), float((walls-wall2).abs().max()),
               float((flux-q2).abs().max())]
        errors.append(row)
        for name, error in zip(("field", "walls", "flux"), row):
            record_property(f"{name}_coefficient_error_{n}", error)
    for coarse, fine in zip(errors, errors[1:]):
        assert all(3.8 < a/b < 4.2 for a,b in zip(coarse, fine))


def test_quadratic_film_response_predicts_contact_slab_at_small_frequency(record_property):
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_films, solve_low_frequency_layered_films

    x = (torch.arange(12,dtype=torch.float64)+.5)/12
    contacts = torch.zeros_like(x[:-1])
    contacts[5] = .25
    options = dict(conductivity=1.+x, volumetric_heat_capacity=2.+x, contact_resistance_x=contacts,
                   film_coefficients=(2.,3.), wall_areal_heat_capacity=(.3,.8), transverse_wavenumber=1.)
    args = (x*.2j, 1/12, (1.+.25j, .3j))
    coefficients = solve_low_frequency_layered_films(*args, order=2, **options)
    errors = []
    for omega in (.05, .025):
        direct = solve_harmonic_layered_films(*args, omega, **options)
        row = [float((actual-(dc+1j*omega*first-omega**2*second)).abs().max())
               for actual, dc, first, second in zip(direct, *coefficients)]
        errors.append(row)
        for name, error in zip(("field", "walls", "flux"), row):
            record_property(f"{name}_remainder_{omega}", error)
    assert all(7.8 < a/b < 8.2 for a,b in zip(*errors))


def test_fixed_wall_temperature_step_heat_at_every_section(record_property):
    """A unit slab starts cold, then its left wall jumps to 1 K above the right."""
    from tensorsph.heat_conduction_2d import solve_transient_slab_2d

    # k = C = L = 1. Integrating the continuum Fourier flux in time gives
    # a static cosine series plus an exponentially converging transient.
    # Sum the static series analytically, including at the heated wall.
    final_time = .04
    modes = torch.arange(1, 25, dtype=torch.float64) * math.pi
    decay = torch.exp(-modes.square() * final_time)
    errors = []
    for n in (12, 24, 48):
        dx = 1 / n
        faces = torch.arange(n + 1, dtype=torch.float64) * dx
        centers = faces[:-1] + dx / 2
        exact_heat = (final_time + 1/3 - faces + faces.square()/2
                      - 2 * (torch.cos(faces[:, None]*modes) @ (decay/modes.square())))
        exact_temp = 1 - centers - 2 * (torch.sin(centers[:, None]*modes) @ (decay/modes))
        times = torch.linspace(0, final_time, n + 1, dtype=torch.float64)
        history, heat = solve_transient_slab_2d(
            torch.zeros((1, n), dtype=torch.float64), times, dx, (1., 0.),
            rannacher=True, return_x_heat=True)
        total = heat.sum(0)
        heat_error = (total - exact_heat).abs().max().item()
        temp_error = (history[-1, 0] - exact_temp).abs().max().item()
        balance = (history[-1].sum()*dx - (total[0]-total[-1])).abs().item()
        record_property(f"max_section_heat_error_{n}_J_per_m2", heat_error)
        record_property(f"max_temperature_error_{n}_K", temp_error)
        record_property(f"energy_balance_error_{n}_J_per_m2", balance)
        assert balance < 5e-13
        errors.append((heat_error, temp_error))
    for coarse, fine in zip(errors, errors[1:]):
        assert all(a/b > 3.5 for a,b in zip(coarse, fine))


@pytest.mark.parametrize("method", ["theta", "sdirk2"])
def test_fixed_temperature_pulse_cooling_against_fourier_series(method, record_property):
    """Post-pulse stored heat and wall exchange for a homogeneous unit slab."""
    from tensorsph.heat_conduction_2d import solve_transient_slab_2d

    # The left wall is raised by 1 K, then reset between requested outputs.
    # With k=C=L=1, subtract two delayed step responses. After shutoff the
    # steady terms cancel, leaving sine modes with exponential decay.
    shutoff, start, end = .023, .04, .08
    modes = math.pi * torch.arange(1, 41, dtype=torch.float64)
    rate = modes.square()

    def amplitudes(time):
        return torch.exp(-rate*(time-shutoff)) * (-torch.expm1(-rate*shutoff))

    integrated = amplitudes(start) * (-torch.expm1(-rate*(end-start))) / rate
    wall_heat = -2 * torch.stack((integrated.sum(), integrated @ modes.cos()))
    stored_heat = 2 * (amplitudes(end) * (1-modes.cos()) / rate).sum()
    errors = []
    # Hold the event position within the time step fixed across spatial grids.
    steps = 192
    times = torch.linspace(0., end, steps+1, dtype=torch.float64)
    for n in (12, 24, 48):
        field, heat = solve_transient_slab_2d(
            torch.zeros((1, n), dtype=torch.float64), times, 1/n,
            (lambda t: float(t < shutoff), 0.), wall_jump_times=(shutoff,),
            return_x_heat=True, method=method, rannacher=(method=="theta"))
        post = heat[steps//2:].sum(0)
        assert post[0] < 0 < post[-1]
        stored = field[-1].mean()
        balance = abs(float(stored - field[steps//2].mean() - post[0] + post[-1]))
        assert balance < 5e-13
        row = [(post[[0,-1]]-wall_heat).abs().max().item(), abs(float(stored-stored_heat))]
        record_property(f"max_cooling_wall_heat_error_{n}_J_per_m2", row[0])
        record_property(f"residual_stored_heat_error_{n}_J_per_m2", row[1])
        record_property(f"cooling_balance_error_{n}_J_per_m2", balance)
        errors.append(row)
    for coarse, fine in zip(errors, errors[1:]):
        assert all(a/b > 3.4 for a,b in zip(coarse, fine))


@pytest.mark.parametrize("method", ["theta", "sdirk2"])
def test_fixed_wall_pulse_time_order_on_one_lattice(method, record_property):
    from tensorsph.heat_conduction_2d import (
        lattice_diffusion_factor_2d, solve_transient_slab_2d, stencil_rhs_with_wall_power_2d,
    )

    n = 16
    dx = 1/n
    offsets = lattice_offsets(dx, 1.3*dx)
    factor = lattice_diffusion_factor_2d(dx=dx, mode="moment")
    zero = torch.zeros((1, n), dtype=torch.float64)
    # Augment T with cumulative inward heat at each wall and a constant
    # coordinate for the inhomogeneous wall drive. One unit-thickness row
    # has wall area dx. Construct columns directly from the particle stencil.
    def rates(field, left):
        rate, power = stencil_rhs_with_wall_power_2d(field, offsets, dx, (left, 0.))
        return torch.cat((rate.flatten(), power/dx))/factor

    generator = zero.new_zeros((n+3, n+3))
    for i, basis in enumerate(torch.eye(n, dtype=zero.dtype)):
        generator[:-1, i] = rates(basis[None], 0.)
    reference = zero.new_zeros(n+3)
    reference[-1] = 1.
    shutoff, end = .02, .08
    for duration, left in ((shutoff, 1.), (end-shutoff, 0.)):
        generator[:-1, -1] = rates(zero, left)
        reference = torch.matrix_exp(duration*generator) @ reference

    errors = []
    # The shutoff lies halfway through a step on each grid.
    for steps in (14, 42, 126):
        times = torch.linspace(0., end, steps+1, dtype=zero.dtype)
        history, heat = solve_transient_slab_2d(
            zero, times, dx, (lambda t: float(t < shutoff), 0.),
            wall_jump_times=(shutoff,), return_x_heat=True, method=method,
            rannacher=(method=="theta"))
        inward = heat.sum(0)[[0,-1]] * zero.new_tensor([1.,-1.])
        field_error = (history[-1,0]-reference[:n]).abs().max().item()
        wall_error = (inward-reference[n:n+2]).abs().max().item()
        record_property(f"max_field_error_{steps}_K", field_error)
        record_property(f"max_wall_heat_error_{steps}_J_per_m2", wall_error)
        assert abs(float(history[-1].mean() - inward.sum())) < 5e-13
        errors.append((field_error, wall_error))
    for coarse, fine in zip(errors, errors[1:]):
        assert all(7.5 < a/b < 11 for a,b in zip(coarse, fine))


@pytest.mark.parametrize("method", ["theta", "sdirk2"])
def test_unequal_wall_temperature_ramps_against_cubic_slab(method, record_property):
    """Two prescribed temperature ramps drive storage and interior heat transport."""
    from tensorsph.heat_conduction_2d import solve_transient_slab_2d

    # Unit k, C and length. T=(1+x)t+p(x) solves T_t=T_xx with p''=1+x.
    def profile(x):
        return x**3 / 6 + x**2 / 2 - x / 2

    end = 0.125
    errors = []
    for n in (12, 24, 48):
        dx = 1 / n
        faces = torch.arange(n + 1, dtype=torch.float64) * dx
        centers = faces[:-1] + dx / 2
        initial = profile(centers)[None, :]
        steps = round(end / (0.5 * dx**2))
        times = torch.linspace(0, end, steps + 1, dtype=torch.float64)
        history, heat = solve_transient_slab_2d(
            initial, times, dx, (lambda t: t, lambda t: 1 / 6 + 2*t),
            method=method, return_x_heat=True)
        time = times[1:, None]
        exact_field = profile(centers) + (1 + centers) * time
        exact_heat = -(faces**2 / 2 + faces - 0.5) * time - time**2 / 2
        cumulative = heat.cumsum(dim=0)
        field_error = (history[1:, 0] - exact_field).abs().max().item()
        heat_error = (cumulative - exact_heat).abs().max().item()
        # Check energy at each cell, including the two boundary cells.
        residual = (history[1:, 0] - initial)*dx + cumulative.diff(dim=-1)
        balance_error = residual.abs().max().item()
        record_property(f"max_field_error_{n}_K", field_error)
        record_property(f"max_section_heat_error_{n}_J_per_m2", heat_error)
        record_property(f"max_cell_balance_error_{n}_J_per_m2", balance_error)
        assert balance_error < 5e-13
        errors.append((field_error, heat_error))
    for coarse, fine in zip(errors, errors[1:]):
        assert all(a/b > 3.5 for a, b in zip(coarse, fine))


@pytest.mark.parametrize("method", ["theta", "sdirk2"])
def test_transient_rowwise_wall_heat_against_slab_mode(method, record_property):
    from tensorsph.heat_conduction_2d import solve_transient_slab_2d

    # Unit material and periodic unit width. A decaying sine mode creates
    # opposite local heat flows on parts of the same fixed-temperature wall.
    end = 0.025
    rate = 5 * math.pi**2
    errors = []
    for n in (12, 24, 48):
        dx = 1/n
        x = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
        lateral = torch.cos(2 * math.pi * x)
        initial = 1 - x + lateral[:, None] * torch.sin(math.pi*x)
        times = torch.linspace(0, end, n+1, dtype=x.dtype)
        _, sections, rows = solve_transient_slab_2d(
            initial, times, dx, (1., 0.), method=method,
            return_x_heat=True, return_wall_heat_by_row=True)
        t = times[1:, None]
        mode_heat = math.pi * torch.expm1(-rate*t) * lateral / rate
        exact = torch.stack((t+mode_heat, -t+mode_heat), dim=1)
        error = (rows.cumsum(0) - exact).abs().max().item()
        means = torch.stack((sections[:, 0], -sections[:, -1]), dim=1)
        mean_error = (rows.mean(dim=-1) - means).abs().max().item()
        record_property(f"max_row_heat_error_{n}_J_per_m2", error)
        record_property(f"max_interval_mean_error_{n}_J_per_m2", mean_error)
        assert mean_error < 2e-14
        errors.append(error)
    assert all(a / b > 3.5 for a, b in zip(errors, errors[1:]))


@pytest.mark.parametrize("flux_side", [0, 1])
@pytest.mark.parametrize("method", ["theta", "sdirk2"])
def test_mixed_wall_row_heat_with_decaying_fourier_load(flux_side, method, record_property):
    from tensorsph.heat_conduction_2d import solve_transient_slab_2d

    # T=(1+x) cos(2*pi*y) exp(-4*pi²*t) on the unit square.
    rate = 4 * math.pi**2
    errors = []
    for n in (12, 24, 48):
        x = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        lateral = torch.cos(2*math.pi*x)
        initial = lateral[:, None] * (1+x)
        fixed_side = 1 - flux_side
        walls = [None, None]
        walls[fixed_side] = lambda t: (1+fixed_side)*math.exp(-rate*t)*lateral
        loads = [0., 0.]
        loads[flux_side] = lambda t: (2*flux_side-1)*math.exp(-rate*t)*lateral
        times = torch.linspace(0., 0.025, n+1, dtype=x.dtype)
        _, rows = solve_transient_slab_2d(
            initial, times, 1/n, walls, x_wall_inward_flux=loads,
            return_wall_heat_by_row=True, method=method)
        west = torch.expm1(-rate*times[1:, None])*lateral/rate
        exact = torch.stack((west, -west), dim=1)
        error = (rows.cumsum(0)-exact).abs().amax(dim=(0, 2))
        record_property(f"fixed_wall_heat_error_{n}_J_per_m2", error[fixed_side].item())
        record_property(f"flux_wall_heat_error_{n}_J_per_m2", error[flux_side].item())
        errors.append(error)
    assert all((a/b > 3.5).all() for a, b in zip(errors, errors[1:]))


def test_prescribed_row_heat_integrates_split_flux_pulse():
    from tensorsph.heat_conduction_2d import solve_transient_slab_2d

    field = torch.zeros(3, 4, dtype=torch.float64)
    profile = field.new_tensor([1.5, -0.5, 0.25])
    switch = 0.025
    _, rows = solve_transient_slab_2d(
        field, [0., 0.1], 0.25, (None, None),
        x_wall_inward_flux=(lambda t: profile if t < switch else -profile, 0.5),
        wall_jump_times=(switch,), rannacher=True, return_wall_heat_by_row=True)
    torch.testing.assert_close(rows[0, 0], (2*switch-0.1)*profile)
    torch.testing.assert_close(rows[0, 1], torch.full_like(profile, 0.05))


@pytest.mark.parametrize("backend", ["torch", "banded"])
def test_harmonic_film_section_fluxes_match_prescribed_wall_solution(backend):
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_films, solve_harmonic_layered_slab

    dx = 0.125
    x = (torch.arange(8, dtype=torch.float64)+0.5)*dx
    source = (1.+0.5j)*x
    options = dict(conductivity=1.+x, volumetric_heat_capacity=2.-x, backend=backend,
                   transverse_wavenumber=1.3, contact_resistance_x=[0., 0., 0., .25, 0., 0., 0.])
    field, walls, inward, cuts = solve_harmonic_layered_films(
        source, dx, (1.+0.3j, -0.4j), 2.5, film_coefficients=(1.2, 2.3),
        wall_areal_heat_capacity=(.4, .7), return_x_flux=True, **options)
    direct, expected = solve_harmonic_layered_slab(source, dx, walls, 2.5, return_x_flux=True, **options)
    torch.testing.assert_close(field, direct, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(cuts, expected, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(inward, torch.stack((cuts[0], -cuts[-1])), rtol=0., atol=1e-13)


def test_harmonic_wall_flux_and_low_frequency_with_one_uncoated_end():
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_films, solve_low_frequency_layered_films
    source = torch.zeros(12, dtype=torch.float64)
    applied = torch.tensor([.7-.2j, -.4j], dtype=torch.complex128)
    opts = dict(conductivity=torch.full_like(source, 1.7), volumetric_heat_capacity=2.1, wall_areal_heat_capacity=(.3,.8),
                wall_applied_flux=applied, film_coefficients=(0.,2.))
    omega = 1e-4
    terms = solve_low_frequency_layered_films(source, 1/12, (0.,.5), order=2, **opts)
    actual = solve_harmonic_layered_films(source, 1/12, (0.,.5), omega, **opts)
    for field, steady, lag, second in zip(actual, *terms):
        torch.testing.assert_close(field, steady+1j*omega*lag-omega**2*second, rtol=0., atol=1e-10)
    opts["film_coefficients"] = (0.,0.)
    with pytest.raises(ValueError, match="temperature reference"):
        solve_harmonic_layered_films(source, 1/12, (0.,0.), 0., **opts)
    _, wall, inward = solve_harmonic_layered_films(source, 1/12, (0.,0.), 3., **opts)
    torch.testing.assert_close(inward+3j*wall*wall.real.new_tensor([.3,.8]), applied)


@pytest.mark.parametrize("backend", ["torch", "banded"])
def test_steady_flux_wall_mean_shifts_only_temperatures(backend):
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_films

    source = torch.linspace(-.7, .3, 12, dtype=torch.float64)
    dx = 1/12
    capacity = 1.5+source
    surface = source.new_tensor([.3, .7])
    applied = torch.stack((source.new_tensor(.8), -.8-dx*source.sum()))
    opts = dict(film_coefficients=(0., 0.), wall_applied_flux=applied, backend=backend,
                conductivity=2.+source, volumetric_heat_capacity=capacity,
                wall_areal_heat_capacity=surface, return_x_flux=True)
    args = (source, dx, (0., 0.), 0.)
    a = solve_harmonic_layered_films(*args, mean_temperature=.2j, **opts)
    b = solve_harmonic_layered_films(*args, mean_temperature=3.+.2j, **opts)
    field, walls, inward, cuts = a
    stored = dx*(capacity*field).sum()+(surface*walls).sum()
    torch.testing.assert_close(stored, .2j*(dx*capacity.sum()+surface.sum()))
    torch.testing.assert_close(inward, applied.to(inward.dtype))
    torch.testing.assert_close(cuts.diff(), (dx*source).to(cuts.dtype))
    for i, (first, second) in enumerate(zip(a, b)):
        torch.testing.assert_close(second, first+3 if i < 2 else first)
    with pytest.raises(ValueError, match="balanced wall inputs"):
        solve_harmonic_layered_films(*args, mean_temperature=0., **dict(opts, wall_applied_flux=(0., 0.)))
    with pytest.raises(ValueError, match="only to an unanchored"):
        solve_harmonic_layered_films(*args, mean_temperature=0., **dict(opts, film_coefficients=(1., 0.)))


@pytest.mark.parametrize("backend", ["torch", "banded"])
def test_balanced_flux_only_low_frequency_coefficients(backend):
    from tensorsph.heat_conduction_2d import solve_low_frequency_layered_films, solve_harmonic_layered_films

    source = torch.linspace(-1., .5, 12, dtype=torch.float64)
    dx = 1/12
    applied = (.8, -.8-dx*source.sum().item())
    contacts = torch.zeros_like(source[:-1])
    contacts[5] = .35
    cap = source+2.
    surface = source.new_tensor([.3, .7])
    opts = dict(film_coefficients=(0., 0.), wall_applied_flux=applied, conductivity=2.+source,
                volumetric_heat_capacity=cap, contact_resistance_x=contacts,
                wall_areal_heat_capacity=surface, backend=backend)
    args = (source, dx, (0., 0.))
    terms = solve_low_frequency_layered_films(*args, order=2, **opts)
    for field, walls, _ in terms:
        assert abs(dx*(cap*field).sum()+(surface*walls).sum()) < 1e-13
    errors = []
    for omega in (.1, .05, .025):
        direct = solve_harmonic_layered_films(*args, omega, **opts)
        errors.append(max((value-a-1j*omega*b+omega**2*c).abs().max().item()
                          for value, a, b, c in zip(direct, *terms)))
    assert min(errors[0]/errors[1], errors[1]/errors[2]) > 7.8
    with pytest.raises(ValueError, match="balanced wall inputs"):
        solve_low_frequency_layered_films(*args, **dict(opts, wall_applied_flux=(0., 0.)))


@pytest.mark.parametrize("backend", ["torch", "banded"])
def test_unbalanced_closed_slab_laurent_response(backend):
    from tensorsph.heat_conduction_2d import solve_low_frequency_layered_films, solve_harmonic_layered_films

    x = torch.arange(12, dtype=torch.float64)/12
    capacity = 1.2+x
    surface = x.new_tensor([.4, .7])
    applied = torch.tensor([.4+.1j, -.12j], dtype=torch.complex128)
    args = (x-.1j, 1/12, (0., 0.))
    opts = dict(film_coefficients=(0., 0.), conductivity=1.+x, volumetric_heat_capacity=capacity,
                wall_areal_heat_capacity=surface, wall_applied_flux=applied, backend=backend)
    residue, terms = solve_low_frequency_layered_films(*args, order=2, return_pole=True, **opts)
    torch.testing.assert_close(residue, (args[0].mean()+applied.sum())/(capacity.mean()+surface.sum()))
    errors = []
    for omega in (.2, .1, .05):
        direct = solve_harmonic_layered_films(*args, omega, **opts)
        errors.append(max((value-(residue/(1j*omega) if i < 2 else 0)-a-1j*omega*b+omega**2*c).abs().max().item()
                          for i, (value, a, b, c) in enumerate(zip(direct, *terms))))
    assert min(errors[0]/errors[1], errors[1]/errors[2]) > 7.7

    # Independently drive every store in proportion to its capacity.
    rate = .27+.34j
    residue, terms = solve_low_frequency_layered_films(capacity*rate, 1/12, (0., 0.), return_pole=True,
        **dict(opts, wall_applied_flux=surface*rate))
    torch.testing.assert_close(residue, residue.new_tensor(rate))
    assert max(v.abs().max().item() for term in terms for v in term) < 1e-13


@pytest.mark.parametrize("backend", ["torch", "banded"])
def test_uniform_closed_slab_has_zero_cut_flux_at_tiny_frequency(backend):
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_films

    cap = torch.linspace(.9, 2., 32, dtype=torch.float64)
    surface = cap.new_tensor([.3, .75])
    rate = .27+.34j
    opts = dict(conductivity=torch.ones_like(cap), volumetric_heat_capacity=cap, backend=backend,
                wall_areal_heat_capacity=surface, wall_applied_flux=surface*rate,
                film_coefficients=(0., 0.), return_x_flux=True)
    for omega in (1e-12, 1., 100.):
        args = (cap*rate, 1/32, (0., 0.), omega)
        field, wall, inward, cuts, mean = solve_harmonic_layered_films(*args, separate_mean=True, **opts)
        torch.testing.assert_close(1j*omega*mean, mean.new_tensor(rate), atol=1e-14, rtol=0)
        assert max(x.abs().max().item() for x in (field, wall, inward, cuts)) < 1e-13
        full = solve_harmonic_layered_films(*args, **opts)
        for value, expected in zip(full, (mean+field, mean+wall, inward, cuts)):
            torch.testing.assert_close(value, expected)


@pytest.mark.parametrize("material", ["uniform", "layered", "contact"])
def test_four_column_fixed_wall_flux(material, record_property):
    from tensorsph.heat_conduction_2d import recover_dirichlet_wall_flux_2d

    dx = .25
    k = torch.tensor([1., 2., .8, 1.5], dtype=torch.float64)
    if material == "uniform":
        k.fill_(1.)
    resistance = torch.tensor([.08, .2, .11], dtype=k.dtype) if material == "contact" else torch.zeros_like(k[:-1])
    edges = torch.cumsum(dx / k, 0)
    centers = edges - .5 * dx / k
    # T = amplitude * (r + r^3) plus the jumps R * dT/dr at each contact.
    jumps = torch.cat((k.new_zeros(1), (resistance * (1 + 3 * edges[:-1]**2)).cumsum(0)))
    amplitude = torch.tensor([1+.2j, -.3+.5j], dtype=torch.complex128)
    field = amplitude[:, None] * (centers + centers**3 + jumps)
    walls = (0., amplitude * (edges[-1] + edges[-1]**3 + jumps[-1]))
    expected = torch.stack((-amplitude, amplitude * (1 + 3 * edges[-1]**2)))
    flux = recover_dirichlet_wall_flux_2d(field, dx, 1. if material == "uniform" else k,
        expected.mean(1), contact_resistance_x=resistance if material == "contact" else None,
        x_wall_temperatures=walls)
    error = float((flux - expected).abs().max())
    record_property("max_flux_error_W_m2", error)
    assert error < 1e-12
    with pytest.raises(ValueError):
        recover_dirichlet_wall_flux_2d(field, dx, k, expected.mean(1))


def test_traveling_temperature_wall_row_flux_refines():
    from tensorsph.heat_conduction_2d import solve_transient_slab_2d

    # A periodic transverse mode propagates from the west wall by diffusion.
    omega = 8.
    mu = complex((2*math.pi)**2, omega)**.5
    errors = []
    for n in (16, 32, 64):
        x = (torch.arange(n, dtype=torch.float64)+.5)/n
        lateral = torch.exp(2j*math.pi*x)
        amplitude = lateral[:, None]*torch.sinh(mu*(1-x))/cmath.sinh(mu)
        times = torch.linspace(0., .2, n+1, dtype=x.dtype)
        _, heat = solve_transient_slab_2d(
            amplitude.real, times, 1/n,
            (lambda t: (lateral*cmath.exp(1j*omega*t)).real, 0.),
            return_wall_heat_by_row=True, method="sdirk2", wall_gradient_fit="interior")
        average_phase = torch.exp(1j*omega*(times[1:]+times[:-1])/2)*torch.sinc(omega*times.diff()/(2*math.pi))
        flux = mu/cmath.tanh(mu)*lateral
        exact = (average_phase[:, None]*flux).real
        errors.append((heat[:, 0]/times.diff()[:, None]-exact).abs().max().item())
    assert all(a/b > 3.5 for a, b in zip(errors, errors[1:]))


def test_fixed_wall_flux_resolves_one_ulp_temperature_difference():
    from tensorsph.heat_conduction_2d import solve_steady_slab_2d

    source = torch.zeros((6, 32), dtype=torch.float64)
    drop = 2**-44
    _, flux = solve_steady_slab_2d(source, 1/32, (305., 305.+drop),
                                  conductivity=1.7, return_x_flux=True)
    torch.testing.assert_close(flux, torch.full_like(flux, -1.7*drop), atol=0., rtol=1e-9)


@pytest.mark.parametrize("backend", ["torch", "banded"])
def test_layered_sampled_dc_flux_resolves_small_wall_differences(backend):
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_slab

    source = torch.zeros((5, 16), dtype=torch.float64)
    drop = 2**-44
    k = torch.ones(16, dtype=source.dtype)
    k[8:] = 4
    resistance = torch.zeros(15, dtype=source.dtype)
    resistance[7] = 0.25
    _, cuts = solve_harmonic_layered_slab(
        source, 1/16, (305., 305.+drop), 0., conductivity=k, volumetric_heat_capacity=1.,
        contact_resistance_x=resistance, backend=backend, return_x_flux="by_row",
    )
    torch.testing.assert_close(cuts, torch.full_like(cuts, -drop/0.875), atol=0., rtol=1e-8)


def test_steady_slab_reference_preserves_weak_heating_temperature_field():
    from tensorsph.heat_conduction_2d import solve_steady_slab_2d

    source = torch.full((4, 16), 2**-44, dtype=torch.float64)
    opts = dict(conductivity=1.7, return_x_flux=True)
    field, cuts = solve_steady_slab_2d(source, 1/16, (305., 305.), temperature_reference=305., **opts)
    control, control_cuts = solve_steady_slab_2d(source, 1/16, (0., 0.), **opts)
    torch.testing.assert_close(field, control, rtol=0., atol=0.)
    torch.testing.assert_close(cuts, control_cuts, rtol=0., atol=0.)
    assert field.max() > 0
    with pytest.raises(ValueError, match="temperature_reference"):
        solve_steady_slab_2d(source, 1/16, (305., 305.), temperature_reference=float("nan"))


@pytest.mark.parametrize("method", ["theta", "sdirk2"])
def test_transient_slab_weak_pulse_is_independent_of_temperature_datum(method):
    from tensorsph.heat_conduction_2d import solve_transient_slab_2d

    field = torch.zeros((5, 16), dtype=torch.float64)
    load = torch.arange(16, dtype=field.dtype).expand_as(field)*2**-44
    options = dict(method=method, heat_source=lambda t: load if t < .02 else 0.,
                   wall_jump_times=[.02], return_x_heat=True, return_wall_heat_by_row=True)
    times = [0., .01, .02, .03, .04]
    control = solve_transient_slab_2d(field, times, 1/16, (0., 0.), **options)
    shifted = solve_transient_slab_2d(field+305., times, 1/16, (305., 305.),
                                      temperature_reference=305., **options)
    for actual, expected in zip(shifted, control):
        torch.testing.assert_close(actual, expected, rtol=0., atol=0.)


@pytest.mark.parametrize("options", [dict(theta=.75), dict(rannacher=True), dict(method="sdirk2")])
def test_interval_mean_temperature_closes_bulk_cooling_balance(options):
    from tensorsph.heat_conduction_2d import solve_transient_slab_2d

    initial = torch.full((3, 8), 307., dtype=torch.float64)
    times = [0., .15, .31]
    fields, means = solve_transient_slab_2d(
        initial, times, 1/8, (None, None), heat_loss_coefficient=2.,
        heat_source=2.*305., temperature_reference=305., wall_jump_times=[.1],
        return_interval_temperatures=True, **options)
    dt = initial.new_tensor([b-a for a,b in zip(times, times[1:])])
    torch.testing.assert_close(fields.diff(dim=0), -2*dt[:, None, None]*means, atol=1e-12, rtol=0.)


@pytest.mark.parametrize("columns", [2, 3])
def test_narrow_fixed_wall_resistance_gradient(columns, record_property):
    from tensorsph.heat_conduction_2d import recover_dirichlet_wall_flux_2d

    k = torch.tensor([1.2, .8, 2.1][:columns], dtype=torch.float64)
    dx = 1/columns
    edges = (dx/k).cumsum(0)
    centers = edges - dx/(2*k)
    contacts = k.new_tensor([.2, .08][:columns-1])
    # The conductive-coordinate polynomial and its derivative determine the
    # temperature on either side of each contact and both exact inward fluxes.
    curvature = .35 if columns == 3 else 0.
    jumps = (contacts*(1 + 2*curvature*edges[:-1])).cumsum(0)
    amplitude = torch.tensor([1.+.2j, -.3+.5j, .6-.9j], dtype=torch.complex128)
    field = amplitude[:, None]*(centers + curvature*centers**2 + torch.cat((k.new_zeros(1), jumps)))
    east = amplitude*(edges[-1] + curvature*edges[-1]**2 + jumps[-1])
    expected = torch.stack((-amplitude, amplitude*(1+2*curvature*edges[-1])))
    flux = recover_dirichlet_wall_flux_2d(field, dx, k, expected.mean(1),
                                          x_wall_temperatures=(0., east), contact_resistance_x=contacts)
    record_property("max_flux_error_W_m2", float(abs(flux-expected).max()))
    torch.testing.assert_close(flux, expected, atol=2e-13, rtol=0)


@pytest.mark.parametrize("field_dtype,mean", [
    (torch.float64, 1 + 2**-32),
    (torch.float64, (1 + 2**-32) + (1 - 2**-32)*1j),
    (torch.complex128, (1 + 2**-32) + (1 - 2**-32)*1j),
])
def test_python_wall_flux_means_keep_double_precision(field_dtype, mean):
    from tensorsph.heat_conduction_2d import recover_dirichlet_wall_flux_2d

    # A linear temperature profile has spatially constant wall flux. A complex
    # mean can also supply the phase of a spatially uniform flux component.
    field = torch.arange(4, dtype=torch.float64).to(field_dtype).expand(3, 4)
    flux = recover_dirichlet_wall_flux_2d(field, 1., 1., [mean, -mean], x_wall_temperatures=(-.5, 3.5))
    expected = torch.tensor([mean, -mean], dtype=torch.complex128 if isinstance(mean, complex) else torch.float64)
    torch.testing.assert_close(flux, expected[:, None].expand_as(flux), atol=0., rtol=0.)


@pytest.mark.parametrize("method", ["theta", "sdirk2"])
def test_transient_relative_bulk_exchange_with_moving_ambient(method):
    from tensorsph.heat_conduction_2d import solve_transient_slab_2d

    # Uniform insulated slab: T = 305 + t, Ta = 305 + 2*t,
    # H = 3 and source = 1 - 3*t give dT/dt = 1 exactly.
    initial = torch.full((2, 8), 305., dtype=torch.float64)
    times = [0., .125, .25, .5]
    fields = solve_transient_slab_2d(
        initial, times, .125, (None, None), method=method, heat_loss_coefficient=3.,
        ambient_temperature=lambda t: initial + 2*t, heat_source=lambda t: 1 - 3*t,
        temperature_reference=305.)
    expected = initial.new_tensor(times)[:, None, None].expand_as(fields)
    torch.testing.assert_close(fields, expected, atol=1e-13, rtol=0)

    # Adding this source directly to H*305 would round it away.
    def excess(base):
        return solve_transient_slab_2d(
            torch.full_like(initial, base), times, .125, (base, base), method=method,
            heat_loss_coefficient=3., ambient_temperature=base, heat_source=2**-46,
            temperature_reference=base)
    reference = excess(0.)
    assert reference[-1].min() > 0
    torch.testing.assert_close(excess(305.), reference, atol=0, rtol=0)


@pytest.mark.parametrize("walls", [(None, None), (305., None), (305., 305.)])
def test_steady_weak_source_at_prescribed_bulk_ambient(walls):
    from tensorsph.heat_conduction_2d import solve_steady_slab_2d

    source = torch.full((3, 12), 2**-48, dtype=torch.float64)
    relative_walls = tuple(None if w is None else 0. for w in walls)
    def solve(boundary, ambient, reference):
        return solve_steady_slab_2d(source, 1/12, boundary, heat_loss_coefficient=2.,
                                    ambient_temperature=ambient, temperature_reference=reference,
                                    return_x_flux=True)
    reference = solve(relative_walls, 0., 0.)
    field, cuts = solve(walls, torch.full_like(source, 305.), 305.)
    torch.testing.assert_close(field, reference[0], rtol=0, atol=0)
    torch.testing.assert_close(cuts, reference[1], rtol=0, atol=0)
    assert field.min() > 0
    if walls == (None, None):
        torch.testing.assert_close(field, source/2, rtol=1e-14, atol=0)


def test_steady_spatially_varying_ambient_balances_local_heat_source():
    from tensorsph.heat_conduction_2d import solve_steady_slab_2d

    ambient = 305. + torch.arange(24, dtype=torch.float64).reshape(3, 8)/32
    source = 3*(305.-ambient)
    field, flux = solve_steady_slab_2d(source, .125, (305., 305.), ambient_temperature=ambient,
                                       heat_loss_coefficient=3., temperature_reference=305., return_x_flux=True)
    assert torch.count_nonzero(field) == torch.count_nonzero(flux) == 0


@pytest.mark.parametrize("loss", [0., 1.75])
def test_harmonic_ambient_drives_exchange_but_not_storage(loss):
    from tensorsph.heat_conduction_2d import solve_harmonic_slab_2d

    source = torch.full((3, 10), .7-.2j, dtype=torch.complex128)
    ambient = .3+.8j
    omega, capacity = 5., 1.2
    exact = (source + loss*ambient)/(loss + 1j*omega*capacity)
    for walls in ((None, None), (exact[0, 0], None), (exact[0, 0], exact[0, 0])):
        field, flux = solve_harmonic_slab_2d(source, .1, walls, omega,
                                              volumetric_heat_capacity=capacity, heat_loss_coefficient=loss,
                                              ambient_temperature=ambient, return_x_flux=True)
        torch.testing.assert_close(field, exact, rtol=1e-13, atol=1e-14)
        assert flux.abs().max() < 1e-13


@pytest.mark.parametrize("backend", ["torch", "banded"])
def test_layered_ambient_phasor_with_discontinuous_exchange(backend):
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_slab

    n = 8
    k = torch.linspace(1., 3., n, dtype=torch.float64)
    loss = (k > 2).to(k.dtype)*2
    ambient = (.5-.2j)*k
    target = .3+.4j
    source = 1j*5*k*target + loss*(target-ambient)
    for load in (source, source.expand(3, n)):
        field, flux = solve_harmonic_layered_slab(load, 1/n, (target, None), 5.,
            conductivity=k, volumetric_heat_capacity=k, heat_loss_coefficient=loss,
            ambient_temperature=ambient, backend=backend, return_x_flux=True)
        torch.testing.assert_close(field, torch.full_like(field, target), atol=1e-13, rtol=0)
        assert flux.abs().max() < 1e-12


@pytest.mark.parametrize("method", ["theta", "sdirk2", "exponential", "exponential_quadratic"])
def test_layered_wall_heat_components_across_flux_jump(method):
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    times = [0., .15, .3]
    field = torch.zeros(6, dtype=torch.float64)
    options = dict(conductivity=field+1., volumetric_heat_capacity=1., method=method,
                   x_wall_inward_flux=(lambda t: .8 if t < .1 else -.4, 0.),
                   x_wall_films=(None, (2., 1.)), load_jump_times=[.1],
                   return_wall_heat_components=True)
    _, external, parts = solve_transient_layered_slab(field, times, 1/6, (None,None),
                                                       return_external_heat=True, **options)
    torch.testing.assert_close(parts["applied"][:,0], field.new_tensor([.06,-.06]), atol=1e-14, rtol=0)
    assert parts["applied"][:,1].count_nonzero() == parts["film"][:,0].count_nonzero() == 0
    torch.testing.assert_close(parts["film"]+parts["applied"], external, atol=1e-13, rtol=0)
    _, alone = solve_transient_layered_slab(field, times, 1/6, (None,None), **options)
    torch.testing.assert_close(parts["film"], alone["film"])


@pytest.mark.parametrize("method", ["theta", "sdirk2", "exponential", "exponential_quadratic"])
def test_massless_film_with_heater_reconstructs_wall_after_jump(method):
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    k = torch.ones(8, dtype=torch.float64)
    switch, h = .03, 4.
    def heater(t):
        return .8 if t < switch else -.3
    field, cuts, surface, mean, parts = solve_transient_layered_slab(
        k*0., [0., switch, .1], .125, (None, 0.), conductivity=k, volumetric_heat_capacity=1.,
        x_wall_inward_flux=(heater, 0.), x_wall_films=((h, .5), None), method=method,
        load_jump_times=[switch], return_x_heat=True, return_wall_temperatures=True,
        return_interval_wall_temperatures=True, return_wall_heat_components=True)
    half_r = .125/2
    load = k.new_tensor([.8,-.3,-.3])
    torch.testing.assert_close(surface[:,0], (field[:,0]+half_r*(load+h*.5))/(1+half_r*h))
    dt = k.new_tensor([switch, .1-switch])
    torch.testing.assert_close(parts["applied"][:,0], load[:2]*dt)
    torch.testing.assert_close(parts["film"][:,0], dt*h*(.5-mean[:,0]))
    torch.testing.assert_close(cuts[:,0], (parts["applied"]+parts["film"])[:,0])


@pytest.mark.parametrize("side", [0, 1])
def test_quadrature_wall_flux_rotates_harmonic_solution(side):
    from tensorsph.heat_conduction_2d import solve_harmonic_slab_2d

    source = torch.zeros((6, 12), dtype=torch.float64)
    walls = (None, 0.) if side == 0 else (0., None)

    def response(load, conditions):
        fluxes = (load, 0.) if side == 0 else (0., load)
        return solve_harmonic_slab_2d(source, 1/12, conditions, 3.,
                                      x_wall_inward_flux=fluxes, return_x_flux=True)

    real_field, real_flux = response(1., walls)
    field, flux = response(1j, walls)
    torch.testing.assert_close(field, 1j * real_field)
    torch.testing.assert_close(flux, 1j * real_flux)
    with pytest.raises(ValueError, match="prescribe heat flux"):
        response(1j, (0., 0.))


def test_closed_harmonic_section_flux_resolves_small_gradients_on_uniform_pole():
    from tensorsph.heat_conduction_2d import solve_harmonic_slab_2d

    columns = 32
    x = (torch.arange(columns, dtype=torch.float64) + 0.5) / columns
    source = (1 + torch.cos(torch.pi * x)).repeat(8, 1)
    omega = 1e-14
    field, flux = solve_harmonic_slab_2d(
        source, 1 / columns, (None, None), omega,
        x_wall_inward_flux=(1j, 2j), return_x_flux=True,
    )
    storage = 1j * omega * field.mean(dim=0)
    torch.testing.assert_close(storage.mean(), storage.new_tensor(1+3j))
    assert (flux.diff() + (storage-source[0])/columns).abs().max() < 1e-12


def test_closed_steady_flux_with_weak_volumetric_cooling():
    from tensorsph.heat_conduction_2d import solve_steady_slab_2d

    columns = 32
    x = (torch.arange(columns, dtype=torch.float64) + 0.5) / columns
    source = (1 + torch.cos(torch.pi * x)).repeat(8, 1)
    loss = 1e-14
    excess, flux = solve_steady_slab_2d(
        source, 1 / columns, (None, None), heat_loss_coefficient=loss,
        x_wall_inward_flux=(1., 2.), ambient_temperature=300.,
        temperature_reference=300., return_x_flux=True,
    )
    cooling = loss * excess.mean(0)
    torch.testing.assert_close(cooling.mean(), cooling.new_tensor(4.))
    assert (flux.diff() + (cooling - source[0])/columns).abs().max() < 1e-12


@pytest.mark.parametrize("loss", [0., 1e-18])
def test_steady_separated_mean_resolves_two_flux_wall_profile(loss):
    from tensorsph.heat_conduction_2d import solve_steady_slab_2d

    n = 16
    x = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    # The uniform source balances the weak loss at a 1e18 K mean.
    # A linear temperature profile carries a unit through-flux.
    source = torch.full((4, n), loss * 1e18, dtype=x.dtype)
    options = dict(x_wall_inward_flux=(1., -1.), heat_loss_coefficient=loss,
                   mean_temperature=None if loss else 1e18, separate_mean=True)
    profile, cuts, mean = solve_steady_slab_2d(source, 1/n, (None, None), return_x_flux=True, **options)
    torch.testing.assert_close(profile, (0.5-x).expand_as(source))
    torch.testing.assert_close(cuts, torch.ones_like(cuts))
    torch.testing.assert_close(mean, x.new_tensor(1e18))
    profile_only, relative_mean = solve_steady_slab_2d(
        source, 1/n, (None, None), temperature_reference=1e18, **options)
    torch.testing.assert_close(profile_only, profile)
    assert relative_mean.abs() < 256


@pytest.mark.parametrize("loss", [0., 1e-18])
def test_harmonic_separated_mean_with_balanced_wall_flux(loss):
    from tensorsph.heat_conduction_2d import solve_harmonic_slab_2d

    n = 16
    omega = 1e-18
    x = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    through_flux = 0.4 - 1j
    source = torch.full((4, n), 2+3j, dtype=torch.complex128)
    options = dict(x_wall_inward_flux=(through_flux, -through_flux),
                   heat_loss_coefficient=loss, separate_mean=True)
    profile, flux, mean = solve_harmonic_slab_2d(
        source, 1/n, (None, None), omega, return_x_flux=True, **options)
    torch.testing.assert_close(mean * (loss + 1j*omega), source[0, 0])
    torch.testing.assert_close(profile, ((0.5-x)*through_flux).expand_as(source))
    torch.testing.assert_close(flux, torch.full_like(flux, through_flux))
    pair = solve_harmonic_slab_2d(source, 1/n, (None, None), omega, **options)
    torch.testing.assert_close(pair, (profile, mean))


def test_layered_harmonic_weak_wall_flux_reflects_with_the_heater():
    from tensorsph.heat_conduction_2d import solve_harmonic_layered_slab

    n = 128
    k = torch.ones(n, dtype=torch.float64)
    k[:n//2] = 4.
    capacity = 1.25 + (k == 4.) * 1.25
    source = torch.zeros_like(k)
    source[n//8:n//4] = 12.
    contact = torch.zeros(n-1, dtype=k.dtype)
    contact[n//2-1] = 0.35

    def solve(load, material, storage):
        return solve_harmonic_layered_slab(
            load, 1/n, (0., 0.), 800., conductivity=material, volumetric_heat_capacity=storage,
            contact_resistance_x=contact, h_factor=0.9, backend="banded", return_x_flux=True)[1]

    cuts = solve(source, k, capacity)
    reflected = solve(source.flip(0), k.flip(0), capacity.flip(0))
    torch.testing.assert_close(cuts, -reflected.flip(0), atol=0., rtol=1e-10)


@pytest.mark.parametrize("h_factor", [0.9, 1.6])
def test_split_interface_links_resolve_transverse_conduction_at_a_material_jump(h_factor):
    from tensorsph.heat_conduction_2d import layered_slab_modal_cut_flux, solve_harmonic_layered_slab

    # Across a perfectly bonded interface the exact field exp(i beta y)
    # has no normal heat flow, even though k and the required heating jump.
    n = 48
    dx, beta, omega = 1/n, 2*math.pi, 3.
    k = torch.ones(n, dtype=torch.float64)
    k[n//2:] = 4.
    # The uniform lattice's exact discrete transverse eigenvalue isolates
    # the interface error from the ordinary bulk dispersion error.
    offsets = lattice_offsets(dx, h_factor*dx)
    factor = -sum(a*(j*dx)**2 for (_, j), a in offsets)
    decay = -4*sum(a*math.sin(m*beta*dx/2)**2 for (m, _), a in offsets)/factor
    field, cuts = solve_harmonic_layered_slab(
        k*decay + 1j*omega, dx, (1., 1.), omega, conductivity=k, volumetric_heat_capacity=1.,
        h_factor=h_factor, transverse_wavenumber=beta, split_interface_links=True, return_x_flux=True)
    torch.testing.assert_close(field, torch.ones_like(field), rtol=0., atol=2e-12)
    assert cuts[[0, n//2, -1]].abs().max() < 1e-11
    for face in (0, n//2-1, n//2, n):
        recovered = layered_slab_modal_cut_flux(field, dx, face, (1., 1.), conductivity=k,
                                                h_factor=h_factor, transverse_wavenumber=beta,
                                                split_interface_links=True)
        torch.testing.assert_close(recovered, cuts[face], rtol=0., atol=1e-13)


@pytest.mark.parametrize("contact", [0., 0.25])
def test_curved_half_cells_share_one_complex_fourier_flux(contact):
    from tensorsph.heat_conduction_2d import recover_layered_interior_flux

    dx = 0.2
    q = 1.1 - 0.4j
    gradient = torch.tensor([2+3j, -0.5j], dtype=torch.complex128)
    k = torch.tensor([1.5, 3.], dtype=torch.float64)
    t = torch.stack(((q*dx/2-gradient[0]*dx**2/8)/k[0],
                     -contact*q-(q*dx/2+gradient[1]*dx**2/8)/k[1]))
    rows = torch.stack((t, -t))
    flux = recover_layered_interior_flux(rows, dx, k, contact_resistance_x=[contact],
                                         axial_flux_gradient=torch.stack((gradient, -gradient)))
    torch.testing.assert_close(flux[:, 0], t.new_tensor([q, -q]), atol=1e-14, rtol=1e-14)


def test_contact_flux_from_independent_interior_anchors():
    from tensorsph.heat_conduction_2d import extrapolate_layered_face_flux

    # q(x) = x² on the left and 2x²-1 on the right, continuous at x=1.
    dx = 0.25
    x = dx*torch.arange(1, 8, dtype=torch.float64)
    centers = dx*(torch.arange(8, dtype=x.dtype)+0.5)
    q = torch.where(x <= 1, x**2, 2*x**2-1)
    q[3] = 99  # the direct estimate at the requested cut is deliberately wrong
    gradient = torch.where(centers < 1, 2*centers, 4*centers)
    phases = torch.tensor([1+0j, 1-2j], dtype=torch.complex128)
    rows = phases[:, None]
    for value in extrapolate_layered_face_flux(rows*q, dx, 4, 2, axial_flux_gradient=rows*gradient):
        torch.testing.assert_close(value, phases, rtol=0., atol=1e-14)
    with pytest.raises(ValueError, match="anchors"):
        extrapolate_layered_face_flux(q, dx, 4, 4, axial_flux_gradient=gradient)


@pytest.mark.parametrize("anchor", [True, False])
def test_single_fixed_wall_with_layered_contact(anchor):
    from tensorsph.heat_conduction_2d import recover_dirichlet_wall_flux_2d

    dx = .2
    k = torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float64)
    contacts = k.new_tensor([.1, 0., 0., .3])
    resistance = dx/k
    centers = resistance.cumsum(0)-resistance/2 + torch.cat((k.new_zeros(1), contacts.cumsum(0)))
    flux = torch.tensor([1+2j, -3j, .2+.4j], dtype=torch.complex128)
    field = -flux[:, None]*centers
    for side, sign, walls in (("left", 1, (0., None)),
                               ("right", -1, (None, -flux*(resistance.sum()+contacts.sum())))):
        recovered = recover_dirichlet_wall_flux_2d(
            field, dx, k, sign*flux.mean(), side=side, contact_resistance_x=contacts,
            x_wall_temperatures=walls if anchor else None)
        torch.testing.assert_close(recovered, sign*flux, rtol=1e-12, atol=1e-12)


def test_quadratic_flux_reconstructs_both_cubic_contact_traces():
    from tensorsph.heat_conduction_2d import layered_slab_contact_temperatures

    h = .25
    k = torch.tensor([1.5, 3.], dtype=torch.float64)
    curvature = k.new_tensor([2., -5.])
    # q(x)=q0+b*x+c*x²/2 with x measured from the shared contact.
    q0, slope = .8, k.new_tensor([.4, -.2])
    z = k.new_tensor([-h/2, h/2])
    traces = k.new_tensor([2., 1.7])
    center = traces-(q0*z+slope*z**2/2+curvature*z**3/6)/k
    outer_z = 2*z
    outer = q0+slope*outer_z+curvature*outer_z**2/2
    flux = k.new_tensor([outer[0], q0, outer[1]])
    phase = torch.tensor([[1+.3j], [-.4j]], dtype=torch.complex128)
    left, right = layered_slab_contact_temperatures(
        phase*center, phase*flux, h, k, axial_flux_curvature=phase*curvature)
    torch.testing.assert_close(torch.cat((left, right), dim=-1), phase*traces, atol=1e-14, rtol=0.)


def test_storing_contact_retains_distinct_one_sided_material_fluxes():
    from tensorsph.heat_conduction_2d import layered_slab_contact_temperatures

    # Constant flux in each bulk. Their difference supplies sheet storage.
    dx = .4
    k = torch.tensor([1., 3.], dtype=torch.float64)
    flux = k.new_tensor([[2., 2.], [-.5, -.5]])
    traces = k.new_tensor([1.2, .9])
    centers = traces + dx/2 * k.new_tensor([2., .5])/k
    phase = torch.tensor([[1+.4j], [-.7j]], dtype=torch.complex128)
    left, right = layered_slab_contact_temperatures(phase*centers, phase[..., None]*flux, dx, k)
    torch.testing.assert_close(torch.cat((left, right), dim=-1), phase*traces)


@pytest.mark.parametrize("options", [{}, {"rannacher": True}, {"method": "sdirk2"}])
def test_fixed_wall_bulk_exchange_across_load_switch(options, record_property):
    from tensorsph.heat_conduction_2d import solve_transient_slab_2d

    dx, jump, end = .125, .125, .5
    x = (torch.arange(8, dtype=torch.float64)+.5)*dx
    initial = (305.+x).repeat(2, 1)
    pattern = torch.stack((x, -x))
    def exchange(t):
        return pattern*(1. if t < jump else -2.)

    # T=305+x+t gives unit temperature rate and constant rightward flux -1.
    # C=2, H=4. The external source compensates the patterned bulk exchange.
    field, heat, means, bulk, supplied = solve_transient_slab_2d(
        initial, [0., end], dx, (lambda t: 305.+t, lambda t: 306.+t),
        ambient_temperature=lambda t: initial+t+exchange(t)/4,
        heat_source=lambda t: 2.-exchange(t), heat_loss_coefficient=4.,
        volumetric_heat_capacity=2., wall_jump_times=[jump], temperature_reference=305.,
        return_x_heat=True, return_interval_temperatures=True, return_bulk_exchange=True, return_source_heat=True, **options)
    expected = (jump-2*(end-jump))*pattern
    # Rannacher uses backward-Euler endpoint quadrature on two half steps
    # in each segment; its reported stage mean includes that quadrature bias.
    mean_time = end/2
    if options.get("rannacher"):
        mean_time += (jump**2+(end-jump)**2)/(4*end)
    errors = dict(bulk_J_m3=(bulk[0]-expected).abs().max(),
                  source_J_m3=(supplied[0]-(2*end-expected)).abs().max(),
                  final_K=(field[-1]-x-end).abs().max(),
                  mean_K=(means[0]-x-mean_time).abs().max(),
                  section_J_m2=(heat+end).abs().max(),
                  budget_J_m3=(2*(field[-1]-field[0])-bulk[0]-supplied[0]
                               +heat[0].diff()/dx).abs().max())
    for name, error in errors.items():
        record_property(name, float(error))
        assert error < 5e-12


@pytest.mark.parametrize("method", ["theta", "sdirk2"])
def test_weak_source_heat_with_implicit_zero_ambient(method, record_property):
    from tensorsph.heat_conduction_2d import solve_transient_slab_2d

    # A weak scalar load on top of the much larger sink referenced to 0 K.
    # Its integral must retain the physical input and its sign on both intervals.
    initial = torch.full((2, 8), 305., dtype=torch.float64)
    power = 2.**-50
    times = [0., .25, .75]
    def heating(t):
        return power*(1.-4*t)
    _, supplied = solve_transient_slab_2d(
        initial, times, .125, (305., 305.), heat_source=heating,
        heat_loss_coefficient=4., method=method, return_source_heat=True)
    primitive = initial.new_tensor([t-2*t*t for t in times])
    expected = primitive.diff()[:, None, None].expand_as(supplied)
    error = float((supplied/power-expected).abs().max())
    record_property("source_error_divided_by_power_s", error)
    assert supplied.shape == (2, 2, 8) and error < 1e-14


def test_source_history_can_reuse_its_particle_buffer():
    from tensorsph.heat_conduction_2d import solve_transient_slab_2d

    initial = torch.zeros((2, 8), dtype=torch.float64)
    buffer = torch.empty_like(initial)
    def source(t):
        return buffer.fill_(t)
    field, supplied = solve_transient_slab_2d(
        initial, [0., 1.], .125, (None, None), heat_source=source, return_source_heat=True)
    torch.testing.assert_close(supplied[0], torch.full_like(initial, .5))
    torch.testing.assert_close(field[-1], supplied[0])


@pytest.mark.parametrize("options", [{}, {"rannacher": True}, {"method": "sdirk2"},
                                     {"method": "exponential"}, {"method": "exponential_quadratic"}])
def test_layered_source_heat_closes_each_column(options, record_property):
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    dx, jump = .125, .125
    k = torch.tensor([1.]*4+[4.]*4, dtype=torch.float64)
    c = k.new_tensor([2.]*4+[3.]*4)
    thickness = dx/k
    initial = thickness.cumsum(0)-thickness/2
    right = float(thickness.sum())
    times = [0., .25, .5]
    # Unit rightward temperature gradient in resistance coordinates gives
    # constant flux -1. The source and ambient together drive dT/dt=1.
    def exchange(t):
        return k*(1. if t < jump else -2.)
    source_buffer = torch.empty_like(k)
    def heating(t):
        return source_buffer.copy_(c-exchange(t))
    field, heat, bulk, supplied = solve_transient_layered_slab(
        initial, times, dx, (lambda t: t, lambda t: right+t), conductivity=k,
        volumetric_heat_capacity=c, heat_source=heating, heat_loss_coefficient=1.,
        bulk_ambient_temperature=lambda t: initial+t+exchange(t), load_jump_times=[jump],
        return_x_heat=True, return_bulk_exchange=True, return_source_heat=True, **options)
    expected_bulk = dx*k*k.new_tensor([jump-2*(.25-jump), -.5])[:, None]
    expected_storage = .25*dx*c
    errors = dict(source_J_m2=(supplied-(expected_storage-expected_bulk)).abs().max(),
                  exchange_J_m2=(bulk-expected_bulk).abs().max(),
                  section_J_m2=(heat+.25).abs().max(),
                  field_K=(field-initial-k.new_tensor(times)[:, None]).abs().max(),
                  budget_J_m2=(dx*c*field.diff(dim=0)+heat.diff(dim=1)-bulk-supplied).abs().max())
    for name, error in errors.items():
        record_property(name, float(error))
        assert error < 5e-12


def test_layered_source_heat_integrates_quadratic_midpoint():
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    k = torch.ones(8, dtype=torch.float64)
    state, supplied = solve_transient_layered_slab(
        torch.zeros_like(k), [0., .5], .125, (None, None), conductivity=k,
        volumetric_heat_capacity=1., heat_source=lambda t: t*t, method="exponential_quadratic",
        return_source_heat=True)
    torch.testing.assert_close(supplied[0], torch.full_like(k, .125*.5**3/3), atol=1e-15, rtol=0)
    torch.testing.assert_close(state[-1]*.125, supplied[0], atol=1e-14, rtol=0)


@pytest.mark.parametrize("phase", [0., .7])
def test_travelling_fixed_wall_cycle_heat_refinement(phase, record_property):
    from tensorsph.heat_conduction_2d import harmonic_fixed_wall_cycle_heat_2d, solve_harmonic_slab_2d

    # T=x*exp(i*2*pi*y+i*phase), k=C=1. Opposite wall flux phasors have
    # unit amplitude in every row and sum to zero along either periodic wall.
    omega = 3.
    previous = None
    for n in (16,32,64):
        dx = 1/n
        x = (torch.arange(n,dtype=torch.float64)+.5)*dx
        wave = torch.exp(1j*(2*math.pi*x+phase))
        exact = wave[:,None]*x
        walls = (0.,wave)
        field,sections = solve_harmonic_slab_2d(
            (1j*omega+(2*math.pi)**2)*exact,dx,walls,omega,return_x_flux=True)
        heat = harmonic_fixed_wall_cycle_heat_2d(field,dx,1.,torch.stack((sections[0],-sections[-1])),
                                                 omega,x_wall_temperatures=walls)
        error = float((heat-2/omega).abs().max())
        record_property(f"n{n}_max_cycle_heat_error_J_m2",error)
        assert float(sections[[0,-1]].abs().max()) < 1e-12
        if previous is not None:
            order = math.log2(previous/error)
            record_property(f"n{n}_order",order)
            assert order > 1.8
        previous = error


def test_biased_wall_cycle_heat_for_opposing_rows(record_property):
    from scipy.integrate import quad
    from tensorsph.heat_conduction_2d import directional_wall_cycle_heat, recover_dirichlet_wall_flux_2d

    # Linear-in-x temperature profiles, sampled at centres. Both the DC
    # slope and the AC phase vary by row. The right wall has opposite flux.
    bias = torch.tensor([-2.,-1.,-.375,0.,.375,1.,2.],dtype=torch.float64)
    phase = torch.arange(len(bias),dtype=torch.float64)*.5
    phasor = torch.exp(1j*phase)
    dx,omega = .125,2.5
    x = (torch.arange(8,dtype=torch.float64)+.5)*dx
    def recovered(slope):
        return recover_dirichlet_wall_flux_2d(slope[:,None]*x,dx,1.,
            torch.stack((-slope.mean(),slope.mean())),x_wall_temperatures=(0.,slope))
    inward = recovered(-phasor)
    steady = recovered(-bias)
    absorption,release = directional_wall_cycle_heat(inward,omega,steady_flux=steady)
    rows = []
    for b,p in zip(bias.tolist(),phase.tolist()):
        roots = ([(sign*math.acos(-b)-p)%(2*math.pi) for sign in (-1,1)] if abs(b)<1 else [])
        rows.append([quad(lambda t: max(sign*(b+math.cos(t+p)),0.),0.,2*math.pi,points=roots)[0]/omega
                     for sign in (1,-1)])
    reference = bias.new_tensor(rows).T
    error = max(float((absorption-reference).abs().max()),float((release-reference.flip((0,))).abs().max()))
    balance = float((absorption-release-2*math.pi*steady/omega).abs().max())
    record_property("max_directional_error_J_m2",error)
    record_property("max_net_error_J_m2",balance)
    assert error < 1e-12 and balance < 1e-14


def test_biased_wall_cycle_heat_resolves_grazing_reversal():
    from tensorsph.heat_conduction_2d import directional_wall_cycle_heat

    gap = 2.**-50
    _,released = directional_wall_cycle_heat(1.,1.,steady_flux=1-gap)
    assert abs(released.item()/(4*math.sqrt(2)/3*gap**1.5)-1)<3e-15


def _directional_heat_quadrature(wave, dc, omega, start, end):
    from scipy.integrate import quad

    phase = math.atan2(wave.imag, wave.real)
    amplitude = abs(wave)
    roots = []
    if abs(dc) < amplitude:
        angle = math.acos(-dc/amplitude)
        for root in (-angle, angle):
            first = math.ceil((omega*start+phase-root)/(2*math.pi))
            last = math.floor((omega*end+phase-root)/(2*math.pi))
            roots.extend((root-phase+2*math.pi*k)/omega for k in range(first, last+1))
    return [quad(lambda t: max(0., sign*(dc+amplitude*math.cos(omega*t+phase))),
                 start, end, points=roots)[0] for sign in (1, -1)]


def _traveling_wall_fluxes(n, omega):
    from tensorsph.heat_conduction_2d import (
        recover_dirichlet_wall_flux_2d, solve_harmonic_slab_2d, solve_steady_slab_2d,
    )

    # T_dc = -x*(0.2 + 1.4*cos(2πy)), T_ac = -x*exp(i*2πy).
    wavenumber = 2 * math.pi
    dx = 1 / n
    x = (torch.arange(n, dtype=torch.float64) + .5) * dx
    phase = wavenumber * x
    varying_bias = 1.4 * phase.cos()
    bias = .2 + varying_bias
    wave = torch.exp(1j * phase)
    steady_walls = (0., -bias)
    harmonic_walls = (0., -wave)
    steady, dc_sections = solve_steady_slab_2d(
        -wavenumber**2 * varying_bias[:, None] * x, dx, steady_walls, return_x_flux=True)
    harmonic, ac_sections = solve_harmonic_slab_2d(
        -(wavenumber**2 + 1j * omega) * wave[:, None] * x, dx, harmonic_walls, omega,
        return_x_flux=True)
    dc_flux = recover_dirichlet_wall_flux_2d(
        steady, dx, 1., torch.stack((dc_sections[0], -dc_sections[-1])),
        x_wall_temperatures=steady_walls)
    ac_flux = recover_dirichlet_wall_flux_2d(
        harmonic, dx, 1., torch.stack((ac_sections[0], -ac_sections[-1])),
        x_wall_temperatures=harmonic_walls)
    return bias, phase, wave, dc_flux, ac_flux


def test_biased_cycle_heat_from_steady_and_harmonic_slab_solves(record_property):
    from scipy.integrate import quad
    from tensorsph.heat_conduction_2d import (
        directional_wall_cycle_heat, directional_wall_interval_heat, harmonic_wall_interval_heat,
    )

    # Left inward fluxes equal the respective slopes with their signs reversed.
    omega = 3.
    previous_directional = {}
    interval_errors = []
    errors = []
    for n in (16, 32, 64):
        bias, phase, wave, dc_flux, ac_flux = _traveling_wall_fluxes(n, omega)
        absorbed, released = directional_wall_cycle_heat(ac_flux, omega, steady_flux=dc_flux)

        # A full-cycle integral is invariant under a phase shift. On the right
        # wall both fluxes reverse, exchanging the absorbed and released heat.
        reference = []
        for b in bias.tolist():
            cuts = [math.acos(-b), 2 * math.pi - math.acos(-b)] if abs(b) < 1 else []
            reference.append([
                quad(lambda a: max(sign * (b + math.cos(a)), 0.), 0., 2 * math.pi,
                     points=cuts)[0] / omega for sign in (1, -1)])
        left, right = bias.new_tensor(reference).T
        expected = torch.stack((left, right))
        error = max((absorbed - expected).abs().max().item(),
                    (released - expected.flip((0,))).abs().max().item())
        record_property(f"n{n}_max_directional_error_J_m2", error)
        if errors:
            order = math.log2(errors[-1] / error)
            record_property(f"n{n}_order", order)
            assert order > 1.8
        errors.append(error)

        start, end = -.13, .57
        heat = harmonic_wall_interval_heat(ac_flux, omega, start, end, steady_flux=dc_flux)
        reference = bias*(end-start) + ((phase+omega*end).sin()-(phase+omega*start).sin())/omega
        interval_error = (heat-torch.stack((reference,-reference))).abs().max().item()
        record_property(f"n{n}_interval_error_J_m2", interval_error)
        if interval_errors:
            order = math.log2(interval_errors[-1]/interval_error)
            record_property(f"n{n}_interval_order", order)
            assert order > 1.8
        interval_errors.append(interval_error)

        # Use analytic zero crossings of the exact traveling wave for each row.
        # Integrate the two signs independently, before averaging any rows.
        exact_rows = [_directional_heat_quadrature(w, b, omega, start, end)
                      for w, b in zip(wave.tolist(), bias.tolist())]
        left_absorbed, left_released = bias.new_tensor(exact_rows).T
        expected_absorbed = torch.stack((left_absorbed, left_released))
        absorbed, released = directional_wall_interval_heat(ac_flux, omega, start, end, steady_flux=dc_flux)
        for name, actual, target in (("absorbed", absorbed, expected_absorbed),
                                     ("released", released, expected_absorbed.flip((0,)))):
            differences = (actual-target).abs()
            # Cell centres move on refinement, changing the set of reversing rows.
            reversing = (left_absorbed > 0) & (left_released > 0)
            assert reversing.any()
            record_property(f"n{n}_reversing_{name}_error_J_m2", differences[:, reversing].max().item())
            error = differences.max().item()
            record_property(f"n{n}_partial_{name}_error_J_m2", error)
            if name in previous_directional:
                order = math.log2(previous_directional[name]/error)
                record_property(f"n{n}_partial_{name}_order", order)
                assert order > 1.8
            previous_directional[name] = error
    assert errors[-1] < .04


def test_directional_wall_interval_heat_against_split_quadrature(record_property):
    from tensorsph.heat_conduction_2d import directional_wall_interval_heat, harmonic_wall_interval_heat

    omega = 2.7
    flux = torch.tensor([0j, 1+.5j, -.3+1.2j, -1j, 1j], dtype=torch.complex128)
    bias = torch.tensor([.23, -2., .3, 1., -.4], dtype=torch.float64)
    worst, balance = 0., 0.
    for start, end in ((-.83, .21), (.2, 9.7), (1.1, 1.10000001), (2., 2.)):
        absorbed, released = directional_wall_interval_heat(flux, omega, start, end, steady_flux=bias)
        for j, (wave, dc) in enumerate(zip(flux.tolist(), bias.tolist())):
            references = _directional_heat_quadrature(wave, dc, omega, start, end)
            for reference, result in zip(references, (absorbed, released)):
                worst = max(worst, abs(result[j].item()-reference))
        net = harmonic_wall_interval_heat(flux, omega, start, end, steady_flux=bias)
        balance = max(balance, (absorbed-released-net).abs().max().item())
    record_property("directional_error_J_m2", worst)
    record_property("balance_error_J_m2", balance)
    assert max(worst, balance) < 1e-12

    gap = 2.**-50
    _, released = directional_wall_interval_heat(-1., 1., 0., .1, steady_flux=1-gap)
    assert abs(released.item()/(2*math.sqrt(2)/3*gap**1.5)-1) < 1e-12

def test_partial_wall_heat_at_fixed_reversing_rows(record_property):
    from tensorsph.heat_conduction_2d import directional_wall_interval_heat, directional_wall_interval_duration

    omega, start, end = 3., -.13, .57
    base = 16
    centres = (torch.arange(base, dtype=torch.float64)+.5)/base
    phase = 2*math.pi*centres
    reference = [_directional_heat_quadrature(w, b, omega, start, end)
                 for w, b in zip(torch.exp(1j*phase).tolist(), (.2+1.4*phase.cos()).tolist())]
    rows = centres.new_tensor(reference)
    reversing = (rows > 0).all(dim=1)
    assert reversing.any()
    record_property("reversing_positions", centres[reversing].tolist())
    exact = rows[reversing].T
    duration_reference = [_reference_wall_durations(w, b, omega, start, end)
                          for w, b in zip(torch.exp(1j*phase).tolist(), (.2+1.4*phase.cos()).tolist())]
    durations = centres.new_tensor(duration_reference)[reversing].T
    assert (durations > 0).all() and (durations < end-start).all()
    previous = {}
    for factor in (1, 3, 9):
        n = base*factor
        # An odd refinement factor retains every original particle centre.
        indices = (factor*torch.arange(base) + factor//2)[reversing]
        _, _, _, dc_flux, ac_flux = _traveling_wall_fluxes(n, omega)
        absorbed, released = directional_wall_interval_heat(
            ac_flux[:, indices], omega, start, end, steady_flux=dc_flux[:, indices])
        inward, outward = directional_wall_interval_duration(
            ac_flux[:, indices], omega, start, end, steady_flux=dc_flux[:, indices])
        balance = (inward+outward-(end-start)).abs().max().item()
        record_property(f"n{n}_duration_sum_error_s", balance)
        assert balance < 1e-14
        for name, actual, target, unit in (
                ("inward_duration", inward, durations, "s"),
                ("outward_duration", outward, durations.flip((0,)), "s"),
                ("absorbed", absorbed, exact, "J_m2"),
                ("released", released, exact.flip((0,)), "J_m2")):
            error = (actual-target).abs().max().item()
            record_property(f"n{n}_{name}_error_{unit}", error)
            if name in previous:
                order = math.log(previous[name]/error, 3)
                record_property(f"n{n}_{name}_order", order)
                assert order > 1.8
            previous[name] = error


def test_wall_reversal_times_at_fixed_rows(record_property):
    from scipy.optimize import brentq
    from tensorsph.heat_conduction_2d import directional_wall_cycle_duration, harmonic_wall_reversal_times

    omega = 3.
    period = 2*math.pi/omega
    # Reference roots bracketed by consecutive extrema of the exact flux.
    base = 16
    indices = torch.tensor([3, 4, 10, 11])
    phase = (indices.to(torch.float64)+.5)*2*math.pi/base
    roots = []
    for phi in phase.tolist():
        def inward(t):
            return .2+1.4*math.cos(phi)+math.cos(omega*t+phi)
        peak = -phi/omega
        roots.append([brentq(inward, peak-period/2, peak)%period,
                      brentq(inward, peak, peak+period/2)%period])
    rising, falling = phase.new_tensor(roots).T
    expected = torch.stack((rising, falling))
    previous = {}
    for factor in (1, 3, 9):
        n = base*factor
        selected = factor*indices + factor//2
        _, _, _, dc, ac = _traveling_wall_fluxes(n, omega)
        absorption, release = harmonic_wall_reversal_times(ac[:,selected], omega, steady_flux=dc[:,selected])
        inward_time, outward_time = directional_wall_cycle_duration(ac[:,selected], omega, steady_flux=dc[:,selected])
        positive = torch.remainder(expected.flip((0,))-expected, period)
        for label, actual, target in (("inward_duration", inward_time, positive),
                                      ("outward_duration", outward_time, period-positive),
                                      ("absorption", absorption, expected),
                                      ("release", release, expected.flip((0,)))):
            if label.endswith("duration"):
                difference = actual-target
                metric = f"{label}_error_s"
            else:
                difference = torch.remainder(actual-target+period/2, period)-period/2
                metric = f"{label}_onset_error_s"
            error = difference.abs().max().item()
            record_property(f"n{n}_{metric}", error)
            assert math.isfinite(error)
            if label in previous:
                order = math.log(previous[label]/error, 3)
                record_property(f"n{n}_{label}_order", order)
                assert order > 1.8
            previous[label] = error


def test_wall_reversal_times_skip_tangencies_and_constant_flux():
    from tensorsph.heat_conduction_2d import harmonic_wall_reversal_times

    for times in harmonic_wall_reversal_times([1j, -1., 0., .5], 2., steady_flux=[-1., 1., 0., 3.]):
        assert torch.isnan(times).all()
    rising, falling = harmonic_wall_reversal_times(-1j, 2.)
    assert rising.item() == pytest.approx(0., abs=1e-15)
    assert falling.item() == pytest.approx(math.pi/2)


def test_wall_cycle_durations_at_grazing_and_without_crossings(record_property):
    from tensorsph.heat_conduction_2d import directional_wall_cycle_duration

    omega = 3.
    period = 2*math.pi/omega
    gap = 2.**-52
    # Use unit phasors with the short window centered on or away from phase zero.
    flux = torch.tensor([-1., 1j, 1., 1., 0., 0., 1j], dtype=torch.complex128)
    bias = torch.tensor([1-gap, -(1-gap), 1., -1., 0., 2., 0.], dtype=torch.float64)
    absorbed, released = directional_wall_cycle_duration(flux, omega, steady_flux=bias)
    short = 2*math.acos(1-gap)/omega
    expected = bias.new_tensor([[period-short, short, period, 0., 0., period, period/2],
                                [short, period-short, 0., period, 0., 0., period/2]])
    error = (torch.stack((absorbed, released))-expected).abs().max().item()
    relative = max(abs(released[0].item()/short-1), abs(absorbed[1].item()/short-1))
    record_property("grazing_duration_s", short)
    record_property("max_error_s", error)
    record_property("grazing_relative_error", relative)
    assert error < 1e-15 and relative < 1e-15


def _reference_wall_durations(flux, bias, omega, start, end):
    """Partition the exact waveform at all sign changes in the observation window."""
    if abs(bias) >= abs(flux):
        return (end-start if bias > 0 else 0., end-start if bias < 0 else 0.)
    phase = math.atan2(flux.imag, flux.real)
    period = 2*math.pi/omega
    angle = math.acos(-bias/abs(flux))
    cuts = [start, end]
    for k in range(math.floor(start/period)-2, math.ceil(end/period)+2):
        for root in (-angle, angle):
            t = (root-phase+2*math.pi*k)/omega
            if start < t < end:
                cuts.append(t)
    cuts.sort()
    durations = [0., 0.]
    for left, right in zip(cuts, cuts[1:]):
        q = bias + abs(flux)*math.cos(phase+omega*(left+right)/2)
        durations[int(q < 0)] += right-left
    return durations


def test_directional_wall_interval_duration(record_property):
    from tensorsph.heat_conduction_2d import directional_wall_interval_duration

    omega = 2.3
    period = 2*math.pi/omega
    fluxes = [1+0j, -.3+.7j, -1j, 0j, 1j, 0j]
    biases = [.3, -.2, 1., 0., -2., .7]
    worst = 0.
    for start, end in ((-.63,.51), (.12,4.7*period), (period-.01,period+.02), (1.,1.)):
        inward, outward = directional_wall_interval_duration(fluxes, omega, start, end, steady_flux=biases)
        for j, (flux,bias) in enumerate(zip(fluxes,biases)):
            durations = _reference_wall_durations(flux, bias, omega, start, end)
            worst = max(worst, abs(inward[j].item()-durations[0]), abs(outward[j].item()-durations[1]))
    record_property("max_duration_error_s",worst)
    assert worst < 1e-13

    # Clip a grazing release window on both sides of the cycle origin.
    gap = 2.**-52
    half = math.acos(1-gap)/omega
    inward, outward = directional_wall_interval_duration(-1., omega, -2*half, half/2, steady_flux=1-gap)
    expected = 1.5*half
    record_property("clipped_release_s",outward.item())
    relative = abs(outward.item()/expected-1)
    record_property("clipped_relative_error",relative)
    assert relative < 1e-14 and abs(inward.item()/half-1) < 1e-14


def test_layered_fixed_wall_pulse_mean_temperature(record_property):
    """Compare mean particle temperatures across and after a wall pulse."""
    import numpy as np
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    shutoff, start, end = .023, .04, .08
    wave = np.pi*np.arange(1, 81)
    rate = wave**2
    # Difference of two wall step responses, integrated over the cooling window.
    mean_amplitude = (2/wave * np.exp(-rate*(start-shutoff)) * -np.expm1(-rate*shutoff)
                      * -np.expm1(-rate*(end-start))/(rate*(end-start)))
    # Over [0,start] the integral of the two steady profiles lasts only
    # until shutoff. The remaining Fourier terms decay for at least start-shutoff.
    pulse_correction = 2*np.exp(-rate*(start-shutoff))*(-np.expm1(-rate*shutoff))/wave**3
    errors = []
    for n in (12, 24, 48):
        _, mean = solve_transient_layered_slab(
            np.zeros(n), [0., start, end], 1/n, (lambda t: float(t < shutoff), 0.),
            conductivity=np.ones(n), volumetric_heat_capacity=1.,
            load_jump_times=[shutoff], method="exponential", return_interval_temperature=True)
        centers = (np.arange(n)+.5)/n
        basis = np.sin(centers[:, None]*wave)
        reference = basis @ mean_amplitude
        pulse_reference = (shutoff*(1-centers) - basis @ pulse_correction)/start
        pulse_error = float(np.max(np.abs(mean[0].numpy()-pulse_reference)))
        record_property(f"n{n}_max_cross_switch_mean_error_K", pulse_error)
        error = float(np.max(np.abs(mean[-1].numpy()-reference)))
        errors.append(np.array([error, pulse_error]))
        record_property(f"n{n}_max_cooling_mean_error_K", error)
    for coarse, fine in zip(errors, errors[1:]):
        assert np.all(coarse > 3.4*fine)


def test_layered_fixed_wall_startup_mean_temperature(record_property):
    """Mean temperature from the initial wall step through the end of heating."""
    import numpy as np
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    duration = .023
    wave = np.pi*np.arange(1, 81)
    tail = 2*np.exp(-wave**2*duration)/wave**3
    # Integrating the step response leaves a nondecaying sine series.
    # Its sum solves -v'' = 1-x with zero wall values: v=x*(1-x)*(2-x)/6.
    bulk_reference = .5 - (1/24 - (1-np.cos(wave))/wave @ tail)/duration
    errors = []
    for n in (12, 24, 48):
        _, mean = solve_transient_layered_slab(
            np.zeros(n), [0., duration], 1/n, (1., 0.), conductivity=np.ones(n),
            volumetric_heat_capacity=1., method="exponential", return_interval_temperature=True)
        x = (np.arange(n)+.5)/n
        primitive = x*(1-x)*(2-x)/6
        reference = 1-x - (primitive - np.sin(x[:, None]*wave) @ tail)/duration
        field = mean[0].numpy()
        probe_error = float(np.max(np.abs(field-reference)))
        bulk_error = abs(field.mean()-bulk_reference)
        errors.append(np.array([probe_error, bulk_error]))
        record_property(f"n{n}_max_startup_mean_error_K", probe_error)
        record_property(f"n{n}_bulk_startup_mean_error_K", bulk_error)
    for coarse, fine in zip(errors, errors[1:]):
        assert np.all(coarse > 3.4*fine)


def test_opposing_wall_ramps_particle_temperatures(record_property):
    """A zero bulk mean can hide local errors during antisymmetric heating."""
    import numpy as np
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    end = .04
    slope = 1/end
    # Lift the linear wall data with slope*t*(1-2*x). Only even sine
    # modes remain in the homogeneous-wall response to the constant forcing.
    wave = 2*np.pi*np.arange(1, 257)
    rate = wave**2
    decay_integral = -np.expm1(-rate*end)/rate
    # Fourier's law applied to the time-integrated profile at x=1/2.
    # Here v'(1/2)=-1/12 and the cosine takes alternating signs.
    center_heat = slope*(end**2-end/12-np.cos(wave/2) @ (4*decay_integral/rate))
    record_property("reference_center_heat_J_m2", center_heat)
    errors = []
    heat_errors = []
    for n in (12, 24, 48):
        states, heat, means = solve_transient_layered_slab(
            np.zeros(n), [0., end], 1/n, (lambda t: slope*t, lambda t: -slope*t),
            conductivity=np.ones(n), volumetric_heat_capacity=1.,
            method="exponential", return_x_heat=True, return_interval_temperature=True)
        x = (np.arange(n)+.5)/n
        basis = np.sin(x[:, None]*wave)
        # -v'' = 1-2*x, v(0)=v(1)=0. Removing this polynomial leaves
        # an exponentially small endpoint tail and a fifth-order mean tail.
        v = x*(1-x)*(1-2*x)/6
        final_ref = slope*(end*(1-2*x)-v+basis @ (4*np.exp(-rate*end)/wave**3))
        mean_ref = slope*(end*(1-2*x)/2-v+basis @ (4*decay_integral/(end*wave**3)))
        final = states[-1].numpy()
        center = heat[0, n//2].item()
        west = heat[0, 0].item()
        # Unit volumetric capacity and initially uniform zero temperature.
        balance = abs(west-center-final[:n//2].sum()/n)
        heat_errors.append(abs(center-center_heat))
        record_property(f"n{n}_center_heat_error_J_m2", heat_errors[-1])
        record_property(f"n{n}_west_half_balance_J_m2", balance)
        assert balance < 1e-12
        mean = means[0].numpy()
        assert max(abs(final.mean()), abs(mean.mean())) < 1e-12
        local = np.array([np.max(np.abs(final-final_ref)), np.max(np.abs(mean-mean_ref))])
        errors.append(local)
        record_property(f"n{n}_endpoint_max_error_K", local[0])
        record_property(f"n{n}_time_mean_max_error_K", local[1])
    for coarse, fine in zip(errors, errors[1:]):
        assert np.all(coarse > 3*fine)
    assert np.all(errors[-1] < .003)
    for coarse, fine in zip(heat_errors, heat_errors[1:]):
        assert coarse > 3*fine
    assert heat_errors[-1] < .01*center_heat


def test_equal_wall_ramp_and_hold_heat_storage(record_property):
    """Both walls supply heat during the ramp and continue to do so during the hold."""
    import numpy as np
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    ramp_end, end = .04, .09
    wave = np.pi*np.arange(1, 80, 2)

    def stored_under_ramp(t):
        # Integrate the unit slab's odd sine modes over space. The constant
        # 1/12 is the integral of x*(1-x)/2, the steady lag for a unit ramp.
        return (t-1/12+np.sum(8*np.exp(-wave**2*t)/wave**4))/ramp_end

    stored = [0., stored_under_ramp(ramp_end),
              stored_under_ramp(end)-stored_under_ramp(end-ramp_end)]
    inward = np.diff(stored)/2
    wall = lambda t: min(t/ramp_end, 1.)
    errors = []
    for phase, reference in zip(("ramp", "hold"), inward):
        record_property(f"{phase}_wall_reference_J_m2", reference)
    for n in (12, 24, 48):
        states, heat = solve_transient_layered_slab(
            np.zeros(n), [0., ramp_end, end], 1/n, (wall, wall),
            conductivity=np.ones(n), volumetric_heat_capacity=1.,
            method="exponential", return_x_heat=True)
        west, east = heat[:, 0].numpy(), -heat[:, -1].numpy()
        uptake = np.diff(states.numpy().mean(axis=1))
        balance = np.max(np.abs(west+east-uptake))
        center_heat = heat[:, n//2].abs().max().item()
        assert max(balance, center_heat) < 1e-12
        record_property(f"n{n}_max_balance_J_m2", balance)
        wall_error = np.maximum(np.abs(west-inward), np.abs(east-inward))
        for phase, error in zip(("ramp", "hold"), wall_error):
            record_property(f"n{n}_{phase}_wall_error_J_m2", error)
        errors.append(wall_error)
    for coarse, fine in zip(errors, errors[1:]):
        assert np.all(coarse > 3*fine)
    assert np.all(errors[-1] < .01*inward)


def test_matched_effusivity_interface_ramp_and_hold(record_property):
    """Native section heat through a perfectly bonded two-material slab."""
    import numpy as np
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    ramp_end, end = .04, .09
    # The resistance coordinate y=integral(dx/k) spans [0,1]. Since k*C=1,
    # T_t=T_yy and the rightward flux at the physical midpoint is -T_y(.25).
    y = .25
    wave = np.pi*np.arange(1, 80, 2)

    def ramp_heat(t):
        # Integral of the odd-mode flux for a continuing linear wall ramp.
        tail = np.sum(4*np.cos(wave*y)*np.exp(-wave**2*t)/wave**4)
        return (t*(.5-y)-1/24+y*y/4-y**3/6+tail)/ramp_end

    expected = np.diff([0., ramp_heat(ramp_end), ramp_heat(end)-ramp_heat(end-ramp_end)])
    wall = lambda t: min(t/ramp_end, 1.)
    errors = []
    for phase, value in zip(("ramp", "hold"), expected):
        record_property(f"{phase}_interface_reference_J_m2", value)
    for n in (12, 24, 48):
        mid = n//2
        k = np.repeat([2., 2/3], mid)
        states, heat = solve_transient_layered_slab(
            np.zeros(n), [0., ramp_end, end], 1/n, (wall, wall),
            conductivity=k, volumetric_heat_capacity=1/k,
            method="exponential", return_x_heat=True)
        stored = states.numpy()/(n*k)
        west, through, east = heat.numpy()[:, [0, mid, -1]].T
        for side, net, layer in (("west", west-through, stored[:, :mid]),
                                 ("east", through-east, stored[:, mid:])):
            residual = np.max(abs(net-np.diff(layer.sum(axis=1))))
            record_property(f"n{n}_{side}_balance_J_m2", residual)
            assert residual < 1e-12
        error = abs(through-expected)
        for phase, value in zip(("ramp", "hold"), error):
            record_property(f"n{n}_{phase}_interface_error_J_m2", value)
        errors.append(error)
    for coarse, fine in zip(errors, errors[1:]):
        assert np.all(coarse > 3*fine)
    assert np.all(errors[-1] < .01*expected)


def test_resistive_decay_mode_interface_traces_and_heat(record_property):
    """Recover the two temperatures at a resistive cut from interval particle means."""
    import numpy as np
    from tensorsph.heat_conduction_2d import (
        layered_slab_contact_temperatures, solve_transient_layered_slab,
    )

    wave = 1.5*np.pi
    times = np.array([0., .025, .05])
    dt = np.diff(times)
    decay_integral = np.exp(-wave**2*times[:-1])*(-np.expm1(-wave**2*dt))/wave**2
    # With k=C=3 on the east, its reflected mode has amplitude -1/3:
    # both materials give the same positive rightward flux at the contact.
    traces = np.sin(wave/2)*np.array([1., -1/3])
    flux = -wave*np.cos(wave/2)
    resistance = (traces[0]-traces[1])/flux
    exact_heat = flux*decay_integral
    exact_means = (decay_integral/dt)[:, None]*traces
    errors = []
    for n in (12, 24, 48):
        mid = n//2
        dx = 1/n
        x = (np.arange(n)+.5)*dx
        k = np.repeat([1., 3.], mid)
        initial = np.where(x < .5, np.sin(wave*x), -np.sin(wave*(1-x))/3)
        contacts = np.zeros(n-1)
        contacts[mid-1] = resistance
        states, heat, means = solve_transient_layered_slab(
            initial, times, dx, (0., 0.), conductivity=k, volumetric_heat_capacity=k,
            contact_resistance_x=contacts, method="exponential", return_x_heat=True,
            return_interval_temperature=True)
        left, right = layered_slab_contact_temperatures(means, heat.numpy()/dt[:, None], dx, k)
        observed = np.column_stack((left[:, mid-1], right[:, mid-1]))
        transfer = heat[:, mid].numpy()
        uptake = dx*np.diff(states[:, :mid].numpy() @ k[:mid])
        balance = max(abs(heat[:, 0].numpy()-transfer-uptake))
        assert balance < 1e-12
        record_property(f"n{n}_left_balance_J_m2", balance)
        error = np.r_[abs(observed-exact_means).max(axis=0), max(abs(transfer-exact_heat))]
        errors.append(error)
        for label, value in zip(("west_trace_K", "east_trace_K", "contact_heat_J_m2"), error):
            record_property(f"n{n}_max_error_{label}", value)
    for coarse, fine in zip(errors, errors[1:]):
        assert np.all(coarse > 3*fine)
    assert np.all(errors[-1] < [1e-3, 1e-3, 1e-4])


@pytest.mark.parametrize("method", ["theta", "sdirk2", "exponential", "exponential_quadratic"])
def test_native_directional_section_heat_affine_reversal(method, record_property):
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    n = 8
    x = (torch.arange(n, dtype=torch.float64)+.5)/n
    # T=(1-t)*(1-x) yields a spatially constant rightward flux 1-t.
    # With nearest-column support, ghost reflection reproduces the linear field.
    times = [0., .5, 1.5, 2.]
    _, net, parts = solve_transient_layered_slab(
        1-x, times, 1/n, (lambda t: 1-t, 0.), conductivity=torch.ones_like(x),
        volumetric_heat_capacity=1., heat_source=x-1, h_factor=.9, method=method,
        return_x_heat=True, return_directional_x_heat=True)
    for name, values in (("rightward", [.375, .125, 0.]), ("leftward", [0., .125, .375])):
        error = (parts[name]-x.new_tensor(values)[:, None]).abs().max().item()
        record_property(f"{name}_max_error_J_m2", error)
        assert error < 1e-12
    torch.testing.assert_close(parts["rightward"]-parts["leftward"], net, rtol=0., atol=1e-13)


def test_directional_section_heat_keeps_hidden_neumann_pulse():
    from tensorsph.heat_conduction_2d import solve_transient_layered_slab

    def pulse(t):
        return -3. if .25 <= t < .5 else 1.

    initial = torch.zeros(8, dtype=torch.float64)
    _, parts = solve_transient_layered_slab(
        initial, [0., 1.], 1/8, (None, 0.), conductivity=torch.ones_like(initial),
        volumetric_heat_capacity=1., x_wall_inward_flux=(pulse, 0.), load_jump_times=[.25, .5],
        method="sdirk2", return_directional_x_heat=True)
    assert parts["rightward"][0, 0].item() == pytest.approx(.75, abs=1e-13, rel=0.)
    assert parts["leftward"][0, 0].item() == pytest.approx(.75, abs=1e-13, rel=0.)
