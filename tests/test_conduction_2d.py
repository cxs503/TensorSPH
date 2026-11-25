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


@pytest.mark.parametrize("wall,flux,film",[(1.,0.,(1.,2.)),(None,.3,(1.,2.)),(None,0.,(-1.,2.)),
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
