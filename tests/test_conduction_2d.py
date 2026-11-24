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
from pathlib import Path

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
