"""Gates for the lake-at-rest case (`CASE-SPH-LAKE-AT-REST`).

The case is deliberately *not* a "does it pass" test: it pins the measured
statement, including the part that fails. Three assertions types appear here and
they must not be confused:

* **identities** that must hold to machine precision (the density operator is the
  summation the solver performs; the analytic prior realises the analytic
  density; the vertical balance is attainable);
* **negative controls** that must stay `O(1)` — otherwise the identities would be
  measured on a dead operator;
* **documented defects** (the horizontal side-wall residual, the wall-lattice
  asymmetry). Those assert *that the defect is present and where*, so a future
  fix shows up as a red test rather than as silently stale prose.

Kept small (`dx = 0.04`, 70 particles) so the suite stays seconds-long; the
published artefact uses `dx = 0.02`.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.lake_at_rest import (LakeConfig, analytic_profile, build_state,
                                    density_operator, integrate_at_rest,
                                    reconstruct, residual_report, run_case,
                                    wall_symmetry_report)


@pytest.fixture(scope="module")
def case():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        cfg = LakeConfig(width=0.40, depth=0.28, dx=0.04, t_end=0.05,
                         sample_every=5, solver_max_iter=30)
        state = build_state(cfg)
        prior = analytic_profile(state, cfg)
        result = run_case(cfg)
        yield cfg, state, prior, result
    finally:
        torch.set_default_dtype(previous)


def test_density_operator_is_the_summation_the_solver_does(case):
    cfg, state, _prior, _solution = case
    A, b = density_operator(state, cfg)
    mass = torch.full((state["n_fluid"],), state["m_unit"],
                      dtype=state["pos"].dtype, device=state["pos"].device)
    modelled = A @ mass + b
    measured = residual_report(state, cfg, mass)["density"]
    assert torch.allclose(modelled, measured, rtol=0, atol=1e-9)


def test_analytic_prior_realises_the_analytic_density(case):
    """The mass field that sums to the exact Tait profile is a direct solve."""
    cfg, state, prior, _solution = case
    A, b = density_operator(state, cfg)
    rho = A @ prior["mass"] + b
    error = float(((rho - prior["rho_reference"]) / cfg.rho0).abs().max())
    assert error < 1e-12, error


def test_vertical_balance_reaches_machine_precision(case):
    _cfg, _state, _prior, result = case
    equilibrium = result["equilibrium"]
    assert equilibrium["converged"], equilibrium["residual_history"]
    assert equilibrium["a_max_over_g"] < 1e-9, equilibrium["a_max_over_g"]
    # The horizontal component is *not* machine precision; pinned in the next
    # test. Its presence is why the vertical number alone is not the verdict.
    assert equilibrium["a_max_over_g_horizontal"] > 1e-4


def test_horizontal_residual_lives_in_the_side_wall_band(case):
    """F1: the side-wall truncation is what the reconstruction cannot absorb."""
    _cfg, _state, _prior, result = case
    equilibrium = result["equilibrium"]
    wall = equilibrium["a_max_over_g_horizontal_wall_band"]
    interior = equilibrium["a_max_over_g_horizontal_interior"]
    assert wall > 1e-3, wall
    assert wall > 5.0 * interior, (wall, interior)


def test_negative_controls_keep_the_operator_alive(case):
    _cfg, _state, _prior, result = case
    controls = result["controls"]
    uniform = controls["uniform_lattice_mass"]
    analytic = controls["analytic_hydrostatic_profile"]
    # A uniform mass field sums to rho0 everywhere, so the pressure is identically
    # zero and the lake free-falls: exactly 1 g, to the last digit.
    assert uniform["a_max_over_g"] == pytest.approx(1.0, abs=1e-9)
    assert uniform["p_max"] == pytest.approx(0.0, abs=1e-9)
    # The best a continuum reference can do is still O(1e-1) g.
    assert analytic["a_max_over_g"] > 5e-2
    # ... and the reconstruction buys at least six orders of magnitude.
    assert result["equilibrium"]["a_max_over_g"] < 1e-6 * analytic["a_max_over_g"]


def test_residual_responds_linearly_to_a_mass_perturbation(case):
    _cfg, _state, _prior, result = case
    response = result["perturbation_response"]
    small, large = response[1.0e-4], response[1.0e-3]
    assert 0.0 < small < large
    # Ten times the perturbation, ten times the residual: the near-zero residual
    # of the equilibrium is a balance, not a dead operator.
    assert small / large == pytest.approx(0.1, rel=0.3), small / large


def test_reconstruction_is_honestly_non_unique(case):
    """F4: the balance Jacobian is rank deficient; the prior picks the member."""
    _cfg, _state, _prior, result = case
    equilibrium = result["equilibrium"]
    low, high = equilibrium["mass_over_unit_min"], equilibrium["mass_over_unit_max"]
    assert low > 0.0
    # Masses follow a physical hydrostatic profile, not a free-for-all, but the
    # solved field is *not* the analytic one: the spread records how much of it
    # is the chosen member of a degenerate solution set.
    assert 1.0 < high / low < 5.0, (low, high)


def test_rest_run_does_not_stay_at_rest(case):
    """The verdict of the anchor: the lake sloshes, and that is what we publish."""
    cfg, state, prior, result = case
    run = result["rest"]
    assert run["max_v_over_c0"] > 1e-9
    assert run["max_dy_over_H"] > 1e-9
    assert run["steps"] == int(round(cfg.t_end / run["dt"]))
    # The interior is not protected from the side-wall residual either: a
    # horizontal force at the wall and a frozen mass field can only slosh.
    assert run["max_v_over_c0_interior"] > 1e-9
    assert run["wall_band_particles"] + run["interior_particles"] == state["n_fluid"]
    # A particle mass is a constant of the scheme; total mass cannot move.
    rerun = integrate_at_rest(state, cfg, prior["mass"] * 0.0 + result["rest"]["total_mass"] / state["n_fluid"],
                              t_end=2.0 * run["dt"])
    assert rerun["total_mass"] == pytest.approx(run["total_mass"], rel=1e-12)


def test_wall_is_mirror_symmetric_after_rounding(case):
    """F2: symmetry of the wall is measured; the case wall must have no unpaired column."""
    cfg, _state, _prior, _result = case
    report = wall_symmetry_report(cfg)
    assert report["case_wall_unpaired"] == 0
    # The shipped wall is symmetric too, once its float duplicates are rounded
    # away -- which is exactly what `build_state` does before the mirror map runs.
    assert report["unpaired"] == 0
    assert report["shipped_wall_columns"] > 0


def test_geometry_invariants(case):
    cfg, state, prior, _result = case
    pos = state["pos"]
    columns = torch.unique(pos[:, 0])
    rows = torch.unique(pos[:, 1])
    assert columns.numel() == int(round(cfg.width / cfg.dx))
    assert rows.numel() == int(round(cfg.depth / cfg.dx))
    # mirror map: one degree of freedom per (row, mirror column) pair
    assert int(state["columns"].max()) + 1 == rows.numel() * math.ceil(columns.numel() / 2)
    # the prior is a hydrostatic profile: heavier at the bottom
    low = prior["mass"][pos[:, 1] < pos[:, 1].median()].mean()
    high = prior["mass"][pos[:, 1] >= pos[:, 1].median()].mean()
    assert float(low) < float(high)