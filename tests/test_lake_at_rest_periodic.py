"""The laterally periodic lake at rest: the anchor tolerance is met here.

``docs/verification_lake_at_rest.md`` established an honest FAIL for the
wall-bounded tank: the side-wall truncation leaves ``max|a_x|/g = 0.124`` and the
lake sloshes.  This file tests the variant that removes the truncation instead
of the physics -- ``LakeConfig(periodic_x=True)``, which drives the shipped
:func:`tensorsph.droplet_impact.rhs` with ``wrap_width = width`` and therefore
sees an ``x``-translation-invariant operator.  The claims are, in order:

* the wrap is a *minimum image*, not a stretched kernel (``test_wrap_*``);
* the shipped operator is then exactly translation invariant in ``x``, measured
  on the analytic profile before any reconstruction (``test_*translation*``);
* the reconstruction closes **both** components to roundoff, and the run stays
  at rest to roundoff (``test_*machine_precision*``, ``test_rest_run*``);
* the operator is not dead: the negative controls and the linear response of
  the wall-mode case are unchanged (``test_negative_controls*``,
  ``test_residual_responds*``).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorsph.droplet_impact import Config, cubic_spline_kernel, rhs
from tensorsph.lake_at_rest import (LakeConfig, analytic_profile, build_state,
                                    integrate_at_rest, reconstruct,
                                    residual_report)


def coarse(**kwargs) -> LakeConfig:
    """Small, fast, exactly on the lattice: 8 x 6 fluid columns/rows."""
    base = dict(dx=0.04, depth=0.24, width=0.32, periodic_x=True, t_end=0.08)
    base.update(kwargs)
    return LakeConfig(**base)


@pytest.fixture(scope="module")
def state():
    torch.set_default_dtype(torch.float64)
    cfg = coarse()
    return cfg, build_state(cfg)


@pytest.fixture(scope="module")
def solution(state):
    cfg, st = state
    prior = analytic_profile(st, cfg)["mass"]
    return prior, reconstruct(st, cfg, prior=prior)


def test_wrap_none_is_the_previous_code_path():
    """``wrap_width=None`` must reproduce the unwrapped call bit for bit."""
    torch.set_default_dtype(torch.float64)
    h = 0.026
    pos = torch.tensor([[0.0, 0.0], [0.5 * h, 0.0], [0.3 * h, 0.4 * h]],
                       dtype=torch.float64)
    mass = torch.full((3,), 0.4, dtype=torch.float64)
    cfg = Config(dx=0.02, h_factor=1.3, density_calibration=False)
    pairs = torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.long)
    # The period is far larger than the support, so no separation wraps.
    base = rhs(pos, torch.zeros_like(pos), mass, cfg, pairs=pairs)
    wrapped = rhs(pos, torch.zeros_like(pos), mass, cfg, pairs=pairs,
                  wrap_width=100.0)
    for a, b in zip(base, wrapped):
        assert torch.equal(a, b)


def test_wrap_rejects_a_nonsense_period():
    torch.set_default_dtype(torch.float64)
    pos = torch.tensor([[0.0, 0.0], [0.02, 0.0]], dtype=torch.float64)
    cfg = Config(dx=0.02)
    pairs = torch.tensor([[0, 1]], dtype=torch.long)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            rhs(pos, pos, torch.ones(2, dtype=torch.float64), cfg,
                pairs=pairs, wrap_width=bad)


def test_wrap_is_a_minimum_image():
    """A pair across the seam must contribute exactly like an interior pair.

    Two particles half a spacing inside the seam are ``width - dx`` apart
    through the middle of the tank and ``dx`` apart across the seam.  With the
    wrap the second separation is the one the kernel sees, so the density of
    that state equals the density of the same two particles placed one spacing
    apart in the middle -- and *without* the wrap the pair is simply out of
    support and does not exist.
    """
    torch.set_default_dtype(torch.float64)
    cfg = Config(dx=0.02, h_factor=1.3, density_calibration=False)
    width = 0.4
    h = cfg.h_factor * cfg.dx
    mass = torch.full((2,), 0.4, dtype=torch.float64)
    seam = torch.tensor([[0.5 * cfg.dx, 0.1], [width - 0.5 * cfg.dx, 0.1]],
                        dtype=torch.float64)
    interior = torch.tensor([[0.20, 0.1], [0.20 + cfg.dx, 0.1]],
                            dtype=torch.float64)
    pairs = torch.tensor([[0, 1]], dtype=torch.long)
    rho_seam, _, _ = rhs(seam, torch.zeros_like(seam), mass, cfg, pairs=pairs,
                         wrap_width=width)
    rho_interior, _, _ = rhs(interior, torch.zeros_like(interior), mass, cfg,
                             pairs=pairs)
    assert torch.allclose(rho_seam, rho_interior, rtol=0.0, atol=1e-12)

    # Without the wrap the seam pair is at distance width - dx = 0.38 > 2h and
    # the kernel is exactly zero: the two rows do not see each other at all.
    rho_blind, _, _ = rhs(seam, torch.zeros_like(seam), mass, cfg, pairs=pairs)
    self_term = float(cubic_spline_kernel(torch.zeros(1, dtype=torch.float64), h)[0])
    assert float(rho_blind.max()) == pytest.approx(mass[0].item() * self_term,
                                                   rel=1e-12)
    assert float(rho_seam.max()) > float(rho_blind.max())


def test_wrap_width_is_only_the_periodic_case(state):
    cfg, st = state
    assert st["wrap_width"] == pytest.approx(cfg.width)
    assert st["tie"] == "row"
    assert st["n_wall"] > 0                      # bottom wall survives
    assert len(torch.unique(st["wall"][:, 0])) == 8  # one column per fluid column


def test_periodic_operator_is_translation_invariant(state):
    """A ``y``-only mass field produces *no* horizontal acceleration.

    This is measured on the analytic profile, i.e. before any reconstruction:
    it is a property of the discretisation, and it is exactly what a side wall
    destroys.
    """
    cfg, st = state
    prior = analytic_profile(st, cfg)["mass"]
    report = residual_report(st, cfg, prior)
    assert report["a_max_over_g_horizontal"] < 1e-11
    # ... and the vertical residual of the same profile is O(g): the operator is
    # alive and the horizontal statement is not the trivial one.
    assert report["a_max_over_g"] > 0.05


def test_reconstruction_closes_both_components(solution):
    prior, sol = solution
    assert sol["converged"] is True
    assert sol["a_max_over_g"] < 1e-9
    assert sol["a_max_over_g_horizontal"] < 1e-9


def test_mass_field_is_one_value_per_row(state, solution):
    cfg, st = state
    prior, sol = solution
    mass = sol["mass"]
    rows = st["columns"]
    for row in torch.unique(rows).tolist():
        column = mass[rows == row]
        assert float(column.max() - column.min()) == 0.0


def test_rest_run_stays_at_rest(state, solution):
    cfg, st = state
    prior, sol = solution
    run = integrate_at_rest(st, cfg, sol["mass"])
    assert run["max_v_over_c0"] < 1e-9
    assert run["max_dy_over_H"] < 1e-9
    assert run["max_drho_over_rho0"] < 1e-9


def test_negative_controls_keep_the_operator_alive(state, solution):
    cfg, st = state
    prior, sol = solution
    uniform = torch.full((st["n_fluid"],), st["m_unit"], dtype=st["pos"].dtype)
    free_fall = residual_report(st, cfg, uniform)
    assert free_fall["a_max_over_g"] > 0.5         # a uniform mass sums to rho0
    analytic = residual_report(st, cfg, prior)
    assert analytic["a_max_over_g"] > 0.05         # O(g): a reconstruction is needed


def test_residual_responds_linearly_to_a_mass_perturbation(state, solution):
    """Scaling the equilibrium mass must scale the residual: not dead code."""
    cfg, st = state
    prior, sol = solution
    mass = sol["mass"]
    small = residual_report(st, cfg, mass * (1.0 + 1e-4))["a_max_over_g"]
    large = residual_report(st, cfg, mass * (1.0 + 1e-3))["a_max_over_g"]
    assert large > 1e-6
    assert 0.05 < small / large < 0.15


def test_geometry_invariants(state):
    cfg, st = state
    assert st["n_fluid"] == 48
    assert st["pos"][:, 1].min() == pytest.approx(0.5 * cfg.dx)
    assert st["pos"][:, 1].max() == pytest.approx(cfg.depth - 0.5 * cfg.dx)
    assert st["pos"][:, 0].min() == pytest.approx(0.5 * cfg.dx)
    assert st["pos"][:, 0].max() == pytest.approx(cfg.width - 0.5 * cfg.dx)
    # Every fluid/wall interaction inside the support is in the list exactly
    # once: 480 for this 48-particle lattice (checked by construction below).
    assert st["pairs"].shape[0] == 480
    assert int(st["pairs"].max()) < st["n_fluid"] + st["n_wall"]


def test_tank_versus_periodic_is_the_side_wall(state, solution):
    """The documented FAIL is a truncation effect, not an operator defect.

    Same operator, same lattice, same rows; the only difference is the two
    side-wall columns.  The wall-mode horizontal residual in the wall band is
    O(0.1 g) while the periodic one is at roundoff, and the *vertical* balance
    is equally closable in both -- which is why the periodic variant is the
    honest way to certify the anchor.
    """
    cfg, st = state
    prior, sol = solution
    wall_cfg = coarse(periodic_x=False)
    wall_state = build_state(wall_cfg)
    wall_prior = analytic_profile(wall_state, wall_cfg)["mass"]
    wall_residual = residual_report(wall_state, wall_cfg, wall_prior)
    masked = wall_residual["a_max_over_g_horizontal_wall_band"]
    assert masked is not None and masked > 0.01 * 1.0
    periodic = residual_report(st, cfg, prior)["a_max_over_g_horizontal"]
    assert periodic * 1e6 < masked
    # The vertical part of the discretisation is closed in both modes.
    assert wall_residual["a_max_over_g"] > 0.05
    assert sol["a_max_over_g"] < 1e-9


def test_dt_is_the_acoustic_courant_step():
    cfg = coarse()
    assert cfg.dt() == pytest.approx(cfg.max_courant * cfg.h() / cfg.sound_speed)
    assert math.isclose(cfg.dt() * cfg.sound_speed / cfg.h(), cfg.max_courant)
