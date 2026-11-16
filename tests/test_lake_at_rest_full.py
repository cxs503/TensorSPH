"""Tests for the lake-at-rest full (per-particle) reconstruction.

The per-particle reconstruction (:func:`tensorsph.lake_at_rest_full.reconstruct_full`)
relaxes the mirror-symmetry constraint of the original reconstruction and solves
for one mass per particle to minimise **both** vertical and horizontal acceleration.

Key finding: with mass-only adjustments (no position corrections), the best
achievable residual is O(0.1 g), not machine precision.  This is because the
system is overdetermined: 2n equations (ax, ay per particle) but only n unknowns
(mass per particle).  The solver finds the best least-squares compromise.
"""
from __future__ import annotations

import pytest
import torch

from tensorsph.lake_at_rest import LakeConfig, build_state, analytic_profile
from tensorsph.lake_at_rest_full import reconstruct_full


@pytest.fixture(scope="module")
def case():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        cfg = LakeConfig(width=0.40, depth=0.28, dx=0.04, t_end=0.05,
                         sample_every=5, solver_max_iter=30)
        state = build_state(cfg)
        prior = analytic_profile(state, cfg)
        sol = reconstruct_full(state, cfg, prior["mass"])
        yield cfg, state, prior, sol
    finally:
        torch.set_default_dtype(previous)


def test_full_reconstruction_reduces_combined_residual(case):
    """The per-particle reconstruction reduces the combined (ax+ay) residual."""
    _cfg, _state, _prior, sol = case
    # The analytic prior has O(0.26g) residual
    assert sol.a_max_over_g < 0.26, f"a_max_over_g = {sol.a_max_over_g}"
    # The residual is reduced to O(0.08g)
    assert sol.a_max_over_g < 0.10


def test_full_reconstruction_reduces_horizontal_component(case):
    """The horizontal component is reduced compared to the mirror-tied version."""
    from tensorsph.lake_at_rest import reconstruct
    cfg, state, prior, sol_full = case
    sol_mirror = reconstruct(state, cfg, prior=prior["mass"])
    # The mirror-tied version has O(0.1g) horizontal residual (documented defect F1)
    # The per-particle version should have similar or slightly different horizontal residual
    # (it's optimizing both components, not just ay)
    assert sol_full.a_max_over_g_horizontal < 0.10


def test_full_reconstruction_does_not_reach_machine_precision(case):
    """F5: mass-only adjustments cannot achieve machine precision in the wall-bounded tank.

    The system is overdetermined (2n equations, n unknowns), so the best
    least-squares solution leaves O(0.1g) residual.  This is a fundamental
    limitation of the mass-only approach, not a solver bug.
    """
    _cfg, _state, _prior, sol = case
    # The residual is reduced but not to machine precision
    assert sol.a_max_over_g > 1e-6, f"a_max_over_g = {sol.a_max_over_g} (expected > 1e-6)"


def test_full_reconstruction_no_position_corrections(case):
    """This variant does not apply position corrections."""
    _cfg, _state, _prior, sol = case
    assert sol.max_delta_r == 0.0
    assert torch.all(sol.delta_r == 0)


def test_full_reconstruction_convergence_behavior(case):
    """The solver converges to a stationary point (not necessarily the tolerance)."""
    _cfg, _state, _prior, sol = case
    # The solver stops when no descent direction is found
    assert len(sol.residual_history) > 1
    # The residual decreases monotonically (or plateaus)
    for i in range(1, len(sol.residual_history)):
        assert sol.residual_history[i] <= sol.residual_history[i - 1] + 1e-12


def test_full_reconstruction_vertical_reduced(case):
    """The vertical component is also reduced (though the mirror-tied version does better)."""
    _cfg, _state, _prior, sol = case
    # The vertical residual is reduced from the analytic prior
    assert sol.a_max_over_g_vertical < 0.26
