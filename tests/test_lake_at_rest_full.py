"""Tests for the lake-at-rest horizontal residual analysis.

These tests document the horizontal residual defect (F1) and verify the
convergence behavior of the side-wall truncation error.
"""
from __future__ import annotations

import pytest
import torch

from tensorsph.lake_at_rest import (LakeConfig, analytic_profile, build_state,
                                    reconstruct, residual_report)


@pytest.fixture(scope="module")
def case():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        cfg = LakeConfig(width=0.40, depth=0.28, dx=0.04, t_end=0.05,
                         sample_every=5, solver_max_iter=30)
        state = build_state(cfg)
        prior = analytic_profile(state, cfg)
        result = reconstruct(state, cfg, prior=prior["mass"])
        yield cfg, state, prior, result
    finally:
        torch.set_default_dtype(previous)


def test_original_reconstruction_achieves_vertical_machine_precision(case):
    """G1: the original reconstruction achieves machine precision for vertical balance."""
    _cfg, _state, _prior, result = case
    assert result["a_max_over_g"] < 1e-9, result["a_max_over_g"]


def test_horizontal_residual_at_side_walls(case):
    """F1: the horizontal residual lives in the side-wall band.

    This is the documented defect: the mirror-symmetric mass field cannot
    cancel the horizontal force at side walls because the force is odd under
    mirror reflection.
    """
    _cfg, _state, _prior, result = case
    wall_band = result["a_max_over_g_horizontal_wall_band"]
    interior = result["a_max_over_g_horizontal_interior"]
    # Wall band residual is O(0.1)
    assert wall_band > 1e-2, f"wall band residual too small: {wall_band}"
    # Interior is much smaller
    assert interior < wall_band / 5, f"interior not smaller: {interior} vs {wall_band}"


def test_horizontal_residual_scales_with_dx():
    """The horizontal residual should decrease with finer resolution.

    This documents the convergence behavior of the side-wall truncation error.
    """
    results = {}
    for dx in [0.08, 0.04, 0.02]:
        torch.set_default_dtype(torch.float64)
        cfg = LakeConfig(width=0.40, depth=0.28, dx=dx, t_end=0.05,
                         sample_every=5, solver_max_iter=30)
        state = build_state(cfg)
        prior = analytic_profile(state, cfg)
        result = reconstruct(state, cfg, prior=prior["mass"])
        results[dx] = result["a_max_over_g_horizontal_wall_band"]

    # The residual should decrease with dx (roughly first order)
    assert results[0.04] < results[0.08], "residual should decrease with dx"
    assert results[0.02] < results[0.04], "residual should decrease with dx"


def test_negative_controls_still_valid(case):
    """The negative controls (uniform mass, analytic profile) still show O(1) residuals."""
    _cfg, state, _prior, _result = case
    from tensorsph.lake_at_rest import run_case
    cfg = LakeConfig(width=0.40, depth=0.28, dx=0.04, t_end=0.05, sample_every=5)
    full_result = run_case(cfg)
    uniform = full_result["controls"]["uniform_lattice_mass"]
    analytic = full_result["controls"]["analytic_hydrostatic_profile"]
    # Uniform mass: free fall, exactly 1g
    assert uniform["a_max_over_g"] == pytest.approx(1.0, abs=1e-6)
    # Analytic profile: O(0.1)g
    assert analytic["a_max_over_g"] > 0.05
