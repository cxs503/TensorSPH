"""Regression gate: the *default* curvature of the projection-SPH Young-Laplace
approach must pass the owner 15 % gate.

The board audit (``scripts/audit/m17736_board_audit.py``) flagged that
``laplace_isph_case()`` shipped with ``curvature_method="fit_quadratic"`` as its
default, which measures ``kappa ~ 30 %`` low on the *two-phase carrier* (the
colour shell is a smooth multi-layer transition, not the sharp free-surface
layer the quadratic fit was written for) and produces a ``-24.5 %`` jump that
fails the 15 % gate.  The XVAL harness only passed because ``run.py`` passed
``mls`` explicitly -- a report that covered up a regression.

This test pins the fix: the default must be the validated ``mls`` (``div n``)
operator and must clear the gate, so a future re-introduction of the broken
default fails loudly instead of being hidden by an explicit argument.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.laplace_isph import laplace_isph_case  # noqa: E402

torch.set_num_threads(2)
TOL = 0.15  # owner acceptance: < 15 % on the Young-Laplace jump


def test_default_curvature_is_mls():
    """The shipped default must be the validated ``mls`` (div n) operator."""
    rep = laplace_isph_case(radius=0.02, dx=0.001, sigma=0.072, rho0=1000.0,
                            outer=0.012, h_factor=1.3)
    assert rep["curvature_method"] == "mls"


def test_default_jump_within_gate():
    """Default configuration (no explicit curvature) must clear the 15 % gate."""
    rep = laplace_isph_case(radius=0.02, dx=0.001, sigma=0.072, rho0=1000.0,
                            outer=0.012, h_factor=1.3)
    assert abs(rep["dp_error_pct"]) < 100 * TOL
    # and be inside the 3 % claim line at the operating point
    assert abs(rep["dp_error_pct"]) < 3.0


def test_default_curvature_recovers_one_over_R():
    """The default curvature product must recover ``kappa R = 1``."""
    rep = laplace_isph_case(radius=0.02, dx=0.001, sigma=0.072, rho0=1000.0,
                            outer=0.012, h_factor=1.3)
    assert abs(rep["kappaR_delta_weighted"] - 1.0) < 0.05


def test_fit_quadratic_remains_a_documented_failing_option():
    """``fit_quadratic`` is still available but must stay a known-failing option.

    It is not removed (it is the correct operator for the sharp single-phase
    free surface), but on this two-phase carrier it must not be silently
    promoted back to a passing default: the jump it produces is outside the
    gate, and that is the documented defect (not a tuning choice).
    """
    rep = laplace_isph_case(radius=0.02, dx=0.001, sigma=0.072, rho0=1000.0,
                            outer=0.012, h_factor=1.3,
                            curvature_method="fit_quadratic")
    assert abs(rep["dp_error_pct"]) > 100 * TOL
