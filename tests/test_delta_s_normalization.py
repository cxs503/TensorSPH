"""Delta-s normalization deficit analysis for the CSF surface-tension model.

The CSF delta function must satisfy ``int delta_s dn = 1``.  In the current
consistent implementation the measured value is 0.74-0.76 (dx = 1.5-3 mm),
i.e. a systematic -24% to -26% deficit.  This test suite:

1. **Guards against regression**: the integral must stay within [0.6, 1.15].
2. **Documents the deficit**: a dedicated test asserts the *measured* range
   so that the honest number is visible in the test output.
3. **Provides a target**: when the ghost-colour or local-Shepard fix lands,
   the threshold should be tightened to [0.9, 1.1].

See ``docs/verification_surface_tension_synthesis.md`` §3.1 for the root-cause
analysis and proposed fixes.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tensorsph.csf_consistent import csf_fields
from tensorsph.neighbor_search import cell_linked_pairs

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "csf_consistent"
RADIUS = 0.02
RHO0 = 1000.0


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("csf_bench", BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _delta_s_integral(pos, mass, rho0, delta_s, radius):
    """Radial integral of delta_s over the interface shell."""
    r = torch.linalg.vector_norm(pos, dim=1)
    m = (r > 0.6 * radius) & (r < 1.05 * radius)
    d_v = mass / rho0
    contrib = (delta_s[m] * d_v[m] / (2.0 * math.pi * r[m].clamp_min(1e-30))).sum()
    return float(contrib)


@pytest.mark.parametrize("dx", [0.003, 0.002, 0.0015])
def test_delta_s_integral_regression_guard(harness, dx):
    """The delta_s integral must not silently regress below the legacy value (0.33)."""
    cfg = harness.Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=20.0,
                         gamma=7.0, gravity=0.0, viscosity=0.01,
                         surface_tension_coefficient=7.2)
    h = cfg.h_factor * cfg.dx
    pos, mass = harness.hexagonal_droplet(RADIUS, dx, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    state = csf_fields(pos, mass, pairs, h, RHO0)
    integral = _delta_s_integral(pos, mass, RHO0, state.delta_s, RADIUS)
    # Must be well above legacy (0.33) and below the analytic ceiling (1.0 + margin)
    assert 0.6 < integral < 1.15, (
        f"delta_s integral {integral:.4f} at dx={dx} is outside [0.6, 1.15]; "
        "legacy was ~0.33, analytic target is 1.0")


@pytest.mark.parametrize("dx", [0.003, 0.002, 0.0015])
def test_delta_s_integral_documents_deficit(harness, dx):
    """Explicitly document the measured delta_s deficit (honest number)."""
    cfg = harness.Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=20.0,
                         gamma=7.0, gravity=0.0, viscosity=0.01,
                         surface_tension_coefficient=7.2)
    h = cfg.h_factor * cfg.dx
    pos, mass = harness.hexagonal_droplet(RADIUS, dx, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    state = csf_fields(pos, mass, pairs, h, RHO0)
    integral = _delta_s_integral(pos, mass, RHO0, state.delta_s, RADIUS)
    rel_error = (integral - 1.0) / 1.0 * 100.0
    # This test *documents* the deficit; it passes as long as the number is
    # in the historically observed range.  When the fix lands, tighten to
    # assert abs(rel_error) < 15.
    assert 0.65 < integral < 0.85, (
        f"delta_s integral {integral:.4f} ({rel_error:+.1f}%) at dx={dx} "
        f"is outside the expected deficit range [0.65, 0.85]; "
        "if this fails because the value improved, update the threshold "
        "and the synthesis document")


def test_delta_s_integral_with_fit_quadratic_curvature(harness):
    """Delta-s integral is independent of the curvature scheme (same grad_c)."""
    dx = 0.002
    cfg = harness.Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=20.0,
                         gamma=7.0, gravity=0.0, viscosity=0.01,
                         surface_tension_coefficient=7.2,
                         csf_curvature="fit_quadratic")
    h = cfg.h_factor * dx
    pos, mass = harness.hexagonal_droplet(RADIUS, dx, RHO0)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    state = csf_fields(pos, mass, pairs, h, RHO0,
                       curvature_scheme="fit_quadratic")
    integral = _delta_s_integral(pos, mass, RHO0, state.delta_s, RADIUS)
    # Same as mls scheme (delta_s depends on grad_c, not curvature)
    assert 0.65 < integral < 0.85, integral
