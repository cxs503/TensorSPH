"""Hydrostatic equilibrium test for the transport-velocity correction.

The TVC operator must **not** introduce spurious motion in a fluid at rest.
This is a fundamental requirement: a resting column with hydrostatic initial
conditions should remain at rest (up to the solver's intrinsic noise floor)
when the correction is applied every step.

This test is distinct from ``test_uniform_lattice_is_a_fixed_point`` in
``test_transport_velocity_correction.py``, which checks the operator on an
infinite lattice with no gravity or walls.  Here we test the *full solver
integration*: gravity, wall particles, hydrostatic density initialization,
boundary clamping, and the TVC position correction all active simultaneously.

Key finding (2026-11-14, XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION):
    The **bulk** scope (TVC applied to interior particles only, excluding
    free-surface particles) causes instability in the hydrostatic case:
    max_speed reaches 2.9 m/s and density compression exceeds 10%, compared
    to 0.38 m/s and 2.4% for the **all** scope at the same alpha=0.05.
    This is because correcting only the interior while leaving the surface
    uncorrected creates a mismatch that drives spurious motion.  The
    ``all`` scope is the safe default for general use.

Reference: Adami, Hu & Adams (2013) JCP 241:292-307, §3.1 — the
transport-velocity formulation is designed to preserve equilibrium states.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.dam_break import DamBreakConfig, run_case  # noqa: E402


def _hydrostatic_cfg(dx: float, steps: int, tvc_every: int = 1,
                     tvc_alpha: float = 0.05, device: str = "cpu",
                     tvc_scope: str = "all"):
    """A resting column: same geometry as the dam break but the gate stays closed."""
    return DamBreakConfig(
        dx=dx, dt=1.0e-4, steps=steps, output_every=max(1, steps // 4),
        device=device, column_width=0.2, column_height=0.4,
        pool_width=2.0, tank_height=0.75,
        hydrostatic_init=True,
        tvc_every=tvc_every,
        tvc_alpha=tvc_alpha,
        tvc_scope=tvc_scope,
    )


# The baseline (no TVC) resting column has intrinsic noise ~0.27 m/s due to
# the lattice initialization not being a perfect SPH equilibrium (defect F1,
# docs/verification_startup_quiescence.md).  TVC must not increase this by
# more than a factor of 2.
_BASELINE_MAX_SPEED = 0.27  # m/s, measured at dx=0.04, 200 steps


@pytest.mark.parametrize("dx", [0.04, 0.02])
@pytest.mark.parametrize("tvc_alpha", [0.05, 0.10])
def test_tvc_does_not_disturb_hydrostatic(dx, tvc_alpha):
    """A resting column with TVC (all scope) must stay near rest.

    The column is initialized with hydrostatic density and zero velocity.
    With TVC applied every step (scope='all'), the max velocity must not
    exceed 2x the baseline solver noise.
    """
    cfg = _hydrostatic_cfg(dx, steps=200, tvc_every=1, tvc_alpha=tvc_alpha,
                           tvc_scope="all")
    run = run_case(cfg)

    # Mass is exactly conserved (particle masses are constants).
    assert run.mass_drift == 0.0, f"mass drift = {run.mass_drift}"

    # Velocity must stay within 2x the baseline noise floor.
    speed_limit = 2.0 * _BASELINE_MAX_SPEED + 0.05  # 0.59 m/s
    assert run.max_speed < speed_limit, (
        f"max_speed = {run.max_speed:.4f} m/s exceeds {speed_limit:.2f} m/s "
        f"(2x baseline + tolerance; TVC alpha={tvc_alpha}, dx={dx})"
    )

    # Density compression must be small (hydrostatic, no dynamic compression).
    assert run.max_density_compression < 0.05, (
        f"max_density_compression = {run.max_density_compression:.4f} "
        f"exceeds 5% (TVC alpha={tvc_alpha}, dx={dx})"
    )

    # TVC shift must be bounded.
    h = cfg.h_factor * dx
    assert run.max_tvc_shift < 0.2 * h, (
        f"max_tvc_shift = {run.max_tvc_shift:.6f} m exceeds 0.2 h = {0.2 * h:.6f} m "
        f"(TVC alpha={tvc_alpha}, dx={dx})"
    )


def test_tvc_hydrostatic_vs_no_tvc():
    """TVC must not increase the resting-column noise beyond 2x the baseline.

    Run the same resting column with and without TVC; the max speed with TVC
    must not exceed 2x the baseline + tolerance.
    """
    dx = 0.04
    steps = 200

    # Baseline: no TVC
    cfg_base = DamBreakConfig(
        dx=dx, dt=1.0e-4, steps=steps, output_every=max(1, steps // 4),
        device="cpu", column_width=0.2, column_height=0.4,
        pool_width=2.0, tank_height=0.75,
        hydrostatic_init=True,
        tvc_every=0,
    )
    run_base = run_case(cfg_base)

    # With TVC (all scope, alpha=0.05)
    cfg_tvc = DamBreakConfig(
        dx=dx, dt=1.0e-4, steps=steps, output_every=max(1, steps // 4),
        device="cpu", column_width=0.2, column_height=0.4,
        pool_width=2.0, tank_height=0.75,
        hydrostatic_init=True,
        tvc_every=1,
        tvc_alpha=0.05,
        tvc_scope="all",
    )
    run_tvc = run_case(cfg_tvc)

    # TVC must not increase the noise by more than 2x.
    assert run_tvc.max_speed < 2.0 * run_base.max_speed + 0.05, (
        f"TVC max_speed = {run_tvc.max_speed:.4f} exceeds "
        f"2x baseline ({run_base.max_speed:.4f}) + tolerance"
    )


def test_tvc_bulk_scope_is_unstable_in_hydrostatic():
    """TVC with scope='bulk' causes instability in the hydrostatic case.

    This is a documented limitation: correcting only interior particles while
    leaving free-surface particles uncorrected creates a mismatch that drives
    spurious motion.  The all scope is the safe default.

    This test *expects* the bulk scope to fail the stability gate, documenting
    the limitation rather than hiding it.
    """
    dx = 0.04
    cfg = DamBreakConfig(
        dx=dx, dt=1.0e-4, steps=200, output_every=50,
        device="cpu", column_width=0.2, column_height=0.4,
        pool_width=2.0, tank_height=0.75,
        hydrostatic_init=True,
        tvc_every=1,
        tvc_alpha=0.05,
        tvc_scope="bulk",
        tvc_bulk_c=0.95,
    )
    run = run_case(cfg)

    # The bulk scope is known to be unstable: max_speed >> baseline.
    # We assert the instability is present (documenting the limitation).
    assert run.max_speed > 1.0, (
        f"bulk scope max_speed = {run.max_speed:.4f} should be > 1.0 m/s "
        f"(known instability; use scope='all' instead)"
    )
    assert run.max_density_compression > 0.05, (
        f"bulk scope compression = {run.max_density_compression:.4f} "
        f"should be > 5% (known instability)"
    )


def test_tvc_alpha_scaling_hydrostatic():
    """Larger alpha should produce larger (but still bounded) corrections.

    This verifies the alpha parameter has the expected monotone effect on the
    correction magnitude, even in the hydrostatic case where the grad-1
    inconsistency is small but non-zero (due to the lattice truncation at walls
    and the free surface).
    """
    dx = 0.04
    shifts = {}
    for alpha in [0.05, 0.10, 0.15]:
        cfg = _hydrostatic_cfg(dx, steps=100, tvc_every=1, tvc_alpha=alpha,
                               tvc_scope="all")
        run = run_case(cfg)
        shifts[alpha] = run.max_tvc_shift

    # Larger alpha -> larger shift (monotone).
    assert shifts[0.05] <= shifts[0.10] <= shifts[0.15], (
        f"shifts not monotone: {shifts}"
    )

    # Even at alpha=0.15, the shift must be bounded.
    h = cfg.h_factor * dx
    assert shifts[0.15] < 0.3 * h, (
        f"max shift at alpha=0.15 = {shifts[0.15]:.6f} exceeds 0.3 h = {0.3 * h:.6f}"
    )
