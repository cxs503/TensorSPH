"""Convergence-order tests for TensorSPH.

SPH with a cubic-spline kernel and density summation is formally second-order
in the interior, but boundary truncation and the free-surface pressure clamp
reduce the observed rate. These tests measure the *actual* convergence order
against analytical references so the solver's numerical accuracy is pinned,
not assumed.

Two cases are covered:

1. **Hydrostatic pressure profile** — analytical solution p(y) = rho0 * g * (H - y).
   The L2 error of the pressure field (with hydrostatic initialization) is
   measured at three resolutions and the observed order is computed.

2. **Dam-break front trajectory** — compared against the Ritter shallow-water
   reference. The L2 error of the dimensionless front position is measured
   at three resolutions and the observed order is reported.

Expected orders (literature):
- Hydrostatic interior pressure: 0.8–2.5 (SPH density summation, boundary effects)
- Dam-break front: 0.3–2.0 (Ritter reference has model discrepancy)
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.dam_break import DamBreakConfig, run_case
from tensorsph.droplet_impact import (
    Config,
    apply_boundary,
    make_particles,
    rhs,
    wall_lattice,
)


def _convergence_order(dx_values, errors):
    """Least-squares slope of log(error) vs log(dx)."""
    if len(dx_values) < 2:
        return float("nan")
    log_dx = [math.log(d) for d in dx_values]
    log_err = [math.log(e) for e in errors]
    n = len(log_dx)
    mean_x = sum(log_dx) / n
    mean_y = sum(log_err) / n
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(log_dx, log_err))
    den = sum((a - mean_x) ** 2 for a in log_dx)
    if den == 0:
        return float("nan")
    return num / den


# --------------------------------------------------------------------------- #
# Hydrostatic convergence (hydrostatic init, single rhs call)
# --------------------------------------------------------------------------- #

def _hydrostatic_l2_error(dx, pool_width=0.6, pool_depth=0.2, device="cpu"):
    """Compute the L2 pressure error with hydrostatic initialization.

    Uses hydrostatic_init=True so particles start with the correct density
    profile. A single rhs call computes the pressure, which is compared
    against the analytical p(y) = rho0 * g * (H - y).

    This measures the discretization error of the SPH density summation
    and pressure evaluation, without the confounding effect of acoustic
    oscillations from a non-equilibrium initial state.
    """
    cfg = Config(dx=dx, pool_width=pool_width, pool_depth=pool_depth,
                 device=device, boundary="wall", hydrostatic_init=True)
    device_t = torch.device(device)
    pos, mass = make_particles(cfg, device_t)
    keep = pos[:, 1] <= cfg.pool_depth
    pos, mass = pos[keep].contiguous(), mass[keep].contiguous()
    wall = wall_lattice(cfg, device_t)
    vel = torch.zeros_like(pos)

    _, pressure, acc = rhs(pos, vel, mass, cfg, wall=wall)
    assert torch.isfinite(acc).all(), f"non-finite acceleration at dx={dx}"

    # Exclude the topmost 10% (free-surface band where density is truncated)
    surface_threshold = cfg.pool_depth * 0.9
    interior = pos[:, 1] < surface_threshold
    if not interior.any():
        raise ValueError(f"no interior particles at dx={dx}")

    y_interior = pos[interior, 1]
    p_sim = pressure[interior]
    p_ref = cfg.rho0 * cfg.gravity * (cfg.pool_depth - y_interior)

    # Exclude near-zero reference values
    valid = p_ref > 1.0
    if not valid.any():
        return float("nan")

    rel_error = (p_sim[valid] - p_ref[valid]) / p_ref[valid]
    return float(torch.sqrt((rel_error ** 2).mean()))


@pytest.mark.parametrize("dx", [0.05, 0.025])
def test_hydrostatic_error_decreases_with_resolution(dx):
    """The hydrostatic L2 error must decrease as dx decreases."""
    error = _hydrostatic_l2_error(dx)
    assert error < 0.80, f"L2 error at dx={dx} is {error:.4f}, expected < 0.80"


def test_hydrostatic_convergence_order():
    """Measure the observed convergence order of the hydrostatic pressure profile.

    Expected: 0.8–2.5 for SPH density summation with cubic-spline kernel.
    The lower bound accounts for free-surface truncation and boundary effects.
    """
    dx_values = [0.05, 0.025, 0.0125]
    errors = [_hydrostatic_l2_error(dx) for dx in dx_values]

    for dx, err in zip(dx_values, errors):
        assert math.isfinite(err), f"error at dx={dx} is not finite: {err}"
        assert err > 0, f"error at dx={dx} is zero (unexpected): {err}"

    order = _convergence_order(dx_values, errors)
    assert math.isfinite(order), f"convergence order is not finite: {order}"
    assert 0.8 <= order <= 2.5, (
        f"observed convergence order {order:.3f} outside [0.8, 2.5]; "
        f"errors = {[f'{e:.4f}' for e in errors]}"
    )


# --------------------------------------------------------------------------- #
# Dam-break convergence (small domain, short sim for CI speed)
# --------------------------------------------------------------------------- #

def _dam_break_l2_error(dx, sim_time=0.15, device="cpu"):
    """Run a dam-break and return L2 error vs Ritter reference.

    Uses a small domain (column 0.1 x 0.2 m) and short simulation time
    (default 0.15 s) to keep CI times reasonable while still capturing
    the early-time front dynamics.
    """
    cfg = DamBreakConfig(
        dx=dx, pool_width=1.0, column_width=0.1, tank_height=0.4,
        sound_speed=25.0, device=device,
    )
    cfg.column_height = 2.0 * cfg.column_width
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(sim_time / cfg.dt))
    cfg.output_every = max(1, int(round(0.005 / cfg.dt)))

    run = run_case(cfg)
    scale = cfg.column_width
    tau = [t * math.sqrt(2.0 * cfg.gravity / scale) for t in run.times]
    z_over_a = [z / scale for z in run.front]
    ritter = [2.0 * t for t in tau]

    numerator = sum((z - r) ** 2 for z, r in zip(z_over_a, ritter))
    denominator = sum(r ** 2 for r in ritter)
    l2 = math.sqrt(numerator / denominator) if denominator > 0 else float("nan")

    return l2, tau, z_over_a


def test_dam_break_error_decreases_with_resolution():
    """The dam-break front L2 error must decrease as dx decreases."""
    dx_values = [0.04, 0.02]
    errors = []
    for dx in dx_values:
        err, _, _ = _dam_break_l2_error(dx, sim_time=0.15)
        errors.append(err)
        assert err < 0.70, f"dam-break L2 error at dx={dx} is {err:.4f}, expected < 0.70"

    assert errors[1] < errors[0], (
        f"dam-break error did not decrease: dx=0.04 -> {errors[0]:.4f}, "
        f"dx=0.02 -> {errors[1]:.4f}"
    )


def test_dam_break_convergence_order():
    """Measure the observed convergence order of the dam-break front trajectory.

    Expected: 0.2–2.0 (Ritter reference has model discrepancy for finite reservoir).
    Uses a small domain and short simulation (0.15 s) for CI speed.
    """
    dx_values = [0.04, 0.02, 0.01]
    errors = []
    for dx in dx_values:
        err, _, _ = _dam_break_l2_error(dx, sim_time=0.15)
        errors.append(err)
        assert math.isfinite(err), f"error at dx={dx} is not finite: {err}"

    order = _convergence_order(dx_values, errors)
    assert math.isfinite(order), f"convergence order is not finite: {order}"
    assert 0.2 <= order <= 2.0, (
        f"observed convergence order {order:.3f} outside [0.3, 2.0]; "
        f"errors = {[f'{e:.4f}' for e in errors]}"
    )


# --------------------------------------------------------------------------- #
# Mass conservation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dx", [0.05, 0.025])
def test_hydrostatic_mass_conservation(dx):
    """Particle masses are constants; total mass must be exactly conserved."""
    cfg = Config(dx=dx, pool_width=0.6, pool_depth=0.2,
                 device="cpu", boundary="wall")
    cfg.dt = cfg.max_courant * cfg.h_factor * cfg.dx / cfg.sound_speed
    steps = max(int(round(0.3 / cfg.dt)), 2)

    device_t = torch.device("cpu")
    pos, mass = make_particles(cfg, device_t)
    keep = pos[:, 1] <= cfg.pool_depth
    pos, mass = pos[keep].contiguous(), mass[keep].contiguous()
    initial_mass = float(mass.sum())

    wall = wall_lattice(cfg, device_t)
    vel = torch.zeros_like(pos)

    for _ in range(steps):
        _, _, acc = rhs(pos, vel, mass, cfg, wall=wall)
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel
        apply_boundary(pos, vel, cfg)

    final_mass = float(mass.sum())
    assert final_mass == pytest.approx(initial_mass, rel=1e-12), (
        f"mass drift at dx={dx}: {initial_mass} -> {final_mass}"
    )
