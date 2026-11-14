"""Tests for the FSI2 rod sub-step criterion and the strong-coupling stepper.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

The two claims under test are the ones the coupled FSI2 run depends on:

1. ``FSI2Case.rod_substeps_effective()`` must size the rod sub-cycling on the
   rod's *highest* mode (bending), not on the axial mode -- the axial-only
   criterion leaves the shortest bending mode unresolvable and the free beam
   gains 1e249 in energy over 200 zero-load steps at ``rho_s/rho_f = 1``.
2. The interface accelerators (Aitken, IQN-ILS) must cancel the fixed-point
   residual of an added-mass-like interface map with gain > 1, and
   ``step_fsi2_strong`` must reproduce the shipped step exactly when the
   iteration is switched off (so the new stepper cannot silently change the
   physics it is meant to stabilise).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorsph.fsi2_similarity import similarity_corrected_case
from tensorsph.fsi2_turek_hron import (
    beam_advance,
    beam_energy,
    beam_initial_nodes,
    beam_max_frequency,
    initial_state,
    step_fsi2,
)
from tensorsph.fsi2_strong_coupling import (
    AitkenRelaxation,
    IQNILS,
    step_fsi2_strong,
)


def _case(ratio: float = 1.0, dx: float = 0.05, c0: float = 5.0):
    case = similarity_corrected_case(U_mean=0.2)
    case.dx = dx
    case.sound_speed = c0
    case.beam_rho = 1000.0 * ratio
    return case


# --------------------------------------------------------------------------- #
# 1. Rod sub-cycling criterion
# --------------------------------------------------------------------------- #

def test_rod_substeps_cover_highest_mode():
    """dt_sub * omega_max <= rod_stability_target for every case (derived gate)."""
    for ratio in (1.0, 2.0, 10.0):
        case = _case(ratio)
        n_sub = case.rod_substeps_effective()
        bounded = case.dt() * beam_max_frequency(case) / n_sub
        assert bounded <= case.rod_stability_target + 1e-12, (ratio, n_sub, bounded)


def test_rod_substeps_ratio_one_differs_from_axial_only():
    """The axial-only count is demonstrably too small at rho_s/rho_f = 1."""
    case = _case(1.0)
    axial_only = case.dt() * case.beam_axial_frequency() / case.rod_stability_target
    assert case.rod_substeps_effective() > max(1.0, math.ceil(axial_only - 1e-12))


@pytest.mark.parametrize("ratio", [1.0, 10.0])
def test_free_beam_energy_bounded_at_auto_substeps(ratio):
    """A freely vibrating clamped beam must not gain energy (zero load)."""
    case = _case(ratio, dx=0.04)
    dt = case.dt()
    n_sub = case.rod_substeps_effective()
    nodes = beam_initial_nodes(case, torch.device("cpu")).clone()
    n = nodes.shape[0]
    nodes[:, 1] += 1.0e-5 * torch.sin(
        math.pi * torch.arange(n, dtype=torch.float64) / (n - 1))
    nodes[0, 1] = case.cylinder_y
    vel = torch.zeros_like(nodes)
    load = torch.zeros_like(nodes)
    e0 = float(beam_energy(nodes, case))
    for _ in range(120):
        for _ in range(n_sub):
            nodes, vel = beam_advance(nodes, vel, load, case, dt / n_sub)
        assert math.isfinite(float(beam_energy(nodes, case)))
    ratio_e = float(beam_energy(nodes, case)) / e0
    assert ratio_e <= 1.0 + 1e-9, (ratio, n_sub, ratio_e)


def test_explicit_override_still_wins():
    case = _case(1.0)
    case.rod_substeps = 3
    assert case.rod_substeps_effective() == 3


# --------------------------------------------------------------------------- #
# 2. Interface accelerators
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("gain", [1.5, 4.35, 20.0])
@pytest.mark.parametrize("factory", [AitkenRelaxation, IQNILS])
def test_accelerator_cancels_added_mass_map(gain, factory):
    """Fixed point x -> x* + g (x - x*): both accelerators must converge."""
    xstar = np.array([1.0, -2.0, 0.5])
    acc = factory()
    x = xstar + np.array([1.0e-3, -2.0e-3, 0.5e-3])
    err0 = np.linalg.norm((gain - 1.0) * (x - xstar))
    err = err0
    for _ in range(8):
        r = (gain - 1.0) * (x - xstar)
        x, _ = acc.next(x, r)
        err = np.linalg.norm((gain - 1.0) * (x - xstar))
        if err == 0.0:
            break
    assert err <= 1.0e-9 * err0, (gain, factory.__name__, err / err0)


def test_aitken_can_produce_negative_relaxation():
    """The added-mass regime needs a negative factor; Aitken must allow it."""
    xstar = np.zeros(2)
    acc = AitkenRelaxation()
    x = np.array([1.0, -1.0])
    used = []
    for _ in range(4):
        r = 4.0 * x  # gain 4.35 map, r = (g-1)(x - x*)
        x, w = acc.next(x, r)
        used.append(w)
    assert min(used) < 0.0


# --------------------------------------------------------------------------- #
# 3. Strong stepper
# --------------------------------------------------------------------------- #

def test_strong_step_matches_loose_step_without_iteration():
    """max_iter=1 with no accelerator must reproduce ``step_fsi2`` bit-for-bit."""
    c1, c2 = _case(1.0), _case(1.0)
    s1, s2 = initial_state(c1), initial_state(c2)
    r1 = step_fsi2(s1, c1, c1.dt())
    r2, stats = step_fsi2_strong(s2, c2, c2.dt(), interface_solver="none",
                                 max_iter=1)
    assert stats.iterations == 1
    for key in ("tip_x", "tip_y", "drag", "lift", "max_speed"):
        assert r1[key] == pytest.approx(r2[key], rel=1e-12, abs=1e-15)
    assert torch.allclose(s1.beam_nodes, s2.beam_nodes)


def test_strong_step_reduces_interface_residual():
    """IQN-ILS must reach the predeclared residual-reduction gate."""
    case = _case(1.0, dx=0.05)
    state = initial_state(case)
    _, stats = step_fsi2_strong(state, case, case.dt(), interface_solver="iqn-ils",
                                max_iter=15, tol=1e-2)
    assert stats.converged
    assert stats.residual_rel <= 1e-2


def test_loose_run_at_benchmark_density_ratio_stays_bounded():
    """Regression on the round's headline result.

    With the bending-aware rod sub-cycling and the acoustic wall-impedance term
    off, the shipped loose driver must stay bounded at ``rho_s/rho_f = 1``
    (before this round it reached 3.5e50 m/s inside 100 steps).  The impedance
    variant is deliberately not exercised here; it is the documented
    destabiliser (docs/xval_fsi2/STRONG_COUPLING_AND_ROD_STABILITY.md).
    """
    case = _case(1.0, dx=0.02)  # the probe resolution of the reported experiment
    case.wall_impedance = False
    state = initial_state(case)
    dt = case.dt()
    for k in range(60):
        row = step_fsi2(state, case, dt)
        v = float(state.beam_vel.norm(dim=1).max())
        assert math.isfinite(v) and math.isfinite(row["max_speed"])
        # measured 5.4e-3 m/s at step 60 (drag-driven scale is ~3e-2 m/s);
        # the gate is the "not the 1e50 blow-up" regression
        assert v < 0.3, (k, v)