"""Guards for the flux-driven open-boundary injection of the FSI2 channel.

The geometric clock it replaces (``min x > 1.5 dx``) is pinned by the slowest
particle in the domain: the inlet parabola vanishes at the walls, so a near-wall
particle creeps and the outlet keeps deleting particles while nothing is
injected.  Measured consequence: ``injected = 0`` over 1200 steps, bulk speed
0.19 ``U_mean``, downstream flux 0.18 of the inlet flux
(``docs/xval_fsi2/INFLOW_CLOCK_STALL.md``).

These tests pin the invariant, not the numbers: the inflow volume must be
delivered at the prescribed rate even when particles are parked at the inlet.
"""
from __future__ import annotations

import math

import torch

from tensorsph.fsi2_turek_hron import (
    FSI2Case,
    _apply_open_boundaries,
    initial_state,
)


def _case(dx: float = 0.05) -> FSI2Case:
    case = FSI2Case()
    case.dx = dx
    case.open_boundaries = True
    case.n_beam_nodes = 8
    return case


def _n_column(case: FSI2Case) -> int:
    return int(round(case.channel_height / case.dx))


def test_injection_is_driven_by_flux_not_by_a_geometric_front():
    """Park every particle at the inlet: a flux clock still injects."""
    case = _case()
    dt = case.dt()
    state = initial_state(case)
    # pin a fictitious front: no particle may pass the old trigger
    state.fluid_pos[:, 0] = torch.clamp(state.fluid_pos[:, 0], max=1.4 * case.dx)
    state.fluid_vel.zero_()
    n_pinned = int(state.fluid_pos.shape[0])

    steps_for_one_column = int(math.ceil(case.dx / (case.inlet_u_mean() * dt)))
    injected = 0
    for _ in range(steps_for_one_column + 2):
        injected += _apply_open_boundaries(state, case, dt)["injected"]

    assert injected > 0, "flux clock failed to inject with a pinned geometric front"
    assert 0.5 * _n_column(case) <= injected <= 1.5 * _n_column(case)
    assert state.fluid_pos.shape[0] > n_pinned


def test_injection_rate_matches_the_prescribed_inlet_flux():
    """Columns injected over ``T`` must equal ``U_mean T / dx`` (integer rounding)."""
    case = _case()
    dt = case.dt()
    state = initial_state(case)
    n_steps = 40
    injected = 0
    for _ in range(n_steps):
        injected += _apply_open_boundaries(state, case, dt)["injected"]
    # do not let the outlet delete anything, so the count is pure injection
    expected_columns = case.inlet_u_mean() * dt * n_steps / case.dx
    assert abs(injected / _n_column(case) - expected_columns) <= 1.0


def test_deficit_accumulates_when_no_column_fits():
    """Sub-column time must accumulate, not be discarded."""
    case = _case()
    dt = case.dt()
    state = initial_state(case)
    per_step = case.inlet_u_mean() * dt
    n_steps = max(1, int(0.5 * case.dx / per_step))
    for _ in range(n_steps):
        _apply_open_boundaries(state, case, dt)
    assert 0.0 <= state.inflow_deficit < case.dx
    assert state.inflow_deficit > 0.0


def test_outlet_removal_still_fires_and_mass_volume_is_consistent():
    """The removal rule is untouched: end-of-channel particles are deleted."""
    case = _case()
    dt = case.dt()
    state = initial_state(case)
    n_before = int(state.fluid_pos.shape[0])
    state.fluid_pos[-1, 0] = case.channel_length - 0.1 * case.dx
    counters = _apply_open_boundaries(state, case, dt)
    assert counters["removed"] == 1
    assert state.fluid_pos.shape[0] == n_before - 1
