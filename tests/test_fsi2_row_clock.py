"""Guards for the *per-streamline* (per-row) inflow clock of the FSI2 channel.

Round ``INFLOW_CLOCK_STALL.md`` replaced the geometric injection trigger by a
single **global** flux clock driven by ``U_mean``.  That delivers the right
total volume but distributes it uniformly over the inlet rows, while the inlet
profile ``u(y) = U_max 4 y (H - y) / H^2`` delivers ~7x less volume on the
near-wall rows (``u = 0.14 U_mean`` at ``y = dx/2``).  The surplus particles
have nowhere to go: they stack up in the first lattice spacing, and the pile-up
is a density/pressure spike, not through-flow.

Measured on the rigid-body precursor at ``dx = 0.02`` (800 steps, same run,
only the clock changed -- ``benchmarks/xval_fsi2/rigid_cfd_precursor.py``
``FSI2_INFLOW_CLOCK``):

| clock | particles in the first spacing | max in one row cell | max abs u |
|---|---|---|---|
| global (control) | 49 | 6 | 2.298 m/s |
| per-row (this file) | 12 | 1 | 0.956 m/s |

These tests pin the *invariant* (no row is fed faster than its own flux), not
the numbers above.
"""
from __future__ import annotations

import math

import torch

from tensorsph.fsi2_turek_hron import FSI2Case, _apply_open_boundaries, initial_state


def _case(dx: float = 0.05) -> FSI2Case:
    case = FSI2Case()
    case.dx = dx
    case.open_boundaries = True
    case.n_beam_nodes = 8
    case.inlet_ramp_time = 0.0     # isolate the clock from the start-up ramp
    return case


def _rows(case: FSI2Case) -> torch.Tensor:
    return torch.arange(0.5 * case.dx, case.channel_height, case.dx,
                        dtype=torch.float64)


def _row_u(case: FSI2Case, ys: torch.Tensor) -> torch.Tensor:
    H = case.channel_height
    return case.inlet_u_max * 4.0 * ys * (H - ys) / (H * H)


def _empty_state_with_one_far_particle(case: FSI2Case):
    """Nothing near the inlet or the outlet, so counts are pure injections."""
    state = initial_state(case)
    keep = torch.zeros(state.fluid_pos.shape[0], dtype=torch.bool)
    state.fluid_pos = state.fluid_pos[:1].clone()
    state.fluid_vel = state.fluid_vel[:1].clone()
    state.fluid_mass = state.fluid_mass[:1].clone()
    state.fluid_pos[0, 0] = 0.5 * case.channel_length
    state.fluid_pos[0, 1] = 0.5 * case.channel_height
    del keep
    return state


def test_each_row_injects_at_its_own_rate():
    """Injection count per row must follow ``u(y_j)``, not ``U_mean``."""
    case = _case()
    dt = case.dt()
    state = _empty_state_with_one_far_particle(case)
    ys = _rows(case)
    n_steps = int(4.0 * case.dx / (case.inlet_u_mean() * dt))
    for _ in range(n_steps):
        _apply_open_boundaries(state, case, dt)
        state.time += dt
    t = n_steps * dt
    expect = _row_u(case, ys) * t / case.dx           # fractional injections
    got = torch.zeros_like(expect)
    y_inj = state.fluid_pos[1:, 1]                    # skip the far particle
    for j, y in enumerate(ys):
        got[j] = float((y_inj - y).abs().lt(1e-9).sum())
    err = (got - expect).abs()
    assert float(err.max()) < 1.0, (got.tolist(), expect.tolist())
    # the near-wall row must be fed *less* than the centre row
    assert float(got[0]) < float(got[len(ys) // 2])
    ratio = float(expect[0] / expect[len(ys) // 2])
    # parabola: u(y0)/u(y_mid) = y0 (H - y0) / (y_mid (H - y_mid))
    H = case.channel_height
    y0, ym = float(ys[0]), float(ys[len(ys) // 2])
    analytic = (y0 * (H - y0)) / (ym * (H - ym))
    assert abs(ratio - analytic) < 1e-9, (ratio, analytic)
    assert ratio < 0.5, ratio


def test_rows_are_not_fed_uniformly():
    """With the lattice frozen in place, strip occupancy must follow ``u(y_j)``.

    The global clock fed every row ``U_mean / u(y_j)`` times faster than the row
    vacates (6 particles in one row cell on the rigid precursor); the per-row
    clock must keep the counts ordered by ``u(y_j)`` and the near-wall row
    strictly emptier than mid-channel.
    """
    case = _case()
    dt = case.dt()
    state = initial_state(case)                  # full lattice, nothing advected
    n_steps = int(3.0 * case.dx / (case.inlet_u_mean() * dt))
    for _ in range(n_steps):
        _apply_open_boundaries(state, case, dt)
        state.time += dt
    strip = state.fluid_pos[:, 0] < case.dx
    rid = torch.round(state.fluid_pos[strip, 1] / case.dx - 0.5).long()
    counts = torch.bincount(rid - rid.min(), minlength=int(rid.max() - rid.min() + 1))
    assert counts.numel() >= 4
    assert float(counts[0]) < float(counts[counts.numel() // 2])
    # a row can never hold more than its own delivered volume / spacing + 1
    ys = _rows(case)
    bound = _row_u(case, ys) * n_steps * dt / case.dx + 1.5
    assert float(counts.max()) <= float(bound.max()) + 1e-9, (
        counts.tolist(), bound.tolist())


def test_total_injected_flux_still_equals_the_mean_inlet_flux():
    """Per-row feeding must not change the total volume the inlet delivers."""
    case = _case()
    dt = case.dt()
    state = initial_state(case)
    n_steps = int(6.0 * case.dx / (case.inlet_u_mean() * dt))
    injected = 0
    n_rows = int(_rows(case).numel())
    for _ in range(n_steps):
        injected += _apply_open_boundaries(state, case, dt)["injected"]
        state.time += dt
    columns = injected / n_rows
    expected = case.inlet_u_mean() * dt * n_steps / case.dx
    assert abs(columns - expected) < 1.5, (columns, expected)


def test_deficit_is_bounded_by_one_spacing():
    case = _case()
    dt = case.dt()
    state = initial_state(case)
    for _ in range(int(0.5 * case.dx / (case.inlet_u_mean() * dt))):
        _apply_open_boundaries(state, case, dt)
        state.time += dt
        assert 0.0 <= state.inflow_deficit < case.dx
    assert state.inflow_deficit_rows is not None
    assert float(state.inflow_deficit_rows.max()) < case.dx
    assert abs(float(state.inflow_deficit)
               - float(state.inflow_deficit_rows.mean())) < 1e-12


def test_ramped_startup_delivers_no_volume_at_t_zero():
    """The clock must track the *ramped* profile, not just ``U_mean``."""
    case = _case()
    case.inlet_ramp_time = 1.0
    dt = case.dt()
    state = initial_state(case)
    for _ in range(5):
        _apply_open_boundaries(state, case, dt)
    assert float(state.inflow_deficit_rows.abs().max()) == 0.0
    # half-way through the ramp the delivered volume is ~half of the steady one
    state.time = 0.5
    for _ in range(int(2.0 * case.dx / (case.inlet_u_mean() * dt))):
        _apply_open_boundaries(state, case, dt)
    ramp = 0.5 * (1.0 - math.cos(math.pi * 0.5))
    assert abs(ramp - 0.5) < 1e-12
    assert 0.0 < float(state.inflow_deficit_rows.max()) < case.dx