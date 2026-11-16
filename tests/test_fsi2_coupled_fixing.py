"""Regression tests for the coupled FSI2 defects found in mission 20552.

The anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`` had no coupled run
until now; wiring one up exposed three defects in
:mod:`tensorsph.fsi2_turek_hron` that made every coupled run explode on the
*first* step (fluid speed 3777 m/s, beam force 2.1e8 N/m for a load whose
physical scale is ~6 N/m).  Each defect gets an invariant here, plus the
added-mass limit of the loosely coupled scheme, which is *pinned*, not fixed.

See ``docs/xval_fsi2/COUPLED_RUN_DEFECTS.md``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph import fsi2_turek_hron as fsi2
from tensorsph.neighbor_search import cell_linked_pairs


def _case(dx=0.02, **kw):
    case = fsi2.FSI2Case(dx=dx, device="cpu", **kw)
    return case


# --------------------------------------------------------------------------- #
# D1: the node -> boundary interpolation collapsed onto the first segment
# --------------------------------------------------------------------------- #
def test_beam_boundary_spans_the_whole_beam():
    """17 boundary stations must span 0.35 m, not the first 12 mm."""
    case = _case(dx=0.02)
    nodes = fsi2.beam_initial_nodes(case, torch.device("cpu"))
    pos, normals, interp, trib = fsi2.beam_boundary_geometry(nodes, case)

    x = pos[:, 0]
    assert float(x.max() - x.min()) == pytest.approx(case.beam_length, rel=1e-6)
    # station spacing is the arc length per station, ~ one particle spacing
    assert trib == pytest.approx(case.beam_length
                                / (math.floor(case.beam_length / case.dx) - 1),
                                rel=1e-12)
    assert case.dx / 3.0 < trib < 2.0 * case.dx

    # interpolation rows are convex combinations of adjacent nodes
    assert torch.allclose(interp.sum(dim=1), torch.ones(interp.shape[0],
                                                        dtype=interp.dtype))
    assert float(interp.min()) >= 0.0
    assert float(interp.max()) <= 1.0


def test_interpolation_matrix_spans_node_index_space():
    mat = fsi2.interpolation_matrix_beam(30, 17, torch.device("cpu"))
    rows = torch.arange(17, dtype=mat.dtype)
    station = mat @ torch.arange(30, dtype=mat.dtype)
    assert float(station[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(station[-1]) == pytest.approx(29.0, abs=1e-9)
    assert torch.all(station[1:] > station[:-1])
    assert rows.shape == station.shape


# --------------------------------------------------------------------------- #
# D2: two-sided boundary layer (one row per beam face)
# --------------------------------------------------------------------------- #
def test_beam_boundary_has_two_faces_with_opposite_normals():
    case = _case(dx=0.01)
    nodes = fsi2.beam_initial_nodes(case, torch.device("cpu"))
    pos, normals, interp, trib = fsi2.beam_boundary_geometry(nodes, case)
    m = pos.shape[0] // 2
    assert torch.allclose(normals[:m], -normals[m:])
    assert torch.allclose(torch.linalg.vector_norm(normals, dim=1),
                          torch.ones(pos.shape[0], dtype=pos.dtype))
    # straight beam along +x: the faces sit at y = y0 +- t/2
    assert float(pos[:m, 1].mean() - case.cylinder_y) == pytest.approx(
        0.5 * case.beam_thickness, rel=1e-9)
    assert float(pos[m:, 1].mean() - case.cylinder_y) == pytest.approx(
        -0.5 * case.beam_thickness, rel=1e-9)


def test_beam_normals_follow_the_deformed_centreline():
    """After a bend the face normals rotate with the rod tangent."""
    case = _case(dx=0.02)
    nodes = fsi2.beam_initial_nodes(case, torch.device("cpu"))
    ramp = torch.linspace(0.0, 1.0, nodes.shape[0], dtype=nodes.dtype)
    nodes[:, 1] += 0.02 * ramp  # bend the beam towards +y
    pos, normals, _, _ = fsi2.beam_boundary_geometry(nodes, case)
    m = pos.shape[0] // 2
    upper, lower = normals[:m], normals[m:]
    assert torch.allclose(upper, -lower)
    assert float(upper[:, 1].min()) > 0.0        # upper face points up
    # a bent rod has a tilted tangent downstream, so |n_x| grows along the beam
    assert float(upper[-1, 0].abs()) > float(upper[0, 0].abs())
    # the two rows stay one thickness apart, measured along the normal
    gap = (pos[:m] - pos[m:]) * upper
    assert torch.allclose(gap.sum(dim=1),
                          torch.full((m,), case.beam_thickness, dtype=pos.dtype),
                          rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------- #
# D3: no fluid particle on a beam surface (zero-distance neighbour)
# --------------------------------------------------------------------------- #
def test_no_fluid_particle_sits_on_a_beam_surface():
    for dx in (0.01, 0.02):
        case = _case(dx=dx)
        pos, mass = fsi2.build_fluid_particles(case, torch.device("cpu"))
        nodes = fsi2.beam_initial_nodes(case, torch.device("cpu"))
        beam_pos, _, _, _ = fsi2.beam_boundary_geometry(nodes, case)
        d = torch.cdist(pos, beam_pos).min()
        assert float(d) > 0.45 * dx, (dx, float(d))


def test_uniform_pressure_gives_no_net_force():
    """A symmetric (constant) pressure field must produce a zero net load.

    Before the fixes this number was 2.1e8 N/m, because the collapsed boundary
    row produced ~0.04 dx separations.
    """
    case = _case(dx=0.02)
    state = fsi2.initial_state(case)
    beam_pos, normals, interp, trib = fsi2.beam_boundary_geometry(
        state.beam_nodes, case)
    wall = torch.cat((state.wall_static, state.cylinder_particles, beam_pos), dim=0)
    wall_start = state.wall_static.shape[0] + state.cylinder_particles.shape[0]
    pairs = cell_linked_pairs(torch.cat((state.fluid_pos, wall), dim=0), 2.0 * case.h())
    p = torch.full((state.fluid_pos.shape[0],), 1234.5, dtype=state.fluid_pos.dtype)
    p_wall, w_sum = fsi2.boundary_pressure_from_fluid(
        state.fluid_pos, p, wall, wall_start, pairs, case.h())
    assert float((w_sum > 0).to(torch.float64).mean()) > 0.99  # full coverage
    force = fsi2.boundary_force_from_pressure(p_wall, normals, trib)
    assert float(force.sum(dim=0).abs().max()) < 1.0e-9
    # ... while each face individually carries the full load p*A/2
    m = force.shape[0] // 2
    assert float(force[:m, 1].sum()) == pytest.approx(
        -1234.5 * trib * m, rel=0.05)


def test_acoustic_impedance_sign():
    """A face closing on the fluid must see a *higher* pressure (Riemann star state).

    Corrected in the impedance-sign round: the earlier version of this test
    moved the *upper* face *downwards* (away from its fluid, i.e. expansion) and
    called it compression, thereby pinning the anti-damping sign.  The upper
    row's normal is ``+y``, so ``+y`` motion is the closing one.  See
    ``tests/test_fsi2_impedance_sign.py`` for the quantitative invariants and
    ``docs/fsi2_impedance_sign/01_report.md`` for the coupled measurement.
    """
    case = _case(dx=0.02)
    state = fsi2.initial_state(case)
    beam_pos, normals, interp, trib = fsi2.beam_boundary_geometry(
        state.beam_nodes, case)
    wall = torch.cat((state.wall_static, state.cylinder_particles, beam_pos), dim=0)
    wall_start = state.wall_static.shape[0] + state.cylinder_particles.shape[0]
    pairs = cell_linked_pairs(torch.cat((state.fluid_pos, wall), dim=0), 2.0 * case.h())
    p = torch.zeros((state.fluid_pos.shape[0],), dtype=state.fluid_pos.dtype)
    vel = torch.zeros_like(state.fluid_pos)
    n_b = beam_pos.shape[0] // 2

    still, _ = fsi2.boundary_pressure_from_fluid(
        state.fluid_pos, p, wall, wall_start, pairs, case.h(),
        normals=normals, vel=vel,
        wall_vel=torch.zeros_like(wall), rho0=case.rho0,
        sound_speed=case.sound_speed)

    # upper face moving up (+y, normal +y): it closes on the fluid above
    wall_vel = torch.zeros_like(wall)
    wall_vel[wall_start:wall_start + n_b, 1] = +0.5
    closing, _ = fsi2.boundary_pressure_from_fluid(
        state.fluid_pos, p, wall, wall_start, pairs, case.h(),
        normals=normals, vel=vel, wall_vel=wall_vel,
        rho0=case.rho0, sound_speed=case.sound_speed)
    assert float(still.abs().max()) < 1.0e-12
    assert float(closing[:n_b].min()) == pytest.approx(
        case.rho0 * case.sound_speed * 0.5, rel=1.0e-10)

    # ... and moving down (-y) is expansion: the pressure drops by the same amount
    wall_vel = torch.zeros_like(wall)
    wall_vel[wall_start:wall_start + n_b, 1] = -0.5
    expanding, _ = fsi2.boundary_pressure_from_fluid(
        state.fluid_pos, p, wall, wall_start, pairs, case.h(),
        normals=normals, vel=vel, wall_vel=wall_vel,
        rho0=case.rho0, sound_speed=case.sound_speed)
    assert float(expanding[:n_b].max()) == pytest.approx(
        -case.rho0 * case.sound_speed * 0.5, rel=1.0e-10)


# --------------------------------------------------------------------------- #
# Open boundaries: the benchmark is a through-flow channel, not a tank
# --------------------------------------------------------------------------- #
#
# D4 was first closed with a *geometric* inlet clock -- "re-inject once the
# inlet has drifted one lattice spacing" -- and the round below pinned exactly
# that.  The clock was then replaced by a per-streamline *flux* clock, because a
# geometric trigger is pinned by the slowest particle in the domain (the inlet
# parabola vanishes at the wall, so a near-wall particle creeps at ~0.1 U_max and
# keeps ``min x`` below the trigger while the outlet keeps deleting particles:
# ``injected = 0`` over 1200 steps, ``docs/xval_fsi2/INFLOW_CLOCK_STALL.md``).
# The tests below therefore pin the *current* contract, not the withdrawn one:
# a row injects once it has been delivered one lattice spacing of volume,
# ``sum(u(y_j) dt) >= dx``, and nothing else.

def _shifted_state(case, shift_lattice=1.5, stall_particle=True):
    """Initial state with a gap at the inlet and (optionally) a stalled particle.

    Shifting everything downstream by ``1.5 dx`` opens the inlet: ``min x`` is
    then ``2 dx``, i.e. past the old ``1.5 dx`` geometric trigger, so a geometric
    clock would inject on the first call and a flux clock must not.  The extra
    particle parked at ``x = dx/2`` with zero velocity is the *stall* case: it
    holds ``min x`` at half a lattice spacing forever, which killed the
    geometric clock outright.
    """
    state = fsi2.initial_state(case)
    n0 = state.fluid_pos.shape[0]
    shifted = state.fluid_pos.clone()
    shifted[:, 0] += shift_lattice * case.dx
    keep = shifted[:, 0] < case.channel_length - 0.25 * case.dx
    state.fluid_pos = shifted[keep]
    state.fluid_vel = state.fluid_vel[keep]
    state.fluid_mass = state.fluid_mass[keep]
    n_removed = int((~keep).sum())
    if stall_particle:
        park = torch.tensor([[0.5 * case.dx, 0.5 * case.dx]], dtype=state.fluid_pos.dtype)
        state.fluid_pos = torch.cat((state.fluid_pos, park), dim=0)
        state.fluid_vel = torch.cat(
            (state.fluid_vel, torch.zeros_like(park)), dim=0)
        state.fluid_mass = torch.cat(
            (state.fluid_mass, torch.full((1,), float(state.fluid_mass[0]),
                                          dtype=state.fluid_mass.dtype)), dim=0)
    return state, n0, n_removed


def _row_speeds(case):
    """Local inlet speed of every injection row, exactly as the driver lays them."""
    dx = case.dx
    ys = torch.arange(0.5 * dx, case.channel_height, dx, dtype=torch.float64)
    H = case.channel_height
    return ys, case.inlet_u_max * 4.0 * ys * (H - ys) / (H * H)


def test_open_boundaries_remove_at_the_outlet_but_wait_for_the_flux_clock():
    """One fluid step removes the outlet row and injects nothing yet.

    The flux clock delivers ``u(y_j) dt`` per streamline; one step delivers
    ``3e-4 dx`` at the fastest row, so no row has earned a particle.  A
    geometric clock would have fired here (``min x = 0.5 dx`` because of the
    parked particle, or ``2 dx`` without it).
    """
    case = _case(dx=0.02, inlet_ramp_time=0.0)
    for stall in (True, False):
        state, n0, n_removed = _shifted_state(case, stall_particle=stall)
        assert n_removed > 0                     # the shift really did cross the outlet
        counters = fsi2._apply_open_boundaries(state, case, dt=case.dt())
        assert counters["removed"] == 0          # the shift removed them already
        assert counters["injected"] == 0, counters
        expected = n0 - n_removed + (1 if stall else 0)
        assert state.fluid_pos.shape[0] == expected


def test_inlet_injection_delivers_the_local_volume_per_streamline():
    """A row injects once its own flux has filled one spacing -- not on position.

    Driven with ``dt = 3 dx / U_max`` so the accumulated volume is three
    spacings at the channel centre and less than one near the wall: the deep
    rows inject (up to three particles), the near-wall rows stay silent.  That
    is the per-streamline behaviour the flux clock exists for; a global clock
    fed by ``U_mean`` would have over-fed the wall rows by ``U_mean/u(y) ~ 7``.
    """
    case = _case(dx=0.02, inlet_ramp_time=0.0)
    state, n0, n_removed = _shifted_state(case, stall_particle=True)
    n_before = state.fluid_pos.shape[0]
    ys, u_row = _row_speeds(case)
    dt = 3.0 * case.dx / case.inlet_u_max

    counters = fsi2._apply_open_boundaries(state, case, dt=dt)
    injected = counters["injected"]
    assert injected > 0

    # the parked particle (min x = dx/2 < 1.5 dx) does not stall the clock
    assert float(state.fluid_pos[:, 0].min()) <= 0.5 * case.dx
    # particle bookkeeping is exact: nothing else is created or destroyed
    assert state.fluid_pos.shape[0] == n_before + injected

    # every injected particle sits on the inlet column, none at the wall rows
    fresh_pos = state.fluid_pos[n_before:]
    fresh_vel = state.fluid_vel[n_before:]
    assert torch.allclose(fresh_pos[:, 0],
                          torch.full((injected,), 0.5 * case.dx,
                                     dtype=fresh_pos.dtype))
    fired = u_row * dt >= case.dx
    assert bool(fired.any()) and not bool(fired.all())     # near-wall rows silent
    fired_ys = ys[fired]
    assert float(fresh_pos[:, 1].min()) >= float(fired_ys.min()) - 1e-12
    assert float(fresh_pos[:, 1].max()) <= float(fired_ys.max()) + 1e-12

    # the injected velocity is the *analytic* parabola at its own row, not a slug
    H = case.channel_height
    u_exact = case.inlet_u_max * 4.0 * fresh_pos[:, 1] * (H - fresh_pos[:, 1]) / (H * H)
    assert torch.allclose(fresh_vel[:, 0].to(torch.float64), u_exact, rtol=1e-10)
    assert float(fresh_vel[:, 0].max() / fresh_vel[:, 0].min()) > 2.0

    # delivered volume: one particle per spacing per streamline, to rounding
    expected = float((u_row * dt / case.dx).sum())
    assert abs(injected - expected) <= int(fired.sum()) + 1, (injected, expected)
    # the clock's residual never exceeds one spacing on any streamline
    assert float(state.inflow_deficit_rows.max()) < case.dx
    assert state.inflow_deficit == pytest.approx(
        float(state.inflow_deficit_rows.mean()), rel=1e-12)


def test_open_boundaries_are_on_by_default():
    assert _case().open_boundaries is True


# --------------------------------------------------------------------------- #
# Added-mass limit of the loosely coupled scheme (pinned, not fixed)
# --------------------------------------------------------------------------- #
def _beam_speed_after(case, n_steps):
    state = fsi2.initial_state(case)
    dt = case.dt()
    for _ in range(n_steps):
        row = fsi2.step_fsi2(state, case, dt)
        if not math.isfinite(row["max_speed"]):
            return float("nan")
    return float(state.beam_vel.norm(dim=1).max())


@pytest.mark.parametrize("ratio,diverges", [(1.0, True), (10.0, False)])
def test_added_mass_limit_of_loose_coupling(ratio, diverges):
    """rho_s/rho_f = 1 and 10 both stay bounded -- verdict revised in m20761.

    The earlier claim ("rho_s/rho_f = 1 blows up, 10 stays bounded") is
    **falsified**: the blow-up was the rod sub-cycling criterion, which sized
    the explicit rod sub-steps on the *axial* frequency only and left the
    shortest bending mode unresolvable at rho_s/rho_f = 1 (free-beam energy
    +1e249 over 200 zero-load steps).  With the bending-aware criterion
    (``FSI2Case.beam_max_frequency`` / ``rod_substeps_effective``, m20761) the
    same loose driver at the same dt gives max|v_beam| = 5.4e-3 m/s after 60
    steps at ratio 1 instead of a divergence.  What still diverges is the
    acoustic wall-impedance term, which is off here.
    See docs/xval_fsi2/STRONG_COUPLING_AND_ROD_STABILITY.md.

    Run at the similarity-corrected stiffness (Ae = 5600), which is the point
    the benchmark comparison has to be made at; the shipped FSI2Case default is
    18x stiffer.
    """
    from tensorsph.fsi2_similarity import similarity_corrected_case
    case = similarity_corrected_case(U_mean=0.2)
    case.dx = 0.02
    case.sound_speed = 5.0
    case.beam_rho = 1000.0 * ratio
    case.wall_impedance = False
    v = _beam_speed_after(case, 60)
    # both density ratios are bounded; the pre-m20761 run left this branch
    # asserting a divergence at ratio 1
    assert math.isfinite(v) and v < 0.3, (ratio, v)
