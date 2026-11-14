"""Acoustic (Riemann star-state) wall impedance of the FSI2 load transfer.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.  The load transfer
(:func:`tensorsph.fsi2_turek_hron.boundary_pressure_from_fluid`) extrapolates the
fluid pressure onto the moving beam boundary and adds the acoustic term

    p* = p_f + rho_f c_f (u_wall - u_f).n ,      n = outward normal (solid->fluid)

which is the heavy-wall limit (``Z_wall >> Z_fluid``) of the linearised Riemann
problem across the interface.  These tests pin the *sign* of that term against
physics that does not depend on the implementation:

* the piston relation ``dp = rho c u_n`` for a wall closing on a fluid at rest,
* the reaction ``-p* n`` of a rigidly translating beam must be anti-parallel to
  its velocity (dissipation), and of magnitude ``2 rho c u_n trib per station``,
* only the *relative* velocity may enter, and
* the pre-fix convention (``wall_impedance_sign = -1``) must reproduce the
  anti-damping force -- that is the defect these tests guard against
  (``docs/fsi2_impedance_sign/01_report.md``).

Nothing here calls the coupled solver: the invariants are algebraic, so they
fail on a sign flip even when every run happens to stay bounded.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph import fsi2_turek_hron as fsi2
from tensorsph.neighbor_search import cell_linked_pairs


def _beam_setup(dx=0.02):
    case = fsi2.FSI2Case(dx=dx, device="cpu")
    state = fsi2.initial_state(case)
    beam_pos, normals, interp, trib = fsi2.beam_boundary_geometry(
        state.beam_nodes, case)
    wall = torch.cat((state.wall_static, state.cylinder_particles, beam_pos), dim=0)
    wall_start = state.wall_static.shape[0] + state.cylinder_particles.shape[0]
    pairs = cell_linked_pairs(torch.cat((state.fluid_pos, wall), dim=0), 2.0 * case.h())
    return case, state, beam_pos, normals, trib, wall, wall_start, pairs


def _wall_pressure(case, state, wall, wall_start, pairs, normals, wall_vel, vel,
                   p, sign=1.0):
    return fsi2.boundary_pressure_from_fluid(
        state.fluid_pos, p, wall, wall_start, pairs, case.h(),
        normals=normals, vel=vel, wall_vel=wall_vel,
        rho0=case.rho0, sound_speed=case.sound_speed, impedance_sign=sign)


def test_default_sign_is_the_dissipative_one():
    assert fsi2.FSI2Case().wall_impedance_sign == 1.0


def test_piston_relation_for_a_wall_closing_on_a_fluid_at_rest():
    """``dp = rho c u_n`` exactly: the wall is heavy, the fluid is uniform."""
    case, state, beam_pos, normals, trib, wall, wall_start, pairs = _beam_setup()
    n_b = beam_pos.shape[0] // 2
    v = 0.5
    wall_vel = torch.zeros_like(wall)
    wall_vel[wall_start:wall_start + n_b, 1] = +v   # upper row: n = +y, closing
    p = torch.zeros((state.fluid_pos.shape[0],), dtype=state.fluid_pos.dtype)
    vel = torch.zeros_like(state.fluid_pos)

    p_wall, w_sum = _wall_pressure(case, state, wall, wall_start, pairs,
                                   normals, wall_vel, vel, p)
    expected = case.rho0 * case.sound_speed * v
    assert float((w_sum > 0).to(torch.float64).mean()) > 0.99
    assert float(p_wall[:n_b].min()) == pytest.approx(expected, rel=1.0e-10)
    assert float(p_wall[:n_b].max()) == pytest.approx(expected, rel=1.0e-10)
    # the lower face has n = -y, so moving *down* also closes on its fluid
    wall_vel2 = torch.zeros_like(wall)
    wall_vel2[wall_start + n_b:, 1] = -v
    p_wall2, _ = _wall_pressure(case, state, wall, wall_start, pairs,
                                normals, wall_vel2, vel, p)
    assert float(p_wall2[n_b:].min()) == pytest.approx(expected, rel=1.0e-10)
    # ... and moving *up* (away from the fluid below) is expansion: p = -rho c v
    wall_vel3 = torch.zeros_like(wall)
    wall_vel3[wall_start + n_b:, 1] = +v
    p_wall3, _ = _wall_pressure(case, state, wall, wall_start, pairs,
                                normals, wall_vel3, vel, p)
    assert float(p_wall3[n_b:].max()) == pytest.approx(
        -case.rho0 * case.sound_speed * v, rel=1.0e-10)


def test_rigid_transverse_translation_is_damped():
    """A beam translating in +y must feel a net force in -y of ``2 rho c v trib M``.

    This is the property the added-mass regime needs: the impedance term has to
    remove energy, not inject it.
    """
    case, state, beam_pos, normals, trib, wall, wall_start, pairs = _beam_setup()
    n_b = beam_pos.shape[0] // 2
    v = 0.4
    wall_vel = torch.zeros_like(wall)
    wall_vel[wall_start:, 1] = +v          # whole beam (both faces) moves up
    p = torch.zeros((state.fluid_pos.shape[0],), dtype=state.fluid_pos.dtype)
    vel = torch.zeros_like(state.fluid_pos)

    p_wall, _ = _wall_pressure(case, state, wall, wall_start, pairs,
                               normals, wall_vel, vel, p)
    force = fsi2.boundary_force_from_pressure(p_wall, normals, trib)
    fy = float(force[:, 1].sum())
    expected = -2.0 * case.rho0 * case.sound_speed * v * trib * n_b
    assert fy == pytest.approx(expected, rel=5.0e-3)
    assert fy < 0.0                      # opposes the +y motion
    # isolating each face: upper carries -rho c v trib, lower carries the same
    assert float(force[:n_b, 1].sum()) == pytest.approx(
        -case.rho0 * case.sound_speed * v * trib * n_b, rel=5.0e-3)
    assert float(force[n_b:, 1].sum()) == pytest.approx(
        -case.rho0 * case.sound_speed * v * trib * n_b, rel=5.0e-3)


def test_pre_fix_sign_anti_damps():
    """``impedance_sign = -1`` must flip the reaction: energy injection, the defect."""
    case, state, beam_pos, normals, trib, wall, wall_start, pairs = _beam_setup()
    n_b = beam_pos.shape[0] // 2
    v = 0.4
    wall_vel = torch.zeros_like(wall)
    wall_vel[wall_start:, 1] = +v
    p = torch.zeros((state.fluid_pos.shape[0],), dtype=state.fluid_pos.dtype)
    vel = torch.zeros_like(state.fluid_pos)

    p_phys, _ = _wall_pressure(case, state, wall, wall_start, pairs,
                               normals, wall_vel, vel, p, sign=+1.0)
    p_pre, _ = _wall_pressure(case, state, wall, wall_start, pairs,
                              normals, wall_vel, vel, p, sign=-1.0)
    f_phys = fsi2.boundary_force_from_pressure(p_phys, normals, trib)
    f_pre = fsi2.boundary_force_from_pressure(p_pre, normals, trib)
    assert float(f_phys[:, 1].sum()) < 0.0 < float(f_pre[:, 1].sum())
    assert float((f_phys + f_pre).abs().max()) < 1.0e-9


def test_only_the_relative_velocity_enters():
    case, state, beam_pos, normals, trib, wall, wall_start, pairs = _beam_setup()
    v = 0.25
    wall_vel = torch.zeros_like(wall)
    wall_vel[wall_start:, 1] = +v
    vel = torch.zeros_like(state.fluid_pos)
    vel[:, 1] = +v                      # fluid comoving with the wall
    p = torch.zeros((state.fluid_pos.shape[0],), dtype=state.fluid_pos.dtype)

    p_wall, _ = _wall_pressure(case, state, wall, wall_start, pairs,
                               normals, wall_vel, vel, p)
    assert float(p_wall.abs().max()) < 1.0e-10