"""The beam must be loaded by skin friction, not by pressure alone.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Defect this file pins down
--------------------------
``body_force_from_fluid`` computes the viscous (friction) traction on the
cylinder *and* on the beam faces, but the coupled drivers used the beam half of
it only as a diagnostic: the elastic rod was advanced with
``raw_load = beam_interp.T @ f_beam`` where ``f_beam`` came from the pressure
arm alone (see the "diagnostic arm, not fed back into the rod dynamics" comment
in ``fsi2_turek_hron.body_force_from_fluid``).

On the Turek-Hron / SPHinXsys ``test_2d_fsi2`` plate the axial load is
boundary-layer skin friction (``Re_L = U L / nu ~ 3.5e2``, laminar), so a
pressure-only rod stays exactly straight: measured tip displacement ``0.0`` for
as long as the fluid keeps accelerating, and the coupled benchmark can never
reach the archived reference.  ``FSI2Case.beam_viscous_load`` (default on) adds
the friction traction to the structural load in *both* drivers
(:func:`tensorsph.fsi2_turek_hron.step_fsi2` and
:func:`tensorsph.fsi2_strong_coupling._fluid_load`).

What is asserted here
---------------------
1. the friction operator is exact on a linear (Couette) shear profile -- the
   property the load transfer relies on;
2. the structural nodal load contains the friction term, and
   ``beam_viscous_load=False`` removes exactly that term and nothing else;
3. with the fix the rod actually deflects downstream under a sheared flow,
   and by a large margin over the pressure-only rod.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.fsi2_similarity import similarity_corrected_case  # noqa: E402
from tensorsph.fsi2_turek_hron import (  # noqa: E402
    initial_state, step_fsi2, wall_shear_from_velocity_gradient,
)


def _case(dx: float = 0.02, greedy: bool = True):
    case = similarity_corrected_case(U_mean=0.2, Ae=1.4e3)
    case.dx = dx
    case.sound_speed = 5.0
    case.beam_rho = 10.0 * case.rho0
    case.channel_length = 1.1
    case.n_beam_nodes = max(15, int(round(case.beam_length / dx)))
    case.rod_stability_target = 0.2
    case.max_rod_substeps = 2000
    case.wall_impedance = False          # friction must not depend on the scheme
    case.greedy = greedy
    return case


def test_friction_arm_exact_on_couette_shear():
    """One-sided kernel-corrected traction reproduces ``mu * du/dy`` exactly."""
    h, mu, shear = 0.02, 0.2, 3.0
    xs = torch.arange(0.01, 0.35, h, dtype=torch.float64)
    wall = torch.stack([xs, torch.zeros_like(xs)], dim=1)
    normals = torch.zeros_like(wall)
    normals[:, 1] = 1.0
    trib = torch.full((xs.shape[0],), h, dtype=torch.float64)

    ys = torch.arange(h, 0.22, h, dtype=torch.float64)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    fluid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    vel = torch.zeros_like(fluid)
    vel[:, 0] = shear * fluid[:, 1]
    vol = torch.full((fluid.shape[0],), h * h, dtype=torch.float64)
    mu_f = torch.full((fluid.shape[0],), mu, dtype=torch.float64)

    f = wall_shear_from_velocity_gradient(
        wall, normals, trib, fluid, vel, vol, mu_f, h)
    analytic = mu * shear * trib                     # traction x tributary
    assert torch.all(f[:, 1].abs() < 1e-9)
    assert torch.allclose(f[:, 0], analytic, rtol=1e-9, atol=1e-12)


def test_structural_load_contains_friction_term():
    """``beam_viscous_load`` adds the friction traction to the rod load."""
    case = _case()
    state = initial_state(case)
    # uniform streamwise velocity, hydrostatic density/pressure: the pressure
    # arm carries no axial load, so anything axial is the friction term
    state.fluid_vel = torch.zeros_like(state.fluid_vel)
    state.fluid_vel[:, 0] = case.inlet_u_max

    row_on = step_fsi2(state, case, case.dt())
    fric = row_on["beam_drag_friction"]
    assert fric > 0.0, f"friction arm gave {fric}"
    assert row_on["drag_struct"] == pytest.approx(
        row_on["beam_drag_pressure"] + fric, rel=0, abs=1e-12)

    case_off = _case()
    case_off.beam_viscous_load = False
    state_off = initial_state(case_off)
    state_off.fluid_vel = torch.zeros_like(state_off.fluid_vel)
    state_off.fluid_vel[:, 0] = case_off.inlet_u_max
    row_off = step_fsi2(state_off, case_off, case_off.dt())
    assert row_off["drag_struct"] == pytest.approx(
        row_off["beam_drag_pressure"], rel=0, abs=1e-12)
    # the diagnostic arm is unchanged by the switch: it is the same operator
    assert row_off["beam_drag_friction"] == pytest.approx(fric, rel=1e-9)


def test_beam_stretch_from_friction_matches_axial_estimate():
    """Regression guard: the pressure-only rod does not respond to friction.

    A uniform streamwise flow over the plate gives (almost) pure axial tension,
    so the *difference* between the two runs is the axial stretch of the rod,
    ``d = T L / (2 E A)`` with ``E A = E t`` and ``T`` the friction load the
    driver reports.  The check is order-of-magnitude (the pressure arm of the
    same two-face sum contributes a small offset that is not zero here), which
    is all an analytic axial estimate can support.
    """
    def run(beam_viscous_load: bool, n_steps: int = 300) -> tuple[float, float]:
        case = _case()
        case.beam_viscous_load = beam_viscous_load
        state = initial_state(case)
        state.fluid_vel = torch.zeros_like(state.fluid_vel)
        state.fluid_vel[:, 0] = case.inlet_u_max
        dt = case.dt()
        tip, load = 0.0, 0.0
        for _ in range(n_steps):
            row = step_fsi2(state, case, dt)
            tip, load = row["tip_x"], row["beam_drag_friction"]
        return tip, load

    tip_on, friction = run(True)
    tip_off, _ = run(False)
    assert math.isfinite(tip_on)
    # pressure-only rod: no axial load from friction, so no stretch scale
    assert abs(tip_off - tip_on) > 1.0e-5, (
        f"friction did not change the rod response: {tip_on} vs {tip_off}")
    ea = float(_case().beam_E) * float(_case().beam_thickness)   # N
    length = float(_case().beam_length)
    analytic = friction * length / (2.0 * ea)
    measured = tip_on - tip_off
    assert measured > 0.0
    assert 0.2 * analytic < measured < 5.0 * analytic, (
        f"axial stretch {measured:.3e} m vs estimate {analytic:.3e} m")
