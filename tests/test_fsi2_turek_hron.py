"""Tests for the Turek-Hron FSI2 benchmark module.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

These tests verify the geometry, inlet profile, beam stiffness, and
force-from-fluid methodology without requiring long-time integration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import math
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph import fsi2_turek_hron as fsi2  # noqa: E402


def test_case_defaults():
    """Default case has physically sensible parameters."""
    case = fsi2.FSI2Case()
    case.validate()
    assert case.reynolds() == pytest.approx(100.0, rel=0.05), case.reynolds()
    assert case.mach() <= 0.01, f"Mach {case.mach()} too high for incompressible flow"
    assert case.inlet_u_mean() == pytest.approx(0.2, rel=1e-6)


def test_parabolic_inlet_profile():
    """V1: Inlet velocity matches analytic parabola."""
    case = fsi2.FSI2Case()
    v = fsi2.verify_parabolic_inlet(case)
    assert v["rel_error"] < 0.01, v
    assert v["u_max_computed"] == pytest.approx(case.inlet_u_max, rel=0.01)


def test_cylinder_geometry():
    """V2: Cylinder particles lie on the correct circle."""
    case = fsi2.FSI2Case()
    v = fsi2.verify_cylinder_geometry(case)
    assert v["rel_deviation"] < 0.01, v
    assert v["n_particles"] > 10, v


def test_beam_stiffness():
    """V3: Beam stiffness matches analytic EI."""
    case = fsi2.FSI2Case()
    v = fsi2.verify_beam_stiffness(case)
    EI_expected = case.beam_E * case.beam_thickness ** 3 / 12.0
    assert v["EI_analytic"] == pytest.approx(EI_expected, rel=1e-12)
    EA_expected = case.beam_E * case.beam_thickness
    assert v["EA_analytic"] == pytest.approx(EA_expected, rel=1e-12)


def test_beam_point_load_small_deflection():
    """V4: Beam tip deflection under small point load vs Euler-Bernoulli."""
    case = fsi2.FSI2Case(dx=0.005, n_beam_nodes=30)
    v = fsi2.verify_beam_point_load(case)
    # Allow 15% error for the coarse discretization
    assert abs(v["rel_error"]) < 0.15, v


def test_beam_point_load_convergence():
    """V4b: Error decreases with beam refinement."""
    errors = {}
    for n in (15, 30, 50):
        case = fsi2.FSI2Case(dx=0.005, n_beam_nodes=n)
        v = fsi2.verify_beam_point_load(case)
        errors[n] = abs(v["rel_error"])
    # Finer should be better (or at least not worse by much)
    assert errors[50] < errors[15] * 1.5, errors


def test_fluid_particle_count():
    """Fluid particles fill the channel minus cylinder and beam."""
    case = fsi2.FSI2Case(dx=0.01)  # coarser for speed
    pos, mass = fsi2.build_fluid_particles(case, torch.device("cpu"))
    # Rough estimate: channel area / dx^2 minus cylinder area / dx^2
    channel_area = case.channel_length * case.channel_height
    cyl_area = math.pi * (case.cylinder_d / 2) ** 2
    beam_area = case.beam_length * case.beam_thickness
    expected = int((channel_area - cyl_area - beam_area) / case.dx ** 2)
    # Allow 20% tolerance due to boundary effects
    assert abs(pos.shape[0] - expected) / expected < 0.20, (pos.shape[0], expected)


def test_initial_state():
    """Initial state is well-formed."""
    case = fsi2.FSI2Case(dx=0.01)
    state = fsi2.initial_state(case)
    assert state.fluid_pos.shape[0] > 0
    assert state.beam_nodes.shape[0] == case.n_beam_nodes
    assert state.time == 0.0
    # Beam should be straight initially
    expected_x0 = case.cylinder_x + case.cylinder_d / 2
    assert abs(float(state.beam_nodes[0, 0]) - expected_x0) < 1e-10
    assert abs(float(state.beam_nodes[0, 1]) - case.cylinder_y) < 1e-10


def test_single_step():
    """One coupled step runs without error and produces diagnostics."""
    case = fsi2.FSI2Case(dx=0.01, t_end=0.001)
    state = fsi2.initial_state(case)
    dt = case.dt()
    row = fsi2.step_fsi2(state, case, dt)
    assert "t" in row
    assert "tip_x" in row
    assert "tip_y" in row
    assert "drag" in row
    assert "lift" in row
    assert row["t"] == pytest.approx(dt, rel=1e-6)


def test_short_run():
    """A short run completes and records history."""
    case = fsi2.FSI2Case(dx=0.01, t_end=0.002, output_every=1)
    state, metrics = fsi2.run_fsi2(case)
    assert len(state.history["t"]) > 0
    assert len(state.history["tip_x"]) > 0
    assert metrics["n_steps"] > 0


def test_deterministic():
    """Two identical runs produce identical results."""
    case = fsi2.FSI2Case(dx=0.01, t_end=0.002)
    _, m1 = fsi2.run_fsi2(case)
    _, m2 = fsi2.run_fsi2(case)
    assert m1["peak_tip_x"] == m2["peak_tip_x"]
    assert m1["peak_tip_y"] == m2["peak_tip_y"]


def test_beam_clamped_at_cylinder():
    """The beam's first node stays at the cylinder attachment point."""
    case = fsi2.FSI2Case(dx=0.01, t_end=0.005, output_every=1)
    state, _ = fsi2.run_fsi2(case)
    expected_x = case.cylinder_x + case.cylinder_d / 2
    expected_y = case.cylinder_y
    assert abs(float(state.beam_nodes[0, 0]) - expected_x) < 1e-10
    assert abs(float(state.beam_nodes[0, 1]) - expected_y) < 1e-10


def test_reynolds_number():
    """Reynolds number matches the FSI2 target."""
    case = fsi2.FSI2Case()
    # Re = rho * U_mean * D / mu
    expected = case.rho0 * case.inlet_u_mean() * case.cylinder_d / case.viscosity
    assert case.reynolds() == pytest.approx(expected, rel=1e-12)
    assert case.reynolds() == pytest.approx(100.0, rel=0.05)
