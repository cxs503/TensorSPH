"""Tests for Stokes' second problem: reference solution, moving mirror wall, driver.

The reference is not asserted, it is *re-derived*: the sympy branch proves the
closed forms satisfy the governing equation and both boundary conditions, and
the numeric branch does the same with finite differences on a fine grid.  The
wall tests pin the ghost-velocity construction, including the static-wall limit
that every earlier published number depends on.  The driver test is a coarse
smoke run: it proves the machinery executes and produces rows, not that the
physics is right (the accuracy evidence lives in
``docs/verification_stokes_second.md``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorsph.stokes_second import (
    finite_depth,
    finite_depth_amplitude,
    harmonic_fit,
    penetration_depth,
    semi_infinite,
    semi_infinite_amplitude,
    verify_numeric,
    verify_symbolic,
)
from tensorsph.wall_mirror import mirror_plane, plane_walls


# --------------------------------------------------------------------------- #
# Reference solution
# --------------------------------------------------------------------------- #

def test_delta_is_the_stokes_penetration_depth():
    # delta = sqrt(2 nu / omega); the 1/e decay is the defining property
    nu, om = 0.01, 2.0 * math.pi
    delta = penetration_depth(nu, om)
    assert delta == pytest.approx(math.sqrt(2.0 * nu / om), rel=1e-15)
    ratio = semi_infinite_amplitude(delta, 1.0, nu, om)
    assert ratio == pytest.approx(math.exp(-1.0), rel=1e-12)


def test_symbolic_re_derivation_of_both_closed_forms():
    check = verify_symbolic()
    if not check["available"]:
        pytest.skip(f"sympy unavailable: {check.get('reason')}")
    assert check["pde_residual_is_zero"], check["pde_residual_u_t_minus_nu_u_yy"]
    assert check["plate_bc_is_satisfied"], check["plate_bc_residual"]
    assert check["amplitude_at_delta_is_one_over_e"]
    assert check["finite_depth_pde_residual_is_zero"]
    assert check["finite_depth_bc_y0_satisfied"]
    assert check["finite_depth_bc_yH_satisfied"]


def test_numeric_re_derivation_of_both_closed_forms():
    check = verify_numeric()
    assert check["all_checks_pass"], check
    assert check["pde_relative_residual"] < 5.0e-3
    assert check["amplitude_at_delta_error"] < 1e-4
    # a channel of depth 6 delta is within exp(-5) of the half space
    assert check["finite_depth_vs_half_space_max_diff"] < math.exp(-5.0)


def test_finite_depth_solution_approaches_the_half_space_one():
    nu, om, u0 = 0.01, 2.0 * math.pi, 0.01
    delta = penetration_depth(nu, om)
    y = np.linspace(0.0, 3.0 * delta, 61)
    err = {}
    for h_over_delta in (3.0, 6.0):
        fd = finite_depth_amplitude(y, h_over_delta * delta, u0, nu, om)
        half = semi_infinite_amplitude(y, u0, nu, om)
        err[h_over_delta] = float(np.max(np.abs(fd - half)) / u0)
    assert err[6.0] < err[3.0] * 0.05
    # the largest difference sits at the far-wall row, where the half-space
    # value itself is 1e-3 of U0
    assert err[6.0] < 5e-4


def test_plate_boundary_condition_and_wave_speed():
    nu, om, u0 = 0.01, 2.0 * math.pi, 0.01
    t = np.linspace(0.0, 2.0 * math.pi / om, 200, endpoint=False)
    plate = semi_infinite(np.zeros_like(t), t, u0, nu, om)
    assert np.allclose(plate, u0 * np.cos(om * t), atol=1e-15)
    # the shear wave travels at sqrt(2 nu omega) = omega delta: track one phase
    delta = penetration_depth(nu, om)
    # at t = pi/(2 omega) the crest has travelled to y = delta * omega * t
    y = np.linspace(0.0, 2.0 * math.pi * delta, 4000)
    t_peak = 0.5 * math.pi / om
    field = semi_infinite(y, t_peak, u0, nu, om)
    peak = y[int(np.argmax(field))]
    assert peak == pytest.approx(delta * om * t_peak, rel=0.01)


# --------------------------------------------------------------------------- #
# Harmonic estimator
# --------------------------------------------------------------------------- #

def test_harmonic_fit_recovers_amplitude_phase_and_mean():
    om = 2.0 * math.pi
    t = np.arange(0.0, 4.0, 1.0 / 400.0)
    signal = 0.7 * np.cos(om * t - 0.9) + 0.05
    fit = harmonic_fit(t, signal, om)
    assert fit["amplitude"] == pytest.approx(0.7, rel=1e-9)
    assert fit["phase"] == pytest.approx(-0.9, abs=1e-9)
    assert fit["mean"] == pytest.approx(0.05, abs=1e-9)
    assert fit["residual"] < 1e-9


def test_harmonic_fit_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        harmonic_fit([0.0, 1.0, 2.0], [1.0, 2.0], 1.0)


# --------------------------------------------------------------------------- #
# Mirror (ghost) wall: moving planes
# --------------------------------------------------------------------------- #

def _case(top_speed: float = 0.0):
    pos = torch.tensor([[0.25, 0.3], [0.75, 0.3], [0.25, 0.9], [0.75, 0.9]])
    vel = torch.tensor([[top_speed, 0.1], [top_speed, -0.1],
                        [top_speed, 0.2], [top_speed, -0.2]])
    mass = torch.full((4,), 1.0)
    return pos, vel, mass


def test_static_mirror_is_unchanged_by_the_moving_plane_support():
    pos, vel, mass = _case(top_speed=0.3)
    gp, gv, _ = mirror_plane(pos, vel, mass, 0.0, 1.0, side=+1)
    assert torch.allclose(gv, -vel)              # the historical static form
    gp2, gv2, _ = mirror_plane(pos, vel, mass, 0.0, 1.0, side=+1,
                               plane_vel=(0.0, 0.0))
    assert torch.equal(gv, gv2)


def test_moving_no_slip_ghost_pair_has_the_wall_velocity_as_its_mean():
    pos, vel, mass = _case(top_speed=0.3)
    w = torch.tensor([0.4, 0.0])
    gp, gv, _ = mirror_plane(pos, vel, mass, 0.0, 1.0, side=+1, plane_vel=w)
    assert torch.allclose(0.5 * (gv + vel), w.expand_as(gv))
    # positions are still mirrored about the plane
    assert torch.allclose(gp[:, 1], -pos[:, 1])


def test_moving_free_slip_ghost_has_no_tangential_jump():
    pos, vel, mass = _case(top_speed=0.3)
    gp, gv, _ = mirror_plane(pos, vel, mass, 0.0, 1.0, side=+1,
                             slip="free_slip", plane_vel=(0.4, 0.0))
    assert torch.allclose(gv[:, 0], vel[:, 0])          # no tangential shear
    assert torch.allclose(gv[:, 1], -vel[:, 1])         # normal component mirrored


def test_plane_walls_accepts_a_per_plane_velocity():
    pos, vel, mass = _case(top_speed=0.0)
    w = (0.5, 0.0)
    gp, gv, gm = plane_walls(pos, vel, mass, [(0.0, +1, w), (1.2, -1)], 1.0)
    assert gp.shape[0] == 8 and gm.shape[0] == 8
    bottom = gv[:4]
    assert torch.allclose(0.5 * (bottom + vel), torch.tensor(w).expand_as(bottom))
    top = gv[4:]
    assert torch.allclose(top, -vel)        # the far wall is at rest


# --------------------------------------------------------------------------- #
# Driver smoke test (machinery, not accuracy)
# --------------------------------------------------------------------------- #

def test_driver_produces_a_decaying_amplitude_profile():
    from benchmarks.stokes_second import run as bench

    device = "cpu"
    res = bench.run_one(6, device, periods=1.0, measure_periods=0.5,
                        sound_speed=4.0, label="test")
    rows = res["rows"]
    assert len(rows) >= 8
    amps = [r["amplitude_over_U0"] for r in rows]
    assert all(np.isfinite(a) for a in amps)
    # the wave must decay away from the plate and the plate row must move
    assert amps[0] > 0.5
    assert amps[-1] < 0.1 * amps[0]
    assert res["n_fluid"] > 0 and res["n_ghost_max"] > 0
    assert res["metrics"]["n_rows_stokes_layer"] >= 5
    # wall clock evidence, and the wall model is a mirror wall, not the blend
    assert res["wall_friction"] == 1.0
    assert "amplitude_l2_half_space_pct" in res["metrics"]