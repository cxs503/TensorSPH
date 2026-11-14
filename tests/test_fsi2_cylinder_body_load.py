"""Gates for the force-from-fluid path on the rigid part of the FSI2 body.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.  Before this round the
coupled driver accumulated only the *beam* load, so the Turek-Hron drag/lift
coefficients -- defined on the whole body -- could not be formed.  These tests
pin the instrument to exact answers instead of merely executing it:

* bookkeeping (analytic field on the stations) is exact,
* the OSS kernel-average arm carries a radial over-load that decays as O(h/r),
* the weighted plane fit is exact on a linear field,
* the viscous arm reproduces ``mu * gamma`` on a flat wall (sign included),
* the coupled step reports body = cylinder + beam.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_fsi2"))

from tensorsph.fsi2_turek_hron import (  # noqa: E402
    FSI2Case, boundary_force_from_pressure, boundary_pressure_from_fluid,
    boundary_pressure_from_fluid_linear_exact, cylinder_surface_geometry,
    initial_state, step_fsi2,
)
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402

GRAD_P = torch.tensor([0.0, -1.0], dtype=torch.float64)
P0 = 1.0


def _field(x):
    return P0 + x @ GRAD_P


def _operator_arms(dx: float):
    case = FSI2Case()
    case.dx = dx
    pos, nrm, trib = cylinder_surface_geometry(case)
    n = pos.shape[0]
    trib_t = torch.full((n,), trib, dtype=torch.float64)
    area = math.pi * (0.5 * case.cylinder_d) ** 2
    exact = -(GRAD_P * area)

    r0, h = 0.5 * case.cylinder_d, case.h()
    ext = r0 + 3.0 * h
    xs = torch.arange(-ext, ext + dx, dx, dtype=torch.float64)
    gx, gy = torch.meshgrid(xs, xs, indexing="xy")
    d = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    keep = d.norm(dim=1) > (r0 + 0.5 * dx)
    centre = torch.tensor([case.cylinder_x, case.cylinder_y], dtype=torch.float64)
    fpos = d[keep] + centre
    pf = _field(fpos)
    pairs = cell_linked_pairs(torch.cat((fpos, pos), dim=0), 2.0 * h)

    p_a = _field(pos)
    p_avg, w_avg = boundary_pressure_from_fluid(
        fpos, pf, pos, 0, pairs, h, wall_stop_idx=n)
    p_fit, _ = boundary_pressure_from_fluid_linear_exact(
        fpos, pf, pos, 0, n, pairs, h)
    f = {k: boundary_force_from_pressure(p, nrm, trib_t).sum(dim=0)
         for k, p in (("analytic", p_a), ("avg", p_avg), ("fit", p_fit))}
    return exact, f, float((w_avg > 1e-30).to(torch.float64).mean())


def test_cylinder_station_identity_is_exact():
    """sum(-p n trib) with the analytic field = -grad p A, to machine precision."""
    exact, f, coverage = _operator_arms(0.02)
    assert coverage == 1.0
    assert abs(float((f["analytic"][1] - exact[1]) / exact[1])) < 1e-12
    # transverse component cancels by symmetry of the regular station set
    assert abs(float(f["analytic"][0] / exact[1])) < 1e-12


def test_kernel_average_curved_bias_is_o_h_over_r():
    """The OSS arm over-loads a convex body by O(h/r): large, slowly decaying."""
    exact, coarse, _ = _operator_arms(0.04)
    _, fine, _ = _operator_arms(0.01)
    e_coarse = float((coarse["avg"][1] - exact[1]) / exact[1])
    e_fine = float((fine["avg"][1] - exact[1]) / exact[1])
    assert e_coarse > 0.4 and e_fine > 0.1     # +60 % / +19 % measured
    assert e_fine < 0.6 * e_coarse             # halving dx halves the bias
    # the drag-direction component is blind to the defect (<1 %), which is why
    # the flat-wall audits did not catch it
    assert abs(float(coarse["avg"][0] / exact[1])) < 0.01


def test_plane_fit_is_exact_on_linear_field():
    for dx in (0.04, 0.01):
        exact, f, _ = _operator_arms(dx)
        assert abs(float((f["fit"][1] - exact[1]) / exact[1])) < 1e-4
        assert abs(float(f["fit"][0] / exact[1])) < 1e-4


def test_viscous_arm_reproduces_mu_gamma_on_flat_wall():
    from cylinder_load_audit import run_gate_d
    gate = run_gate_d()
    assert abs(gate["rel_err_mid_station"]) < 1e-6
    assert gate["measured_fx_per_station"] > 0.0  # fluid drags the wall downstream


def _tiny_case(dx: float = 0.02) -> FSI2Case:
    case = FSI2Case()
    case.dx = dx
    case.sound_speed = 5.0
    case.inlet_u_max = 1.0e-3       # numerically still; Re must stay > 0
    case.n_beam_nodes = max(15, int(case.beam_length / dx))
    case.max_rod_substeps = 200
    return case


def test_coupled_step_reports_body_and_cylinder_loads():
    case = _tiny_case()
    state = initial_state(case)
    dt = case.dt()
    for _ in range(3):
        row = step_fsi2(state, case, dt)
    for key in ("cyl_drag", "cyl_lift", "body_drag", "body_lift",
                "cyl_drag_fit", "body_drag_fit", "drag_viscous", "cyl_coverage"):
        assert key in row, key
        assert math.isfinite(float(row[key])), key
    assert row["body_drag"] == pytest.approx(row["cyl_drag"] + row["drag"], abs=1e-12)
    assert row["cyl_coverage"] == 1.0


def test_still_fluid_produces_no_spurious_body_load():
    """u_inlet = 0: the fluid load on the whole body must stay at round-off."""
    case = _tiny_case()
    state = initial_state(case)
    dt = case.dt()
    for _ in range(5):
        row = step_fsi2(state, case, dt)
    # gate: the residual load must stay far below the acoustic pressure scale
    # rho c^2 D, i.e. no spurious load is manufactured at rest
    acoustic = case.rho0 * case.sound_speed ** 2 * case.cylinder_d
    assert abs(row["body_drag"]) < 1.0e-3 * acoustic
    assert abs(row["body_drag_fit"]) < 1.0e-3 * acoustic
    assert row["body_drag"] == pytest.approx(row["body_drag_fit"], abs=1e-3 * acoustic)
