"""Tests for the Navier-slip Poiseuille *reference* and *metric* pipeline.

Two things are asserted:

1. the closed forms used as the reference really solve the stated problem --
   ``nu u'' + g = 0`` and ``u = lam du/dn`` at both walls, checked numerically
   on a fine grid, plus the integral identity for the flux;
2. the metric pipeline in ``benchmarks/poiseuille_slip/run.py`` grades a profile
   that *is* the analytic solution as zero error and rejects a profile that is
   off by a known amount -- otherwise a broken metric could pass for physics.

A short end-to-end channel run (``H/dx = 8``, ``t_end = 0.4 s``) then checks the
boundary condition is actually *active* in the solver: the measured slip
velocity must be a large fraction of the analytic slip and must vanish when the
slip length is zero.
"""

from __future__ import annotations

import math

import pytest
import torch

from benchmarks.poiseuille_slip.run import (
    analytic_slip,
    evaluate,
    flux_ref,
    gates,
    slip_ref,
    u_max_ref,
)

H = 0.1
NU = 0.01
G = 9.81
RHO = 1000.0


@pytest.mark.parametrize("lam_over_h", [0.0, 0.125, 0.25, 0.5])
def test_analytic_solution_satisfies_the_ode_and_both_slip_bcs(lam_over_h):
    lam = lam_over_h * H
    n = 20001
    ys = torch.linspace(0.0, H, n, dtype=torch.float64)
    u = torch.tensor([analytic_slip(float(y), H, G, NU, lam) for y in ys],
                     dtype=torch.float64)
    dy = float(ys[1] - ys[0])
    # ODE residual on the interior
    residual = NU * (u[2:] - 2 * u[1:-1] + u[:-2]) / dy ** 2 + G
    assert float(residual.abs().max()) < 1e-6 * G
    # Navier condition at both walls, one-sided difference
    du0 = (u[1] - u[0]) / dy
    duH = (u[-1] - u[-2]) / dy
    assert u[0].item() == pytest.approx(lam * du0.item(), rel=0, abs=2e-4)
    assert u[-1].item() == pytest.approx(-lam * duH.item(), rel=0, abs=2e-4)


@pytest.mark.parametrize("lam_over_h", [0.0, 0.125, 0.25, 0.5])
def test_reference_integrals_match_the_profile(lam_over_h):
    lam = lam_over_h * H
    n = 20001
    ys = torch.linspace(0.0, H, n, dtype=torch.float64)
    u = torch.tensor([analytic_slip(float(y), H, G, NU, lam) for y in ys],
                     dtype=torch.float64)
    assert float(u.max()) == pytest.approx(
        u_max_ref(H, G, NU, lam), rel=1e-9)
    # the maximum sits at the centre for every slip length
    y_max = float(ys[int(u.argmax())])
    assert y_max == pytest.approx(H / 2, abs=2 * H / (n - 1))
    assert float(u[0]) == pytest.approx(slip_ref(H, G, NU, lam), rel=1e-12)
    q = float(torch.trapezoid(u, ys))
    assert q == pytest.approx(flux_ref(H, G, NU, lam), rel=1e-7)
    # wall shear is independent of the slip length (momentum budget)
    du0 = float((u[1] - u[0]) / (ys[1] - ys[0]))
    assert RHO * NU * abs(du0) == pytest.approx(RHO * G * H / 2, rel=1e-3)


# An odd number of rows puts one row exactly at y = H/2, so the row-based
    # centre-line maximum is the continuum maximum; with an even row count the
    # sampling bias (g/2nu)(dy)^2/u_max (~0.7 % at H/dx = 12) would mask a
    # metric bug. The bias itself is documented in the verification note.
def _synthetic_case(lam_over_h: float, resolution: int = 15, scale: float = 1.0):
    """A case dict whose profile lies exactly on the closed form (times scale)."""
    lam = lam_over_h * H
    dx = H / resolution
    rows = []
    for k in range(resolution):
        y = dx * 0.5 + k * dx
        u = scale * analytic_slip(y, H, G, NU, lam)
        rows.append({"row": k, "y": y, "y_over_H": y / H, "u_mean": u,
                     "u_std_x": 0.0, "count": 10})
    return {
        "profile": rows,
        "rows": list(range(resolution)),
        "field": {"x": [0.0], "y": [0.0], "u": [0.0]},
        "trace": [{"t": 0.0, "u_max": rows[-1]["u_mean"], "u_mean": 0.0,
                   "max_mach": 0.0, "wall_clock_s": 0.0}],
        "dt": 1e-4,
        "steps": 1,
        "n_fluid": 10 * resolution,
        "n_ghost": 4 * resolution,
        "n_averaged_samples": 10,
        "steady_drift": 0.0,
        "y_min": dx / 2,
        "y_max": H - dx / 2,
        "n_outside_channel": 0,
        "max_particle_speed": 1.0,
        "y_at_max_particle_speed": H / 2,
        "wall_clock_s": 0.0,
    }


@pytest.mark.parametrize("lam_over_h", [0.0, 0.125, 0.25, 0.5])
def test_metric_pipeline_scores_the_exact_solution_as_zero_error(lam_over_h):
    res = evaluate(_synthetic_case(lam_over_h), H, G, NU, RHO, 15,
                   lam_over_h * H)
    assert abs(res["u_max_error_pct"]) < 1e-6
    assert res["profile_rms_interior_pct"] < 1e-6
    assert abs(res["wall_shear_bottom_error_pct"]) < 1e-4
    assert abs(res["flux_enhancement_error_pct"]) < 1e-2
    if lam_over_h > 0:
        assert abs(res["u_slip_error_pct"]) < 1e-2
    assert all(gates(res).values())


def test_metric_pipeline_rejects_a_known_three_percent_offset():
    ratio = 0.25
    res = evaluate(_synthetic_case(ratio, scale=1.03), H, G, NU, RHO, 15,
                   ratio * H)
    assert res["u_max_error_pct"] == pytest.approx(3.0, abs=1e-4)
    # a 3 % uniform bias must be visible but still inside the 5 % gate
    assert gates(res)["A_centreline_5pct"]
    res8 = evaluate(_synthetic_case(ratio, scale=1.08), H, G, NU, RHO, 15,
                    ratio * H)
    assert not gates(res8)["A_centreline_5pct"]


def test_flux_enhancement_is_one_plus_six_lam_over_h():
    for ratio in (0.0, 0.125, 0.25, 0.375, 0.5):
        assert (flux_ref(H, G, NU, ratio * H) / flux_ref(H, G, NU, 0.0)
                == pytest.approx(1.0 + 6.0 * ratio, rel=1e-12))


def test_solver_slip_is_active_and_scales_with_lambda():
    """Short channel run: slip must appear for lam > 0 and vanish at lam = 0."""
    torch.set_num_threads(2)
    from benchmarks.poiseuille.run import make_config
    from benchmarks.poiseuille_slip.run import run_case_slip

    resolution, t_end = 8, 0.4
    dx = H / resolution
    cfg = make_config(dx, NU, 1.0, "cpu")
    out = {}
    for ratio in (0.0, 0.25):
        lam = ratio * H
        case = run_case_slip(cfg, H, 12 * dx, G, t_end, 0.1, "cpu", lam)
        out[ratio] = evaluate(case, H, G, NU, RHO, resolution, lam)
    # no-slip: the fitted wall intercept stays near zero
    assert abs(out[0.0]["u_slip_measured_bottom"]) / out[0.0][
        "u_max_measured"] < 0.05
    # slip: a wall slip velocity of the right order appears
    slip = out[0.25]["u_slip_measured_bottom"]
    assert slip > 0.3 * out[0.25]["u_max_measured"]
    assert slip > 3.0 * abs(out[0.0]["u_slip_measured_bottom"])
    # and the centre line has grown by roughly the predicted 1 + 4 lam/H
    # t_end = 0.4 s leaves the slip member further from its (larger) steady
    # state than the no-slip member, so the growth is a lower bound here.
    growth = out[0.25]["u_max_measured"] / out[0.0]["u_max_measured"]
    assert 1.3 < growth < 2.3