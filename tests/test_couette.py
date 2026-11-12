"""Plane Couette flow: closed-form reference, analysis machinery, and a real run.

The case is a 2-D channel with a stationary wall at ``y = 0`` and a wall sliding
at ``U`` at ``y = H``; the exact steady solution is ``u = U y/H`` (Batchelor
1967, §4.2).  See ``benchmarks/couette/run.py`` and
``benchmarks/couette/DESIGN.md``.

Three kinds of checks live here:

* exact, simulation-free checks of the reference and of the analysis
  (``profile_metrics``) -- a slip profile ``u = U (y+d)/(H+2d)`` has a *known*
  flux ratio of exactly 1, which pins the flux estimator without any fluid;
* construction checks: the sliding wall must carry ``U`` on its top rows and
  zero on its bottom rows, otherwise the case is not a Couette flow;
* one short real run (coarse, ~0.24 viscous times) plus its free-slip negative
  control: the moving wall must drive the fluid and the free-slip wall must not.
"""

import pytest
import torch

from benchmarks.couette.run import (
    HEIGHT,
    RHO0,
    analytic,
    analytic_scalars,
    build_walls,
    make_config,
    profile_metrics,
    run_one,
)
from tensorsph.droplet_impact import resolve_device

U = 0.2
NU = 0.02
DX = HEIGHT / 8


def _synthetic(profile_fn, rows=8, dx=DX):
    """Sample ``profile_fn(y)`` on lattice row centres, as ``bin_rows`` does."""
    return [
        {
            "row": k,
            "y": dx * 0.5 + k * dx,
            "y_over_H": (dx * 0.5 + k * dx) / HEIGHT,
            "u_mean": profile_fn(dx * 0.5 + k * dx),
            "u_std_x": 0.0,
            "count": 12,
        }
        for k in range(rows)
    ]


def _continuum_l2(profile_fn, lo=0.15, hi=0.85, n=20001):
    """RMS of ``(u_f - u_analytic)/U`` over the interior band, by quadrature.

    The discrete metric averages over the *rows* that fall in the band, this one
    integrates the continuum error; agreeing to a few percent is the check that
    the row selection and the normalisation of ``interior_l2`` are right.
    """
    y = torch.linspace(lo * HEIGHT, hi * HEIGHT, n, dtype=torch.float64)
    err = (torch.tensor([profile_fn(float(v)) for v in y], dtype=torch.float64)
           - U * y / HEIGHT) / U
    return float(torch.sqrt(torch.trapz(err * err, y) / (y[-1] - y[0])))


# --------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------- #

def test_analytic_profile_is_the_couette_line():
    assert analytic(0.0, HEIGHT, U) == pytest.approx(0.0)
    assert analytic(HEIGHT, HEIGHT, U) == pytest.approx(U)
    assert analytic(HEIGHT / 2, HEIGHT, U) == pytest.approx(U / 2)
    # the slope is independent of y: that is what "linear" means here
    for frac in (0.1, 0.3, 0.7, 0.9):
        y = frac * HEIGHT
        assert analytic(y, HEIGHT, U) == pytest.approx(U * frac, rel=1e-15)


def test_analytic_scalars_match_their_independent_formulas():
    ref = analytic_scalars(HEIGHT, U, NU)
    assert ref["slope"] == pytest.approx(U / HEIGHT)
    assert ref["flux"] == pytest.approx(0.5 * U * HEIGHT)      # area of a triangle
    assert ref["wall_shear"] == pytest.approx(RHO0 * NU * U / HEIGHT)


# --------------------------------------------------------------------------- #
# Analysis machinery (no fluid)
# --------------------------------------------------------------------------- #

def test_analysis_is_exact_on_the_analytic_profile():
    metrics = profile_metrics(_synthetic(lambda y: analytic(y, HEIGHT, U)),
                              HEIGHT, U, NU)
    assert metrics["interior_l2"] < 1e-12
    assert metrics["flux_ratio"] == pytest.approx(1.0, rel=1e-12)
    assert metrics["wall_shear_ratio"] == pytest.approx(1.0, rel=1e-12)
    assert abs(metrics["linearity_deviation_over_U"]) < 1e-12
    assert metrics["slip_length_bottom_over_dx"] == pytest.approx(0.0, abs=1e-12)


def test_flux_is_invariant_under_a_symmetric_wall_slip():
    """``u = U (y+d)/(H+2d)`` has ``Q = U H / 2`` for every slip length ``d``.

    That identity is why the flux is a robust gate: the slip that the SPH wall
    inevitably has cancels in ``Q``, while the wall shear picks up the factor
    ``H/(H+2d)`` -- so the two metrics separate slip from a viscous-term error.
    """
    for d_over_dx in (0.0, 0.5, 1.0, 2.0):
        d = d_over_dx * DX

        def profile(y, d=d):
            return U * (y + d) / (HEIGHT + 2 * d)

        metrics = profile_metrics(_synthetic(profile), HEIGHT, U, NU)
        assert metrics["flux_ratio"] == pytest.approx(1.0, rel=1e-12)
        assert metrics["wall_shear_ratio"] == pytest.approx(
            HEIGHT / (HEIGHT + 2 * d), rel=1e-12)
        assert metrics["slip_length_bottom_over_dx"] == pytest.approx(
            d_over_dx, rel=1e-9, abs=1e-12)
        assert metrics["interior_l2"] == pytest.approx(
            _continuum_l2(profile), rel=0.10)


def test_half_the_analytic_slope_is_reported_as_a_fifty_percent_error():
    metrics = profile_metrics(_synthetic(lambda y: 0.5 * analytic(y, HEIGHT, U)),
                              HEIGHT, U, NU)
    assert metrics["wall_shear_ratio"] == pytest.approx(0.5, rel=1e-12)
    assert metrics["flux_ratio"] == pytest.approx(0.5, rel=1e-12)
    assert metrics["interior_l2"] == pytest.approx(
        _continuum_l2(lambda y: 0.5 * U * y / HEIGHT), rel=0.05)   # 0.5 * rms(y/H) on the band


# --------------------------------------------------------------------------- #
# Wall construction
# --------------------------------------------------------------------------- #

def test_sliding_wall_carries_u_on_top_and_zero_on_bottom():
    cfg = make_config(DX, NU, 1.0, torch.device("cpu"))
    wall, wall_vel = build_walls(cfg, HEIGHT, cfg.pool_width, cfg.wall_layers,
                                 U, torch.device("cpu"))
    bottom = wall[:, 1] < 0.0
    top = wall[:, 1] > HEIGHT
    assert bottom.any() and top.any()
    assert torch.all(wall_vel[bottom] == 0.0)
    assert torch.allclose(wall_vel[top, 0], torch.tensor(U, dtype=wall.dtype))
    assert torch.all(wall_vel[top, 1] == 0.0)
    # the first wall layer sits half a spacing outside the first fluid row
    assert wall[bottom][:, 1].max() == pytest.approx(-DX * 0.5)
    assert wall[top][:, 1].min() == pytest.approx(HEIGHT + DX * 0.5)


# --------------------------------------------------------------------------- #
# A real (short) run and its negative control
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def short_runs():
    device = resolve_device("cpu")
    t_end = 0.12                       # 0.24 viscous times: transient still live
    no_slip = run_one(8, NU, U, 1.0, HEIGHT, t_end, 0.02, device, label="test-f1")
    free_slip = run_one(8, NU, U, 0.0, HEIGHT, t_end, 0.02, device, label="test-f0")
    return no_slip, free_slip


def test_sliding_wall_drives_the_fluid(short_runs):
    no_slip, _ = short_runs
    m = no_slip["metrics"]
    # analytic flux at t=0.12 s is 0.92 of the steady value; the gate is loose on
    # purpose -- the sharp claims are made by the benchmark ladder, not here
    assert 0.6 < m["flux_ratio"] < 1.2
    assert m["interior_l2"] < 0.20
    assert m["wall_shear_ratio"] > 0.3
    assert no_slip["max_mach"] < 0.1


def test_free_slip_wall_does_not_drive_the_fluid(short_runs):
    _, free_slip = short_runs
    m = free_slip["metrics"]
    assert m["flux_ratio"] < 0.2, (
        f"free-slip control was driven: flux ratio {m['flux_ratio']:.3f}")
    # and the no-slip run must be the driven one
    assert free_slip["metrics"]["flux_ratio"] < 3.0 * 0.0 + 0.2


def test_no_slip_run_is_two_dimensional_and_steady_in_the_mean(short_runs):
    no_slip, _ = short_runs
    # the flow is x-invariant: the scatter inside a row is small compared to U
    worst = max(p["u_std_x"] for p in no_slip["profile"])
    assert worst < 0.02 * U
    assert no_slip["steps"] > 100
    assert no_slip["dt"] < 3.0e-4