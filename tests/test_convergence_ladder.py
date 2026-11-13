"""Tests for the TVC resolution-ladder analysis (``convergence.py``).

The analysis has no experimental input in its core leg (Cauchy
self-convergence), so it can be verified against *synthetic* curves whose
convergence order is known exactly.  That is what most of this file does: if
the machinery cannot recover a prescribed order from clean data, its output on
the dam-break data means nothing.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_sphinx_tvc"))

import convergence as cv  # noqa: E402


TAU = [i * 0.05 for i in range(0, 61)]          # 0 .. 3.0


def synthetic_level(dx: float, order: float, coeff: float = 3.0) -> list[float]:
    """True curve plus a pure ``dx**order`` error mode."""
    return [math.sin(1.4 * t) + coeff * dx ** order * math.exp(-0.3 * t)
            for t in TAU]


def record(dx: float, za: list[float]) -> dict:
    return {"variant": "synthetic", "dx": dx, "tau": TAU, "Z_over_a": za}


def exp_payload(points=6) -> dict:
    return {"data": [{"tau": 0.1 + 0.4 * i, "Z_over_a": 1.0 + 0.2 * i}
                     for i in range(points)]}


# --------------------------------------------------------------------------- #
# curve helpers
# --------------------------------------------------------------------------- #
def test_interp_is_exact_on_linear_data():
    xs = [0.0, 1.0, 2.0]
    ys = [0.0, 2.0, 4.0]
    for x in (0.0, 0.25, 0.5, 1.75, 2.0):
        assert cv.interp(xs, ys, x) == pytest.approx(2.0 * x)


def test_interp_clamps_outside_the_range():
    xs, ys = [0.0, 1.0], [1.0, 2.0]
    assert cv.interp(xs, ys, -5.0) == 1.0
    assert cv.interp(xs, ys, 5.0) == 2.0


def test_common_grid_uses_the_shared_interval():
    grid = cv.common_grid([[0.0, 1.0], [0.4, 3.0]], n=5)
    assert grid[0] == pytest.approx(0.4)
    assert grid[-1] == pytest.approx(1.0)
    assert len(grid) == 5


def test_common_grid_rejects_disjoint_series():
    with pytest.raises(ValueError):
        cv.common_grid([[0.0, 1.0], [2.0, 3.0]])


def test_cauchy_norm_of_identical_curves_is_zero():
    l2, linf = cv.cauchy_norm([1.0, 2.0], [1.0, 2.0])
    assert l2 == 0.0 and linf == 0.0


# --------------------------------------------------------------------------- #
# order / Richardson machinery
# --------------------------------------------------------------------------- #
def test_observed_order_recovers_second_order():
    e_coarse, e_fine = 1.0, 0.25          # halving dx, p = 2
    assert cv.observed_order(e_coarse, e_fine, 2.0) == pytest.approx(2.0)


def test_observed_order_recovers_first_order():
    assert cv.observed_order(1.0, 0.5, 2.0) == pytest.approx(1.0)


def test_observed_order_is_none_when_degenerate():
    assert cv.observed_order(0.0, 1.0, 2.0) is None
    assert cv.observed_order(1.0, 1.0, 1.0) is None


def test_richardson_removes_a_pure_error_mode():
    """``f(dx) = g + C dx^p s`` -> Richardson with the true ``p`` returns ``g``."""
    for order in (1.0, 2.0, 1.5):
        coarse = synthetic_level(0.04, order)
        fine = synthetic_level(0.02, order)
        exact = synthetic_level(0.0, order)
        got = cv.richardson(coarse, fine, r=2.0, p=order)
        assert max(abs(a - b) for a, b in zip(got, exact)) == pytest.approx(
            0.0, abs=1e-12)


def test_richardson_of_identical_curves_is_the_curve():
    fine = synthetic_level(0.02, 2.0)
    got = cv.richardson(fine, fine, r=2.0, p=2.0)
    assert max(abs(a - b) for a, b in zip(got, fine)) == pytest.approx(
        0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# end-to-end on synthetic payloads
# --------------------------------------------------------------------------- #
def test_analyse_variant_recovers_prescribed_order_and_limit():
    records = {dx: record(dx, synthetic_level(dx, 2.0))
               for dx in (0.08, 0.04, 0.02)}
    res = cv.analyse_variant(records, exp_payload())

    assert res["levels"] == [0.08, 0.04, 0.02]
    # Cauchy differences must fall like dx**2, i.e. factor 4 per halving
    assert len(res["pairs"]) == 2
    ratio = res["pairs"][0]["cauchy_l2"] / res["pairs"][1]["cauchy_l2"]
    assert ratio == pytest.approx(4.0, rel=0.05)
    for pair in res["pairs"]:
        assert pair["r"] == pytest.approx(2.0)
    for o in res["orders"]:
        assert o["observed_order_l2"] == pytest.approx(2.0, abs=0.05)
    assert res["cauchy_slope_fit"]["slope"] == pytest.approx(2.0, abs=0.05)

    limit = res["richardson"]
    assert limit["p_source"].startswith("measured")
    exact = synthetic_level(0.0, 2.0)
    grid = cv.common_grid([TAU, TAU], n=res["grid_points"])
    ref = cv.on_grid(TAU, exact, grid)
    from_fig = cv.richardson(
        cv.on_grid(TAU, synthetic_level(0.04, 2.0), grid),
        cv.on_grid(TAU, synthetic_level(0.02, 2.0), grid),
        r=2.0, p=limit["p_used"])
    assert max(abs(a - b) for a, b in zip(from_fig, ref)) < 1e-6


def test_analyse_variant_flags_a_single_level():
    res = cv.analyse_variant({0.04: record(0.04, synthetic_level(0.04, 2.0))},
                             exp_payload())
    assert res["levels"] == [0.04]
    assert "note" in res and "pairs" not in res


def test_error_vs_experiment_is_zero_on_the_experiment_itself():
    exp = exp_payload()
    ts = [d["tau"] for d in exp["data"]]
    zs = [d["Z_over_a"] for d in exp["data"]]
    out = cv.error_vs_experiment(ts, zs, exp)
    assert out["mae"] == pytest.approx(0.0, abs=1e-12)
    assert out["mean_rel_error"] == pytest.approx(0.0, abs=1e-12)
    assert out["within_digitisation_uncertainty"] == 1.0


def test_error_vs_experiment_ignores_points_outside_the_curve():
    exp = exp_payload()
    out = cv.error_vs_experiment([0.1, 0.5], [0.0, 0.0], exp)
    assert out["points"] == 2          # only tau = 0.1, 0.5 are covered


# --------------------------------------------------------------------------- #
# figure smoke test
# --------------------------------------------------------------------------- #
def test_figure_is_written(tmp_path):
    records = {dx: record(dx, synthetic_level(dx, 2.0))
               for dx in (0.04, 0.02)}
    payloads = {"baseline": records, "tvc_a010_all": records}
    png = tmp_path / "fig.png"
    out = cv.make_figure(payloads, exp_payload(),
                         ["baseline", "tvc_a010_all"], png)
    assert out == [str(png)]
    assert png.exists() and png.stat().st_size > 1000


# --------------------------------------------------------------------------- #
# integration: real solver output at a short window
# --------------------------------------------------------------------------- #
def test_ladder_on_live_solver_output_is_finite():
    """Two short real runs: the analysis must not produce NaN/negative norms."""
    from tensorsph.dam_break import DamBreakConfig, run_case

    from convergence import (analyse_variant, error_vs_experiment,  # noqa: F401
                             observed_order)

    records = {}
    for dx in (0.04, 0.02):
        cfg = DamBreakConfig(dx=dx, dt=DamBreakConfig(dx=dx).dt_from_courant(),
                             steps=48, output_every=4, device="cpu")
        run = run_case(cfg)
        tau = [t * math.sqrt(2 * cfg.gravity / cfg.column_width)
               for t in run.times]
        za = [(f - cfg.column_width) / cfg.column_width for f in run.front]
        records[dx] = {"variant": "baseline", "dx": dx, "tau": tau,
                       "Z_over_a": za}

    res = analyse_variant(records, exp_payload())
    assert len(res["pairs"]) == 1
    pair = res["pairs"][0]
    assert pair["cauchy_l2"] >= 0.0 and math.isfinite(pair["cauchy_l2"])
    assert math.isfinite(pair["cauchy_linf"])
    # a single pair cannot yield an order -> the field must be present but empty
    assert res["orders"] == []
    assert math.isfinite(observed_order(pair["cauchy_l2"], 1.0, 2.0))