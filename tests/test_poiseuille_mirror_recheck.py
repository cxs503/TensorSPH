"""Invariant tests for the independent Poiseuille recheck estimators.

The recheck benchmark (``benchmarks/poiseuille_mirror/adversarial_recheck.py``)
is expensive to run, so its *estimators* are pinned here on synthetic profiles
whose answer is known in closed form.  What is being protected:

* the three-point one-sided wall shear stencil returns ``C H`` exactly on the
  analytic parabola ``u = C y (H - y)``, i.e. it is not biased by the row
  spacing in the way the one-``dx`` secant is;
* the order-of-accuracy fit recovers the exponent of a power law, and
  ``resolution_needed`` inverts it;
* ``measure()`` reproduces the analytic peak, the analytic wall shear and a
  zero row error when it is handed the exact solution, and *fails* the 5 % gates
  by the right amount when the profile is deliberately scaled by 1.1;
* ``gates()`` is a real threshold comparison, not a constant.
"""

from __future__ import annotations

import math

import pytest

from benchmarks.poiseuille_mirror.adversarial_recheck import (
    GX,
    HEIGHT,
    NU,
    RHO0,
    _secant_wall_gradient,
    _three_point_wall_gradient,
    analytic,
    gates,
    measure,
    order_fit,
    resolution_needed,
)

RES = 16
DX = HEIGHT / RES
U_MAX = GX * HEIGHT**2 / (8.0 * NU)
TAU = RHO0 * GX * HEIGHT / 2.0


def synthetic_profile(scale: float = 1.0, res: int = RES) -> list[dict]:
    """Rows on the same lattice the solver uses, sampled from the exact profile."""
    dx = HEIGHT / res
    rows = []
    for k in range(res):
        y = dx * 0.5 + k * dx
        u = scale * analytic(y)
        rows.append({
            "row": k,
            "y": y,
            "y_over_H": y / HEIGHT,
            "u": u,
            "u_mean": u,
            "u_std_x": 0.0,
        })
    return rows


def test_three_point_gradient_is_exact_on_the_parabola():
    rows = synthetic_profile()
    # du/dy at y = 0 is C H with C = g/(2 nu); the stencil must return it.
    exact = GX * HEIGHT / (2.0 * NU)
    got = _three_point_wall_gradient(rows, DX)
    assert got == pytest.approx(exact, rel=1e-12)


def test_secant_and_three_point_estimators_differ():
    """The merged secant reads a different (biased) number on the same rows."""
    rows = synthetic_profile()
    exact = GX * HEIGHT / (2.0 * NU)
    sec = _secant_wall_gradient(rows, DX)
    # u(dx/2)/(dx/2) = C (H - dx/2), i.e. low by dx/(2H) -- a known bias.
    assert sec == pytest.approx(exact * (1.0 - DX / (2.0 * HEIGHT)), rel=1e-12)
    assert sec < exact


def test_order_fit_recovers_a_known_exponent():
    pts = [(0.1 / r, 3.0 * (0.1 / r) ** 2) for r in (8, 12, 16, 24, 32)]
    fit = order_fit(pts)
    assert fit["p"] == pytest.approx(2.0, abs=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)
    assert fit["n_points"] == 5
    assert fit["p_finest_pair"] == pytest.approx(2.0, abs=1e-9)


def test_order_fit_ignores_non_positive_errors():
    fit = order_fit([(0.1 / 8, 0.0), (0.1 / 16, float("nan")), (0.1 / 32, 1e-3)])
    assert fit["n_points"] == 1
    assert math.isnan(fit["p"])


def test_resolution_needed_inverts_the_fit():
    pts = [(0.1 / r, 3.0 * (0.1 / r) ** 2) for r in (8, 16, 32)]
    fit = order_fit(pts)
    # the synthetic law is err = 3 dx^2 with dx = H/R, so err = 3 H^2 / R^2 and
    # the resolution at which it reaches 5 is R = H sqrt(3/5)
    assert resolution_needed(fit, 5.0) == pytest.approx(HEIGHT * math.sqrt(3.0 / 5.0),
                                                        rel=1e-9)


def test_measure_is_zero_error_on_the_exact_solution():
    case = {"profile": synthetic_profile(), "n_fluid": RES, "n_ghost": 0,
            "dt": 1e-4, "steps": 10, "trace": [{"max_mach": 0.06}],
            "steady_drift": 0.0, "wall_clock_s": 0.0}
    # the row averages sit at y = dx/2 + k dx, so the row maximum is the
    # analytic value *at a row*, which is a fraction of a percent below the
    # continuum peak: that offset is a property of the lattice, not an error
    row_peak = max(analytic(r["y"]) for r in case["profile"])
    res = measure(case, RES)
    assert res["u_max_measured"] == pytest.approx(row_peak, rel=1e-12)
    assert abs(res["u_max_error_pct"]) < 0.5
    assert res["profile_rms_interior_pct"] < 1e-9
    assert res["tau_bottom_3pt"] == pytest.approx(TAU, rel=1e-12)
    assert res["tau_top_3pt"] == pytest.approx(TAU, rel=1e-12)
    assert abs(res["tau_bottom_error_pct"]) < 1e-9
    assert all(gates(res).values())


def test_measure_flags_a_ten_percent_scale_error():
    case = {"profile": synthetic_profile(scale=1.1), "n_fluid": RES, "n_ghost": 0,
            "dt": 1e-4, "steps": 10, "trace": [{"max_mach": 0.06}],
            "steady_drift": 0.0, "wall_clock_s": 0.0}
    row_peak = max(analytic(r["y"]) for r in case["profile"])
    expected = 100.0 * (1.1 * row_peak - U_MAX) / U_MAX
    res = measure(case, RES)
    assert expected > 9.0                     # the injected defect is real
    assert res["u_max_error_pct"] == pytest.approx(expected, rel=1e-9)
    assert res["tau_bottom_error_pct"] == pytest.approx(10.0, rel=1e-9)
    g = gates(res)
    assert g["A_centreline_5pct"] is False
    assert g["C_wall_shear_5pct"] is False
    assert g["B_profile_rms_5pct"] is False


def test_gates_are_threshold_comparisons():
    ok = {"u_max_error_pct": 4.9, "profile_rms_interior_pct": 4.9,
          "tau_bottom_error_pct": -4.9, "tau_top_error_pct": -4.9,
          "wall_row_bottom_error_pct": 9.9, "wall_row_top_error_pct": -9.9}
    assert all(gates(ok).values())
    bad = dict(ok, tau_top_error_pct=-5.1)
    assert gates(bad)["C_wall_shear_5pct"] is False
    assert gates(dict(ok, u_max_error_pct=5.1))["A_centreline_5pct"] is False