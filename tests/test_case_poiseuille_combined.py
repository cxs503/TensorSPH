"""Tests for R3 of CASE-POISEUILLE-SPH: combined Couette-Poiseuille.

Everything here is cheap (closed forms, a 1-D Crank-Nicolson solve, two 20 ms
SPH runs); the graded resolution ladder lives in
``benchmarks/case_poiseuille/combined.py`` and its artifact
``out/combined.json``.

The tests are written as *invariants*, not as a replay of the driver:
the closed form must satisfy the PDE and both Dirichlet data, its wall shears
must equal the numerical derivative, the superposition identity must hold to
machine precision, the quadratic estimator must recover both shears from an
exact quadratic, the CN reference with a moving wall must reproduce the closed
form, and the new ``planes`` hook of ``run.drive`` must leave the static-wall
path bit-identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmarks.case_poiseuille import combined as cb  # noqa: E402
from benchmarks.case_poiseuille import ref as refmod  # noqa: E402
from benchmarks.case_poiseuille import run as caserun  # noqa: E402

H = cb.HEIGHT
NU = cb.VISCOSITY
UW = cb.WALL_SPEED
G = cb.G_ADVERSE


# --------------------------------------------------------------------------- #
# Closed form
# --------------------------------------------------------------------------- #

def test_profile_solves_the_ode_and_both_walls():
    y = np.linspace(0.0, H, 401)
    u = cb.analytic_profile(y)
    # second derivative of a parabola is constant -> nu u_yy + g = 0
    d2 = np.gradient(np.gradient(u, y), y)
    assert np.allclose(-NU * d2[2:-2], G, rtol=1e-6, atol=1e-6)
    assert u[0] == pytest.approx(0.0, abs=1e-15)
    assert u[-1] == pytest.approx(UW, rel=0, abs=1e-12)


def test_superposition_holds_exactly_in_the_closed_form():
    y = np.linspace(0.0, H, 101)
    total = cb.analytic_profile(y, wall_speed=UW)
    pois = cb.analytic_profile(y, wall_speed=0.0)
    cout = cb.analytic_profile(y, g=0.0, wall_speed=UW)
    assert np.allclose(total, pois + cout, rtol=0, atol=1e-14)


def test_wall_shears_are_the_numerical_derivative():
    y = np.linspace(0.0, H, 200001)
    u = cb.analytic_profile(y)
    du = np.gradient(u, y)
    mu = cb.MU
    ana = cb.analytic_wall_shear()
    # np.gradient is one-sided at the ends: 1e-5 is its own truncation, not the
    # formula's, so the tolerance sits just above it
    assert mu * du[0] == pytest.approx(ana["tau_bottom"], rel=1e-4)
    assert mu * du[-1] == pytest.approx(ana["tau_top"], rel=1e-4)


def test_wall_shears_close_the_momentum_balance():
    ana = cb.analytic_wall_shear()
    balance = cb.RHO0 * G * H + ana["tau_top"] - ana["tau_bottom"]
    assert balance == pytest.approx(0.0, abs=1e-9)
    assert ana["net_bottom_minus_top"] == pytest.approx(ana["body_force_per_area"])


def test_reversal_is_inside_the_channel_and_signs_are_opposite():
    y_star = cb.analytic_zero_crossing()
    assert y_star is not None and 0.0 < y_star < H
    assert cb.analytic_profile([0.5 * y_star])[0] < 0.0     # adverse-gradient side
    assert cb.analytic_profile([0.5 * (H + y_star)])[0] > 0.0
    ext = cb.analytic_extrema()
    assert 0.0 < ext["y"] < H and ext["u"] < 0.0
    # the reversal must stay inside the case's declared Mach envelope
    assert abs(ext["u"]) / cb.SOUND_SPEED < 0.1
    assert UW / cb.SOUND_SPEED < 0.1


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #

def test_quadratic_estimator_recovers_both_shears_from_an_exact_profile():
    """The estimator's only error source is the rows: exact data -> exact shears."""
    y = np.linspace(0.0, H, 17)
    u = cb.analytic_profile(y)
    fit = cb.quadratic_wall_shear(y, u, H)
    ana = cb.analytic_wall_shear()
    assert fit["tau_bottom"] == pytest.approx(ana["tau_bottom"], rel=1e-9)
    assert fit["tau_top"] == pytest.approx(ana["tau_top"], rel=1e-9)
    assert fit["body_force_implied"] == pytest.approx(G, rel=1e-9)
    assert fit["wall_speed_implied"] == pytest.approx(UW, rel=1e-9)


def test_measured_zero_crossing_interpolates_the_sign_change():
    y = np.linspace(0.0, H, 17)
    u = cb.analytic_profile(y)
    found = cb.measured_zero_crossing(y, u)
    assert found == pytest.approx(cb.analytic_zero_crossing(), abs=0.02 * H)


def test_measured_zero_crossing_returns_none_without_reversal():
    y = np.linspace(0.0, H, 17)
    u = cb.analytic_profile(y, wall_speed=0.0)      # pure Poiseuille, no crossing
    assert cb.measured_zero_crossing(y, u) is None


def test_reference_check_drives_the_moving_wall():
    """CN-FD with ``u(H) = U_w`` must reproduce the closed form (gate C7)."""
    out = cb.reference_check(nx=201, n_steps=4000)
    assert out["profile_rms_pct"] < 1.0
    assert out["zero_crossing_pct_of_H"] < 1.0


def test_reference_default_is_still_the_static_wall_problem():
    """``wall_values`` defaults must not perturb the existing reference."""
    t_end = 0.05
    sol = refmod.cn_solve(lambda t: G, H, NU, t_end, [t_end], nx=101, n_steps=500)
    assert sol["samples"][-1][0] == 0.0
    assert sol["samples"][-1][-1] == 0.0


# --------------------------------------------------------------------------- #
# The new ``planes`` hook of run.drive
# --------------------------------------------------------------------------- #

def test_static_wall_path_is_unchanged_by_the_moving_wall_hook():
    """Gate C8: ``planes=None`` and two static planes are bit-identical."""
    out = cb.static_equivalence("cpu", resolution=8, t_end=0.01)
    assert out["bit_identical"], f"max |diff| = {out['max_abs_diff']}"


def test_moving_wall_actually_drives_the_flow():
    """A negative control for the hook: with ``U_w = 0`` the wall cannot drag fluid."""
    dx = H / 8
    cfg = caserun.make_case_config(dx, NU, "cpu", sound_speed=cb.SOUND_SPEED)
    length = caserun.COLUMNS * dx
    kw = dict(t_end=0.01, sample_times=[0.01], device="cpu")
    moved = caserun.drive(cfg, H, length, lambda t: 0.0,
                          planes=[(0.0, +1), (H, -1, (UW, 0.0))], **kw)
    still = caserun.drive(cfg, H, length, lambda t: 0.0,
                          planes=[(0.0, +1), (H, -1, (0.0, 0.0))], **kw)
    assert float(np.max(np.abs(moved["u_samples"]))) > 0.05 * UW
    assert float(np.max(np.abs(still["u_samples"]))) < 1e-3 * UW


def test_adverse_gradient_alone_reverses():
    """With the moving wall removed the adverse gradient must give ``u < 0``."""
    dx = H / 8
    cfg = caserun.make_case_config(dx, NU, "cpu", sound_speed=cb.SOUND_SPEED)
    length = caserun.COLUMNS * dx
    case = caserun.drive(cfg, H, length, lambda t: G, 0.05, [0.05], "cpu",
                         planes=[(0.0, +1), (H, -1)])
    assert float(np.mean(case["u_samples"][-1])) < 0.0