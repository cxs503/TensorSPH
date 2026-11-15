"""Start-up transient of plane Poiseuille flow with mirror/ghost no-slip walls.

Guards three things:

1. the Boussinesq series used as reference is cross-checked against an
   independent Crank-Nicolson solve of the same initial-boundary-value problem;
2. ``benchmarks/poiseuille_startup/run.py`` really imposes the walls -- it
   regressed once, silently, by building an *empty* wall set
   (``build_mirror_walls`` -> ``torch.empty((0, 2))``), which turns the run into
   a wall-less rigid-body acceleration that can pass a loose 15 % gate;
3. the two viscosity estimators of
   ``benchmarks/poiseuille_startup_mirror/run.py`` (amplitude and rate) recover
   a known viscosity from synthetic exact data, so a measured mismatch is
   attributable to the solver and not to the estimator.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from benchmarks.poiseuille.run import make_config
from benchmarks.poiseuille_startup.run import boussinesq_series, run_startup_case
from benchmarks.poiseuille_startup_mirror.run import (
    GATES,
    check_reference_provenance,
    fit_rate_viscosity,
    series_centre,
    smooth,
    solve_amplitude_viscosity,
)

H = 0.1
NU = 0.01
G = 9.81
TAU = H ** 2 / (NU * math.pi ** 2)
U_MAX_INF = G * H ** 2 / (8.0 * NU)


# --------------------------------------------------------------------------- #
# 1. Reference provenance
# --------------------------------------------------------------------------- #

def test_reference_series_matches_independent_fd_solution():
    """The typed series must equal an independent Crank-Nicolson solve."""
    prov = check_reference_provenance(H, G, NU)
    assert prov["agrees_to_1e-4"], prov["comparison"]
    assert prov["worst_rel_to_umax"] < 1e-4


def test_reference_provenance_catches_a_wrong_viscosity():
    """A 20 % viscosity error must be visible to the same check (no tautology)."""
    prov = check_reference_provenance(H, G, NU)
    # same IVP, different viscosity: the pair must no longer agree
    from benchmarks.poiseuille_startup_mirror.run import fd_reference
    times = [0.5 * TAU, 1.0 * TAU, 2.0 * TAU]
    fd = fd_reference(times, H, G, NU)
    worst = 0.0
    for t in times:
        u_series = boussinesq_series(fd["y"], t, H, G, 1.2 * NU, n_terms=400)
        worst = max(worst, float(np.abs(fd["profiles"][t][1:-1]
                                        - u_series[1:-1]).max() / U_MAX_INF))
    assert worst > 1e-2, f"20 % nu error must show: worst={worst}"
    assert prov["worst_rel_to_umax"] < 1e-4


def test_smoothing_bias_is_small_in_the_graded_window():
    """The box average used on the history must not move the reference by >0.5 %."""
    t = np.linspace(0.2 * TAU, 5.5 * TAU, 400)
    ref = series_centre(t, H, G, NU)
    sel = (t >= 0.25 * TAU) & (t <= 5.0 * TAU)
    bias = np.max(np.abs(smooth(t, ref, 0.05 * TAU)[sel] - ref[sel])
                  / np.abs(ref[sel]))
    assert bias < 5e-3, f"smoothing bias {bias:.3%} is too large to grade at 5 %"


# --------------------------------------------------------------------------- #
# 2. The repaired startup harness must impose walls
# --------------------------------------------------------------------------- #

SAMPLE_TIMES = (0.5 * TAU, 2.0 * TAU)


@pytest.fixture(scope="module")
def short_runs():
    torch.set_num_threads(1)
    cfg = make_config(H / 8, NU, 1.0, "cpu")
    t_end = SAMPLE_TIMES[-1]
    out = {}
    for name, kw in (("walled", {}), ("wallless", {"wallless": True})):
        res = run_startup_case(cfg, H, 0.4, G, t_end=t_end,
                               sample_times=list(SAMPLE_TIMES), device="cpu",
                               **kw)
        out[name] = {}
        for t in SAMPLE_TIMES:
            prof = res["profiles"][t]["profile"]
            rows = np.array([r["y"] for r in prof])
            u = np.array([r["u_mean"] for r in prof])
            ic = int(np.argmin(np.abs(rows - 0.5 * H)))
            ref = float(boussinesq_series(rows[ic:ic + 1], t, H, G, NU)[0])
            out[name][t] = {"centre": float(u[ic]), "ref": ref,
                            "err_pct": 100.0 * (u[ic] / ref - 1.0),
                            "rows": rows, "u": u}
    return out


def test_walled_startup_tracks_the_series(short_runs):
    """With ghosts at both planes the centre line is within 8 % of the series."""
    for t in SAMPLE_TIMES:
        assert abs(short_runs["walled"][t]["err_pct"]) < 8.0, short_runs["walled"][t]


def test_wallless_startup_is_far_from_the_series(short_runs):
    """Regression guard: an empty wall set must give a grossly wrong answer.

    This is the failure the module docstring of
    ``benchmarks/poiseuille_startup_mirror`` documents: a wall-less run
    accelerates like a rigid body (``u = g t``) and would sail through a loose
    gate.  If a future edit re-introduces an empty wall list, the walled and
    wall-less numbers collapse onto each other and this test fails.
    """
    err = short_runs["wallless"][SAMPLE_TIMES[-1]]["err_pct"]
    assert err > 25.0, (
        f"wall-less run deviates only {err:+.1f}% at t=2 tau -- walls may be "
        "missing from the walled path too")


def test_short_horizon_cannot_detect_a_missing_wall(short_runs):
    """Documented trap: at ``t = 0.5 tau`` a wall-less run is *accidentally* close.

    Rigid-body acceleration gives ``u_centre = g t``; the exact solution at
    ``t = 0.5 tau`` sits within a few per cent of that, so a short-horizon gate
    (or the 15 % gate of the old startup harness) cannot tell a wall-less run
    from a walled one.  This test pins that coincidence so the graded window of
    the benchmark cannot silently shrink back below ~2 tau.
    """
    early = short_runs["wallless"][SAMPLE_TIMES[0]]["err_pct"]
    late = short_runs["wallless"][SAMPLE_TIMES[-1]]["err_pct"]
    assert abs(early) < 15.0, early          # the illusion
    assert late > 5.0 * max(abs(early), 1.0)  # ... gone by t = 2 tau


def test_walled_and_wallless_are_distinguishable(short_runs):
    """The two runs must not produce the same field (silent-stub detector)."""
    a = np.max(np.abs(short_runs["walled"][SAMPLE_TIMES[-1]]["u"]))
    b = np.max(np.abs(short_runs["wallless"][SAMPLE_TIMES[-1]]["u"]))
    assert abs(a - b) / max(b, 1e-12) > 0.05, (a, b)


# --------------------------------------------------------------------------- #
# 3. Viscosity estimators recover a known viscosity
# --------------------------------------------------------------------------- #

def test_rate_viscosity_estimator_recovers_a_synthetic_viscosity():
    nu_true = 1.15 * NU
    t = np.linspace(0.25 * TAU, 5.0 * TAU, 60)
    u = series_centre(t, H, G, nu_true)
    u_late = float(series_centre([8.0 * TAU], H, G, nu_true)[0])
    fit = fit_rate_viscosity(t, u, u_late, H, G, NU)
    assert abs(fit["nu_fit"] / nu_true - 1.0) < 0.02, fit
    assert fit["shape_rms"] < 5e-3, fit


def test_amplitude_viscosity_estimator_recovers_a_synthetic_viscosity():
    nu_true = 1.1 * NU
    u_late = float(series_centre([8.0 * TAU], H, G, nu_true)[0])
    nu_amp = solve_amplitude_viscosity(8.0 * TAU, u_late, H, G, NU)
    assert abs(nu_amp / nu_true - 1.0) < 0.02, (nu_amp, nu_true)


# --------------------------------------------------------------------------- #
# 4. Gate bookkeeping
# --------------------------------------------------------------------------- #

def _fake_case(**kw):
    base = {"H_over_dx": 16, "history_max_err_pct": 2.0,
            "profile_rms_max_pct": 2.0, "nu_amp_over_nu": 1.01,
            "nu_rate_over_nu": 1.01, "nu_rate_over_nu_amp": 1.0}
    base.update(kw)
    return base


def test_gates_pass_on_clean_measurements():
    from benchmarks.poiseuille_startup_mirror.run import gates
    ladder = [_fake_case(H_over_dx=8, history_max_err_pct=3.0),
              _fake_case(H_over_dx=16)]
    ctrl = {"max_rigid_body_err_pct": 0.5}
    wl = {"history_max_err_pct": 80.0}
    g = gates(ladder, ctrl, wl)
    assert all(g.values()), g


def test_gates_flag_a_non_monotone_ladder():
    from benchmarks.poiseuille_startup_mirror.run import gates
    ladder = [_fake_case(H_over_dx=8, history_max_err_pct=1.0),
              _fake_case(H_over_dx=16, history_max_err_pct=4.0)]
    ctrl = {"max_rigid_body_err_pct": 0.5}
    wl = {"history_max_err_pct": 80.0}
    g = gates(ladder, ctrl, wl)
    assert g["G3_refinement_monotone"] is False


def test_gates_flag_a_wallless_run_that_looks_good():
    """G6 exists so a wall-less 'pass' can never be reported as a pass."""
    from benchmarks.poiseuille_startup_mirror.run import gates
    ladder = [_fake_case(H_over_dx=8), _fake_case(H_over_dx=16)]
    ctrl = {"max_rigid_body_err_pct": 0.5}
    wl = {"history_max_err_pct": 1.0}   # suspiciously good
    g = gates(ladder, ctrl, wl)
    assert g["G6_wallless_must_miss"] is False


def test_predeclared_gates_are_the_documented_ones():
    assert GATES["G1_centreline_history_5pct"] == 5.0
    assert GATES["G4c_amp_vs_rate_5pct"] == 5.0
    assert GATES["G5_free_slip_rigid_body_2pct"] == 2.0