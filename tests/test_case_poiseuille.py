"""Tests for CASE-POISEUILLE-SPH: Womersley reference, FD reference, SPH smoke.

The unit tests are cheap (closed forms + a 1-D finite-difference solve, well
under a second each).  The last test runs the actual SPH channel for one settle
period plus one measurement period at ``H/dx = 8`` (~30 s on one core), which is
the smallest run that still exercises the whole measurement chain.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmarks.case_poiseuille import ref as refmod  # noqa: E402
from tensorsph import womersley as wm  # noqa: E402
from tensorsph.stokes_second import harmonic_fit  # noqa: E402

H = 0.1
NU = 0.01
G0 = 9.81
OMEGA = 13.0


# --------------------------------------------------------------------------- #
# Library reference
# --------------------------------------------------------------------------- #

def test_symbolic_identities_hold():
    """The closed form solves the ODE and both no-slip conditions (sympy)."""
    out = wm.verify_symbolic()
    if not out.get("available"):
        pytest.skip(f"sympy unavailable: {out.get('reason')}")
    assert out["ode_residual_is_zero"]
    assert out["bc_at_y0_is_zero"] and out["bc_at_yH_is_zero"]
    assert out["alpha_squared_is_iomega_over_nu"]
    assert out["omega_to_zero_limit_equals_steady_parabola"]


def test_amplitude_vanishes_at_both_walls():
    amp = wm.amplitude([0.0, H], H, G0, NU, OMEGA)
    assert np.max(np.abs(amp)) < 1e-12 * G0 / OMEGA


def test_slow_limit_is_the_steady_parabola():
    """Wo -> 0 must return the steady profile amplitude and phase zero."""
    y = np.linspace(0.05 * H, 0.95 * H, 7)
    steady = wm.steady_velocity(y, H, G0, NU)
    amp = wm.amplitude(y, H, G0, NU, 1e-6)
    phase = wm.phase(y, H, G0, NU, 1e-6)
    assert np.max(np.abs(amp - steady)) / np.max(steady) < 1e-6
    assert np.max(np.abs(phase)) < 1e-5      # radians; round-off only


def test_womersley_number_and_layer_thickness():
    wo = wm.womersley_number(H, NU, OMEGA)
    delta = wm.stokes_layer_thickness(NU, OMEGA)
    # Wo = (H/2) sqrt(omega/nu) <-> delta = H/Wo * sqrt(2)/1 ... exact chain:
    assert wo == pytest.approx(0.5 * H * math.sqrt(OMEGA / NU))
    assert delta == pytest.approx(math.sqrt(2.0 * NU / OMEGA))
    # the two groups are not independent: delta = H / (sqrt(2) Wo)
    assert delta == pytest.approx(H / (math.sqrt(2.0) * wo))


def test_series_from_rest_approaches_the_periodic_state():
    t = 6.0 * 2.0 * math.pi / OMEGA       # many transient times later
    y = np.linspace(0.0, H, 41)
    periodic = wm.periodic_velocity(y, t, H, G0, NU, OMEGA)
    series = wm.series_from_rest(y, t, H, G0, NU, OMEGA, n_terms=401)
    assert np.linalg.norm(series - periodic) / np.linalg.norm(periodic) < 1e-3


def test_startup_series_matches_the_repository_boussinesq_form():
    """The omega=0 branch must equal the shipped start-up series exactly."""
    from benchmarks.poiseuille_startup.run import boussinesq_series
    y = np.linspace(0.0, H, 41)
    for t in (0.05, 0.20, 0.50):
        mine = wm.startup_velocity(y, t, H, G0, NU, n_terms=801)
        theirs = boussinesq_series(y, t, H, G0, NU, n_terms=200)
        assert np.max(np.abs(mine - theirs)) / np.max(np.abs(theirs)) < 1e-8


# --------------------------------------------------------------------------- #
# Independent finite-difference reference
# --------------------------------------------------------------------------- #

def test_fd_reference_reproduces_the_closed_form():
    period = 2.0 * math.pi / OMEGA
    times = [2.0 * period + i * period / 32 for i in range(64)]
    sol = refmod.cn_solve(lambda t: G0 * math.cos(OMEGA * t), H, NU,
                          times[-1], times, nx=201, n_steps=8000)
    k = int(np.argmin(np.abs(sol["y"] - H / 2)))
    fit = harmonic_fit(sol["times"], sol["samples"][:, k], OMEGA)
    amp_ana = float(wm.amplitude([sol["y"][k]], H, G0, NU, OMEGA)[0])
    phase_ana = float(wm.phase([sol["y"][k]], H, G0, NU, OMEGA)[0])
    assert abs(fit["amplitude"] - amp_ana) / amp_ana < 1e-3
    assert abs(fit["phase"] - phase_ana) < 1e-3          # radians
    # walls stay at rest to round-off
    assert np.max(np.abs(sol["samples"][:, 0])) < 1e-12
    assert np.max(np.abs(sol["samples"][:, -1])) < 1e-12


def test_fd_reference_reproduces_the_startup_series():
    y_rows = np.array([0.025, 0.05, 0.075])
    times = [0.10, 0.20, 0.40]
    sol = refmod.cn_solve(lambda t: G0, H, NU, times[-1], times, nx=201,
                          n_steps=4000)
    for k, t in enumerate(sol["times"]):
        prof = refmod.interpolate_to_rows(sol["y"], sol["samples"][k], y_rows)
        ref = wm.startup_velocity(y_rows, float(t), H, G0, NU)
        assert np.max(np.abs(prof - ref)) / np.max(np.abs(ref)) < 1e-3


def test_fd_reference_is_grid_converged():
    period = 2.0 * math.pi / OMEGA
    times = [2.0 * period + i * period / 16 for i in range(16)]
    out = refmod.convergence_check(lambda t: G0 * math.cos(OMEGA * t), H, NU,
                                   times[-1], times,
                                   y_rows=np.linspace(0.02, 0.08, 7),
                                   refinements=(51, 101, 201, 401))
    diffs = out["successive_max_rel_diff"]
    assert out["finest_vs_coarsest"] < 2e-3
    # successive grid differences must fall: second order means a factor ~4
    assert all(b < a for a, b in zip(diffs[:-1], diffs[1:]))
    observed_order = math.log2(diffs[0] / diffs[-1]) / 2.0
    assert observed_order > 1.5


# --------------------------------------------------------------------------- #
# Estimator convention
# --------------------------------------------------------------------------- #

def test_harmonic_fit_phase_matches_the_closed_form_convention():
    """``u = A cos(omega t + phi)`` -- fit and reference must use one sign."""
    period = 2.0 * math.pi / OMEGA
    times = np.array([2.0 * period + i * period / 64 for i in range(64)])
    y = np.array([H / 2, 0.75 * H])
    sig = wm.periodic_velocity(y, times, H, G0, NU, OMEGA)   # shape (2, nt)
    for k, yk in enumerate(y):
        fit = harmonic_fit(times, sig[k], OMEGA)
        assert fit["amplitude"] == pytest.approx(
            float(wm.amplitude([yk], H, G0, NU, OMEGA)[0]), rel=1e-10)
        assert fit["phase"] == pytest.approx(
            float(wm.phase([yk], H, G0, NU, OMEGA)[0]), abs=1e-10)


# --------------------------------------------------------------------------- #
# End-to-end SPH smoke
# --------------------------------------------------------------------------- #

def test_sph_channel_smoke_harmonic():
    """One settle + one measurement period at H/dx = 8 must land inside G1..G4."""
    from benchmarks.case_poiseuille.run import run_harmonic
    from tensorsph.droplet_impact import resolve_device
    import torch

    torch.set_num_threads(1)
    res = run_harmonic(8, OMEGA, resolve_device("cpu"), settle_periods=1,
                       window_periods=1)
    assert abs(res["amp_centre_error_pct"]) <= 5.0
    assert res["amp_profile_rms_pct"] <= 5.0
    assert res["phase_centre_pct_of_quarter_turn"] <= 5.0
    assert res["phase_profile_rms_pct_of_quarter_turn"] <= 5.0
    assert res["max_mach"] < 0.1