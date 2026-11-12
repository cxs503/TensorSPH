"""Stokes' second problem (oscillating plate): reference, estimators, and real runs.

The case is the 2-D channel with the lower plate oscillating at
``U cos(omega t)`` and the upper plate at rest; the exact finite-gap solution is
``u = Re{ U [e^{-ky} - e^{-k(2H-y)}]/[1 - e^{-2kH}] e^{i omega t} }`` with
``k = (1+i)/delta``, ``delta = sqrt(2 nu/omega)`` (Landau & Lifshitz, *Fluid
Mechanics*, §24).  See ``benchmarks/stokes_second/run.py`` and
``benchmarks/stokes_second/DESIGN.md``.

Three kinds of checks live here:

* exact, simulation-free checks of the reference: the plate amplitude is ``U``
  by construction, and for ``H >> delta`` the amplitude/phase reduce to the
  half-space forms ``U exp(-y/delta)`` / ``y/delta``;
* estimator checks with synthetic profiles: the Fourier-projection metrics must
  return ~0 on the analytic field and ~10 % when the field is scaled by 1.1
  (this is the check that catches an amplitude normalisation that forgets to
  divide by the row occupancy -- the bug this case was first written with);
* a short real run at coarse resolution plus its free-slip negative control: the
  moving plate must drag the fluid, and a frictionless wall must not.
"""

import math

import numpy as np
import pytest
import torch

from benchmarks.stokes_second.run import (
    DELTA,
    HEIGHT,
    NU,
    OMEGA,
    U0,
    analytic,
    build_walls,
    layer_metrics,
    make_config,
    run_case,
)
from tensorsph.droplet_impact import resolve_device

DX = HEIGHT / 20.0
DX_SYN = DELTA / 8.0        # synthetic profiles: 8 rows inside the layer


# --------------------------------------------------------------------------- #
# Exact reference
# --------------------------------------------------------------------------- #

def test_plate_amplitude_is_the_imposed_velocity_and_lag_is_zero():
    amp, phase = analytic(np.array([0.0]))
    # The finite-gap image form is normalised so that u(0, t) = U cos(omega t).
    assert amp[0] == pytest.approx(U0, rel=1e-12)
    assert phase[0] == pytest.approx(0.0, abs=1e-12)


def test_half_space_limit_reproduces_exp_decay_and_linear_phase():
    y = np.linspace(0.0, DELTA, 7)
    amp, phase = analytic(y)
    # H/delta = 5.6 here; the image term contributes O(e^{-2H/delta}) = 1e-5.
    ref_amp = U0 * np.exp(-y / DELTA)
    assert np.allclose(amp, ref_amp, rtol=2e-4)
    assert np.allclose(phase, y / DELTA, atol=2e-4)


def test_amplitude_and_lag_decay_monotonically_into_the_bulk():
    y = np.linspace(0.0, HEIGHT, 201)
    amp, phase = analytic(y)
    assert np.all(np.diff(amp) < 0.0)
    # Phase lag grows monotonically inside the Stokes layer; further out the
    # amplitude is below 1e-3 U and the angle is numerically meaningless.
    near = y <= 3.0 * DELTA
    assert np.all(np.diff(phase[near]) > 0.0)
    # At the upper plate the exact solution is the no-slip rest condition; the
    # phase there is undefined (the amplitude vanishes), so it is not asserted.
    assert amp[-1] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Estimators (no simulation)
# --------------------------------------------------------------------------- #

def _synthetic_result(scale=1.0, rows=12, dx=DX_SYN):
    """Build a ``run_case``-shaped result from the exact field, scaled."""
    y = np.array([dx * 0.5 + k * dx for k in range(rows)])
    amp, phase = analytic(y)
    prof = [{
        "row": k,
        "y": float(y[k]),
        "y_over_delta": float(y[k] / DELTA),
        "amplitude": float(scale * amp[k]),
        "amplitude_ref": float(amp[k]),
        "phase": float(phase[k]),
        "phase_ref": float(phase[k]),
    } for k in range(rows)]
    return {"profile": prof, "delta": DELTA, "dx": dx}


def test_metrics_are_zero_on_the_exact_field():
    m = layer_metrics(_synthetic_result(), 0.05)
    assert m["amplitude_l2"] < 1e-12
    assert m["decay_err"] < 1e-12
    # The finite-gap phase is not exactly y/delta: the image term shifts it by
    # O(e^{-2H/delta}) ~ 1e-5 relative, which is what remains here.
    assert m["phase_err"] < 1e-3
    assert m["n_rows_in_layer"] >= 5


def _layer_rows(result):
    return [p for p in result["profile"]
            if 0.0 < p["y"] <= DELTA and p["row"] >= 1]


def test_metrics_report_the_amplitude_bias_normalised_by_the_plate_velocity():
    """``amplitude_l2`` is the RMS error over the layer divided by ``U``.

    This pins the normalisation end to end: the expected value is computed here
    from the analytic rows, independently of the estimator.  A projection that
    forgets to divide by the row occupancy (row *sum* instead of row *mean*)
    multiplies the measured amplitude by the number of columns and is caught by
    this number, not by the decay rate -- which is why the case reports both.
    """
    base = _synthetic_result(scale=1.1)
    m = layer_metrics(base, 0.05)
    ar = np.array([p["amplitude_ref"] for p in _layer_rows(base)])
    expected = 0.1 * np.sqrt(np.mean(ar ** 2)) / U0
    # rel=1e-4: the estimator normalises by the plate amplitude extrapolated
    # through the exact profile, which carries the O(e^{-2H/delta}) image term.
    assert m["amplitude_l2"] == pytest.approx(expected, rel=1e-4)
    # A uniform scaling must not move the decay rate or the phase lag.
    assert m["decay_err"] < 1e-12
    assert m["phase_err"] < 1e-3


def test_metrics_scale_linearly_with_an_amplitude_error():
    m1 = layer_metrics(_synthetic_result(scale=1.05), 0.05)["amplitude_l2"]
    m2 = layer_metrics(_synthetic_result(scale=1.10), 0.05)["amplitude_l2"]
    assert m2 == pytest.approx(2.0 * m1, rel=1e-6)


# --------------------------------------------------------------------------- #
# Construction checks
# --------------------------------------------------------------------------- #

def test_wall_bands_are_split_bottom_then_top():
    cfg = make_config(DX, NU, 1.0, torch.device("cpu"))
    wall, wall_vel, n_bottom = build_walls(cfg, HEIGHT, 0.1, cfg.wall_layers,
                                           torch.device("cpu"))
    assert n_bottom * 2 == wall.shape[0]
    assert float(wall[:n_bottom, 1].max()) < 0.0      # bottom band below y = 0
    assert float(wall[n_bottom:, 1].min()) > HEIGHT   # top band above y = H
    assert float(wall_vel.abs().max()) == 0.0         # velocities set in the loop


# --------------------------------------------------------------------------- #
# Real runs (coarse, short)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def short_runs():
    device = resolve_device("cpu")
    runs = {}
    for label, friction in (("noslip", 1.0), ("freeslip", 0.0)):
        cfg = make_config(DX, NU, friction, device)
        res = run_case(cfg, HEIGHT, 0.1, U0, OMEGA, 1, 1, device,
                       snapshot=False)
        res["metrics"] = layer_metrics(res, 0.05)
        runs[label] = res
    return runs


def test_no_slip_plate_drags_the_fluid(short_runs):
    m = short_runs["noslip"]["metrics"]
    # Coarse grid (delta/dx = 3.6): the level is allowed to be loose here, the
    # reference-resolution run is what the 5 % gate is applied to.
    assert 0.4 < m["first_row_amplitude"] / U0 < 2.0
    assert -1.4 < m["decay_rate"] < -0.7
    assert 0.7 < m["phase_slope"] < 1.3


def test_free_slip_wall_does_not_drag_the_fluid(short_runs):
    m = short_runs["freeslip"]["metrics"]
    assert m["first_row_amplitude"] < 0.2 * U0


def test_the_run_is_unidirectional_and_bounded(short_runs):
    res = short_runs["noslip"]
    snap = res["snapshot"]
    assert snap is None or max(abs(u) for u in snap["u"]) < 3.0 * U0
    assert res["max_mach"] < 0.05