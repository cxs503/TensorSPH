"""Tests for the similarity-corrected FSI2 coupled driver.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

What is asserted (invariants, not just execution)
-------------------------------------------------
* the beam first-bending frequency helper reproduces an independently typed
  closed form (``1.8751^2/(2 pi) sqrt(EI/rhoA L^4)``) -- the number that fixes
  the measurement window;
* ``window_stats`` really is a half-range over ``t >= t_start`` (checked on a
  synthetic series with a known answer, including the ``t_start`` cut);
* ``dominant_frequency`` recovers an injected sine frequency;
* the checkpoint round-trip is exact (bitwise on the beam nodes, and it carries
  ``state.time``) so a long run can be chained across queue jobs;
* a 12-step coupled run at the coarsest usable resolution is finite, advances
  ``state.time`` by exactly ``n * dt`` and stays on the similarity point.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_fsi2"))

import run_similarity_corrected as drv  # noqa: E402
from tensorsph.fsi2_similarity import similarity_report  # noqa: E402
from tensorsph.fsi2_turek_hron import initial_state  # noqa: E402
from tensorsph.fsi2_strong_coupling import run_fsi2_strong  # noqa: E402


def test_beam_first_frequency_matches_closed_form():
    case = drv.build_case(dx=0.02, u_mean=0.2, c0=5.0, ratio=1.0)
    got = drv.beam_first_frequency(case)
    t, L = case.beam_thickness, case.beam_length
    ei = case.beam_E * t ** 3 / 12.0
    rho_a = case.beam_rho * t
    want = 1.8751 ** 2 / (2.0 * math.pi) * math.sqrt(ei / (rho_a * L ** 4))
    assert got == pytest.approx(want, rel=1e-12)
    # Turek-Hron FSI2 beam period is ~2.5 s at the similarity point: the run
    # must therefore last several seconds, which is what the driver enforces.
    assert 1.5 < 1.0 / got < 4.0


def test_window_stats_half_range_and_start_cut():
    t = np.arange(0.0, 1.0, 0.01)
    series = {"t": t.tolist(),
              "tip_x": np.zeros_like(t).tolist(),
              "tip_y": np.sin(2 * np.pi * t).tolist()}
    full = drv.window_stats(series, 0.0)
    assert full["tip_y_amplitude"] == pytest.approx(1.0, abs=0.02)
    assert full["tip_y_mean"] == pytest.approx(0.0, abs=0.02)
    # cutting at t = 0.25 keeps only samples at or after the crest: the
    # half-range of a *partial* sine window is not the full amplitude, so only
    # the raw min/max and the sample count are asserted here
    cut = drv.window_stats(series, 0.25)
    assert cut["n_samples"] == 75
    assert cut["tip_y_max"] == pytest.approx(1.0, abs=0.02)
    assert cut["tip_y_min"] == pytest.approx(-1.0, abs=0.02)
    # an all-zero series has zero amplitude, not NaN
    zeros = {"t": t.tolist(), "tip_x": np.zeros_like(t).tolist(),
             "tip_y": np.zeros_like(t).tolist()}
    assert drv.window_stats(zeros, 0.0)["tip_y_amplitude"] == 0.0


def test_dominant_frequency_recovers_injected_sine():
    dt = 0.01
    t = np.arange(0.0, 20.0, dt)
    f0 = 0.4
    y = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.05
    assert drv.dominant_frequency(t, y) == pytest.approx(f0, rel=0.05)


def test_checkpoint_round_trip_is_exact(tmp_path):
    case = drv.build_case(dx=0.05, u_mean=0.2, c0=5.0, ratio=1.0)
    state = initial_state(case)
    # perturb so the checkpoint cannot be confused with a fresh state
    state.beam_nodes = state.beam_nodes + 1.0e-4
    state.beam_vel = state.beam_vel + 2.0e-4
    state.fluid_vel = state.fluid_vel + 1.0e-5
    state.time = 0.123
    hist = {"t": [0.1, 0.123], "tip_x": [0.0, 1e-5], "tip_y": [0.0, -2e-5]}
    path = tmp_path / "ck.pt"
    meta = {"dx": 0.05, "c0": 5.0, "ratio": 1.0, "solver": "iqn-ils",
            "u_mean": 0.2}
    drv.save_checkpoint(path, state, hist, meta)

    fresh = initial_state(case)
    back, prev = drv.load_checkpoint(path, fresh)
    assert back["tip_y"] == hist["tip_y"]
    assert back["t"] == hist["t"]
    assert fresh.time == pytest.approx(0.123)
    assert torch.equal(fresh.beam_nodes, state.beam_nodes)
    assert torch.equal(fresh.beam_vel, state.beam_vel)
    assert torch.equal(fresh.fluid_vel, state.fluid_vel)
    assert prev["solver"] == "iqn-ils"


def test_short_coupled_run_is_finite_and_advances_time():
    """A 12-step run at the coarsest usable dx: finite, on the similarity point."""
    case = drv.build_case(dx=0.05, u_mean=0.2, c0=5.0, ratio=1.0)
    assert similarity_report(case)["passed"] is True
    state = initial_state(case)
    dt = case.dt()
    t_before = state.time
    state, res = run_fsi2_strong(case, 12, interface_solver="iqn-ils",
                                 max_iter=6, tol=1e-3, state=state,
                                 stop_on_abort=True)
    assert res["aborted"] is False
    assert res["n_steps"] == 12
    assert state.time == pytest.approx(t_before + 12 * dt, rel=1e-12)
    tip = np.asarray(res["history"]["tip_y"], dtype=float)
    assert tip.size == 12
    assert np.all(np.isfinite(tip))
    # the beam cannot move more than its own length in 12 steps
    assert np.abs(tip).max() < case.beam_length