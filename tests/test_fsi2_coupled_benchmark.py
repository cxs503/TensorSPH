"""Coupled FSI2 benchmark tests: infrastructure and short-run validation.

Anchor: ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

These tests verify that the coupled benchmark infrastructure works correctly
and that the strong-coupling driver produces a physically plausible response
at the benchmark density ratio (``rho_s/rho_f = 1``).  The full long-time
integration (t_end = 10 s, dx = 0.005 m) is a cluster job, not a unit test.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.fsi2_reference import (  # noqa: E402
    BENCHMARK_TIP, compare_to_benchmark, LENGTH_SCALE,
    load_tip_series, summarise_tip,
)
from tensorsph.fsi2_similarity import similarity_corrected_case  # noqa: E402
from tensorsph.fsi2_turek_hron import initial_state, step_fsi2  # noqa: E402
from tensorsph.fsi2_strong_coupling import (  # noqa: E402
    run_fsi2_strong, step_fsi2_strong,
)


def _case(dx: float = 0.02, ratio: float = 1.0, c0: float = 5.0,
          u_mean: float = 0.2) -> "FSI2Case":
    """Similarity-corrected FSI2 case."""
    case = similarity_corrected_case(U_mean=u_mean)
    case.dx = dx
    case.sound_speed = c0
    case.beam_rho = 1000.0 * ratio
    case.n_beam_nodes = max(15, int(case.beam_length / dx))
    case.max_rod_substeps = 2000
    case.rod_stability_target = 0.2
    return case


# --------------------------------------------------------------------------- #
# 1. Reference archive integrity
# --------------------------------------------------------------------------- #

def test_oss_reference_archive_is_readable():
    """The archived SPHinXsys records must be parseable."""
    for run in ("0", "4", "9"):
        n, pos = load_tip_series(run)
        assert n == 107
        assert pos.shape == (107, 2)
        s = summarise_tip(pos)
        assert s["tip_y_amplitude"] > 0.0
        assert s["tip_x_amplitude"] > 0.0


def test_oss_amplitudes_are_consistent_across_runs():
    """The three upstream runs agree to <10 % on the tip amplitude."""
    amps = [summarise_tip(load_tip_series(r)[1])["tip_y_amplitude"]
            for r in ("0", "4", "9")]
    assert (max(amps) - min(amps)) / np.mean(amps) < 0.10, amps


# --------------------------------------------------------------------------- #
# 2. Coupled benchmark infrastructure
# --------------------------------------------------------------------------- #

def test_strong_coupling_produces_finite_state_at_ratio_one():
    """The strong-coupling driver must stay finite at rho_s/rho_f = 1."""
    case = _case(dx=0.02, ratio=1.0)
    state = initial_state(case)
    dt = case.dt()

    for k in range(20):
        row, stats = step_fsi2_strong(state, case, dt, interface_solver="iqn-ils",
                                      max_iter=10, tol=1e-2)
        assert math.isfinite(row["tip_x"])
        assert math.isfinite(row["tip_y"])
        assert math.isfinite(row["drag"])
        assert math.isfinite(row["lift"])
        assert math.isfinite(row["max_speed"])
        if stats.aborted:
            pytest.fail(f"aborted at step {k}")


def test_strong_coupling_beam_responds_to_flow():
    """The beam tip must move once the flow develops (non-zero displacement)."""
    case = _case(dx=0.02, ratio=10.0)  # ratio 10 for faster response
    state = initial_state(case)
    dt = case.dt()

    tip_y_initial = 0.0
    for k in range(50):
        row, stats = step_fsi2_strong(state, case, dt, interface_solver="iqn-ils",
                                      max_iter=10, tol=1e-2)
        if k == 49:
            tip_y_final = row["tip_y"]
    # After 50 steps the beam should have moved from its initial position
    assert abs(tip_y_final) > 1.0e-12, f"tip_y={tip_y_final} after 50 steps"


def test_loose_coupling_diverges_at_ratio_one_regression():
    """Regression: the loose driver must stay bounded at ratio 1 with the
    bending-aware rod sub-cycling (the fix from the strong-coupling round)."""
    case = _case(dx=0.02, ratio=1.0)
    case.wall_impedance = False
    state = initial_state(case)
    dt = case.dt()

    for k in range(40):
        row = step_fsi2(state, case, dt)
        v = float(state.beam_vel.norm(dim=1).max())
        assert math.isfinite(v)
        assert math.isfinite(row["max_speed"])
        # The beam speed must stay below the drag-driven scale
        assert v < 0.3, f"step {k}: v_beam={v} (expected < 0.3 m/s)"


# --------------------------------------------------------------------------- #
# 3. Acceptance gate mechanics
# --------------------------------------------------------------------------- #

def test_benchmark_gate_passes_for_exact_match():
    """A perfect match must pass both gates."""
    summary = {"tip_x_amplitude": BENCHMARK_TIP["tip_x_amplitude"],
               "tip_y_amplitude": BENCHMARK_TIP["tip_y_amplitude"]}
    gate = compare_to_benchmark(summary)
    assert gate["passed_15"]
    assert gate["passed_3"]
    assert gate["worst_rel_error"] == pytest.approx(0.0)


def test_benchmark_gate_fails_for_large_error():
    """A 50 % error must fail the 15 % gate."""
    summary = {"tip_x_amplitude": BENCHMARK_TIP["tip_x_amplitude"] * 1.5,
               "tip_y_amplitude": BENCHMARK_TIP["tip_y_amplitude"] * 1.5}
    gate = compare_to_benchmark(summary)
    assert not gate["passed_15"]
    assert gate["worst_rel_error"] == pytest.approx(0.5, rel=1e-9)


def test_oss_record_fails_the_benchmark_gate():
    """The archived SPHinXsys record is ~9x the benchmark amplitude."""
    _, pos = load_tip_series("0")
    s = summarise_tip(pos)
    gate = compare_to_benchmark(s)
    assert not gate["passed_15"]
    assert gate["worst_rel_error"] > 1.0  # >100 % error


# --------------------------------------------------------------------------- #
# 4. Short coupled benchmark run (probe)
# --------------------------------------------------------------------------- #

def test_coupled_benchmark_probe_runs_to_completion():
    """A short coupled benchmark run must complete and produce a result file."""
    import subprocess
    out = ROOT / "artifacts" / "xval_fsi2" / "coupled_benchmark_probe.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "xval_fsi2"
                             / "run_coupled_benchmark.py"),
         "--dx", "0.02", "--n-steps", "40",
         "--variants", "loose,strong-iqn-ils",
         "--out", str(out)],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists()
    data = json.loads(out.read_text())
    assert "rows" in data
    assert len(data["rows"]) == 2
    for row in data["rows"]:
        assert row["finite"]
        assert "tip_amplitudes" in row
        assert "gate" in row
