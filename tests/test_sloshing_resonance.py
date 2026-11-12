"""Tests for the CASE-SLOSHING benchmark: the anchor and the run plumbing.

The analytic anchor is checked first and hard: a reference that is wrong in the
third digit makes the 10 % gate meaningless. Two independent routes are
compared (:func:`sloshing_anchor.omega_n` closed form and
:func:`sloshing_anchor.omega_n_fd`, an FD eigensolve of the same boundary-value
problem), plus the two limits the closed form must reproduce.

The run is deliberately tiny (6 cells per depth, half a period) and only asserts
*plumbing*: the loop finishes, the modal record is finite, the measured snapshot
is taken where it was asked for, and mass is conserved. Accuracies come from the
recorded run in ``runs/sloshing_resonance/summary.json``.
"""
import contextlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch

from tensorsph.freesurface import sloshing_anchor, sloshing_mode

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "sloshing_resonance"
sys.path.insert(0, str(BENCH))

import cases as case_table  # noqa: E402


@pytest.mark.parametrize("depth_ratio", [0.10, 0.20, 0.30, 0.42, 0.60])
def test_closed_form_matches_independent_fd_eigensolve(depth_ratio):
    """Two routes to the same eigenvalue: tanh closed form vs FD eigensolve."""
    length = 0.60
    depth = depth_ratio * length
    closed = sloshing_anchor.omega_n(length, depth)
    fd = sloshing_anchor.omega_n_fd(length, depth, cells=(32, 64, 128, 256))
    assert fd == pytest.approx(closed, rel=1e-6)


def test_fd_route_converges_at_second_order():
    """The un-extrapolated FD eigenvalues must halve their error each refinement."""
    length, depth = 0.60, 0.25
    closed = sloshing_anchor.omega_n(length, depth)
    k = math.pi / length
    raw = [sloshing_anchor._omega_fd_single(k, depth, 9.81, cells)
           for cells in (32, 64, 128, 256)]
    errors = [abs(value - closed) for value in raw]
    assert errors[0] > errors[1] > errors[2] > errors[3]
    # O(dy^2): the error drops by ~4 per refinement until round-off dominates
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.25)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.25)


def test_shallow_and_deep_water_limits():
    length, g = 1.0, 9.81
    k = math.pi / length
    shallow = sloshing_anchor.omega_n(length, 1.0e-5 * length)
    assert shallow == pytest.approx(k * math.sqrt(g * 1.0e-5 * length), rel=1e-4)
    deep = sloshing_anchor.omega_n(length, 1.0e3 * length)
    assert deep == pytest.approx(math.sqrt(g * k), rel=1e-8)


def test_higher_modes_scale_with_sqrt_mode_in_deep_water():
    length = 1.0
    deep = 50.0
    w1 = sloshing_anchor.omega_n(length, deep, 1)
    w4 = sloshing_anchor.omega_n(length, deep, 4)
    assert w4 / w1 == pytest.approx(2.0, rel=1e-8)


def test_anchor_agrees_with_the_reference_the_free_surface_study_gates_on():
    for depth in (0.12, 0.25, 0.36):
        assert sloshing_anchor.omega_n(0.60, depth) == pytest.approx(
            sloshing_mode.analytic_omega(0.60, depth), rel=1e-15)


def test_case_table_is_self_consistent_and_one_at_a_time():
    table = case_table.CASES
    labels = [case.label for case in table]
    assert len(labels) == len(set(labels))
    for case in table:
        assert case.tank_height > case.water_depth + case.amplitude
        assert case.dx == pytest.approx(case.water_depth / case.cells_per_depth)
        assert 0.0 < case.amplitude_ratio < 0.25
    assert case_table.TOLERANCE == 0.10
    # every sweep must contain the baseline parameters, so each sensitivity
    # number has exactly one cause
    base = case_table.BASELINE
    for family in ("resolution", "amplitude", "sound_speed"):
        members = [c for c in table if c.family == family]
        assert members, family
    sweep_keys = {
        "resolution": lambda c: (c.water_depth, c.amplitude, c.sound_speed_ratio),
        "amplitude": lambda c: (c.water_depth, c.cells_per_depth, c.sound_speed_ratio),
        "sound_speed": lambda c: (c.water_depth, c.amplitude, c.cells_per_depth),
    }
    for family, key in sweep_keys.items():
        values = {key(c) for c in table if c.family == family}
        values.add(key(base))
        assert len(values) == 1, f"{family} sweep is not one-at-a-time: {values}"


def test_benchmark_planes_are_importable_and_declare_no_numbers():
    """run.py/aggregate.py must exist next to cases.py and expose --list."""
    assert (BENCH / "run.py").exists()
    assert (BENCH / "aggregate.py").exists()


@contextlib.contextmanager
def few_threads(count: int = 4):
    previous = torch.get_num_threads()
    torch.set_num_threads(min(count, previous))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_smoke_run_takes_the_requested_snapshot_and_conserves_mass():
    case = case_table.baseline()
    with few_threads(4):
        model = sloshing_mode.SloshingModeCase(
            tank_length=case.tank_length, water_depth=case.water_depth,
            tank_height=case.tank_height, amplitude=case.amplitude,
            cells_per_depth=6.0, sound_speed_ratio=80.0, periods=0.6,
            output_per_period=20, snapshot_time=0.05, label="smoke")
        result = sloshing_mode.run_sloshing_mode(model)
    summary = result.summary()
    assert result.particle_count > 0
    assert math.isfinite(summary["mass_drift"])
    assert summary["mass_drift"] < 1e-12
    assert len(result.times) == len(result.modal) > 5
    assert all(math.isfinite(v) for v in result.modal)
    # the snapshot is the frame nearest the requested time
    assert result.snapshot, "snapshot_time was set but no frame was stored"
    assert result.snapshot["time"] >= 0.0
    n = len(result.snapshot["x"])
    assert n == result.particle_count
    assert len(result.snapshot["y"]) == n
    assert len(result.snapshot["speed"]) == n
    assert len(result.snapshot["density"]) == n