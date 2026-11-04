"""Regression gate on the measured effect of the free-surface pressure closure.

``benchmarks/laplace_pressure_floor/report_closure.json`` records what the
closure actually bought (see docs/verification_free_surface_pressure_closure.md).
These assertions pin the committed measurement so that a change in the closure,
the EOS or the read-out cannot silently move the published numbers; they are
not a re-run of the physics (that is 1-2 min per case).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure_floor"
CLOSURE = BENCH / "report_closure.json"
CLAMP = BENCH / "report.json"


def _load(path):
    if not path.exists():
        pytest.skip(f"{path.name} missing: run the benchmark first")
    return json.loads(path.read_text())


def test_the_closure_is_what_was_measured():
    closure = _load(CLOSURE)
    assert closure["closure"] == "shell"
    assert len(closure["runs"]) >= 2
    for run in closure["runs"]:
        assert run["finite"], (run["dx_m"], run["sound_speed"])
        assert run["closure"] == "shell"
        assert run["closure_threshold"] == pytest.approx(0.99)


def test_closure_beats_the_clamp_at_every_recorded_point():
    closure = _load(CLOSURE)
    clamp = _load(CLAMP)
    clamp_runs = {(r["dx_m"], r["sound_speed"]): r for r in clamp["runs"]}
    improved = 0
    for run in closure["runs"]:
        key = (run["dx_m"], run["sound_speed"])
        if key not in clamp_runs:
            continue
        before = clamp_runs[key]["absolute_time_average_tail"]["rel_error_pct"]
        after = run["absolute_time_average_tail"]["rel_error_pct"]
        assert abs(after) < abs(before), (key, before, after)
        improved += 1
    assert improved >= 2


def test_at_least_one_point_meets_the_15pct_bar():
    closure = _load(CLOSURE)
    worst = min(abs(r["absolute_time_average_tail"]["rel_error_pct"])
                for r in closure["runs"])
    assert worst < 15.0, worst


def test_spurious_currents_drop_below_the_adami_limit():
    closure = _load(CLOSURE)
    for run in closure["runs"]:
        assert run["spurious_current"]["vmax_over_vref_final"] < 0.1, (
            run["dx_m"], run["sound_speed"],
            run["spurious_current"]["vmax_over_vref_final"])
