"""Guards for the H2 scoring harness (mission 21586).

The harness decides whether the case ledger criterion (peak height and arrival time at
the Kleefsman H2 probe, 10 % tolerance) is met.  These tests pin the properties that
make its verdict trustworthy:

* arrival time is interpolated, not quantised to the output interval;
* a run whose window stops before the experimental global peak is *not* scored as a
  pass -- it is reported as WINDOW_LIMITED;
* a verdict cannot be rescued by choosing the lucky (operator, station) cell;
* the 2D-vs-3D blockage fact that bounds comparability is computed, not asserted.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "benchmarks" / "kleefsman_obstacle"))

import score_h2 as S  # noqa: E402


def test_arrival_time_is_interpolated_between_samples():
    times = [0.0, 0.5, 1.0]
    values = [0.0, 0.01, 0.03]
    # 0.02 is reached half way between t = 0.5 and t = 1.0
    assert S.arrival_time(times, values, 0.02) == pytest.approx(0.75)


def test_arrival_time_is_nan_when_never_reached():
    assert math.isnan(S.arrival_time([0.0, 1.0], [0.0, 0.005], 0.02))


def test_peak_metrics_reports_global_peak_and_time():
    m = S.peak_metrics([0.0, 1.0, 2.0], [0.1, 0.4, 0.2], 0.02)
    assert m["peak_m"] == pytest.approx(0.4)
    assert m["peak_time_s"] == pytest.approx(1.0)


def test_gauge_column_split():
    assert S.split_gauge("slab_x2.426") == ("slab", 2.426)
    assert S.split_gauge("height_x2.228") == ("height", 2.228)


def test_reference_is_anchored():
    ref = S.load_reference()
    exp = ref["experimental_values"]
    assert exp["status"] == "ANCHORED"
    assert exp["peak_height_m"] == pytest.approx(0.37)
    assert exp["arrival_time_s"] == pytest.approx(0.3591, abs=1e-3)
    # the digitised experimental series must reproduce the anchor's peak
    t, h = S.load_reference_series()
    assert t and len(t) == len(h)
    i = max(range(len(h)), key=lambda k: h[k])
    assert h[i] == pytest.approx(exp["peak_height_m"], abs=0.01)


def test_short_window_run_is_window_limited_not_pass():
    """A t_end < 1.703 s run holds only the *local* first-impact peak."""
    ref = S.load_reference()
    run = S.load_run(HERE / "benchmarks" / "kleefsman_obstacle" / "results"
                     / "gauge_ladder" / "dx0.02" / "time_series.csv")
    res = S.score_run(run, ref)
    assert res["window_covers_experimental_peak"] is False
    assert res["cells"], "expected gauge cells"
    assert S.verdict({"ladder_dx0.02": res}, ref["tolerance"]["relative"])["verdict"] \
        == "WINDOW_LIMITED"


def _cell(peak, arrival, op="slab", xs=2.228):
    return {"operator": op, "station_x_m": xs, "peak_m": peak, "peak_time_s": 1.0,
            "arrival_s": arrival, "peak_err_rel": peak, "arrival_err_rel": arrival,
            "peak_pass": peak < 0.1, "arrival_pass": arrival < 0.1}


def test_verdict_cannot_be_rescued_by_a_lucky_cell():
    """One passing cell among failing ones is still a FAIL."""
    per_run = {"r": {"t_end_s": 2.5, "window_covers_experimental_peak": True,
                     "cells": [_cell(0.02, 0.05), _cell(0.40, 0.30, xs=2.426)]}}
    v = S.verdict(per_run, 0.1)
    assert v["verdict"] == "FAIL"
    assert v["best_peak_err_rel"] == pytest.approx(0.02)
    assert v["worst_peak_err_rel"] == pytest.approx(0.40)


def test_verdict_passes_only_on_a_full_window_with_every_cell_inside():
    per_run = {"r": {"t_end_s": 2.5, "window_covers_experimental_peak": True,
                     "cells": [_cell(0.02, 0.05), _cell(0.03, 0.06, op="surface")]}}
    assert S.verdict(per_run, 0.1)["verdict"] == "PASS"


def test_measured_envelope_misses_the_tolerance_on_the_peak():
    """Measured fact, not a target: no (operator, station) cell meets 10 % on the peak."""
    verdict = json.loads((HERE / "benchmarks" / "kleefsman_obstacle" / "results"
                          / "h2_score" / "verdict.json").read_text())
    for op, d in verdict["envelope"].items():
        assert d["any_peak_pass"] is False, f"{op} unexpectedly meets the peak tolerance"
    blk = verdict["obstacle_blockage"]
    assert blk["blocked_fraction_3d_experiment"] == pytest.approx(0.403, abs=1e-9)
    assert blk["blocked_fraction_2d_model"] == pytest.approx(1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))