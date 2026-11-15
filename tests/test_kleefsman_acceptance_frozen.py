"""Pin the frozen acceptance protocol of ``CASE-SPH-KLEEFSMAN-DAMBREAK-OBST``.

The case rule (relative error < 10 % on the water-level peak and on the arrival
time in front of the box) is meaningless unless the *measurement* is fixed.
``reference.json`` froze the protocol in m20773 (slab gauge, 0.08 m aperture,
station 0.05 m upstream of the box, 30 ms symmetric peak filter, 0.02 m arrival
threshold) but until m21324 nothing evaluated it: the solver's scorer took a
plain ``max`` of the raw series, ``run.py`` defaulted to a 0.03 m aperture, and
``reference.json`` did not even carry the reference *peak time*.

These tests pin the evaluation path on synthetic inputs (so a regression cannot
hide behind plausible-looking committed numbers) and pin the two data-plane
defects that m21324 repaired.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "kleefsman_obstacle"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT / "src"))

import acceptance_frozen as af  # noqa: E402

REF = json.loads((BENCH / "reference.json").read_text())
DT = 4.0e-3          # the committed output interval (output_every * dt)


def synthetic(t_end=2.6, dt=DT, primary=0.517, primary_t=0.716,
              secondary=0.40, secondary_t=1.703, onset=0.357):
    """A two-hump H2 history shaped like the committed trajectory.

    Flat (dry) before ``onset``, a primary impact hump at ``primary_t`` and a
    secondary one at ``secondary_t`` -- exactly the structure that makes the
    whole-record maximum and the event-matched maximum *different events*.
    """
    times, values = [], []
    n = int(round(t_end / dt))
    for k in range(n + 1):
        t = k * dt
        if t < onset:
            v = 0.0
        else:
            v = (primary * math.exp(-((t - primary_t) / 0.16) ** 2)
                 + secondary * math.exp(-((t - secondary_t) / 0.20) ** 2))
            v = max(v, 0.05)                       # wet sheet after the front
        times.append(t)
        values.append(v)
    return times, values


# --------------------------------------------------------------------------- #
# series operators
# --------------------------------------------------------------------------- #
def test_moving_average_keeps_length_and_is_symmetric():
    times = [k * 0.01 for k in range(11)]
    values = [0.0] * 10 + [1.0]
    out = af.moving_average(times, values, 0.03)
    assert len(out) == len(values)
    # the single spike is spread over +/- half a window, not shifted
    assert out[10] > out[7] and out[10] > out[9]
    assert out[0] == pytest.approx(0.0)


def test_moving_average_reproduces_a_constant_and_a_linear_ramp():
    times = [k * 0.01 for k in range(21)]
    flat = af.moving_average(times, [0.25] * 21, 0.03)
    assert all(v == pytest.approx(0.25) for v in flat)
    ramp = af.moving_average(times, [3.0 * t for t in times], 0.03)
    # interior samples: the mean of a symmetric linear window is the centre value
    for k in range(3, 18):
        assert ramp[k] == pytest.approx(3.0 * times[k], abs=1e-12)


def test_moving_average_uses_the_window_in_seconds_not_a_sample_count():
    """A non-uniform time base must still average over the same 30 ms."""
    times = [0.0, 0.001, 0.002, 0.5, 0.501, 0.502]
    values = [1.0, 1.0, 1.0, 9.0, 9.0, 9.0]
    out = af.moving_average(times, values, 0.03)
    assert out[0] == pytest.approx(1.0)      # never sees the far cluster
    assert out[5] == pytest.approx(9.0)


def test_peak_filter_suppresses_a_one_sample_spray_spike():
    """The 30 ms filter exists because a single ballistic particle sets the peak."""
    times, values = synthetic(primary=0.50)
    spiked = list(values)
    k = min(range(len(times)), key=lambda i: abs(times[i] - 0.716))
    spiked[k] = 0.90
    raw_peak, _ = af.series_peak(times, spiked)
    smooth_peak, _ = af.series_peak(times, af.moving_average(times, spiked, 0.03))
    assert raw_peak == pytest.approx(0.90)
    assert smooth_peak < 0.62


def test_first_crossing_is_interpolated_between_samples():
    times = [0.0, 0.1, 0.2]
    values = [0.0, 0.01, 0.05]
    # threshold 0.02 lies halfway between 0.01 and 0.05 -> t = 0.1 + 0.25*0.1
    assert af.first_crossing(times, values, 0.02) == pytest.approx(0.125)


def test_first_crossing_is_nan_when_the_threshold_is_never_reached():
    assert math.isnan(af.first_crossing([0.0, 0.1], [0.0, 0.01], 0.02))


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def test_two_hump_record_makes_the_record_peak_and_the_event_peak_different():
    times, values = synthetic(primary=0.517, primary_t=0.716,
                              secondary=0.40, secondary_t=1.703)
    res = af.score(times, values, REF, t_end=times[-1])
    assert res["peak_time_s"] == pytest.approx(0.716, abs=0.02)
    assert res["event_peak_time_s"] == pytest.approx(1.703, abs=0.05)
    assert res["peak_event_mismatch"] is True
    # the record maximum is the *primary* impact, 40 % above the reference peak
    assert res["peak_row"]["rel_error"] > 0.10
    assert res["verdict"] == "FAIL"


def test_a_record_matching_the_reference_passes_and_a_10pct_bias_does_not():
    times, values = synthetic(primary=0.20, primary_t=0.716,
                              secondary=0.37, secondary_t=1.703)
    ok = af.score(times, values, REF, t_end=times[-1])
    assert ok["arrival_row"]["rel_error"] < 0.10
    assert ok["peak_row"]["rel_error"] < 0.10
    assert ok["verdict"] == "PASS"

    biased = synthetic(primary=0.20, primary_t=0.716,
                       secondary=0.41, secondary_t=1.703)   # +10.8 %
    bad = af.score(biased[0], biased[1], REF, t_end=times[-1])
    assert bad["peak_row"]["rel_error"] > 0.10
    assert bad["verdict"] == "FAIL"


def test_a_peak_at_the_end_of_the_record_is_reported_not_comparable():
    """The exact defect that made the peak half of the case rule unevaluable."""
    times, values = synthetic(t_end=1.2, primary=0.517, primary_t=1.196,
                              secondary=0.0, secondary_t=1.703)
    res = af.score(times, values, REF, t_end=1.2)
    assert res["censoring"]["censored"] is True
    assert res["verdict"] == "NOT_COMPARABLE"
    assert "lower bound" in res["censoring"]["reason"]


def test_extending_the_horizon_clears_the_censoring_flag():
    times, values = synthetic(t_end=2.6)
    res = af.score(times, values, REF, t_end=2.6)
    assert res["censoring"]["censored"] is False
    assert res["verdict"] in {"PASS", "FAIL"}


def test_the_protocol_reported_is_the_one_frozen_in_reference_json():
    frozen = REF["metric_definition"]["frozen_primary"]
    times, values = synthetic()
    res = af.score(times, values, REF, t_end=times[-1])
    assert res["protocol"]["aperture_m"] == frozen["aperture_m"]
    assert res["protocol"]["station_offset_m"] == frozen["station_offset_m"]
    assert res["protocol"]["peak_window_s"] == frozen["peak_filter_window_s"]
    assert res["protocol"]["arrival_threshold_m"] == frozen["arrival_threshold_m"]
    assert af.FROZEN["column"].endswith(f"x{REF['geometry_m']['obstacle_upstream_face_x'] - frozen['station_offset_m']:.3f}")


# --------------------------------------------------------------------------- #
# data-plane defects repaired in m21324
# --------------------------------------------------------------------------- #
def test_reference_json_carries_the_peak_time_the_protocol_needs():
    exp = REF["experimental_values"]
    assert "peak_time_s" in exp, "the 30 ms event window is anchored on this field"
    assert exp["peak_time_s"] == pytest.approx(1.703, abs=0.005)


def test_digitised_series_path_resolves_and_reproduces_the_stored_reference():
    exp = REF["experimental_values"]
    path = ROOT / exp["digitised_series"]
    assert path.exists(), f"digitised_series does not resolve: {exp['digitised_series']}"
    times, values = af.load_series(path, column="h_m")
    peak, peak_t = af.series_peak(times, values)
    assert peak == pytest.approx(exp["peak_height_m"], abs=1e-6)
    assert peak_t == pytest.approx(exp["peak_time_s"], abs=1e-3)
    # the stored 0.3591 s is the first *sample* above 0.02 m; the interpolated
    # crossing of the same curve -- the operator the solver reports -- is 0.3481 s
    assert af.first_crossing(times, values, 0.02) == pytest.approx(
        exp["arrival_time_s_interpolated"], abs=1e-3)
    first_above = next(t for t, h in zip(times, values) if h >= 0.02)
    assert first_above == pytest.approx(exp["arrival_time_s"], abs=1e-3)


def test_the_two_readings_of_the_reference_arrival_are_scored_separately():
    """3.1 % of the 10 % tolerance hangs on which arrival operator is used."""
    exp = REF["experimental_values"]
    band = abs(exp["arrival_time_s_interpolated"] - exp["arrival_time_s"]) \
        / exp["arrival_time_s"]
    assert band == pytest.approx(0.031, abs=0.003)
    assert "arrival_time_definition" in exp

    times, values = synthetic(t_end=2.6)
    res = af.score(times, values, REF, t_end=times[-1])
    metrics = {row["metric"] for row in res["rows"]}
    assert {"arrival_time", "arrival_time_first_above_sample"} <= metrics
    assert res["reference"]["arrival_definition_band_rel"] == pytest.approx(band, abs=1e-6)
    # a PASS must hold against both readings, so the verdict is the stricter one
    assert res["arrival_row"]["ref"] == pytest.approx(exp["arrival_time_s_interpolated"])