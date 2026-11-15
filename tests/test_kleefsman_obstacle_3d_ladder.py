"""Invariant tests for the 3-D Kleefsman grid ladder.

These assert *properties* rather than "the code runs": a synthetic ladder built
from a known ``f(dx) = f_inf + C dx^p`` must be recovered, two grids must never
produce a fabricated order, and a protocol disagreement must poison the ladder
instead of silently averaging incomparable runs.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "kleefsman_obstacle_3d"))

import ladder  # noqa: E402


# --------------------------------------------------------------------------- #
# fit_power_law
# --------------------------------------------------------------------------- #
def test_power_law_recovers_a_known_continuum_and_order():
    # synthetic second-order data on the real non-uniform 3-D ladder
    f_inf, C, p = 0.372, 1.4e-3, 2.0
    dxs = [0.08, 0.04, 0.03]
    vals = [f_inf + C * d ** p for d in dxs]
    fit = ladder.fit_power_law(dxs, vals)
    assert fit["status"] == "OK"
    assert abs(fit["f_inf"] - f_inf) < 1e-6
    assert abs(fit["p_observed"] - p) < 1e-3
    assert abs(fit["C"] - C) / C < 1e-3


def test_power_law_handles_a_first_order_accuracy_data_set():
    f_inf, C, p = 0.20, 5.0e-2, 1.0
    dxs = [0.08, 0.04, 0.02]
    vals = [f_inf + C * d for d in dxs]
    fit = ladder.fit_power_law(dxs, vals)
    assert fit["status"] == "OK"
    assert abs(fit["p_observed"] - 1.0) < 1e-3
    assert abs(fit["f_inf"] - f_inf) < 1e-6


def test_two_grids_never_produce_an_order_of_accuracy():
    fit = ladder.fit_power_law([0.08, 0.04], [0.21, 0.42])
    assert fit["status"] == "UNDERDETERMINED"
    assert "p_observed" not in fit, "a two-grid ladder must not fabricate an order"
    assert fit["gap"] == pytest.approx(0.21)
    assert fit["finest"] == pytest.approx(0.42)


def test_single_grid_is_refused():
    fit = ladder.fit_power_law([0.04], [0.42])
    assert fit["status"] == "INSUFFICIENT_LEVELS"


# --------------------------------------------------------------------------- #
# protocol guard
# --------------------------------------------------------------------------- #
def _entry(tag, dx, t_end=2.6, c0=20.0):
    return {"tag": tag, "dx": dx,
            "fingerprint": {"t_end": t_end, "sound_speed": c0, "wall_layers": 3,
                            "obstacle_z": 0.0, "station_offset": 0.05,
                            "gauge_aperture": 0.08, "arrival_threshold": 0.02}}


def test_identical_protocols_are_comparable():
    ok, bad = ladder.check_protocol([_entry("a", 0.08), _entry("b", 0.04)])
    assert ok is True and bad == {}


def test_a_shorter_horizon_poisons_the_ladder():
    ok, bad = ladder.check_protocol([_entry("a", 0.08), _entry("b", 0.04, t_end=1.2)])
    assert ok is False
    assert "t_end" in bad
    assert bad["t_end"]["b"] == pytest.approx(1.2)


def test_a_different_sound_speed_poisons_the_ladder():
    ok, bad = ladder.check_protocol([_entry("a", 0.08), _entry("b", 0.04, c0=80.0)])
    assert ok is False and "sound_speed" in bad


# --------------------------------------------------------------------------- #
# hypothesis adjudication
# --------------------------------------------------------------------------- #
def test_monotone_closing_to_the_reference_supports_the_hypothesis():
    fit = ladder.fit_power_law([0.08, 0.04, 0.02], [0.20, 0.30, 0.35])
    res = ladder.converges_toward(fit, 0.37)
    assert res["verdict"] == "BLOCKAGE_HYPOTHESIS_SUPPORTED"
    assert res["monotone_closing"] is True


def test_a_ladder_moving_away_refutes_the_hypothesis():
    fit = ladder.fit_power_law([0.08, 0.04, 0.02], [0.42, 0.45, 0.47])
    res = ladder.converges_toward(fit, 0.37)
    assert res["verdict"] == "BLOCKAGE_HYPOTHESIS_NOT_SUPPORTED"


def test_a_two_grid_ladder_is_not_decidable():
    fit = ladder.fit_power_law([0.08, 0.04], [0.21, 0.42])
    res = ladder.converges_toward(fit, 0.37)
    assert res["status"] == "NOT_DECIDABLE"


# --------------------------------------------------------------------------- #
# I/O round trip
# --------------------------------------------------------------------------- #
def _write_run(d: Path, tag: str, dx: float, peak: float, arrival: float,
               t_end: float = 2.6, bypass: int = 7) -> None:
    meta = {
        "verdict": "FAIL", "censored": False, "window_covers_reference_peak": True,
        "rows": [{"metric": "peak_height", "ref": 0.37, "got": peak,
                  "rel_error": abs(peak - 0.37) / 0.37, "pass": False}],
        "run": {"dx": dx, "peak_height_m": peak, "peak_time_s": 1.5,
                "arrival_time_s": arrival, "wall_clock_s": 100.0,
                "t_end": t_end, "sound_speed": 20.0, "wall_layers": 3,
                "obstacle_z": 0.0, "station_offset": 0.05,
                "gauge_aperture": 0.08, "arrival_threshold": 0.02},
    }
    (d / f"{tag}.json").write_text(json.dumps(meta))
    with (d / f"{tag}_time_series.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "slab_height_m", "height_operator_m", "front_x_m", "n_bypass"])
        for i in range(11):
            t = i * 0.26
            w.writerow([f"{t:.6f}", f"{peak * (1 - abs(i - 5) / 10):.6f}",
                        f"{peak * (1 - abs(i - 5) / 10) + 0.01:.6f}",
                        f"{2.0 + i * 0.02:.6f}", bypass])


def test_analyse_builds_a_three_grid_ladder_and_keeps_bypass(tmp_path):
    for tag, dx, peak in (("dx0.08", 0.08, 0.20), ("dx0.04", 0.04, 0.30),
                          ("dx0.03", 0.03, 0.35)):
        _write_run(tmp_path, tag, dx, peak, 0.50)
    rep = ladder.analyse(tmp_path, ["dx0.08", "dx0.04", "dx0.03"])
    assert rep["status"] == "OK"
    assert rep["protocol_comparable"] is True
    assert len(rep["runs"]) == 3
    assert rep["runs"][0]["max_bypass"] == 7
    assert rep["peak_ladder"]["status"] == "OK"
    assert rep["peak_vs_reference"]["monotone_closing"] is True


def test_analyse_refuses_an_incomparable_ladder(tmp_path):
    _write_run(tmp_path, "dx0.08", 0.08, 0.20, 0.5, t_end=2.6)
    _write_run(tmp_path, "dx0.04", 0.04, 0.30, 0.5, t_end=1.2)
    rep = ladder.analyse(tmp_path, ["dx0.08", "dx0.04"])
    assert rep["status"] == "INCOMPARABLE"
    assert "peak_ladder" not in rep, "an incomparable ladder must not be extrapolated"


def test_missing_runs_report_no_runs(tmp_path):
    rep = ladder.analyse(tmp_path, ["nope"])
    assert rep["status"] == "NO_RUNS"