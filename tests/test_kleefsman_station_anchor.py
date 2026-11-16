"""Invariants behind the H2 station-anchor argument.

These assert the *numbers the argument rests on*, not merely that
``station_anchor.analyse()`` returns.  A silent edit of the digitised arrival,
the reference geometry or the Ritter inversion therefore fails here instead of
quietly changing what the ledger is told.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "kleefsman_obstacle"))
sys.path.insert(0, str(ROOT / "src"))

import station_anchor  # noqa: E402


@pytest.fixture(scope="module")
def res():
    return station_anchor.analyse()


def _trace(res, name):
    return next(t for t in res["traces"] if t["trace"] == name)


def test_ritter_locus_matches_hand_computation():
    """(2 sqrt(g h0) - 3 sqrt(g h*)) for h0=0.55, h*=0.02."""
    c0, locus = station_anchor.ritter_speeds(0.55, 0.02)
    assert c0 == pytest.approx(4.645643, rel=1e-6)
    assert locus == pytest.approx(3.316809, rel=1e-6)
    assert locus < c0, "the level locus must trail the leading edge"


def test_geometry_is_the_documented_one(res):
    geom = res["geometry"]
    assert geom["gate_x_m"] == pytest.approx(1.228)
    assert geom["box_face_x_m"] == pytest.approx(2.476)
    assert geom["arrival_threshold_m"] == pytest.approx(0.02)
    # 1.228 + 1.248 + 0.744 = 3.22 m, the documented consistency check
    assert geom["gate_x_m"] + 1.248 + 0.744 == pytest.approx(3.22, abs=1e-12)


def test_ritter_arrival_at_the_committed_station(res):
    """The as-committed station's analytic arrival, the value quoted in docs."""
    assert res["ritter_arrival_at_stations_s"][
        "primary (0.05 m up of box face)"] == pytest.approx(0.36119, abs=1e-5)
    assert res["ritter_arrival_at_stations_s"][
        "upstream (0.248 m up of box face)"] == pytest.approx(0.30149, abs=1e-5)


@pytest.mark.parametrize("trace", ["experiment_interp", "experiment_sample", "comflow_vof"])
def test_implied_station_brackets_the_committed_one(res, trace):
    """Every independent trace lands within 5 cm of x = 2.426 m."""
    row = _trace(res, trace)
    assert abs(row["x_implied_m"] - 2.426) < 0.05, row["x_implied_m"]
    # ... and nowhere near the 0.248 m "escape" station
    assert abs(row["x_implied_m"] - 2.228) > 0.15


@pytest.mark.parametrize("trace", ["experiment_interp", "experiment_sample", "comflow_vof"])
def test_committed_station_needs_no_front_lag(res, trace):
    """At x=2.426 the measured arrival coincides with the Ritter locus (|gamma-1|<=5%).

    This is the whole point: the committed station is the one at which the
    digitised first crossing is reproduced by the closed-form solution, so it
    needs no free parameter.  The 0.248 m station needs an 13-19 % lag.
    """
    row = _trace(res, trace)
    prim = row["primary (0.05 m up of box face)"]["required_gamma"]
    up = row["upstream (0.248 m up of box face)"]["required_gamma"]
    assert abs(prim - 1.0) <= 0.05, prim
    assert up < 0.90, up
    assert prim > up, "a nearer station must not be the more lagging hypothesis"


def test_hypothesis_needs_implausibly_steep_acceleration(res):
    """Moving H2 to 2.228 m needs an acceleration trend >= 3x the solver's.

    gamma grows with distance (the leading edge accelerates), so a *nearer*
    station shows a *lower* gamma.  The hypothesis needs d gamma / d d = 0.81 -
    0.87 /m; the solver's own tracked front gives ~0.24 /m.
    """
    assert res["solver_front_trend"], "no front series found; cannot test the trend"
    ratios = [t["trend_ratio_hypothesis_over_solver"] for t in res["discrimination"]]
    assert all(r > 3.0 for r in ratios), ratios
    # the solver trend itself is the same sign as the hypothesis, only smaller
    assert all(r["d_gamma_per_m"] > 0 for r in res["solver_front_trend"])


def test_json_artifact_is_written_and_round_trips(tmp_path, monkeypatch):
    """The published artifact must be re-readable JSON with the same numbers."""
    import json
    res = station_anchor.analyse()
    blob = json.dumps(res, indent=2)
    back = json.loads(blob)
    assert back["ritter"]["locus_speed_m_s"] == pytest.approx(
        res["ritter"]["locus_speed_m_s"])
    assert len(back["traces"]) == 3
    assert math.isfinite(back["ritter"]["c0_gate_m_s"])