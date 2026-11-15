"""Invariants of the frame-ensemble stationarity audit (no solver run).

These tests pin the *audit arithmetic*, not a physical number: a wrong partition,
a silent NaN, or a re-inverted judgeability rule would each turn a report claim
into a falsehood, and each is expressible as a pure invariant.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BENCH = HERE.parent / "benchmarks" / "case_sph_wedge_entry_2d"
sys.path.insert(0, str(BENCH))

import frame_stationarity as fs  # noqa: E402

ART = HERE.parent / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ensemble"


def synth_frames(n=20, n_wet_start=1, cp0=6.0, cp_slope=0.0):
    frames = []
    for k in range(n):
        n_wet = n_wet_start + k
        frames.append({"t": 1e-3 * k, "n_wet": n_wet,
                       "cp": cp0 + cp_slope * math.log(max(1, n_wet)),
                       "tau_at_peak": 0.8 if k % 2 else 0.1,
                       "cp_centre": cp0, "max_rho_ratio": 1.01})
    return frames


# --------------------------------------------------------------------------- #
# frame table
# --------------------------------------------------------------------------- #
def test_frame_table_drops_frames_without_wetted_nodes():
    art = {"frame_scan": [
        {"t": 0.0, "n_wet": 0, "cp_peak_single_frame": float("nan"), "tau_at_peak": float("nan")},
        {"t": 1e-3, "n_wet": 5, "cp_peak_single_frame": 7.0, "tau_at_peak": 0.5},
        {"t": 2e-3, "n_wet": 6, "cp_peak_single_frame": 6.0, "tau_at_peak": 0.9},
    ]}
    frames = fs.frame_table(art)
    assert len(frames) == 2
    assert all(f["cp"] == f["cp"] for f in frames)
    assert [f["n_wet"] for f in frames] == [5, 6]


def test_argmax_frame_of_committed_artifact_is_nan_and_therefore_unusable():
    """Regression on the defect this module works around: the artefact's own
    ``argmax_frame`` is the frame with no wetted node (nan compares False)."""
    p = ART / "ensemble_dx2mm_c0v20.json"
    if not p.exists():
        pytest.skip("committed ensemble artefact missing")
    art = json.loads(p.read_text())
    am = art["argmax_frame"]
    assert am["t"] == 0.0
    assert not (am["cp_peak_single_frame"] == am["cp_peak_single_frame"])


# --------------------------------------------------------------------------- #
# Q1: extreme-value signature
# --------------------------------------------------------------------------- #
def test_ev_strata_partition_every_frame_exactly_once():
    frames = synth_frames(30, n_wet_start=1)
    rows = fs.ev_strata(frames)
    assert sum(r["n"] for r in rows) == len(frames)


def test_ev_slope_recovers_a_known_power_law():
    """cp proportional to the sample count -> log-log slope exactly 1."""
    frames = synth_frames(60, n_wet_start=2)
    for f in frames:
        f["cp"] = float(f["n_wet"])
    assert fs.ev_slope(fs.ev_strata(frames)) == pytest.approx(1.0, abs=1e-9)


def test_ev_slope_positive_when_peak_grows_with_sample_count():
    frames = synth_frames(60, n_wet_start=2, cp0=4.0, cp_slope=1.5)
    assert fs.ev_slope(fs.ev_strata(frames)) > 0.15


def test_ev_slope_zero_when_peak_is_scale_free():
    frames = synth_frames(60, n_wet_start=2, cp0=4.0, cp_slope=0.0)
    assert abs(fs.ev_slope(fs.ev_strata(frames))) < 1e-9


def test_peak_location_flip_counts_both_regions():
    frames = synth_frames(10)
    loc = fs.peak_location_flip(frames)
    assert loc["n"] == 10
    assert loc["stagnation"] == pytest.approx(0.5)
    assert loc["spray"] == pytest.approx(0.5)


def test_dispersion_of_constant_series_is_zero():
    frames = synth_frames(12, cp_slope=0.0)
    d = fs.dispersion(frames)
    assert d["cov"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Q2: half split
# --------------------------------------------------------------------------- #
def test_half_split_of_constant_series_is_zero():
    frames = synth_frames(20, cp_slope=0.0)
    hs = fs.half_split(frames)
    assert hs["rel_split"] == pytest.approx(0.0, abs=1e-12)
    assert hs["n_early"] + hs["n_late"] == 20


def test_half_split_detects_a_step_between_halves():
    frames = synth_frames(20, cp_slope=0.0)
    for k, f in enumerate(frames[10:]):
        f["cp"] *= 2.0
    hs = fs.half_split(frames)
    assert hs["rel_split"] == pytest.approx(0.5, abs=1e-9)


def test_wet_gate_drops_early_frames_and_is_monotone_in_frame_count():
    frames = synth_frames(30, n_wet_start=1)
    counts = [len(fs.gated(frames, g)) for g in (0, 5, 10, 20)]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == 11          # n_wet >= 20 -> k >= 19


# --------------------------------------------------------------------------- #
# Q3: judgeability ledger
# --------------------------------------------------------------------------- #
def _rec(spread, jackknife, rb_spread, judgeable, peak=6.0):
    return {"spread_rel": spread, "jackknife_rel": jackknife, "judgeable": judgeable,
            "resolution_band": {"peak_median": peak, "spread_rel": rb_spread}}


def test_ledger_calls_out_the_false_positive_that_motivated_the_audit():
    art = {"by_frames": {"61": _rec(0.140, 0.4295, 0.118, judgeable=True)}}
    led = fs.judgeability_ledger(art)
    assert led["artifact_judgeable"] is True
    assert led["declared_uncertainty"] == pytest.approx(0.4295)
    assert led["stationarity_aware_judgeable"] is False
    assert led["contradiction"] is True


def test_ledger_agrees_when_both_rules_agree():
    art = {"by_frames": {"61": _rec(0.05, 0.06, 0.05, judgeable=True)}}
    led = fs.judgeability_ledger(art)
    assert led["stationarity_aware_judgeable"] is True
    assert led["contradiction"] is False


def test_ledger_uses_the_resolution_band_not_the_full_bin_grid():
    """A grid whose full-grid spread is wide but whose band spread is tight is
    judgeable by band *and* jackknife -- the old rule would reject it."""
    art = {"by_frames": {"61": _rec(0.20, 0.03, 0.05, judgeable=False)}}
    led = fs.judgeability_ledger(art)
    assert led["stationarity_aware_judgeable"] is True
    assert led["contradiction"] is True


# --------------------------------------------------------------------------- #
# end to end on the committed artefacts
# --------------------------------------------------------------------------- #
def test_audit_on_committed_artefacts_reports_the_contradictions():
    if not ART.exists():
        pytest.skip("committed ensemble artefacts missing")
    res = fs.audit(ART)
    assert res["gate"] == pytest.approx(0.15)
    assert res["wet_node_gate"] == 16
    got = {t: g.get("ledger", {}).get("contradiction")
           for t, g in res["grids"].items()}
    # dx=2mm: artefact says judgeable, the half-split says 43 % -> contradiction.
    # dx=1mm: artefact says not judgeable, band+jackknife say 6.5 % -> contradiction.
    assert got.get("dx2mm") is True
    assert got.get("dx1mm") is True
    assert res["summary"]["n_judgeable"] == 1
    assert res["summary"]["grids_judgeable"] == ["dx1mm"]