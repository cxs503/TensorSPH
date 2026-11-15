"""Tests for the window-convergence audit of the wedge-entry peak Cp (m22436)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "case_sph_wedge_entry_2d"))

import window_convergence as wc  # noqa: E402

ART = ROOT / "docs/artifacts/case_sph_wedge_entry_2d_ensemble"
GATE = 0.15


def _row(n, peak, spread=0.05, jack=0.02, flag=None, full=None):
    return {"n_frames": n, "peak": peak, "bin_spread": spread, "jackknife": jack,
            "spread_full_grid": full if full is not None else spread,
            "judgeable_committed_flag": bool(flag) if flag is not None else None,
            "pooled_samples": 100 * n, "mean_wet_nodes": 10.0}


def _artifact(rows, dx=1e-3):
    by_frames = {}
    for r in rows:
        by_frames[str(r["n_frames"])] = {
            "resolution_band": {"peak_median": r["peak"], "spread_rel": r["bin_spread"]},
            "spread_rel": r["spread_full_grid"],
            "jackknife_rel": r["jackknife"],
            "judgeable": r["judgeable_committed_flag"],
            "pooled_samples": r["pooled_samples"],
            "mean_wet_nodes": r["mean_wet_nodes"],
        }
    return {"config": {"dx": dx}, "by_frames": by_frames}


# --------------------------------------------------------------------------- #
# the three components
# --------------------------------------------------------------------------- #

def test_window_sensitivity_is_relative_change_between_levels():
    rows = [_row(7, 8.0), _row(13, 8.8), _row(25, 8.8)]
    ws = wc.window_sensitivity(rows)
    assert math.isnan(ws[0]), "the first level has no predecessor"
    assert ws[1] == pytest.approx(0.1, rel=1e-12)
    assert ws[2] == pytest.approx(0.0, abs=1e-12)


def test_window_sensitivity_is_undefined_rather_than_zero_on_bad_input():
    rows = [_row(7, 0.0), _row(13, 5.0), _row(25, float("nan"))]
    ws = wc.window_sensitivity(rows)
    assert math.isnan(ws[1]), "division by a zero predecessor must not be reported as 0"
    assert math.isnan(ws[2]), "a NaN level must not produce a finite sensitivity"


def test_unified_uncertainty_takes_the_worst_component():
    row = _row(61, 6.0, spread=0.05, jack=0.02)
    assert wc.unified_uncertainty(row, 0.22) == pytest.approx(0.22)
    assert wc.unified_uncertainty(row, 0.01) == pytest.approx(0.05)


def test_monotone_detection():
    assert wc._monotone([1.0, 2.0, 3.0]) == 1
    assert wc._monotone([3.0, 2.0, 1.0]) == -1
    assert wc._monotone([1.0, 3.0, 2.0]) == 0
    assert wc._monotone([float("nan"), 1.0]) == 0


# --------------------------------------------------------------------------- #
# the two committed judgeability definitions must be distinguishable
# --------------------------------------------------------------------------- #

def test_committed_flag_and_verdict_rule_are_not_the_same_function():
    """A small full-grid spread with a large jackknife is flagged judgeable by
    ``ensemble_peak.py`` but refused by the verdict rule -- that is the defect."""
    art = _artifact([_row(61, 6.0, spread=0.05, jack=0.40, flag=True, full=0.05)])
    g = wc.audit_grid("dx1mm", art, 6.92)
    row = g["rows"][0]
    assert row["judgeable_committed_flag"] is True
    assert row["judgeable_committed_rule"] is False
    assert g["flag_disagreements"] == [61]


def test_gate_is_not_widened_and_window_sensitivity_can_block_a_grid():
    art = _artifact([_row(7, 7.0), _row(13, 9.0, flag=True, full=0.01, spread=0.01, jack=0.01)])
    g = wc.audit_grid("dx2mm", art, 6.92)
    finest = g["finest"]
    assert finest["window_sensitivity"] == pytest.approx(2.0 / 7.0)
    assert finest["uncertainty_unified"] > GATE
    assert finest["judgeable_unified"] is False


def test_reference_is_the_designated_composite_solution():
    assert wc.lr.designated_cp_max(wc.BETA_DEG) == pytest.approx(6.92, abs=1e-9)


# --------------------------------------------------------------------------- #
# regression against the committed artefacts
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not ART.is_dir(), reason="committed ensemble artefacts absent")
def test_audit_reproduces_the_committed_numbers():
    rep = wc.audit(ART)
    tags = [g["tag"] for g in rep["grids"]]
    assert tags == ["dx4mm", "dx2mm", "dx1mm", "dx0.5mm"]
    finest = {g["tag"]: g["finest"] for g in rep["grids"]}
    # the four committed readings of ladder_closure.py / m21468
    assert finest["dx4mm"]["peak"] == pytest.approx(10.041, abs=5e-3)
    assert finest["dx2mm"]["peak"] == pytest.approx(6.071, abs=5e-3)
    assert finest["dx1mm"]["peak"] == pytest.approx(6.805, abs=5e-3)
    assert finest["dx0.5mm"]["peak"] == pytest.approx(8.984, abs=5e-3)
    # the gate is untouched and only one grid survives the unified rule
    assert rep["design"]["gate"] == GATE
    assert rep["design"]["gate_widened"] is False
    assert rep["grids_judgeable_unified"] == ["dx1mm"]
    assert rep["verdict"]["verdict"] == "NOT_DECIDABLE"


@pytest.mark.skipif(not ART.is_dir(), reason="committed ensemble artefacts absent")
def test_window_sensitivity_is_largest_on_the_coarsest_grid():
    """The first window extension matters most where the wet width is few cells."""
    rep = wc.audit(ART)
    w = {g["tag"]: g["window_sensitivity_max"] for g in rep["grids"]}
    assert w["dx4mm"] > GATE, "coarse grid must show a gate-sized window dependence"
    assert w["dx4mm"] > w["dx0.5mm"]
    assert w["dx1mm"] < 0.10