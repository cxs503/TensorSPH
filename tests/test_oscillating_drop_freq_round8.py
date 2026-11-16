"""Tests for round 8 synthesis of CASE-SPH-OSC-DROP-FREQ.

These tests verify that the meta-analysis correctly ingests round7.json,
computes the right statistics, and produces a consistent verdict.
They do NOT run SPH simulations.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))

from benchmarks.oscillating_drop_freq import round8_synthesis as R8


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def r7_data():
    return R8.load_round7()


@pytest.fixture(scope="module")
def meta(r7_data):
    return R8.meta_analysis(r7_data, None)


@pytest.fixture(scope="module")
def verdict(meta):
    return R8.make_verdict(meta)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def test_round7_json_exists():
    """round7.json must exist for round 8 to work."""
    p = R8.OUT / "round7.json"
    assert p.exists(), f"round7.json not found at {p}"


def test_round7_json_has_required_keys(r7_data):
    d = r7_data
    assert "verdict" in d
    assert "analysis" in d
    assert "rows" in d["analysis"] or "rows" in d.get("analysis", {})


# --------------------------------------------------------------------------- #
# Meta-analysis correctness
# --------------------------------------------------------------------------- #
def test_gate_pass_rate_is_100_pct(meta):
    """All 10 rows must pass the 6% gate at R/dx=24."""
    assert meta["gate"]["pass_rate"] == pytest.approx(1.0)
    assert meta["gate"]["pass_rows"] == meta["gate"]["total_rows"]


def test_validation_pass_rate_is_30_pct(meta):
    """Only 3/10 rows pass ASME V&V 20."""
    assert meta["validation"]["pass_rows"] == 3
    assert meta["validation"]["total_rows"] == 10
    assert meta["validation"]["pass_rate"] == pytest.approx(0.3)


def test_first_order_assumption_falsified(meta):
    """Round 4's p=1 assumption must be marked as not held."""
    assert meta["convergence"]["first_order_assumption_held"] is False


def test_p_median_near_1_93(meta):
    """The median observed order should be ~1.93."""
    assert meta["convergence"]["p_median"] == pytest.approx(1.93, abs=0.05)


def test_l3_systematic_bias(meta):
    """l=3 mode should have a ~-2.5% systematic bias."""
    bias = meta["systematic_bias"]["l3_q_inf_error_mean_pct"]
    assert bias == pytest.approx(-2.54, abs=0.1)
    assert meta["systematic_bias"]["l3_rows_analysed"] >= 3


def test_l2_differential_near_zero(meta):
    """l=2 differential channel should have near-zero bias."""
    bias = meta["systematic_bias"]["l2_differential_q_inf_bias"]
    assert abs(bias) < 0.01  # < 1%


def test_dynamic_arm_failed(meta):
    """Dynamic arm must be marked as FAIL."""
    assert meta["dynamic_arm"]["status"] == "FAIL"
    assert "divergence" in meta["dynamic_arm"]["reason"].lower()


def test_established_facts_non_empty(meta):
    """There must be at least 5 established facts."""
    assert len(meta["established_facts"]) >= 5


def test_open_questions_non_empty(meta):
    """There must be at least 3 open questions."""
    assert len(meta["open_questions"]) >= 3


# --------------------------------------------------------------------------- #
# Verdict correctness
# --------------------------------------------------------------------------- #
def test_verdict_overall_is_partial(verdict):
    """Overall verdict should be PARTIAL (gate pass but V&V 20 partial)."""
    assert verdict["overall"] == "PARTIAL"


def test_verdict_gate_pass(verdict):
    """Gate should be PASS."""
    assert verdict["gate"]["status"] == "PASS"
    assert verdict["gate"]["pass_rate"] == pytest.approx(1.0)


def test_verdict_asymptotic_partial(verdict):
    """Asymptotic validation should be PARTIAL."""
    assert verdict["asymptotic_validation"]["status"] == "PARTIAL"
    assert verdict["asymptotic_validation"]["pass_rate"] == pytest.approx(0.3)


def test_verdict_dynamic_fail(verdict):
    """Dynamic arm should be FAIL."""
    assert verdict["dynamic_arm"]["status"] == "FAIL"


def test_verdict_has_recommendation(verdict):
    """Verdict must include a recommendation."""
    assert "recommendation" in verdict
    assert len(verdict["recommendation"]) > 50


def test_verdict_anchor_matches(verdict):
    """Verdict anchor must match the expected slug."""
    assert verdict["anchor"] == "CASE-SPH-OSC-DROP-FREQ"


# --------------------------------------------------------------------------- #
# Figure generation
# --------------------------------------------------------------------------- #
def test_figure_generated(meta, r7_data):
    """The synthesis figure must be created successfully."""
    fig_path = R8.make_synthesis_figure(r7_data, meta)
    assert pathlib.Path(fig_path).exists()
    assert pathlib.Path(fig_path).stat().st_size > 10000  # > 10 KB


# --------------------------------------------------------------------------- #
# Round 8 JSON output
# --------------------------------------------------------------------------- #
def test_round8_json_structure():
    """Run the full synthesis and check the output structure."""
    r7 = R8.load_round7()
    meta = R8.meta_analysis(r7, None)
    verdict = R8.make_verdict(meta)

    result = {
        "case": "SPH-OSC-DROP-FREQ",
        "round": 8,
        "meta_analysis": meta,
        "verdict": verdict,
    }

    # Check required keys
    assert result["case"] == "SPH-OSC-DROP-FREQ"
    assert result["round"] == 8
    assert "gate" in result["meta_analysis"]
    assert "validation" in result["meta_analysis"]
    assert "convergence" in result["meta_analysis"]
    assert "systematic_bias" in result["meta_analysis"]
    assert "dynamic_arm" in result["meta_analysis"]
    assert "established_facts" in result["meta_analysis"]
    assert "open_questions" in result["meta_analysis"]
    assert "overall" in result["verdict"]
    assert "recommendation" in result["verdict"]


# --------------------------------------------------------------------------- #
# Consistency checks
# --------------------------------------------------------------------------- #
def test_gate_and_validation_counts_consistent(r7_data, meta):
    """Gate and validation counts must match the analysis rows."""
    ok_rows = [r for r in r7_data["analysis"]["rows"] if r["status"] == "ok"]
    assert meta["gate"]["total_rows"] == len(ok_rows)
    assert meta["validation"]["total_rows"] == len(ok_rows)


def test_v_pass_count_matches_analysis(r7_data, meta):
    """V&V 20 pass count must match the analysis."""
    ok_rows = [r for r in r7_data["analysis"]["rows"] if r["status"] == "ok"]
    v_pass = sum(1 for r in ok_rows if r["validation"]["pass"])
    assert meta["validation"]["pass_rows"] == v_pass


def test_l3_bias_rows_are_monotone(r7_data, meta):
    """l=3 rows used for bias calculation must be monotone."""
    ok_rows = [r for r in r7_data["analysis"]["rows"] if r["status"] == "ok"]
    l3_mono = [r for r in ok_rows if r["mode"] == 3 and r["monotone"]]
    assert len(l3_mono) == meta["systematic_bias"]["l3_rows_analysed"]
