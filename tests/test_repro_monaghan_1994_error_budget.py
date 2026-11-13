"""Tests for the REPRO-SPH-MONAGHAN1994 error budget analysis.

These tests verify that the error budget analysis is consistent with the
committed artifacts and that the interpretation is correct.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "repro_monaghan_1994" / "results"


@pytest.fixture(scope="module")
def budget():
    path = RESULTS / "error_budget.json"
    if not path.exists():
        pytest.skip("error_budget.json absent: run error_budget.py")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def summary():
    path = RESULTS / "summary.json"
    if not path.exists():
        pytest.skip("summary.json absent")
    return json.loads(path.read_text())


def test_error_budget_total_matches_summary(budget, summary):
    """The total MAE/amplitude in the budget must match the summary."""
    finest_case = budget["error_budget"]["case"]
    budget_total = budget["error_budget"]["total_mae_over_amplitude"]
    summary_total = summary["comparisons"][finest_case]["crosscheck"]["mae_over_amplitude"]
    assert budget_total == pytest.approx(summary_total, rel=1e-9)


def test_error_budget_bias_matches_summary(budget, summary):
    """The bias in the budget must match the summary."""
    finest_case = budget["error_budget"]["case"]
    budget_bias = budget["error_budget"]["bias"]
    summary_bias = summary["comparisons"][finest_case]["crosscheck"]["bias"]
    assert budget_bias == pytest.approx(summary_bias, rel=1e-9)


def test_reference_uncertainty_is_large(budget):
    """The reference uncertainty should be large (the two transcriptions disagree)."""
    ref_unc = budget["error_budget"]["components"]["reference_uncertainty"]
    assert ref_unc["max_spread_Za"] > 0.5, "The two transcriptions should disagree significantly"
    assert ref_unc["spread_over_amplitude"] > 0.2, "The spread should be >20% of amplitude"


def test_ritter_deviation_is_positive(budget):
    """Ritter should overpredict the front speed (SPH is slower)."""
    ritter = budget["error_budget"]["components"]["ritter_deviation"]
    assert ritter["mae_Za"] > 0, "Ritter deviation should be positive"
    assert ritter["mae_over_amplitude"] > 0.5, "Ritter deviation should be significant"


def test_compressibility_is_small(budget):
    """The compressibility error should be small (weak-compressibility is valid)."""
    comp = budget["error_budget"]["components"]["compressibility"]
    assert comp["max_rho_over_rho0_minus_1"] < 0.1, "Compression should be <10%"


def test_resolution_trend_bias_sign_change(budget):
    """The resolution trend should show a bias sign change."""
    trend = budget["resolution_trend"]
    assert trend["bias_sign_changes"] >= 1, "Bias should change sign at least once"
    assert trend["min_mae_over_amplitude"] < 0.05, "Minimum MAE should be <5%"


def test_ritter_comparison_points(budget):
    """The Ritter comparison should have points in the evaluation window."""
    ritter = budget["ritter_comparison"]
    assert ritter["n_points"] > 10, "Should have enough points in the window"
    assert ritter["mean_ritter_deviation_Za"] > 1.0, "Mean Ritter deviation should be >1 Z/a"


def test_error_budget_interpretation_is_consistent(budget):
    """The interpretation string should mention the key numbers."""
    interp = budget["error_budget"]["interpretation"]
    total = budget["error_budget"]["total_mae_over_amplitude"]
    assert f"{100*total:.2f}%" in interp, "Interpretation should mention total MAE"


def test_resolution_trend_interpretation(budget):
    """The resolution trend interpretation should explain the non-monotone behavior."""
    interp = budget["resolution_trend"]["interpretation"]
    assert "bias" in interp.lower(), "Should mention bias"
    assert "zero" in interp.lower() or "sign" in interp.lower(), "Should explain sign change"
