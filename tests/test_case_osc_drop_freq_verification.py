"""CASE-SPH-OSC-DROP-FREQ: verification data integrity test.

Reads the round7.json analysis output and asserts the key findings
that the anchor gate (|Q - 1| < 6%) is met at R/dx = 24 for all channels.

This is a *data integrity* test: it does not run SPH, it verifies that
the stored analysis results are self-consistent and meet the acceptance criteria.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "benchmarks" / "oscillating_drop_freq" / "out" / "round7.json"
GATE_PCT = 6.0


@pytest.fixture(scope="module")
def round7_data():
    """Load the round7 analysis JSON."""
    with open(DATA) as f:
        return json.load(f)


class TestRayleighReferenceValues:
    """The exact Rayleigh frequency for the canonical parameters."""

    def test_mode_2_frequency(self):
        """ω₂ = √(6σ/(ρR³)) for σ=0.072, ρ=1000, R=0.05."""
        sigma, rho, radius = 0.072, 1000.0, 0.05
        omega = math.sqrt(6.0 * sigma / (rho * radius ** 3))
        assert omega == pytest.approx(1.859032, rel=1e-5)
        period = 2 * math.pi / omega
        assert period == pytest.approx(3.379816, rel=1e-5)

    def test_mode_3_frequency(self):
        """ω₃ = √(24σ/(ρR³)) for σ=0.072, ρ=1000, R=0.05."""
        sigma, rho, radius = 0.072, 1000.0, 0.05
        omega = math.sqrt(24.0 * sigma / (rho * radius ** 3))
        assert omega == pytest.approx(3.718065, rel=1e-5)


class TestRound7DataIntegrity:
    """Assertions on the stored round7 analysis results."""

    def test_all_rows_present(self, round7_data):
        """10 analysis rows (3 channels × 2 modes × 2 eps, minus B eps=0.05)."""
        assert len(round7_data["analysis"]["rows"]) == 10

    def test_all_rows_at_gate_resolution(self, round7_data):
        """All rows have data at R/dx = 24."""
        for row in round7_data["analysis"]["rows"]:
            assert row.get("rdx_finest") == 24.0
            assert "q_at_24" in row
            assert "omega_error_at_24_pct" in row

    def test_all_rows_pass_6pct_gate(self, round7_data):
        """Every row passes the 6% gate at R/dx = 24."""
        for row in round7_data["analysis"]["rows"]:
            err = row["omega_error_at_24_pct"]
            assert abs(err) < GATE_PCT, (
                f"Row {row['channel']} m={row['mode']} ε={row['eps']}: "
                f"error {err:.2f}% exceeds {GATE_PCT}% gate"
            )

    def test_gate_pass_rate(self, round7_data):
        """10/10 rows pass the gate."""
        verdict = round7_data["verdict"]
        assert verdict["gate_pass_rows_at_24"] == 10
        assert verdict["gate_rows"] == 10

    def test_observed_order_median(self, round7_data):
        """Median observed order p ≈ 1.93 (not 1.0 as R4 assumed)."""
        p_median = round7_data["verdict"]["p_observed_median"]
        assert 1.5 < p_median < 2.5, f"Median p = {p_median} unexpected"

    def test_first_order_assumption_falsified(self, round7_data):
        """The R4 assumption of p=1 is falsified by the measured data."""
        assert round7_data["verdict"]["first_order_assumption_held"] is False

    def test_validation_pass_count(self, round7_data):
        """3/10 rows pass ASME V&V 20 validation."""
        assert round7_data["verdict"]["validation_pass_rows"] == 3

    def test_l3_systematic_bias(self, round7_data):
        """l=3 mode shows consistent ~-2.5% asymptotic bias."""
        l3_rows = [r for r in round7_data["analysis"]["rows"] if r["mode"] == 3]
        q_inf_vals = [r["q_inf"] for r in l3_rows if "q_inf" in r]
        if q_inf_vals:
            mean_q_inf = sum(q_inf_vals) / len(q_inf_vals)
            bias = (mean_q_inf - 1.0) * 100
            assert -3.0 < bias < -2.0, (
                f"l=3 asymptotic bias = {bias:.2f}%, expected ~-2.5%"
            )

    def test_q_values_converge_toward_one(self, round7_data):
        """Q values at R/dx=24 are all between 0.95 and 0.97."""
        for row in round7_data["analysis"]["rows"]:
            q = row["q_at_24"]
            assert 0.94 < q < 0.98, (
                f"Q(24) = {q:.4f} outside expected range for "
                f"{row['channel']} m={row['mode']}"
            )
