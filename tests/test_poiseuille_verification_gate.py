"""Machine-checkable verification gate for CASE-POISEUILLE-SPH.

Anchor: CASE-POISEUILLE-SPH
  ref: 解析解 u(y) = g·y·(H-y)/(2ν), u_max = g·H²/(8ν)
  容差: <5%

This test reads the committed benchmark artefact and verifies that the
peak-velocity error against the analytic solution is <5% at the finest
committed resolution (H/dx = 16).  It does NOT run the solver; it only
validates the artefact.

Known limitation: the wall-shear stress is ~50% low at every resolution
due to the Adami wall model's effective slip.  The u_max gate passes
because the over-diffusive bulk operator (ν_eff/ν ≈ 1.15) and the
under-predicted wall shear partially cancel.  This case therefore
certifies the *peak velocity* of the steady parabola, not the full
no-slip boundary layer.  See docs/verification_poiseuille_channel.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS_FILE = REPO / "benchmarks" / "poiseuille" / "out" / "results.json"

TOLERANCE_UMAX = 0.05  # 5% gate on peak velocity


@pytest.fixture(scope="module")
def results():
    assert RESULTS_FILE.exists(), f"Results file not found: {RESULTS_FILE}"
    with open(RESULTS_FILE) as f:
        return json.load(f)


class TestPoiseuilleVerificationGate:
    """CASE-POISEUILLE-SPH verification gate: u_max error < 5%."""

    def test_results_file_exists(self):
        assert RESULTS_FILE.exists(), (
            f"Benchmark results not found at {RESULTS_FILE}. "
            "Run: PYTHONPATH=src python benchmarks/poiseuille/run.py"
        )

    def test_has_fine_enough_resolution(self, results):
        """At least one resolution with H/dx >= 16 must be present."""
        max_hdx = max(r["resolution_H_over_dx"] for r in results["results"])
        assert max_hdx >= 16, (
            f"Need H/dx >= 16 for <5% u_max gate, got max H/dx = {max_hdx}"
        )

    def test_umax_error_below_5_percent_at_finest(self, results):
        """Peak velocity error vs analytic must be < 5% at finest resolution."""
        finest = max(results["results"], key=lambda r: r["resolution_H_over_dx"])
        hdx = finest["resolution_H_over_dx"]
        u_max_ratio = finest["u_max_ratio"]
        u_max_err_pct = abs(u_max_ratio - 1.0) * 100

        assert u_max_err_pct < 5.0, (
            f"CASE-POISEUILLE-SPH FAIL: u_max error = {u_max_err_pct:.2f}% "
            f"(ratio = {u_max_ratio:.4f}) at H/dx = {hdx}, "
            f"gate is < 5%"
        )

    def test_analytic_umax_is_correct(self, results):
        """Verify the analytic u_max = g·H²/(8ν) matches the stored value."""
        g = results["body_force"]
        H = results["height"]
        nu = results["viscosity"]
        expected = g * H ** 2 / (8.0 * nu)
        finest = max(results["results"], key=lambda r: r["resolution_H_over_dx"])
        u_max_analytic = finest["u_max_analytic"]
        assert abs(u_max_analytic - expected) < 1e-10, (
            f"Analytic u_max mismatch: stored {u_max_analytic} vs computed {expected}"
        )

    def test_steady_state_reached_at_finest(self, results):
        """Steady drift at finest resolution must be < 3%."""
        finest = max(results["results"], key=lambda r: r["resolution_H_over_dx"])
        drift = finest["steady_drift"]
        assert drift < 0.03, (
            f"Steady drift = {drift:.4f} at H/dx={finest['resolution_H_over_dx']}, "
            "expected < 0.03 (flow must be statistically steady)"
        )

    def test_mach_number_low(self, results):
        """Max Mach number must be < 0.1 for weakly-compressible validity."""
        for r in results["results"]:
            assert r["max_mach"] < 0.1, (
                f"Max Mach = {r['max_mach']:.4f} at H/dx={r['resolution_H_over_dx']}, "
                "expected < 0.1"
            )

    def test_parabola_fit_quality(self, results):
        """The quadratic fit residual must be small relative to u_max."""
        finest = max(results["results"], key=lambda r: r["resolution_H_over_dx"])
        fit_resid = finest["fit"]["fit_residual_rms"]
        u_max = finest["u_max_measured"]
        resid_ratio = fit_resid / u_max if u_max > 0 else float("inf")
        assert resid_ratio < 0.05, (
            f"Parabola fit residual / u_max = {resid_ratio:.4f} at H/dx={finest['resolution_H_over_dx']}, "
            "expected < 0.05 (profile must be parabolic)"
        )

    def test_free_slip_control_accelerates(self, results):
        """Free-slip negative control: flow must keep accelerating (no steady state)."""
        control = results.get("free_slip_control")
        assert control is not None, "Free-slip control data missing"
        trace = control.get("trace", [])
        assert len(trace) >= 2, "Free-slip control trace too short"
        u_first = trace[0]["u_max"]
        u_last = trace[-1]["u_max"]
        assert u_last > 2 * u_first, (
            f"Free-slip control: u_max went from {u_first:.4f} to {u_last:.4f}, "
            "expected monotonic acceleration (no steady state without wall friction)"
        )

    def test_convergence_trend(self, results):
        """u_max error should decrease with resolution (convergence trend)."""
        errors = [(r["resolution_H_over_dx"], abs(r["u_max_ratio"] - 1.0))
                  for r in results["results"]]
        errors.sort()
        # The finest resolution should have the smallest error
        assert errors[-1][1] <= errors[0][1], (
            f"u_max error not converging: {errors}"
        )
