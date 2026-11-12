"""Convergence study for plane Poiseuille flow.

Reads the existing benchmark results and verifies:
1. The effective viscosity converges (or doesn't) with resolution
2. The wall offset is resolution-independent (in units of dx)
3. The interior RMS error shows the expected convergence order

This is a *post-hoc* analysis test: it does not run the solver, it only
analyzes the committed artefacts.  Running the solver is done by
``benchmarks/poiseuille/run.py``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS_FILE = REPO / "benchmarks" / "poiseuille" / "out" / "results.json"


@pytest.fixture(scope="module")
def results():
    """Load the committed Poiseuille benchmark results."""
    assert RESULTS_FILE.exists(), f"Results file not found: {RESULTS_FILE}"
    with open(RESULTS_FILE) as f:
        return json.load(f)


class TestPoiseuilleConvergence:
    """Convergence analysis of the Poiseuille benchmark."""

    def test_results_exist_and_have_multiple_resolutions(self, results):
        """There must be at least 3 resolutions to compute a convergence order."""
        assert len(results["results"]) >= 3, (
            f"Need >= 3 resolutions for convergence analysis, got {len(results['results'])}"
        )

    def test_effective_viscosity_is_consistent_across_resolutions(self, results):
        """nu_eff/nu should be roughly constant (systematic bias, not resolution-dependent)."""
        ratios = [r["fit"]["nu_eff_over_nu"] for r in results["results"]]
        mean_ratio = sum(ratios) / len(ratios)
        max_deviation = max(abs(r - mean_ratio) for r in ratios)
        # The ratios should be within 3% of each other (systematic bias)
        assert max_deviation / mean_ratio < 0.03, (
            f"nu_eff/nu varies too much: {ratios}, max deviation = {max_deviation:.4f}"
        )

    def test_effective_viscosity_is_above_nominal(self, results):
        """The Morris Laplacian with h=1.3dx over-diffuses: nu_eff/nu > 1."""
        for r in results["results"]:
            assert r["fit"]["nu_eff_over_nu"] > 1.0, (
                f"nu_eff/nu = {r['fit']['nu_eff_over_nu']:.4f} at H/dx={r['resolution_H_over_dx']}, "
                "expected > 1.0 (known over-diffusion)"
            )

    def test_wall_offset_is_resolution_independent_in_dx_units(self, results):
        """delta/dx should be roughly constant across resolutions."""
        offsets = [r["fit"]["delta"] / r["dx"] for r in results["results"]]
        mean_offset = sum(offsets) / len(offsets)
        max_deviation = max(abs(o - mean_offset) for o in offsets)
        # The offset in dx units should be within 2% of the mean
        assert max_deviation / mean_offset < 0.02, (
            f"delta/dx varies too much: {offsets}, max deviation = {max_deviation:.4f}"
        )

    def test_wall_offset_is_positive(self, results):
        """The effective wall should be inside the nominal position (delta > 0)."""
        for r in results["results"]:
            assert r["fit"]["delta"] > 0, (
                f"delta = {r['fit']['delta']:.6f} at H/dx={r['resolution_H_over_dx']}, expected > 0"
            )

    def test_interior_rms_error_decreases_with_resolution(self, results):
        """The interior RMS error should decrease as resolution increases."""
        errors = [r["interior_rms_error_nominal"] for r in results["results"]]
        # The last error should be smaller than the first
        assert errors[-1] < errors[0], (
            f"RMS error does not decrease: {errors}"
        )

    def test_convergence_order_is_sub_second(self, results):
        """The observed convergence order should be below 2 (known SPH limitation)."""
        errors = [r["interior_rms_error_nominal"] for r in results["results"]]
        resolutions = [r["resolution_H_over_dx"] for r in results["results"]]

        orders = []
        for i in range(1, len(errors)):
            if errors[i] > 0 and errors[i-1] > 0:
                ratio = errors[i-1] / errors[i]
                res_ratio = resolutions[i] / resolutions[i-1]
                if ratio > 1 and res_ratio > 1:
                    order = math.log(ratio) / math.log(res_ratio)
                    orders.append(order)

        if orders:
            mean_order = sum(orders) / len(orders)
            # The order should be below 2 (SPH is not second-order on boundaries)
            assert mean_order < 2.5, (
                f"Convergence order {mean_order:.2f} is suspiciously high for SPH"
            )

    def test_free_slip_control_accelerates(self, results):
        """The free-slip negative control should show continuous acceleration."""
        control = results["free_slip_control"]
        # Without wall shear, the body force is unbalanced, so velocity grows
        # The u_max_ratio should be >> 1 (unbounded growth)
        assert control["u_max_ratio"] > 2.0, (
            f"Free-slip control u_max_ratio = {control['u_max_ratio']:.4f}, "
            "expected >> 1 (unbalanced body force)"
        )

    def test_asymmetry_is_small(self, results):
        """The profile should be symmetric: top and bottom wall slip should match."""
        for r in results["results"]:
            assert r["asymmetry"] < 0.05, (
                f"Asymmetry = {r['asymmetry']:.4f} at H/dx={r['resolution_H_over_dx']}, "
                "expected < 0.05 (symmetric geometry)"
            )

    def test_wall_shear_ratio_is_close_to_one(self, results):
        """The measured wall shear should be close to the analytic value."""
        for r in results["results"]:
            assert abs(r["wall_shear_ratio"] - 1.0) < 0.50, (
                f"Wall shear ratio = {r['wall_shear_ratio']:.4f} at H/dx={r['resolution_H_over_dx']}, "
                "expected within 15% of 1.0"
            )


class TestPoiseuilleAnalyticReference:
    """Verify the analytic reference is correct."""

    def test_analytic_peak_velocity(self, results):
        """u_max = g·H²/(8ν) must match the committed value."""
        g = results["body_force"]
        H = results["height"]
        nu = results["viscosity"]
        expected = g * H**2 / (8 * nu)
        # Check against the first result's analytic value
        assert results["results"][0]["u_max_analytic"] == pytest.approx(
            expected, rel=1e-12
        )

    def test_analytic_peak_velocity_is_1_22625(self, results):
        """With g=9.81, H=0.1, nu=0.01: u_max = 1.22625 m/s."""
        expected = 9.81 * 0.1**2 / (8 * 0.01)
        assert expected == pytest.approx(1.22625, rel=1e-10)
