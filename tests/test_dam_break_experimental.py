"""Tests for the dam-break experimental validation infrastructure."""
import pytest

from benchmarks.dam_break.experimental_validation import (
    load_experimental_data,
    interpolate_exp,
    compute_exp_error,
    _interp,
)


@pytest.fixture
def exp_data():
    """Load the bundled Martin & Moyce experimental data."""
    ref = load_experimental_data()
    return ref["data"]


class TestExperimentalData:
    """Test the experimental data loading and structure."""

    def test_data_loads(self):
        ref = load_experimental_data()
        assert "data" in ref
        assert "source" in ref
        assert "digitization" in ref
        assert "experimental_setup" in ref

    def test_data_has_provenance(self, exp_data):
        ref = load_experimental_data()
        assert "paper" in ref["source"]
        assert "doi" in ref["source"]
        assert "method" in ref["digitization"]
        assert "uncertainty" in ref["digitization"]

    def test_data_points(self, exp_data):
        assert len(exp_data) >= 10
        for point in exp_data:
            assert "tau" in point
            assert "Z_over_a" in point
            assert point["tau"] >= 0
            assert point["Z_over_a"] >= 0

    def test_data_monotonic(self, exp_data):
        """Z/a should be monotonically increasing with tau."""
        for i in range(1, len(exp_data)):
            assert exp_data[i]["tau"] > exp_data[i - 1]["tau"]
            assert exp_data[i]["Z_over_a"] >= exp_data[i - 1]["Z_over_a"]

    def test_initial_condition(self, exp_data):
        """At tau=0, Z/a should be 0."""
        assert exp_data[0]["tau"] == 0.0
        assert exp_data[0]["Z_over_a"] == 0.0


class TestInterpolation:
    """Test the interpolation functions."""

    def test_interp_exact(self):
        xs = [0.0, 1.0, 2.0]
        ys = [0.0, 1.0, 4.0]
        assert _interp(xs, ys, 0.0) == pytest.approx(0.0)
        assert _interp(xs, ys, 1.0) == pytest.approx(1.0)
        assert _interp(xs, ys, 2.0) == pytest.approx(4.0)

    def test_interp_linear(self):
        xs = [0.0, 2.0]
        ys = [0.0, 4.0]
        assert _interp(xs, ys, 1.0) == pytest.approx(2.0)

    def test_interp_extrapolation(self):
        xs = [0.0, 1.0]
        ys = [0.0, 1.0]
        assert _interp(xs, ys, -0.5) == pytest.approx(0.0)
        assert _interp(xs, ys, 2.0) == pytest.approx(1.0)

    def test_interp_empty(self):
        assert _interp([], [], 0.5) is None

    def test_exp_interpolation(self, exp_data):
        """Test interpolation of experimental data."""
        # Should return exact values at data points
        for point in exp_data[:5]:
            result = interpolate_exp(exp_data, point["tau"])
            assert result == pytest.approx(point["Z_over_a"])


class TestErrorMetrics:
    """Test the error computation functions."""

    def test_perfect_match(self):
        sim_tau = [0.0, 1.0, 2.0]
        sim_za = [0.0, 1.0, 2.0]
        exp_data = [
            {"tau": 0.0, "Z_over_a": 0.0, "uncertainty": 0.05},
            {"tau": 1.0, "Z_over_a": 1.0, "uncertainty": 0.05},
            {"tau": 2.0, "Z_over_a": 2.0, "uncertainty": 0.05},
        ]
        result = compute_exp_error(sim_tau, sim_za, exp_data)
        assert result["max_absolute_error"] == pytest.approx(0.0)
        assert result["mean_absolute_error"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["points_compared"] == 3
        assert result["within_uncertainty"] == pytest.approx(1.0)

    def test_constant_offset(self):
        sim_tau = [0.0, 1.0, 2.0]
        sim_za = [0.1, 1.1, 2.1]
        exp_data = [
            {"tau": 0.0, "Z_over_a": 0.0, "uncertainty": 0.05},
            {"tau": 1.0, "Z_over_a": 1.0, "uncertainty": 0.05},
            {"tau": 2.0, "Z_over_a": 2.0, "uncertainty": 0.05},
        ]
        result = compute_exp_error(sim_tau, sim_za, exp_data)
        assert result["max_absolute_error"] == pytest.approx(0.1)
        assert result["mean_absolute_error"] == pytest.approx(0.1)
        assert result["rmse"] == pytest.approx(0.1)
        assert result["within_uncertainty"] == pytest.approx(0.0)

    def test_no_overlap(self):
        sim_tau = [10.0, 11.0, 12.0]
        sim_za = [5.0, 6.0, 7.0]
        exp_data = [
            {"tau": 0.0, "Z_over_a": 0.0, "uncertainty": 0.05},
            {"tau": 1.0, "Z_over_a": 1.0, "uncertainty": 0.05},
        ]
        result = compute_exp_error(sim_tau, sim_za, exp_data)
        assert result["points_compared"] == 0
        assert result["mean_absolute_error"] is None

    def test_partial_overlap(self):
        sim_tau = [0.0, 1.0, 2.0]
        sim_za = [0.0, 1.0, 2.0]
        exp_data = [
            {"tau": 0.5, "Z_over_a": 0.5, "uncertainty": 0.05},
            {"tau": 5.0, "Z_over_a": 5.0, "uncertainty": 0.05},
        ]
        result = compute_exp_error(sim_tau, sim_za, exp_data)
        assert result["points_compared"] == 1
        assert result["mean_absolute_error"] == pytest.approx(0.0)
