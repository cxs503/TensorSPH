"""Validation tests for the Kleefsman obstacle case — m19936 deliverable.

These tests assert the *invariants* of the Kleefsman case:
- geometry self-consistency
- mass conservation and boundary integrity
- convergence of arrival time (not peak height)
- c0 sensitivity monotonicity
- gauge operator bias bounds
"""
import json
import math
from pathlib import Path

import pytest

REF = json.loads(Path(__file__).resolve().parents[1].joinpath(
    "benchmarks/kleefsman_obstacle/reference.json").read_text())
RESULTS = Path(__file__).resolve().parents[1].joinpath(
    "benchmarks/kleefsman_obstacle/results")


# --------------------------------------------------------------------------- #
# Geometry invariants
# --------------------------------------------------------------------------- #
class TestGeometryInvariants:
    """Geometry must be self-consistent regardless of solver state."""

    def test_tank_length_partition(self):
        """1.228 + 1.248 + 0.744 = 3.220 exactly."""
        g = REF["geometry_m"]
        residual = g["column_length"] + (g["obstacle_upstream_face_x"] - g["column_length"]) \
            + g["downstream_gap"] - g["tank_length"]
        assert abs(residual) < 1e-12, f"tank partition residual = {residual}"

    def test_obstacle_fits_in_tank(self):
        g = REF["geometry_m"]
        assert g["obstacle_upstream_face_x"] + g["obstacle_length"] < g["tank_length"]
        assert g["obstacle_height"] < g["tank_height"]

    def test_probe_between_gate_and_box(self):
        g = REF["geometry_m"]
        probe_x = g["obstacle_upstream_face_x"] - 0.05  # default probe_offset
        assert g["column_length"] < probe_x < g["obstacle_upstream_face_x"]


# --------------------------------------------------------------------------- #
# Result file integrity
# --------------------------------------------------------------------------- #
class TestResultIntegrity:
    """Each result JSON must satisfy basic physical invariants."""

    @pytest.fixture(params=[0.04, 0.02, 0.01])
    def result(self, request):
        dx = request.param
        path = RESULTS / f"dx{dx:g}.json"
        if not path.exists():
            pytest.skip(f"dx{dx:g}.json not found")
        return json.loads(path.read_text())

    def test_mass_conservation(self, result):
        assert result["mass_error_rel"] == 0.0, \
            f"mass error {result['mass_error_rel']}"

    def test_no_boundary_escape(self, result):
        assert result["boundary_escapes"] == 0, \
            f"{result['boundary_escapes']} particles escaped"

    def test_positive_peak(self, result):
        assert result["peak_height_m"] > 0, "peak height must be positive"

    def test_arrival_before_peak(self, result):
        assert result["arrival_time_s"] < result["peak_time_s"], \
            f"arrival {result['arrival_time_s']} >= peak {result['peak_time_s']}"

    def test_front_within_tank(self, result):
        g = REF["geometry_m"]
        # front is max fluid x; must not exceed tank length
        # (we check the time series if available)
        ts_path = RESULTS / f"dx{result['dx']:g}_time_series.csv"
        if ts_path.exists():
            with open(ts_path) as f:
                next(f)
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        front = float(parts[2])
                        assert front <= g["tank_length"] + 1e-6, \
                            f"front {front} exceeds tank {g['tank_length']}"


# --------------------------------------------------------------------------- #
# Convergence assertions
# --------------------------------------------------------------------------- #
class TestConvergence:
    """Assert convergence properties from the ladder analysis."""

    @pytest.fixture
    def ladder(self):
        path = RESULTS / "convergence_ladder.json"
        if not path.exists():
            pytest.skip("convergence_ladder.json not found")
        return json.loads(path.read_text())

    def test_arrival_time_converged(self, ladder):
        """Arrival time must be CONVERGED with GCI < 10%."""
        arrival = ladder["metrics"]["arrival_time_s"]
        assert arrival["verdict"] == "CONVERGED", \
            f"arrival time verdict: {arrival['verdict']}"
        assert arrival["gci_fine"] < 0.10, \
            f"arrival GCI {arrival['gci_fine']:.4f} >= 10%"

    def test_arrival_time_order_positive(self, ladder):
        """Observed order p should be positive (monotone convergence)."""
        arrival = ladder["metrics"]["arrival_time_s"]
        p = arrival["p_finest"]
        assert math.isfinite(p) and p > 0, f"arrival order p = {p}"

    def test_peak_height_not_converged(self, ladder):
        """Peak height is known to be NOT_CONVERGED (splash-dominated)."""
        peak = ladder["metrics"]["peak_height_m_raw"]
        assert peak["verdict"] == "NOT_CONVERGED", \
            "peak height should NOT be converged (splash-dominated)"


# --------------------------------------------------------------------------- #
# c0 sensitivity
# --------------------------------------------------------------------------- #
class TestC0Sensitivity:
    """c0 sensitivity scan must show monotone decreasing peak with increasing c0."""

    @pytest.fixture
    def c0_scan(self):
        path = RESULTS / "c0_scan" / "c0_scan_summary.json"
        if not path.exists():
            pytest.skip("c0_scan_summary.json not found")
        return json.loads(path.read_text())

    def test_dx02_monotone_decreasing(self, c0_scan):
        """At dx=0.02, peak height must decrease as c0 increases."""
        per_grid = c0_scan["per_grid"]["dx0.02"]
        assert per_grid["monotone_decreasing"], \
            "peak height should decrease with increasing c0 at dx=0.02"

    def test_dx04_monotone_decreasing(self, c0_scan):
        """At dx=0.04, peak height must decrease as c0 increases."""
        per_grid = c0_scan["per_grid"]["dx0.04"]
        assert per_grid["monotone_decreasing"], \
            "peak height should decrease with increasing c0 at dx=0.04"

    def test_compression_below_5pct_at_max_c0(self, c0_scan):
        """At max c0, density compression should be below 5%."""
        for grid in ["dx0.02", "dx0.04"]:
            assert c0_scan["per_grid"][grid]["compression_below_5pct_at_max_c0"], \
                f"{grid} compression > 5% at max c0"


# --------------------------------------------------------------------------- #
# Gauge ladder
# --------------------------------------------------------------------------- #
class TestGaugeLadder:
    """Gauge operator bias must be bounded."""

    @pytest.fixture
    def gauge_analysis(self):
        path = RESULTS / "gauge_ladder" / "analysis.json"
        if not path.exists():
            pytest.skip("gauge_ladder/analysis.json not found")
        return json.loads(path.read_text())

    def test_slab_gauge_upstream_low_spread(self, gauge_analysis):
        """Slab gauge at x=2.228 (upstream 0.248 m) should have low spread."""
        slab = gauge_analysis["metrics"]["slab@x2.228"]
        assert slab["pair_spread"]["peak"] < 0.05, \
            f"slab@x2.228 peak spread {slab['pair_spread']['peak']:.4f} >= 5%"

    def test_height_gauge_near_box_high_spread(self, gauge_analysis):
        """Height gauge at x=2.426 (near box) should have high spread."""
        height = gauge_analysis["metrics"]["height@x2.426"]
        assert height["pair_spread"]["peak"] > 0.10, \
            f"height@x2.426 peak spread {height['pair_spread']['peak']:.4f} <= 10% " \
            "(expected high spread due to splash)"
