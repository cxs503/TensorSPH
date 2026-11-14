"""Validation test for CASE-SPH-KLEEFSMAN-DAMBREAK-OBST against experimental data.

This test asserts the comparison logic between TensorSPH simulation results
and the digitized experimental data from Kleefsman et al. (2005).

It does NOT assert that the simulation passes the 10% tolerance (it doesn't
for arrival time) — rather it asserts that the comparison is computed correctly
and that the verdict is honest.
"""
import csv
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REF = json.loads((ROOT / "benchmarks/kleefsman_obstacle/reference.json").read_text())
EXP = REF["experimental_values"]
TOL = REF["tolerance"]["relative"]

# Experimental digitized data
EXP_CSV = ROOT / "docs/m20884/outputs/kleefsman_h2_ref/h2_fig19.csv"


def load_exp_data():
    """Load the digitized experimental H2 time series."""
    data = []
    with open(EXP_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append((float(row["t_s"]), float(row["h_m"])))
    return data


def load_sim_ts(tag):
    """Load simulation time series from results directory."""
    ts_path = ROOT / "benchmarks/kleefsman_obstacle/results" / f"{tag}_time_series.csv"
    if not ts_path.exists():
        return None
    t, h = [], []
    with open(ts_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["t_s"]))
            h.append(float(row["height_m"]))
    return t, h


def load_sim_metrics(tag):
    """Load simulation metrics JSON."""
    m_path = ROOT / "benchmarks/kleefsman_obstacle/results" / f"{tag}.json"
    if not m_path.exists():
        return None
    return json.loads(m_path.read_text())


def compute_error(got, ref):
    """Relative error |got - ref| / |ref|."""
    if ref == 0:
        return float("inf") if got != 0 else 0.0
    return abs(got - ref) / abs(ref)


# --------------------------------------------------------------------------- #
# Experimental data integrity
# --------------------------------------------------------------------------- #
class TestExperimentalData:
    """The digitized experimental data must be self-consistent."""

    def test_exp_csv_exists(self):
        assert EXP_CSV.exists(), f"Experimental CSV not found at {EXP_CSV}"

    def test_exp_data_has_points(self):
        data = load_exp_data()
        assert len(data) > 50, f"Only {len(data)} experimental points"

    def test_exp_peak_matches_anchor(self):
        """The peak of the digitized data should match the anchored value."""
        data = load_exp_data()
        peak_h = max(h for _, h in data)
        assert abs(peak_h - EXP["peak_height_m"]) < 0.01, \
            f"Digitized peak {peak_h:.3f} vs anchored {EXP['peak_height_m']:.3f}"

    def test_exp_arrival_matches_anchor(self):
        """The arrival time (first crossing of 0.02 m) should match the anchor."""
        data = load_exp_data()
        for i, (t, h) in enumerate(data):
            if h >= 0.02:
                # Linear interpolation
                if i > 0:
                    t0, h0 = data[i - 1]
                    t_arr = t0 + (0.02 - h0) * (t - t0) / (h - h0)
                else:
                    t_arr = t
                assert abs(t_arr - EXP["arrival_time_s"]) < 0.02, \
                    f"Digitized arrival {t_arr:.3f} vs anchored {EXP['arrival_time_s']:.3f}"
                return
        pytest.fail("Experimental data never crosses 0.02 m threshold")


# --------------------------------------------------------------------------- #
# Simulation result integrity
# --------------------------------------------------------------------------- #
class TestSimulationResults:
    """Each simulation result must satisfy basic invariants."""

    @pytest.fixture(params=["dx0.04", "dx0.02", "dx0.01"])
    def tag(self, request):
        return request.param

    def test_metrics_file_exists(self, tag):
        m = load_sim_metrics(tag)
        assert m is not None, f"Metrics file for {tag} not found"

    def test_time_series_exists(self, tag):
        ts = load_sim_ts(tag)
        assert ts is not None, f"Time series for {tag} not found"

    def test_mass_conservation(self, tag):
        m = load_sim_metrics(tag)
        assert m["mass_error_rel"] == 0.0, f"{tag}: mass error {m['mass_error_rel']}"

    def test_no_boundary_escape(self, tag):
        m = load_sim_metrics(tag)
        assert m["boundary_escapes"] == 0, f"{tag}: {m['boundary_escapes']} escapes"

    def test_positive_peak(self, tag):
        m = load_sim_metrics(tag)
        assert m["peak_height_m"] > 0, f"{tag}: peak height must be positive"

    def test_arrival_before_peak(self, tag):
        m = load_sim_metrics(tag)
        assert m["arrival_time_s"] < m["peak_time_s"], \
            f"{tag}: arrival {m['arrival_time_s']} >= peak {m['peak_time_s']}"


# --------------------------------------------------------------------------- #
# Experimental comparison (the core validation)
# --------------------------------------------------------------------------- #
class TestExperimentalComparison:
    """Compare simulation results against experimental values."""

    def test_dx01_peak_error(self):
        """dx=0.01 peak height error should be computed correctly."""
        m = load_sim_metrics("dx0.01")
        err = compute_error(m["peak_height_m"], EXP["peak_height_m"])
        assert err < 0.03, f"dx=0.01 peak error {err:.1%} >= 3%"  # 2.0% passes

    def test_dx01_arrival_error(self):
        """dx=0.01 arrival time error should be computed correctly."""
        m = load_sim_metrics("dx0.01")
        err = compute_error(m["arrival_time_s"], EXP["arrival_time_s"])
        # This is expected to FAIL the 10% tolerance — we assert the error is
        # in the expected range (not that it passes).
        assert 0.20 < err < 0.30, \
            f"dx=0.01 arrival error {err:.1%} not in expected [20%, 30%] range"

    def test_dx01_verdict_is_fail(self):
        """The overall verdict for dx=0.01 should be FAIL (arrival time fails)."""
        m = load_sim_metrics("dx0.01")
        peak_err = compute_error(m["peak_height_m"], EXP["peak_height_m"])
        arr_err = compute_error(m["arrival_time_s"], EXP["arrival_time_s"])
        verdict_pass = (peak_err < TOL) and (arr_err < TOL)
        assert not verdict_pass, \
            "dx=0.01 should FAIL the 10% tolerance (arrival time is ~24% off)"

    def test_arrival_error_decreases_with_resolution(self):
        """Arrival time error should decrease (but not vanish) with finer dx."""
        errors = {}
        for tag in ["dx0.04", "dx0.02", "dx0.01"]:
            m = load_sim_metrics(tag)
            errors[tag] = compute_error(m["arrival_time_s"], EXP["arrival_time_s"])
        # Error should decrease monotonically
        assert errors["dx0.04"] > errors["dx0.02"] > errors["dx0.01"], \
            f"Arrival error not monotone: {errors}"

    def test_peak_error_not_monotone(self):
        """Peak height error is NOT monotone (splash-dominated)."""
        errors = {}
        for tag in ["dx0.04", "dx0.02", "dx0.01"]:
            m = load_sim_metrics(tag)
            errors[tag] = compute_error(m["peak_height_m"], EXP["peak_height_m"])
        # dx=0.02 has the WORST error (non-monotone)
        assert errors["dx0.02"] > errors["dx0.04"], \
            f"Peak error at dx=0.02 should be worst: {errors}"


# --------------------------------------------------------------------------- #
# Convergence ladder assertions
# --------------------------------------------------------------------------- #
class TestConvergenceLadder:
    """Assert properties of the convergence analysis."""

    @pytest.fixture
    def ladder(self):
        path = ROOT / "benchmarks/kleefsman_obstacle/results/convergence_ladder.json"
        if not path.exists():
            pytest.skip("convergence_ladder.json not found")
        return json.loads(path.read_text())

    def test_arrival_converged(self, ladder):
        """Arrival time is numerically converged (GCI < 10%)."""
        arr = ladder["metrics"]["arrival_time_s"]
        assert arr["verdict"] == "CONVERGED"
        assert arr["gci_fine"] < 0.10

    def test_arrival_converges_to_wrong_value(self, ladder):
        """Arrival time converges, but to a value ~24% above experiment."""
        arr = ladder["metrics"]["arrival_time_s"]
        finest_val = arr["values"][-1]  # dx=0.01
        err = compute_error(finest_val, EXP["arrival_time_s"])
        assert err > 0.20, \
            f"Converged arrival {finest_val:.3f} should be >20% off experiment"

    def test_peak_not_converged(self, ladder):
        """Peak height is NOT_CONVERGED (splash-dominated)."""
        peak = ladder["metrics"]["peak_height_m_raw"]
        assert peak["verdict"] == "NOT_CONVERGED"


# --------------------------------------------------------------------------- #
# Gauge ladder assertions
# --------------------------------------------------------------------------- #
class TestGaugeLadder:
    """Assert properties of the gauge operator comparison."""

    @pytest.fixture
    def gauge(self):
        path = ROOT / "benchmarks/kleefsman_obstacle/results/gauge_ladder/analysis.json"
        if not path.exists():
            pytest.skip("gauge_ladder/analysis.json not found")
        return json.loads(path.read_text())

    def test_slab_gauge_low_spread_upstream(self, gauge):
        """Slab gauge at x=2.228 (upstream) should have low pair spread."""
        slab = gauge["metrics"]["slab@x2.228"]
        assert slab["pair_spread"]["peak"] < 0.05

    def test_height_gauge_high_spread_near_box(self, gauge):
        """Height gauge at x=2.426 (near box) should have high pair spread."""
        height = gauge["metrics"]["height@x2.426"]
        assert height["pair_spread"]["peak"] > 0.10
