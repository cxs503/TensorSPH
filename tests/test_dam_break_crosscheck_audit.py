"""Tests for the B2 experimental-reference audit.

The audit's job is to say whether the bundled Martin & Moyce table can carry a
validation verdict. These tests pin the *measured* structural facts, so a future
edit to either table cannot quietly restore the old verdict:

* the bundled table stays identifiable as a piecewise-linear sketch
  (arithmetic abscissae, 0.05 ordinate grid, four slope levels, no
  acceleration phase);
* the independent transcription keeps the acceleration phase a gate-release
  experiment must have;
* the cross-source spread stays far above the claimed ±0.05 anywhere it is
  quoted, and the verdict stays ``withheld`` while it does.

Also pins the error metric (`error_vs`) with a self-consistency case: comparing
a curve against an interpolant of a table must return zero error.
"""
import json
from pathlib import Path

import pytest

from benchmarks.dam_break.crosscheck_audit import (
    _interp,
    cross_source_spread,
    evaluate,
    grid_stats,
    load_bundled,
    load_crosscheck,
)
from benchmarks.dam_break import reference_ladder
from benchmarks.dam_break.reference_ladder import error_vs

HERE = Path(__file__).resolve().parent.parent / "benchmarks" / "dam_break"


@pytest.fixture(scope="module")
def bundled():
    return load_bundled()


@pytest.fixture(scope="module")
def crosscheck():
    return load_crosscheck()


@pytest.fixture(scope="module")
def report(bundled, crosscheck):
    return evaluate({"bundled": bundled, "crosscheck": crosscheck})


# --------------------------------------------------------------------------- #
# tables load with the same convention
# --------------------------------------------------------------------------- #
def test_crosscheck_table_converts_to_the_same_frame(crosscheck):
    assert crosscheck["name"] == "crosscheck"
    assert len(crosscheck["tau"]) == 14
    # X = x_front/a, so the front at t = 0 sits one column width from the wall.
    assert crosscheck["za"][0] == pytest.approx(0.0, abs=1e-12)
    assert crosscheck["tau"] == sorted(crosscheck["tau"])
    assert crosscheck["za"] == sorted(crosscheck["za"])
    assert crosscheck["claimed_uncertainty"] == pytest.approx(0.05)


def test_both_tables_cover_the_same_aspect_ratio_window(bundled, crosscheck):
    assert bundled["tau"][0] == crosscheck["tau"][0] == 0.0
    overlap = min(bundled["tau"][-1], crosscheck["tau"][-1])
    assert overlap >= 2.9


# --------------------------------------------------------------------------- #
# structural facts of the bundled table
# --------------------------------------------------------------------------- #
def test_bundled_table_is_a_piecewise_linear_sketch(bundled):
    g = grid_stats(bundled["tau"], bundled["za"])
    assert g["tau_grid_is_arithmetic"] is True
    assert g["tau_step_spread"] < 1e-9
    assert g["ordinate_quantum"] == pytest.approx(0.05)
    assert g["n_slope_levels"] == 4
    # the four levels are separated by much more than the grouping tolerance
    levels = g["slope_levels"]
    assert levels == sorted(levels)
    assert levels[-1] / levels[0] > 2.0


def test_bundled_table_has_no_acceleration_phase(bundled, crosscheck):
    gb = grid_stats(bundled["tau"], bundled["za"])
    gc = grid_stats(crosscheck["tau"], crosscheck["za"])
    # The bundled curve starts at 80 % of its peak front speed
    assert gb["first_slope_to_peak"] > 0.75
    # ... the independent transcription starts at ~20 % and accelerates 6x
    assert gc["first_slope_to_peak"] < 0.25
    assert gc["slope_peak"] / gc["slope_first"] > 4.0
    # and it is not an arithmetic grid read off a round grid
    assert gc["ordinate_quantum"] <= 0.01


def test_cross_source_spread_dwarfs_the_claimed_uncertainty(bundled, crosscheck):
    spread = cross_source_spread(bundled, crosscheck)
    assert spread["n_overlap"] >= 13
    assert spread["max_abs_delta"] > 0.5
    assert spread["rms_delta"] > 0.4
    # The two transcriptions agree only at the end of the common window.
    last = spread["rows"][-1]
    assert abs(last["delta"]) < 0.1
    # ... and disagree strongly in the middle of it.
    assert max(abs(r["delta"]) for r in spread["rows"]) > 0.5


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def test_all_three_structural_gates_fail_on_the_bundled_table(report):
    gates = report["gates"]
    assert gates["bundled_is_pointwise"]["result"] == "fail"
    assert gates["acceleration_phase_present"]["result"] == "fail"
    assert gates["claimed_uncertainty_consistent"]["result"] == "fail"
    assert gates["claimed_uncertainty_consistent"]["measured"] > 10.0


def test_verdict_withholds_physical_validation(report):
    assert report["verdict"]["physical_validation_status"] == "withheld"
    assert "does not support" in report["verdict"]["reason"]


def test_committed_audit_artifact_is_reproducible(report):
    """The JSON in the repo must equal what the code computes now."""
    path = HERE / "crosscheck_audit.json"
    assert path.exists(), "run benchmarks/dam_break/crosscheck_audit.py"
    on_disk = json.loads(path.read_text())
    for name, gate in report["gates"].items():
        assert on_disk["gates"][name]["result"] == gate["result"]
    fresh = report["cross_source_spread"]
    disk = on_disk["cross_source_spread"]
    for key in ("max_abs_delta", "rms_delta", "mean_signed_delta"):
        assert disk[key] == pytest.approx(fresh[key])
    assert disk["n_overlap"] == fresh["n_overlap"]
    assert on_disk["verdict"]["physical_validation_status"] == \
        report["verdict"]["physical_validation_status"]


# --------------------------------------------------------------------------- #
# the error metric itself
# --------------------------------------------------------------------------- #
def test_error_metric_is_zero_against_its_own_interpolation(crosscheck):
    """A curve that *is* the reference interpolant must measure as exact."""
    tau = [0.005 * i for i in range(int(crosscheck["tau"][-1] / 0.005) + 1)]
    za = [_interp(crosscheck["tau"], crosscheck["za"], t) for t in tau]
    err = error_vs(tau, za, crosscheck)
    assert err["n_points"] == len(crosscheck["tau"])
    assert err["mae"] == pytest.approx(0.0, abs=1e-12)
    assert err["rmse"] == pytest.approx(0.0, abs=1e-12)


def test_error_metric_detects_a_known_bias(crosscheck):
    """Add a constant offset: MAE must return exactly that offset."""
    tau = [0.005 * i for i in range(int(crosscheck["tau"][-1] / 0.005) + 1)]
    za = [_interp(crosscheck["tau"], crosscheck["za"], t) + 0.25 for t in tau]
    err = error_vs(tau, za, crosscheck)
    assert err["mae"] == pytest.approx(0.25, abs=1e-12)
    assert err["mean_signed"] == pytest.approx(0.25, abs=1e-12)


def test_gate_table_marks_the_limit_correctly():
    fake = [{"dx": 0.04, "particles": 50,
             "error_vs_crosscheck": {"mae": 0.10, "rmse": 0.11,
                                     "within_claimed_uncertainty": 0.5}},
            {"dx": 0.02, "particles": 200,
             "error_vs_crosscheck": {"mae": 0.20, "rmse": 0.21,
                                     "within_claimed_uncertainty": 0.3}}]
    table = reference_ladder.gate_table(fake, "crosscheck", 0.15)
    assert [r["result"] for r in table["rows"]] == ["pass", "fail"]
    assert table["finest"]["result"] == "fail"
    assert table["all_pass"] is False
