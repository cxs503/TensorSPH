"""Regression tests pinned to the committed free-surface study.

``docs/benchmarks/free-surface/report.json`` is the audit trail: it holds every
number the verification page quotes. These tests make three promises about it:

* the verdict can be recomputed from the measured numbers and the criterion
  table (nobody can flip a status in the JSON);
* the honest failures (C1 front position, C2 ten-wavelength decay) are *present*
  in the report -- a later change that quietly drops them would turn this file
  red instead of turning the report optimistic;
* the numbers stay put. They come from a recorded run; a code change that moves
  them has to move them in the report (and in
  ``docs/verification_free_surface.md``) as well.
"""
import json
import math
from pathlib import Path

import pytest

from tensorsph.freesurface import gates

REPORT_PATH = (Path(__file__).resolve().parents[1]
               / "docs" / "benchmarks" / "free-surface" / "report.json")


@pytest.fixture(scope="module")
def report():
    assert REPORT_PATH.is_file(), (
        f"{REPORT_PATH} is missing; run scripts/run_freesurface_verification.py "
        "and commit the report")
    return json.loads(REPORT_PATH.read_text())


def test_the_study_completed_every_part(report):
    parts = report["parts"]
    assert set(parts) >= {"ritter", "ladder", "sloshing", "solitary"}
    for name in ("ritter", "ladder", "sloshing", "solitary"):
        assert report[name] is not None, f"{name} part missing from the report"


def test_every_gate_status_follows_from_its_measured_value(report):
    rescored = gates.evaluate(report["measured"])
    recorded = {row["id"]: row["status"] for row in report["gates"]}
    for row in rescored["gates"]:
        assert row["status"] == recorded[row["id"]], row["id"]


def test_the_report_keeps_the_honest_failures(report):
    statuses = {row["id"]: row["status"] for row in report["gates"]}
    assert statuses["C3"] == "PASS"       # sloshing frequency
    assert statuses["C4"] == "PASS"       # convergence slope
    assert statuses["C5"] == "PASS"       # mass conservation
    assert statuses["C1"] == "FAIL"       # Ritter front position
    assert statuses["C2"] == "FAIL"       # ten-wavelength decay
    assert report["maturity"]["verdict"] == "HONEST_FAIL_OPEN"
    assert set(report["maturity"]["failed"]) == {"C1", "C2"}


def test_ritter_numbers_are_pinned(report):
    ritter = report["ritter"]
    assert ritter["depth0"] == pytest.approx(0.10)
    assert ritter["column_width"] == pytest.approx(0.50)
    assert ritter["dx"] == pytest.approx(0.01)
    assert ritter["depth_l2"] == pytest.approx(0.0760, abs=0.01)
    assert ritter["depth_l2_resolved"] == pytest.approx(0.0994, abs=0.01)
    assert ritter["velocity_l2"] == pytest.approx(0.1000, abs=0.015)
    assert ritter["front_error_max"] == pytest.approx(0.1404, abs=0.02)
    # the zero-depth front is much worse: that gap is the unresolvable film
    assert ritter["front_error_zero_depth"] == pytest.approx(0.4703, abs=0.05)
    assert ritter["released_volume_error"] == pytest.approx(0.1146, abs=0.02)
    # mass is a code invariant, not an accuracy claim
    assert ritter["mass_drift_relative"] == 0.0
    assert ritter["particles_escaped"] == 0


def test_the_ladder_converges_and_is_monotone_in_the_resolved_norm(report):
    ladder = report["ladder"]
    resolved = ladder["depth_l2_resolved"]
    assert resolved == sorted(resolved, reverse=True)
    assert ladder["spacings"] == sorted(ladder["spacings"], reverse=True)
    slope = report["convergence"]["slope_resolved"]
    assert slope == pytest.approx(0.847, abs=0.05)
    assert 0.8 <= slope <= 1.2
    # the full-fan norm converges more slowly; both are recorded
    assert report["convergence"]["slope_full_fan"] < slope


def test_sloshing_frequency_is_pinned_and_passing(report):
    sloshing = report["sloshing"]
    assert sloshing["omega_analytic"] == pytest.approx(6.6619, abs=1e-3)
    assert sloshing["omega_measured"] == pytest.approx(6.5901, abs=0.05)
    assert sloshing["frequency_error"] == pytest.approx(0.0108, abs=0.01)
    assert sloshing["frequency_error"] <= 0.10
    # the damping is numerical and is reported, not used as a pass criterion
    assert 0.0 < sloshing["decay_per_period"] < 0.5
    assert sloshing["mass_drift"] == 0.0


def test_solitary_celerity_and_shape_are_pinned(report):
    solitary = report["solitary"]
    assert solitary["celerity_error"] == pytest.approx(0.0081, abs=0.005)
    assert solitary["celerity_error"] <= 0.05
    assert solitary["shape_l2"] == pytest.approx(0.264, abs=0.05)
    assert solitary["shape_correlation"] >= 0.85
    crests = [gauge["crest_height"] for gauge in solitary["gauges"]]
    assert crests == sorted(crests, reverse=True)     # the crest decays downstream


def test_the_ten_wavelength_decay_is_an_extrapolation_and_says_so(report):
    solitary = report["solitary"]
    assert solitary["distance_in_wavelengths"] < 1.0   # the flume is shorter than 1 λ
    assert "decay_10_lambda_extrapolated" in solitary
    # the measured loss over the traversed distance already exceeds the gate
    assert 1.0 - solitary["amplitude_ratio"] > 0.05
    assert solitary["decay_10_lambda_extrapolated"] > 0.05
    assert report["measured"]["C2"] == solitary["decay_10_lambda_extrapolated"]


def test_the_figures_and_the_cross_case_comparison_are_committed(report):
    root = REPORT_PATH.parent
    names = {Path(p).name for p in report["figures"]}
    assert {"ritter_profiles.png", "ritter_convergence.png", "sloshing_mode.png",
            "solitary_wave.png", "summary_gates.png"} <= names
    for relative in report["figures"]:
        path = root / relative
        assert path.is_file(), relative
        assert path.stat().st_size > 5000, relative


def test_the_mass_drift_is_reported_per_case(report):
    drifts = report["mass_drift_by_case"]
    assert set(drifts) == {"ritter", "sloshing", "solitary"}
    for name, value in drifts.items():
        assert math.isfinite(value)
        assert abs(value) <= 1.0e-3, name
