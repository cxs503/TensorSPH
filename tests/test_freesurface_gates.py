"""Tests for the acceptance-gate scoring.

The gates are the only place a threshold lives, so they get their own tests: a
gate that can be talked into PASS is worse than no gate at all.
"""
import math

import pytest

from tensorsph.freesurface import gates


def test_the_criterion_table_covers_the_maturity_node():
    ids = [c.identifier for c in gates.CRITERIA]
    assert ids == ["C1", "C2", "C3", "C4", "C5"]
    thresholds = {c.identifier: c.threshold for c in gates.CRITERIA}
    assert thresholds["C1"] == pytest.approx(0.10)     # Ritter front
    assert thresholds["C2"] == pytest.approx(0.05)     # solitary decay
    assert thresholds["C3"] == pytest.approx(0.10)     # sloshing frequency
    assert thresholds["C4"] == pytest.approx(1.00)     # convergence slope
    assert thresholds["C5"] == pytest.approx(0.01)     # mass


def test_a_gate_passes_at_the_threshold_and_fails_just_above_it():
    at = gates.evaluate({"C1": 0.10}, criteria=(gates.CRITERIA[0],))
    above = gates.evaluate({"C1": 0.1001}, criteria=(gates.CRITERIA[0],))
    below = gates.evaluate({"C1": 0.0}, criteria=(gates.CRITERIA[0],))
    assert at["gates"][0]["status"] == "PASS"
    assert above["gates"][0]["status"] == "FAIL"
    assert below["gates"][0]["status"] == "PASS"


def test_a_missing_or_nan_measurement_is_unmeasured_not_pass():
    report = gates.evaluate({"C1": float("nan")}, criteria=(gates.CRITERIA[0],))
    assert report["gates"][0]["status"] == "UNMEASURED"
    report = gates.evaluate({}, criteria=(gates.CRITERIA[0],))
    assert report["gates"][0]["status"] == "UNMEASURED"
    verdict = gates.maturity(report)
    assert verdict["verdict"] == "HONEST_FAIL_OPEN"
    assert "C1" in verdict["failed"]


def test_the_convergence_gate_is_a_band_around_first_order():
    criterion = (gates.CRITERIA[3],)
    for value, expected in ((1.0, "PASS"), (0.8, "PASS"), (1.2, "PASS"),
                            (1.21, "FAIL"), (0.79, "FAIL"), (2.0, "FAIL")):
        report = gates.evaluate({"C4": value}, criteria=criterion)
        assert report["gates"][0]["status"] == expected, value


def test_maturity_needs_every_gate_and_lists_the_open_ones():
    passing = {"C1": 0.02, "C2": 0.01, "C3": 0.05, "C4": 0.95, "C5": 0.002}
    report = gates.evaluate(passing)
    verdict = gates.maturity(report)
    assert verdict["verdict"] == "PASS"
    assert verdict["failed"] == []

    failing = dict(passing, C2=0.30)
    verdict = gates.maturity(gates.evaluate(failing))
    assert verdict["verdict"] == "HONEST_FAIL_OPEN"
    assert verdict["failed"] == ["C2"]
    assert any(line.startswith("C2") for line in verdict["open_items"])


def test_notes_are_carried_into_the_report():
    report = gates.evaluate({"C1": 0.5}, notes={"C1": "front smeared over 4 spacings"})
    assert report["gates"][0]["note"] == "front smeared over 4 spacings"
    assert math.isfinite(report["gates"][0]["measured"])
