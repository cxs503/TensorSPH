"""Gates for the four-level solenoidal analysis (``solenoid_scaling.py``).

All tests are offline: they read the shipped records and synthetic ladders, never
call a solver.  The point of the suite is that the *classifier* is bound to the
pre-registered gate threshold in the record -- not to a threshold chosen after
the numbers were seen -- and that the analysis cannot mutate its input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "laplace_sph"))

import solenoid_scaling as S  # noqa: E402


def test_fit_recovers_a_known_exponent():
    x = [8.0, 12.0, 16.0, 24.0]
    y = [0.05 * xx ** 0.75 for xx in x]
    fit = S.fit_power_law(x, y)
    assert abs(fit["exponent"] - 0.75) < 1e-9
    assert abs(fit["decay_order"] + 0.75) < 1e-9
    assert fit["rmse_rel"] < 1e-12


def test_classifier_is_bound_to_the_declared_gate_not_to_the_data():
    """A decaying ladder passes; the shipped ladder fails -- same code path."""
    dec = S.fit_power_law([8.0, 12.0, 16.0, 24.0],
                          [0.40 * (8.0 / xx) ** 0.7 for xx in (8.0, 12.0, 16.0, 24.0)])
    assert S.classify(dec["decay_order"], 0.5) is True
    rec = S.load_levels(S.SOLENOID)
    order = rec["orders"]["solenoidal_fraction_decay_order"]
    assert S.classify(order, rec["declared_order_min"]) is False


def test_shipped_record_has_four_levels_and_the_merge_is_not_hand_typed():
    rec = S.load_levels(S.SOLENOID)
    assert rec["x"] == [8.0, 12.0, 16.0, 24.0]
    assert rec["declared_order_min"] == 0.5          # the pre-registered gate
    # every level record exists as its own file (per-level provenance)
    for n in (16, 24):
        lv = S.LADDERDIR / "solenoid_levels" / ("Rdx%d.json" % n)
        assert lv.exists()
        assert abs(json.loads(lv.read_text())["R_over_dx"] - n) < 1e-12


def test_holdout_prediction_is_recomputed_from_the_record_only():
    rec = S.load_levels(S.SOLENOID)
    ho = S.holdout_power_law(rec["x"], rec["fraction"])
    assert ho["held_out_level"] == 24.0
    # predicted/measured relationship checked against the raw record numbers
    assert ho["measured"] == pytest.approx(rec["fraction"][-1])
    assert ho["rel_error_pct"] == pytest.approx(
        100.0 * abs(ho["predicted"] - ho["measured"]) / ho["measured"])
    # the best-fitting power law overshoots the withheld level: growth decelerates
    assert ho["predicted"] > ho["measured"]


def test_analysis_does_not_mutate_its_input_record():
    before = S.SOLENOID.read_bytes()
    res = S.analyse()
    assert S.SOLENOID.read_bytes() == before
    assert res["monotone_increasing"] is True
    assert res["verdict"]["pass"] is False
    # the Laplace-jump acceptance is answered from records, and it passes
    assert res["young_laplace_all_under_15pct"] is True
    assert res["young_laplace_all_under_3pct"] is True
    assert res["young_laplace_max_error_pct"] < 3.0