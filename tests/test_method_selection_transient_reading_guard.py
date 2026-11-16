"""Tests for ``benchmarks/method_selection/transient_reading_guard.py``.

The guard may only hand out ``transient_reading`` when the record's own grid
recovers inside the standard at a longer terminal time; every other shape must
keep the resolution/accuracy label alive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "method_selection"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import transient_reading_guard as trg  # noqa: E402


def _levels(pairs, admissible=()):
    return {float(vt): {"error_pct": float(pct), "admissible": float(vt) in admissible,
                        "steady_drift": 0.0, "protocol": f"{vt}vt",
                        "source": "synthetic"}
            for vt, pct in pairs.items()}


def test_declared_classes_are_the_documented_ones():
    assert set(trg.GUARD_CLASSES) == {"transient_reading", "reading_window_time",
                                      "insufficient_time_ladder"}
    for spec in trg.RECORDS:
        verdict = trg.classify_record(spec["extractor"])
        assert verdict is not None
        assert verdict["class"] in trg.GUARD_CLASSES


def test_couette_record_is_graded_transient_from_committed_artefacts():
    verdict = trg.classify_record("x_couette_wall_shear")
    # the record's own artefact is the --quick run on the coarsest grid
    assert verdict["record"]["resolution_H_over_dx"] == 8
    assert verdict["record"]["viscous_times"] == pytest.approx(0.3)
    assert verdict["record"]["error_pct"] > trg.OWNER_TOLERANCE_PCT
    assert verdict["class"] == "transient_reading"
    assert verdict["every_longer_level_below_record"] is True
    assert "1.2" in verdict["passing_levels"]
    # honest caveat: no passing level cleared the repository's steadiness gate
    assert verdict["passing_levels_steady"] == []
    assert "steadiness gate" in verdict["reason"]


def test_record_value_is_the_registry_value():
    """The guard reads the same artefact the adjudication banks -- no re-typing."""
    registry = json.loads((ROOT / "docs/reports/method_selection_registry.json").read_text())
    banked = {r["extractor"]: float(r["measured_pct"]) for r in registry["records"]}
    verdict = trg.classify_record("x_couette_wall_shear")
    assert verdict["record"]["error_pct"] == pytest.approx(banked["x_couette_wall_shear"],
                                                           rel=1e-9)


def test_rising_longer_levels_refute_the_transient_claim():
    levels = _levels({0.3: 32.6, 0.6: 22.6, 1.2: 33.0}, admissible=(1.2,))
    verdict = trg.classify(levels, 32.6, 0.3)
    assert verdict["class"] == "reading_window_time"
    assert verdict["every_longer_level_below_record"] is False
    assert "transient claim is refuted" in verdict["reason"]


def test_no_passing_level_is_not_a_transient_reading():
    levels = _levels({0.3: 32.6, 0.6: 22.6, 1.2: 18.0})
    verdict = trg.classify(levels, 32.6, 0.3)
    assert verdict["class"] == "reading_window_time"
    assert "inside 15%" in verdict["reason"]


def test_a_pass_needs_every_longer_level_below_the_record():
    levels = _levels({0.3: 32.6, 0.6: 22.6, 1.2: 4.0, 2.4: 33.0})
    verdict = trg.classify(levels, 32.6, 0.3)
    assert verdict["class"] == "reading_window_time"


def test_two_levels_are_not_a_time_ladder():
    verdict = trg.classify(_levels({0.3: 32.6, 1.2: 4.0}), 32.6, 0.3)
    assert verdict["class"] == "insufficient_time_ladder"


def test_shared_terminal_time_must_agree_across_artefacts(tmp_path):
    def write(name, value):
        doc = {"cells": [{"protocol": "design", "viscous_times": 1.2,
                          "resolution_H_over_dx": 8,
                          "wall_shear_error_pct_bottom": value,
                          "admissible": False, "steady_drift": 0.03}]}
        path = tmp_path / name
        path.write_text(json.dumps(doc))
        return str(path)

    first = trg.time_levels(write("a.json", 4.338367283344291), 8)
    assert first[1.2]["error_pct"] == pytest.approx(4.338367283344291)
    with pytest.raises(ValueError, match="disagrees across artefacts"):
        merged = dict(first)
        for vt, level in trg.time_levels(write("b.json", 9.0), 8).items():
            if vt in merged and abs(merged[vt]["error_pct"] - level["error_pct"]) > 1e-6:
                raise ValueError("shared terminal time disagrees across artefacts")
            merged[vt] = level


def test_matched_protocol_resolution_is_banked_when_the_matrix_exists():
    verdict = trg.classify_record("x_couette_wall_shear")
    ladder = verdict["matched_protocol_resolution"]
    if not ladder:                       # the H/dx = 32 matrix has not landed yet
        pytest.skip("protocol matrix not committed")
    for protocol, res in ladder.items():
        assert res["class"] in ("resolution_attributable", "resolution_window",
                                "resolution_pair_only")
        assert res["rungs"]
