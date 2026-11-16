"""Guards for the resolution-ladder ruling of an over-tolerance record (m22957).

m22787 called an over-tolerance record ``resolution_attributable`` when *one*
finer rung was inside the owner standard.  One pair is not convergence: the
transport-velocity record's own committed four-level ladder is non-monotone and
its finest rung is back above 15 %.  These tests pin the three outcomes of
``benchmarks/method_selection/resolution_ladder_guard.py`` and the propagation
of the ruling into the adjudication artefacts.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "method_selection"
sys.path.insert(0, str(BENCH))

import resolution_ladder_guard as rlg  # noqa: E402

TOL = rlg.OWNER_TOLERANCE_PCT


@pytest.fixture(scope="module")
def adjudication() -> dict:
    return json.loads(
        (ROOT / "docs/reports/method_selection_open_flags_adjudication_m22787.json")
        .read_text())


def test_monotone_three_rung_ladder_is_attributable():
    verdict = rlg.classify({0.04: 30.0, 0.02: 20.0, 0.01: 9.0}, "decreasing")
    assert verdict["class"] == "resolution_attributable"
    assert verdict["monotone"] is True
    assert verdict["finest_within_standard"] is True


def test_reversed_ladder_is_a_window_not_attributable():
    """Negative control: finer than the passing rung the error comes back."""
    verdict = rlg.classify({0.04: 17.1, 0.03: 10.6, 0.02: 12.5, 0.01: 16.4},
                           "decreasing")
    assert verdict["class"] == "resolution_window"
    assert verdict["monotone"] is False
    assert verdict["finest_within_standard"] is False
    assert verdict["window"] == {"min": 0.02, "max": 0.03}
    assert len(verdict["reversals"]) == 2


def test_two_rung_ladder_is_pair_only():
    verdict = rlg.classify({8.0: 32.0, 16.0: 3.0}, "increasing")
    assert verdict["class"] == "resolution_pair_only"
    assert verdict["n_rungs"] == 2


def test_finest_rung_failing_alone_is_a_window():
    verdict = rlg.classify({0.04: 40.0, 0.02: 18.0, 0.01: 16.0}, "decreasing")
    assert verdict["class"] == "resolution_window"
    assert verdict["monotone"] is True          # monotone but never passes
    assert verdict["window"] is None


def test_increasing_direction_is_honoured():
    fine_first = rlg.classify({8.0: 20.0, 16.0: 9.0, 32.0: 14.0}, "increasing")
    assert fine_first["class"] == "resolution_window"
    assert fine_first["reversals"][0]["coarse_rung"] == 16.0


def test_unknown_direction_is_refused():
    with pytest.raises(ValueError):
        rlg.classify({1.0: 2.0}, "sideways")


def test_guard_reads_the_committed_tvc_ladder():
    """The four rungs are recomputed from the artefact, not typed in."""
    verdict = rlg.classify_method("transport_velocity_correction")
    assert verdict is not None
    ladder = json.loads(
        (ROOT / "docs/xval_sphinx_tvc/data/ladder_dense.json").read_text())
    n_tvc = sum(1 for r in ladder["records"] if r["variant"] == rlg.TVC_VARIANT)
    assert verdict["n_rungs"] == n_tvc
    assert verdict["class"] == "resolution_window"
    assert verdict["finest_pct"] > TOL
    rows = {f"{float(r['dx']):g}": r for r in ladder["records"]
            if r["variant"] == rlg.TVC_VARIANT}
    for rung, pct in verdict["rungs"].items():
        rel = [abs(float(x["rel_error"])) for x in rows[rung]["score"]["rows"]
               if x["rel_error"] is not None]
        assert math.isclose(pct, sum(rel) / len(rel) * 100.0, rel_tol=1e-9)


def test_guard_reads_the_committed_couette_ladder():
    verdict = rlg.classify_method("wcsph_wall_no_slip")
    assert verdict is not None
    assert verdict["class"] == "resolution_pair_only"
    design = json.loads(
        (ROOT / "benchmarks/couette/out_design/results.json").read_text())
    assert verdict["n_rungs"] == len(design["ladder"])


def test_transport_velocity_record_is_no_longer_called_attributable(adjudication):
    entry = next(a for a in adjudication["adjudications"]
                 if a["extractor"] == "x_transport_velocity_coarse")
    assert entry["class"] == "resolution_window"
    guard = entry["ladder_guard"]
    assert guard["finest_rung"] == float(min(guard["rungs"], key=float))
    assert guard["window"] == {"min": 0.02, "max": 0.03}
    # the verdict is carried in its own channel, so the md/json cannot read as a pass
    assert entry["problems"] == []
    assert any("resolution_window" in d for d in entry["downgrades"])


def test_couette_wall_shear_record_is_a_transient_reading(adjudication):
    """The resolution label was never the cause: the same grid recovers."""
    entry = next(a for a in adjudication["adjudications"]
                 if a["extractor"] == "x_couette_wall_shear")
    # the grid verdict is still banked, unhidden...
    assert entry["ladder_guard"]["n_rungs"] < rlg.MIN_RUNGS
    assert entry["ladder_guard"]["class"] == "resolution_pair_only"
    # ...but it does not carry the record: the terminal-time axis does
    assert entry["class"] == "transient_reading"
    guard = entry["transient_guard"]
    assert guard["resolution_H_over_dx"] == guard["record"]["resolution_H_over_dx"]
    assert guard["every_longer_level_below_record"] is True
    assert guard["passing_levels"]
    assert any("transient_reading overrides resolution_pair_only" in d
               for d in entry["downgrades"])


def test_no_record_is_called_attributable_without_a_clean_ladder(adjudication):
    for entry in adjudication["adjudications"]:
        if entry["class"] != "resolution_attributable":
            continue
        guard = entry.get("ladder_guard")
        assert guard is not None
        assert guard["monotone"] is True
        assert guard["finest_within_standard"] is True


def test_guard_summary_is_banked_in_the_artefact(adjudication):
    summary = adjudication["resolution_ladder_guard"]
    for spec in rlg.LADDERS:
        assert spec["method"] in summary
        verdict = summary[spec["method"]]
        assert verdict["class"] in rlg.GUARD_CLASSES
        assert verdict["sources"], "a graded method must name its artefacts"


def test_producer_is_a_fresh_build():
    proc = subprocess.run(
        [sys.executable, str(BENCH / "adjudicate_over_tolerance.py"), "--check"],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
