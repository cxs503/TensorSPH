"""Artifact-level checks for the case-pool entry ``CASE-WAVE-FLUME``.

Two things are asserted here, and they are different in kind:

1. **The committed numbers are a recomputation, not prose.**  The report
   (``docs/reports/wave_flume_report.json``) and the numbers bundle
   (``docs/reports/wave_flume_numbers.json``) are regenerated from the raw gauge
   records in ``runs/`` and compared field by field, so editing a digit by hand
   breaks CI instead of the review.
2. **The acceptance criterion is evaluated on the real solver records.**  The
   primary production record (``H/dx = 3.2``, ``T = 1 s``, ``H/h = 0.16``) must
   put every gauge inside the 10 % height gate; a red test here means the case
   did *not* meet its case-pool entry, which is a reportable FAIL and not
   something to be smoothed over.

The records are the JSON written by ``benchmarks/wave_flume/run.py``; the grading
lives in :mod:`tensorsph.wave_flume_grade` and is unit-tested on synthetic signals
in ``tests/test_wave_flume_grade.py``.  No solver runs in this file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorsph.wave_flume_grade import grade_regular_record

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
REPORT = REPO / "docs" / "reports" / "wave_flume_report.json"
NUMBERS = REPO / "docs" / "reports" / "wave_flume_numbers.json"
DOC = REPO / "docs" / "reports" / "wave_flume_case.md"
TOLERANCE = 0.10


def _records():
    return sorted(RUNS.glob("wf_*.json"))


def _report():
    if not REPORT.exists():
        pytest.skip(f"report artifact missing: {REPORT}")
    return json.loads(REPORT.read_text())


def test_committed_records_are_completed_solver_runs():
    """A committed record must be a finished solver run, not a stub."""
    paths = _records()
    assert paths, f"no solver record under {RUNS}; run benchmarks/wave_flume/run.py"
    for path in paths:
        record = json.loads(path.read_text())
        assert record.get("case") == "regular", path
        assert not record.get("failed"), f"{path}: {record.get('error')}"
        diag = record["diagnostics"]
        assert diag["steps_taken"] > 20000, (path, diag["steps_taken"])
        assert diag["simulated_time"] > 3.0, (path, diag["simulated_time"])
        assert len(record["probe_x"]) >= 3, (path, record["probe_x"])
        assert len(record["times"]) == len(record["probe_eta"][0]), path
        finite = [sum(math.isfinite(v) for v in eta) for eta in record["probe_eta"]]
        assert min(finite) > 0.9 * len(record["times"]), (path, finite)


def test_report_is_a_recomputation_of_the_records():
    """Every published height/attenuation number must come from the records."""
    report = _report()
    tolerance = float(report["tolerance"])
    assert abs(tolerance - TOLERANCE) < 1e-12
    for case in report["cases"]:
        record_path = Path(case["record"])
        if not record_path.is_absolute():
            record_path = REPO / record_path
        assert record_path.exists(), record_path
        start, end = case["window"]
        fresh = grade_regular_record(json.loads(record_path.read_text()),
                                     start=start, end=end, tolerance=tolerance)
        assert fresh["passed"] == case["passed"], case["record"]
        assert abs(fresh["target_height"] - case["target_height"]) < 1e-12
        assert abs(fresh["worst_error_vs_target"] - case["worst_error_vs_target"]) < 1e-12
        assert abs(fresh["attenuation"]["mu_per_m"] - case["attenuation"]["mu_per_m"]) < 1e-12
        for got, ref in zip(case["gauges"], fresh["gauges"]):
            assert got["x"] == pytest.approx(ref["x"], abs=1e-12)
            assert got["height"] == pytest.approx(ref["height"], rel=1e-9, abs=1e-12)
            assert got["error_vs_target"] == pytest.approx(ref["error_vs_target"],
                                                           rel=1e-9, abs=1e-12)


def test_numbers_bundle_matches_the_report():
    """``numbers.json`` is what the write-up quotes; it may not drift."""
    report = _report()
    if not NUMBERS.exists():
        pytest.skip(f"numbers bundle missing: {NUMBERS}")
    numbers = json.loads(NUMBERS.read_text())
    assert numbers["criterion"].startswith("CASE-WAVE-FLUME")
    assert abs(numbers["tolerance"] - TOLERANCE) < 1e-12
    assert numbers["cases_graded"] == len([c for c in report["cases"] if not c["failed"]])
    worst = max(c["worst_error_vs_target"] for c in report["cases"] if not c["failed"])
    assert numbers["worst_error_vs_target"] == pytest.approx(worst, rel=1e-12)
    by_record = {c["record"]: c for c in numbers["cases"]}
    for case in report["cases"]:
        published = by_record[case["record"]]
        assert published["target_height"] == pytest.approx(case["target_height"], rel=1e-12)
        assert published["attenuation"]["mu_per_m"] == pytest.approx(
            case["attenuation"]["mu_per_m"], rel=1e-9, abs=1e-18)
        for got, ref in zip(published["gauges"], case["gauges"]):
            assert got["height"] == pytest.approx(ref["height"], rel=1e-9, abs=1e-12)


def test_primary_case_meets_the_ten_percent_height_gate():
    """The case-pool entry itself: H within 10 % of the piston-transfer target."""
    report = _report()
    primary = [c for c in report["cases"]
               if (c.get("spec") or {}).get("period") == pytest.approx(1.0)
               and (c.get("spec") or {}).get("height", 0.0)
               / (c.get("spec") or {}).get("depth", 1.0) == pytest.approx(0.16, abs=0.005)]
    assert primary, "the primary production case (T = 1 s, H/h = 0.16) is not in the report"
    for case in primary:
        for gauge in case["gauges"]:
            err = gauge["error_vs_target"]
            assert abs(err) <= TOLERANCE, (
                f"{case['record']} gauge x={gauge['x']:.3f} m: "
                f"H_got={gauge['height']:.4f} m vs H_ref={case['target_height']:.4f} m "
                f"-> {100 * err:+.2f} % (gate {100 * TOLERANCE:.0f} %)")


def test_write_up_quotes_the_repro_command_and_the_error_table():
    """The report must carry the repro command and ref-vs-got rows, or it is prose."""
    if not DOC.exists():
        pytest.skip(f"report prose missing: {DOC}")
    text = DOC.read_text()
    assert "benchmarks/wave_flume/run.py" in text
    assert "benchmarks/wave_flume/analyse.py" in text
    assert "H_ref" in text and "H_got" in text and "误差" in text
    assert "打勾" in text or "打叉" in text
    assert "μ_theory" in text