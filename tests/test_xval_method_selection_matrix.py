"""The XVAL matrix driver must be a recomputation, and must not invent numbers.

Nothing here runs the solver: the two records are synthetic gauge series with a
*known* amplitude offset against the piston-transfer target, so the driver's error
column can be checked against a hand-derivable value.  The second test pins the
report artifact to those records, so a hand-edited digit in
``docs/reports/xval_method_selection_m17501.json`` breaks CI instead of review.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "benchmarks" / "xval_method_selection" / "grade_matrix.py"
REPORT = REPO / "docs" / "reports" / "xval_method_selection_m17501.json"
GRAVITY = 9.81


def _driver():
    spec = importlib.util.spec_from_file_location("xval_grade_matrix", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(height_scale=1.0, depth_cells=13, height_cells=2.08, period=1.0,
            depth=0.25, duration=10.0, gauge_x=(0.9, 1.1, 1.3, 1.5)):
    """Analytic wave record shaped like ``benchmarks/wave_flume/run.py`` output."""
    from tensorsph.wave_flume_grade import wave_height_target
    from tensorsph.wave_flume import wavenumber_from_period

    k = wavenumber_from_period(period, depth, GRAVITY)
    omega = 2.0 * math.pi / period
    spec = {"kind": "regular", "height": 0.16 * depth, "depth": depth,
            "gravity": GRAVITY, "period": period, "k": k, "omega": omega,
            "celerity": omega / k, "wavelength": 2.0 * math.pi / k,
            "length_scale": 2.0 * math.pi / k, "stroke": 0.1}
    target = wave_height_target(spec)
    dt = 0.004
    times = [i * dt for i in range(int(round(duration / dt)) + 1)]
    amp = 0.5 * target * height_scale
    probe_eta = [[amp * math.cos(k * x - omega * t) for t in times] for x in gauge_x]
    return {
        "case": "regular",
        "spec": spec,
        "config": {"depth_cells": depth_cells, "height_cells": height_cells,
                   "dx": depth / depth_cells, "particle_count": 2000,
                   "density_diffusion": 0.1, "settle_time": 1.0,
                   "viscosity_settle": 0.02, "xsph_settle": 0.1, "sound_speed": 30.0},
        "diagnostics": {"steps_taken": 2500, "dt": dt, "simulated_time": duration,
                        "max_speed": 1.0, "max_density_compression": 0.01,
                        "escaped": 0},
        "probe_x": list(gauge_x),
        "times": times,
        "probe_eta": probe_eta,
    }


def test_matrix_reports_the_known_height_offset_and_the_gap(tmp_path):
    """A +3 % record must read +3 %; an absent record must be a gap, not a zero."""
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "wf_D13_settle_d010.json").write_text(json.dumps(_record(height_scale=1.03)))
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps([
        ["D13_hn016_d010_settle", "wf_D13_settle_d010.json", "ladder", "ref A"],
        ["D16_hn016_d010_settle", "wf_D16_settle_d010.json", "ladder", "ref A"],
    ]))
    out = tmp_path / "report.json"
    figs = tmp_path / "figs"
    driver = _driver()
    assert driver.main(["--run-dir", str(runs), "--out", str(out),
                        "--figures", str(figs), "--matrix", str(matrix)]) == 0
    payload = json.loads(out.read_text())
    assert payload["counts"] == {"matrix": 2, "present": 1, "failed": 0,
                                 "within_owner_tol": 1}
    row = payload["rows"][0]
    assert row["present"] is True
    assert row["worst_error_vs_target"] == pytest.approx(0.03, abs=1e-3)
    assert all(g["error_vs_target"] == pytest.approx(0.03, abs=1e-3)
               for g in row["gauges"])
    missing = payload["rows"][1]
    assert missing["present"] is False
    assert "missing" in missing["note"]
    # the single-point matrix still produces its figures
    made = {f["name"]: f["made"] for f in payload["figures"]}
    assert made["ladder.png"] is True and made["sensitivity.png"] is True
    assert (figs / "ladder.png").stat().st_size > 1000
    # strictly valid JSON: a diverged run must be written as null, never NaN
    json.loads(out.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(
        AssertionError(f"non-finite constant {c} leaked into the report")))


def test_report_artifact_matches_a_fresh_regrade():
    """``docs/reports/xval_method_selection_m17501.json`` is a recomputation."""
    if not REPORT.exists():
        pytest.skip(f"report artifact missing: {REPORT}")
    payload = json.loads(REPORT.read_text())
    assert payload["anchor"] == "XVAL-SPH-METHOD-SELECTION-2026-11-12"
    for row in payload["rows"]:
        record = Path(row["record"])
        if not record.is_absolute():
            record = REPO / record
        if not row.get("present"):
            assert not record.exists(), f"{record} exists but the report calls it missing"
            continue
        if row.get("failed"):
            assert json.loads(record.read_text()).get("failed") is True
            continue
        fresh = payload["rows"][0] and json.loads(record.read_text())
        assert not fresh.get("failed"), row["label"]
        # the published target height must equal the piston-transfer value of the
        # record's own spec (no free parameter anywhere)
        from tensorsph.wave_flume_grade import wave_height_target
        assert row["target_height"] == pytest.approx(wave_height_target(fresh["spec"]),
                                                     rel=1e-12)
        for gauge, series in zip(row["gauges"], fresh["probe_eta"]):
            assert len(series) == len(fresh["times"])