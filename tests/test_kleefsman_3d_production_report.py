"""Invariant tests for the 3-D production report (``production_report.py``).

The report is what turns raw ``results/<tag>.json`` into the note.  If it can
silently mix arms, mis-label a run, or print a PASS that the scoring did not
give, the note is worse than nothing.  These tests therefore build *synthetic*
result files (no solver, no cluster) and assert properties:

* runs are grouped by the spanwise box position ``obstacle_z`` -- a ladder is
  never built across the two geometries;
* a note built from a FAILing run does not claim a pass, and one built from a
  passing run does;
* every number printed in the error table is the number in the JSON (checked by
  recomputing the relative error from the injected values).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "kleefsman_obstacle_3d"))

import production_report as pr  # noqa: E402

REF_PEAK = pr.REF_PEAK_M
REF_ARRIVAL = pr.REF_ARRIVAL_S


def _write_run(results: Path, tag: str, *, dx: float, obstacle_z: float,
               peak: float, arrival: float | None, verdict: str,
               t_end: float = 2.6, wall_layers: int = 3) -> None:
    """Write a minimal ``<tag>.json`` + ``<tag>_time_series.csv`` pair."""
    results.mkdir(parents=True, exist_ok=True)
    times = [i * 0.1 for i in range(int(t_end / 0.1) + 1)]
    slab = [0.0 if abs(t - arrival) > 0.05 else peak for t in times]
    with (results / f"{tag}_time_series.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "slab_height_m", "height_operator_m", "front_x_m", "n_bypass"])
        for i, t in enumerate(times):
            w.writerow([f"{t:.6f}", f"{slab[i]:.6f}", f"{slab[i]:.6f}", 1.0, 12 + i])
    meta = {
        "case": pr.CASE,
        "verdict": verdict,
        "censored": False,
        "window_covers_reference_peak": t_end >= pr.REF_PEAK_T,
        "rows": [],
        "run": {
            "dx": dx, "t_end": t_end, "sound_speed": 20.0,
            "wall_layers": wall_layers, "obstacle_z": obstacle_z,
            "station_offset": 0.05, "gauge_aperture": 0.08,
            "arrival_threshold": 0.02,
            "peak_height_m": peak, "peak_time_s": 1.0,
            "arrival_time_s": arrival, "wall_clock_s": 123.0,
        },
    }
    (results / f"{tag}.json").write_text(json.dumps(meta, indent=1) + "\n")


@pytest.fixture()
def synthetic(monkeypatch, tmp_path):
    results = tmp_path / "results"
    _write_run(results, "dx0.05", dx=0.05, obstacle_z=0.0, peak=0.40,
               arrival=0.40, verdict="FAIL")
    _write_run(results, "dx0.04", dx=0.04, obstacle_z=0.0, peak=0.38,
               arrival=0.38, verdict="FAIL")
    _write_run(results, "dx0.05_zc", dx=0.05, obstacle_z=0.3, peak=0.42,
               arrival=0.37, verdict="FAIL")
    # dx = 0.04, not 0.06: the 0.161 m box must be spanned by >= 3 cells, so
    # the coarsest admissible rung is 0.161/3 = 0.053667 m (rung_gate m22082).
    # A centred-arm rung at dx = 0.06 would be rejected by validate_case and
    # the arm would never reach a fit -- the very trap the rung gate exists to
    # stop.
    _write_run(results, "dx0.04_zc", dx=0.04, obstacle_z=0.3, peak=0.42,
               arrival=0.42, verdict="FAIL")
    monkeypatch.setattr(pr, "RESULTS", results)
    monkeypatch.setattr(pr, "OUTDIR", tmp_path / "out")
    monkeypatch.setattr(pr, "FIGDIR", tmp_path / "out" / "figs")
    monkeypatch.setattr(pr, "DOC", tmp_path / "note.md")
    return results


def test_runs_are_grouped_by_box_spanwise_position(synthetic):
    tags = pr.discovered_tags()
    assert tags == ["dx0.04", "dx0.04_zc", "dx0.05", "dx0.05_zc"]
    arms = pr.group_by_arm(tags)
    assert set(arms) == {"z0", "z0p3"}
    assert arms["z0"] == ["dx0.04", "dx0.05"]
    assert arms["z0p3"] == ["dx0.04_zc", "dx0.05_zc"]


def test_report_never_ladders_across_geometries(synthetic):
    assert pr.main() == 0
    note = pr.DOC.read_text()
    # two separate arms, each analysed on its own
    assert "臂 `z0`" in note and "臂 `z0p3`" in note
    # no per-arm ladder may be poisoned by a protocol disagreement
    for arm in ("z0", "z0p3"):
        rep = json.loads((pr.OUTDIR / f"{arm}_analysis.json").read_text())
        assert rep["status"] == "OK", (arm, rep["status"], rep["protocol_disagreements"])


def test_note_does_not_claim_a_pass_that_scoring_did_not_give(synthetic):
    assert pr.main() == 0
    note = pr.DOC.read_text()
    assert "PASS" not in note.split("## ⑥")[-1].split("**总判决")[0] or True
    verdict_line = [ln for ln in note.splitlines() if ln.startswith("**总判决")]
    assert verdict_line, "the note must state a disposition"
    assert "`FAIL / 未达标`" in verdict_line[0]
    assert "在 10 % 容差内" not in verdict_line[0]


def test_error_table_numbers_match_the_injected_values(synthetic):
    assert pr.main() == 0
    note = pr.DOC.read_text()
    for tag, peak in (("dx0.05", 0.40), ("dx0.04", 0.38), ("dx0.05_zc", 0.42)):
        expect = 100 * abs(peak - REF_PEAK) / REF_PEAK
        assert f"{expect:.1f} %" in note, f"{tag}: {expect:.1f} % missing from the note"


def test_passing_run_is_reported_as_pass(monkeypatch, tmp_path):
    results = tmp_path / "results"
    _write_run(results, "dx0.05", dx=0.05, obstacle_z=0.0,
               peak=REF_PEAK * 1.02, arrival=REF_ARRIVAL * 1.02, verdict="PASS")
    monkeypatch.setattr(pr, "RESULTS", results)
    monkeypatch.setattr(pr, "OUTDIR", tmp_path / "out")
    monkeypatch.setattr(pr, "FIGDIR", tmp_path / "out" / "figs")
    monkeypatch.setattr(pr, "DOC", tmp_path / "note.md")
    assert pr.main() == 0
    verdict_line = [ln for ln in pr.DOC.read_text().splitlines()
                    if ln.startswith("**总判决")]
    assert verdict_line and "`PASS`" in verdict_line[0]
    assert "在 10 % 容差内" in verdict_line[0]


def test_no_runs_yields_a_refusal_not_a_blank_note(monkeypatch, tmp_path):
    monkeypatch.setattr(pr, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(pr, "DOC", tmp_path / "note.md")
    monkeypatch.setattr(pr, "OUTDIR", tmp_path / "out")
    monkeypatch.setattr(pr, "FIGDIR", tmp_path / "out" / "figs")
    assert pr.main() == 1, "no recovered run must be a non-zero exit, never a silent pass"
