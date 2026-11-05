"""Regression gate on the *measured* effect of the free-surface pressure closure.

``benchmarks/laplace_pressure_floor/report_closure.json`` records what the
closure bought and what it cost (see
``docs/verification_free_surface_pressure_closure.md``).  The physics runs take
minutes, so they are not repeated here; instead these assertions pin the
committed measurement -- the numbers, the comparison against the clamp baseline
in ``report.json``, and the document that quotes them -- so that a change in the
closure, the EOS or the read-out cannot silently move the published claims.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure_floor"
ROOT = BENCH.parents[1]
CLOSURE = BENCH / "report_closure.json"
CLAMP = BENCH / "report.json"
DOC = ROOT / "docs" / "verification_free_surface_pressure_closure.md"
PNG = BENCH / "free_surface_pressure_closure.png"

# The two points actually measured (dx [m], c0 [m/s]) and the tail-averaged
# error the closure reached there.  Pinned values are the record, not physics:
# a legitimate improvement must update the JSON, the document and this table.
MEASURED = {
    (0.002, 20.0): 8.8,
    (0.002, 5.0): 4.9,
}


def _load(path):
    if not path.exists():
        pytest.skip(f"{path.name} missing: run the benchmark first")
    return json.loads(path.read_text())


def _runs():
    return {(r["dx_m"], r["sound_speed"]): r for r in _load(CLOSURE)["runs"]}


def test_the_report_records_the_shell_closure():
    report = _load(CLOSURE)
    assert report["closure"] == "shell"
    assert len(report["runs"]) >= 2
    for run in report["runs"]:
        assert run["finite"], (run["dx_m"], run["sound_speed"])
        assert run["closure"] == "shell"
        assert run["closure_donors"] == "interior"
        assert run["closure_layers"] == 2
        assert run["closure_threshold"] == pytest.approx(0.99)


def test_every_measured_point_is_pinned():
    runs = _runs()
    for key, expected in MEASURED.items():
        assert key in runs, key
        got = runs[key]["absolute_time_average_tail"]["rel_error_pct"]
        assert got == pytest.approx(expected, abs=0.25), (key, got)


def test_closure_beats_the_clamp_at_every_recorded_point():
    runs = _runs()
    clamp = {(r["dx_m"], r["sound_speed"]): r for r in _load(CLAMP)["runs"]}
    improved = 0
    for key, run in runs.items():
        if key not in clamp:
            continue
        before = clamp[key]["absolute_time_average_tail"]["rel_error_pct"]
        after = run["absolute_time_average_tail"]["rel_error_pct"]
        assert abs(after) < 0.5 * abs(before), (key, before, after)
        improved += 1
    assert improved >= 2


def test_at_least_one_point_meets_the_platform_15pct_bar():
    worst = min(abs(r["absolute_time_average_tail"]["rel_error_pct"])
                for r in _load(CLOSURE)["runs"])
    assert worst < 15.0, worst


def test_the_3pct_bar_is_not_claimed():
    """Honest record: the closure did *not* reach the strict 3 % standard."""
    errors = [abs(r["absolute_time_average_tail"]["rel_error_pct"])
              for r in _load(CLOSURE)["runs"]]
    assert min(errors) > 3.0, errors


def test_spurious_currents_are_recorded_as_a_regression():
    """The closure raised the shell currents; that cost must stay visible."""
    runs = _runs()
    clamp = {(r["dx_m"], r["sound_speed"]): r for r in _load(CLAMP)["runs"]}
    for key, run in runs.items():
        if key not in clamp:
            continue
        after = run["spurious_current"]["vmax_over_vref_final"]
        before = clamp[key]["spurious_current"]["vmax_over_vref_final"]
        assert after > before, (key, before, after)
    assert max(r["spurious_current"]["vmax_over_vref_final"]
               for r in runs.values()) > 0.1


def test_the_document_quotes_the_measured_numbers():
    assert DOC.exists(), "make_closure_doc.py writes the record"
    assert PNG.exists(), "make_closure_doc.py writes the figure"
    doc = DOC.read_text()
    for run in _load(CLOSURE)["runs"]:
        tail = run["absolute_time_average_tail"]["rel_error_pct"]
        assert f"{tail:+.1f} %" in doc, tail
    assert "surface_pressure_closure" in doc


def test_clamp_baseline_report_is_the_default_path():
    """``report.json`` must stay the closure-off baseline it has always been."""
    report = _load(CLAMP)
    assert report.get("closure", "clamp") == "clamp"
    assert all(run.get("closure", "clamp") == "clamp" for run in report["runs"])


def test_relax_rejects_an_unknown_closure_mode():
    spec = importlib.util.spec_from_file_location(
        "laplace_pressure_floor_run", BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError):
        module.relax(0.002, 20.0, closure="bogus")
