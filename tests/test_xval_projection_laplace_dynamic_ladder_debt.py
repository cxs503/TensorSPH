"""Visibility gate for the *evidence debt* of the dynamic Young-Laplace ladder.

``tests/test_xval_projection_laplace_dynamic_ladder.py`` is written as a module
with ``skipif(not SUMMARY.exists())``.  A skipped module reports green, so before
this file existed the ``XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE`` dynamic ladder
had **no** enforcement at all: its harness (``benchmarks/laplace_sph/
run_dynamic_ladder.py``), its doc generator and its eight pre-registered gates
were in the tree, the record was not, and nothing in CI said so.

This file makes the absence a *counted* test outcome instead:

* ``test_dynamic_ladder_record_present`` is a **strict** xfail while the record
  is missing.  It is not a skip -- pytest counts it as ``xfailed`` -- and because
  it is strict, the moment ``results_dynamic_ladder/summary.json`` lands the test
  **xpasses and turns the suite red**, which forces the debt to be closed (the
  record consumed by the doc generator) instead of quietly satisfied.
* ``test_debt_is_documented`` is a normal test: someone who cannot run the ladder
  still has to leave a page with the reproduce command, the declared gates and
  the reason, so the next unit does not have to rediscover them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_sph"
RES = BENCH / "results_dynamic_ladder"
SUMMARY = RES / "summary.json"
DOC = ROOT / "docs" / "xval_projection_laplace_dynamic_ladder_debt.md"

sys.path.insert(0, str(BENCH))


@pytest.mark.xfail(strict=True, reason=(
    "dynamic projection-SPH Young-Laplace ladder record absent: harness, gates "
    "and doc generator are in tree but results_dynamic_ladder/summary.json is "
    "not; see docs/xval_projection_laplace_dynamic_ladder_debt.md"))
def test_dynamic_ladder_record_present():
    assert SUMMARY.exists(), (
        "no ladder record: run benchmarks/laplace_sph/run_dynamic_ladder.py "
        "(7 jobs) then --summarize; see the debt page for the job list")


def test_debt_is_documented():
    assert DOC.exists(), "the ladder evidence debt must carry a page"
    text = DOC.read_text()
    for needle in ("run_dynamic_ladder.py", "--summarize",
                   "results_dynamic_ladder/summary.json", "< 3 %"):
        assert needle in text, "debt page does not mention %r" % needle


def test_debt_page_lists_the_declared_gates_verbatim():
    """Every pre-registered gate must appear on the page, with its threshold."""
    import run_dynamic_ladder as harness

    text = DOC.read_text()
    for name, spec in harness.GATES_DECLARED.items():
        assert "`%s`" % name in text, "gate %r absent from the debt page" % name
        assert ("%.4g" % float(spec["value"])) in text, (
            "threshold of gate %r (%r) absent from the debt page"
            % (name, spec["value"]))


def test_debt_page_states_the_unmeasured_levels():
    """Honest scope: the page must say which levels are *not* measured."""
    import run_dynamic_ladder as harness

    text = DOC.read_text()
    unmeasured = 0
    for name in harness.LEVELS:
        if (RES / ("dx%d_nu01.json" % name)).exists():
            continue
        unmeasured += 1
        assert ("`R/dx` = %d" % name) in text, (
            "level R/dx=%d has no job file and is not listed as unmeasured"
            % name)
    assert unmeasured > 0, "every level is measured: retire this page"

def test_summarize_survives_a_side_diagnostic_in_the_same_directory(tmp_path, monkeypatch):
    """The shipped summariser must not read a file that is not a ladder job.

    ``results_dynamic_ladder/`` also holds ``diagnose_solenoid.json`` (the one-shot
    Helmholtz-split diagnostic).  Before this test it made ``--summarize`` die with
    ``KeyError: 'role'``, i.e. the record could not be merged at all.
    """
    import json

    import run_dynamic_ladder as harness

    dx8 = RES / "dx8_nu01.json"
    if not dx8.exists():
        pytest.skip("no job file to copy")
    (tmp_path / "diagnose_solenoid.json").write_text(json.dumps(
        {"R_over_dx": 8, "solenoidal_fraction": 0.23}))
    (tmp_path / "dx8_nu01.json").write_text(dx8.read_text())
    monkeypatch.setattr(harness, "OUTDIR", tmp_path)

    summary = harness.summarize()

    jobs = {r["job"] for r in summary["rows"]}
    assert "diagnose_solenoid" not in jobs
    assert jobs == {"dx8_nu01"}
    assert (tmp_path / "summary.json").exists()
