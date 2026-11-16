"""A killed shard must not kill the deliverable: partial dynamic-ladder records.

``benchmarks/laplace_sph/run_dynamic_ladder.py`` is built as one record per job
("a shard that dies leaves the others usable", its module docstring).  That
promise held for the harness but **not** for the doc generator: with fewer than
two levels carrying a positive spurious-current reading, ``make_figure`` raised

    ValueError: Data has no positive values, and therefore cannot be log-scaled.

from ``matplotlib``'s log locator, so the deliverable could not be rendered at
all from a partial record -- exactly the situation the per-job design exists
for.  This file pins the degraded path:

* a partial record renders (figure + page), with the missing panels empty rather
  than invented;
* a missing gate is written as ``—`` / ``**FAIL**``, never as a number;
* a *complete* record still uses log axes (the fix must not degrade the real
  deliverable).

No physics is asserted here; the numbers are gated by
``tests/test_xval_projection_laplace_dynamic_ladder.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_sph"
RES = BENCH / "results_dynamic_ladder"
sys.path.insert(0, str(BENCH))

matplotlib = pytest.importorskip("matplotlib")


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    import run_dynamic_ladder as h
    monkeypatch.setattr(h, "OUTDIR", tmp_path)
    return h


@pytest.fixture()
def docgen(tmp_path, monkeypatch):
    import make_doc_dynamic_ladder as g
    monkeypatch.setattr(g, "OUTDIR", tmp_path)
    monkeypatch.setattr(g, "SUMMARY", tmp_path / "summary.json")
    monkeypatch.setattr(g, "DOC", tmp_path / "doc.md")
    monkeypatch.setattr(g, "FIG", tmp_path / "fig.png")
    return g


def _partial_summary(harness, tmp_path):
    """Only the coarsest level is 'measured'; the other six jobs are absent."""
    src = RES / "dx8_nu01.json"
    if not src.exists():
        pytest.skip("no job record in the tree to build a partial record from")
    (tmp_path / "dx8_nu01.json").write_text(src.read_text())
    summary = harness.summarize()
    # the record must land in the sandbox: reloading the module here would reset
    # its OUTDIR to the repository path and write into benchmarks/ .  That is
    # what a stray summary.json in the tree does to the strict-xfail debt marker
    # (it xpasses and turns the suite red), so assert it explicitly.
    assert (tmp_path / "summary.json").exists(), "summarize wrote outside OUTDIR"
    return summary


def test_partial_record_renders_a_figure_and_a_page(harness, docgen, tmp_path):
    s = _partial_summary(harness, tmp_path)
    assert s["n_jobs_present"] < s["n_jobs_expected"]

    docgen.make_figure(s, docgen.FIG)      # used to raise ValueError
    docgen.make_doc(s)

    assert docgen.FIG.exists() and docgen.FIG.stat().st_size > 0
    assert docgen.DOC.exists()


def test_partial_record_invents_no_number(harness, docgen, tmp_path):
    s = _partial_summary(harness, tmp_path)
    docgen.make_doc(s)
    text = docgen.DOC.read_text()
    # every gate that could not be measured is declared, not silently dropped
    for name in s["gates_declared"]:
        assert "`%s`" % name in text, "gate %r missing from a partial page" % name
    # unmeasurable rows are dashes
    assert "—" in text
    # and the unmeasured gates are marked as such, never as pass
    failed = [k for k, v in s["gate_verdicts"].items() if not v["pass"]]
    assert failed, "a 1/7 record must not report all gates green"
    for name in failed:
        row = [ln for ln in text.splitlines() if "`%s`" % name in ln]
        assert any("**FAIL**" in ln for ln in row), (
            "gate %r is not marked FAIL on the partial page" % name)


def test_full_record_still_uses_log_axes(tmp_path):
    """Guard against 'fixing' the crash by always plotting linear."""
    import make_doc_dynamic_ladder as g
    assert g._positive_curve([0.26, 0.1, 0.05]) is True
    assert g._positive_curve([0.26]) is False             # one level only
    assert g._positive_curve([0.26, 0.0]) is False        # exact rest
    assert g._positive_curve([0.26, float("nan")]) is False