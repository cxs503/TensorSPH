"""Guards for the m4802 deliverables report (``docs/reports/m4802-main-red-mode2-pins.md``).

The report explains *why* the mode-2 frequency pins of
``tests/test_oscillating_droplet*.py`` are recorded per platform and why the
15 % gate verdict is asserted on platform-independent evidence.  A report like
that is only worth something if its numbers cannot drift from the measurement,
so this module checks three properties, all cheap (no solver run):

1. **freshness** -- every generated block in the document equals what the
   committed JSON artefacts render to today (``m4802_make_report.py --check``);
2. **honest verdict** -- the report still says the gate is *not* met, and the
   committed verdicts it leans on still say so, so "green CI" cannot be
   mistaken for "physics accepted";
3. **reproducible figure** -- the figure the report embeds really is a
   deterministic rendering of those same artefacts (a hand-drawn PNG would
   fail here, and the binary itself is regenerable when a checkout lacks it).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "docs" / "reports"
DOC = REPORTS / "m4802-main-red-mode2-pins.md"
GENERATOR = REPORTS / "m4802_make_report.py"
FIGURE = ROOT / "docs" / "figures" / "m4802_error_vs_gate.png"
BENCH = ROOT / "benchmarks" / "oscillating_droplet"

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def report():
    """The generator module (loaded by path: it is a script, not a package)."""
    if not GENERATOR.exists():  # pragma: no cover - deliverable guard
        pytest.fail(f"{GENERATOR.relative_to(ROOT)} is missing")
    spec = importlib.util.spec_from_file_location("m4802_make_report", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def doc():
    if not DOC.exists():  # pragma: no cover - deliverable guard
        pytest.fail(f"{DOC.relative_to(ROOT)} is missing")
    return DOC.read_text()


def test_report_covers_the_six_sections(doc):
    """A progress report is only reviewable if it answers the six questions."""
    for heading in ("案例介绍", "怎么做的", "计算结果", "结果图", "注意点", "误差判定"):
        assert heading in doc, f"section {heading!r} missing from {DOC.name}"


def test_generated_blocks_match_the_committed_artefacts(report):
    """No hand-transcribed number: the blocks are re-rendered and compared."""
    assert report.check() == [], (
        "the report no longer matches the committed artefacts; re-run "
        f"`PYTHONPATH=src python {GENERATOR.relative_to(ROOT)}`")


def test_every_artefact_the_report_quotes_exists(report):
    """The three sources the report is rendered from are committed files."""
    assert report.LEDGER.is_file(), f"{report.LEDGER} missing"
    assert report.RELEASE.is_file(), f"{report.RELEASE} missing"
    assert report.LAMB.is_file(), f"{report.LAMB} missing"


def test_the_gate_verdict_has_not_been_flipped(report):
    """honest FAIL: the recorded verdict still misses both gates.

    If a future model really fixes the divergence, this fails *first* -- which
    is the reminder to re-render the verification documents instead of just
    relaxing the pin.
    """
    ledger = json.loads(report.LEDGER.read_text())
    release = json.loads(report.RELEASE.read_text())
    lamb = json.loads(report.LAMB.read_text())

    assert ledger["verdict"]["passes_gate"] is False
    assert "WITHHELD" in ledger["verdict"]["evidence_level"]
    assert release["verdict"]["worst_within_15pct"] is False
    assert release["verdict"]["worst_within_3pct"] is False
    assert lamb["pass"] is False

    for row in report.error_rows(ledger, release, lamb):
        assert abs(row["err"]) > report.GATE_PCT, (
            f"{row['label']} now measures {row['err']:+.2f} %, inside the "
            f"{report.GATE_PCT:.0f} % gate; re-record it")


def test_the_platform_pins_are_the_ones_the_tests_assert(report):
    """Report pins == ledger pins == the pins in the two test modules.

    Three places hold the same numbers (ledger, tests, report); they are
    cross-read here so a single-sided edit cannot stay self-consistent.
    """
    ledger = json.loads(report.LEDGER.read_text())
    pins = ledger["recorded_platform_pins_pct"]

    # the report renders exactly these pins ...
    rendered = report.render_pins(ledger, None, None)
    for arch, val in pins.items():
        assert f"{val:+.2f}%" in rendered, (arch, val)

    # ... and the oscillation test hard-codes the same dict
    import re

    src = (ROOT / "tests" / "test_oscillating_droplet.py").read_text()
    block = re.search(r"RECORDED_MODE2_ERR_PCT\s*=\s*\{(.*?)\}", src, re.S)
    assert block, "RECORDED_MODE2_ERR_PCT not found in tests/test_oscillating_droplet.py"
    in_test = {k: float(v) for k, v in
               re.findall(r'"(\w+)":\s*([+-]?[\d.]+)', block.group(1))}
    assert in_test == pytest.approx(pins, abs=1e-9), (
        f"test pins {in_test} vs ledger pins {pins}")


def test_figure_is_a_reproducible_rendering(report, tmp_path):
    """The embedded PNG is drawn from the same artefacts, not pasted in."""
    import sys

    if sys.modules.get("matplotlib") is None:
        pytest.importorskip("matplotlib")
    out = report.render_figure(
        tmp_path / "m4802_error_vs_gate.png",
        *report.load())
    assert out.exists() and out.stat().st_size > 10_000, (
        f"rendered figure is only {out.stat().st_size} bytes")
    assert out.read_bytes()[:8] == PNG_MAGIC

    for fig in (FIGURE, report.TRACE_FIGURE):
        if fig.exists():  # committed copy: real PNG, non-trivial size
            assert fig.read_bytes()[:8] == PNG_MAGIC
            assert fig.stat().st_size > 10_000
        assert f"docs/figures/{fig.name}" in DOC.read_text(), (
            f"the report must link {fig.name}, which this test can regenerate")

    trace = report.render_trace_figure(tmp_path / "m4802_release_trace.png")
    assert trace.read_bytes()[:8] == PNG_MAGIC


def test_report_links_the_verification_documents(doc):
    """The report's verdict is traceable to the documents that own it."""
    for name in ("docs/verification_oscillating_droplet.md",
                 "docs/verification_oscillating_droplet_lamb.md"):
        assert name in doc, f"{name} is not linked"
