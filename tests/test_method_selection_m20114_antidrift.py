"""Anti-drift gate for the m20114 artefacts (PR#869 follow-through).

The m20114 tables/summary are *derived* documents: they print registry-derived
counts (``records``, ``methods_without_evidence``) and draw a coverage figure from
the registry.  Missions merged after m20114 banked more evidence records, so the
committed counts went stale (``records: 12`` while the registry held 29) and the
coverage figure kept drawing the old bar set -- and nothing failed, because
``build_registry.py --check`` only guards the m18056 documents.

These tests make the m20114 deliverables a checked recomputation:

* ``--check`` is clean on the committed artefacts **and** writes nothing;
* the summary agrees with the registry it claims to summarise;
* the gate fires when an artefact is hand-edited, and the figure digest is
  sensitive to the plotted series (so the gate is not vacuous);
* the coverage figure on disk is the one the current registry draws.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmarks" / "method_selection" / "plot_method_selection_m20114.py"
TABLES = REPO / "docs" / "reports" / "method_selection_m20114_tables.md"
SUMMARY = REPO / "docs" / "reports" / "method_selection_m20114.json"
REGISTRY = REPO / "docs" / "reports" / "method_selection_registry.json"
FIG_DIR = REPO / "docs" / "figures" / "method_selection_m20114"
ARTEFACTS = (TABLES, SUMMARY, FIG_DIR / "figure_tvc_front.png",
             FIG_DIR / "figure_registry_coverage.png")


def _module():
    spec = importlib.util.spec_from_file_location("plot_method_selection_m20114", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_check(*extra: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    return subprocess.run([sys.executable, str(SCRIPT), "--check", *extra],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=600)


def test_check_mode_is_clean_and_read_only():
    before = {p: (_sha(p), p.stat().st_mtime_ns) for p in ARTEFACTS}
    proc = _run_check()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "up to date" in proc.stdout
    after = {p: (_sha(p), p.stat().st_mtime_ns) for p in ARTEFACTS}
    assert before == after, "check mode must not touch the artefacts"


def test_summary_tracks_the_current_registry():
    summary = json.loads(SUMMARY.read_text())
    registry = json.loads(REGISTRY.read_text())
    assert summary["records"] == len(registry["records"])
    assert summary["records"] > 12, "regression: registry evidence records never shrink"
    assert summary["methods_without_evidence"] == registry["methods_without_evidence"]
    banked = next(r["measured_pct"] for r in registry["records"]
                  if r["extractor"] == "x_transport_velocity")
    assert summary["registry_backed_pct"] == pytest.approx(banked, abs=1e-9)
    # the tables carry the same counts (they are generated from the same registry)
    tables = TABLES.read_text()
    assert f"- records: {len(registry['records'])}" in tables
    assert f"- methods with evidence: {len(registry['methods_with_evidence'])}" in tables


def test_gate_fires_on_a_hand_edited_artefact(tmp_path):
    """Negative control: a tampered copy must produce drift, so the gate is real."""
    mod = _module()
    assert mod.check() == []

    tampered = tmp_path / "summary.json"
    summary = json.loads(SUMMARY.read_text())
    summary["records"] = 12  # exactly the stale value that shipped
    tampered.write_text(json.dumps(summary, indent=2))
    drift = mod.check(summary_path=tampered)
    assert drift, "hand-edited summary must be reported as drift"
    assert any("records" in d for d in drift)

    tampered_tables = tmp_path / "tables.md"
    tampered_tables.write_text(TABLES.read_text().replace("- records: ", "- records: 1"))
    assert mod.check(tables_path=tampered_tables)


def test_figure_digest_tracks_the_plotted_series():
    """The digest must depend on the data the figure draws, not on nothing."""
    mod = _module()
    registry = json.loads(REGISTRY.read_text())
    base = mod._digest(mod._registry_payload(registry))

    extended = json.loads(json.dumps(registry))
    extra = dict(extended["records"][0])
    extra["measured_pct"] = 14.0  # a new worst record for that method
    extended["records"].append(extra)
    assert mod._digest(mod._registry_payload(extended)) != base

    summary = json.loads(SUMMARY.read_text())
    assert summary["figure_digests"] == {
        "figure_tvc_front.png": mod._digest(mod._front_payload(
            {dx: json.loads((REPO / "docs/xval_sphinx_tvc/data" /
                             f"grid_dx{str(dx).replace('.', '')}.json").read_text())
             for dx in (0.03, 0.04)},
            json.loads((REPO / "benchmarks/dam_break/martin_moyce_1952.json").read_text()))),
        "figure_registry_coverage.png": base,
    }


def test_committed_figures_exist_and_are_substantial():
    for name in ("figure_tvc_front.png", "figure_registry_coverage.png"):
        fig = FIG_DIR / name
        assert fig.exists(), name
        assert fig.stat().st_size > 20_000, f"{name} looks like a placeholder"
        assert fig.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"