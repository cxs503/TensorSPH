"""The committed 3-D shear artifact must describe the case list the harness runs.

``benchmarks/threed_shear/report.json`` was published when ``CASES`` was smaller
(the h/dx sweep stopped at 1.2 and the fourth-order ladder was a single rung).
``run.py`` later grew ``H_SWEEP`` to 1.5/1.8 and the fourth-order ladder to
``n = 16/20/24``, and the artifact was never regenerated: it advertised
``missing_cases = ['h1.8']`` while ``merge()`` over the committed per-case JSONs
reports six missing rungs.  The verification ledger grades that artifact
(``threed_shear.decay_rate.finest`` is credited PASS while the artifact's own
``verdict`` is ``fail``), so a stale case list silently changes what the credit
means.

These tests make the artifact/code pair self-consistent: the artifact must be
regenerable from the committed per-case JSONs and must name exactly the rungs
that are absent.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "threed_shear"
REPORT = BENCH / "report.json"
CASES_DIR = BENCH / "cases"


def _load_run_module():
    spec = importlib.util.spec_from_file_location("threed_shear_run", BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT.read_text())


@pytest.fixture(scope="module")
def run_module():
    return _load_run_module()


@pytest.fixture(scope="module")
def committed_cases() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in CASES_DIR.glob("*.json")}


def test_artifact_matches_the_committed_per_case_json(report, run_module, committed_cases):
    """merge(cases/*.json) reproduces the committed report bit-for-bit."""
    regenerated = run_module.merge(list(committed_cases.values()))
    assert regenerated["missing_cases"] == report["missing_cases"]
    assert regenerated["verdict"] == report["verdict"]
    assert regenerated["criteria"] == report["criteria"]
    assert regenerated["resolution_ladder"] == report["resolution_ladder"]
    assert regenerated["h_over_dx_sweep"] == report["h_over_dx_sweep"]


def test_missing_cases_are_exactly_the_absent_rungs(report, run_module, committed_cases):
    """No rung may be reported missing while its case JSON is committed, and vice versa."""
    expected = sorted(set(run_module.CASES) - set(committed_cases))
    assert report["missing_cases"] == expected


def test_every_committed_case_file_belongs_to_the_case_list(run_module, committed_cases):
    stray = sorted(set(committed_cases) - set(run_module.CASES))
    assert stray == [], f"case files not reachable from run.py CASES: {stray}"


def test_verdict_follows_from_criteria_and_missing_cases(report):
    """The published verdict is a function of the gates, never a free flag."""
    expected = "pass" if all(report["criteria"].values()) and not report["missing_cases"] else "fail"
    assert report["verdict"] == expected