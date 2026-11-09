"""Ledger-registration guard for ``benchmarks/ice_floe_added_mass``.

Closes the ``uncovered_benchmark`` finding of the verification ledger: the
harness must keep a committed ``report.json`` (so the verification doc's
numbers are reproducible from the repo), the ledger must carry the registered
claims, and the committed report must stay self-declaring.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "benchmarks" / "verification_ledger" / "build_ledger.py"
REPORT = REPO / "benchmarks" / "ice_floe_added_mass" / "report.json"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "verification_ledger_builder_m7556", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = _load_builder()


@pytest.fixture(scope="module")
def ledger():
    return B.build()


def test_report_is_committed_and_self_declaring():
    assert REPORT.is_file(), (
        "benchmarks/ice_floe_added_mass/report.json must be committed: the "
        "verification doc injects its numbers from it"
    )
    d = json.loads(REPORT.read_text())
    assert d["verdict"] in ("pass", "fail")
    assert d["criteria"] and all(
        isinstance(v, bool) for v in d["criteria"].values())
    assert (d["verdict"] == "pass") == all(d["criteria"].values())


def test_ledger_registers_the_claims(ledger):
    by_id = {e["claim_id"]: e for e in ledger["entries"]}
    for cid in (
        "ice_floe_added_mass.anchor.c_m_unbounded",
        "ice_floe_added_mass.anchor.c_m_richardson",
        "ice_floe_added_mass.plane_limits.max_deviation",
        "ice_floe_added_mass.launch.speed_ratio",
    ):
        assert cid in by_id, f"claim {cid} missing from the ledger"
        assert by_id[cid]["artifact"] == "benchmarks/ice_floe_added_mass/report.json"


def test_no_uncovered_benchmark_finding_for_ice_floe(ledger):
    offenders = [
        f for f in ledger["findings"]
        if f["rule"] == "uncovered_benchmark" and "ice_floe_added_mass" in f["claim_id"]
    ]
    assert not offenders, f"ice_floe_added_mass is uncovered again: {offenders}"
