"""Pool-row conformance for CASE-POISEUILLE-SPH (算例池·泊肃叶流, ref: analytic, <5 %).

The case is graded by a family of studies (R1 ladder, R2 Womersley, combined
Couette-Poiseuille, steady-window, drift, uncertainty), each writing its own
artifact.  This test binds the *whole* record to the raw data with
``benchmarks/case_poiseuille/pool_conformance.py`` and asks four questions:

1. does every graded functional still sit inside its declared tolerance;
2. can every stored summary percentage still be re-derived from the raw rows
   (to 1e-6 relative) -- i.e. has no artifact drifted away from its verdict;
3. is the set of binding failures empty -- the one historical exception (the
   combined case's C1 at H/dx = 8 used a different profile normalisation than
   the rungs 12/16) was repaired at root by ``rebase_gauge.py``, so any binding
   failure now is a regression;
4. are the source artifacts the same bytes the record was generated from
   (SHA-256 pins), so an artifact edit forces a regeneration instead of a
   silent re-grade.

Regenerate with:
    python benchmarks/case_poiseuille/pool_conformance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "case_poiseuille"
RECORD = BENCH / "out" / "pool_conformance.json"

sys.path.insert(0, str(BENCH))
import pool_conformance as pc  # noqa: E402


@pytest.fixture(scope="module")
def record():
    assert RECORD.exists(), (
        f"missing {RECORD}; regenerate with "
        "python benchmarks/case_poiseuille/pool_conformance.py")
    return json.loads(RECORD.read_text())


@pytest.fixture(scope="module")
def fresh():
    return pc.build()


def test_record_matches_a_fresh_recompute(record, fresh):
    """The committed record is the artifact; a stale record is not a verdict."""
    assert record["summary"] == fresh["summary"]
    assert record["source_artifacts_sha256"] == fresh["source_artifacts_sha256"]


def test_source_artifacts_are_unchanged(record):
    """Artifact bytes are pinned: editing one silently would break the binding."""
    for rel, digest in record["source_artifacts_sha256"].items():
        path = REPO / rel
        assert path.exists(), f"pinned artifact vanished: {rel}"
        assert pc.sha256(path) == digest, f"artifact changed since the record: {rel}"


def test_recomputation_is_wired_to_raw_data(record):
    """A gate that only reads stored percentages would not be a conformance check."""
    s = record["summary"]
    assert s["n_recomputed_from_raw"] >= 30, s
    assert s["n_functionals"] == (s["n_recomputed_from_raw"] + s["n_stored"]
                                  + s["n_not_recomputable"])
    # the recomputed anchor functionals must actually bind
    assert s["binding_max_rel_diff"] is not None


def test_no_binding_violation_remains(record):
    """Stored summary vs raw rows must now agree everywhere (no exception left).

    The single historical exception -- ``CP_profile_rms@Hdx8`` graded on ``U_w``
    while rungs 12/16 used ``max|u_a|`` over their rows -- was repaired at root
    (one closed-form continuum gauge in ``combined.py`` and here, plus
    ``rebase_gauge.py`` rebasing the committed summary from its own rows), so a
    binding failure is a regression, not a tolerated state.
    """
    assert pc.KNOWN_BINDING_EXCEPTIONS == {}
    assert record["binding_violations"] == [], record["binding_violations"]
    assert record["summary"]["n_unknown_binding_violations"] == 0


def test_anchor_scope_within_tolerance(record):
    """Every functional graded at the anchor's 5 % must be inside 5 %."""
    anchor = [r for r in record["functionals"]
              if r["error_percent"] is not None
              and r["tolerance_percent"] == pc.TOL_PCT]
    assert anchor
    worst = max(anchor, key=lambda r: abs(r["error_percent"]))
    failing = [r["id"] for r in anchor if r["pass"] is False]
    # the only tolerated failures are the combined case's honest C6 backflow gate
    assert set(failing) <= {"CP_backflow_peak@Hdx8", "CP_backflow_peak@Hdx12"}, failing
    assert abs(worst["error_percent"]) <= pc.TOL_PCT or worst["id"] in failing


def test_honest_fail_is_not_re_graded(record):
    """C6 fails at H/dx = 8 and 12 and the record must keep saying so."""
    ids = {r["id"]: r for r in record["functionals"]}
    assert ids["CP_backflow_peak@Hdx8"]["pass"] is False
    assert ids["CP_backflow_peak@Hdx12"]["pass"] is False
    assert ids["CP_backflow_peak@Hdx16"]["pass"] is True
    assert record["verdict"] == "FAIL"


def test_supporting_studies_declare_pass(record):
    """Steady-window / drift / uncertainty / audit gates are carried, not dropped."""
    support = [r for r in record["functionals"] if r["id"].startswith("support:")]
    assert len(support) >= 12
    assert all(r["pass"] for r in support), [r["id"] for r in support if not r["pass"]]


def test_not_graded_is_declared(record):
    """Nothing silently missing: an absent artifact must be listed."""
    names = " ".join(record["not_graded"])
    assert "missing" not in names.lower() or "not graded" in names.lower()
