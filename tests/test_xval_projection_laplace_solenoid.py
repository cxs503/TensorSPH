"""Gates on the one-shot solenoidal diagnostic of the projection-SPH Laplace drop.

The measurement is cheap (one Poisson solve per level) but its *interpretation*
is the deliverable, so the record is gated: the operator identities must be
exact, the split must be complete, and the H1 verdict must be recorded as
falsified if the solenoidal share does not decay -- a future round that closes
it must edit this file deliberately.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_sph"
REC = BENCH / "results_dynamic_ladder" / "diagnose_solenoid.json"
DOC = ROOT / "docs" / "xval_projection_laplace_solenoid_ladder.md"
sys.path.insert(0, str(BENCH))

pytestmark = pytest.mark.skipif(not REC.exists(), reason="diagnostic record absent")


@pytest.fixture(scope="module")
def rec():
    return json.loads(REC.read_text())


def test_declared_gates_are_the_harness_constants(rec):
    import diagnose_csf_solenoid as d
    assert rec["gates_declared"] == json.loads(json.dumps(d.GATES_DECLARED))


def test_operator_identities_are_round_off(rec):
    for lv in rec["levels"]:
        assert lv["adjointness_defect"] <= 1e-10
        assert lv["reconstruction_defect"] <= 1e-6
        # the split must be orthogonal: <irrotational, solenoidal> ~ 0
        assert abs(lv["orthogonality"]) <= 1e-6


def test_verdicts_agree_with_their_own_numbers(rec):
    for name, v in rec["gate_verdicts"].items():
        m, d, op = v["measured"], v["declared"], v["op"]
        if m != m:  # nan -> cannot pass
            assert v["pass"] is False, "%s passed on nan" % name
            continue
        expect = m <= d if op == "<=" else m >= d
        if name == "consistency_with_march":
            expect = (1.0 / 3.0) <= m <= 3.0
        assert bool(v["pass"]) == bool(expect), name


def test_levels_are_the_ones_reported(rec):
    xs = [lv["R_over_dx"] for lv in rec["levels"]]
    assert xs == sorted(xs)
    assert len(xs) == len(set(xs))
    assert set(xs) == set(rec["setup"]["levels_R_over_dx"])


def test_h1_verdict_is_recorded_not_hidden(rec):
    """H1 (the residual is a dx-artefact) must be stated as supported or not."""
    sup = rec["orders"]["H1_artifact_hypothesis_supported"]
    order = rec["orders"]["solenoidal_fraction_decay_order"]
    assert isinstance(sup, bool)
    if order == order:  # finite
        assert sup == (order >= 0.50), "H1 flag disagrees with the fitted order"


def test_doc_reports_every_failed_gate(rec):
    if not DOC.exists():
        pytest.skip("doc not rendered")
    text = DOC.read_text()
    for name, v in rec["gate_verdicts"].items():
        if not v["pass"]:
            rows = [ln for ln in text.splitlines() if "`%s`" % name in ln]
            assert rows, "failed gate %r missing from the doc" % name
            assert any("**FAIL**" in ln for ln in rows), name
