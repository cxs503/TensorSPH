"""Regression gates for the *verdict* of the Young-Laplace evidence, not its numbers.

The numbers live in three records (`report_force_balance.json`,
`raw/relax_ladder.json`, `laplace_pressure_floor/report.json`) and each was
verified for arithmetic in its own test module.  What was never gated is the
step from numbers to grading, and that step went wrong twice in the document
these tests now guard:

1. the headline "best closure variant" was taken over *all* variants, so the
   +8.8 % row -- a run whose surface breathes (`v_max/v_ref = 0.388`) and whose
   final snapshot is +142.8 % off -- was presented as the winner, although
   `test_closure_improves_the_relaxed_laplace_readout` ranks only variants that
   satisfy Adami's spurious-current criterion;
2. the same sentence then called that number "not inside the 15 % acceptance
   band", which is arithmetically false because ``|8.8| < 15``.

Both are now computed in `benchmarks/laplace_pressure_closure/verdict_audit.py`
and pinned here, plus the two evidence defects the audit surfaced (duplicate
"variant" rows, and a relaxation group that lives on a single grid).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
BENCH = HERE / "benchmarks" / "laplace_pressure_closure"
DOC = HERE / "docs" / "verification_laplace_pressure_closure.md"
MATRIX = BENCH / "verdict_matrix.json"
GATE = 15.0


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def matrix():
    if not MATRIX.exists():
        pytest.skip("run benchmarks/laplace_pressure_closure/verdict_audit.py first")
    return json.loads(MATRIX.read_text())


@pytest.fixture(scope="module")
def rows_by_label(matrix):
    return {row["label"]: row for row in matrix["readouts"]}


def test_matrix_flags_match_their_own_arithmetic(matrix):
    """A row may never be labelled 'outside the band' while its error is inside it.

    This is the exact error the document made, reduced to a bit: the flag and
    the number it summarises are recomputed independently here.
    """
    for row in matrix["readouts"]:
        err = row["rel_error_pct"]
        assert row["within_15pct"] == (abs(err) < GATE), row["label"]
        assert row["within_3pct"] == (abs(err) < 3.0), row["label"]
    # The gate itself must be the platform's 15 %, not a locally tuned number.
    assert matrix["gate"]["band_pct"] == GATE


def test_capability_question_is_answered_on_the_relaxation_group_only(matrix):
    """The static flux integral is the identity dp/dr = rho a_r: it cannot grade.

    It sits inside the band by construction (-0.3 % at dx = 2 mm), and an earlier
    revision of the audit let it flip the verdict to "provisional" -- the
    inflation this test forbids.
    """
    groups = matrix["summary"]["by_group"]
    assert groups["relaxation"]["answers_capability_question"] is True
    for name in ("static_force_balance", "floor_reference"):
        assert groups[name]["answers_capability_question"] is False, name

    asserted = " ".join(matrix["verdict"]["not_capability_evidence"])
    assert "flux integral" in asserted
    # And the headline best read-out of the capability group is a pressure one.
    cap_best = groups["relaxation"]["best_admissible"]
    assert "relaxed" in cap_best["label"]


def test_verdict_stays_withheld_because_no_admissible_readout_is_in_band(matrix):
    """Honest FAIL pin: admissible evidence is -20.1 %, well outside the 15 % band."""
    cap = matrix["summary"]["answers_capability_question"]
    assert cap["n_within_15pct_admissible"] == 0, cap
    assert matrix["verdict"]["status"] == "WITHHELD"
    best = cap["best_admissible"]
    assert best["within_15pct"] is False
    assert abs(best["rel_error_pct"]) > GATE
    assert best["rel_error_pct"] == pytest.approx(-20.1, abs=1.0)


def test_in_band_rows_are_exactly_the_inadmissible_ones(matrix):
    """The only in-band pressure readings belong to the breathing variant.

    If a *stable* variant ever lands inside the band this fails on purpose, so the
    grading and the document get updated instead of drifting.
    """
    cap_rows = [r for r in matrix["readouts"] if r["group"] == "relaxation"]
    in_band = [r for r in cap_rows if r["within_15pct"]]
    assert in_band, "the in-band-but-inadmissible trap no longer exists; revise the doc"
    for row in in_band:
        assert row["admissible"] is False, row["label"]
        assert row["spurious_current_over_limit"] > 1.0, row["label"]
        # It wins the tail average only by breathing: its snapshot is far off.
        assert abs(row["final_snapshot_rel_error_pct"]) > 50.0, row["label"]


def test_duplicate_ladder_rows_are_not_counted_as_two_pieces_of_evidence(matrix):
    """`surface_pressure_layers` 2 vs 4 is bit-identical on a one-particle surface."""
    by_case = {r["label"]: r for r in matrix["readouts"] if r["group"] == "relaxation"}
    two = by_case["relaxed <p_core>, closure_on_interior_layers2"]
    four = by_case["relaxed <p_core>, closure_on_interior_layers4"]
    assert two["value_Pa"] == pytest.approx(four["value_Pa"], rel=0, abs=1e-12)
    assert two["vmax_over_vref"] == pytest.approx(four["vmax_over_vref"], rel=0, abs=1e-12)
    assert two["independent_evidence"] is True
    assert four["independent_evidence"] is False
    assert four["duplicate_of"] == two["label"]
    # Exactly one duplicate is on record; if the layers knob starts to matter the
    # audit must be re-read (and the document regenerated) rather than silently
    # gaining a row.
    dups = [r for r in matrix["readouts"] if not r["independent_evidence"]]
    assert len(dups) == 1


def test_resolution_defect_is_recorded(matrix):
    """The relaxation group is single-grid, and the audit must say so."""
    ids = [d["id"] for d in matrix["defects"]]
    assert "resolution-not-laddered" in ids
    defect = next(d for d in matrix["defects"] if d["id"] == "resolution-not-laddered")
    assert "1.5 mm" in defect["consequence"]


def test_document_states_the_verdict_the_matrix_computes(matrix, rows_by_label):
    """The document is generated from the matrix, so the two may not disagree."""
    if not DOC.exists():
        pytest.skip("run make_doc.py first")
    text = DOC.read_text()
    assert matrix["verdict"]["status"] in text
    # The false sentence is banned outright: with |8.8| < 15 it cannot be true.
    assert "not inside the 15 % acceptance band" not in text
    # The headline number of the admissible winner appears with its sign.
    best = matrix["summary"]["answers_capability_question"]["best_admissible"]
    assert f"{best['rel_error_pct']:+.1f}" in text
    # The in-band trap is spelled out, not hidden.
    assert "not admissible evidence" in text
    assert "one piece of evidence, not two" in text
    # And the machine-checked table is present with both flag columns.
    assert "in +/-15 % band" in text and "independent evidence" in text


def test_ladder_evidence_files_never_mix_resolutions():
    """The rung that is still open gets its own file; the historical one keeps its name."""
    ladder = _load("relax_ladder", BENCH / "relax_ladder.py")
    assert ladder.ladder_path(0.002, 20.0).name == "relax_ladder.json"
    finer = ladder.ladder_path(0.0015, 20.0)
    assert finer.name != "relax_ladder.json"
    assert "1.5000" in finer.name
    other_c0 = ladder.ladder_path(0.002, 5.0)
    assert other_c0.name != ladder.ladder_path(0.002, 20.0).name
    # An existing record belonging to another grid must not be merged into a new run.
    assert ladder.ladder_path(0.0015, 20.0) != ladder.ladder_path(0.0015, 10.0)
