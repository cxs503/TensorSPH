"""Regression gates for the *verdict* of the Young-Laplace evidence, not its numbers.

The numbers live in the records (`report_force_balance.json`,
`raw/relax_ladder*.json`, `laplace_pressure_floor/report.json`) and each was
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
and pinned here, plus the evidence defects the audit surfaces (duplicate
"variant" rows, in-band-but-inadmissible rows) and the provenance of the
ladder evidence files the matrix reads.

The verdict pinned here is PROVISIONAL, not WITHHELD: the ``dx``/``c0`` ladder
recorded after the audit was written (PR #317) added points at which the
interior-2-layers variant is inside the band *and* admissible (1.5 mm/20 m/s:
-3.4 %, 2.0 mm/10 m/s: -4.0 %), while the same variant is not admissible at
2.0 mm/20 m/s (breathing) and 2.0 mm/5 m/s (marginal current).  If a future
rung makes the variant uniform, this file must be re-read and the grading
upgraded -- not left to drift.
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


def test_verdict_is_provisional_with_admissible_in_band_readouts(matrix):
    """Honest pin: admissible in-band read-outs exist, but the variant is not uniform.

    The two in-band admissible read-outs are the interior-2-layers variant at
    1.5 mm/20 m/s (-3.4 %) and 2.0 mm/10 m/s (-4.0 %); the same variant is not
    admissible at 2.0 mm/20 m/s (0.388) and 2.0 mm/5 m/s (0.104).  A uniform
    upgrade to VERIFIED or a regression to WITHHELD must fail this test and
    force the document to be regenerated, not hand-edited.
    """
    cap = matrix["summary"]["answers_capability_question"]
    assert cap["n_within_15pct_admissible"] == 2, cap
    assert matrix["verdict"]["status"] == "PROVISIONAL"
    best = cap["best_admissible"]
    assert best["within_15pct"] is True
    assert best["admissible"] is True
    assert "closure_on_interior_layers2" in best["label"]
    assert best["config"]["dx_m"] == pytest.approx(0.0015)
    assert best["config"]["sound_speed"] == pytest.approx(20.0)
    assert best["rel_error_pct"] == pytest.approx(-3.4, abs=1.0)
    # The reason must name the points where the variant is not admissible.
    assert "2.0 mm, c0 = 20 m/s" in matrix["verdict"]["reason"]
    assert "2.0 mm, c0 = 5 m/s" in matrix["verdict"]["reason"]


def test_in_band_rows_split_admissible_from_inadmissible(matrix):
    """Every in-band capability row is either admissible or flagged as a defect.

    The breathing rows (final snapshot far off) are the inadmissible ones; the
    admissible in-band rows sit below the Adami limit.  If a *stable* variant
    ever lands inside the band at a new point, the defect list and the verdict
    move together -- this test fails on purpose so the record is updated.
    """
    cap_rows = [r for r in matrix["readouts"] if r["group"] == "relaxation"]
    in_band = [r for r in cap_rows if r["within_15pct"]]
    assert len(in_band) == 5, [r["label"] for r in in_band]
    adm = [r for r in in_band if r["admissible"]]
    inad = [r for r in in_band if not r["admissible"]]
    assert len(adm) == 2 and len(inad) == 3, [r["label"] for r in in_band]
    for row in adm:
        assert row["spurious_current_over_limit"] < 1.0, row["label"]
    for row in inad:
        assert row["spurious_current_over_limit"] > 1.0, row["label"]
    # The breathing rows are exactly the two 2.0 mm/20 m/s interior rows.
    breathing = [r for r in in_band if abs(r["final_snapshot_rel_error_pct"]) > 50.0]
    assert {r["config"]["sound_speed"] for r in breathing} == {20.0}
    assert all(r["config"]["dx_m"] == pytest.approx(0.002) for r in breathing)
    # Every inadmissible in-band row is named in the defect list.
    flagged = {d["row"] for d in matrix["defects"] if d["id"] == "in-band-but-inadmissible"}
    assert {r["label"] for r in inad} <= flagged


def test_duplicate_ladder_rows_are_not_counted_as_two_pieces_of_evidence(matrix):
    """`surface_pressure_layers` 2 vs 4 is bit-identical on a one-particle surface."""
    by_case = {r["label"]: r for r in matrix["readouts"] if r["group"] == "relaxation"}
    two = by_case["relaxed <p_core>, closure_on_interior_layers2, dx = 2.0 mm, c0 = 20 m/s"]
    four = by_case["relaxed <p_core>, closure_on_interior_layers4, dx = 2.0 mm, c0 = 20 m/s"]
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


def test_ladder_is_laddered_so_the_single_grid_defect_is_absent(matrix):
    """The relaxation group spans several (dx, c0) points; the old defect is gone.

    The single-grid defect that forced WITHHELD in the single-point revision is
    derived, not asserted: it may only appear while the group really sits on
    one point.
    """
    ids = [d["id"] for d in matrix["defects"]]
    assert "resolution-not-laddered" not in ids
    cap_rows = [r for r in matrix["readouts"] if r["group"] == "relaxation"]
    points = {(r["config"]["dx_m"], r["config"]["sound_speed"]) for r in cap_rows}
    assert len(points) >= 2, points
    assert matrix["summary"]["n_capability_points"] == len(points)


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


def test_matrix_rows_trace_to_their_evidence_files(matrix):
    """A matrix row must trace to a real file whose header matches the row's point.

    The matrix reads every `raw/relax_ladder*.json`; if a file's header ever
    disagrees with the point a row claims (the failure mode that once put 2 mm
    cases under a 1.5 mm header), the row is contaminated evidence.
    """
    for row in matrix["readouts"]:
        if row["group"] != "relaxation":
            continue
        path = BENCH / row["source"]
        assert path.exists(), row["source"]
        rec = json.loads(path.read_text())
        assert (rec["dx_m"], rec["sound_speed"]) == (
            row["config"]["dx_m"], row["config"]["sound_speed"]), row["label"]


def test_ladder_evidence_files_never_mix_resolutions(tmp_path):
    """The tool guarantees: one file per (dx, c0), historical name kept, no foreign resume."""
    ladder = _load("relax_ladder", BENCH / "relax_ladder.py")
    assert ladder.out_path(0.002, 20.0).name == "relax_ladder.json"
    finer = ladder.out_path(0.0015, 20.0)
    assert finer.name != "relax_ladder.json"
    assert "1.5mm" in finer.name
    other_c0 = ladder.out_path(0.002, 5.0)
    assert other_c0.name != ladder.out_path(0.002, 20.0).name
    # An existing record belonging to another grid must not be merged into a new run.
    assert ladder.out_path(0.0015, 20.0) != ladder.out_path(0.0015, 10.0)
    stale = {"dx_m": 0.002, "sound_speed": 20.0, "cases": [{"case": "closure_off"}]}
    target = tmp_path / "relax_ladder_dx1.5mm_c20.json"
    target.write_text(json.dumps(stale))
    assert ladder.resume_cases(target, 0.0015, 20.0) == [], (
        "a record recorded at 2.0 mm/20 m/s must not be resumed into a 1.5 mm run")
