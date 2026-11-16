"""Guarding test for the CASE-POISEUILLE-SPH run-length note.

This is not a "the file exists" check:

1. **Binding** -- the settled numbers in `out/time_axis.json:evidence` are
   recomputed from the committed `out/ma_envelope.json`, and the published
   markdown cells are compared against the artifact, so the note cannot drift
   away from the data it summarises.
2. **Predeclared predictions stay predeclared** -- the `Psi` table is rebuilt
   from `u_max = g H^2/(8 nu)` (analytic), so a prediction moved after a run
   fails here.
3. **The rule is applied, not narrated** -- `verdict()` is re-run from the raw
   rows and must reproduce the stored status; `NOT_RUN` (no artifacts) is
   accepted *only* together with a document that claims nothing.
4. **No over-certification** -- the note may not present `Psi` as validated
   while the missing runs are missing, and may not quote an error number for a
   point that has no artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks.poiseuille_mirror import study_time_axis as sta  # noqa: E402

ARTIFACT = REPO / "benchmarks" / "poiseuille_mirror" / "out" / "time_axis.json"
DOC = REPO / "docs" / "verification_poiseuille_time_axis.md"
FIG = REPO / "docs" / "figures" / "poiseuille_time_axis.png"
TOL_PCT = 5.0


@pytest.fixture(scope="module")
def doc():
    assert ARTIFACT.exists(), f"missing committed artifact {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


def test_artifact_shape(doc):
    declared = [p["id"] for p in sta.POINTS]
    assert [p["id"] for p in doc["declared_points"]] == declared
    assert doc["tolerance_pct"] == TOL_PCT
    executed = [p["id"] for p in doc["points"]]
    assert set(executed) <= set(declared), "an undeclared point was executed"


def test_predictions_are_analytic(doc):
    op = doc["verdict"]["operating_point"]
    u = op["g"] * op["H"] ** 2 / (8.0 * op["nu"])
    assert u == pytest.approx(op["u_max_analytic"], rel=1e-12)
    for pid, pred in doc["verdict"]["predictions"].items():
        assert pred["psi_analytic"] == pytest.approx(
            u ** 2 * pred["t_end"] / (op["cs"] * op["dx"]), rel=1e-12)
        assert pred["psi_analytic"] == pytest.approx(sta.psi_predicted(pred["t_end"]),
                                                    rel=1e-12)


def test_evidence_recomputed_from_committed_sweep(doc):
    ev = doc["evidence"]
    assert ev["available"], "committed ma_envelope.json is required for the note"
    fresh = sta.existing_evidence()
    assert fresh["dt_p3"] == pytest.approx(ev["dt_p3"], rel=1e-12)
    assert fresh["dts_equal"] == ev["dts_equal"] is True
    assert fresh["psi_ratio_p4_over_p3"] == pytest.approx(ev["psi_ratio_p4_over_p3"],
                                                          rel=1e-12)
    assert fresh["rms_amplification_p4_over_p3"] == pytest.approx(
        ev["rms_amplification_p4_over_p3"], rel=1e-12)
    # the settled statement: same dt, opposite verdicts -> dt is not sufficient
    assert ev["dt_candidate"] == "ELIMINATED_AS_SUFFICIENT_CAUSE"
    assert ev["p3"]["passes_all_gates"] is True
    assert ev["p4"]["passes_all_gates"] is False
    assert ev["p4"]["t_end"] == pytest.approx(2.0 * ev["p3"]["t_end"])


def test_executed_points_are_the_certified_operating_point(doc):
    op = doc["verdict"]["operating_point"]
    for p in doc["points"]:
        assert p["dt"] <= op["certified_dt"] * 1.001
        assert p["max_mach"] <= 0.0610 * 1.02
        assert p["psi_index"] == pytest.approx(p["psi_analytic_prediction"], rel=0.02)
        assert p["passes_all_gates"] == all(p["gates"].values())


def test_verdict_recomputed(doc):
    fresh = sta.verdict(doc["raw_rows"])
    assert fresh["status"] == doc["verdict"]["status"]
    assert fresh["statement"] == doc["verdict"]["statement"]


def test_status_is_one_of_the_declared_outcomes(doc):
    assert doc["verdict"]["status"] in {
        "NOT_RUN", "INCOMPLETE", "H_TIME_SUPPORTED", "H_PSI_SUPPORTED",
        "H_MA_DT_SUPPORTED_PARTIAL", "H_MA_DT_SUPPORTED_BOUNDED",
        "H_MA_DT_SUPPORTED_STRONG", "UNEXPLAINED"}


def test_no_over_certification(doc):
    v = doc["verdict"]
    text = v["statement"]
    if v["status"] == "NOT_RUN":
        assert "makes no claim" in text
    if v["status"] == "H_PSI_SUPPORTED":
        assert "two-point" in text and "not a fitted law" in text
    if v["status"].endswith("PARTIAL"):
        assert "not been executed" in text or "not executed" in text
    assert "validated criterion" not in text


def test_document_binds_to_artifact(doc):
    assert DOC.exists(), f"missing {DOC}"
    md = DOC.read_text()
    assert FIG.exists() and FIG.stat().st_size > 20_000
    assert FIG.read_bytes()[:4] == b"\x89PNG"
    # the verdict status is quoted verbatim
    assert doc["verdict"]["status"] in md
    # the settled numbers appear with the artifact's own formatting
    ev = doc["evidence"]
    for key in ("dt_p3", "dt_over_certified", "psi_ratio_p4_over_p3",
                "rms_amplification_p4_over_p3"):
        assert f"{ev[key]:.3e}" in md or f"{ev[key]:.2f}" in md or f"{ev[key]:.3f}" in md
    for q in ev["p3"], ev["p4"]:
        assert f"{q['profile_rms_interior_pct']:.3f}" in md
        assert f"{q['wall_shear_worst_abs_pct']:.3f}" in md
    # every executed point must appear as a row of the results table
    for p in doc["points"]:
        assert any(ln.startswith(f"| {p['id']} |") for ln in md.splitlines())
    # a NOT_RUN note must say so explicitly instead of quoting errors
    if not doc["points"]:
        assert "NOT_RUN" in md
        assert "不做误差判定" in md


def test_document_declares_limits(doc):
    md = DOC.read_text()
    for needle in ("④ 判定", "⑤ 注意点与局限", "⑥ 误差判定", "两点夹逼",
                   "不是拟合出的定律", "排队"):
        assert needle in md, f"document is missing {needle!r}"
    # failing gates of executed points must be visible
    n_fail = sum(1 for p in doc["points"] for ok in p["gates"].values() if not ok)
    assert md.count("✗") >= n_fail
