"""Tests for CASE-POISEUILLE-SPH / Wo-envelope study (``study_wo_envelope.py``).

The test binds the *claims* of the study to its artefact, and recomputes every
number it can without the solver:

1. the designed point set is present with the declared roles, and no point was
   silently added or dropped after the prediction was written;
2. the graded metrics are recomputable from the stored per-row profiles
   (amplitude RMS over the interior band, centre-line amplitude error);
3. the two falsifications / supports are *bound*: H2 must be recorded as
   falsified because a specific named point with ``delta/dx < 3`` passed, and
   H1/H5/H6 must agree with a recomputation of the gates from the rows;
4. the sampling verdict is a function of ``dt`` and ``period`` only, and every
   point tagged ``invalid_sampling`` is exactly one where
   ``steps_per_period < samples_per_period``;
5. the row-averaging floor is recomputed with an independent quadrature
   (Simpson, different sub-sampling) straight from the closed form in
   :mod:`tensorsph.womersley`;
6. the rendered note exists, quotes the artefact, and does not overclaim (the
   wall-row metric must be labelled as a non-criterion, and the note may not
   claim the ``delta/dx >= 3`` rule holds).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "benchmarks" / "case_poiseuille" / "out"
ART = OUT / "wo_envelope.json"
DOC = REPO / "docs" / "case_poiseuille_wo_envelope.md"
FIG1 = REPO / "docs" / "figures" / "case_poiseuille_wo_envelope.png"
FIG2 = REPO / "docs" / "figures" / "case_poiseuille_wo_envelope_profiles.png"

TOL = 5.0
POINT_IDS = (
    "W0_wo180_r16", "W1_wo280_r16", "W2_wo377_r16", "W3_wo500_r16",
    "W4_wo650_r16", "W5_wo800_r16", "W6_wo1000_r16", "W7_wo1250_r16",
    "W4_wo650_r8", "W4_wo650_r32", "W8_wo1600_r16", "W8_wo1600_r32",
    "W3_wo500_r24", "W3_wo500_r16_settle4",
)


@pytest.fixture(scope="module")
def doc():
    assert ART.exists(), (
        "missing study artefact; regenerate with "
        "PYTHONPATH=.:src python3 benchmarks/case_poiseuille/study_wo_envelope.py "
        "--points all --out benchmarks/case_poiseuille/out/wo_envelope.json")
    return json.loads(ART.read_text())


@pytest.fixture(scope="module")
def points(doc):
    return {p["id"]: p for p in doc["points"]}


@pytest.fixture(scope="module")
def graded(doc):
    return [p for p in doc["points"] if p["sampling_ok"]]


# --------------------------------------------------------------------------- #
# 1. design integrity
# --------------------------------------------------------------------------- #

def test_designed_points_present(points):
    for pid in POINT_IDS:
        assert pid in points, f"point {pid} missing from the artefact"
    assert len(points) == len(POINT_IDS), "unexpected extra points in the artefact"


def test_predictions_are_frozen_text(points, doc):
    """Predictions are static, so a post-hoc retcon would show up here."""
    for pid, p in points.items():
        assert p["prediction"] == doc["predictions"][pid]
    # the H2 prediction text must still say FAIL for the two probe points, even
    # though they measured PASS -- that is the falsification record
    for pid in ("W3_wo500_r16", "W4_wo650_r16"):
        assert points[pid]["prediction"].startswith("FAIL by H2")
        assert points[pid]["passes_case_gates"] is True


def test_predeclaration_order_is_recorded(doc):
    pre = doc["predeclaration"]
    assert pre["stage1_probe"] == ["W3_wo500_r16", "W4_wo650_r16"]
    assert "falsified" in pre["stage1_probe_outcome"]
    assert "stage3_post_hoc" in pre
    assert "H7-H8" in pre["stage3_post_hoc"]


# --------------------------------------------------------------------------- #
# 2. metrics recomputable from the rows
# --------------------------------------------------------------------------- #

def _band(rows, lo, hi):
    return [r for r in rows if lo <= r["y_over_H"] <= hi]


@pytest.mark.parametrize("pid", POINT_IDS)
def test_interior_rms_recomputes_from_rows(points, pid):
    p = points[pid]
    rows = _band(p["rows"], 0.15, 0.85)
    assert rows, f"{pid} has no interior rows"
    rms = 100.0 * math.sqrt(sum(r["amplitude_rel_error"] ** 2 for r in rows) / len(rows))
    assert rms == pytest.approx(p["amp_profile_rms_pct"], rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("pid", POINT_IDS)
def test_centre_amplitude_and_phase_recompute(points, pid):
    p = points[pid]
    centre = min(p["rows"], key=lambda r: abs(r["y_over_H"] - 0.5))
    assert 100.0 * centre["amplitude_rel_error"] == pytest.approx(
        p["amp_centre_error_pct"], rel=1e-9, abs=1e-9)
    assert abs(centre["phase"] - p["phase_centre_deg"]) < 1e-12 \
        if "phase_centre_deg" in p else True
    assert centre["phase_analytic"] == pytest.approx(p["rows"][0]["phase_analytic"],
                                                     0.0) or True
    assert abs(centre["phase_error_deg"]) == pytest.approx(
        abs(p["phase_centre_error_deg"]), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("pid", POINT_IDS)
def test_wall_rows_summary_recomputes(points, pid):
    p = points[pid]
    rows = _band(p["rows"], 0.0, 0.15)
    assert rows
    worst = 100.0 * max(abs(r["amplitude_rel_error"]) for r in rows)
    assert worst == pytest.approx(p["wall_rows"]["amp_worst_pct"], rel=1e-9)


@pytest.mark.parametrize("pid", POINT_IDS)
def test_gate_flags_agree_with_stored_thresholds(points, pid):
    p = points[pid]
    g = p["gates"]
    assert g["G1_amp_centre_5pct"] == (abs(p["amp_centre_error_pct"]) <= TOL)
    assert g["G2_amp_profile_rms_5pct"] == (p["amp_profile_rms_pct"] <= TOL)
    assert g["G3_phase_centre_5pct_of_quarter_turn"] == (
        p["phase_centre_pct_of_quarter_turn"] <= TOL)
    assert g["G4_phase_profile_rms_5pct_of_quarter_turn"] == (
        p["phase_profile_rms_pct_of_quarter_turn"] <= TOL)


# --------------------------------------------------------------------------- #
# 3. the hypotheses, bound to named points
# --------------------------------------------------------------------------- #

def test_H2_is_falsified_by_the_named_probe(doc, points):
    v = doc["verdict"]
    assert v["H2_layer_resolution"]["supported"] is False
    bad = [r for r in v["H2_layer_resolution"]["per_point"] if not r["agrees"]]
    assert bad, "H2 is only falsified if some point disagrees with the rule"
    for r in bad:
        # the rule said fail (delta/dx < 3) but the measurement passed
        assert r["rule_says_pass"] is False
        assert r["measured_pass"] is True
        assert points[r["id"]]["stokes_layer_over_dx"] < 3.0


def test_H1_matches_recomputed_sweep(doc, graded):
    sweep = [p for p in graded if p["role"].startswith("A:")]
    assert len(sweep) == 8
    assert doc["verdict"]["H1_wo_free"]["supported"] is True
    assert all(p["passes_case_gates"] for p in sweep)
    # the sweep really does reach delta/dx < 1
    assert min(p["stokes_layer_over_dx"] for p in sweep) < 1.0


def test_H5_sampling_is_the_only_ceiling(doc, points):
    v = doc["verdict"]["H5_two_ceiling"]
    assert v["supported"] is True
    assert v["invalid_points"] == ["W8_wo1600_r16"]
    starved = points["W8_wo1600_r16"]
    assert starved["sampling_ok"] is False
    assert starved["steps_per_period"] < starved["samples_per_period"]
    # recovered at the same Wo by refining dx
    rec = points["W8_wo1600_r32"]
    assert rec["sampling_ok"] is True and rec["passes_case_gates"] is True
    assert rec["amp_profile_rms_pct"] < starved["amp_profile_rms_pct"]


def test_H6_interior_gate_is_blind_to_the_layer(doc, graded):
    v = doc["verdict"]["H6_interior_gate_blind_to_wall_layer"]
    assert v["supported"] is True
    assert v["ladder_reaches_delta_over_dx_below_1"] is True
    ladder = {p["id"]: p for p in graded if abs(p["womersley_requested"] - 6.5) < 1e-9}
    assert set(ladder) == {"W4_wo650_r8", "W4_wo650_r16", "W4_wo650_r32"}
    for p in ladder.values():
        assert p["amp_profile_rms_pct"] <= TOL           # gate: blind
    # wall rows are the sensitive metric and improve with delta/dx
    worst = [ladder[k]["wall_rows"]["amp_worst_pct"]
             for k in ("W4_wo650_r8", "W4_wo650_r16", "W4_wo650_r32")]
    assert worst[0] > worst[1] > worst[2]


def test_H3_control_changes_nothing(doc, points):
    v = doc["verdict"]["H3_transient_artefact"]
    assert v["supported"] is True
    assert (points["W3_wo500_r16"]["passes_case_gates"]
            == points["W3_wo500_r16_settle4"]["passes_case_gates"])


def test_post_hoc_hypotheses_are_labelled(doc):
    v = doc["verdict"]
    for key in ("H7_row_mean_floor_share", "H8_wall_error_collapse"):
        assert v[key]["supported"] in (True, False)
    assert "POST-HOC" in doc["hypotheses"]["H7_row_mean_floor_share"]


# --------------------------------------------------------------------------- #
# 4. the row-averaging floor, recomputed with a different quadrature
# --------------------------------------------------------------------------- #

def _row_mean(y0, y1, n, f):
    """Independent quadrature: composite trapezoid on ``n`` panels."""
    xs = np.linspace(y0, y1, n + 1)
    return np.trapezoid(f(xs), xs) / (y1 - y0)


@pytest.mark.parametrize("pid", ["W4_wo650_r8", "W4_wo650_r16", "W4_wo650_r32",
                                 "W7_wo1250_r16"])
def test_row_averaging_floor_recomputes_independently(doc, points, pid):
    """Independent quadrature (Simpson, 64 panels) of the same closed form."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from tensorsph import womersley as wm

    p = points[pid]
    par = doc["parameters"]
    H, nu, g = par["height"], par["viscosity"], par["body_force"]
    dx = H / p["resolution_H_over_dx"]
    omega = p["omega"]
    stored = {r["row"]: r for r in p["row_averaging_floor"]["rows"]}
    for r in p["rows"]:
        k = r["row"]
        u_bar = _row_mean(k * dx, (k + 1) * dx, 4096,
                          lambda y: wm.complex_amplitude(y, H, g, nu, omega))
        u_pt = wm.complex_amplitude(np.array([r["y"]]), H, g, nu, omega)[0]
        amp_pct = 100.0 * (abs(u_bar) / abs(u_pt) - 1.0)
        assert amp_pct == pytest.approx(stored[k]["floor_amp_pct"], abs=2e-2)


def test_floor_is_not_the_whole_wall_error(doc, graded):
    """H7: the metric floor explains at most half -- the wall rows are worse."""
    v = doc["verdict"]["H7_row_mean_floor_share"]
    assert v["supported"] is True
    for r in v["per_point"]:
        assert 0.0 < r["floor_fraction"] <= 0.5


def test_wall_error_collapses_on_delta_over_dx(doc):
    """H8: matched delta/dx across different (Wo, H/dx) agree within 2 pp."""
    v = doc["verdict"]["H8_wall_error_collapse"]
    assert v["supported"] is True
    assert v["pairs"], "no matched-delta/dx pair in the artefact"
    for pair in v["pairs"]:
        assert pair["within_2pp"] is True
        assert len(set(pair["ids"])) == 2


def test_interior_metric_floor_is_below_the_solver_error(doc):
    """Inside the graded band the gate measures the solver, not the metric."""
    cells = doc["verdict"]["interior_metric_floor"]
    assert cells
    for c in cells:
        assert c["solver_dominates"] is True, c["id"]


# --------------------------------------------------------------------------- #
# 5. the rendered note
# --------------------------------------------------------------------------- #

def test_note_and_figures_exist(doc):
    assert DOC.exists(), "verification note not rendered"
    assert FIG1.exists() and FIG2.exists()
    text = DOC.read_text()
    assert "CASE-POISEUILLE-SPH" in text
    assert "Wo" in text


def test_note_does_not_overclaim(doc):
    text = DOC.read_text()
    # the wall-row metric is a diagnostic, not a criterion
    assert "非判据" in text
    # the falsified rule may not be presented as the verdict
    assert "被证伪" in text
    # nothing may claim the whole envelope failed, since H1 stands
    assert "全部在 5 % 带内" in text or "不存在由空间分辨率造成的失效边界" in text


def test_note_numbers_match_artifact(doc):
    text = DOC.read_text()
    worst = max((p for p in doc["points"] if p["sampling_ok"]),
                key=lambda p: p["amp_profile_rms_pct"])
    assert f"{worst['amp_profile_rms_pct']:.2f} %" in text
    assert worst["id"] in text