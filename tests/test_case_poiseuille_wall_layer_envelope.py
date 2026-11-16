"""Tests for the near-wall resolution envelope (``wall_layer_envelope.py``).

The parent study's tests bind its claims to the Wo-envelope artefact.  This file
does the same for the secondary analysis, without any solver:

1. the analysis reads the *committed* parent artefact (sha256 recorded and
   re-checked), and every number it uses is the parent's own summary, not a
   re-derivation that could drift;
2. the fit / hold-out split is reproduced from the declared rule alone (role,
   ``sampling_ok``, ``delta/dx`` window) -- no hand-picked points;
3. the power-law coefficients, R^2, AIC and the required ``delta/dx`` values
   are recomputed independently and must agree with the artefact;
4. the verdicts are a function of the frozen thresholds: the falsified
   hypothesis (A2 plateau) must stay falsified if the stored numbers do not
   reach the predeclared Delta AIC;
5. the extrapolation warning is not cosmetic: below the calibrated domain the
   stored prediction must be far off the measured plateau, and the measured
   plateau itself must be flat;
6. the row-level H8 extension is recomputed from the parent per-row profiles
   by an independent interpolation;
7. the note exists, quotes the artefact, links the figure, and does not
   overclaim (it must answer the "< 3 %" question with an explicit "不成立"
   for the predictive MAPE, and must not present the 5 % requirement as a
   measurement).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "benchmarks" / "case_poiseuille" / "out"
ART = OUT / "wo_envelope_wall_layer.json"
PARENT = OUT / "wo_envelope.json"
DOC = REPO / "docs" / "case_poiseuille_wo_envelope_wall_layer.md"
FIG = REPO / "docs" / "figures" / "case_poiseuille_wo_envelope_wall_layer.png"

CASE_GATE = 5.0
STRICT_GATE = 3.0


@pytest.fixture(scope="module")
def art():
    assert ART.exists(), (
        "missing artefact; regenerate with PYTHONPATH=src:. python3 "
        "benchmarks/case_poiseuille/wall_layer_envelope.py")
    return json.loads(ART.read_text())


@pytest.fixture(scope="module")
def parent():
    return json.loads(PARENT.read_text())


@pytest.fixture(scope="module")
def pp(parent):
    return {p["id"]: p for p in parent["points"]}


@pytest.fixture(scope="module")
def doc_text():
    return DOC.read_text()


# --------------------------------------------------------------------------- #
# 1. provenance
# --------------------------------------------------------------------------- #

def test_parent_fingerprint_is_the_committed_one(art):
    recorded = art["parent_artefact"]["sha256"]
    assert recorded == hashlib.sha256(PARENT.read_bytes()).hexdigest(), (
        "parent artefact changed after this analysis; re-run "
        "wall_layer_envelope.py")


def test_no_solver_run_was_made(art):
    assert art["provenance"]["solver_runs"] == 0
    assert art["provenance"]["command"].endswith("wall_layer_envelope.py")
    assert art["parent_artefact"]["wall_clock_total_s"] > 0


def test_predeclaration_is_recorded(art):
    for key in ("decision", "alternatives", "hypotheses", "thresholds",
                "design", "epistemic_status"):
        assert art[key], f"predeclaration field {key} missing"
    assert len(art["alternatives"]) == 3
    assert "not blind" in art["epistemic_status"]
    assert art["thresholds"]["target_gates_pct"] == [CASE_GATE, STRICT_GATE]


def test_wall_metric_agrees_with_parent_summary(art, pp):
    seen = ({r["id"] for r in art["fit_set"]}
            | {r["id"] for r in art["holdout_set"]}
            | {r["id"] for r in art["excluded"]})
    assert seen == set(pp), "some parent point is neither fitted nor held out"
    for r in art["fit_set"] + art["holdout_set"]:
        assert r["err_wall_pct"] == pytest.approx(
            pp[r["id"]]["wall_rows"]["amp_worst_pct"], abs=1e-3)


# --------------------------------------------------------------------------- #
# 2. split follows the declared rule
# --------------------------------------------------------------------------- #

def test_split_follows_the_declared_rule(art, pp):
    fit, hold, excl = set(), set(), set()
    for pid, p in pp.items():
        if not p["sampling_ok"]:
            excl.add(pid)
        elif (p["role"].startswith("A:")
              and 1.4 <= p["stokes_layer_over_dx"] <= 6.5):
            fit.add(pid)
        else:
            hold.add(pid)
    assert {r["id"] for r in art["fit_set"]} == fit
    assert {r["id"] for r in art["holdout_set"]} == hold
    assert {r["id"] for r in art["excluded"]} == excl
    assert excl == {"W8_wo1600_r16"}


def test_fit_set_spans_the_declared_window(art):
    xs = [r["delta_over_dx"] for r in art["fit_set"]]
    assert min(xs) >= 1.4 and max(xs) <= 6.5
    assert len(art["fit_set"]) == 6


# --------------------------------------------------------------------------- #
# 3. coefficients, AIC, requirements recomputed
# --------------------------------------------------------------------------- #

def _power_law(fit):
    x = np.array([p["delta_over_dx"] for p in fit])
    y = np.array([p["err_wall_pct"] for p in fit])
    a = np.polyfit(np.log(x), np.log(y), 1)
    return math.exp(a[1]), -a[0]


def test_power_law_coefficients_recompute(art):
    K, m = _power_law(art["fit_set"])
    stored = art["models"]["A1_power_law"]
    # the artefact stores the fit-set inputs rounded to 4 dp, so the
    # recomputation can only match to that precision (1e-3 is conservative)
    assert stored["K"] == pytest.approx(K, rel=1e-3)
    assert stored["m"] == pytest.approx(m, rel=1e-3)


def test_r2_recomputes(art):
    K, m = _power_law(art["fit_set"])
    x = np.array([p["delta_over_dx"] for p in art["fit_set"]])
    y = np.array([p["err_wall_pct"] for p in art["fit_set"]])
    pred = K * x ** (-m)
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    assert art["models"]["A1_power_law"]["r2"] == pytest.approx(r2, abs=1e-6)
    assert r2 >= art["thresholds"]["fit_r2_min"]


def test_aic_and_the_falsified_plateau_are_consistent(art):
    aic = art["models"]["aic"]
    n = len(art["fit_set"])
    sse1 = art["models"]["A1_power_law"]["sse_rel"]
    sse2 = art["models"]["A2_saturating"]["sse_rel"]
    assert aic["A1_power_law"] == pytest.approx(n * math.log(sse1 / n) + 4, abs=1e-3)
    assert aic["A2_saturating"] == pytest.approx(n * math.log(sse2 / n) + 6, abs=1e-3)
    delta = aic["A1_power_law"] - aic["A2_saturating"]
    assert delta == pytest.approx(aic["delta_aic_favouring_A2"], abs=2e-3)
    # the verdict must follow the frozen threshold, not the narrative
    expected = "supported" if delta >= art["thresholds"]["delta_aic_min"] else "falsified"
    assert art["verdict"]["W2_plateau_is_real"] == expected


def test_requirements_recompute_and_are_flagged_as_extrapolation(art):
    K, m = _power_law(art["fit_set"])
    dom = art["anchors"]["rule_domain"]
    for r in art["requirements"]:
        if r["model"] != "A1_power_law":
            continue
        x = (K / r["target_wall_err_pct"]) ** (1.0 / m)
        assert r["delta_over_dx_required"] == pytest.approx(x, rel=1e-3)
        inside = bool(dom["delta_over_dx_min"] <= x <= dom["delta_over_dx_max"])
        assert bool(r["inside_calibrated_domain"]) is inside
        assert inside is False, (
            "a requirement inside the calibrated domain would need a measured "
            "confirmation row in the note")
    x3 = next(r["delta_over_dx_required"] for r in art["requirements"]
              if r["model"] == "A1_power_law" and r["target_wall_err_pct"] == STRICT_GATE)
    x5 = next(r["delta_over_dx_required"] for r in art["requirements"]
              if r["model"] == "A1_power_law" and r["target_wall_err_pct"] == CASE_GATE)
    assert x3 > x5, "the stricter gate must need the finer mesh"


def test_holdout_predictions_follow_the_adopted_model(art):
    K, m = _power_law(art["fit_set"])
    detail = art["holdout"]["A1_power_law"]["detail"]
    assert len(detail) == len(art["holdout_set"])
    for h in detail:
        assert h["predicted_pct"] == pytest.approx(
            K * h["delta_over_dx"] ** (-m), abs=2e-3)
        # the artefact stores predicted/measured rounded to 3 dp
        assert h["rel_error_pct"] == pytest.approx(
            (h["predicted_pct"] - h["measured_pct"]) / h["measured_pct"] * 100.0,
            abs=2e-2)
    inside = [abs(h["rel_error_pct"]) for h in detail if h["in_fit_domain"]]
    outside = [abs(h["rel_error_pct"]) for h in detail if not h["in_fit_domain"]]
    # the per-point relative errors are stored rounded to 3 dp
    assert art["holdout"]["A1_power_law"]["interpolation_mape_pct"] == pytest.approx(
        float(np.mean(inside)), abs=1e-3)
    assert art["holdout"]["A1_power_law"]["extrapolation_abs_rel_error_pct"] == pytest.approx(
        float(np.mean(outside)), abs=1e-3)


# --------------------------------------------------------------------------- #
# 4. the domain boundary is a measured statement, not a decoration
# --------------------------------------------------------------------------- #

def test_unresolved_branch_is_flat_and_the_law_is_not_usable_there(art, pp):
    dom = art["anchors"]["rule_domain"]
    unb = art["anchors"]["unresolved_branch"]
    below = [p for p in pp.values()
             if p["sampling_ok"] and p["stokes_layer_over_dx"] < dom["delta_over_dx_min"]]
    assert len(below) == unb["n_points"]
    errs = [p["wall_rows"]["amp_worst_pct"] for p in below]
    assert max(errs) - min(errs) <= 2.0, "the measured plateau must be flat"
    # ... while the resolved-branch law predicts a much larger error there
    for h in art["holdout"]["A1_power_law"]["detail"]:
        if not h["in_fit_domain"]:
            assert h["predicted_pct"] > h["measured_pct"] * 1.1
    assert art["anchors"]["rule_domain"]["hard_lower_bound"] is True


def test_no_measured_point_is_inside_the_wall_gate(art, pp):
    best = art["anchors"]["best_measured_wall_point"]
    measured_best = min((p for p in pp.values() if p["sampling_ok"]),
                        key=lambda p: p["wall_rows"]["amp_worst_pct"])
    assert best["id"] == measured_best["id"]
    assert best["err_wall_pct"] == pytest.approx(
        measured_best["wall_rows"]["amp_worst_pct"], abs=1e-3)
    inside = [p for p in pp.values()
              if p["sampling_ok"] and p["wall_rows"]["amp_worst_pct"] <= CASE_GATE]
    anchor = art["anchors"]["smallest_measured_delta_over_dx_under_gate"]
    if inside:
        assert anchor is not None
    else:
        assert anchor is None
        assert best["err_wall_pct"] > CASE_GATE


def test_sensitivity_reports_the_excluded_point(art):
    sens = art["sensitivity"]
    assert sens["excluded_id"] == art["excluded"][0]["id"]
    assert sens["excluded_verdict"] == "invalid_sampling"
    assert sens["in_calibrated_domain"] is False
    assert "reported only" in sens["note"]


# --------------------------------------------------------------------------- #
# 5. row-level H8, recomputed from the parent profiles
# --------------------------------------------------------------------------- #

Y_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def _profile(p):
    y = np.array([r["y"] for r in p["rows"]]) / p["stokes_layer"]
    e = np.array([r["amplitude_rel_error"] for r in p["rows"]]) * 100.0
    return y, e


def test_matched_pairs_recompute(art, pp):
    ok = [p for p in pp.values() if p["sampling_ok"]]
    expected = {}
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            a, b = ok[i], ok[j]
            if abs(a["stokes_layer_over_dx"] - b["stokes_layer_over_dx"]) / \
                    a["stokes_layer_over_dx"] <= 0.02:
                ya, ea = _profile(a)
                yb, eb = _profile(b)
                ga = np.interp(Y_GRID, ya, ea)
                gb = np.interp(Y_GRID, yb, eb)
                expected[frozenset((a["id"], b["id"]))] = float(
                    np.abs(ga - gb).max())
    assert len(expected) == len(art["matched_pairs"])
    for pr in art["matched_pairs"]:
        key = frozenset(pr["pair"])
        assert key in expected
        assert pr["max_abs_disagreement_pp"] == pytest.approx(
            expected[key], abs=5e-3)


def test_row_level_verdict_follows_the_threshold(art):
    v = art["verdict"]
    th = art["thresholds"]
    maxd = v["max_matched_pair_disagreement_pp"]
    assert maxd == pytest.approx(
        max(p["max_abs_disagreement_pp"] for p in art["matched_pairs"]), abs=1e-6)
    if maxd <= th["row_pair_max_disagreement_pp"]:
        assert v["W3_H8_at_row_level"] == "supported"
    elif maxd > th["row_pair_falsify_pp"]:
        assert v["W3_H8_at_row_level"] == "falsified"
    else:
        assert v["W3_H8_at_row_level"] == "partial"


def test_adopted_model_matches_the_winner(art):
    v = art["verdict"]
    h1 = art["holdout"]["A1_power_law"]
    h2 = art["holdout"]["A2_saturating"]
    delta = art["models"]["aic"]["delta_aic_favouring_A2"]
    w1 = (art["models"]["A1_power_law"]["r2"] >= art["thresholds"]["fit_r2_min"]
          and h1["usable"])
    w2 = (delta >= art["thresholds"]["delta_aic_min"]
          and h2["interpolation_mape_pct"] < h1["interpolation_mape_pct"])
    assert v["W1_power_law_is_predictive"] == ("supported" if w1 else "falsified")
    expected = ("A0_no_envelope" if not w1 and not w2
                else "A2_saturating" if w2 else "A1_power_law")
    assert v["adopted"] == expected


# --------------------------------------------------------------------------- #
# 6. the note
# --------------------------------------------------------------------------- #

def test_note_quotes_the_artefact(art, doc_text):
    K = art["models"]["A1_power_law"]["K"]
    m = art["models"]["A1_power_law"]["m"]
    r2 = art["models"]["A1_power_law"]["r2"]
    assert f"{K:.2f}" in doc_text and f"{m:.3f}" in doc_text
    assert f"{r2:.4f}" in doc_text
    x5 = next(r["delta_over_dx_required"] for r in art["requirements"]
              if r["model"] == "A1_power_law" and r["target_wall_err_pct"] == CASE_GATE)
    assert f"{x5:.2f}" in doc_text
    assert "figures/case_poiseuille_wo_envelope_wall_layer.png" in doc_text
    assert "out/wo_envelope_wall_layer.json" in doc_text


def test_note_does_not_overclaim(art, doc_text):
    # the predictive MAPE is > 3 %, so the "< 3 %" answer must not be "yes"
    assert art["holdout"]["A1_power_law"]["interpolation_mape_pct"] > STRICT_GATE
    assert "不成立" in doc_text
    # the 5 % requirement is an extrapolation and must be labelled as such
    assert "外推" in doc_text
    assert all("inside_calibrated_domain" in doc_text for _ in (0,))
    # the wall-row metric must stay labelled as a non-criterion (parent rule)
    assert "非判据" in doc_text


def test_note_reports_the_domain_boundary(art, doc_text):
    dom = art["anchors"]["rule_domain"]
    assert f"{dom['delta_over_dx_min']}" in doc_text
    assert "只能向上用" in doc_text


def test_figure_exists_and_is_substantive():
    assert FIG.exists()
    assert FIG.stat().st_size > 20_000, "figure looks empty"


# --------------------------------------------------------------------------- #
# 7. reproducibility of the whole note (no hand transcription)
# --------------------------------------------------------------------------- #

def test_note_regenerates_from_the_artefact():
    """Re-rendering the note from the committed artefact must not change it."""
    import subprocess
    import sys
    script = REPO / "benchmarks" / "case_poiseuille" / "make_wall_layer_envelope_doc.py"
    before = DOC.read_text()
    png_before = hashlib.sha256(FIG.read_bytes()).hexdigest()
    subprocess.run([sys.executable, str(script)], cwd=REPO, check=True,
                   capture_output=True, timeout=300,
                   env={"PYTHONPATH": f"{REPO / 'src'}:{REPO}", "PATH": "/usr/bin:/bin"})
    assert DOC.read_text() == before, "the note is not a pure function of the artefact"
    assert hashlib.sha256(FIG.read_bytes()).hexdigest() == png_before