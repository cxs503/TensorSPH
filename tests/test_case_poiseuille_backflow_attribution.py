"""CASE-POISEUILLE-SPH -- gates for the C6 backflow-peak attribution record.

The record under test is ``benchmarks/case_poiseuille/out/backflow_attribution.json``,
written by ``benchmarks/case_poiseuille/backflow_attribution.py``.  These tests
are deliberately *claim* tests, not smoke tests: they re-derive every relation
the note asserts (the kernel-support criterion, the falsification of the
estimator and quadrature hypotheses, the gauge-coherence arithmetic, and the
bit-exact reproduction of the published C6 numbers), and they fail if the
record and the published combined artifact ever disagree -- which is what
stops a re-grading of the honest FAIL from slipping in silently.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmarks.case_poiseuille import backflow_attribution as ba  # noqa: E402

ART = REPO / "benchmarks" / "case_poiseuille" / "out" / "backflow_attribution.json"
COMBINED = REPO / "benchmarks" / "case_poiseuille" / "out" / "combined.json"


@pytest.fixture(scope="module")
def rec():
    if not ART.exists():
        pytest.skip(f"attribution record not generated: {ART}")
    return json.loads(ART.read_text())


def _rows(rec):
    rows = list(rec["resolution_sweep"]) + list(rec["boundary_probe"])
    for key in rec["column_sweep"]:
        rows.extend(rec["column_sweep"][key])
    return rows


# --------------------------------------------------------------------------- #
# 1. schema and provenance
# --------------------------------------------------------------------------- #

def test_record_is_self_describing(rec):
    for key in ("case", "anchor", "title", "generated_by", "parameters",
                "analytic", "hypotheses", "declared_gate", "resolution_sweep",
                "verdict_table", "column_effect", "summary"):
        assert key in rec, key
    assert rec["anchor"] == "CASE-POISEUILLE-SPH"
    assert rec["parameters"]["gate_C6_pct"] == 5.0
    assert rec["generated_by"].split("/")[-1] in (
        "backflow_attribution.py", "backflow_attribution_from_rows.py")
    assert rec["summary"]["wall_clock_s"] > 0.0


def test_analytic_extremum_matches_closed_form(rec):
    """The peak height used everywhere is the closed form's, not a fitted one."""
    h, nu, g, uw = 0.1, 0.01, -4.905, 1.47
    a, b = -g / (2 * nu), g * h / (2 * nu) + uw / h
    y_ext = -b / (2 * a)
    u_ext = a * y_ext**2 + b * y_ext
    assert math.isclose(rec["analytic"]["extremum_y"], y_ext, rel_tol=1e-12)
    assert math.isclose(rec["analytic"]["extremum_u"], u_ext, rel_tol=1e-12)
    # it sits at 0.2003 H, i.e. one fifth of the channel up from the lower wall
    assert 0.19 < y_ext / h < 0.21


# --------------------------------------------------------------------------- #
# 2. the declared (raw) estimator is preserved bit-for-bit
# --------------------------------------------------------------------------- #

def test_declared_numbers_reproduce_the_published_artifact(rec):
    """No silent re-grading: the raw-min values must equal combined.json's."""
    if not COMBINED.exists():
        pytest.skip("combined.json absent")
    stored = {r["resolution_H_over_dx"]: r["backflow"]["peak_pct"]
              for r in json.loads(COMBINED.read_text())["runs"]}
    seen = 0
    for row in rec["resolution_sweep"]:
        res = row["resolution_H_over_dx"]
        if res in stored and row["n_cols"] == 8:
            assert math.isclose(row["declared"]["peak_pct"], stored[res],
                                rel_tol=1e-9, abs_tol=1e-9), res
            seen += 1
    assert seen == 3, f"expected to re-check the three published rungs, got {seen}"


def test_honest_fail_is_not_covered_up(rec):
    """The coarse rungs still fail the declared 5 % gate in this record."""
    coarse = {r["resolution_H_over_dx"]: r for r in rec["resolution_sweep"]}
    for res in (8, 12):
        assert coarse[res]["declared"]["pass_5pct"] is False, res
        assert abs(coarse[res]["declared"]["peak_pct"]) > 5.0, res
    assert "FAIL" in rec["declared_gate"]["verdict_preserved"]


# --------------------------------------------------------------------------- #
# 3. H1 -- the kernel-support criterion predicts the verdict
# --------------------------------------------------------------------------- #

def test_h1_criterion_predicts_every_verdict(rec):
    table = rec["verdict_table"]
    assert table["n"] == len(_rows(rec))
    assert table["misses"] == [], f"criterion missed: {table['misses']}"
    assert table["hypothesis_H1_supported"] is True


def test_h1_criterion_is_not_vacuous(rec):
    """Both labels must occur, otherwise 'predicts everything' means nothing."""
    rows = rec["resolution_sweep"]
    inside = [r for r in rows if r["inside_kernel_support"]]
    outside = [r for r in rows if not r["inside_kernel_support"]]
    assert inside and outside
    assert any(not r["declared"]["pass_5pct"] for r in inside)
    assert all(r["declared"]["pass_5pct"] for r in outside)
    # the coarse rungs are the ones whose extremum row is inside one support
    for r in rows:
        expected = r["extremum_row_over_dx"] < rec["parameters"]["support_over_dx"]
        assert r["inside_kernel_support"] is expected


def test_error_decays_with_resolution(rec):
    rows = sorted(rec["resolution_sweep"], key=lambda r: r["resolution_H_over_dx"])
    err = [abs(r["declared"]["peak_pct"]) for r in rows]
    assert err[-1] < err[0] / 4.0, err
    # observed order: fit a slope of log(err) vs log(H/dx) over the failing rungs
    fail = [r for r in rows if not r["declared"]["pass_5pct"]]
    if len(fail) >= 2:
        x = np.log([r["resolution_H_over_dx"] for r in fail])
        yl = np.log([abs(r["declared"]["peak_pct"]) for r in fail])
        slope = float(np.polyfit(x, yl, 1)[0])
        assert slope < 0.0, slope


# --------------------------------------------------------------------------- #
# 4. H2 falsified -- the estimator is not the cause
# --------------------------------------------------------------------------- #

def test_h2_estimator_swap_does_not_rescue_the_coarse_rungs(rec):
    """The interior-parabola fit fails wherever the raw minimum fails."""
    for r in rec["resolution_sweep"]:
        if r["inside_kernel_support"]:
            assert r["fit"]["pass_5pct"] is False, r["resolution_H_over_dx"]
            # same sign, same order of magnitude: the bias is in the profile
            assert (r["fit"]["pct"] * r["declared"]["peak_pct"]) > 0
            assert 0.5 < abs(r["fit"]["pct"] / r["declared"]["peak_pct"]) < 2.0
        else:
            assert abs(r["fit"]["pct"]) < 5.0, r["resolution_H_over_dx"]


def test_fit_extremum_recovers_an_exact_parabola():
    """Synthetic control: on clean data the estimator is exact to 1e-10."""
    y = np.linspace(0.005, 0.095, 9)
    a, b, c = +245.25, -9.825, 0.0   # u_yy > 0, interior minimum (physical)
    u = a * y**2 + b * y + c
    got = ba.fit_extremum(y, u)
    y_true = -b / (2 * a)
    assert math.isclose(got["y"], y_true, rel_tol=1e-10)
    assert math.isclose(got["u"], a * y_true**2 + b * y_true, rel_tol=1e-10)
    assert got["a"] > 0.0  # u_yy = -g/nu > 0: upward parabola, interior minimum


def test_fit_rejects_a_wrong_curvature_sign():
    y = np.linspace(0.005, 0.095, 9)
    u = -200.0 * y**2 - 5.0 * y        # opens down: no interior minimum
    got = ba.fit_extremum(y, u)
    assert math.isnan(got["u"]) and math.isnan(got["y"])


# --------------------------------------------------------------------------- #
# 5. H0 falsified -- transverse quadrature noise is not the cause
# --------------------------------------------------------------------------- #

def test_h0_column_sweep_keeps_the_verdict(rec):
    for res, block in rec["column_effect"].items():
        if not isinstance(block, dict):
            continue
        assert len(block["peak_pct_by_n_cols"]) >= 3
        # same verdict for every transverse sample count
        assert block["all_fail"] != block["all_pass"]
        # and the error does not collapse: it stays on the same side of the band
        assert block["spread_pp"] < 30.0, (res, block["spread_pp"])


# --------------------------------------------------------------------------- #
# 6. the scale-coherence arithmetic (why the gate is 15x tighter in absolute terms)
# --------------------------------------------------------------------------- #

def test_scale_block_arithmetic(rec):
    peak = abs(rec["analytic"]["extremum_u"])
    for r in rec["resolution_sweep"]:
        sc = r["scale"]
        assert math.isclose(sc["peak_abs"], peak, rel_tol=1e-12)
        assert math.isclose(sc["gauge_over_peak"], sc["gauge"] / peak, rel_tol=1e-12)
        # 5 % of the peak, expressed against the gauge used by every other gate
        assert math.isclose(sc["C6_tolerance_in_gauge_pct"],
                            100.0 * 0.05 * peak / sc["gauge"], rel_tol=1e-9)
        assert math.isclose(sc["C6_tolerance_m_s"], 0.05 * peak, rel_tol=1e-12)
        # the tolerance really is an order of magnitude tighter than C1's 5 %
        assert sc["C6_tolerance_in_gauge_pct"] < 1.0
        # the identity that explains the FAIL: peak-unit error = gauge-unit error
        assert math.isclose(sc["rms_error_pct_of_peak"],
                            sc["rms_error_pct_of_gauge"] * sc["gauge_over_peak"],
                            rel_tol=1e-9)


def test_fail_is_reachable_from_the_profile_error_alone(rec):
    """A C1-compliant profile is already allowed to fail C6, by construction.

    C1 permits 5 % of the gauge anywhere in the interior band; C6 permits 5 % of
    the peak, which is only 1/gauge_over_peak of that.  So the coarse rungs can
    be inside C1 and outside C6 at the same time -- which is exactly what the
    record shows, and what makes the FAIL a gate-scale artefact rather than a
    solver defect.
    """
    for r in rec["resolution_sweep"]:
        if r["inside_kernel_support"]:
            assert r["profile_rms_pct"] < 5.0
            assert not r["declared"]["pass_5pct"]
            assert r["scale"]["max_row_error_pct_of_gauge"] < 5.0


def test_profile_rows_are_consistent(rec):
    for r in rec["resolution_sweep"]:
        rows = r["rows"]
        assert len(rows) == round(0.1 / r["dx"])
        for row in rows:
            assert math.isclose(row["residual_pct_of_peak"],
                                100.0 * (row["u"] - row["u_analytic"])
                                / abs(rec["analytic"]["extremum_u"]), rel_tol=1e-9)