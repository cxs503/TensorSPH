"""Grading test for the POISEUILLE-SPH validity study.

The anchor ``CASE-POISEUILLE-SPH`` is certified at one operating point
(``g = 9.81``, ``Ma = 0.06``).  ``benchmarks/poiseuille_mirror/study_ma_envelope.py``
asks what that certificate covers away from the declared point.  This test grades
the *committed artifact only* -- it never runs the solver -- and it does three
jobs an ordinary "read the number and compare" test would not:

1. **Binding** -- every published metric is recomputed from the stored per-row
   profile means, the stored gates are re-derived from the stored metrics, and
   the control point is cross-checked against the committed resolution ladder in
   ``out/results.json``.  The verdict cannot drift away from its data.

2. **No over-certification** -- the outcome must be one of the predeclared
   statuses, and the statement must not claim a Mach envelope: the measured
   pattern falsifies the "Ma <= 0.1 cliff" reading, so a future artifact that
   concludes ``H2_SUPPORTED`` would contradict its own data and fail here.

3. **No un-controlled confounder presented as a law** -- the accumulated-time
   index ``Psi`` is allowed only as a *post-hoc* correlation: the test requires
   the artifact to carry its open item (the ``c_s`` / ``dt`` confounder is not
   separated) and refuses any claim that the index is validated.

Regenerate with:
    PYTHONPATH=src:. python benchmarks/poiseuille_mirror/study_ma_envelope.py --points all
    PYTHONPATH=src:. python benchmarks/poiseuille_mirror/make_envelope_doc.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "benchmarks" / "poiseuille_mirror" / "out" / "ma_envelope.json"
LADDER = REPO / "benchmarks" / "poiseuille_mirror" / "out" / "results.json"
TOL = 5.0
POINT_IDS = ("P0_baseline", "P1_low_ma", "P2_high_ma", "P3_cert_re_high_ma",
             "P4_cert_re_high_ma_long", "P5_re_bracket", "P6_short_time_high_ma")


@pytest.fixture(scope="module")
def doc():
    assert ART.exists(), (
        "missing study artifact; regenerate with "
        "python benchmarks/poiseuille_mirror/study_ma_envelope.py --points all")
    return json.loads(ART.read_text())


@pytest.fixture(scope="module")
def points(doc):
    return {r["id"]: r for r in doc["points"]}


# --------------------------------------------------------------------------- #
# 1. binding: recompute the published metrics from the raw rows
# --------------------------------------------------------------------------- #

def test_all_designed_points_are_present(points):
    for pid in POINT_IDS:
        assert pid in points, f"point {pid} missing from the artifact"


@pytest.mark.parametrize("pid", POINT_IDS)
def test_metrics_are_recomputable_from_rows(points, pid):
    r = points[pid]
    rows = r["rows"]
    assert rows, f"{pid} carries no per-row profile"
    H, nu, g = r["height"], r["viscosity"], r["body_force"]

    u_max = max(row["u_mean"] for row in rows)
    u_ref = g * H * H / (8.0 * nu)
    assert u_max == pytest.approx(r["u_max_measured"], rel=1e-12)
    assert u_ref == pytest.approx(r["u_max_reference"], rel=1e-12)
    assert 100.0 * (u_max / u_ref - 1.0) == pytest.approx(r["u_max_error_pct"],
                                                         abs=1e-9)

    interior = [row for row in rows if 0.1 <= row["y_over_H"] <= 0.9]
    rms = math.sqrt(sum(row["rel_error"] ** 2 for row in interior) / len(interior))
    assert 100.0 * rms == pytest.approx(r["profile_rms_interior_pct"], abs=1e-9)

    for row in rows:                      # the analytic column is the parabola
        u_a = g * (H - row["y"]) * row["y"] / (2.0 * nu)
        assert u_a == pytest.approx(row["u_analytic"], rel=1e-12)

    # accumulated-time index is reported, never hand-entered
    psi = u_max ** 2 * r["t_end"] / (r["sound_speed"] * r["dx"])
    assert psi == pytest.approx(r["psi_index"], rel=1e-12)


@pytest.mark.parametrize("pid", POINT_IDS)
def test_stored_gates_follow_from_stored_metrics(points, pid):
    r = points[pid]
    g = r["gates"]
    assert g["A_centreline_5pct"] == (abs(r["u_max_error_pct"]) <= TOL)
    assert g["B_profile_rms_5pct"] == (r["profile_rms_interior_pct"] <= TOL)
    assert g["C_wall_shear_5pct"] == (
        abs(r["wall_shear_bottom_error_pct"]) <= TOL
        and abs(r["wall_shear_top_error_pct"]) <= TOL)
    assert r["passes_all_gates"] == all(g.values())


def test_control_point_matches_the_committed_ladder(points):
    """P0 must reproduce the H/dx = 12 rung of the committed anchor artifact."""
    r = points["P0_baseline"]
    ladder = json.loads(LADDER.read_text())
    rung = [x for x in ladder["results"]
            if x["resolution_H_over_dx"] == r["resolution_H_over_dx"]][0]
    assert r["u_max_error_pct"] == pytest.approx(rung["u_max_error_pct"], abs=0.5)
    assert r["profile_rms_interior_pct"] == pytest.approx(
        rung["profile_rms_interior_pct"], abs=0.5)
    assert r["max_mach"] == pytest.approx(rung["max_mach"], rel=0.05)


# --------------------------------------------------------------------------- #
# 2. the predeclared discrimination rule and its (recorded) revisions
# --------------------------------------------------------------------------- #

def test_verdict_is_one_of_the_predeclared_outcomes(doc):
    assert doc["verdict"]["status"] in (
        "H1_SUPPORTED", "H2_SUPPORTED", "H3_SUPPORTED", "H4_SUPPORTED",
        "MIXED_MA_TIME", "UNEXPLAINED")


def test_rule_version_is_recorded(doc):
    assert doc["verdict"]["rule_version"] == "v3"


def test_ma_envelope_is_not_claimed(doc):
    """The measured pattern falsifies the "Ma <= 0.1 cliff" reading."""
    v = doc["verdict"]
    assert v["status"] != "H2_SUPPORTED"
    assert "H2_acoustic_envelope" not in v or not v["H2_acoustic_envelope"]


def test_measured_pattern_holds(points):
    """Pass = {P0,P1,P3,P5,P6}, fail = {P2,P4} -- with the matched-Re contrast."""
    for pid in ("P0_baseline", "P1_low_ma", "P3_cert_re_high_ma",
                "P5_re_bracket", "P6_short_time_high_ma"):
        assert points[pid]["passes_all_gates"] is True, pid
    for pid in ("P2_high_ma", "P4_cert_re_high_ma_long"):
        assert points[pid]["passes_all_gates"] is False, pid


def test_matched_re_contrast_isolates_ma(doc, points):
    """P0 vs P3: same Re, Ma doubled by lowering c_s, band untouched."""
    c = doc["verdict"]["same_re_contrast"]
    assert c["re_matched_rel"] < 0.01
    assert c["ma_high"] > 1.95 * c["ma_low"]
    assert c["rms_ratio"] < 1.1, "raising Ma at fixed Re must not move the band"


def test_accumulated_time_effect_at_fixed_re(points):
    """Same Ma and Re, longer run: 0.6 s and 1.0 s clean, 2.0 s outside."""
    p3, p4, p6 = (points["P3_cert_re_high_ma"], points["P4_cert_re_high_ma_long"],
                  points["P6_short_time_high_ma"])
    assert p6["t_end"] < p3["t_end"] < p4["t_end"]
    assert abs(p3["max_mach"] - p4["max_mach"]) < 0.01
    assert p3["profile_rms_interior_pct"] == pytest.approx(
        p6["profile_rms_interior_pct"], abs=1.0)
    assert p4["profile_rms_interior_pct"] > 2.0 * p3["profile_rms_interior_pct"]


def test_psi_index_separates_but_is_not_sold_as_a_law(doc):
    """Psi must separate the points *and* be flagged post-hoc with an open item."""
    v = doc["verdict"]
    sep = v["psi"]["separation"]
    assert sep["cleanly_separated"] is True
    assert sep["psi_min_failing"] > 1.4 * sep["psi_max_passing"]
    assert "post-hoc" in v["statement"]
    assert v.get("open_item"), "the c_s/dt confounder must stay declared as open"


def test_declared_operating_point_sits_inside_the_certificate(doc, points):
    env = doc["verdict"]["certificate_envelope"]
    p0 = points["P0_baseline"]
    assert env["declared_point"]["psi"] == pytest.approx(p0["psi_index"])
    assert env["declared_point"]["psi"] < 0.5 * doc["verdict"]["psi_failing_min"]
    assert "not covered" in env["statement"] or "未" in env["statement"]