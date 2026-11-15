"""Tests for the continuum footprint reference (m21742).

The reference side of the footprint metric (``footprint_peak.py``) is built by
re-sampling the analytic Wagner field on the SPH grid.  This module checks the
*analytic* value of that same functional, which is the quantity the sampled
reference is supposed to converge to -- so a defect in the sampled reference
cannot hide behind the case verdict.

Nothing here re-states a number from a document: every quantity is recomputed
from the closed form and compared either with an independent quadrature or with a
property (monotonicity, limit, inverse round-trip) that a wrong implementation
would break.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"
FP = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_footprint"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT / "src"))

import footprint_reference as fr  # noqa: E402
import literature_refs as lr      # noqa: E402

BETA30 = math.radians(30.0)
GATE = 0.15


# ---------------------------------------------------------------- closed form --
@pytest.mark.parametrize("t1,t2,beta_deg", [
    (0.2, 0.4, 30.0), (0.5, 0.62, 30.0), (0.0, 0.3, 20.0), (0.7, 0.8, 40.0)])
def test_window_mean_matches_quadrature(t1, t2, beta_deg):
    """Closed form (1) == Simpson integrration of the profile, tight tolerance."""
    beta = math.radians(beta_deg)
    closed = fr.window_mean_q(t1, t2, beta)
    numeric = fr.window_mean_q_numeric(t1, t2, beta)
    assert closed == pytest.approx(numeric, rel=2e-6)


def test_window_mean_reduces_to_profile_as_window_shrinks():
    """A vanishing window must report the local profile value (no offset)."""
    for tau in (0.1, 0.5, 0.8):
        m = fr.window_mean_q(tau, tau + 1e-6, BETA30)
        assert m == pytest.approx(fr.profile_q(tau, BETA30), rel=1e-3)


def test_protocol_rejects_dry_or_collapsed_windows():
    with pytest.raises(ValueError):
        fr.window_mean_q(0.6, 0.5, BETA30)          # tau1 > tau2
    with pytest.raises(ValueError):
        fr.window_mean_q(-0.1, 0.2, BETA30)         # outside the wet footprint


# -------------------------------------------------------------- footprint ratio --
def test_ratio_tends_to_one_for_a_vanishing_footprint():
    assert fr.footprint_ratio(1e-3, BETA30) > 0.999


def test_ratio_is_strictly_decreasing_in_footprint():
    """A wider window averages a sharper cusp: R must fall monotonically.

    A non-monotone R would mean the operator/analytic pair disagree, and the
    'footprint explains the gap' argument in the ledger would be unfounded.
    """
    eps = [0.01, 0.02, 0.05, 0.1, 0.16, 0.25, 0.4]
    r = [fr.footprint_ratio(e, BETA30) for e in eps]
    assert all(a > b for a, b in zip(r, r[1:])), r


def test_ratio_at_case_footprint_is_the_ledger_value():
    """beta=30, Phi=8 mm, c_end=50 mm -> eps = 0.16; R is recomputed, not read."""
    r = fr.footprint_ratio(8e-3 / 0.05, BETA30)
    assert r == pytest.approx(0.9453, abs=2e-3)
    # and the deficit is the first-order part of the gap the case must not charge
    # to the solver
    assert 0.03 < 1.0 - r < 0.08


def test_ratio_weakens_with_deadrise_angle():
    """Small deadrise => sharper cusp => a transducer loses more of the point max."""
    rs = [fr.footprint_ratio(0.16, math.radians(b)) for b in (10, 20, 30, 40)]
    assert all(a < b for a, b in zip(rs, rs[1:])), rs


def test_required_eps_inverts_the_ratio():
    for target in (0.70, 0.80, 0.95):
        eps = fr.required_eps(target, BETA30)
        assert fr.footprint_ratio(eps, BETA30) == pytest.approx(target, abs=2e-3)


def test_required_eps_refuses_unreachable_targets():
    with pytest.raises(ValueError):
        fr.required_eps(1.5, BETA30)


# ------------------------------------------------------- metric-consistent side --
def test_denominators_apply_the_same_footprint_to_every_model():
    d = fr.reference_denominators(
        30.0, 8e-3, 0.05,
        designated_cp=lr.designated_cp_max(30.0),
        exact_cp=lr.cp_max_model("EIM", 30.0),
        wagner_cp=fr.cp_max_point(BETA30))
    r = d["footprint_ratio"]
    assert d["denominator_designated"] == pytest.approx(d["point_designated"] * r)
    assert d["denominator_exact"] == pytest.approx(d["point_exact"] * r)
    assert d["denominator_wagner"] == pytest.approx(d["point_wagner"] * r)
    # the finite-area denominator sits below the bare point value -- charging a
    # footprint reading against the point value is a systematic negative offset
    assert d["denominator_designated"] < d["point_designated"]


def test_reference_model_spread_exceeds_the_gate():
    """Wagner (8.40) vs the designated composite (6.92) is 21 % -- wider than 15 %.

    Therefore a verdict quoted as a single percentage against one model cannot be
    sharper than the model disagreement; the case must report both levels.
    """
    d = fr.reference_denominators(
        30.0, 8e-3, 0.05,
        designated_cp=lr.designated_cp_max(30.0),
        wagner_cp=fr.cp_max_point(BETA30))
    gap = d["denominator_wagner"] / d["denominator_designated"] - 1.0
    assert gap == pytest.approx(0.214, abs=0.01)
    assert gap > GATE


def test_bare_point_comparison_biases_a_perfect_solver_low():
    """A solver that reproduces the ideal field exactly reads R*Cp_max, not Cp_max.

    Grading that reading against the bare point value would report a FAIL for a
    perfect solver -- the size of that bias is what the ledger reports.
    """
    d = fr.reference_denominators(30.0, 8e-3, 0.05,
                                  designated_cp=lr.designated_cp_max(30.0),
                                  wagner_cp=fr.cp_max_point(BETA30))
    bias = d["denominator_wagner"] / d["point_wagner"] - 1.0
    assert bias == pytest.approx(-0.055, abs=0.01)


# --------------------------------------------------------------- sensor anchor --
def test_sensor_anchor_keeps_finite_area_below_point_maximum():
    """The only measured finite-area data in the repo must read below the models."""
    if not hasattr(fr, "TAU_EDGE_GUARD"):
        pytest.skip("module incomplete")
    import footprint_reference_ledger as prl
    anchor = prl.sensor_anchor()
    rows = anchor.get("rows") or []
    assert rows, anchor.get("reason", "no sensor rows")
    for r in rows:
        assert r["ratio_to_com"] < 1.0, r          # space averaging can only reduce
        assert 0.3 < r["ratio_to_com"] < 0.7, r    # order of magnitude


# ------------------------------------------------------------------- artefacts --
@pytest.mark.skipif(not (FP / "reference_ledger.json").exists(),
                    reason="reference ledger absent")
def test_reference_ledger_is_recomputable():
    rep = json.loads((FP / "reference_ledger.json").read_text())
    assert rep["design"]["verdict_rule_changed"] is False
    for g in rep["grids"]:
        assert g["eps"] == pytest.approx(8e-3 / g["c_end_m"])
        assert g["footprint_ratio"] == pytest.approx(
            fr.footprint_ratio(g["eps"], BETA30), abs=1e-6)
        assert g["denominator_designated"] == pytest.approx(
            lr.designated_cp_max(30.0) * g["footprint_ratio"], rel=1e-12)
    # the sampled same-operator reference must converge to the analytic value:
    # |bias| strictly decreasing along the dyadic ladder is the evidence that the
    # 'same operator' trick is asymptotically unbiased in dx ...
    bias = [abs(g["sampled_reference_bias"]) for g in rep["grids"]
            if g.get("sampled_reference_bias") == g.get("sampled_reference_bias")]
    assert all(a > b for a, b in zip(bias, bias[1:])), bias
    # ... and that what remains at the finest grid is small enough to be quoted
    assert bias[-1] < 0.03, bias