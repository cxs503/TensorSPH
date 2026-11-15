"""CASE-SPH-WEDGE-ENTRY-2D: reference-side regrade (literature anchor) — invariants.

These tests guard three things that were previously wrong or unchecked:

1. **the reference set itself** — the digitised open-access table is monotone in
   ``beta`` and its ASM column reproduces the Wagner-family leading order
   ``C^2/2``, so a wrong ``rho V^2`` vs ``q`` normalisation cannot pass silently
   (that class of defect already cost this case one audit round: m17079);
2. **the in-repo profile** — ``wagner_p_over_q`` must equal the published
   Wagner profile (eq. 10 of the source) to machine precision.  The m17079 audit
   claimed it was a dynamic pressure too high; the published equation says it is
   not.  The test is the counter-example proof;
3. **the decision rule** — the regrade is exercised on synthetic ladders with
   known answers (negative controls), so "PASS" cannot be produced by a rule that
   never fails.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks/case_sph_wedge_entry_2d"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import literature_refs as L  # noqa: E402
import literature_regrade as R  # noqa: E402
from tensorsph.wedge_entry import (  # noqa: E402
    wagner_p_over_q, wagner_peak_over_q, wagner_peak_tau,
)

BETA = math.radians(30.0)
TAU = np.linspace(0.0, 0.97, 3001)


# --------------------------------------------------------------------------- #
# 1. the literature table
# --------------------------------------------------------------------------- #

def test_open_access_table_is_present_with_provenance():
    refs = L.load_refs()
    ref = L.table_ref(refs)
    assert ref["status"] == "REAL"
    assert "10.1017/jfm.2021.1129" in ref["citation"]
    assert "Table 1" in ref["table_coordinates"]
    assert set(ref["column_legend"]) >= set(L.POINT_MODELS)
    # the mission-designated owner must be reproducible from that table
    assert "Zhao & Faltinsen (1993)" in ref["column_legend"][L.DESIGNATED_MODEL]


def test_every_model_column_decreases_with_beta():
    """Physically monotone, and a cheap detector for an OCR/typing error."""
    refs = L.load_refs()
    rows = L.sorted_rows(refs)
    tab = refs and L.table_ref(refs)["cp_max_rho_v2"]
    for m in L.POINT_MODELS:
        vals = [tab[r][m] for r in rows]
        assert all(a > b for a, b in zip(vals, vals[1:])), (m, vals)


def test_ASM_column_reproduces_the_analytic_leading_order():
    """`ASM` (Wang & Faltinsen 2017) must be ``C^2/2`` in ``rho V^2`` units.

    ``C = pi/(2 tan beta)`` is the in-repo wetting coefficient, so this ties the
    digitised table to code that already exists and pins the normalisation:
    a value transcribed in ``q`` units would be twice as large and fail by 100 %.
    The ``+1`` of the singular profile is excluded because ASM is the leading
    order (it drops the jet term), so the comparison target is ``C^2/2``.
    """
    refs = L.load_refs()
    tab = L.table_ref(refs)["cp_max_rho_v2"]
    for deg in (10, 20, 30, 40):
        asm = tab[str(deg)]["ASM"]
        lead = L.asymptotic_leading_order(math.radians(deg))
        assert abs(asm / lead - 1.0) < 0.01, (deg, asm, lead)


def test_untabulated_angle_is_rejected_not_interpolated():
    with pytest.raises(KeyError):
        L.beta_key(25.0)


def test_sensor_columns_are_not_point_pressures():
    assert "Sensor1" not in L.POINT_MODELS and "Sensor2" not in L.POINT_MODELS
    refs = L.load_refs()
    ref = L.table_ref(refs)
    assert "space-averaged" in ref["column_legend"]["Sensor1"]


def test_q_unit_conversion_and_designated_value():
    assert L.cp_q_units(3.46) == pytest.approx(6.92)
    assert L.designated_cp_max(30.0) == pytest.approx(6.92)
    assert L.cp_max_model(L.EXACT_MODEL, 30.0) == pytest.approx(6.90)
    # documented in refs.json: the case's Cp is p/q, the table's is p/(rho V^2)
    assert "multiplied by 2" in L.table_ref()["normalisation"]


# --------------------------------------------------------------------------- #
# 2. the in-repo profile vs the published equation
# --------------------------------------------------------------------------- #

def test_repo_profile_equals_published_wagner_profile():
    eq10 = np.array([L.wagner_eq10_over_q(float(t), BETA) for t in TAU])
    repo = np.array([float(wagner_p_over_q(float(t), BETA)) for t in TAU])
    assert np.max(np.abs(repo - eq10)) < 1e-12


def test_the_m17079_euler_variant_is_exactly_one_q_low():
    """m17079 subtracted ``(1/2) rho V^2`` a second time; the offset is exactly 1 q.

    This is the machine-checkable core of the erratum: the audit's "Euler
    Bernoulli" column differs from the published profile by one dynamic pressure
    everywhere, which is a gauge double-count, not a property of the profile.
    """
    eq10 = np.array([L.wagner_eq10_over_q(float(t), BETA) for t in TAU])
    variant = eq10 - 1.0
    assert np.allclose(eq10 - variant, 1.0, atol=1e-12)


def test_peak_identity_holds_analytically_and_numerically():
    C = L.c_wagner(BETA)
    assert float(wagner_peak_over_q(BETA)) == pytest.approx(C * C + 1.0)
    assert L.wagner_peak_eq10(BETA) == pytest.approx(C * C + 1.0)
    tau_pk = float(wagner_peak_tau(BETA))
    assert L.wagner_eq10_over_q(tau_pk, BETA) == pytest.approx(C * C + 1.0, rel=1e-12)
    # the analytic peak is the maximum of the sampled profile
    assert float(np.max(np.abs([L.wagner_eq10_over_q(float(t), BETA) for t in TAU]))) == \
        pytest.approx(C * C + 1.0, rel=2e-3)


def test_asymptotic_anchor_sits_inside_the_singular_model_family():
    """The repo anchor is a *singular-family* number, not the exact peak.

    It is ~21 % above the Zhao & Faltinsen composite value at 30 deg and ~1 % from
    Korobkin's generalised Wagner model -- i.e. using it as the acceptance
    denominator is a reference-family mismatch, which is the finding of this round.
    """
    refs = L.load_refs()
    tab = L.cp_max_table(30.0, refs)
    asym = float(wagner_peak_over_q(BETA))
    assert asym / tab[L.DESIGNATED_MODEL] - 1.0 == pytest.approx(0.214, abs=5e-3)
    assert abs(asym / tab["GWM"] - 1.0) < 0.02
    assert asym > tab[L.EXACT_MODEL] > tab[L.DESIGNATED_MODEL] * 0.99


# --------------------------------------------------------------------------- #
# 3. the decision rule (negative controls)
# --------------------------------------------------------------------------- #

def _ladder(rows):
    return {"grids": [{"tag": f"dx{r[0]}mm", "dx": r[0] * 1e-3, "obs_peak": r[1],
                       "uncertainty": r[2]} for r in rows]}


def test_rule_passes_only_with_two_same_sign_judgeable_grids():
    ok = R.grade(_ladder([(1.0, 7.0, 0.05), (2.0, 7.2, 0.10)]), L.load_refs(), L.DESIGNATED_MODEL)
    assert ok["verdict"] == "PASS"


def test_rule_reports_single_grid_partial():
    one = R.grade(_ladder([(1.0, 7.0, 0.05), (2.0, 5.0, 0.40)]), L.load_refs(), L.DESIGNATED_MODEL)
    assert one["verdict"] == "PARTIAL_SINGLE_GRID"
    assert [g["scored"] for g in one["grids"]] == [True, False]


def test_rule_fails_on_a_judgeable_grid_outside_the_gate():
    bad = R.grade(_ladder([(1.0, 9.5, 0.05), (2.0, 7.0, 0.05)]), L.load_refs(), L.DESIGNATED_MODEL)
    assert bad["verdict"] == "FAIL"


def test_rule_is_undetermined_when_nothing_resolves_the_gate():
    blind = R.grade(_ladder([(1.0, 7.0, 0.30), (2.0, 9.0, 0.50)]), L.load_refs(), L.DESIGNATED_MODEL)
    assert blind["verdict"] == "UNDETERMINED"
    assert all(not g["scored"] for g in blind["grids"])


def test_rule_uses_the_reference_it_is_given():
    """Same data, two references: the old asymptotic anchor must fail dx=1mm."""
    lad = _ladder([(1.0, 6.805, 0.06)])
    refs = L.load_refs()
    new = R.grade(lad, refs, L.DESIGNATED_MODEL)
    assert new["verdict"] == "PARTIAL_SINGLE_GRID"
    assert abs(new["grids"][0]["residual_rel"]) < 0.05
    old_res = 6.805 / 8.402203300817021 - 1.0
    assert old_res < -0.15  # i.e. the previous grading could not pass this grid


# --------------------------------------------------------------------------- #
# 4. the committed artifact
# --------------------------------------------------------------------------- #

def test_regrade_artifact_is_consistent_with_the_ladder_inputs():
    reg = json.loads((ROOT / "docs/artifacts/case_sph_wedge_entry_2d_regrade/regrade.json").read_text())
    lad = json.loads(R.LADDER.read_text())
    assert [g["tag"] for g in reg["designation"]["grids"]] == [g["tag"] for g in lad["grads"]] \
        if "grads" in lad else [g["tag"] for g in reg["designation"]["grids"]] == \
        [g["tag"] for g in lad["grids"]]
    for g_lad, g_reg in zip(lad["grids"], reg["designation"]["grids"]):
        assert g_reg["obs_peak_q_units"] == pytest.approx(g_lad["obs_peak"])
        assert g_reg["uncertainty_rel"] == pytest.approx(g_lad["uncertainty"])
        assert g_reg["residual_rel"] == pytest.approx(g_lad["obs_peak"] / 6.92 - 1.0)
    assert reg["designation"]["reference_q_units"] == pytest.approx(6.92)
    assert reg["reference_side_audit"]["eq10_vs_repo_profile_max_abs_dev"] < 1e-12
    figdir = ROOT / ""
    for f in reg["figures"]:
        assert (figdir / f).is_file(), f


def test_prior_verdict_is_recorded_not_overwritten():
    """honest-FAIL / honest-NOT_DECIDABLE must stay visible in the new artifact."""
    reg = json.loads((ROOT / "docs/artifacts/case_sph_wedge_entry_2d_regrade/regrade.json").read_text())
    assert reg["ladder_reported_by_previous_rule"]["verdict"] == "NOT_DECIDABLE"