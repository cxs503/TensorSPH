"""Deadrise-response axis of the wedge-entry case (m23230).

Three groups of checks, none of which needs an SPH run:

* **reference side** -- the in-repo Wagner anchor must be asymptotically exact
  (it is the reason the case can be graded at all) and must *not* be inside the
  case tolerance at the case angle; both statements are read out of
  ``refs.json`` (Jain et al. 2022, Table 1), never typed here.
* **no silent interpolation** -- a deadrise angle that has no published row must
  raise, so a ratio can never be formed from an interpolated literature table.
* **decision rule** -- the predeclared rule must be able to return PASS,
  FAIL *and* NOT_JUDGEABLE on synthetic artefacts, must group by grid (a ratio
  across two dx values is a grid error wearing a deadrise costume), and must
  widen the ratio uncertainty with the uncertainty of both readings.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
CASE = HERE / "benchmarks" / "case_sph_wedge_entry_2d"
for _p in (HERE / "src", CASE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import deadrise_response as dr  # noqa: E402
import literature_refs as lr  # noqa: E402


# --------------------------------------------------------------------------- #
# Reference side
# --------------------------------------------------------------------------- #

def test_envelope_asymptote_matches_wagner():
    """beta -> 0: the Wagner peak must coincide with the exact solution.

    This is the assumption the whole case rests on (the anchor is only usable
    because it is asymptotically exact).  At the smallest tabulated angle the
    two published families must agree with the closed form to well under 1 %.
    """
    env = dr.reference_envelope()
    assert env["asymptote"]["beta_deg"] <= 2.0
    assert abs(env["asymptote"]["wagner_vs_eim_rel"]) <= 0.01


def test_envelope_gap_grows_monotonically_with_beta():
    """Linearisation error must increase with deadrise (no sign flips)."""
    env = dr.reference_envelope()
    gaps = [r["wagner_vs_eim_rel"] for r in env["rows"]]
    assert gaps == sorted(gaps)
    assert gaps[0] > 0.0
    assert all(not math.isnan(g) for g in gaps)


def test_case_angle_anchor_is_outside_the_case_gate():
    """At 30 deg the asymptotic anchor alone eats the 15 % tolerance.

    Pinned so that no future round can quietly present "SPH vs Wagner within
    15 % at 30 deg" as agreement with the mission-designated reference.
    """
    env = dr.reference_envelope()
    validity = env["validity"]
    assert validity["case_angle"] == 30.0
    assert abs(validity["case_angle_wagner_vs_eim_rel"]) > validity["gate"]
    assert validity["max_angle_inside_gate"] < validity["case_angle"]


def test_reference_values_come_from_refs_json():
    """No literature number may be typed into the axis module."""
    refs = lr.load_refs()
    table = lr.table_ref(refs)["cp_max_rho_v2"]
    for beta, row in ((20.0, table["20"]), (30.0, table["30"]),
                      (40.0, table["40"])):
        assert dr.reference_cp_max(beta, "COM") == pytest.approx(2.0 * row["COM"])
        assert dr.reference_cp_max(beta, "EIM") == pytest.approx(2.0 * row["EIM"])


def test_wagner_branch_is_the_in_repo_closed_form():
    """``wagner_1932`` bypasses the table on purpose (the table has no row)."""
    from tensorsph.wedge_entry import wagner_peak_over_q

    for beta in (20.0, 30.0, 40.0):
        assert dr.reference_cp_max(beta, dr.DIAGNOSTIC) == pytest.approx(
            wagner_peak_over_q(math.radians(beta)))


def test_ratio_at_reference_angle_is_unity():
    for model in (dr.PRIMARY, dr.CROSSCHECK, dr.DIAGNOSTIC):
        assert dr.ratio_reference(dr.REFERENCE_ANGLE, model) == pytest.approx(1.0)


def test_untabulated_angle_raises_instead_of_interpolating():
    with pytest.raises(KeyError):
        dr.reference_cp_max(25.0, "COM")
    with pytest.raises(KeyError):
        dr.ratio_reference(35.0, "EIM")


# --------------------------------------------------------------------------- #
# Synthetic artefacts
# --------------------------------------------------------------------------- #

def _art(beta: float, dx: float, seed: int, obs: float,
         spread: float = 0.05) -> dict:
    return {
        "beta_deg": float(beta),
        "design": {"dx": float(dx)},
        "realisation": {"seed": int(seed)},
        "observed": {"peak_median": float(obs), "spread_rel": float(spread)},
        "reference": {"cp_com": dr.reference_cp_max(beta, dr.PRIMARY),
                      "cp_eim": dr.reference_cp_max(beta, dr.CROSSCHECK),
                      "cp_wagner": dr.reference_cp_max(beta, dr.DIAGNOSTIC)},
    }


def _synthetic(ratio_measured: float, dx: float = 4.0e-3,
               spread: float = 0.05) -> list[dict]:
    """Three seeds per angle with the *measured* ratio fixed by the caller."""
    ref30 = 8.0
    ref20 = ref30 * ratio_measured
    return [
        _art(20.0, dx, s, ref20 * (1.0 + 0.001 * s), spread) for s in (1, 2, 3)
    ] + [
        _art(30.0, dx, s, ref30 * (1.0 + 0.001 * s), spread) for s in (1, 2, 3)
    ]


def test_rule_passes_on_the_designated_ratio():
    summary = dr.analyse(_synthetic(dr.ratio_reference(20.0, dr.PRIMARY)))
    grid = summary["grids"]["0.004"]
    assert grid["ratios"][0]["verdict"] == "PASS_COM"
    assert grid["verdict"] == "RESPONSE_CONSISTENT"


def test_rule_fails_on_a_mismatched_ratio():
    """Negative control: a flat deadrise response cannot pass."""
    summary = dr.analyse(_synthetic(1.0))
    grid = summary["grids"]["0.004"]
    assert grid["ratios"][0]["verdict"] == "FAIL"
    assert grid["verdict"] == "RESPONSE_NOT_CONSISTENT"


def test_rule_refuses_to_grade_a_noisy_reading():
    """A reading whose own spread exceeds the gate is not judged either way."""
    summary = dr.analyse(_synthetic(dr.ratio_reference(20.0, dr.PRIMARY),
                                    spread=0.60))
    grid = summary["grids"]["0.004"]
    assert all(r["verdict"] == "NOT_JUDGEABLE" for r in grid["ratios"])
    assert grid["verdict"] == "NO_JUDGEABLE_ANGLE"


def test_grids_are_never_pooled():
    """Two grids with different readings stay separate (ratio is per grid)."""
    arts = _synthetic(dr.ratio_reference(20.0, dr.PRIMARY), dx=4.0e-3)
    arts += _synthetic(1.0, dx=2.0e-3)
    summary = dr.analyse(arts)
    assert set(summary["grids"]) == {"0.004", "0.002"}
    assert summary["grids"]["0.004"]["ratios"][0]["verdict"] == "PASS_COM"
    assert summary["grids"]["0.002"]["ratios"][0]["verdict"] == "FAIL"


def test_ratio_uncertainty_grows_with_both_readings():
    tight = dr.analyse(_synthetic(dr.ratio_reference(20.0, dr.PRIMARY),
                                  spread=0.02))
    loose = dr.analyse(_synthetic(dr.ratio_reference(20.0, dr.PRIMARY),
                                  spread=0.10))
    u_tight = tight["grids"]["0.004"]["ratios"][0]["ratio_uncertainty"]
    u_loose = loose["grids"]["0.004"]["ratios"][0]["ratio_uncertainty"]
    assert 0.0 < u_tight < u_loose


def test_gate_is_the_case_tolerance_not_a_new_one():
    assert dr.GATE == 0.15


# --------------------------------------------------------------------------- #
# Committed artefacts (skipped when the axis has not been run)
# --------------------------------------------------------------------------- #

ART = HERE / "docs" / "artifacts" / "case_sph_wedge_entry_2d_deadrise"
SUMMARY = ART / "deadrise_response.json"


@pytest.mark.skipif(not SUMMARY.exists(), reason="deadrise axis not run")
def test_committed_summary_matches_committed_artefacts():
    """The committed summary is reproducible from the committed artefacts."""
    summary = json.loads(SUMMARY.read_text())
    recomputed = dr.analyse(dr.load_artefacts(ART))
    assert set(summary["grids"]) == set(recomputed["grids"])
    for tag, grid in recomputed["grids"].items():
        committed = summary["grids"][tag]
        for beta, reading in grid["readings"].items():
            assert committed["readings"][beta]["peak_median"] == pytest.approx(
                reading["peak_median"], rel=1e-12)
        for got, ref in zip(grid["ratios"], committed["ratios"]):
            assert got["beta_deg"] == ref["beta_deg"]
            assert got["verdict"] == ref["verdict"]


@pytest.mark.skipif(not SUMMARY.exists(), reason="deadrise axis not run")
def test_committed_summary_declares_its_own_scope():
    summary = json.loads(SUMMARY.read_text())
    assert summary["design"]["statistic"] == dr.STATISTIC
    assert summary["design"]["gate"] == dr.GATE
    assert summary["design"]["primary"] == "COM"
    for tag, grid in summary["grids"].items():
        assert grid["dx"] == pytest.approx(float(tag))
        for beta, reading in grid["readings"].items():
            # every committed reading carries its uncertainty and judgeability
            assert reading["estimator_uncertainty"] == pytest.approx(
                max(v for v in (reading["rel_sd"], reading["band_spread_median"])
                    if v == v))
            assert reading["judgeable"] == (
                reading["estimator_uncertainty"] <= dr.GATE)