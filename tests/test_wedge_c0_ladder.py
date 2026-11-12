"""Tests for the CASE-SPH-WEDGE-ENTRY-2D sound-speed ladder (artefact-gated).

The heavy numbers come from ``benchmarks/case_sph_wedge_entry_2d/c0_ladder.py``
(written to ``artifacts/.../c0_ladder/summary.json``).  The tests that need that
artefact skip when it is absent (CI does not run the benchmarks); the tests that
pin the design and the estimator always run.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"
sys.path.insert(0, str(BENCH))

import c0_ladder as ladder  # noqa: E402


def test_plan_is_a_sound_speed_sweep_not_a_grid_ladder():
    specs = ladder.plan()
    assert specs, "the plan must not be empty"
    for s in specs:
        # grade() runs its GCI branch on kind == "ladder"; a c0 sweep at fixed dx
        # would divide by a spacing ratio of one, so it must not carry that kind.
        assert s["kind"] != "ladder"
        assert s["kind"] == "c0_sweep"
        assert s["sound_speed_factor"] in ladder.C0_FACTORS
        assert s["dx"] in ladder.DX_LADDER
        assert s["beta_deg"] == pytest.approx(30.0)
    # every (dx, c0) pair of the predeclared matrix is present exactly once
    pairs = {(s["dx"], s["sound_speed_factor"]) for s in specs}
    assert len(pairs) == len(ladder.DX_LADDER) * len(ladder.C0_FACTORS)


def test_loglog_slope_recovers_the_acoustic_scaling():
    c0 = [20.0, 40.0, 80.0]
    eps = [4.0e-2, 1.0e-2, 0.25e-2]          # exact (c0/V)^-2
    assert ladder._loglog_slope(c0, eps) == pytest.approx(-2.0, abs=1e-9)
    # a floor (constant offset) flattens the slope, as it should
    assert ladder._loglog_slope(c0, [e + 1e-3 for e in eps]) > -2.0
    assert math.isnan(ladder._loglog_slope([20.0], [1e-2]))


SUMMARY = ladder.ART / "summary.json"


@pytest.mark.skipif(not SUMMARY.exists(), reason="benchmark artefacts not generated")
def test_summary_errors_recompute_from_the_artefacts():
    s = json.loads(SUMMARY.read_text())
    ref = s["reference"]["wagner_cp_max"]
    assert ref == pytest.approx(ladder.case.wagner_peak_over_q(math.radians(30.0)))
    for row in s["cases"]:
        assert row["err_band_peak"] == pytest.approx(
            (row["cp_band_peak"] - ref) / ref, rel=1e-9)
        assert row["c0_over_V"] in ladder.C0_FACTORS
        # the run must stay finite and weakly compressible
        assert math.isfinite(row["cp_band_peak"])
        assert row["max_rho_ratio"] > 1.0


@pytest.mark.skipif(not SUMMARY.exists(), reason="benchmark artefacts not generated")
def test_credit_verdict_follows_the_predeclared_rule():
    s = json.loads(SUMMARY.read_text())
    credit = s["credit"]
    if credit["verdict"] == "NO_DATA":
        pytest.skip("no artefacts")
    lad = s["ladders"][credit["evaluated_at"]]
    eps = lad["density_excursion"]
    errs = lad["err_band_peak"]
    cond1 = (eps[0] / eps[-1]) >= ladder.MIN_DENSITY_DROP
    cond2 = (abs(errs[0]) - abs(errs[-1])) >= ladder.MIN_CP_ERROR_IMPROVEMENT
    cond3 = all(abs(v - 1.0) <= ladder.MASS_GATE for v in lad["max_rho_ratio"])
    assert credit["cond1_density_falls"] is bool(cond1)
    assert credit["cond2_cp_error_improves"] is bool(cond2)
    assert credit["cond3_mass_gate_intact"] is bool(cond3)
    assert credit["verdict"] == ("CREDITED" if (cond1 and cond2 and cond3)
                                 else "NOT_CREDITED")


@pytest.mark.skipif(not SUMMARY.exists(), reason="benchmark artefacts not generated")
def test_no_reference_value_is_invented():
    s = json.loads(SUMMARY.read_text())
    # the mission-designated reference stays withheld; only the analytic Wagner
    # anchor is graded, and it is computed in-repo, never typed
    assert s["reference"]["zhao_faltinsen_1993"] == "WITHHELD"
    for row in s["cases"]:
        assert "zhao" not in json.dumps(row).lower()
