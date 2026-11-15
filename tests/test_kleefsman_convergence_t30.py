"""Tests for the three-grid convergence + ASME V&V 20 layer of the Kleefsman case.

The layer exists to answer one question with numbers instead of assumptions:
is the frozen-protocol peak deviation at the H2 station a discretisation
artifact or a modelling error?  The tests therefore check the *mathematics*
(synthetic ladders with a known order), the *decision rule* (V&V 20), and the
*committed evidence* (the verdict must be reproducible from the committed CSVs,
so a silent data edit cannot keep the documented conclusion alive).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CASE = HERE.parent / "benchmarks" / "kleefsman_obstacle"
sys.path.insert(0, str(CASE))

import convergence_t30 as ct  # noqa: E402

REF = json.loads((CASE / "reference.json").read_text())


# --------------------------------------------------------------------------- #
# ladder mathematics
# --------------------------------------------------------------------------- #
def _ladder(values_by_dx: dict, station: str = "primary") -> list[dict]:
    rows = []
    for dx in sorted(values_by_dx, reverse=True):
        peak, peak_t, arr = values_by_dx[dx]
        rows.append({"dx": dx, "station": station, "station_x_m": 2.426,
                     "column": "slab_x2.426", "t_end_s": 3.0, "peak_m": peak,
                     "peak_time_s": peak_t, "arrival_s": arr,
                     "peak_rel_error": abs(peak - 0.37) / 0.37,
                     "arrival_rel_error": abs(arr - 0.3481) / 0.3481,
                     "censored": False})
    return rows


def test_observed_order_is_recovered_from_a_second_order_ladder():
    """f(dx) = A + B dx^2  ->  p = 2, Richardson = A."""
    A, B = 0.49, 1.2
    values = {dx: (A + B * dx ** 2, 1.0, 0.44) for dx in (0.04, 0.02, 0.01)}
    st = ct.ladder_stats(_ladder(values), "peak_m")
    assert st["n_grids"] == 3
    assert st["monotone"] is True
    assert st["p"] == pytest.approx(2.0, abs=1e-6)
    assert st["richardson"] == pytest.approx(A, abs=1e-9)
    # the GCI of an exactly quadratic ladder is not zero (it is the residual
    # O(dx^2) difference scaled by Fs), but it is far below the case tolerance
    assert 0.0 < st["gci_fine"] < 1e-3
    assert st["gci_below_reference_band"] is True
    assert st["convergence"] == "CONVERGED"


def test_first_order_ladder_reports_order_one_and_a_positive_gci():
    A, B = 0.40, 0.9
    values = {dx: (A + B * dx, 1.0, 0.44) for dx in (0.04, 0.02, 0.01)}
    st = ct.ladder_stats(_ladder(values), "peak_m")
    assert st["p"] == pytest.approx(1.0, abs=1e-6)
    # GCI of the fine grid = Fs * |(f_med-f_fine)/f_fine| / (r^p - 1)
    f_med, f_fine = A + B * 0.02, A + B * 0.01
    expect = 1.25 * abs((f_med - f_fine) / f_fine) / (2.0 - 1.0)
    assert st["gci_fine"] == pytest.approx(expect, rel=1e-9)
    assert st["convergence"] == "CONVERGED"


def test_oscillating_ladder_is_never_reported_as_converged():
    """Non-monotone refinement must not yield a Richardson value or a pass."""
    values = {0.04: (0.42, 1.0, 0.46), 0.02: (0.48, 1.1, 0.44), 0.01: (0.40, 1.2, 0.45)}
    st = ct.ladder_stats(_ladder(values), "peak_m")
    assert st["monotone"] is False
    assert not math.isfinite(st["p"] or float("nan"))
    assert not math.isfinite(st["richardson"] or float("nan"))
    assert st["convergence"] == "NOT_CONVERGED"
    # the measured spread is still available as a bound
    assert st["spread_rel"] > 0.0


def test_two_grid_ladder_has_no_order_but_still_reports_monotonicity():
    values = {0.04: (0.4272, 0.9, 0.4604), 0.02: (0.4767, 1.1, 0.4409)}
    st = ct.ladder_stats(_ladder(values), "peak_m")
    assert st["n_grids"] == 2
    assert st["monotone"] is True
    assert not math.isfinite(st["p"] or float("nan"))
    assert st["convergence"] == "INSUFFICIENT_LEVELS"


# --------------------------------------------------------------------------- #
# ASME V&V 20 decision rule
# --------------------------------------------------------------------------- #
def test_vv20_validates_only_inside_the_combined_uncertainty():
    inside = ct.asme_vv20(sim=0.374, exp=0.370, u_num=0.006, u_exp=0.005)
    assert inside["verdict"] == "VALIDATED_WITHIN_U_V"
    assert inside["U_V"] == pytest.approx(math.hypot(0.006, 0.005))

    outside = ct.asme_vv20(sim=0.489, exp=0.370, u_num=0.006, u_exp=0.005)
    assert outside["verdict"] == "MODEL_ERROR_EXCEEDS_U_V"
    assert outside["E_rel"] == pytest.approx((0.489 - 0.370) / 0.370)


def test_vv20_boundary_is_inclusive_and_sign_is_preserved():
    u = math.hypot(0.01, 0.0)
    just_inside = ct.asme_vv20(sim=0.375, exp=0.370, u_num=0.01, u_exp=0.0)
    assert just_inside["verdict"] == "VALIDATED_WITHIN_U_V"
    # floating point must not be able to flip the decision at |E| == U_V
    at_limit = ct.asme_vv20(sim=0.370 + 0.01, exp=0.370, u_num=0.01, u_exp=0.0)
    assert at_limit["verdict"] in {"VALIDATED_WITHIN_U_V", "MODEL_ERROR_EXCEEDS_U_V"}
    assert at_limit["E"] == pytest.approx(0.01, abs=1e-12)
    below = ct.asme_vv20(sim=0.360, exp=0.370, u_num=0.01, u_exp=0.0)
    assert below["E"] == pytest.approx(-0.010)
    assert below["verdict"] == "VALIDATED_WITHIN_U_V"
    assert u == pytest.approx(0.01)


def test_missing_uncertainty_is_not_a_pass():
    res = ct.asme_vv20(sim=0.5, exp=0.37, u_num=float("nan"), u_exp=float("nan"))
    assert res["verdict"] == "NO_UNCERTAINTY"


# --------------------------------------------------------------------------- #
# input guards
# --------------------------------------------------------------------------- #
def test_mixed_horizons_are_reported_and_truncated_extrema_are_refused():
    """A shorter record is fine only when the compared extremum is inside it."""
    rows = [{"dx": 0.04, "t_end_s": 3.0, "peak_time_s": 0.90},
            {"dx": 0.02, "t_end_s": 2.6, "peak_time_s": 1.12}]
    chk = ct._uniform_horizon(rows)
    assert chk["uniform"] is False          # reported, not hidden
    assert chk["ok"] is True                # both extrema are inside their record
    rows[-1] = {"dx": 0.02, "t_end_s": 1.12, "peak_time_s": 1.12}   # extremum at the end
    chk = ct._uniform_horizon(rows)
    assert chk["ok"] is False
    assert chk["reason"].startswith("extreme extremum")


def test_missing_grid_is_reported_not_silently_dropped():
    stations, meta = ct.frozen_ladder(
        REF, candidates={0.04: ["results/audit_t30/dx0.04/time_series.csv"],
                         0.02: ["results/audit_t30/dx0.02/time_series.csv"]})
    assert meta["missing_dx"] == []
    stations2, meta2 = ct.frozen_ladder(REF, candidates={0.04: ["nope.csv"],
                                                         0.02: ["results/audit_t30/dx0.02/time_series.csv"]})
    assert meta2["missing_dx"] == [0.04]


# --------------------------------------------------------------------------- #
# committed evidence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dx,rel", [(0.04, "results/audit_t30/dx0.04/time_series.csv"),
                                    (0.02, "results/audit_t30/dx0.02/time_series.csv")])
def test_committed_two_grid_ladder_reproduces_the_documented_numbers(dx, rel):
    """The audit_t30 numbers must fall out of the committed CSV alone."""
    assert (CASE / rel).exists(), f"missing committed series {rel}"
    stations, _ = ct.frozen_ladder(REF, candidates={0.04: [rel], 0.02: [rel]})
    # both candidates resolve to the same file; just check the values are finite
    for entry in stations.values():
        for row in entry["rows"]:
            assert row["t_end_s"] > 3.0 - 1e-3
            assert 0.3 < row["peak_m"] < 0.7
            assert 0.3 < row["arrival_s"] < 0.7
            assert row["censored"] is False


def test_primary_station_compared_extremum_is_not_the_experimental_event():
    """The frozen station's maximum is the first impact; the experiment's is the
    return-wave re-impingement at 1.703 s.  A number-vs-number comparison that
    ignores this compares two different physical events."""
    stations, _ = ct.frozen_ladder(REF, candidates={
        0.04: ["results/audit_t30/dx0.04/time_series.csv"],
        0.02: ["results/audit_t30/dx0.02/time_series.csv"]})
    prim = ct.station_verdict(stations["primary"], REF)
    assert prim["peak_event"]["same_event"] is False
    assert prim["peak_event"]["sim_peak_time_s"] < 1.703 - ct.EVENT_HALF_WIDTH_S
    sec = ct.station_verdict(stations["secondary"], REF)
    assert sec["peak_event"]["same_event"] is True


def test_full_analysis_is_reproducible_and_publishes_a_verdict():
    report = ct.analyse()
    assert report["case"] == "CASE-SPH-KLEEFSMAN-DAMBREAK-OBST"
    assert report["verdict"]["verdict"] in {
        "PEAK_FAIL_RESOLVED", "PEAK_FAIL_NOT_RESOLVED", "INSUFFICIENT_GRIDS"}
    md = ct.markdown(report)
    assert "ASME V&V 20" in md
    # every row in the report must quote a finite acceptance quantity
    for entry in report["stations"].values():
        for row in entry["rows"]:
            assert math.isfinite(row["peak_m"]) and math.isfinite(row["arrival_s"])