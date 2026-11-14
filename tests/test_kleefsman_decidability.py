"""Tests for the Kleefsman decidability audit (acceptance-metric uncertainty).

The case-pool rule for ``CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`` is relative
(``< 10 %``), so the stored evidence must be free of *definitional* and
*numerical* ambiguity larger than that tolerance before any PASS/FAIL can be
issued.  These tests pin the audit's individual checks on synthetic
proof-of-defect inputs (so a regression cannot hide behind plausible-looking
committed numbers) and pin the findings the historical evidence bundle must
keep reporting.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "kleefsman_obstacle"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT / "src"))

import decidability as dc  # noqa: E402


def run_row(dx=0.02, t_end=1.2, peak=0.5, peak_t=0.7, steps=6001,
            sound_speed=20.0, compression=0.01, max_speed=6.6):
    return {"dx": dx, "t_end": t_end, "peak_height_m": peak, "peak_time_s": peak_t,
            "steps": steps, "simulated_time_s": t_end, "sound_speed": sound_speed,
            "max_density_compression": compression, "max_speed": max_speed,
            "arrival_time_s": 0.45}


# --------------------------------------------------------------------------- #
# horizon censoring
# --------------------------------------------------------------------------- #
def test_peak_on_the_last_sample_is_censoring_not_a_measurement():
    row = run_row(t_end=1.2, peak_t=1.200128)          # stored dx0.02_c040 run
    res = dc.check_horizon(row)
    assert res["censored"] is True
    assert res["slack_s"] < res["margin_s"]


def test_peak_well_inside_the_horizon_is_accepted():
    res = dc.check_horizon(run_row(t_end=3.0, peak_t=1.1))
    assert res["censored"] is False
    assert res["slack_s"] == pytest_approx(1.9)


def test_censoring_uses_an_absolute_floor_for_short_runs():
    # 1.0 s horizon: 5 % is 0.05 s, so a peak at 0.966 s must be flagged
    assert dc.check_horizon(run_row(t_end=1.0, peak_t=0.966))["censored"] is True


# --------------------------------------------------------------------------- #
# acoustic Courant
# --------------------------------------------------------------------------- #
def test_courant_recomputes_dt_from_the_stored_step_count():
    # dx = 0.01, c0 = 20 m/s -> limit 1.95e-4 s; 2056 steps over 1.0 s -> 4.86e-4 s
    res = dc.check_courant(run_row(dx=0.01, t_end=1.0, steps=2056), )
    assert res["ok"] is False
    assert res["ratio_to_limit"] == pytest_approx(2.49, rel=0.02)
    assert res["courant"] > res["max_courant"]


def test_courant_accepts_a_step_at_the_limit():
    dx, c0 = 0.02, 20.0
    dt = dc.dt_acoustic(dx, c0)
    res = dc.check_courant(run_row(dx=dx, sound_speed=c0, t_end=1.2, steps=round(1.2 / dt)))
    assert res["ok"] is True
    assert 0.99 * dc.MAX_COURANT <= res["courant"] <= 1.001 * dc.MAX_COURANT


def test_courant_constants_match_the_solver_defaults():
    from tensorsph.droplet_impact import Config

    cfg = Config()
    assert dc.MAX_COURANT == cfg.max_courant
    assert dc.H_FACTOR == cfg.h_factor


# --------------------------------------------------------------------------- #
# compressibility
# --------------------------------------------------------------------------- #
def test_compression_criterion_and_ten_u_rule():
    res = dc.check_compression(run_row(compression=0.12, max_speed=6.64, sound_speed=20.0))
    assert res["ok"] is False
    assert res["ten_u_rule_ok"] is False
    assert res["sound_speed_for_10U"] > res["sound_speed"]
    good = dc.check_compression(run_row(compression=0.004, max_speed=6.0, sound_speed=80.0))
    assert good["ok"] is True and good["ten_u_rule_ok"] is True


# --------------------------------------------------------------------------- #
# sound-speed family
# --------------------------------------------------------------------------- #
def _family(peaks, speeds=(20.0, 40.0, 80.0)):
    return [{"sound_speed": c, "peak_height_m": p} for c, p in zip(speeds, peaks)]


def test_family_fit_recovers_the_incompressible_limit():
    # peak = 0.40 + 20 / c0^2 exactly
    rows = _family([0.40 + 20.0 / c ** 2 for c in (20.0, 40.0, 80.0)])
    fit = dc.compressibility_family(rows)
    assert fit["status"] == "OK"
    assert fit["a_incompressible_peak_m"] == pytest_approx(0.40)
    assert fit["monotone_decreasing"] is True
    assert fit["max_abs_residual_m"] < 1e-12


def test_family_flags_a_non_monotone_sound_speed_response():
    fit = dc.compressibility_family(_family([0.4255, 0.4475, 0.5131]))  # stored dx0.04 row
    assert fit["monotone_decreasing"] is False
    assert fit["b_m2_per_s2"] < 0            # negative slope: peak grows with c0


def test_family_needs_three_points():
    assert dc.compressibility_family(_family([0.5]))["status"] == "INSUFFICIENT_POINTS"


# --------------------------------------------------------------------------- #
# budget arithmetic
# --------------------------------------------------------------------------- #
def test_combine_is_root_sum_square_and_skips_nan():
    assert dc.combine(0.03, 0.04) == pytest_approx(math.hypot(0.03, 0.04))
    assert dc.combine(0.03, float("nan")) == pytest_approx(0.03)
    assert math.isnan(dc.combine(float("nan"))) or dc.combine(float("nan")) == 0.0


def test_decide_requires_uncertainty_below_tolerance():
    assert dc.decide(0.05, 0.10)["decidable"] is True
    bad = dc.decide(0.34, 0.10)
    assert bad["decidable"] is False and bad["headroom_rel"] < 0


# --------------------------------------------------------------------------- #
# committed evidence
# --------------------------------------------------------------------------- #
RESULTS = BENCH / "results"
HISTORICAL = ("dx0.04", "dx0.02", "dx0.01")


def test_committed_evidence_reports_the_blocking_defects():
    rep = dc.audit(RESULTS, tags=HISTORICAL)
    ids = {f["id"] for f in rep["findings"]}
    assert "MIXED-HORIZON" in ids              # 1.0 s vs 1.2 s horizons
    assert "CENSORED-dx0.01" in ids            # peak at 0.966 s of a 1.0 s run
    assert "COURANT-dx0.01" in ids             # implied step 2.5x the limit
    assert "COMPRESSION-dx0.02" in ids         # 12 % density error at c0 = 20
    blocking = {f["id"] for f in rep["findings"] if f["severity"] == "blocking"}
    assert "CENSORED-dx0.01" in blocking and "MIXED-HORIZON" in blocking


def test_historical_bundle_cannot_decide_a_ten_percent_tolerance():
    rep = dc.audit(RESULTS, tags=HISTORICAL)
    budget = rep["uncertainty_budget"]
    assert budget["verdict"]["decidable"] is False
    assert budget["total_rel"] > rep["tolerance"]
    assert budget["n_uncensored_grids"] >= 2


def test_audit_discovers_every_stored_run_record():
    records = dc.discover(RESULTS)
    for tag in HISTORICAL:
        assert tag in records
    assert all({"dx", "peak_height_m"} <= set(r) for r in records.values())


def test_gauge_operator_spread_is_attached_when_available():
    rep = dc.audit(RESULTS, tags=HISTORICAL)
    spread = rep["grids"]["dx0.02"]["operator_spread"]
    assert 2.426 in spread
    assert spread[2.426]["spread_rel"] > 0.05        # 11.9 % stored


def pytest_approx(value, rel=1e-6):
    import pytest

    return pytest.approx(value, rel=rel)

# --------------------------------------------------------------------------- #
# frozen acceptance definition
# --------------------------------------------------------------------------- #
def test_reference_json_freezes_exactly_one_primary_operator():
    ref = json.loads((BENCH / "reference.json").read_text())
    frozen = ref["metric_definition"]["frozen_primary"]
    assert frozen["operator"] == "slab"
    assert frozen["aperture_m"] == pytest_approx(0.08)
    assert frozen["station_offset_m"] == pytest_approx(0.05)
    assert frozen["peak_filter_window_s"] == pytest_approx(0.03)
    # the frozen aperture must be the one the ladder used, otherwise the frozen
    # definition and the evidence would describe different gauges
    ladder = json.loads((RESULTS / "gauge_ladder" / "ladder.json").read_text())
    assert {row["gauge_aperture_m"] for row in ladder["rows"]} == {frozen["aperture_m"]}
    # and the operator must be one the solver actually implements
    import tensorsph.kleefsman_obstacle as ko

    assert hasattr(ko, "slab_level")
