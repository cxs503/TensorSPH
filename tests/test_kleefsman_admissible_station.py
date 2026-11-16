"""Invariants for the admissible-station scoring of the Kleefsman H2 criterion.

These tests exist because the acceptance verdict for
``CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`` is a *number plus a rule*, and both can
silently drift: the digitised reference, the frozen protocol, the station, and
the pass/borderline classification are all editable.  The tests therefore

* pin the measurement primitives (peak / arrival) to an **analytic** series with
  a known crossing and a known maximum -- so a broken interpolator or a changed
  filter width cannot pass as "a small numerical difference";
* pin the reference controls to the anchored Fig. 19 values;
* pin the station lattice arithmetic (2.426 m excluded, 2.228 m inside 1 cm of
  the gate anchor) so the station cannot be moved back by editing one number;
* pin the decision rule with synthetic rows (PASS / BORDERLINE / FAIL /
  WITHHELD), independently of whether the repository's CSV records exist;
* lock the committed scored numbers as a regression, so a re-run that changes
  the verdict has to also change this file and the doc.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "benchmarks" / "kleefsman_obstacle"
sys.path.insert(0, str(CASE))
sys.path.insert(0, str(ROOT / "src"))

import score_admissible_station as sas  # noqa: E402
from score_t30 import arrival, peak  # noqa: E402


# --------------------------------------------------------------------------
# analytic control for the measurement primitives
# --------------------------------------------------------------------------
def analytic_series(dt: float = 0.005, t_end: float = 3.0) -> tuple[list, list]:
    """Ramp -> plateau -> spike, with every feature known in closed form.

    h(t) = 0                                   t < 0.2
         = 0.1 * (t - 0.2)                     0.2 <= t < 0.4   (crosses 0.02 at 0.4)
         = 0.4                                 0.4 <= t < 1.0
         = 0.5                                 1.0 <= t < 1.2   (the maximum)
         = 0.4 - 0.05*(t-1.2)                  t >= 1.2
    """
    t, h = [], []
    n = int(round(t_end / dt)) + 1
    for i in range(n):
        x = i * dt
        if x < 0.2:
            v = 0.0
        elif x < 0.4:
            v = 0.1 * (x - 0.2) * 5.0   # 0.02 m at t = 0.24
        elif x < 1.0:
            v = 0.4
        elif x < 1.2:
            v = 0.5
        else:
            v = 0.4 - 0.05 * (x - 1.2)
        t.append(x)
        h.append(v)
    return t, h


def test_arrival_interpolation_lands_on_the_analytic_crossing() -> None:
    t, h = analytic_series()
    # ramp reaches 0.02 m at t = 0.24 s (see analytic_series docstring)
    assert arrival(t, h, 0.02, "interp") == pytest.approx(0.24, abs=1e-9)
    # sample mode returns the first sample at or above the threshold, which is
    # reported separately precisely because it is quantised
    assert arrival(t, h, 0.02, "sample") >= arrival(t, h, 0.02, "interp") - 1e-12


def test_peak_is_exact_and_filter_does_not_shift_a_plateau() -> None:
    t, h = analytic_series()
    val, when = peak(t, h, 0.0)
    assert val == pytest.approx(0.5, abs=1e-12)
    assert when == pytest.approx(1.0, abs=1e-9)
    val_f, when_f = peak(t, h, 0.03)   # frozen 30 ms symmetric filter
    # a 30 ms boxcar over a 120 ms plateau must not move the maximum
    assert val_f <= 0.5 + 1e-12
    assert when_f < 1.2


def test_arrival_threshold_not_reached_is_nan() -> None:
    t, h = analytic_series()
    assert math.isnan(arrival(t, h, 0.9, "interp"))


def test_front_speed_is_distance_over_time() -> None:
    assert sas.front_speed(1.228, 2.228, 0.5) == pytest.approx(2.0)
    assert math.isnan(sas.front_speed(1.228, 2.228, float("nan")))


def test_observed_order_recovers_a_known_second_order_sequence() -> None:
    # q(dx) = q0 + C dx^2 with q0 = 1.0, C = 1.0 -> exact second order
    fine, mid, coarse = 1.0 + 0.01 ** 2, 1.0 + 0.02 ** 2, 1.0 + 0.04 ** 2
    assert sas.observed_order(fine, mid, coarse) == pytest.approx(2.0, abs=1e-6)
    # a constant sequence carries no order information
    assert math.isnan(sas.observed_order(1.0, 1.0, 1.0))


def test_richardson_recovers_the_exact_limit_of_a_known_sequence() -> None:
    assert sas.richardson(1.0001, 1.0004, 2.0, 2.0) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# reference and station bookkeeping
# --------------------------------------------------------------------------
def test_reference_controls_reproduce_the_anchored_values() -> None:
    ref, series = sas.load_reference()
    t, h = sas.load_series(series, "h_m")
    proto = sas.frozen_protocol(ref)
    ev = ref["experimental_values"]
    pk, _ = peak(t, h, proto["peak_filter_window_s"])
    assert sas.rel_error(pk, float(ev["peak_height_m"])) < 5e-3
    for mode, key in (("interp", "arrival_time_s"), ("sample", "arrival_time_s")):
        got = arrival(t, h, proto["arrival_threshold_m"], mode)
        assert sas.rel_error(got, float(ev["arrival_time_s"])) < 0.05
    # the two digitisation conventions must differ, and not by more than 5 %
    interp = arrival(t, h, proto["arrival_threshold_m"], "interp")
    sample = arrival(t, h, proto["arrival_threshold_m"], "sample")
    assert sample > interp
    assert (sample - interp) / interp < 0.05


def test_station_lattice_excludes_the_committed_station() -> None:
    """The station must be re-derivable from the printed probe spacing alone."""
    ref, _ = sas.load_reference()
    geometry = ref["geometry_m"]
    pool = float(geometry["column_length"])          # reservoir / tank gate
    spacing = 0.496                                  # Amaro Junior et al. 2020
    gate_anchor = pool + 2 * spacing                 # H4 at the gate -> H2 = 2.220 m
    assert gate_anchor == pytest.approx(2.220, abs=1e-9)
    # the station this job scores at is within one coarse cell of that anchor
    assert abs(2.228 - gate_anchor) < 0.02
    # the historically committed station would put H4 inside the tank
    implied_h4 = 2.426 - 2 * spacing
    assert implied_h4 > pool


# --------------------------------------------------------------------------
# the decision rule, tested without any simulation data
# --------------------------------------------------------------------------
def _row(metric: str, ref: float, got: float, pass_: bool) -> dict:
    return {"metric": metric, "ref": ref, "got": got,
            "signed_rel_error": (got - ref) / ref, "rel_error": abs(got - ref) / ref,
            "pass": pass_}


def _grid(rows: list[dict], *, censored: bool = False, dx: float = 0.01) -> list[dict]:
    rec = {"status": "measured", "censored": censored,
           "peak_m": rows[0]["got"], "peak_time_s": 1.0,
           "arrival_time_s_interp": rows[1]["got"], "rows": rows}
    return [{"tag": "synthetic", "dx": dx, "status": "scored",
             "gauges": {"slab@2.228": rec}}]


REF = {"arrival_definition_band_rel": 0.032, "peak_time_s": 1.7027}


def test_decide_does_not_let_a_good_arrival_rescue_a_bad_peak() -> None:
    """Both acceptance quantities must pass; a metric may not be traded off."""
    grids = _grid([_row("peak_height", 0.37, 0.48, False),      # +29.7 %
                   _row("arrival_time_interp", 0.3481, 0.35, True),
                   _row("arrival_time_sample", 0.3591, 0.35, True)])
    out = sas.decide(grids, 0.10, REF)
    assert out["verdict"] == "FAIL"
    assert "peak" in out["why"]


def test_decide_clean_pass() -> None:
    grids = _grid([_row("peak_height", 0.37, 0.37, True),
                   _row("arrival_time_interp", 0.3481, 0.35, True),
                   _row("arrival_time_sample", 0.3591, 0.35, True)])
    assert sas.decide(grids, 0.10, REF)["verdict"] == "PASS"


def test_decide_borderline_when_the_two_definitions_disagree() -> None:
    # 0.3869 s: +11.1 % against 0.3481 but +8.5 % against 0.3591
    grids = _grid([_row("peak_height", 0.37, 0.3674,
                        abs(0.3674 - 0.37) / 0.37 < 0.10),
                   _row("arrival_time_interp", 0.3481, 0.3869, False),
                   _row("arrival_time_sample", 0.3591, 0.3869, True)])
    out = sas.decide(grids, 0.10, REF)
    assert out["verdict"] == "BORDERLINE"
    assert "not decidable" in out["why"]
    # the shortfall against the stricter definition is reported, not rounded away
    assert out["arrival_needed_for_10pct_s"]["shortfall_s"] > 0


def test_decide_fails_when_both_definitions_are_outside() -> None:
    grids = _grid([_row("peak_height", 0.37, 0.3674, True),
                   _row("arrival_time_interp", 0.3481, 0.44, False),
                   _row("arrival_time_sample", 0.3591, 0.44, False)])
    assert sas.decide(grids, 0.10, REF)["verdict"] == "FAIL"


def test_decide_withholds_on_a_censored_peak() -> None:
    grids = _grid([_row("peak_height", 0.37, 0.37, True),
                   _row("arrival_time_interp", 0.3481, 0.35, True),
                   _row("arrival_time_sample", 0.3591, 0.35, True)], censored=True)
    assert sas.decide(grids, 0.10, REF)["verdict"] == "WITHHELD"


def test_decide_withholds_when_no_grid_is_scored() -> None:
    assert sas.decide([{"status": "not_run", "gauges": {}}], 0.10, REF)["verdict"] == "WITHHELD"


# --------------------------------------------------------------------------
# regression lock on the committed record
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scored() -> dict:
    path = CASE / "results" / "admissible_station" / "score.json"
    if not path.is_file():
        pytest.skip("run score_admissible_station.py to regenerate results")
    import json
    return json.loads(path.read_text())


def test_committed_verdict_is_borderline_not_pass(scored: dict) -> None:
    """Honest-state lock: the case is NOT a clean PASS at the 10 % level."""
    assert scored["verdict"]["verdict"] == "BORDERLINE"
    assert scored["verdict"]["grid_tag"] == "dx0.01"
    assert scored["verdict"]["station_x_m"] == 2.228


def test_committed_dx001_numbers(scored: dict) -> None:
    v = scored["verdict"]
    assert v["peak"]["got"] == pytest.approx(0.3674, abs=1e-3)
    assert v["peak"]["signed_rel_error"] == pytest.approx(-0.007, abs=2e-3)
    assert v["arrival"]["got"] == pytest.approx(0.3869, abs=1e-3)
    assert v["arrival"]["interp"]["signed_rel_error"] == pytest.approx(0.111, abs=3e-3)
    assert v["arrival"]["sample"]["signed_rel_error"] == pytest.approx(0.085, abs=3e-3)


def test_committed_station_defect_has_the_expected_sign(scored: dict) -> None:
    """At every grid the excluded station must be *worse* than the admissible one."""
    n = 0
    for g in scored["grids"]:
        if g["status"] != "scored":
            continue
        a = g["gauges"].get("slab@2.228")
        b = g["gauges"].get("slab@2.426")
        if not (a and b) or a.get("status") != "measured":
            continue
        ea = {r["metric"]: r["rel_error"] for r in a["rows"]}
        eb = {r["metric"]: r["rel_error"] for r in b["rows"]}
        assert eb["peak_height"] > ea["peak_height"]
        assert eb["arrival_time_interp"] > ea["arrival_time_interp"]
        n += 1
    assert n >= 3


def test_arrival_error_is_converged_so_refinement_cannot_absorb_it(scored: dict) -> None:
    """The residual arrival bias must be a converged quantity, not a grid effect.

    If this fails (GCI grows), the whole "model error, not resolution" reading of
    the case collapses -- so it is asserted rather than described.
    """
    lad = scored["ladder"]["arrival"]
    assert lad["status"] == "ok"
    assert lad["order_observed"] == pytest.approx(2.9, abs=0.5)
    assert lad["gci_fine_rel"] < 0.01
    # continuous-limit arrival stays outside 10 % w.r.t. the interpolated
    # reference and inside it w.r.t. the sample-quantised one
    ref = scored["reference"]
    v0 = lad["value_h0"]
    assert (v0 - ref["arrival_interp_s"]) / ref["arrival_interp_s"] > 0.10
    assert (v0 - ref["arrival_sample_s"]) / ref["arrival_sample_s"] < 0.10


def test_peak_agreement_is_not_asymptotic_and_peak_time_diverges(scored: dict) -> None:
    """Guards the honest caveat: the -0.7 % peak is a zero crossing, not convergence."""
    lad = scored["ladder"]["peak"]
    assert lad["status"] == "ok"
    assert lad["order_observed"] < 1.0        # non-asymptotic
    # ... so no Richardson limit may be quoted for the peak
    assert lad["non_asymptotic"] is True
    assert lad["value_h0"] is None
    # peak *time* error grows monotonically as the grid is refined
    errs = []
    for g in sorted((g for g in scored["grids"] if g["status"] == "scored"),
                    key=lambda d: -d["dx"]):
        rec = g["gauges"].get("slab@2.228")
        if rec and rec.get("status") == "measured":
            errs.append(abs(rec["peak_time_s"] - scored["reference"]["peak_time_s"])
                        / scored["reference"]["peak_time_s"])
    assert len(errs) >= 3
    assert errs[0] < errs[1] < errs[2]
    assert errs[-1] > 0.15


def test_front_speed_deficit_is_consistent_across_operators(scored: dict) -> None:
    rows = [r for r in scored["front_speed"]["rows"] if r["dx"] == 0.01]
    assert len(rows) == 3
    for r in rows:
        assert r["speed_deficit_rel"] == pytest.approx(-0.10, abs=0.02)
        assert r["speed_m_s"] == pytest.approx(
            sas.front_speed(scored["front_speed"]["gate_x_m"], 2.228,
                            r["arrival_interp_s"]))

def test_eight_mm_station_ambiguity_is_immaterial(scored: dict) -> None:
    """2.220 m vs 2.228 m must not be able to flip the verdict on its own.

    Both the interpolated estimate and the two measured station-lattice pairs are
    asserted, because the verdict is quoted at 2.228 m while the lattice anchor
    is 2.220 m -- if this shifted arrival by more than the 3.2 % reference band,
    the station choice itself would be the deciding factor and the verdict would
    have to be re-taken at 2.220 m.
    """
    for row in scored["station_8mm_estimate"]:
        assert abs(row["implied_shift_rel"]) < 0.015
    measured = scored.get("station_sensitivity_measured") or []
    assert measured, "the m22476 station-lattice records should be pickable up"
    for row in measured:
        assert row["aperture_m_lattice_run"] == pytest.approx(0.08)
        assert abs(row["arrival_delta_rel"]) < 0.02
