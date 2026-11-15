"""Guards for the ``t_end = 3.0 s`` Kleefsman scoring harness (score_t30.py).

Context: the two committed long runs (``results/audit_t30/dx0.02`` and
``dx0.04``) were produced while the experimental reference was still
``WITHHELD``, so their ``comparison`` block says ``WITHHELD`` and the peak
criterion was recorded as "not comparable, run window too short".  Both runs
actually reach 3.0 s, which covers the reference peak at 1.703 s: the criterion
is decidable from data already on disk.  ``score_t30.py`` does that scoring
under the frozen protocol.

These tests protect the three things that can silently break such a verdict:

1. the frozen protocol (operator / station / filter / threshold) still matches
   ``reference.json``;
2. the digitised reference still resolves despite its relocated path, and the
   scoring operators reproduce the *anchored* numbers and the paper's own
   COMFLOW cross-check;
3. the grading operators behave analytically (interpolation, smoothing,
   threshold crossing, robustness to the digitisation convention).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "benchmarks", "kleefsman_obstacle", "score_t30.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("kleefsman_score_t30", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sc = _load_module()


@pytest.fixture(scope="module")
def scored():
    return sc.score()


# --------------------------------------------------------------------------
# 1. protocol
# --------------------------------------------------------------------------
def test_frozen_protocol_matches_reference_json():
    ref, _ = sc.load_reference()
    proto = sc.frozen_protocol(ref)
    fp = ref["metric_definition"]["frozen_primary"]
    assert proto["operator"] == "slab"
    assert proto["station_offset_m"] == pytest.approx(fp["station_offset_m"])
    assert proto["peak_filter_window_s"] == pytest.approx(0.03)
    assert proto["arrival_threshold_m"] == pytest.approx(0.02)
    assert fp["frozen_by"] and fp["frozen_on"], "the freeze must stay attributable"


def test_protocol_drift_is_rejected(monkeypatch):
    """A silent edit of reference.json must not silently change the verdict."""
    ref, _ = sc.load_reference()
    ref["metric_definition"]["frozen_primary"]["peak_filter_window_s"] = 0.10
    with pytest.raises(ValueError, match="acceptance protocol changed"):
        sc.frozen_protocol(ref)


# --------------------------------------------------------------------------
# 2. reference integrity
# --------------------------------------------------------------------------
def test_digitised_series_path_is_root_relative_and_resolves():
    """The recorded path used to be ``../../outputs/...``, i.e. a top-level tree
    that does not exist; it is now repo-root-relative and must resolve."""
    ref, path = sc.load_reference()
    recorded = ref["experimental_values"]["digitised_series"]
    assert not recorded.startswith(".."), recorded
    assert path.is_file() and path.name == "h2_fig19.csv"
    assert path == (sc.ROOT / recorded).resolve()
    # the relocation target the loader falls back to must keep existing, so a bad
    # string can never make the harness silently skip the reference
    assert (sc.ROOT / "docs/m20884/outputs/kleefsman_h2_ref/h2_fig19.csv").is_file()


def test_scoring_operators_reproduce_the_anchor(scored):
    c = scored["reference"]["controls"]
    anchored_peak = c["anchored_peak_m"]
    anchored_arrival = c["anchored_arrival_time_s"]
    assert c["peak_m"] == pytest.approx(anchored_peak, rel=1e-6)
    # the anchored arrival is sample-quantised on the 11.6 ms digitisation grid,
    # so it must be reproduced by the "sample" convention exactly ...
    assert c["arrival_time_s_sample"] == pytest.approx(anchored_arrival, abs=1e-9)
    # ... and the interpolated convention is one sample earlier, not a bug
    assert c["arrival_time_s"] < anchored_arrival
    assert 0.0 < c["arrival_time_s_sample"] - c["arrival_time_s"] < 0.02
    assert c["t_span_s"][0] < anchored_arrival < c["peak_time_s"]


def test_t30_scoring_block_is_recorded_in_reference_json(scored):
    ref, _ = sc.load_reference()
    block = ref["t30_scoring"]
    assert block["verdict"] == scored["verdict"] == "FAIL"
    assert block["peak_criterion"]["decidable"] is True
    assert block["results"].endswith("score.json")
    assert "NOT COMPARABLE" not in ref["arrival_attribution"]["verdict"]


def test_digitisation_cross_check_reproduced(scored):
    """The paper's COMFLOW curve, traced independently, must still land within
    the recorded 3.2 % of the experimental arrival."""
    cc = scored["reference"]["comflow_control"]
    assert cc is not None
    assert cc["reproduced_arrival_rel"] == pytest.approx(cc["recorded_arrival_rel"],
                                                        abs=5e-3)
    assert cc["reproduced_arrival_rel"] < 0.05


# --------------------------------------------------------------------------
# 3. operator behaviour (analytic, no simulation)
# --------------------------------------------------------------------------
def test_arrival_interpolation_is_exact_on_a_ramp():
    t = [0.0, 0.1, 0.2, 0.3]
    h = [0.0, 0.0, 0.04, 0.08]            # crosses 0.02 midway in [0.1, 0.2]
    assert sc.arrival(t, h, 0.02) == pytest.approx(0.15, abs=1e-12)
    assert sc.arrival(t, h, 0.02, "sample") == pytest.approx(0.2, abs=1e-12)


def test_arrival_returns_nan_when_never_reached_or_already_wet():
    assert math.isnan(sc.arrival([0.0, 1.0], [0.0, 0.01], 0.02))
    assert math.isnan(sc.arrival([0.0, 1.0], [0.5, 0.6], 0.02))


def test_moving_average_preserves_constants_and_damps_a_spike():
    t = [i * 0.01 for i in range(21)]
    flat = [0.3] * 21
    assert sc.moving_average(t, flat, 0.03) == pytest.approx(flat)
    spiky = [0.0] * 21
    spiky[10] = 1.0
    sm = sc.moving_average(t, spiky, 0.03)
    assert max(sm) < 0.4, "a 30 ms boxcar over 3 samples must cut a 1-sample spike"
    assert max(sm) == pytest.approx(1.0 / 3.0, abs=1e-12)
    pk, pk_t = sc.peak(t, spiky, 0.03)
    assert pk == pytest.approx(1.0 / 3.0, abs=1e-12)
    # the smoothed plateau spans [0.09, 0.11]; ties resolve to the earliest sample
    assert pk_t == pytest.approx(0.09, abs=1e-12)


def test_resample_is_exact_for_a_linear_series():
    t = [0.0, 1.0, 2.0]
    h = [0.0, 2.0, 4.0]
    assert sc.resample(t, h, [0.5, 1.5]) == pytest.approx([1.0, 3.0])
    assert math.isnan(sc.resample(t, h, [-0.5])[0])


def test_richardson_sanity():
    # for a first-order error the two assumed orders bracket the exact value
    p1 = sc.richardson(0.382, 0.400, 2.0, 1.0)
    p2 = sc.richardson(0.382, 0.400, 2.0, 2.0)
    assert p1 < p2 < 0.382, "the finer grid is the upper bound for either order"


# --------------------------------------------------------------------------
# 4. the verdict and its preconditions
# --------------------------------------------------------------------------
def test_long_runs_cover_the_reference_peak(scored):
    """The whole point: the committed window reaches past 1.703 s, so the peak
    criterion is decidable without new integration."""
    t_peak = scored["reference"]["controls"]["peak_time_s"]
    assert scored["rows"], "no committed long runs were found"
    for row in scored["rows"]:
        assert row["window_covers_ref_peak"], row
        assert row["t_end_s"] >= t_peak


def test_every_gauge_and_grid_is_scored(scored):
    grids = {r["grid"] for r in scored["rows"]}
    stations = {r["station_x_m"] for r in scored["rows"]}
    ops = {r["operator"] for r in scored["rows"]}
    assert grids == {"dx0.02", "dx0.04"}
    assert ops == {"height", "surface", "slab"}
    assert len(stations) == 2
    assert len(scored["rows"]) == 2 * 3 * 2
    assert sum(1 for r in scored["rows"] if r["primary"]) == 2


def test_verdict_is_honest_fail_on_the_frozen_protocol(scored):
    assert scored["verdict"] == "FAIL"
    prim = [r for r in scored["rows"] if r["primary"]]
    for row in prim:
        assert not (row["peak_pass"] and row["arrival_robust"])
        assert not row["arrival_pass"], "arrival must fail under the interpolated convention too"


def test_arrival_robustness_invariant(scored):
    """A criterion that flips when the reference arrival is read one digitisation
    sample later is not robustly decidable and must not be counted as a pass."""
    for row in scored["rows"]:
        assert row["arrival_robust"] == (row["arrival_pass"] and row["arrival_pass_anchored"])
        if row["arrival_robust"]:
            assert row["arrival_rel_error"] < scored["tolerance"]
            assert row["arrival_rel_error_anchored"] < scored["tolerance"]


def test_peak_is_decidable_and_station_dependent(scored):
    """New result of this harness: the peak now has a verdict, and it depends on
    which side of the box the gauge sits -- the upstream station is inside the
    tolerance on both grids, the box-adjacent one is not."""
    def key(grid, x):
        return (grid, round(x, 6))

    by = {key(r["grid"], r["station_x_m"]): r for r in scored["rows"]
          if r["operator"] == "slab"}
    secondary = round(scored["station_secondary_x_m"], 6)
    primary = round(scored["station_primary_x_m"], 6)
    for grid in ("dx0.02", "dx0.04"):
        assert by[key(grid, secondary)]["peak_pass"], (grid, "upstream station")
        if grid == "dx0.02":
            assert not by[key(grid, primary)]["peak_pass"], \
                "box-adjacent station must fail"
        # the box-adjacent peak is the higher of the two: nearest the surge
        assert by[key(grid, primary)]["peak_m"] > by[key(grid, secondary)]["peak_m"]


def test_continuum_peak_bounds(scored):
    cont = scored["continuum"]
    key = f"station_x{scored['station_secondary_x_m']:.3f}"
    assert key in cont, cont.keys()
    entry = cont[key]
    # both assumed orders land inside 3 % of the experiment upstream of the box
    for order in ("1", "2"):
        assert entry[f"peak_rel_error_richardson_p{order}"] < 0.03
    # ... while the box-adjacent station diverges away from the experiment
    pkey = f"station_x{scored['station_primary_x_m']:.3f}"
    assert cont[pkey]["peak_rel_error_richardson_p1"] > 0.30


def test_protocol_sensitivity_is_reported(scored):
    """run.py's own comparison block scores a different gauge than the one the
    reference froze; the harness must expose both rather than mixing them."""
    sens = scored["protocol_sensitivity"]
    assert set(sens) == {"dx0.02", "dx0.04"}
    for grid, entry in sens.items():
        assert entry["run_py_verdict_recorded"] == "WITHHELD", \
            "the committed metrics predate the anchor; re-running would overwrite them"
        # run.py grades the legacy top-N operator at probe_x, not the frozen slab
        assert entry["scored_gauge"].startswith("height@")
        assert entry["frozen_slab_peak_m"] is not None
        assert entry["run_py_peak_rel_error"] > 0.10


def test_markdown_table_is_complete(scored):
    md = sc.markdown(scored)
    assert md.count("\n") >= len(scored["rows"]) + 2
    # on the frozen protocol every row is a FAIL: the peak passes upstream of the
    # box but the arrival never does, so no row can be graded PASS
    assert md.count("FAIL") >= len(scored["rows"])
    assert scored["verdict"] == "FAIL"


def test_score_json_is_serialisable(scored):
    blob = json.dumps(scored)
    assert "verdict" in json.loads(blob)
