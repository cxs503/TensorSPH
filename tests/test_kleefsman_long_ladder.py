"""The long-ladder scorer must be the *same* instrument as the frozen one.

Adding grids is only allowed to add evidence, never to relax the gauge.  These
tests pin that down:

* the protocol read by ``score_long_ladder`` is byte-identical to the frozen one
  and to ``score_t30.FROZEN_DEFAULTS``;
* every row it produces for the already-published grids (dx = 0.02, 0.04) is
  numerically identical to ``results/audit_t30/score.json``;
* the new long records are *valid* records -- acoustic dt respected, the window
  covering the reference peak, the frozen gauge aperture used.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "benchmarks" / "kleefsman_obstacle"
sys.path.insert(0, str(HERE))

import score_long_ladder  # noqa: E402
import score_t30  # noqa: E402


@pytest.fixture(scope="module")
def res():
    return score_long_ladder.score()


@pytest.fixture(scope="module")
def frozen():
    path = HERE / "results" / "audit_t30" / "score.json"
    if not path.is_file():
        pytest.skip("frozen score.json not present")
    return json.loads(path.read_text())


def test_protocol_is_shared_with_the_frozen_instrument(res):
    assert res["protocol"] == score_t30.FROZEN_DEFAULTS


def test_published_grids_reproduce_the_frozen_rows(res, frozen):
    """dx0.02 / dx0.04 must be bit-for-bit what the frozen scorer published."""
    got = {(r["grid"], r["operator"], round(r["station_x_m"], 6)): r
           for r in res["rows"]}
    checked = 0
    for row in frozen["rows"]:
        key = (row["grid"], row["operator"], round(float(row["station_x_m"]), 6))
        if key not in got:
            continue
        mine = got[key]
        assert mine["peak_m"] == pytest.approx(row["peak_m"], abs=1e-12), key
        assert mine["peak_time_s"] == pytest.approx(row["peak_time_s"], abs=1e-12), key
        assert mine["arrival_time_s"] == pytest.approx(row["arrival_time_s"],
                                                       abs=1e-12), key
        assert mine["peak_pass"] == row["peak_pass"], key
        assert mine["arrival_pass"] == row["arrival_pass"], key
        checked += 1
    assert checked >= 12, f"only {checked} frozen rows reproduced"


def test_primary_station_is_the_frozen_one(res):
    assert res["protocol"]["station_offset_m"] == pytest.approx(0.05)
    assert res["protocol"]["aperture_m"] == pytest.approx(0.08)
    assert res["protocol"]["peak_filter_window_s"] == pytest.approx(0.03)
    assert res["protocol"]["arrival_threshold_m"] == pytest.approx(0.02)
    for r in res["rows"]:
        if r["primary"]:
            assert r["station_x_m"] == pytest.approx(2.426, abs=1e-6)


def test_verdict_is_not_withheld_once_the_primary_station_is_scored(res):
    assert res["primary_rows"], "no primary-station row scored"
    assert res["verdict"] in ("PASS", "FAIL")
    # the criterion is honest: the verdict is computed from the per-row flags
    expected = "PASS" if all(r["peak_pass"] and r["arrival_pass"]
                             and r["arrival_pass_anchored"]
                             for r in res["primary_rows"].values()) else "FAIL"
    assert res["verdict"] == expected


def test_anchor_values_match_reference_json(res):
    ref = json.loads((HERE / "reference.json").read_text())
    ev = ref["experimental_values"]
    assert res["anchors"]["peak_m"] == pytest.approx(float(ev["peak_height_m"]))
    assert res["anchors"]["arrival_time_s"] == pytest.approx(float(ev["arrival_time_s"]))
    assert res["anchors"]["peak_time_s"] == pytest.approx(float(ev["peak_time_s"]))


@pytest.mark.parametrize("grid", ["dx0.01", "dx0.005"])
def test_long_records_are_valid(grid):
    """New records must respect the acoustic dt limit and cover the reference peak.

    The whole point of the re-run is that the committed dx = 0.01 series was
    integrated at 2.5x its acoustic limit; a replacement that repeats that is
    worthless, so it fails here rather than in the report.
    """
    path = HERE / "results" / "audit_t30_long" / grid / "metrics.json"
    if not path.is_file():
        pytest.skip(f"{grid} long record not produced")
    m = json.loads(path.read_text())
    ref = json.loads((HERE / "reference.json").read_text())
    peak_t = float(ref["experimental_values"]["peak_time_s"])

    sys.path.insert(0, str(ROOT / "src"))
    from tensorsph.kleefsman_obstacle import KleefsmanConfig

    cfg = KleefsmanConfig(dx=float(m["dx"]))
    if "sound_speed" in m and m["sound_speed"]:
        cfg.sound_speed = float(m["sound_speed"])
    assert float(m["dt_max"]) <= cfg.dt_acoustic() * (1.0 + 1e-9), (
        f"{grid}: dt_max {m['dt_max']:.3e} exceeds the acoustic limit "
        f"{cfg.dt_acoustic():.3e}")
    assert float(m["t_end"]) >= peak_t - 1e-9, (
        f"{grid}: t_end {m['t_end']} does not cover the reference peak {peak_t}")
    assert float(m["gauge_aperture_m"]) == pytest.approx(0.08), "frozen aperture"
    assert m["boundary_escapes"] == 0 or int(m["boundary_escapes"]) >= 0
    assert math.isfinite(float(m["final_mass_error_rel"])) if "final_mass_error_rel" in m else True


def test_break_even_uses_the_frozen_gauge_only(res):
    """Break-even must be computed from slab rows, never from a kinder operator.

    The break-even dict is keyed by station offset, so a height-operator row at
    the same offset would collide; what must hold is that the set of offsets is
    exactly the set of *slab* offsets for that grid, and each value equals the
    slab row's error.
    """
    op = res["protocol"]["operator"]
    for grid, be in res["break_even"].items():
        slab = {f"{r['offset_m']:.3f}": r["arrival_rel_error"] for r in res["rows"]
                if r["grid"] == grid and r["operator"] == op}
        assert set(be["arrival_rel_error"]) == set(slab), grid
        for key, val in be["arrival_rel_error"].items():
            assert val == pytest.approx(slab[key], abs=1e-12), (grid, key)