"""Invariant tests for the 3-D ladder's *admission* rules.

The 3-D production rungs submitted on 11-15 (cj3661 dx=0.04, cj3662 dx=0.03,
cj3758 dx=0.08) taught three things that a fitting test cannot catch:

* dx = 0.08 is rejected by ``validate_case`` itself (a 0.161 m box cannot be
  spanned by 3 cells of 0.08 m), so the coarsest rung of the ladder as designed
  could never produce a number;
* a run that dies mid-way must not silently become a level of the ladder;
* ``check_protocol`` skipped every key that the run records did not carry, so a
  guard that compared *nothing* still returned "comparable".

These tests pin all three.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "kleefsman_obstacle_3d"))
sys.path.insert(0, str(ROOT / "src"))

import ladder  # noqa: E402


# --------------------------------------------------------------------------- #
# rung admissibility
# --------------------------------------------------------------------------- #
def test_the_coarse_rung_that_was_submitted_is_inadmissible():
    v = ladder.rung_validity(0.08)
    assert v["valid"] is False
    assert "under-resolved" in v["why"]


def test_the_admissible_rung_set_is_bounded_by_the_three_cell_rule():
    r = ladder.valid_rungs([0.08, 0.05, 0.04, 0.03])
    assert r["valid"] == [0.05, 0.04, 0.03]
    assert [x["dx"] for x in r["rejected"]] == [0.08]
    assert r["usable"] is True
    # 3 cells across the 0.161 m box
    assert abs(r["max_admissible_dx"] - 0.161 / 3.0) < 1e-12
    assert ladder.rung_validity(r["max_admissible_dx"])["valid"] is True
    assert ladder.rung_validity(r["max_admissible_dx"] * 1.001)["valid"] is False


def test_two_valid_rungs_are_not_enough_for_a_three_level_ladder():
    r = ladder.valid_rungs([0.08, 0.04], min_levels=3)
    assert r["n_levels"] == 1 and r["usable"] is False


# --------------------------------------------------------------------------- #
# the comparability guard must not pass vacuously
# --------------------------------------------------------------------------- #
FINGERPRINT = {"t_end": 2.6, "sound_speed": 20.0, "wall_layers": 3,
               "obstacle_z": 0.0, "station_offset": 0.05,
               "gauge_aperture": 0.08, "arrival_threshold": 0.02}


def test_identical_fingerprints_are_comparable():
    entries = [{"tag": "a", "fingerprint": dict(FINGERPRINT)},
               {"tag": "b", "fingerprint": dict(FINGERPRINT)}]
    ok, bad = ladder.check_protocol(entries)
    assert ok is True and bad == {}


def test_empty_fingerprints_do_not_pass_the_guard():
    entries = [{"tag": "a", "fingerprint": {}}, {"tag": "b", "fingerprint": {}}]
    ok, bad = ladder.check_protocol(entries)
    assert ok is False
    assert "_fingerprint" in bad


def test_a_key_recorded_by_only_some_runs_poisons_the_ladder():
    a = dict(FINGERPRINT)
    b = dict(FINGERPRINT)
    b.pop("wall_layers")
    ok, bad = ladder.check_protocol([{"tag": "a", "fingerprint": a},
                                     {"tag": "b", "fingerprint": b}])
    assert ok is False
    assert "wall_layers" in bad and bad["wall_layers"]["_missing"] == ["b"]


def test_a_protocol_disagreement_is_reported_by_key():
    other = dict(FINGERPRINT, t_end=3.0)
    ok, bad = ladder.check_protocol([{"tag": "a", "fingerprint": dict(FINGERPRINT)},
                                     {"tag": "b", "fingerprint": other}])
    assert ok is False
    assert bad["t_end"] == {"a": 2.6, "b": 3.0}


# --------------------------------------------------------------------------- #
# analyse(): rejected rungs and aborted runs never reach the fit
# --------------------------------------------------------------------------- #
def _write_run(results_dir: Path, tag: str, dx: float, peak: float,
               complete: bool = True, fingerprint: dict | None = None) -> None:
    fp = dict(fingerprint or FINGERPRINT)
    (results_dir / f"{tag}.json").write_text(json.dumps({
        "tag": tag, "complete": complete, "verdict": "FAIL",
        "censored": False, "window_covers_reference_peak": True,
        "protocol_comparable": True,
        "rows": [{"metric": "peak_height", "ref": 0.370, "got": peak, "pass": False}],
        "run": {"dx": dx, "peak_height_m": peak, "peak_time_s": 1.7,
                "arrival_time_s": 0.36, "fingerprint": fp},
    }) + "\n")
    with (results_dir / f"{tag}_time_series.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "slab_height_m", "height_operator_m",
                    "front_x_m", "n_bypass"])
        for k in range(9):
            w.writerow([f"{0.3 * k:.6f}", f"{peak * (0.5 + 0.05 * k):.6f}",
                        f"{peak * (0.6 + 0.05 * k):.6f}", "2.4", "0"])


def test_an_inadmissible_rung_stops_the_ladder(tmp_path):
    for tag, dx, peak in (("dx0.08", 0.08, 0.30), ("dx0.04", 0.04, 0.42),
                          ("dx0.03", 0.03, 0.40)):
        _write_run(tmp_path, tag, dx, peak)
    rep = ladder.analyse(tmp_path, ["dx0.08", "dx0.04", "dx0.03"])
    assert rep["status"] == "RUNGS_REJECTED"
    assert "dx0.08" in rep["inadmissible_rungs"]
    assert rep["usable_rungs"] == ["dx0.04", "dx0.03"]
    assert "peak_ladder" not in rep        # no fit is fabricated from them


def test_an_aborted_run_stops_the_ladder(tmp_path):
    _write_run(tmp_path, "dx0.05", 0.05, 0.44)
    _write_run(tmp_path, "dx0.04", 0.04, 0.42, complete=False)
    _write_run(tmp_path, "dx0.03", 0.03, 0.40)
    rep = ladder.analyse(tmp_path, ["dx0.05", "dx0.04", "dx0.03"])
    assert rep["status"] == "RUNGS_REJECTED"
    assert rep["incomplete_runs"] == ["dx0.04"]
    assert rep["usable_rungs"] == ["dx0.05", "dx0.03"]


def test_a_clean_valid_ladder_does_fit(tmp_path):
    f_inf, C, p = 0.372, 1.4e-3, 2.0
    for dx in (0.05, 0.04, 0.03):
        tag = f"dx{dx:g}"
        _write_run(tmp_path, tag, dx, f_inf + C * dx ** p)
    rep = ladder.analyse(tmp_path, ["dx0.05", "dx0.04", "dx0.03"])
    assert rep["status"] == "OK"
    assert rep["inadmissible_rungs"] == {} and rep["incomplete_runs"] == []
    assert abs(rep["peak_ladder"]["f_inf"] - f_inf) < 1e-5


# --------------------------------------------------------------------------- #
# the driver streams its series and records the protocol it ran under
# --------------------------------------------------------------------------- #
def test_the_run_record_carries_the_full_protocol_fingerprint():
    from tensorsph.kleefsman_obstacle_3d import Kleefsman3DConfig, Kleefsman3DRun

    run = Kleefsman3DRun(cfg=Kleefsman3DConfig(dx=0.05))
    fp = run.to_dict()["fingerprint"]
    assert set(fp) == ladder.FROZEN_PROTOCOL - {"wall_layers"} | {"wall_layers"}
    assert set(ladder.FROZEN_PROTOCOL) <= set(fp)


def test_the_solver_hands_out_every_stored_sample():
    from tensorsph.kleefsman_obstacle_3d import Kleefsman3DConfig, run_case

    seen = []
    cfg = Kleefsman3DConfig(dx=0.05, t_end=0.05, output_every=10)
    run = run_case(cfg, device="cpu", on_sample=lambda i, s: seen.append((i, s)))
    assert len(seen) == len(run.times) > 0
    assert [i for i, _ in seen] == list(range(len(run.times)))
    assert all(abs(s["t_s"] - t) < 1e-12 for (_, s), t in zip(seen, run.times))
    assert all(s["slab_height_m"] == h for (_, s), h in zip(seen, run.slab))