"""Guards for the recovered 3-D verdict of CASE-SPH-KLEEFSMAN-DAMBREAK-OBST.

These are invariant tests, not smoke tests:

* a record that stops before the experimental peak instant can never be scored
  PASS on its truncated maximum, even when both metrics happen to look good;
* a maximum sitting on the last two samples of an unfinished record is flagged
  censored;
* a decided metric outside the 10 % band makes the verdict FAIL, and dropping
  the censored metric is not allowed to change that;
* the committed numbers in ``results/recovered/verdict.json`` are re-derived
  here from the raw series (a record whose provenance is missing is not
  admissible);
* the height-operator column of the two recovered records is identically zero
  while the slab column is not -- a *known, unfixed* record defect, pinned with
  ``xfail(strict=True)`` so that the day it is fixed this test turns red and
  the note has to be updated instead of quietly rotting.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "benchmarks" / "kleefsman_obstacle_3d"
sys.path.insert(0, str(HERE))

import recovered_verdict as rv  # noqa: E402


def _write_record(tmp: Path, times, slab, height=None, t_end=2.6):
    height = [0.0] * len(times) if height is None else height
    csv_path = tmp / "synthetic_time_series.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "slab_height_m", "height_operator_m", "front_x_m", "n_bypass"])
        for k, t in enumerate(times):
            w.writerow([format(t, ".6f"), format(slab[k], ".6f"),
                        format(height[k], ".6f"), "1.500000", "0"])
    meta = {"tag": "synthetic", "dx": 0.05, "t_end": t_end, "sound_speed": 20.0,
            "wall_layers": 3, "obstacle_z": 0.0, "station_offset": 0.05,
            "gauge_aperture": 0.08, "arrival_threshold": 0.02, "output_every": 5,
            "complete": False}
    (tmp / "synthetic_meta.json").write_text(json.dumps(meta))
    return {"tag": "synthetic", "dx": 0.05, "job": "cjTEST", "node": "test", "rc": 0,
            "csv": "synthetic_time_series.csv", "meta": "synthetic_meta.json"}


def _ramp(t_end: float, peak: float, peak_t: float, arrival: float):
    """Coarse series: zero until ``arrival``, linear to ``peak`` at ``peak_t``."""
    n = 400
    ts = [t_end * k / (n - 1) for k in range(n)]
    vals = []
    rise = 0.02          # a surge arrives as a steep front, not a slow ramp
    for t in ts:
        if t <= arrival:
            vals.append(0.0)
        elif t <= arrival + rise:
            vals.append(peak * (t - arrival) / rise)
        elif t <= peak_t:
            vals.append(peak)                      # wet plateau, then a second peak
        else:
            vals.append(peak * max(0.0, 1.0 - 0.3 * (t - peak_t)))
    return ts, vals


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    def build(run, times, slab, **kw):
        run = _write_record(tmp_path, times, slab, **kw)
        monkeypatch.setattr(rv, "RECOVERED", tmp_path)
        monkeypatch.setattr(rv, "RUNS", [run])
        return rv.score_run(run)
    return build


def test_good_looking_metrics_in_a_truncated_window_are_not_promoted(sandbox):
    """A record ending before the reference peak instant cannot be PASS."""
    ts, vals = _ramp(t_end=1.30, peak=0.372, peak_t=1.20, arrival=0.360)
    run = sandbox(None, ts, vals)
    rows = {r["metric"]: r for r in run["rows"]}
    # both numbers are inside the 10 % band on paper ...
    assert rows["arrival_time_s"]["decided"] is True
    assert rows["arrival_time_s"]["pass"] is True
    # ... but the maximum is censored by the window, so the peak is WITHHELD
    assert rows["peak_height_m"]["decided"] is False
    assert rows["peak_height_m"]["pass"] is None
    assert run["window_covers_reference_peak"] is False
    assert run["verdict"] == "WITHHELD"


def test_full_window_can_pass_and_fails_only_on_real_error(sandbox):
    ts, vals = _ramp(t_end=2.10, peak=0.372, peak_t=1.60, arrival=0.360)
    run = sandbox(None, ts, vals)
    assert run["window_covers_reference_peak"] is True
    assert run["verdict"] == "PASS"

    ts, vals = _ramp(t_end=2.10, peak=0.52, peak_t=1.60, arrival=0.360)
    run = sandbox(None, ts, vals)
    assert run["verdict"] == "FAIL"


def test_censored_maximum_on_last_samples_is_flagged(sandbox):
    ts = [2.0 * k / 399 for k in range(400)]
    vals = [max(0.0, 0.02 + 0.20 * t) for t in ts]      # still growing at t_last
    run = sandbox(None, ts, vals)
    assert run["peak_censored"] is True


def test_dropping_the_censored_metric_cannot_turn_fail_into_pass():
    """The committed verdict is FAIL because a *decided* metric fails."""
    payload = rv.build()
    decided = [(r["tag"], row) for r in payload["runs"]
               for row in r["rows"] if row["decided"]]
    failed = [(tag, row) for tag, row in decided if not row["pass"]]
    assert failed, "no decided metric fails: the committed FAIL would be unjustified"
    assert payload["verdict"] == "FAIL"


def test_provenance_is_required_and_series_are_raw():
    for run in rv.RUNS:
        assert run["job"].startswith("cj"), "a record without a job id is not admissible"
        assert (rv.RECOVERED / run["csv"]).exists()
        assert (rv.RECOVERED / run["meta"]).exists()


def test_committed_numbers_match_the_raw_series():
    """Recompute peak/arrival from the CSV without using the module's helpers."""
    payload = json.loads((rv.OUT / "verdict.json").read_text())
    by_tag = {r["tag"]: r for r in payload["runs"]}
    for run in rv.RUNS:
        rows = list(csv.DictReader((rv.RECOVERED / run["csv"]).open(newline="")))
        t = [float(r["t_s"]) for r in rows]
        slab = [float(r["slab_height_m"]) for r in rows]
        k = max(range(len(slab)), key=lambda i: slab[i])
        got = by_tag[run["tag"]]
        assert abs(got["peak_height_m"] - slab[k]) < 1e-9
        assert abs(got["peak_time_s"] - t[k]) < 1e-9
        assert abs(got["t_last_s"] - t[-1]) < 1e-9
        # the censoring rule itself, restated from the committed data
        assert got["window_covers_reference_peak"] == (t[-1] >= rv.REF_PEAK_T)


def test_two_dimensional_arm_is_never_scored_with_the_three_dimensional_record():
    """No cross-arm comparison: the recovered records must all be the 3-D arm."""
    for run in rv.RUNS:
        meta = json.loads((rv.RECOVERED / run["meta"]).read_text())
        assert meta["gauge_aperture"] == 0.08
        assert meta["station_offset"] == 0.05
        assert meta["obstacle_z"] == 0.0
        assert meta["arrival_threshold"] == 0.02


@pytest.mark.xfail(strict=True, reason=(
    "known unfixed defect: the height operator is written as 0.0 for every "
    "sample of both recovered 3-D runs while the slab operator is non-zero; "
    "peak_height() returns the correct value in isolation, so the loss happens "
    "inside the run/serialisation path and has not been located yet"))
def test_height_operator_column_is_not_identically_zero():
    for run in rv.RUNS:
        rows = list(csv.DictReader((rv.RECOVERED / run["csv"]).open(newline="")))
        assert max(float(r["height_operator_m"]) for r in rows) > 0.0