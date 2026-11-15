"""Guards for the text-anchored landmark audit of the Kleefsman case (m21534).

The audit's whole claim is that its reference instants come from *text in the
paper*, not from a number somebody typed.  These tests therefore check
(a) the quote bundle is complete and carries provenance, (b) the landmark table
only cites quotes that exist, and (c) the measurement/verdict logic does what it
says on synthetic records (including the "no data -> UNKNOWN" case).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "kleefsman_obstacle"))

import text_landmarks as tl  # noqa: E402


def _write_series(tmp_path: Path, rows: list[tuple[float, float, float, float]]) -> Path:
    p = tmp_path / "series.csv"
    with p.open("w") as fh:
        fh.write("t_s,height_m,slab_x2.476,slab_x2.426,slab_x2.228\n")
        for t, box, h2, up in rows:
            fh.write(f"{t:.6f},0.0,{box:.6f},{h2:.6f},{up:.6f}\n")
    return p


def test_quote_bundle_is_complete_and_provenanced():
    quotes = tl.load_quotes()
    ids = {q["id"] for q in quotes["quotes"]}
    for lm in tl.LANDMARKS:
        for qid in lm["quote_ids"]:
            assert qid in ids, f"landmark {lm['id']} cites missing quote {qid}"
    for src, meta in quotes["sources"].items():
        assert meta.get("sha256") and meta.get("url") and meta.get("citation"), src
        assert len(meta["sha256"]) == 64
    aliases = {"K2005": "kleefsman2005", "AJ2023": "amaro2023"}
    for q in quotes["quotes"]:
        assert len(q["text"]) > 25, q
        assert aliases.get(q["source"], q["source"]) in quotes["sources"], q["source"]


def test_landmark_reference_instants_grow_and_sit_inside_their_windows():
    # BOX_FIRST_IMPACT/BACKWALL_RETURN are early, RETURN_IMPACT_ON_BOX/SECOND_WAVE_AT_H2
    # late.  The two late windows overlap on purpose (they are different
    # observables of the same return-wave event), so only the ordering and the
    # "reference inside its own window" property are asserted here.
    refs = [lm["ref_s"] for lm in tl.LANDMARKS]
    assert refs == sorted(refs), refs
    for lm in tl.LANDMARKS:
        lo, hi = lm["window"]
        assert lo <= lm["ref_s"] <= hi, lm["id"]
        assert lm["tol_s"] > 0.0, lm["id"]


def test_crossing_interpolates_between_samples():
    times = [0.0, 1.0, 2.0]
    values = [0.0, 0.01, 0.03]
    # 0.02 sits half way between the 0.01 and 0.03 samples
    assert tl.crossing(times, values, 0.02) == pytest.approx(1.5)


def test_crossing_respects_t_min_and_returns_nan_when_absent():
    times = [0.0, 1.0, 2.0]
    values = [0.0, 0.05, 0.0]
    assert tl.crossing(times, values, 0.02) == pytest.approx(0.4)
    assert math.isnan(tl.crossing(times, values, 0.02, t_min=0.5))
    assert math.isnan(tl.crossing(times, [0.0, 0.0, 0.0], 0.02))


def test_score_passes_a_record_that_hits_the_text_landmarks(tmp_path):
    # a synthetic record that puts every landmark on its reference instant
    rows = []
    for k in range(0, 6001):
        t = k * 0.001
        box = 0.0
        if t >= 0.47:
            box = 0.05
        if 4.60 <= t <= 4.80:
            box = 0.30   # second impact bump centred on 4.70 s
        h2 = 0.40 if 1.75 <= t <= 1.85 else (0.05 if t > 0.47 else 0.0)
        if 4.95 <= t <= 5.05:
            h2 = 0.25    # second wave bump centred on 5.00 s
        rows.append((t, box, h2, h2))
    path = _write_series(tmp_path, rows)
    out = tl.score(path)
    got = {r["landmark"]: r for r in out["rows"] if not r.get("proxy")}
    assert got["BOX_FIRST_IMPACT"]["verdict"] == "PASS"
    assert got["BOX_FIRST_IMPACT"]["measured_s"] == pytest.approx(0.47, abs=0.01)
    assert got["BACKWALL_RETURN_AT_H2"]["verdict"] == "PASS"
    assert got["RETURN_IMPACT_ON_BOX"]["verdict"] == "PASS"
    assert got["SECOND_WAVE_AT_H2"]["verdict"] == "PASS"
    assert out["verdict"] == "PASS"


def test_score_reports_unknown_for_a_short_record(tmp_path):
    rows = [(k * 0.001, 0.05, 0.05, 0.05) for k in range(0, 1201)]  # 1.2 s only
    path = _write_series(tmp_path, rows)
    out = tl.score(path)
    late = [r for r in out["rows"] if r["landmark"] in ("RETURN_IMPACT_ON_BOX", "SECOND_WAVE_AT_H2")]
    assert late and all(r["verdict"] == "UNKNOWN" for r in late)
    assert out["verdict"] == "INCOMPLETE"


def test_score_flags_a_miss_as_fail(tmp_path):
    # arrival at 0.90 s is outside the 0.40-0.56 s bracket
    rows = [(k * 0.001, 0.05 if k >= 900 else 0.0, 0.05 if k >= 900 else 0.0,
             0.05 if k >= 900 else 0.0) for k in range(0, 1201)]
    path = _write_series(tmp_path, rows)
    out = tl.score(path)
    box = [r for r in out["rows"] if r["landmark"] == "BOX_FIRST_IMPACT" and not r.get("proxy")]
    assert box and box[0]["verdict"] == "FAIL"
    assert out["verdict"] == "FAIL"


def test_proxy_station_is_labelled(tmp_path):
    p = tmp_path / "series.csv"
    with p.open("w") as fh:
        fh.write("t_s,slab_x2.426\n0.0,0.0\n1.0,0.1\n")
    cols = tl.load_series(p)
    dist, x, name = tl.pick_slab(cols, 2.476)
    assert x == pytest.approx(2.426) and dist == pytest.approx(0.05) and name == "slab_x2.426"
