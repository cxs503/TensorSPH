"""Guards for the Kleefsman H2 experimental anchor (mission 20884).

The case-pool entry CASE-SPH-KLEEFSMAN-DAMBREAK-OBST needs an *external* anchor for the water level in
front of the box.  Until this mission that field was WITHHELD; it is now ANCHORED by digitising the
experimental trace of Kleefsman et al. (2005, JCP 206:363-393) Fig. 19, right panel (probe H2).
These tests keep the anchor honest: they check provenance, the digitisation's own self-consistency
(the two independently traced curves must reproduce the paper's claim that the impact instant is
"perfectly captured"), and the behaviour of the grading function.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the digitisation bundle was relocated under docs/ (see
# docs/dev_report_m17086_ledger_artifact_relocation.md); accept either place so the
# guard follows the artifacts instead of silently failing on a stale path
_CANDIDATES = [
    os.path.join(ROOT, "outputs", "kleefsman_h2_ref"),
    os.path.join(ROOT, "docs", "m20884", "outputs", "kleefsman_h2_ref"),
]
REF_DIR = next((c for c in _CANDIDATES if os.path.isdir(c)), _CANDIDATES[0])
CASE_REF = os.path.join(ROOT, "benchmarks", "kleefsman_obstacle", "reference.json")
sys.path.insert(0, REF_DIR)


def _anchor():
    with open(os.path.join(REF_DIR, "reference_h2.json")) as f:
        return json.load(f)


def test_anchor_has_provenance_and_is_no_longer_withheld():
    a = _anchor()
    for key in ("probe", "ref", "source_pdf", "peak_height_m", "arrival_time_s",
                "digitised_from", "cross_check", "uncertainty"):
        assert key in a, key
    assert "10.1016/j.jcp.2004.12.007" in a["ref"]
    with open(CASE_REF) as f:
        case = json.load(f)
    ev = case["experimental_values"]
    assert ev["status"] == "ANCHORED", "the case-pool anchor must stay resolved"
    assert ev["peak_height_m"] == pytest.approx(a["peak_height_m"], abs=1e-9)
    assert ev["arrival_time_s"] == pytest.approx(a["arrival_time_s"], abs=1e-9)
    assert ev["probe_id"] == "H2"


def test_digitised_series_is_physical():
    t, h = [], []
    with open(os.path.join(REF_DIR, "h2_fig19.csv")) as f:
        header = f.readline()
        for line in f:
            a, b = line.split(",")
            t.append(float(a))
            h.append(float(b))
    t, h = np.array(t), np.array(h)
    assert header.strip() == "t_s,h_m"
    assert t.size > 300 and np.all(np.diff(t) > 0)
    assert t.max() <= 6.0 + 1e-9
    # a water height in a 1 m tank cannot exceed the tank and cannot be negative
    assert h.min() >= -1e-9 and h.max() <= 1.0
    # the digitised peak/arrival must be reproduced from the series itself
    a = _anchor()
    ip = int(np.argmax(h))
    assert h[ip] == pytest.approx(a["peak_height_m"], abs=0.01)
    assert t[ip] == pytest.approx(a["peak_time_s"], abs=0.05)
    hit = np.where(h > a["arrival_threshold_m"])[0]
    assert hit.size and t[hit[0]] == pytest.approx(a["arrival_time_s"], abs=0.02)


def test_digitisation_reproduces_the_paper_claim_of_matching_impact_instant():
    """Two curves traced independently (solid COMFLOW vs dashed experiment) must agree on arrival:
    the paper states the impact instant is 'perfectly captured'.  A tracing error that mixes the two
    curves would break this, so it is a real self-check, not a tautology."""
    cc = _anchor()["cross_check"]
    rel = abs(cc["comflow_arrival_time_s"] - _anchor()["arrival_time_s"]) / _anchor()["arrival_time_s"]
    assert rel < 0.05, f"digitised curves disagree on arrival by {rel:.1%}"


def _grade_monkey(monkeypatch, t, h, window=None):
    import compare_h2
    monkeypatch.setattr(compare_h2, "series", lambda tag: (list(t), list(h)))
    return compare_h2.grade("synthetic", window=window)


def test_grading_flags_a_matching_run_as_pass(monkeypatch):
    a = _anchor()
    t = np.linspace(0, 3, 601)
    arr, tp, pk = a["arrival_time_s"], a["peak_time_s"], a["peak_height_m"]
    # triangular rise from exactly the reference arrival to the reference peak, then an exponential tail
    h = np.where(t < arr, 0.0,
                 np.where(t <= tp, np.maximum(0.10, pk * (t - arr) / (tp - arr)),
                          pk * np.exp(-(t - tp) / 0.6)))
    r = _grade_monkey(monkeypatch, t, h)
    assert r["arrival_err_rel"] < 0.1 and r["arrival_pass"]
    assert r["peak_err_rel"] < 0.1 and r["peak_pass"]


def test_grading_flags_overshoot_and_late_arrival_as_fail(monkeypatch):
    a = _anchor()
    t = np.linspace(0, 3, 601)
    h = np.clip(1.4 * a["peak_height_m"] * np.exp(-((t - a["peak_time_s"]) / 0.4) ** 2), 0, None)
    h[t < 1.4 * a["arrival_time_s"]] = 0.0
    r = _grade_monkey(monkeypatch, t, h)
    assert not r["peak_pass"] and not r["arrival_pass"]


def test_window_limit_is_visible_in_the_result(monkeypatch):
    """A run shorter than the experimental peak time must report a window-limited peak whose time is
    inside the window - this is the trap that made the pre-mission numbers look comparable when they
    were not."""
    import compare_h2
    a = _anchor()
    t = np.linspace(0, 1.2, 301)
    h = np.clip(0.52 * np.exp(-((t - 0.7) / 0.3) ** 2), 0, None)
    r = _grade_monkey(monkeypatch, t, h, window=1.2)
    assert r["peak_time_s"] <= 1.2
    assert r["t_end"] <= 1.2
    assert abs(r["peak_time_s"] - a["peak_time_s"]) > 0.5, "window-limited peak is not the global peak"
