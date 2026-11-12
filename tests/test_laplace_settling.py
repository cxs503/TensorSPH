"""Gates for the long-window settling study of the Laplace read-out.

The point of these tests is not that the run finished: it is that the *rules* of
`settled_verdict.py` are the ones applied.  In particular

* admissibility must be the maximum of the late window, never its mean (a single
  breathing excursion must fail it);
* a record shorter than the minimum window may not grade the settling question
  (negative control below);
* the verdict must follow from the per-point checks, not from prose;
* every number quoted in `docs/verification_laplace_pressure_settling.md` must
  be present in `settling_matrix.json`.

    python -m pytest tests/test_laplace_settling.py -q
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_pressure_closure"
MATRIX = BENCH / "settling_matrix.json"
DOC = ROOT / "docs" / "verification_laplace_pressure_settling.md"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sv():
    return _module("settled_verdict", BENCH / "settled_verdict.py")


@pytest.fixture(scope="module")
def matrix():
    assert MATRIX.exists(), "run benchmarks/laplace_pressure_closure/settled_verdict.py"
    return json.loads(MATRIX.read_text())


def test_evidence_is_recorded_with_provenance(matrix):
    """P1: every graded record spans the minimum window and was not dt-clipped."""
    assert matrix["readouts"], "no settling evidence recorded"
    for r in matrix["readouts"]:
        assert r["window_over_t_sigma"] >= matrix["rules"]["min_window_over_t_sigma"], r["label"]
        assert r["checks"]["dt_not_clipped"], r["label"]
        assert r["steps"] > 0 and r["n_particles"] > 0
        assert len(r["vmax_trace"]) == len(r["p_core_trace"]) > 5


def test_matrix_is_recomputed_from_the_raw_files(matrix, sv):
    """The matrix may not carry a number that the raw evidence does not produce."""
    raw = {}
    for r in matrix["readouts"]:
        raw[(r["source"], r["config"]["closure"], r["config"]["donors"],
             r["config"]["layers"])] = r
    points = sv.load_points()
    assert points, "no raw/settling_*.json"
    for point in points:
        for case in point["cases"]:
            recomputed = sv.row(point, case)
            key = (recomputed["source"], recomputed["config"]["closure"],
                   recomputed["config"]["donors"], recomputed["config"]["layers"])
            assert key in raw, f"{key} missing from the matrix"
            stored = raw[key]
            for field in ("value_Pa", "rel_error_pct", "vmax_over_vref_late",
                          "vmax_over_vref_early", "decay_ratio", "window_over_t_sigma"):
                assert stored[field] == pytest.approx(recomputed[field], rel=1e-12,
                                                      abs=1e-12), (key, field)


def test_admissibility_uses_the_maximum_of_the_late_window(matrix):
    """A mean would hide a breathing excursion; the rule is the late-window maximum."""
    for r in matrix["readouts"]:
        trace = r["vmax_trace"]
        n = len(trace)
        late = trace[int(0.75 * n):]
        assert late, r["label"]
        assert r["vmax_over_vref_late"] == pytest.approx(max(v for _, v in late), rel=1e-12)
        assert r["vmax_over_vref_late"] >= sum(v for _, v in late) / len(late) - 1e-12
        assert r["admissible"] == (r["vmax_over_vref_late"]
                                   < matrix["rules"]["spurious_current_limit"])


def test_decay_ratio_is_measured_from_the_trace(matrix):
    for r in matrix["readouts"]:
        assert r["decay_ratio"] == pytest.approx(
            r["vmax_over_vref_late"] / r["vmax_over_vref_early"], rel=1e-9)
        if not r["checks"]["decayed"]:
            assert any(d["id"] == "spurious-current-does-not-decay"
                       and r["label"] in d["rows"] for d in matrix["defects"]), r["label"]


def test_acceptance_ladder_is_complete_or_reported(matrix):
    points = matrix["rules"]["acceptance_points"]
    assert len(points) == 4
    recorded = {(a["dx_m"], a["sound_speed"]) for a in matrix["acceptance"] if a["recorded"]}
    missing = {(a["dx_m"], a["sound_speed"]) for a in matrix["acceptance"] if not a["recorded"]}
    assert recorded | missing == {(p["dx_m"], p["sound_speed"]) for p in points}
    if missing:
        assert any(d["id"] == "acceptance-point-not-recorded" for d in matrix["defects"])
        assert matrix["verdict"]["status"] == "INCOMPLETE"


def test_verdict_follows_from_the_checks(matrix):
    recorded = [a for a in matrix["acceptance"] if a["recorded"]]
    clean = bool(recorded) and all(a["in_band"] and a["admissible"] for a in recorded)
    status = matrix["verdict"]["status"]
    if status == "SETTLED":
        assert clean and len(recorded) == 4
    else:
        assert not (clean and len(recorded) == 4), (
            "verdict is conservative: a clean ladder must be graded SETTLED")
        if clean and len(recorded) < 4:
            assert status == "INCOMPLETE"


def test_band_membership_matches_the_error(matrix):
    for a in matrix["acceptance"]:
        if not a["recorded"]:
            continue
        assert a["in_band"] == (abs(a["rel_error_pct"]) < matrix["rules"]["band_pct"])


def test_negative_control_short_window_is_rejected(sv, matrix):
    """A 1.5-t_sigma record -- the earlier ladder's window -- may not grade the question."""
    point = {"analytic_dp_Pa": matrix["reference"]["analytic_dp_Pa"],
             "dx_m": 0.002, "sound_speed": 20.0, "source_file": "raw/synthetic.json"}
    case = {"case": "closure_on_interior_layers2",
            "config": {"closure": True, "donors": "interior", "layers": 2},
            "t_over_t_sigma_max": 1.5, "dt_was_clipped": False, "steps": 2000,
            "n_particles": 314, "tail40_Pa": 391.8, "tail40_rel_error_pct": 8.8,
            "tail25_rel_error_pct": 8.8, "final_rel_error_pct": 142.8,
            "vmax_over_vref_early20": 0.5, "vmax_over_vref_late_max": 0.388,
            "vmax_over_vref_late_mean": 0.2, "decay_ratio": 0.776,
            "passes_adami_late": False, "final_abs_Pa": 874.0, "n_samples": 27,
            "vmax_trace_over_t_sigma": [[0.1 * i, 0.3] for i in range(27)],
            "p_core_trace_over_t_sigma": [[0.1 * i, 300.0] for i in range(27)]}
    row = sv.row(point, case)
    assert row["checks"]["window"] is False
    assert row["checks"]["admissible"] is False
    # same evidence with a long window would pass P1 -- the window is what fails
    case_long = dict(case, t_over_t_sigma_max=6.0)
    assert sv.row(point, case_long)["checks"]["window"] is True


def test_doc_quotes_the_matrix(matrix):
    assert DOC.exists(), "run benchmarks/laplace_pressure_closure/make_doc_settling.py"
    text = DOC.read_text()
    assert matrix["verdict"]["status"] in text
    for a in matrix["acceptance"]:
        if a["recorded"]:
            assert f"{a['rel_error_pct']:+.1f}" in text
    for r in matrix["readouts"]:
        assert f"{r['vmax_over_vref_late']:.3f}" in text