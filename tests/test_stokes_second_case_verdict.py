"""Tests for the case-level verdict of CASE-SPH-STOKES-2ND-PROBLEM.

The six gates of ``benchmarks/stokes_second/DESIGN.md`` used to be graded in
three different places (per-rung gates in ``results_ladder.json``, the negative
control in ``results_sensitivity.json``, convergence in its own block) with no
machine-readable object saying whether the *case* passed.  ``benchmark.json``
claimed "one row per judged quantity" but carried no G5/G6 row, so a reader
could not tell a qualified case from one whose control never ran -- the first
delivery of this case shipped a one-rung "ladder" with ``G6_not_growing: true``.

``benchmarks/stokes_second/case_verdict.py`` is the missing object.  These tests
pin its contract:

* on the published artefacts it reports PASS for every predeclared gate, and the
  gate inputs are cross-checked against an independent re-derivation from each
  rung's persisted rows (so a hand-edited metric cannot certify itself);
* missing or mismatched control evidence **withholds** G5 instead of passing it
  (fail-closed), and a withheld verdict exits 2 rather than 0;
* an incomplete ladder or a stored convergence flag the rows do not support is
  a FAIL, not a vacuous pass (the F1 defect of the first delivery);
* the committed ``case_verdict.json`` and the generated page are what the code
  produces now (no stale artefact).

No solver is run here: the verdict is a bookkeeping device over published
artefacts, and the physics is graded by ``tests/test_stokes_second.py``.  The
synthetic payloads are copies of the published ladder with one field perturbed.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cv():
    return _load("case_verdict", "benchmarks/stokes_second/case_verdict.py")


@pytest.fixture(scope="module")
def published(cv):
    ladder, sensitivity, provenance = cv.load_artifacts(cv.OUT)
    return ladder, sensitivity, provenance


@pytest.fixture(scope="module")
def verdict(cv, published):
    ladder, sensitivity, provenance = published
    return cv.case_verdict(ladder, sensitivity, provenance=provenance)


def _gate(verdict: dict, gid: str) -> dict:
    return next(g for g in verdict["gates"] if g["id"] == gid)


# --------------------------------------------------------------------------- #
# The published artefacts
# --------------------------------------------------------------------------- #

def test_published_case_passes_every_predeclared_gate(verdict):
    ids = [g["id"] for g in verdict["gates"]]
    assert ids == ["G1", "G2", "G3", "G4", "rows_in_layer", "G6", "G5"]
    assert verdict["overall"] == "PASS"
    assert verdict["counts"] == {"pass": 7, "fail": 0, "withheld": 0}
    assert verdict["all_predeclared_gates_pass"] is True
    assert verdict["consistency"]["ok"], verdict["consistency"]["problems"]


def test_gate_inputs_are_recomputed_from_the_persisted_rows(verdict):
    """The stored metrics must be supported by the rows they claim to describe."""
    by_rung = {r["resolution_delta_over_dx"]: r
               for r in verdict["recomputed_per_rung"]}
    assert sorted(by_rung) == [8, 12, 18]
    for gid in ("G1", "G2", "G3", "G4"):
        g = _gate(verdict, gid)
        for rung, published_value in g["per_rung"].items():
            from_rows = g["per_rung_from_rows"][rung]
            # relative errors ~1e-4 rounded to 6 significant digits: 1e-3 pp
            assert abs(published_value - from_rows) <= 1e-3, (gid, rung)
    # and the worst rung is the one the gate is graded on
    g1 = _gate(verdict, "G1")
    assert abs(g1["measured"]) == max(abs(v) for v in g1["per_rung"].values())


def test_g6_uses_the_predeclared_ladder_and_the_persisted_l2_series(verdict):
    g6 = _gate(verdict, "G6")
    m = g6["measured"]
    assert m["resolutions"] == [8, 12, 18]
    assert m["complete"] and m["not_growing"]
    assert m["l2_pct"][-1] <= m["l2_pct"][0] + 0.5
    assert g6["status"] == "PASS"


def test_g5_is_the_free_slip_control_with_matching_configuration(verdict, published):
    ladder, sensitivity, _ = published
    g5 = _gate(verdict, "G5")
    control = next(r for r in sensitivity["sensitivity"] if r["slip"] == "free_slip")
    assert g5["measured"] == control["metrics"]["amplitude_max_over_U0"]
    assert g5["status"] == "PASS"
    assert verdict["fingerprint"]["match"] is True
    assert verdict["fingerprint"]["ladder"]["c0"] == \
        verdict["fingerprint"]["sensitivity"]["c0"] == 4.0


def test_thresholds_are_the_predeclared_ones(cv):
    """Drift between this module, run.py and DESIGN.md must break the build."""
    run = _load("stokes_second_run", "benchmarks/stokes_second/run.py")
    design = (REPO / "benchmarks/stokes_second/DESIGN.md").read_text()
    mapping = {
        "G1": "G1_amplitude_l2_pct",
        "G2": "G2_worst_row_pct",
        "G3": "G3_decay_rate_pct",
        "G4": "G4_finite_depth_l2_pct",
    }
    for gid, key in mapping.items():
        assert cv.PREDECLARED[gid]["threshold_pct"] == run.GATES[key], gid
        assert f"{run.GATES[key]:g}" in design, gid
    assert cv.PREDECLARED["G5"]["threshold"] == 0.05
    assert "5 %" in design


# --------------------------------------------------------------------------- #
# Fail-closed behaviour
# --------------------------------------------------------------------------- #

def test_missing_control_artifact_withholds_g5_instead_of_passing(cv, published):
    ladder, _, provenance = published
    v = cv.case_verdict(ladder, None, provenance=provenance)
    g5 = _gate(v, "G5")
    assert g5["status"] == "WITHHELD"
    assert "not run" in g5["reason"]
    assert v["overall"] == "WITHHELD"
    assert v["all_predeclared_gates_pass"] is False


def test_mismatched_control_configuration_withholds_g5(cv, published):
    ladder, sensitivity, provenance = published
    bad = copy.deepcopy(sensitivity)
    bad["provenance"]["case"]["c0"] = 10.0   # the pre-fix header value
    v = cv.case_verdict(ladder, bad, provenance=provenance)
    g5 = _gate(v, "G5")
    assert g5["status"] == "WITHHELD"
    assert "does not share the ladder's case configuration" in g5["reason"]
    assert "c0" in g5["reason"]
    assert v["fingerprint"]["match"] is False
    assert v["overall"] == "WITHHELD"


def test_still_wall_run_is_not_accepted_as_the_free_slip_control(cv, published):
    ladder, sensitivity, provenance = published
    only_still = copy.deepcopy(sensitivity)
    only_still["sensitivity"] = [r for r in only_still["sensitivity"]
                                 if r["slip"] == "free_slip"
                                 or r["label"] == "sensitivity_static_wall_control"]
    only_still["sensitivity"] = [r for r in only_still["sensitivity"]
                                 if r["label"] != "sensitivity_free_slip_control"]
    v = cv.case_verdict(ladder, only_still, provenance=provenance)
    g5 = _gate(v, "G5")
    assert g5["status"] == "WITHHELD"
    assert "found 0" in g5["reason"]


def test_one_rung_ladder_fails_g6(cv, published):
    """The historical F1 defect: a one-rung 'ladder' must not pass convergence."""
    ladder, sensitivity, provenance = published
    one = copy.deepcopy(ladder)
    one["runs"] = [copy.deepcopy(max(ladder["runs"],
                                     key=lambda r: r["resolution_delta_over_dx"]))]
    # the defective assembler also wrote G6_not_growing: true and all_gates: true
    one["convergence"] = {"resolutions": [18], "complete": True,
                          "G6_not_growing": True, "all_gates": True,
                          "l2_by_resolution": [one["runs"][0]["metrics"]
                                               ["amplitude_l2_half_space_pct"]]}
    v = cv.case_verdict(one, sensitivity, provenance=provenance)
    assert _gate(v, "G6")["status"] == "FAIL"
    assert v["overall"] == "FAIL"


def test_stored_flag_the_rows_do_not_support_fails(cv, published):
    ladder, sensitivity, provenance = published
    tampered = copy.deepcopy(ladder)
    tampered["convergence"]["G6_not_growing"] = False
    v = cv.case_verdict(tampered, sensitivity, provenance=provenance)
    g6 = _gate(v, "G6")
    assert g6["status"] == "FAIL"
    assert "contradict" in g6["reason"]
    assert v["overall"] == "FAIL"


def test_hand_edited_metric_that_the_rows_do_not_support_fails(cv, published):
    ladder, sensitivity, provenance = published
    tampered = copy.deepcopy(ladder)
    finest = max(tampered["runs"], key=lambda r: r["resolution_delta_over_dx"])
    finest["metrics"]["amplitude_l2_half_space_pct"] = 1e-4   # flattering edit
    v = cv.case_verdict(tampered, sensitivity, provenance=provenance)
    assert v["consistency"]["ok"] is False
    assert any("amplitude_l2_half_space_pct" in p
               for p in v["consistency"]["problems"])
    assert v["overall"] == "FAIL"


# --------------------------------------------------------------------------- #
# CLI / committed artefact
# --------------------------------------------------------------------------- #

def test_cli_exit_codes_encode_the_verdict(cv, published, tmp_path):
    ladder, sensitivity, provenance = published
    (tmp_path / "results_ladder.json").write_text(json.dumps(ladder))
    (tmp_path / "results_sensitivity.json").write_text(json.dumps(sensitivity))
    assert cv.main(["--out-dir", str(tmp_path), "--check"]) == 0

    (tmp_path / "results_sensitivity.json").unlink()
    assert cv.main(["--out-dir", str(tmp_path), "--check"]) == 2  # withheld

    bad = copy.deepcopy(ladder)
    bad["convergence"]["complete"] = False
    (tmp_path / "results_ladder.json").write_text(json.dumps(bad))
    assert cv.main(["--out-dir", str(tmp_path), "--check"]) == 1  # fail


def test_committed_verdict_and_page_match_the_code(cv, verdict):
    committed = json.loads((REPO / "benchmarks/stokes_second/case_verdict.json").read_text())
    assert committed["overall"] == verdict["overall"]
    assert committed["counts"] == verdict["counts"]
    for gid in ("G1", "G2", "G3", "G4", "G5", "G6", "rows_in_layer"):
        assert _gate(committed, gid)["status"] == _gate(verdict, gid)["status"]
        assert _gate(committed, gid)["measured"] == _gate(verdict, gid)["measured"]
    assert committed["artifacts"] == verdict["artifacts"]   # hashes of the bundle
    page = (REPO / "docs/stokes_second_case_verdict.md").read_text()
    assert f"**Overall: {verdict['overall']}**" in page
    for gid in ("G1", "G2", "G3", "G4", "G5", "G6"):
        assert f"| {gid} |" in page
    assert "Stored `metrics` cross-check: consistent" in page


def test_json_verdict_is_finite_and_serialisable(verdict):
    text = json.dumps(verdict, default=verdict and (lambda o: o))
    assert "NaN" not in text and "Infinity" not in text
    for r in verdict["recomputed_per_rung"]:
        for value in r.values():
            if isinstance(value, float):
                assert math.isfinite(value)
