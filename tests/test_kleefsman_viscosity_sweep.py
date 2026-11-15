"""Invariants of the Kleefsman viscosity sweep (fast: no SPH re-run, < 1 s).

The claim under test is that the +24 % arrival-time bias of the obstacle case is
a *viscosity-setting* effect, not a discretisation limit.  These tests pin the
chain of evidence rather than the prose:

* the sweep script really puts the requested ν into the solver configuration
  it hands to ``run_case`` (a silently ignored knob would fake the whole
  result);
* every recorded relative error / pass flag in ``summary.json`` is recomputed
  from the recorded raw values and the registered experiment;
* Re and the bed boundary layer δ are recomputed from ν, L and the arrival
  time, so the physical interpretation table cannot drift from the numbers;
* the monotonicity verdict is recomputed from the arrival series;
* the generated document carries the same numbers as the artifacts.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "kleefsman_obstacle"
RESULTS = BENCH / "results" / "viscosity_sweep"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Sweep = _load_module(BENCH / "viscosity_sweep.py", "kleefsman_viscosity_sweep")

SUMMARYS = sorted(RESULTS.glob("*/summary.json"))
needs_artifacts = pytest.mark.skipif(
    not SUMMARYS, reason="viscosity sweep artifacts not generated in this checkout")


def _summaries():
    return [json.loads(p.read_text()) for p in SUMMARYS]


def test_config_builder_applies_the_requested_viscosity():
    ref = json.loads((BENCH / "reference.json").read_text())
    cfg = Sweep.build_cfg(ref, dx=0.04, t_end=1.2, viscosity=1e-4,
                          sound_speed=None, gauge_aperture=0.08)
    assert cfg.viscosity == 1e-4
    # dt must still satisfy the acoustic limit for the configured c0
    assert cfg.dt <= cfg.dt_acoustic() * (1 + 1e-12)
    # the coarse configuration must pass the case validator, otherwise the
    # sweep would silently be running an invalid setup
    from tensorsph.kleefsman_obstacle import validate_case
    validate_case(cfg)


def test_config_builder_differs_only_in_viscosity():
    ref = json.loads((BENCH / "reference.json").read_text())
    a = Sweep.build_cfg(ref, 0.04, 1.2, 0.01, None, 0.08)
    b = Sweep.build_cfg(ref, 0.04, 1.2, 0.0001, None, 0.08)
    assert a.viscosity != b.viscosity
    for field in ("dx", "t_end", "sound_speed", "rho0", "gamma", "gravity",
                  "h_factor", "dt", "gauge_aperture", "probe_offset",
                  "arrival_threshold", "wall_layers"):
        assert getattr(a, field) == getattr(b, field), field


@needs_artifacts
def test_recorded_errors_recompute_from_raw_values():
    tol = _summaries()[0]["tolerance_relative"]
    assert tol == pytest.approx(0.10)
    seen = 0
    for s in _summaries():
        exp = s["experiment"]
        for r in s["runs"]:
            for key, got, want in (
                    ("peak_height", r["peak_height_m"], exp["peak_height_m"]),
                    ("arrival_time", r["arrival_time_s"], exp["arrival_time_s"])):
                assert key in r["rel_error"], (r["tag"], key)
                err = abs(got - want) / abs(want)
                assert r["rel_error"][key] == pytest.approx(err, rel=1e-12)
                assert r["pass"][key] is bool(err < tol)
                seen += 1
    assert seen >= 2 * 7          # two metrics for every recorded run


@needs_artifacts
def test_reynolds_and_boundary_layer_are_the_formulas_claimed():
    ref = json.loads((BENCH / "reference.json").read_text())
    # gate -> probe: the probe sits 0.05 m upstream of the box face by the
    # code convention recorded in reference.json
    travel = (ref["geometry_m"]["obstacle_upstream_face_x"] - 0.05
              - ref["geometry_m"]["column_length"])
    for s in _summaries():
        for r in s["runs"]:
            u = travel / r["arrival_time_s"]
            assert r["mean_front_speed"] == pytest.approx(u, rel=1e-12)
            if r["viscosity"] > 0:
                assert r["reynolds"] == pytest.approx(u * travel / r["viscosity"], rel=1e-12)
                delta = math.sqrt(r["viscosity"] * travel / u)
                assert r["boundary_layer_m"] == pytest.approx(delta, rel=1e-12)
                assert r["boundary_layer_over_dx"] == pytest.approx(delta / r["dx"], rel=1e-12)


@needs_artifacts
def test_monotonicity_flag_matches_the_arrival_series():
    for s in _summaries():
        for tag, v in s["monotonic_checks"].items():
            arr = v["arrival_s"]
            expect = all(b <= a + 1e-12 for a, b in zip(arr, arr[1:]))
            assert v["arrival_monotonically_decreasing_in_1_over_nu"] is expect, tag
            assert v["arrival_span_s"] == pytest.approx(max(arr) - min(arr), rel=1e-12)
            # viscosity ordering must be strictly descending in the sweep list
            nu = v["viscosity_desc"]
            assert all(b < a for a, b in zip(nu, nu[1:])), tag


@needs_artifacts
def test_sweep_is_stable_where_it_claims_payoff():
    """A viscosity that buys accuracy by blowing the scheme up is not a fix."""
    for s in _summaries():
        for r in s["runs"]:
            assert r["finished"], r["tag"]
            assert r["mass_error_rel"] < 1e-9, r["tag"]
            assert r["boundary_escapes"] == 0, r["tag"]
            assert r["max_density_compression"] < 0.30, r["tag"]
            assert r["max_speed"] < 20.0, r["tag"]


@needs_artifacts
def test_document_quotes_the_artifact_numbers():
    doc = ROOT / "docs" / "verification_kleefsman_viscosity_ordering.md"
    assert doc.exists(), "run make_doc_viscosity.py"
    text = doc.read_text()
    runs = [r for s in _summaries() for r in s["runs"]]
    best = min(runs, key=lambda r: max(r["rel_error"].values()))
    assert f"`{best['tag']}`" in text
    for r in runs:
        assert f"| `{r['tag']}` |" in text
        assert f"{r['arrival_time_s']:.4f}" in text or r["arrival_time_s"] is None
    assert "Kleefsman" in text and "10.1016/j.jcp.2004.12.007" in text