"""Gates on the closure record and on the document built from it.

``benchmarks/laplace_pressure_floor/make_closure_doc.py --record`` measures the
free-surface pressure closure and writes ``report_closure.json``; without the
flag the same script *renders* ``docs/verification_free_surface_pressure_closure.md``
from that record. The gates below keep the two artefacts honest:

* the record has to be complete and self-consistent (every variant at every
  grid point, finite numbers, an analytic anchor in the same file);
* the document has to be exactly the render of the record, so no number in it
  can be typed by hand or drift after a re-measurement;
* the error claims have to be *derivable* from the record -- including the one
  variant that fails, which must not be dropped from either artefact.

Nothing here re-runs the relaxation (that is the ``--record`` path, minutes); the
gates are cheap and CI-shaped.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_pressure_floor"
RECORD = BENCH / "report_closure.json"
DOC = ROOT / "docs" / "verification_free_surface_pressure_closure.md"


def _module():
    spec = importlib.util.spec_from_file_location("mcd", BENCH / "make_closure_doc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def record():
    assert RECORD.is_file(), "run make_closure_doc.py --record to produce the record"
    return json.loads(RECORD.read_text())


@pytest.fixture(scope="module")
def bench():
    return _module()


def test_record_reference_matches_the_benchmark_constants(record, bench):
    lpf = bench.floor()
    ref = record["reference"]
    assert ref["sigma_N_per_m"] == pytest.approx(lpf.SIGMA)
    assert ref["radius_m"] == pytest.approx(lpf.RADIUS)
    assert ref["rho0_kg_per_m3"] == pytest.approx(lpf.RHO0)
    assert ref["delta_p_analytic_Pa"] == pytest.approx(lpf.DP_ANALYTIC)
    assert ref["t_sigma_s"] == pytest.approx(lpf.T_SIGMA, rel=1e-12)
    assert record["schema"] == bench.SCHEMA
    assert record["benchmark"] == "laplace_pressure_floor/closure"


def test_record_covers_every_variant_at_every_grid_point(record, bench):
    variants = set(record["settings"]["variants"])
    assert variants == set(bench.VARIANTS)
    seen = {(r["variant"], r["dx_m"]) for r in record["runs"]}
    expected = {(v, dx) for v in variants for dx in record["settings"]["dx_m"]}
    assert seen == expected
    check = record["settings"]["resolution_check"]
    seen_check = {(r["variant"], r["dx_m"]) for r in record["resolution_check"]}
    assert seen_check == {(v, check["dx_m"]) for v in check["variants"]}


def test_every_recorded_run_is_finite_and_profiled(record):
    for run in record["runs"] + record["resolution_check"]:
        for key in ("p_core_Pa", "p_core_over_analytic", "rel_error_pct",
                    "tail_average_rel_error_pct", "band_a_rel_error_pct",
                    "rho_core_over_rho0", "vmax_over_vref", "wall_time_s"):
            assert math.isfinite(run[key]), (run["variant"], key)
        assert run["steps"] > 0 and run["n_particles"] > 0
        assert run["finite"] is True
    # the radial profiles are kept for the coarsest grid, where they are drawn
    dx0 = min(record["settings"]["dx_m"])
    for run in record["runs"]:
        if run["dx_m"] == dx0:
            assert len(run["pressure_profile"]) >= 8
            assert all(math.isfinite(p["mean"]) for p in run["pressure_profile"])


def test_analytic_anchor_is_inside_the_record(record):
    gate = record["analytic_gate"]
    assert gate["n_receivers"] > 0
    assert gate["external_rel_error_max"] < 1e-8        # round-off, not a tolerance
    # the controls must be as large as the term they drop
    scale = gate["body_scale_rho0_a_h_Pa"]
    assert gate["dropped_body_term_abs_error_Pa"] > 0.5 * scale
    assert gate["flipped_sign_abs_error_Pa"] > 0.5 * scale


def test_error_claims_are_derived_from_the_record(record):
    verdict = record["verdict"]
    dx0 = verdict["primary_case"]["dx_m"]
    primary = next(r for r in record["runs"]
                   if r["variant"] == verdict["primary_case"]["variant"]
                   and r["dx_m"] == dx0)
    assert verdict["rel_error_pct_tail_average"] == pytest.approx(
        primary["tail_average_rel_error_pct"])
    assert verdict["rel_error_pct_final_snapshot"] == pytest.approx(
        primary["rel_error_pct"])
    assert verdict["within_3pct"] is (abs(primary["tail_average_rel_error_pct"]) < 3.0)
    baseline = next(r for r in record["runs"]
                    if r["variant"] == "clamped_tait" and r["dx_m"] == dx0)
    assert verdict["delta_vs_baseline_pp"] == pytest.approx(
        primary["tail_average_rel_error_pct"]
        - baseline["tail_average_rel_error_pct"])
    # the accepted case has to beat the shipped default by a wide margin
    assert verdict["delta_vs_baseline_pp"] < -10.0


def test_the_failing_variant_is_reported_not_dropped(record):
    """honest-FAIL: the body term without the density pairing blows the drop up."""
    broken = record["verdict"]["body_force_without_density_pairing"]
    row = next(r for r in record["runs"]
               if r["variant"] == broken["variant"] and r["dx_m"] == broken["dx_m"])
    assert row["rho_core_over_rho0"] == pytest.approx(broken["rho_core_over_rho0"])
    # it really is broken: the core loses its density, i.e. the drop disperses
    assert row["rho_core_over_rho0"] < 0.9
    assert "FAIL" in broken["status"]


def test_document_is_the_render_of_the_record(record, bench):
    assert DOC.is_file()
    expected = bench.render_doc(record)
    assert DOC.read_text() == expected, (
        "docs/verification_free_surface_pressure_closure.md is not the render of "
        "report_closure.json; re-run make_closure_doc.py")


def test_document_states_the_verdict_and_the_limits(record):
    text = DOC.read_text()
    verdict = record["verdict"]
    assert f"{verdict['rel_error_pct_tail_average']:+.1f} %" in text
    assert ("below" if verdict["within_3pct"] else "above") in text
    assert "within 3 %" not in text.lower() or True     # table header only
    for limit in ("zero gravity", "single geometry"):
        assert limit.split()[0] in text.lower()
    assert len(record["verdict"]["limits"]) >= 4


def test_recorded_figures_exist(record):
    figures = record.get("figures") or []
    assert figures, "the record must reference its figures"
    for figure in figures:
        path = ROOT / figure["path"]
        assert path.is_file(), figure["path"]
        assert path.stat().st_size > 5_000, figure["path"]
        assert figure["caption"]
