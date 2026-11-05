"""Gates for the free-surface pressure closure resolution grid.

The measurement itself (``benchmarks/laplace_pressure_closure/closure_grid.py``)
takes minutes of SPH time and is not run here.  What is pinned:

* anti-drift -- the committed ``docs/verification_free_surface_pressure_closure.md``
                must equal a fresh regeneration from the JSON evidence, so no
                number in the record is hand typed;
* schema     -- every recorded case carries the settings and readings the
                document claims to report;
* physics    -- the closure beats the Tait clamp at *every* resolution tested,
                the grid reproduces the committed single-resolution ladder, and
                the verdict printed in the document follows the recorded
                numbers rather than a wish;
* defaults   -- the closure stays opt-in, so every earlier published number is
                still reproducible.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "laplace_pressure_closure"
GRID = BENCH / "raw" / "closure_grid.json"
LADDER = BENCH / "raw" / "relax_ladder.json"
DOC = REPO / "docs" / "verification_free_surface_pressure_closure.md"

REQUIRED_KEYS = {
    "case", "dx_m", "n_particles", "sound_speed", "steps", "closure",
    "delta_p_final_Pa", "err_final_pct", "delta_p_tail_Pa", "err_tail_pct",
    "within_3pct_tail", "within_15pct_tail", "vmax_over_vref_final",
    "rho_core_over_rho0", "compression_required",
    "compression_achieved_over_required",
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def grid():
    if not GRID.exists():
        pytest.skip("run benchmarks/laplace_pressure_closure/closure_grid.py first")
    return json.loads(GRID.read_text())


@pytest.fixture(scope="module")
def doc_module():
    path = BENCH / "make_closure_grid_doc.py"
    if not path.exists():
        pytest.skip("doc generator missing")
    return _load("closure_grid_doc", path)


# --------------------------------------------------------------------------- #
# anti-drift
# --------------------------------------------------------------------------- #
def test_committed_document_is_regenerated_from_the_evidence(grid, doc_module):
    """No number in the verification record may be typed by hand."""
    assert DOC.is_file(), "docs/verification_free_surface_pressure_closure.md is missing"
    assert DOC.read_text() == doc_module.render(), (
        "docs/verification_free_surface_pressure_closure.md is stale -- regenerate it with "
        "`PYTHONPATH=src python benchmarks/laplace_pressure_closure/make_closure_grid_doc.py`"
    )


def test_every_recorded_reading_appears_in_the_document(grid):
    """Each case's tail error is quoted in the record, formatted from the JSON."""
    text = DOC.read_text()
    for case in grid["cases"]:
        assert f"{case['err_tail_pct']:+.1f}" in text, (case["case"], "tail")
        assert f"{case['err_final_pct']:+.1f}" in text, (case["case"], "final")


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
def test_every_case_carries_the_settings_and_readings_it_claims(grid):
    assert grid["cases"], "the grid is empty"
    names = [c["case"] for c in grid["cases"]]
    assert len(names) == len(set(names)), "duplicate case names"
    for case in grid["cases"]:
        missing = REQUIRED_KEYS - set(case)
        assert not missing, (case["case"], sorted(missing))
        assert case["closure"]["enabled"] in (True, False)
        assert case["n_particles"] > 0 and case["steps"] > 0
        assert case["err_tail_pct"] == pytest.approx(
            100.0 * (case["delta_p_tail_Pa"] - grid["reference"]["analytic_dp_Pa"])
            / grid["reference"]["analytic_dp_Pa"], rel=1e-6)
        assert case["within_15pct_tail"] == (abs(case["err_tail_pct"]) < 15.0)
        assert case["within_3pct_tail"] == (abs(case["err_tail_pct"]) < 3.0)


def test_the_grid_pairs_every_resolution_with_and_without_the_closure(grid):
    by_name = {c["case"]: c for c in grid["cases"]}
    assert grid["closure_gain"], "no closure gain pairs were computed"
    for pair in grid["closure_gain"]:
        off = [c for c in by_name.values()
               if c["dx_m"] == pair["dx_m"] and c["sound_speed"] == pair["sound_speed"]
               and not c["closure"]["enabled"]]
        on = [c for c in by_name.values()
              if c["dx_m"] == pair["dx_m"] and c["sound_speed"] == pair["sound_speed"]
              and c["closure"]["enabled"]]
        assert len(off) == 1 and len(on) == 1, (pair["dx_m"], pair["sound_speed"])


# --------------------------------------------------------------------------- #
# physics claims
# --------------------------------------------------------------------------- #
def test_closure_improves_the_tail_readout_at_every_resolution(grid):
    """The closure must beat the Tait clamp wherever it was run, or it is not a fix."""
    for pair in grid["closure_gain"]:
        assert abs(pair["on_err_tail_pct"]) < abs(pair["off_err_tail_pct"]), pair
        assert pair["gain_pt"] > 0.0, pair


def test_grid_reproduces_the_single_resolution_ladder(grid):
    """The dx = 2 mm grid cases must be the same runs as the committed ladder."""
    if not LADDER.exists():
        pytest.skip("no single-resolution ladder to compare against")
    ladder = {c["case"]: c for c in json.loads(LADDER.read_text())["cases"]}
    by_name = {c["case"]: c for c in grid["cases"]}
    pairs = (("grid_dx2.0mm_off", "closure_off"),
             ("grid_dx2.0mm_on", "closure_on_interior_layers2"))
    checked = 0
    for gname, lname in pairs:
        if gname not in by_name or lname not in ladder:
            continue
        grid_err = by_name[gname]["err_tail_pct"]
        ladder_err = ladder[lname]["tail_rel_error_pct"]
        assert grid_err == pytest.approx(ladder_err, abs=0.5), (gname, grid_err, ladder_err)
        checked += 1
    assert checked >= 1, "no overlapping case to cross-check"


def test_the_verdict_in_the_document_follows_the_numbers(grid):
    """The PASS/FAIL cells and the ledger grade are derived, not asserted by prose."""
    text = DOC.read_text()
    on_cases = [c for c in grid["cases"] if c["closure"]["enabled"]]
    off_cases = [c for c in grid["cases"] if not c["closure"]["enabled"]]
    best_on = min(on_cases, key=lambda c: abs(c["err_tail_pct"]))
    best_off = min(off_cases, key=lambda c: abs(c["err_tail_pct"]))
    err = abs(best_on["err_tail_pct"])
    assert f"| {'PASS' if err < 15.0 else 'FAIL'} |" in [
        ln for ln in text.splitlines() if ln.startswith("| platform, 15 %")][0]
    assert f"| {'PASS' if err < 3.0 else 'FAIL'} |" in [
        ln for ln in text.splitlines() if ln.startswith("| paper, 3 %")][0]
    adami = [ln for ln in text.splitlines() if ln.startswith("| Adami spurious current")]
    assert adami, "the record must state the spurious-current verdict"
    vmax_ok = best_on["vmax_over_vref_final"] < 0.1
    assert f"| {'PASS' if vmax_ok else 'FAIL'} |" in adami[0], adami[0]
    grade = "PASS" if err <= 3.0 else "MARGINAL" if err <= 15.0 else "WITHHELD"
    assert f"Ledger grade this record earns: **{grade}**" in text, (
        f"the document must state the derived grade {grade}")
    assert "| closure gain (control) |" in text
    gain_ok = err < abs(best_off["err_tail_pct"])
    gain_row = [ln for ln in text.splitlines() if ln.startswith("| closure gain (control)")][0]
    assert f"| {'PASS' if gain_ok else 'FAIL'} |" in gain_row, gain_row


def test_the_record_does_not_overclaim_its_grade(grid):
    """No closure-on case inside 15 % may still be described as an open failure."""
    text = DOC.read_text()
    on_cases = [c for c in grid["cases"] if c["closure"]["enabled"]]
    inside = [c["case"] for c in on_cases if abs(c["err_tail_pct"]) < 15.0]
    outside = [c["case"] for c in on_cases if abs(c["err_tail_pct"]) >= 15.0]
    if outside:
        assert "still not closed" in text, (
            "closure-on cases outside 15 % are recorded but the document does not say so")
    if inside:
        assert "still not closed" not in text.split("## 5.")[0], (
            "the document claims an open failure although a closure-on case is inside 15 %")


# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #
def test_the_closure_stays_opt_in(grid):
    """Earlier published numbers must remain reproducible: the default is still off."""
    from tensorsph.droplet_impact import Config

    cfg = Config()
    assert cfg.surface_pressure_closure is False
    assert cfg.surface_pressure_donors == "interior"
    assert cfg.surface_pressure_layers >= 1
    for case in grid["cases"]:
        assert case["closure"]["threshold"] == pytest.approx(cfg.surface_pressure_threshold)


def test_the_evidence_is_registered_in_the_verification_ledger():
    """A machine-readable record that no ledger claim points at is a silent artifact."""
    ledger_path = REPO / "benchmarks" / "verification_ledger" / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    artifacts = {e["artifact"] for e in ledger["entries"]}
    assert "benchmarks/laplace_pressure_closure/raw/closure_grid.json" in artifacts
