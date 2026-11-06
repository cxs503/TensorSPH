"""Gates on the ``dx``/``c0`` closure ladder of the Young-Laplace verification.

``docs/verification_laplace_pressure_closure.md`` used to rest on a single
``(dx, c0)`` point, which cannot separate a closure effect from a one-mesh
accident.  ``relax_ladder.py`` now writes one evidence file per point and
``ladder_table.py`` collapses them; these tests pin the two things that could
silently go wrong again:

* **provenance** -- a file's header must be the point its cases were measured
  at.  The earlier script resumed *any* existing file into the new record, so
  ``--dx 0.0015`` produced a 1.5 mm header holding 2 mm cases.
* **the statements the table licenses** -- which variants are resolution
  independent, and that being inside the +/-15 % gate does *not* by itself
  mean the spurious current (Adami's transient criterion) is acceptable.

The numeric expectations are pinned to ``raw/ladder_table.json`` as recorded,
in the same style as ``tests/test_laplace_pressure_closure.py``: a re-run that
moves them is meant to fail here and force the record to be updated.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure_closure"
RAW = BENCH / "raw"
TABLE = RAW / "ladder_table.json"
GATE_PCT = 15.0
ADAMI_VMAX = 0.1


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def table():
    if not TABLE.exists():
        pytest.skip("run benchmarks/laplace_pressure_closure/ladder_table.py first")
    return json.loads(TABLE.read_text())


# --------------------------------------------------------------------------- #
# provenance: one file per (dx, c0) point, header checked before resuming
# --------------------------------------------------------------------------- #
def test_each_ladder_file_owns_a_distinct_mesh_and_sound_speed():
    seen = {}
    for path in sorted(RAW.glob("relax_ladder*.json")):
        rec = json.loads(path.read_text())
        key = (rec["dx_m"], rec["sound_speed"])
        assert key not in seen, (
            f"{path.name} and {seen[key]} both claim dx={key[0]}, c0={key[1]}: a point "
            "was overwritten instead of being written to its own file")
        seen[key] = path.name
    assert len(seen) >= 2, "the point of the ladder is more than one (dx, c0)"


def test_baseline_point_keeps_its_historical_filename():
    """Existing readers (make_doc, figures, tests) address the 2 mm/20 m/s file by name."""
    ladder = _load("relax_ladder")
    assert ladder.out_path(0.002, 20.0).name == "relax_ladder.json"
    other = ladder.out_path(0.0015, 20.0)
    assert other.name != "relax_ladder.json"
    assert "1.5mm" in other.name and other.parent == RAW
    assert ladder.out_path(0.002, 5.0).name != "relax_ladder.json"


def test_resume_refuses_mismatched_points(tmp_path, monkeypatch):
    """The defect being guarded: resuming a foreign file merged its cases under a new header."""
    ladder = _load("relax_ladder")
    stale = {"dx_m": 0.002, "sound_speed": 20.0, "cases": [{"case": "closure_off"}]}
    target = tmp_path / "relax_ladder_dx1.5mm_c20.json"
    target.write_text(json.dumps(stale))
    assert ladder.resume_cases(target, 0.0015, 20.0) == [], \
        "a 2 mm record must not be reused for a 1.5 mm request"
    # ... while the very same point does resume, so a partial run stays resumable.
    assert ladder.resume_cases(target, 0.002, 20.0) == [{"case": "closure_off"}]
    assert ladder.resume_cases(tmp_path / "absent.json", 0.002, 20.0) == []


# --------------------------------------------------------------------------- #
# the measured table
# --------------------------------------------------------------------------- #
def test_table_covers_a_resolution_ladder_at_one_sound_speed(table):
    at_c0 = [p for p in table["points"] if abs(p["sound_speed"] - 20.0) < 1e-9]
    dxs = sorted({p["dx_m"] for p in at_c0})
    assert len(dxs) >= 2, f"need >= 2 resolutions at c0 = 20 m/s, got {dxs}"
    for dx in dxs:
        names = {c["case"] for p in at_c0 if p["dx_m"] == dx for c in p["cases"]}
        assert {"closure_off", "closure_on_interior_layers2"} <= names, (dx, names)


def test_cases_carry_their_source_file(table):
    for point in table["points"]:
        for case in point["cases"]:
            assert case["source_file"] == point["source_file"]
            assert case["n_particles"] > 0


def test_tail_errors_are_finite_everywhere(table):
    for point in table["points"]:
        for case in point["cases"]:
            assert math.isfinite(case["tail_rel_error_pct"]), (point["source_file"], case["case"])
            assert math.isfinite(case["vmax_over_vref"])


def test_the_tait_clamp_deficit_does_not_follow_resolution(table):
    """C3 at the ladder scale: with the closure off the deficit is the same at every dx.

    A residual that moved with dx would be a discretisation effect (and could be
    engineered away by refining); one that does not move is the closure signature
    established in ``docs/verification_laplace_pressure_floor.md``.
    """
    rows = [r for r in table["by_variant"]["closure_off"]
            if abs(r["sound_speed"] - 20.0) < 1e-9]
    assert len(rows) >= 2
    errs = [r["tail_rel_error_pct"] for r in rows]
    assert max(errs) - min(errs) < 5.0, errs
    assert all(e < -30.0 for e in errs), errs


def test_interior_extrapolation_is_recorded_against_the_gate(table):
    """Pin the measured value of the best variant at every tested point."""
    rows = table["by_variant"]["closure_on_interior_layers2"]
    assert rows, "the interior variant was not run"
    for r in rows:
        inside = abs(r["tail_rel_error_pct"]) < GATE_PCT
        assert r["within_15pct"] is inside, (r, inside)
        assert abs(r["tail_rel_error_pct"]) < 60.0, r


def test_gate_passes_are_reported_with_their_spurious_current(table):
    """A Laplace gate pass is not a capillary capability on its own.

    Adami et al. (2012) require the parasitic current to stay below
    ``v_max/v_ref = 0.1``; a drop that reads the right core pressure while
    still sloshing has simply traded one error for another.  The table must
    carry both numbers for every case, and the evidence is that refinement --
    not a looser tolerance -- is what brings them together.
    """
    for name, rows in table["by_variant"].items():
        for r in rows:
            assert math.isfinite(r["vmax_over_vref"]), (name, r)
    clean = [(r["dx_m"], r["sound_speed"]) for r in
             table["by_variant"]["closure_on_interior_layers2"]
             if r["within_15pct"] and r["vmax_over_vref"] < ADAMI_VMAX]
    assert clean, ("the interior closure meets neither gate jointly anywhere any more; "
                   "re-measure before rewriting this test")
    # the coarsest point is the one that is *not* clean: the spurious current,
    # not the pressure error, is what resolution fixes first
    coarse = [r for r in table["by_variant"]["closure_on_interior_layers2"]
              if abs(r["dx_m"] - max(x["dx_m"] for x in
                                     table["by_variant"]["closure_on_interior_layers2"])) < 1e-12
              and abs(r["sound_speed"] - 20.0) < 1e-9]
    if coarse:
        assert coarse[0]["vmax_over_vref"] > ADAMI_VMAX, coarse[0]


def test_the_cheapest_variant_is_not_the_most_accurate_one(table):
    """Sanity on the donor semantics: `all` dilutes with clamped zeros, so or-else interior wins."""
    def best(name):
        rows = [r for r in table["by_variant"].get(name, [])
                if abs(r["sound_speed"] - 20.0) < 1e-9 and abs(r["dx_m"] - 0.002) < 1e-12]
        return abs(rows[0]["tail_rel_error_pct"]) if rows else None

    interior, lit = best("closure_on_interior_layers2"), best("closure_on_all_donors")
    if interior is None or lit is None:
        pytest.skip("variant not measured at the baseline point")
    assert interior < lit, (interior, lit)
