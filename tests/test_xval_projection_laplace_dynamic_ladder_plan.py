"""Gates for the cost plan of the dynamic ladder (mission m22740).

The plan is an estimate, so it may not be quoted as evidence.  These tests keep
it tied to the tree: the job list is the declared one, the particle counts are
the runner's own lattice, a row is measured XOR estimated, and the total is the
sum of the rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_sph"
PLAN = BENCH / "results_dynamic_ladder" / "cost_plan.json"
DOC = ROOT / "docs" / "dev_report_m22740_dynamic_ladder_cost_plan.md"
sys.path.insert(0, str(BENCH))

pytestmark = pytest.mark.skipif(not PLAN.exists(), reason="no cost plan record")


@pytest.fixture(scope="module")
def plan():
    return json.loads(PLAN.read_text())


def test_plan_covers_every_declared_job(plan):
    import run_dynamic_ladder as h
    assert [r["job"] for r in plan["rows"]] == [s["name"] for s in h.job_specs()]


def test_particle_counts_are_the_runners_lattice(plan):
    import run_dynamic_ladder as h
    from tensorsph.laplace_isph import hex_block
    for row in plan["rows"]:
        pos, _m, _l, _h = hex_block(h.R, h.LEVELS[int(row["R_over_dx"])], h.OUTER, h.RHO)
        assert row["n_particles"] == int(pos.shape[0]), row["job"]


def test_a_row_is_measured_xor_estimated(plan):
    for row in plan["rows"]:
        assert not (row["measured_wall_s"] is not None
                    and row["estimated_wall_s"] is not None), row["job"]


def test_estimate_grows_with_resolution(plan):
    lad = sorted([r for r in plan["rows"]
                  if r["role"] == "ladder" and r["estimated_wall_s"] is not None],
                 key=lambda r: r["R_over_dx"])
    if len(lad) < 2:
        pytest.skip("fewer than two estimated ladder levels")
    walls = [r["estimated_wall_s"] for r in lad]
    assert walls == sorted(walls), "cost must grow with R/dx"


def test_total_is_the_sum_of_rows(plan):
    tot = sum((r["measured_wall_s"] if r["measured_wall_s"] is not None
               else (r["estimated_wall_s"] or 0.0)) for r in plan["rows"])
    assert abs(tot - plan["total_wall_s"]) <= 1e-6 * max(1.0, tot)


def test_doc_is_injected_from_the_plan(plan):
    text = DOC.read_text()
    assert "cost_plan.json" in text
    for row in plan["rows"][:3]:
        assert "`" + row["job"] + "`" in text
        assert str(row["n_particles"]) in text, row["job"]
    assert "NOT_RUN" in text
