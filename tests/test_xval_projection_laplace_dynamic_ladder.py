"""Gates for the dynamic-arm resolution ladder of XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE.

These tests do **not** re-run the ladder (the finest level is a 120-step march on
~7k particles with a PCG solve per step).  They gate the *record*
``benchmarks/laplace_sph/results_dynamic_ladder/summary.json`` and the claims
``docs/xval_projection_laplace_dynamic_ladder.md`` makes from it, so that a
change of the model, of the probes or of the doc cannot silently leave a
converging story behind.

Layers:

1. **provenance** — every declared gate threshold in the summary is identical to
   the constant in ``run_dynamic_ladder.py``: a later round may not relax a gate
   by editing the JSON.
2. **self-consistency** — the recorded verdicts are recomputed from the recorded
   measured values and thresholds; the fitted decay order is re-derived from the
   ladder rows.  A verdict that disagrees with its own numbers fails.
3. **physics gates** — the owner 15 % gate on every level, area drift, the exact
   projection (leftover divergence), the Adami spurious-current line, the
   monotone-in-resolution decay (H1), the horizon hold (H2) and the viscosity
   independence (H3).  These are the assertions the round stands or falls on.
4. **honest-FAIL lock** — if a declared gate is broken, the doc must say so.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_sph"
RES = BENCH / "results_dynamic_ladder"
SUMMARY = RES / "summary.json"
DOC = ROOT / "docs" / "xval_projection_laplace_dynamic_ladder.md"
sys.path.insert(0, str(BENCH))

pytestmark = pytest.mark.skipif(not SUMMARY.exists(),
                                reason="ladder record not present")


@pytest.fixture(scope="module")
def summary():
    return json.loads(SUMMARY.read_text())


@pytest.fixture(scope="module")
def harness():
    import run_dynamic_ladder as h
    return h


def fit_order(xs, ys):
    lx = [math.log10(x) for x in xs]
    ly = [math.log10(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    return (sum((a - mx) * (b - my) for a, b in zip(lx, ly))
            / sum((a - mx) ** 2 for a in lx))


# --------------------------------------------------------------------------- #
# 1. provenance
# --------------------------------------------------------------------------- #
def test_gate_thresholds_are_the_harness_constants(summary, harness):
    declared = summary["gates_declared"]
    assert declared == json.loads(json.dumps(harness.GATES_DECLARED)), (
        "declared gates in the record differ from run_dynamic_ladder.GATES_DECLARED "
        "-- a gate was edited after the fact")


def test_every_declared_job_is_present(summary, harness):
    expected = {s["name"] for s in harness.job_specs()}
    present = {r["job"] for r in summary["rows"]}
    assert present == expected, "missing jobs: %s" % (expected - present)
    assert summary["n_jobs_present"] == summary["n_jobs_expected"]


# --------------------------------------------------------------------------- #
# 2. self-consistency of the recorded verdicts
# --------------------------------------------------------------------------- #
def test_verdicts_agree_with_their_own_numbers(summary):
    for name, v in summary["gate_verdicts"].items():
        m, d, op = v["measured"], v["declared"], v["op"]
        if op == "<=":
            expect = m <= d
        elif op == ">=":
            expect = m >= d
        elif op == "==":
            expect = m == d
        else:  # pragma: no cover
            raise AssertionError("unknown op %r in gate %r" % (op, name))
        assert bool(v["pass"]) == bool(expect), (
            "gate %r records pass=%s but %.6g %s %.6g is %s"
            % (name, v["pass"], m, op, d, expect))


def test_orders_rederive_from_the_ladder_rows(summary):
    lad = summary["ladder_primary"]
    xs = [r["R_over_dx"] for r in lad]
    order = fit_order(xs, [r["umax_over_vref"] for r in lad])
    assert order == pytest.approx(summary["orders"]["spurious_in_R_over_dx"],
                                  rel=1e-9, abs=1e-12)


def test_every_run_is_backed_by_its_own_job_file(summary):
    for row in summary["rows"]:
        p = RES / (row["job"] + ".json")
        assert p.exists(), "summary lists %r but %s is absent" % (row["job"], p)
        blob = json.loads(p.read_text())
        for key in ("dp_rel_error_pct", "umax_over_vref_final",
                    "area_ratio_final", "nu_factor", "R_over_dx"):
            assert row[key] == pytest.approx(blob[key], rel=1e-12, abs=0.0), (
                "%s: summary %s=%r != job file %r" % (row["job"], key, row[key], blob[key]))


# --------------------------------------------------------------------------- #
# 3. the physics gates
# --------------------------------------------------------------------------- #
def test_laplace_jump_inside_owner_gate_at_every_level(summary):
    """Owner acceptance: |dp/(sigma/R) - 1| <= 15 % everywhere."""
    for r in summary["rows"]:
        assert abs(r["dp_rel_error_pct"]) <= 15.0, (
            "%s: dp error %+.2f %% > 15 %%" % (r["job"], r["dp_rel_error_pct"]))


def test_projection_is_exact_at_every_sampled_step(summary):
    worst = max((h.get("div_after_over_div_before", 0.0), r["job"])
                for r in summary["rows"]
                for h in json.loads((RES / (r["job"] + ".json")).read_text())["history"])
    assert worst[0] <= 1e-5, "leftover divergence %.2e in %s" % worst


def test_spurious_currents_fall_with_resolution(summary):
    """H1: the residual current is a lattice artefact, not a formulation defect."""
    lad = sorted(summary["ladder_primary"], key=lambda r: r["R_over_dx"])
    y = [r["umax_over_vref"] for r in lad]
    # monotone (within 10 % sampling noise) and a positive fitted order
    for a, b in zip(y, y[1:]):
        assert b <= a * 1.10, "spurious current grew with resolution: %r" % y
    assert summary["orders"]["spurious_in_R_over_dx"] >= 0.5


def test_area_is_conserved(summary):
    for r in summary["rows"]:
        assert abs(r["area_ratio_final"] - 1.0) <= 0.02, (
            "%s: area ratio %.5f" % (r["job"], r["area_ratio_final"]))


def test_horizon_holds(summary):
    """H2: a static solution must be invariant in time."""
    hz = summary["horizon"]
    assert hz, "horizon job missing"
    for h in hz:
        assert not h["diverged"]
        assert h["steps"] >= summary["setup"]["horizon_steps"]
        assert abs(h["dp_rel_error_pct_final"]) <= 15.0
        assert abs(h["area_ratio_final"] - 1.0) <= 0.02


def test_equilibrium_is_independent_of_the_damping_knob(summary):
    """H3: viscosity damps spurious currents, it must not set the jump."""
    vs = summary["viscosity_sweep"]
    assert len(vs) >= 3
    dp = [r["dp_rel_error_pct"] for r in vs]
    assert max(dp) - min(dp) <= 3.0, "dp spread over nu_factor = %.2f %%" % (max(dp) - min(dp))


def test_no_viscous_run_diverges(summary):
    viscous = [r for r in summary["rows"] if r["nu_factor"] > 0.0]
    assert viscous
    assert all(not r["diverged"] for r in viscous)


# --------------------------------------------------------------------------- #
# 4. the doc must not overstate
# --------------------------------------------------------------------------- #
def test_doc_reports_every_failed_gate_as_fail(summary):
    if not DOC.exists():
        pytest.skip("doc not generated")
    text = DOC.read_text()
    for name, v in summary["gate_verdicts"].items():
        if not v["pass"]:
            assert "`%s`" % name in text, "failed gate %r not in the doc" % name
            row = [ln for ln in text.splitlines() if "`%s`" % name in ln]
            assert any("**FAIL**" in ln for ln in row), (
                "failed gate %r is not marked FAIL in the doc" % name)


def test_doc_states_the_three_percent_answer(summary):
    if not DOC.exists():
        pytest.skip("doc not generated")
    text = DOC.read_text()
    assert "< 3 % ?" in text or "< 3 %% ?" in text, "doc must answer the 3 % question"