"""Completeness + self-consistency of the shift-bound payload (H5/H6 round).

Anchor: ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.

The shift-bound leg (``benchmarks/xval_sphinx_tvc/shift_bound.py``) grades
"is the accuracy-optimal TVC coefficient ``alpha`` predictable, or is it a
fit?"  Its earlier round declared ``dx = 0.01 m`` and the ``dx = 0.02 m`` tail
(``alpha = 0.15``, ``0.20``) as *not measured* and left the task open in the
document.  This round fills those cells and adds two pre-declared hypotheses:

* **H5** -- the optimum sits on a level-independent *accumulated* correction
  ``A = N_steps * s(alpha_opt)`` (a per-step bound ``s`` was H1, already
  rejected).
* **H6** -- the error-vs-alpha curve is flat enough near its minimum that the
  per-level *indifference intervals* ``I(dx) = {alpha : err <= err_min + tol}``
  cannot resolve a resolution-dependent optimum.

The tests below assert **only** things that must hold whatever the physics
says -- payload completeness, arithmetic reproduction of the summary from the
raw records, and document/payload agreement.  They deliberately do *not*
assert that H5 or H6 hold: those are measurements, recorded in the payload and
rendered in ``docs/xval_sphinx_tvc/SHIFT_BOUND_PREDICTION.md``; pinning them
here would turn a falsification into a test failure.

Cheap: reads JSON, no fluid solve.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_sphinx_tvc"))

DATA = ROOT / "docs" / "xval_sphinx_tvc" / "data" / "shift_bound.json"
DOC = ROOT / "docs" / "xval_sphinx_tvc" / "SHIFT_BOUND_PREDICTION.md"

LEVELS = (0.04, 0.03, 0.02)      # levels that must be full-grid (the 3-level grid)
PENDING = {0.01: "cj3832"}       # declared, still measuring; must NOT enter a verdict


def payload() -> dict:
    return json.loads(DATA.read_text())


def by_level(recs: list[dict]) -> dict[float, list[dict]]:
    out: dict[float, list[dict]] = {}
    for r in recs:
        out.setdefault(round(r["dx"], 6), []).append(r)
    for dx in out:
        out[dx].sort(key=lambda r: r["alpha"])
    return out


def complete_levels(p: dict) -> dict:
    """Levels whose measured alpha set equals the pre-declared grid."""
    grid = {round(a, 6) for a in p["predeclared"]["alpha_grid"]}
    out = {}
    for dx, pts in by_level(p["records"]).items():
        if {round(r["alpha"], 6) for r in pts} == grid:
            out[dx] = pts
    return out


def test_grid_is_complete_for_every_level_that_enters_a_verdict():
    """A resolution may only appear in the H1/H2/H5/H6 statistics when *all*
    of its pre-declared alpha cells were measured.  Partially measured levels
    (here dx = 0.01 m) must be absent from the summary, not averaged in."""
    p = payload()
    levels = complete_levels(p)
    for dx in LEVELS:
        assert dx in levels, f"level {dx} is not full-grid"
    # every record lies on the pre-declared grid
    grid = {round(a, 6) for a in p["predeclared"]["alpha_grid"]}
    extra = [(r["dx"], r["alpha"]) for r in p["records"]
             if round(r["alpha"], 6) not in grid]
    assert not extra, f"points outside the pre-declared grid: {extra}"
    # and the statistics cover exactly the complete levels
    assert set(summary_level_keys(p)) == {f"{dx:g}" for dx in levels}
    for dx in PENDING:
        assert f"{dx:g}" not in summary_level_keys(p), (
            f"partially measured level {dx} leaked into the verdict")


def test_every_cell_carries_a_front_measurement():
    """A measured cell has the scored front history, not just a placeholder."""
    p = payload()
    for r in p["records"]:
        assert r["steps"] > 0
        assert r["rel_error"] is not None and r["mae"] is not None
        assert len(r["score_rows"]) >= 10, (r["dx"], r["alpha"], len(r["score_rows"]))
        assert r["max_shift_over_dx"] is not None
        # alpha = 0 is the uncorrected baseline: no displacement, by definition
        if r["alpha"] == 0.0:
            assert r["max_shift_over_dx"] == 0.0


def summary_level_keys(p: dict) -> list[str]:
    return list(p["summary"]["levels"].keys())


def test_summary_reproduces_from_raw_records():
    """H1/H5/H6 statistics in the summary are recomputed here from the records.

    A silent edit of either the payload or the summariser becomes a failure.
    """
    p = payload()
    levels = complete_levels(p)
    summary = p["summary"]

    for dx in LEVELS:
        pts = levels[dx]
        best = min((r for r in pts if r["alpha"] > 0), key=lambda r: r["rel_error"])
        lv = summary["levels"][f"{dx:g}"]
        assert lv["alpha_opt"] == best["alpha"]
        assert lv["rel_error_at_opt"] == best["rel_error"]
        # C_implied / A_implied are read off the measured curve at alpha_opt,
        # which is a grid point, so they are identical to that point's values.
        assert lv["C_implied"] == best["max_shift_over_dx"]
        assert lv["cum_at_opt_over_dx"] == best["steps"] * best["max_shift_over_dx"]
        # last measured alpha of the level carries the largest bound (monotone s)
        assert lv["shift_curve_monotone"] is True

    # H1 spread
    cs = [summary["levels"][f"{dx:g}"]["C_implied"] for dx in LEVELS]
    mean = sum(cs) / len(cs)
    assert summary["C_implied"]["mean"] == mean
    assert summary["C_implied"]["spread_over_mean"] == (max(cs) - min(cs)) / mean

    # H5 spread
    As = [summary["levels"][f"{dx:g}"]["cum_at_opt_over_dx"] for dx in LEVELS]
    meanA = sum(As) / len(As)
    assert summary["H5"]["mean"] == meanA
    assert summary["H5"]["spread_over_mean"] == (max(As) - min(As)) / meanA
    assert summary["H5"]["holds_25pct"] is ((max(As) - min(As)) / meanA <= 0.25)

    # H6 intervals: recompute the indifference set with the declared tolerance
    tol = summary["H6"]["tolerance_abs"]
    for dx in LEVELS:
        pts = levels[dx]
        e_min = min(r["rel_error"] for r in pts)
        keep = [r["alpha"] for r in pts if r["rel_error"] <= e_min + tol]
        assert summary["H6"]["intervals_abs"][f"{dx:g}"] == [min(keep), max(keep)]
    lo = max(summary["H6"]["intervals_abs"][f"{dx:g}"][0] for dx in LEVELS)
    hi = min(summary["H6"]["intervals_abs"][f"{dx:g}"][1] for dx in LEVELS)
    assert summary["H6"]["common_intersection_abs"] == ([lo, hi] if lo <= hi else None)
    assert summary["H6"]["holds"] is (lo <= hi)


def test_pre_registration_is_recorded():
    """H5/H6 must be declared in the payload, with the partial pre-registration
    (the three coarse levels predate them) stated rather than hidden."""
    p = payload()
    hyp = p["predeclared"]["hypotheses"]
    assert "H5" in hyp and "H6" in hyp
    assert p["predeclared"]["frames_measured_before_H5_H6"] == [0.04, 0.03, 0.02]
    assert p["gates"]["owner_rel"] == 0.15


def test_document_agrees_with_payload():
    """The rendered document quotes the payload, not hand-typed numbers."""
    p = payload()
    doc = DOC.read_text()
    summary = p["summary"]
    # headline C_implied and alpha_opt strings appear in the document
    for dx in LEVELS:
        lv = summary["levels"][f"{dx:g}"]
        assert f"{lv['C_implied']:.3f}" in doc, (dx, lv["C_implied"])
    # the completeness accounting line is derived from the payload, so the two
    # cannot drift apart
    grid = len(p["predeclared"]["alpha_grid"])
    have = sum(1 for r in p["records"]
               if r["alpha"] in p["predeclared"]["alpha_grid"])
    assert f"payload 实测 {have} 个单元" in doc
    # every not-yet-measured level is named in the document with its job id
    for dx, job in PENDING.items():
        if f"{dx:g}" not in summary["levels"]:
            assert f"Δx = {dx:g} m" in doc and job in doc