"""Closure round (m21468) for CASE-SPH-WEDGE-ENTRY-2D: did refining to dx = 0.5 mm
make the peak slamming pressure judgeable?

The predeclared question, fixed before the ``dx = 0.5 mm`` run existed
(``benchmarks/case_sph_wedge_entry_2d/ladder_closure.py``, rule unchanged):

    U_g          = max(resolution_band.spread_rel, jackknife_rel) on the 61-frame ensemble
    judgeable_g  <=>  U_g <= 0.15
    verdict      = PASS / FAIL_above / FAIL_below only when the two finest judgeable
                   grids share the sign of r = obs/synth - 1 and |r| <= 0.15;
                   otherwise NOT_DECIDABLE.

These tests grade the *answer* from the committed artefacts, and they also grade
two things the answer would hide if nobody looked:

* the resolution-consistent bin band admits only **one** bin count at
  ``dx = 0.5 mm``, so its ``spread_rel`` is 0 by degeneracy (a one-point
  "spread").  The ladder must still report U > 15 % there, i.e. the jackknife
  must carry the uncertainty;
* the "refine the grid until the peak is judgeable" route (m17548 route b) is
  answered *negatively* by the same artefacts, so the negative result is
  asserted rather than left in prose.

No test here asserts a constant against a threshold: every number is read from
``docs/artifacts/case_sph_wedge_entry_2d_*`` and the verdict is recomputed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENS_DIR = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ensemble"
LADDER_JSON = (ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ladder"
               / "ladder_closure.json")
ERRATUM_JSON = (ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_table_erratum"
                / "erratum.json")

GATE = 0.15
FRAMES = "61"
FINE = "dx0.5mm"


def _json(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not produced in this checkout")
    return json.loads(path.read_text())


def _fine_art() -> dict:
    path = ENS_DIR / f"ensemble_{FINE}_c0v20.json"
    if not path.exists():
        pytest.skip("dx = 0.5 mm ensemble not produced in this checkout")
    return json.loads(path.read_text())


def _ladder():
    return _json(LADDER_JSON)


def _row(tag: str) -> dict:
    for r in _ladder()["grids"]:
        if r["tag"] == tag:
            return r
    pytest.skip(f"{tag} not in the ladder artefact")


# --------------------------------------------------------------------------- #
# 1. the new rung exists and is the run that was asked for
# --------------------------------------------------------------------------- #
class TestFineRungIsBound:
    def test_fine_rung_is_in_the_ladder_and_missing_grids_is_empty(self):
        lad = _ladder()
        assert FINE in {r["tag"] for r in lad["grids"]}
        assert lad["missing_grids"] == []

    def test_fine_rung_has_the_declared_case_and_frame_count(self):
        art = _fine_art()
        assert art["config"]["beta_deg"] == 30.0
        assert art["config"]["velocity"] == 1.0
        assert FRAMES in art["by_frames"]
        assert art["by_frames"][FRAMES]["n_frames"] == 61

    def test_fine_rung_records_its_cost_so_the_run_is_auditable(self):
        art = _fine_art()
        assert art["steps"] > 0
        assert art["wall_seconds"] > 0
        # twice the steps of dx = 1 mm: halving dx doubles the acoustic CFL count
        coarse = json.loads((ENS_DIR / "ensemble_dx1mm_c0v20.json").read_text())
        assert art["steps"] > coarse["steps"]


# --------------------------------------------------------------------------- #
# 2. the uncertainty rule is the declared one, recomputed from the raw artefact
# --------------------------------------------------------------------------- #
class TestUncertaintyRule:
    def test_ladder_U_is_max_spread_jackknife(self):
        for row in _ladder()["grids"]:
            art = json.loads((ENS_DIR / f"ensemble_{row['tag']}_c0v20.json").read_text())
            frames = art["by_frames"][FRAMES]
            expect = max(frames["resolution_band"]["spread_rel"], frames["jackknife_rel"])
            assert row["uncertainty"] == pytest.approx(expect, rel=1e-12), row["tag"]
            assert row["judgeable"] is (expect <= GATE), row["tag"]

    def test_degenerate_spread_is_not_evidence_at_the_fine_rung(self):
        """One surviving bin count => spread_rel == 0 by construction, not by quietness.

        If the ladder had used the bin-count spread alone it would have called
        dx = 0.5 mm judgeable (spread = 0) and turned a 21 % frame-half drift
        into a PASS.  The jackknife is the statistic that carries the answer.
        """
        art = _fine_art()
        rb = art["by_frames"][FRAMES]["resolution_band"]
        if len(rb["n_bins_used"]) != 1:
            pytest.skip("more than one bin count survived the gate; guard not exercised")
        assert rb["spread_rel"] == 0.0
        row = _row(FINE)
        assert row["obs_spread_rel"] == 0.0
        assert row["uncertainty"] == pytest.approx(row["obs_jackknife_rel"], rel=1e-12)
        assert row["uncertainty"] > GATE
        assert row["judgeable"] is False

    def test_band_admits_a_single_candidate_because_the_bin_grid_is_too_coarse(self):
        """Documents the instrument defect rather than hiding it behind a 0.0."""
        art = _fine_art()
        rb = art["by_frames"][FRAMES]["resolution_band"]
        lo, hi = rb["n_bins_range"]
        # the candidate bin counts are declared by the estimator design, not by the run
        grid = json.loads((ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_bias"
                           / "bias_report.json").read_text())["design"]["bin_grid"]
        inside = [b for b in grid if lo <= b <= hi]
        assert rb["n_bins_used"] == inside


# --------------------------------------------------------------------------- #
# 3. the verdict, recomputed, and explained by the artefacts
# --------------------------------------------------------------------------- #
class TestVerdictIsExplained:
    def test_recorded_verdict_equals_the_recomputed_verdict(self):
        lad = _ladder()
        rows = lad["grids"]
        good = [r for r in rows if r["judgeable"]]
        recorded = lad["verdict"]["verdict"]
        if len(good) < 2:
            assert recorded == "NOT_DECIDABLE"
            assert sorted(lad["verdict"]["grids_judgeable"]) == sorted(r["tag"] for r in good)
        else:
            f, f2 = good[-1], good[-2]
            same_sign = (f["residual"] > 0) == (f2["residual"] > 0)
            if not same_sign:
                assert recorded == "NOT_DECIDABLE"
            elif abs(f["residual"]) <= GATE:
                assert recorded == "PASS"
            else:
                assert recorded == ("FAIL_above" if f["residual"] > 0 else "FAIL_below")

    def test_non_pass_verdict_carries_a_reason_bound_to_artefacts(self):
        lad = _ladder()
        if lad["verdict"]["verdict"] == "PASS":
            pytest.skip("case passed; nothing to explain")
        assert isinstance(lad["verdict"].get("reason"), str) and lad["verdict"]["reason"]


# --------------------------------------------------------------------------- #
# 4. the refinement route is falsified, and stays falsified
# --------------------------------------------------------------------------- #
class TestRefinementRouteFalsified:
    def test_uncertainty_is_not_monotone_in_dx(self):
        """m17548 route (b) assumed U falls with dx.  It does not."""
        rows = sorted(_ladder()["grids"], key=lambda r: r["dx"])
        U = [r["uncertainty"] for r in rows]
        assert len(U) >= 4
        assert not all(a > b for a, b in zip(U, U[1:])), U
        fine, coarse = rows[0], rows[1]                     # rows ascend in dx
        assert fine["uncertainty"] > coarse["uncertainty"], (
            "dx = 0.5 mm was quieter than dx = 1 mm; re-open the refinement route")

    def test_consecutive_rungs_do_not_agree_in_sign(self):
        """A sign flip between neighbouring rungs forbids a PASS even if U passed."""
        rows = sorted(_ladder()["grids"], key=lambda r: r["dx"])
        signs = [1 if r["residual"] > 0 else -1 for r in rows]
        assert any(a != b for a, b in zip(signs, signs[1:])), signs


# --------------------------------------------------------------------------- #
# 5. the published table row that this rung was meant to settle
# --------------------------------------------------------------------------- #
class TestTableRowIsSettled:
    def test_dx_0_5mm_row_is_no_longer_unreproducible(self):
        err = _json(ERRATUM_JSON)
        row = [c for c in err["comparisons"] if c["dx"] == 0.0005]
        assert row, "the dx = 0.5 mm table row disappeared"
        row = row[0]
        assert row["match"] in ("reproduced", "unreproducible")
        obs = _row(FINE)["obs_peak"]
        gap = obs / row["table_cp"] - 1.0
        assert row["match"] == ("reproduced" if abs(gap) <= GATE else "unreproducible"), gap

    def test_verdict_is_still_not_usable_as_acceptance(self):
        """The other table rows stay unreproducible, so Table 3.2 is still not evidence."""
        err = _json(ERRATUM_JSON)
        unrepro = [c["dx"] for c in err["comparisons"] if c["match"] == "unreproducible"]
        assert err["summary"]["unreproducible"] == unrepro
        assert err["summary"]["verdict"]["usable_as_acceptance"] is (not unrepro)

    def test_zhao_faltinsen_is_still_withheld(self):
        refs = json.loads((ROOT / "benchmarks" / "case_sph_wedge_entry_2d" / "refs.json")
                          .read_text())
        zf = [r for r in refs["references"] if r["id"] == "zhao_faltinsen_1993"][0]
        assert zf["status"] == "WITHHELD"
        assert zf["table_or_figure_value_beta_30"] is None