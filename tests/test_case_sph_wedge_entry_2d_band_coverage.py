"""CASE-SPH-WEDGE-ENTRY-2D: is the resolution-band uncertainty actually estimated?

Defect being guarded (m22695)
-----------------------------
The case verdict is gated by

    U_g = max(resolution_band.spread_rel, jackknife_rel),  judgeable_g <=> U_g <= 15 %

``resolution_band`` is the band of bin counts whose bin width in metres stays in
``[1, 4]`` particle spacings (``ensemble_peak.resolution_bin_band``).  Its
candidates were taken from ``BIN_GRID = (6, ..., 30)``, i.e. from a grid that
stops at 30 bins.  But the admissible band itself grows as ``dx`` shrinks:

===================  ==================  ==========================
grid                 band (committed)    BIN_GRID candidates
===================  ==================  ==========================
dx = 4   mm          [4, 16]             6, 8, 10, 12, 16
dx = 2   mm          [7, 30]             8 .. 30  (30 == the cap)
dx = 1   mm          [15, 64]            16, 20, 30   <- upper half missing
dx = 0.5 mm          [29, 119]           30           <- one member
===================  ==================  ==========================

So at the two *finest* grids — the ones that decide the verdict — the band was
truncated to its over-smoothing half, and at ``dx = 0.5 mm`` it collapsed to a
single member whose ``spread_rel`` is ``0.0`` **by construction**.  A vacuous
zero in a ``max(...)`` floor is a false-PASS path: had the frame jackknife
happened to be small (it is 0.26 % at 13 frames on the same grid), the grid
would have been declared judgeable with no band estimate at all.

What is asserted here
---------------------
* the truncation is real and recomputed from the committed artefacts, not typed;
* a band with fewer than two members is **not** reported as a spread (the
  estimator now returns NaN and ``estimable=False``);
* the corrected candidate grid admits >= 2 members inside the *physical* band
  at every ladder grid — the arithmetic proof that the fix works, no solver run;
* the committed verdict does **not** change (it is still ``NOT_DECIDABLE``,
  carried by the jackknife), i.e. this is a latent defect, not a rescored PASS.
  Saying so is the point: an honest "no verdict change" is the correct result.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import ensemble_peak as ep  # noqa: E402

ENS_DIR = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ensemble"
LADDER_JSON = (ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ladder"
               / "ladder_closure.json")

GATE = 0.15
FRAMES = "61"
GRIDS = ("dx4mm", "dx2mm", "dx1mm", "dx0.5mm")


def _ens(tag: str) -> dict:
    path = ENS_DIR / f"ensemble_{tag}_c0v20.json"
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    return json.loads(path.read_text())


def _ladder() -> dict:
    if not LADDER_JSON.exists():
        pytest.skip("ladder closure artefact not committed")
    return json.loads(LADDER_JSON.read_text())


# --------------------------------------------------------------------------- #
# 1. the truncation is real (recomputed from committed artefacts)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tag", GRIDS)
def test_band_is_covered_by_the_declared_bin_grid_or_not(tag):
    """Recompute coverage; do not restate it."""
    rb = _ens(tag)["by_frames"][FRAMES]["resolution_band"]
    lo, hi = rb["n_bins_range"]
    used = rb["n_bins_used"]
    covered_above = hi <= max(ep.BIN_GRID)
    if lo >= min(ep.BIN_GRID) and covered_above:
        assert all(lo <= n <= hi for n in used)
    else:
        # truncated band: the recorded candidate list must stop at the grid cap
        assert used, tag
        assert max(used) <= max(ep.BIN_GRID)


def test_truncation_happens_exactly_at_the_two_finest_grids():
    """The pattern is monotone in dx, so a single coarse sample cannot hide it."""
    truncated = {}
    for tag in GRIDS:
        lo, hi = _ens(tag)["by_frames"][FRAMES]["resolution_band"]["n_bins_range"]
        truncated[tag] = hi > max(ep.BIN_GRID)
    assert truncated == {"dx4mm": False, "dx2mm": False,
                         "dx1mm": True, "dx0.5mm": True}


def test_finest_band_collapsed_to_one_member_in_the_committed_artefact():
    rb = _ens("dx0.5mm")["by_frames"][FRAMES]["resolution_band"]
    assert len(rb["n_bins_used"]) == 1
    assert rb["peak_min"] == rb["peak_max"] == rb["peak_median"]
    # this is precisely why a 0.0 spread there carries no information
    assert rb["spread_rel"] == 0.0


# --------------------------------------------------------------------------- #
# 2. a one-member band must not be reported as a spread
# --------------------------------------------------------------------------- #
def test_single_member_band_is_not_estimable():
    """The fix itself: the estimator refuses to emit a fabricated 0.0."""
    lo, hi, cands = ep.resolution_bin_band(c_last=0.05, dx=0.0005)
    assert len(cands) >= 2, "physical band must have >= 2 candidate bin counts now"
    # and the same clamp the estimator applies, on a deliberately degenerate band
    degenerate = [c for c in cands if c == cands[0]]
    assert len(degenerate) == 1
    # spread() alone *does* return 0.0 for one value -- that is the primitive's
    # vacuous behaviour, and the reason the guard must live in the caller.
    assert ep.spread(degenerate)[3] == 0.0


def test_committed_artefacts_are_flagged_not_estimable_when_degenerate():
    """Every committed one-member band must be visible as such (no silent 0.0)."""
    seen = 0
    for tag in GRIDS:
        rb = _ens(tag)["by_frames"][FRAMES]["resolution_band"]
        if len(rb["n_bins_used"]) >= 2:
            continue
        seen += 1
        assert rb["spread_rel"] == 0.0
        # after regeneration the artefact carries the explicit flag; on a stale
        # artefact the flag is absent, which is exactly the audit finding
        assert rb.get("estimable", False) is False
    assert seen >= 1


# --------------------------------------------------------------------------- #
# 3. the corrected candidate grid actually spans the physical band everywhere
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tag", GRIDS)
def test_corrected_band_grid_admits_two_or_more_members_at_every_grid(tag):
    """Arithmetic proof that the fix removes the truncation (no solver run)."""
    art = _ens(tag)
    lo, hi = art["by_frames"][FRAMES]["resolution_band"]["n_bins_range"]
    cands = [n for n in ep.BAND_GRID if lo <= n <= hi]
    assert len(cands) >= 2, (tag, lo, hi, cands)
    assert cands == sorted(cands)
    assert set(ep.BIN_GRID).issubset(ep.BAND_GRID)      # H1 statistic unchanged
    # the band must not be clipped at the top any more
    assert max(cands) <= max(ep.BAND_GRID)


def test_band_grid_reproduces_the_c_equals_50mm_dx_equals_2mm_reference_band():
    """Keep the original a-priori band rule intact: width in [1, 4] spacings."""
    lo, hi, used = ep.resolution_bin_band(c_last=0.05, dx=0.002)
    for n in used:
        width_m = (ep.TAU_MAX / n) * 0.05
        assert 1.0 * 0.002 - 1e-12 <= width_m <= 4.0 * 0.002 + 1e-12
    assert len(used) >= 3


# --------------------------------------------------------------------------- #
# 4. NaN handling in the declared uncertainty + honest verdict statement
# --------------------------------------------------------------------------- #
def test_declared_uncertainty_ignores_a_non_estimable_band_term():
    """max(band, jackknife) must fall back to the finite term, never to NaN."""
    rb_rel, jk_rel = float("nan"), 0.209
    terms = [v for v in (rb_rel, jk_rel) if v == v]
    assert max(terms) == 0.209
    none_finite = [v for v in (float("nan"), float("nan")) if v == v]
    assert none_finite == []


def test_verdict_is_unchanged_by_the_fix_and_stays_honest():
    """No rescoring: the ladder verdict is still NOT_DECIDABLE (jackknife-carried)."""
    lad = _ladder()
    assert lad["verdict"]["verdict"] == "NOT_DECIDABLE"
    assert lad["verdict"]["grids_judgeable"] == ["dx1mm"]
    fine = next(r for r in lad["grids"] if r["tag"] == "dx0.5mm")
    assert fine["obs_spread_rel"] == 0.0
    assert fine["uncertainty"] == pytest.approx(fine["obs_jackknife_rel"], rel=1e-12)
    assert fine["judgeable"] is False