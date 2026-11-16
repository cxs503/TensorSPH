"""CASE-SPH-WEDGE-ENTRY-2D, m23280: the resolution band re-measured on BAND_GRID.

Three things are guarded here, none of which the pre-existing suite could see:

1. **the committed reading is a function of the committed samples** -- the offline
   re-binning tool must reproduce every committed band peak from the frame dump
   to machine precision (0.0 deviation), and must *not* be a lookup of the stored
   numbers (perturbing the samples must move them);
2. **the band is two-sided after the fix** -- every re-measured grid has at least
   two candidate bin counts inside its physical band, so no spread is fabricated
   from a one-member band (the m22695 defect);
3. **the re-measurement reproduced the unchanged grid bit for bit** -- `dx=4 mm`
   has the same candidate list before and after, so its reading must be
   identical: that is the control proving the shift at `dx=2 mm` is the
   instrument, not a different run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import ensemble_peak as ep        # noqa: E402
import rebuild_from_frames as rb  # noqa: E402

OLD = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ensemble"
NEW = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ensemble_bandfix"
REBIN = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_rebin"
LADDER = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ladder_bandfix" / "ladder_closure.json"
DOC = ROOT / "docs" / "case_sph_wedge_entry_2d_band_remeasure_m23280.md"
FRAMES = "61"
GATE = 0.15
REMEASURED = ("dx4mm", "dx2mm")


def _art(base: Path, tag: str) -> dict:
    p = base / f"ensemble_{tag}_c0v20.json"
    if not p.exists():
        pytest.skip(f"{p} not committed")
    return json.loads(p.read_text())


def _frames(tag: str) -> dict:
    p = NEW / f"frames_{tag}_c0v20.json"
    if not p.exists():
        pytest.skip(f"{p} not committed")
    return json.loads(p.read_text())


# --------------------------------------------------------------------------- #
# 1. the artefact is re-derivable from the samples (and only from them)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tag", REMEASURED)
def test_offline_rebin_reproduces_the_committed_band_peaks(tag):
    rep = rb.rebuild(_art(NEW, tag), _frames(tag))
    assert rep["frames_in_dump"] == 61
    assert rep["worst_band_peak_rel_dev"] == pytest.approx(0.0, abs=1e-12)
    for row in rep["by_frames"]:
        assert row["band_members"] >= 2
        assert row["max_band_peak_rel_dev"] == pytest.approx(0.0, abs=1e-12)
        # the full BIN_GRID median (historical H1 statistic) too
        assert row["grid_peak_median_rel_dev"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("tag", REMEASURED)
def test_offline_rebin_is_not_a_lookup(tag):
    """Negative control: move the samples, the reading must move with them."""
    dump = _frames(tag)
    shifted = json.loads(json.dumps(dump))
    for fr in shifted["frames"]:
        fr["cp"] = [1.5 * c for c in fr["cp"]]
    base = rb.rebuild(_art(NEW, tag), dump)["by_frames"][-1]
    moved = rb.rebuild(_art(NEW, tag), shifted)["by_frames"][-1]
    assert moved["peak_median_rebuilt"] / base["peak_median_rebuilt"] - 1.0 == pytest.approx(
        0.5, rel=1e-9)


# --------------------------------------------------------------------------- #
# 2. the fixed estimator never emits a vacuous spread
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tag", ("dx4mm", "dx2mm"))
def test_remeasured_band_has_two_sided_coverage(tag):
    art = _art(NEW, tag)
    band = art["by_frames"][FRAMES]["resolution_band"]
    lo, hi = band["n_bins_range"]
    used = band["n_bins_used"]
    assert len(used) >= 2, "band must admit >= 2 candidates (m22695 fix)"
    assert all(lo <= n <= hi for n in used)
    assert band["spread_rel"] == band["spread_rel"]      # finite, not NaN
    assert band["spread_rel"] > 0.0, "a multi-member band cannot have zero spread"
    assert band["estimable"] is True
    # every candidate comes from the declared grid
    assert set(used).issubset(set(ep.BAND_GRID))
    # and the band is no longer truncated by the grid cap
    assert band["band_truncated"] is False


def test_the_two_finest_grids_are_still_unremeasured():
    """Honest gate: the decisive grids are queued, so no verdict may be claimed."""
    for tag in ("dx1mm", "dx0.5mm"):
        assert not (NEW / f"ensemble_{tag}_c0v20.json").exists(), (
            f"{tag} was re-measured -- update docs/case_sph_wedge_entry_2d_band_remeasure_m23280.md "
            "and the case verdict instead of widening this assertion")
    lad = json.loads(LADDER.read_text()) if LADDER.exists() else None
    if lad is not None:
        assert lad["missing_grids"] == ["dx1mm", "dx0.5mm"]
        assert lad["verdict"]["verdict"] == "NOT_DECIDABLE"
        assert lad["verdict"]["grids_judgeable"] == []


# --------------------------------------------------------------------------- #
# 3. the control: an unchanged candidate list must reproduce the old reading
# --------------------------------------------------------------------------- #
def test_dx4mm_is_bit_identical_to_the_pre_fix_reading():
    old = _art(OLD, "dx4mm")["by_frames"][FRAMES]["resolution_band"]
    new = _art(NEW, "dx4mm")["by_frames"][FRAMES]["resolution_band"]
    assert old["n_bins_used"] == new["n_bins_used"] == [6, 8, 10, 12, 16]
    assert new["peak_median"] == pytest.approx(old["peak_median"], rel=0, abs=1e-12)
    assert new["spread_rel"] == pytest.approx(old["spread_rel"], rel=0, abs=1e-12)


def test_dx2mm_shift_is_the_instrument_not_a_new_run():
    """Only bin 24 was added; the reading moves by a few percent, not by a grid."""
    old = _art(OLD, "dx2mm")["by_frames"][FRAMES]["resolution_band"]
    new = _art(NEW, "dx2mm")["by_frames"][FRAMES]["resolution_band"]
    assert set(old["n_bins_used"]).issubset(set(new["n_bins_used"]))
    assert sorted(set(new["n_bins_used"]) - set(old["n_bins_used"])) == [24]
    shift = new["peak_median"] / old["peak_median"] - 1.0
    assert 0.0 < abs(shift) < 0.10, shift
    # and the grid is still not judgeable: the jackknife term dominates
    u = _art(NEW, "dx2mm")["by_frames"][FRAMES]["uncertainty_declared"]
    assert u > GATE
    assert max(new["spread_rel"], old["spread_rel"]) < u


# --------------------------------------------------------------------------- #
# 4. the document reports the artefacts, it does not restate the author
# --------------------------------------------------------------------------- #
def test_doc_carries_the_remeasured_numbers():
    if not DOC.exists():
        pytest.skip("doc not rendered")
    text = DOC.read_text()
    for tag in REMEASURED:
        peak = _art(NEW, tag)["by_frames"][FRAMES]["resolution_band"]["peak_median"]
        assert f"{peak:.3f}" in text, (tag, peak)
    assert "未重测" in text                      # the queued grids are flagged
    assert "PARTIAL_SINGLE_GRID" in text or "NOT_DECIDABLE" in text