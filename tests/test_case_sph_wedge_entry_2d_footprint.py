"""Tests for the footprint-average peak-pressure estimator (m21677).

Two kinds of test:

* **operator correctness** -- the sliding-window average is checked against a
  field whose window average is known in closed form, so the estimator cannot be
  wrong in a way the case verdict would inherit;
* **the defect and the state of the case** -- the committed ``BIN_GRID`` cannot
  reach a matched bandwidth on the two finest grids (an assertable property of
  the committed artefacts, not a restatement of a claim), and the graded numbers
  are *recomputed* from artefacts rather than re-stated.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"
ENS = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ensemble"
FP = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_footprint"
import sys
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT / "src"))

import footprint_peak as fpm  # noqa: E402
import footprint_ledger as fl  # noqa: E402


def _profile_linear(a: float, b: float, n: int = 2000, tau_max: float = 1.0):
    """Dense samples of ``cp = a + b*tau`` (window mean = value at centre)."""
    taus, cps = [], []
    for k in range(n):
        tau = tau_max * (k + 0.5) / n
        taus.append(tau)
        cps.append(a + b * tau)
    return taus, cps


def test_window_average_of_linear_profile_is_value_at_centre():
    taus, cps = _profile_linear(1.0, 3.0)
    rows = fpm.footprint_peaks(taus, cps, dx=1e-3, c_end=0.05,
                               footprints_cells=(4.0,))
    row = rows[0]
    # the maximum over tau <= 1 is the window pushed as far out as allowed, so
    # its centre sits at tau = 1 (one fine-grid step of slack); the mean of a
    # linear function over a window is its value at the window centre
    assert row["tau_at_peak"] == pytest.approx(1.0, abs=2e-2)
    assert row["cp_peak"] == pytest.approx(1.0 + 3.0 * row["tau_at_peak"], rel=2e-2)


def test_window_is_average_not_max():
    """A single spike must be divided by the window width, not reported."""
    taus, cps = [], []
    for k in range(400):
        tau = 1.2 * (k + 0.5) / 400
        taus.append(tau)
        cps.append(1.0 + (100.0 if abs(tau - 0.5) < 1e-3 else 0.0))
    rows = fpm.footprint_peaks(taus, cps, dx=2e-3, c_end=0.05,
                               footprints_cells=(8.0,))
    peak = rows[0]["cp_peak"]
    assert peak < 10.0, "footprint average must not return the raw spike"


def test_footprint_grows_with_cells():
    taus, cps = _profile_linear(1.0, 3.0)
    rows = fpm.footprint_peaks(taus, cps, dx=1e-3, c_end=0.05,
                               footprints_cells=(2.0, 8.0, 16.0))
    widths = [r["delta_tau"] for r in rows]
    assert widths == sorted(widths) and widths[0] < widths[-1]


@pytest.mark.skipif(not ENS.exists(), reason="committed ensemble artefacts absent")
def test_committed_bin_grid_cannot_reach_matched_bandwidth():
    """The band the committed chain used stops at 30 bins; the two finest grids
    need 60 and 120 bins for one bin per surface cell.  That is the defect."""
    from ensemble_peak import BIN_GRID, TAU_MAX

    for tag, need in (("dx1mm", 60), ("dx0.5mm", 120)):
        art = json.loads((ENS / f"ensemble_{tag}_c0v20.json").read_text())
        c_end = art["C"] * art["config"]["velocity"] * art["t_end"]
        n_cells = TAU_MAX * c_end / art["config"]["dx"]
        assert n_cells == pytest.approx(need, rel=0.1)
        assert max(BIN_GRID) < n_cells, f"{tag}: band is not truncated"
        used = art["by_frames"]["61"]["resolution_band"]["n_bins_used"]
        assert max(used) <= max(BIN_GRID)
        # the finest grid's band degenerates to a single setting, so its
        # "spread" is identically zero -- a false zero, not a small uncertainty
        if tag == "dx0.5mm":
            assert len(used) == 1
            assert art["by_frames"]["61"]["resolution_band"]["spread_rel"] == 0.0


@pytest.mark.skipif(not (FP / "footprint_ledger.json").exists(),
                    reason="footprint ladder artefacts absent")
def test_footprint_ledger_is_recomputable_and_not_overstated():
    sys.path.insert(0, str(BENCH))
    import footprint_ledger as fl

    rep = json.loads((FP / "footprint_ledger.json").read_text())
    recomputed = fl.verdict(rep["grids"])
    assert recomputed["verdict"] == rep["verdict"]["verdict"]
    for r in rep["grids"]:
        # residual must be recomputable from the two numbers it is defined from
        assert r["residual"] == pytest.approx(r["obs_cp"] / r["synth_cp"] - 1.0, rel=1e-12)
        # judgeability must follow the declared rule, not the verdict's wishes
        expected = (r["uncertainty"] <= fl.GATE)
        assert r["judgeable"] == bool(expected)
    # a PASS would require two judgeable grids with the same sign and |r| <= gate
    if rep["verdict"]["verdict"] == "PASS":
        judged = [r for r in rep["grids"] if r["judgeable"]]
        assert len(judged) >= 2
        assert all(abs(r["residual"]) <= fl.GATE for r in judged)


@pytest.mark.skipif(not (FP / "footprint_ledger.json").exists(),
                    reason="footprint ladder artefacts absent")
def test_designated_readout_is_recomputable_and_separate_from_the_verdict():
    """Secondary metric-consistent read-out (m21953) must be derived, not asserted."""
    rep = json.loads((FP / "footprint_ledger.json").read_text())
    d = rep["designated_readout"]
    for tag, r in d["residuals"].items():
        row = next(g for g in rep["grids"] if g["tag"] == tag)
        assert row["judgeable"] is True            # only judgeable grids are usable
        assert r == pytest.approx(row["obs_cp"] / row["designated_footprint_cp"] - 1.0, rel=1e-12)
        assert abs(r) <= fl.GATE or d["inside_gate"] is False
    # the primary verdict is unchanged by the secondary read-out: recompute it
    import footprint_ledger as fl2
    assert fl2.verdict(rep["grids"])["verdict"] == rep["verdict"]["verdict"]
    # and it must never claim a bracket it does not have
    signs = {(x < 0.0) for x in d["residuals"].values()}
    assert d["brackets_target"] == (len(signs) > 1)
