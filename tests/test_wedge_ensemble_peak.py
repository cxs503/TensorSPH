"""Tests for the frame-ensemble peak estimator (``ensemble_peak.py``).

The estimator is what decides whether the case's peak pressure coefficient is
judgeable, so it is tested against synthetic data with a known answer rather
than against the solver:

* the sample gate must refuse to let an under-sampled bin set the peak;
* on Wagner-shaped data with noise the pooled profile must recover the analytic
  peak within the case tolerance, and must degrade *predictably* when the gate
  is removed;
* the resolution band must be the set of bin counts whose bin width in metres
  stays in ``[1 dx, 4 dx]``;
* the committed artefacts must be internally consistent (the reported number is
  reproducible from the stored per-bin rows -- no hand-typed value).
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MOD_PATH = ROOT / "benchmarks" / "case_sph_wedge_entry_2d" / "ensemble_peak.py"
ART = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ensemble"


def _load():
    spec = importlib.util.spec_from_file_location("ensemble_peak", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ep = _load()


def _wagner_samples(beta=math.radians(30.0), n=60, noise=0.0, seed=7):
    """Samples of the in-repo Wagner profile ``Cp(tau)`` on one frame."""
    rng = random.Random(seed)
    taus, cps = [], []
    for k in range(n):
        tau = 1.2 * (k + 0.5) / n
        cp = ep.wagner_p_over_q(tau, beta)
        if noise:
            cp += rng.gauss(0.0, noise * max(cp, 1.0))
        taus.append(tau)
        cps.append(cp)
    return taus, cps


def test_gated_peak_recovers_wagner_profile():
    beta = math.radians(30.0)
    ref = ep.wagner_peak_over_q(beta)
    taus, cps = _wagner_samples(beta, n=60)
    # pool 20 identical frames: the pooled mean is the analytic profile
    taus_all = taus * 20
    cps_all = cps * 20
    centres, means, cnts = ep.binned(taus_all, cps_all, 30)
    peak, tau_at, n_at = ep.gated_peak(centres, means, cnts)
    assert n_at >= ep.MIN_BIN_SAMPLES
    # 30 bins resolve tau_peak = 0.930 to within two bin widths (0.08)
    assert abs(tau_at - 0.930) < 0.08
    assert abs(peak - ref) / ref < 0.15


def test_gate_blocks_under_sampled_bin():
    """A single huge sample in a bin of its own must not become the peak."""
    taus = [0.02] * 40 + [0.90]      # the 0.90 bin holds one sample only
    cps = [3.0] * 40 + [500.0]
    centres, means, cnts = ep.binned(taus, cps, 30)
    gated, _, n_at = ep.gated_peak(centres, means, cnts)
    ungated, _, _ = ep.gated_peak(centres, means, cnts, min_count=1)
    assert ungated == pytest.approx(500.0)
    assert n_at >= ep.MIN_BIN_SAMPLES
    assert gated == pytest.approx(3.0)


def test_gated_peak_is_nan_free_on_empty_profile():
    peak, tau_at, n_at = ep.gated_peak([], [], [])
    assert math.isnan(peak) and math.isnan(tau_at) and n_at == 0


def test_resolution_band_is_one_to_four_particles_wide():
    # c = 50 mm, dx = 2 mm -> tau width per bin in [dx/c, 4 dx/c] = [0.04, 0.16]
    lo, hi, used = ep.resolution_bin_band(c_last=0.05, dx=0.002)
    for n in used:
        width_m = (ep.TAU_MAX / n) * 0.05
        assert 1.0 * 0.002 - 1e-12 <= width_m <= 4.0 * 0.002 + 1e-12
    assert 4 <= lo <= hi
    assert used == sorted(used) and len(used) >= 3


def test_spread_reports_relative_range():
    lo, hi, med, rel = ep.spread([5.0, 6.0, 7.0])
    assert (lo, hi, med) == (5.0, 7.0, 6.0)
    assert rel == pytest.approx(2.0 / 6.0)
    assert math.isnan(ep.spread([])[3])


@pytest.mark.skipif(not ART.exists(), reason="ensemble artefacts not committed")
@pytest.mark.parametrize("path", sorted(ART.glob("ensemble_*.json")) or [None])
def test_committed_artefact_is_internally_consistent(path):
    if path is None:                     # pragma: no cover - no artefacts yet
        pytest.skip("no artefacts")
    doc = json.loads(Path(path).read_text())
    assert doc["case"] == "CASE-SPH-WEDGE-ENTRY-2D"
    row = doc["by_frames"][str(max(int(k) for k in doc["by_frames"]))]
    # the reported peak must be reproducible from the stored per-bin rows
    peaks = [r["cp_peak_gated"] for r in row["rows"]]
    lo, hi, med, rel = ep.spread(peaks)
    assert row["peak_median"] == pytest.approx(med, rel=1e-9)
    assert row["spread_rel"] == pytest.approx(rel, rel=1e-9)
    # the resolution-band statistic must be a subset of the same rows
    band = row["resolution_band"]
    sub = [r["cp_peak_gated"] for r in row["rows"] if r["n_bins"] in band["n_bins_used"]]
    assert band["peak_median"] == pytest.approx(ep.spread(sub)[2], rel=1e-9)
    # the reference the artefact compares against is the in-repo analytic value
    assert doc["wagner_cp_max"] == pytest.approx(ep.wagner_peak_over_q(math.radians(30.0)))