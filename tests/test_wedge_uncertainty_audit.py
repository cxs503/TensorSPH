"""Tests for the frame-bootstrap audit (m22416).

Two things have to be true for the audit to mean anything:

1. **The estimator is untouched.**  The moving-block resampler must sit *around*
   the committed estimator, not replace it: with ``mode="iid"`` and the
   committed seed the audit has to reproduce ``ensemble_peak``'s resample
   sequence exactly.  Otherwise "i.i.d. U" and "block U" would not be the same
   statistic under two dependence assumptions, and the comparison would be
   meaningless.
2. **The machinery recovers a known answer.**  For AR(1) frames with lag-1
   correlation ``rho`` the standard deviation of the sample mean is
   ``sigma / sqrt(n) * sqrt((1+rho)/(1-rho))``.  The i.i.d. bootstrap must
   understate that by ``sqrt((1-rho)/(1+rho))`` and the moving-block bootstrap
   must recover it.  Without this ground truth "the block interval is wider" is
   just an observation, not a correction.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"
sys.path.insert(0, str(BENCH))

import uncertainty_audit as ua  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. the estimator is unchanged
# --------------------------------------------------------------------------- #
def test_iid_resampler_reproduces_committed_sequence():
    """Same seed, same scheme -> byte-identical index sequence as the ledger chain."""
    n = 61
    committed = random.Random(20261113)
    expect = [committed.randrange(n) for _ in range(n)]

    ours = random.Random(20261113)
    assert ua.iid_indices(n, ours) == expect


def test_iid_bootstrap_matches_ensemble_peak_loop():
    """`bootstrap(mode='iid')` == the resampling loop in ensemble_peak.py."""
    rng_seed, n, B = 20261113, 9, 25
    values = [float(i * i % 7) for i in range(n)]

    # verbatim copy of ensemble_peak's loop (means instead of footprint peaks,
    # so the resample *sequence* is what is being compared)
    rng = random.Random(rng_seed)
    ref = []
    for _ in range(B):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        ref.append(sum(sample) / n)

    got = ua.bootstrap(values, lambda s: sum(s) / len(s), resamples=B,
                       seed=rng_seed, mode="iid", block=1)
    assert got == pytest.approx(ref, rel=0)


def test_block_length_one_is_the_iid_scheme():
    a = ua.moving_block_indices(20, random.Random(7), 1)
    b = ua.iid_indices(20, random.Random(7))
    assert a == b


def test_moving_block_indices_are_consecutive_circular_blocks():
    n, L = 10, 3
    idx = ua.moving_block_indices(n, random.Random(3), L)
    assert len(idx) == n
    for k in range(0, n - L + 1, L):
        blk = idx[k:k + L]
        assert all((blk[j + 1] - blk[j]) % n == 1 for j in range(len(blk) - 1))


# --------------------------------------------------------------------------- #
# 2. ground truth: AR(1) frames
# --------------------------------------------------------------------------- #
def ar1(n: int, rho: float, sigma: float = 1.0, seed: int = 11) -> list[float]:
    rng = random.Random(seed)
    innov = sigma * math.sqrt(1.0 - rho ** 2)
    x = [rng.gauss(0.0, sigma)]
    for _ in range(n - 1):
        x.append(rho * x[-1] + rng.gauss(0.0, innov))
    return x


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.8])
def test_iid_bootstrap_understates_ar1_frames_by_the_analytic_factor(rho):
    """The falsifiable claim: i.i.d. resampling of AR(1) frames loses
    ``sqrt((1-rho)/(1+rho))`` of the standard deviation of the mean.

    This is the whole reason the audit exists -- it is a *predictable*
    mis-calibration, not a vague worry.  For ``rho = 0`` the committed scheme is
    correct (ratio 1), which is the negative control: the test can fail.
    """
    n, B = 120, 400
    x = ar1(n, rho, sigma=1.0, seed=2026)
    mean = lambda s: sum(s) / len(s)  # noqa: E731

    true_sd = math.sqrt((1.0 + rho) / (1.0 - rho)) / math.sqrt(n)

    def sd(samples):
        m = sum(samples) / len(samples)
        return math.sqrt(sum((v - m) ** 2 for v in samples) / (len(samples) - 1))

    sd_iid = sd(ua.bootstrap(x, mean, resamples=B, seed=1, mode="iid"))
    predicted = true_sd * math.sqrt((1.0 - rho) / (1.0 + rho))
    assert sd_iid == pytest.approx(predicted, rel=0.2)
    if rho > 0.0:
        assert sd_iid < 0.75 * true_sd


def test_block_resampling_recovers_part_of_the_deficit():
    """Block resampling must move the interval toward the truth, not away.

    It is *not* an exact correction: the moving-block bootstrap of the mean is
    itself biased low when ``L/n`` is not small (the resampled blocks are drawn
    from a finite realisation, so long blocks under-represent the population
    variance).  The report therefore quotes the block interval as a **lower
    bound** on the correction -- this test pins that behaviour down instead of
    pretending the correction is exact.
    """
    n, B, rho = 120, 400, 0.8
    x = ar1(n, rho, sigma=1.0, seed=2026)
    mean = lambda s: sum(s) / len(s)  # noqa: E731
    true_sd = math.sqrt((1.0 + rho) / (1.0 - rho)) / math.sqrt(n)

    sd_iid = statistics_stdev(ua.bootstrap(x, mean, resamples=B, seed=1, mode="iid"))
    sd_blk = statistics_stdev(ua.bootstrap(x, mean, resamples=B, seed=1,
                                           mode="block", block=9))
    assert sd_blk > 1.5 * sd_iid                       # deficit visibly reduced
    assert sd_blk < true_sd                            # still a lower bound
    assert abs(sd_blk - true_sd) < abs(sd_iid - true_sd)


def statistics_stdev(values):
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def test_n_effective_shrinks_with_correlation():
    assert ua.n_effective(61, 0.0) == pytest.approx(61.0)
    assert ua.n_effective(61, 0.5) == pytest.approx(61.0 / 3.0)
    assert ua.n_effective(61, 0.9) == pytest.approx(61.0 * 0.1 / 1.9)
    assert ua.n_effective(61, 1.0) != ua.n_effective(61, 1.0)   # NaN


def test_lag1_recovers_ar1_correlation():
    x = ar1(4000, 0.7, seed=5)
    assert ua.lag1(x) == pytest.approx(0.7, abs=0.05)
    assert ua.lag1([1.0, 2.0]) != ua.lag1([1.0, 2.0])           # too short -> NaN


# --------------------------------------------------------------------------- #
# 3. the half-width is the ledger's definition
# --------------------------------------------------------------------------- #
def test_half_width_matches_footprint_uncertainty_definition():
    import footprint_peak as fpm
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    row = [{"footprint_cells": 4.0, "cp_peak": v} for v in vals]
    ref = fpm.footprint_uncertainty(row, 4.0, [row])
    got = ua.half_width(vals)
    assert got["rel_half_width"] == pytest.approx(ref["rel_half_width"], rel=1e-12)
    assert got["p50"] == pytest.approx(ref["p50"], rel=1e-12)


# --------------------------------------------------------------------------- #
# 4. end-to-end audit on a synthetic dump (no solver)
# --------------------------------------------------------------------------- #
def _synthetic_dump(rho: float, n_frames: int = 21, seed: int = 3) -> dict:
    """Frames whose sample *counts* follow an AR(1) amplitude — fake but finite."""
    rng = random.Random(seed)
    amp = ar1(n_frames, rho, seed=seed)
    frames = []
    for k in range(n_frames):
        a = max(0.0, 1.0 + 0.3 * amp[k])
        tau = [0.05 * j for j in range(20)]
        cp = [a * (6.0 - 3.0 * t) for t in tau]
        frames.append({"t": 1e-4 * k, "tau": tau, "cp": cp})
    return {"case": "SYNTH", "tag": "synth", "dx": 2e-3, "t_end": 1e-2,
            "C": 2.720699046, "q": 500.0, "frames": frames}


def test_audit_runs_and_reports_both_intervals():
    dump = _synthetic_dump(rho=0.8)
    rep = ua.audit(dump, cells=4, resamples=100, seed=7)
    assert rep["iid"]["n_valid"] > 0
    assert set(rep["block"]) == {str(L) for L in ua.BLOCK_LENGTHS}
    assert rep["observed_cp"] > 0.0
    assert len(rep["per_frame_series"]) == len(dump["frames"])
    # the compact form must stay inside the job-log tail budget
    size = len(json.dumps(ua.compact(rep)))
    assert size < 4000, f"compact audit is {size} chars"


def test_audit_block_interval_wider_for_autocorrelated_frames():
    dump = _synthetic_dump(rho=0.9)
    rep = ua.audit(dump, cells=4, resamples=200, seed=11)
    u_iid = rep["iid"]["rel_half_width"]
    u_blk = rep["block"]["15"]["rel_half_width"]
    assert u_blk > u_iid


def test_compact_drops_the_bulky_series():
    rep = ua.audit(_synthetic_dump(rho=0.5), cells=4, resamples=50, seed=5)
    comp = ua.compact(rep)
    assert "per_frame_series" not in comp
    assert "block" in comp and "iid" in comp