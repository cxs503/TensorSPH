"""Invariants of the half-split decomposition (no solver run).

The script's only claim is arithmetic: *if the half-split survives at a common bin
count it is not an operator artefact*.  These tests pin exactly that property, plus
the bootstrap and the wet-node stratification, on synthetic clouds with known
answers.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
BENCH = HERE.parent / "benchmarks" / "case_sph_wedge_entry_2d"
sys.path.insert(0, str(BENCH))

import half_split_decomposition as hd  # noqa: E402


def cloud(n_tau=400, amp=1.0, offset=1.0, t=0.01, n_wet=60):
    """Uniform cloud in tau with a *linear* profile: cp = offset + amp*tau."""
    taus = [(k + 0.5) / n_tau for k in range(n_tau)]
    cps = [offset + amp * s for s in taus]
    return {"t": t, "tau": taus, "cp": cps, "n_wet": n_wet, "c_sph": 0.05}


def identical_shots(n=20, **kw):
    return [dict(cloud(**kw), t=0.001 * k) for k in range(n)]


# --------------------------------------------------------------------------- #
# arithmetic
# --------------------------------------------------------------------------- #
def test_rel_is_relative_to_the_second_argument():
    assert hd.rel(1.0, 1.2) == pytest.approx(0.2 / 1.2)
    assert hd.rel(1.2, 1.0) == pytest.approx(0.2)
    assert math.isnan(hd.rel(float("nan"), 1.0))


def test_half_reading_peak_is_the_highest_populated_bin():
    shots = identical_shots(4, amp=5.0, offset=1.0)
    r = hd.half_reading(shots, 10)
    assert r["n_bins"] == 10
    # linear profile -> the peak sits in the bin nearest tau = 1
    assert 0.85 <= r["tau_at_peak"] <= 1.0
    assert r["peak"] == pytest.approx(1.0 + 5.0 * r["tau_at_peak"], abs=0.3)


def test_common_bin_split_is_operator_independent_by_construction():
    """Flat profile, amplitude x1.2 in the late half: *every* bin count must
    return the identical relative split.  This is the property that makes the
    real-data check decisive -- a split that survives all bin counts cannot be a
    binning artefact."""
    early = identical_shots(10, amp=0.0, offset=5.0)
    late = identical_shots(10, amp=0.0, offset=6.0)
    out = hd.decompose(early + late, dx=0.002)
    splits = [v["rel_split"] for v in out["common_bin_splits"].values()]
    assert len(splits) >= 3
    assert max(splits) - min(splits) < 1e-12
    assert splits[0] == pytest.approx(0.2 / 1.2, abs=1e-9)


def test_common_bin_split_of_a_sloped_profile_stays_within_a_few_percent():
    """With a sloped profile the peak *location* shifts with the bin count, so the
    split is only approximately operator independent -- it must still be stable to
    a few percent, which is enough to separate a real drift from a binning effect."""
    out = hd.decompose(identical_shots(10, amp=5.0) + identical_shots(10, amp=6.0),
                       dx=0.002)
    splits = [v["rel_split"] for v in out["common_bin_splits"].values()]
    assert max(splits) - min(splits) < 0.01


def test_decompose_reports_zero_split_for_identical_halves():
    shots = identical_shots(20, amp=5.0)
    out = hd.decompose(shots, dx=0.002)
    assert out["legacy_half_split"] == pytest.approx(0.0, abs=1e-12)
    assert out["common_band"]["rel_split"] == pytest.approx(0.0, abs=1e-12)
    assert out["resolution_band_peak_median"] > 0.0


def test_decompose_lists_only_bins_admissible_in_both_halves():
    shots = identical_shots(8)
    out = hd.decompose(shots, dx=0.002)
    for tag in ("early", "late"):
        assert set(out["common_bins"]).issubset(set(out["admissible_bins"][tag]["bins"]))


def test_resolution_band_is_derived_from_the_grid_spacing():
    fine = hd.decompose(identical_shots(6, t=0.01), dx=0.0005)
    coarse = hd.decompose(identical_shots(6, t=0.01), dx=0.004)
    assert min(fine["resolution_band_bins"]) > min(coarse["resolution_band_bins"])
    assert max(fine["resolution_band_bins"]) >= max(coarse["resolution_band_bins"])


# --------------------------------------------------------------------------- #
# bootstrap / gating
# --------------------------------------------------------------------------- #
def test_bootstrap_is_degenerate_on_a_frozen_cloud():
    shots = identical_shots(12)
    res = hd.bootstrap(shots, [10, 12], draws=25)
    assert res["sd"] == pytest.approx(0.0, abs=1e-12)
    assert res["peak_median"] > 0.0


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    shots = identical_shots(15)
    for k, s in enumerate(shots):          # perturb so the draws matter
        s["cp"] = [c * (1.0 + 0.02 * ((k * 7 + i) % 5)) for i, c in enumerate(s["cp"])]
    a = hd.bootstrap(shots, [10, 12], draws=40, seed=7)
    b = hd.bootstrap(shots, [10, 12], draws=40, seed=7)
    assert a == b
    assert a["rel_sd"] > 0.0


def test_wet_gate_strata_drop_low_node_frames():
    shots = [dict(cloud(), n_wet=5 + k) for k in range(40)]
    out = hd.decompose(shots, dx=0.002)
    assert "16" in out["wet_gate"]
    assert out["wet_gate"]["16"]["n_frames"] < len(shots)
    assert out["wet_gate"]["30"]["n_frames"] <= out["wet_gate"]["16"]["n_frames"]