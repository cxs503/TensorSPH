"""Tests for the wedge-entry compressibility-closure study (m20914).

The study exists to put the "+234 % Cp" root-cause diagnosis at risk: if the
excess is the weakly-compressible acoustic noise floor, raising ``c0/V`` must
sink it.  These tests pin the three things whose drift would silently turn the
verdict into a rubber stamp:

1. the predeclared design (a tuned grid/sound-speed ladder is the classic way
   to manufacture a PASS after the fact),
2. the closed-form noise floor (it is the yardstick the whole argument uses),
3. the falsifier itself -- a synthetic ladder that does *not* fall must be
   graded FALSIFIED, so the negative control is live rather than decorative.

No SPH simulation is run here (the numbers come from artifacts), so this file
runs in well under a second.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"

# The benchmark directory is a script collection, not an installed package; load
# it by path the same way a user would run it.
sys.path.insert(0, str(BENCH))
_spec = importlib.util.spec_from_file_location(
    "wedge_compressibility", BENCH / "compressibility.py")
comp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comp)


# --------------------------------------------------------------------------- #
# 1. Predeclared design is frozen
# --------------------------------------------------------------------------- #

def test_design_is_frozen_before_results():
    """The ladder literals must not drift; they are what makes the verdict predeclared."""
    assert comp.FROZEN["beta_deg"] == 30.0
    assert comp.FROZEN["velocity"] == 1.0
    assert comp.FROZEN["dx"] == 4.0e-3
    assert tuple(comp.FROZEN["sound_speed_factor_levels"]) == (20.0, 40.0, 80.0, 160.0)
    assert comp.FROZEN["confirm_dx"] == 2.0e-3


def test_sound_ladder_varies_one_knob_only():
    """c0/V varies; beta, V, dx, frames and the aggregation stay identical."""
    ladder = [c for c in comp.case_list() if c["role"] == "sound_ladder"]
    assert len(ladder) == 4
    assert {c["dx"] for c in ladder} == {4.0e-3}
    assert {c["beta_deg"] for c in ladder} == {30.0}
    assert {c["velocity"] for c in ladder} == {1.0}
    assert {c["frames"] for c in ladder} == {13}
    c0 = [c["sound_speed_factor"] for c in ladder]
    assert c0 == sorted(c0), "sound-speed ladder must be monotone so the trend is meaningful"
    # Every level doubles the Mach number ratio -> a clean 2x sequence, which is
    # what lets the deficit be read as a power law rather than as a coincidence.
    assert [c0[i + 1] / c0[i] for i in range(len(c0) - 1)] == [2.0, 2.0, 2.0]


def test_tolerance_is_the_case_pool_tolerance_not_a_relaxed_one():
    """The study may not relax the 15 % case-pool acceptance to pass."""
    assert comp.CP_TOL == pytest.approx(0.15)
    assert comp.MASS_TOL == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# 2. Closed-form acoustic noise floor
# --------------------------------------------------------------------------- #

def test_density_floor_matches_tait_equation():
    """Cp_floor == p/q for p = B[(rho/rho0)^gamma - 1], B = rho0 c0^2/gamma."""
    c0_over_V, rho_ratio, gamma = 20.0, 1.0471, 7.0
    p_over_q = comp.cp_density_floor(c0_over_V, rho_ratio, gamma)
    # independent recomputation in SI units (rho0 and V cancel, keep them explicit)
    rho0, V = 1000.0, 1.0
    B = rho0 * (c0_over_V * V) ** 2 / gamma
    p = B * (rho_ratio ** gamma - 1.0)
    q = 0.5 * rho0 * V ** 2
    assert p_over_q == pytest.approx(p / q, rel=1e-12)
    # The value quoted in docs/case_sph_wedge_entry_2d.md sec. 5.3 is 43.48.
    assert p_over_q == pytest.approx(43.48, rel=0.01)


def test_density_floor_scales_with_mach_squared():
    """Halving the Mach number quarters the floor: the argument's scaling law."""
    rho = 1.01
    base = comp.cp_density_floor(20.0, rho)
    assert comp.cp_density_floor(40.0, rho) == pytest.approx(4.0 * base, rel=1e-12)


def test_floor_at_one_percent_density_error_is_the_size_of_the_reference():
    """Why c0/V = 20 cannot resolve Cp of order 8 even in principle."""
    floor = comp.cp_density_floor(20.0, 1.01)
    assert 7.0 < floor < 9.5, floor  # reference Cp_max = 8.4022


# --------------------------------------------------------------------------- #
# 3. Falsifier / verdict logic (pure function, synthetic ladders)
# --------------------------------------------------------------------------- #

REF = 8.4022


def _ladder(values, c0s=(20.0, 40.0, 80.0, 160.0)):
    return [{"cp_band_peak": v, "c0_over_V": c} for v, c in zip(values, c0s)]


def test_verdict_supported_when_ladder_lands_on_the_anchor():
    out = comp.classify(_ladder([28.0, 18.0, 11.0, 8.9]), REF)
    assert out["verdict"] == "SUPPORTED_FULLY"
    assert out["non_increasing"] is True


def test_verdict_partial_when_trend_is_there_but_residual_remains():
    out = comp.classify(_ladder([28.0, 22.0, 17.0, 13.0]), REF)
    assert out["verdict"] == "PARTIALLY_SUPPORTED"
    assert 0.5 <= out["explained_fraction"] < 1.0


def test_negative_control_flat_ladder_is_falsified():
    """If c0/V does nothing, the hypothesis must die -- not be quietly reworded."""
    out = comp.classify(_ladder([28.0, 28.2, 28.1, 28.3]), REF)
    assert out["verdict"] == "FALSIFIED_NO_TREND"
    assert out["falls_at_stiff_end"] is False


def test_negative_control_rises_with_stiffness_is_falsified():
    out = comp.classify(_ladder([8.1, 9.4, 11.0, 13.2]), REF)
    assert out["verdict"] == "FALSIFIED_NO_TREND"


def test_weak_trend_is_refuted_as_dominant_cause():
    """A small fall (<50 % of the excess) must not be sold as the root cause."""
    out = comp.classify(_ladder([28.0, 26.5, 25.0, 23.5]), REF)
    assert out["verdict"] == "REFUTED_DOMINANT"


def test_verdict_withheld_with_a_single_level():
    assert comp.classify(_ladder([28.0], c0s=(20.0,)), REF)["verdict"] == "WITHHELD"


# --------------------------------------------------------------------------- #
# 4. End-to-end grading path on a synthetic artefact (no SPH)
# --------------------------------------------------------------------------- #

def _fake_artefact(case_id: str, c0_over_V: float, cp_profile: list[float],
                   rho_ratio: float = 1.03, dx: float = 4.0e-3) -> dict:
    n = len(cp_profile)
    taus = [1.2 * k / (n - 1) for k in range(n)]
    series = []
    for t in (0.002, 0.006, 0.010, 0.014, 0.018):
        series.append({
            "t": t,
            "c_sph": 0.05,
            "c_wagner": 0.05,
            "cp_max_footprint": max(cp_profile),
            "cp_max_interior": max(cp_profile),
            "cp_centre": cp_profile[0],
            "x_jet": 0.05,
            "max_rho_ratio": rho_ratio,
            "max_speed_over_V": 1.2,
            "profile_tau": taus,
            "profile_cp": list(cp_profile),
        })
    return {
        "case_id": case_id,
        "kind": "compressibility",
        "config": {"beta_deg": 30.0, "velocity": 1.0, "rho0": 1000.0,
                   "gamma": 7.0, "dx": dx, "sound_speed_factor": c0_over_V,
                   "h_factor": 1.3},
        "derived": {},
        "steps": 100,
        "wall_seconds": 1.0,
        "environment": {},
        "series": series,
    }


def test_synthetic_ladder_reaches_a_verdict_through_the_frozen_grader():
    """The analysis path (run.grade + ensemble profile) is exercised end to end."""
    import run as base  # the frozen case driver

    n = 120
    taus = [1.2 * k / (n - 1) for k in range(n)]
    arts = {}
    for c0, amp in zip((20.0, 40.0, 80.0, 160.0), (28.0, 18.0, 11.0, 8.9)):
        # Wagner-shaped profile scaled so its band peak equals ``amp``
        prof = []
        for tau in taus:
            s = 1.0 / math.sqrt(max(1.0 - min(tau, 0.999) ** 2, 1e-12))
            prof.append(amp * (2 * 2.7207 * s - s * s + 1.0) / (2.7207 ** 2 + 1.0))
        arts[f"c0{int(c0):03d}"] = _fake_artefact(f"c0{int(c0):03d}", c0, prof)
    ref = base.refs_derived()
    graded = base.grade(arts, ref)
    rows = graded["rows"]
    assert len(rows) == 4
    for r in rows:
        assert math.isfinite(r["cp_band_peak"])
        assert r["cp_density_floor"] if "cp_density_floor" in r else True
    peaks = [r["cp_band_peak"] for r in rows]
    assert peaks[0] == pytest.approx(28.0, rel=0.02)
    assert peaks[-1] == pytest.approx(8.9, rel=0.02)
    verdict = comp.classify(
        [{"cp_band_peak": r["cp_band_peak"], "c0_over_V": r["c0_over_V"]} for r in rows],
        ref["wagner"]["cp_max"])
    assert verdict["verdict"] == "SUPPORTED_FULLY"