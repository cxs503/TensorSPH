"""Regression pins for the recorded free-surface closure / force-balance evidence.

The physics runs are too slow for the unit suite; what is pinned here are the
numbers in ``benchmarks/laplace_pressure_closure/report_force_balance.json`` and
``raw/relax_ladder.json`` and the *qualitative* statements the verification
document makes about them, so that a silent change in the closure, the CSF
force, the estimator or the measurement cannot pass unnoticed.

The load-bearing pin is the negative control in
``test_naive_shell_mean_estimator_is_the_wrong_integral``: the first version of
this analysis used a shell-mean integral that is a factor ~2 below the flux
integral, and it would have 'proved' a force-side deficit that does not exist.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure_closure"
FORCE = BENCH / "report_force_balance.json"
LADDER = BENCH / "raw" / "relax_ladder.json"


@pytest.fixture(scope="module")
def force():
    if not FORCE.exists():
        pytest.skip("run benchmarks/laplace_pressure_closure/analysis.py first")
    return json.loads(FORCE.read_text())


@pytest.fixture(scope="module")
def ladder():
    if not LADDER.exists():
        pytest.skip("run benchmarks/laplace_pressure_closure/relax_ladder.py first")
    return json.loads(LADDER.read_text())


def test_force_side_holds_the_full_laplace_jump(force):
    """The discrete CSF force integrates to sigma/R within 10 % at both resolutions."""
    ratios = [c["force_side"]["delta_p_force_over_analytic"] for c in force["cases"]]
    assert len(ratios) == 2
    for value in ratios:
        assert value == pytest.approx(1.0, abs=0.10), ratios
    # Direction: the force is inward, so the core is at the higher pressure.
    for c in force["cases"]:
        assert c["force_side"]["delta_p_force_Pa"] > 0.0


def test_flux_integral_is_bin_insensitive(force):
    """No bin width may be selected to make the result: the spread is < 1 %."""
    for table in force["bin_sensitivity"].values():
        values = list(table.values())
        assert max(values) - min(values) < 0.01 * max(values)


def test_naive_shell_mean_estimator_is_the_wrong_integral(force):
    """The shell-mean estimator is low by a large factor; it must not be used.

    Kept as an explicit negative control: if someone 'fixes' the analysis by
    switching back to it, this assertion fires.
    """
    for c in force["cases"]:
        flux = c["force_side"]["delta_p_force_Pa"]
        naive = c["force_side"]["naive_shell_mean_estimator_Pa"]
        assert naive < 0.85 * flux, (naive, flux)


def test_pressure_side_is_unchanged_on_a_relaxed_lattice(force):
    """The closure maps p = 0 to p = 0: it extrapolates a field, not a constant."""
    for c in force["cases"]:
        for name, row in c["pressure_side"].items():
            assert row["p_core_over_analytic"] == pytest.approx(0.0, abs=1e-9), (name, row)


def test_closure_improves_the_relaxed_laplace_readout(ladder):
    """Closure on must beat closure off on the same run, and stay stable."""
    cases = {c["case"]: c for c in ladder["cases"]}
    assert "closure_off" in cases
    off = cases["closure_off"]
    assert off["tail_rel_error_pct"] == pytest.approx(-37.9, abs=5.0)

    # A variant that only wins by making the surface breathe is not a win, so
    # "best" is taken over the variants that satisfy Adami's spurious-current
    # criterion v_max/v_ref < 0.1. The one-sided variant is excluded by it.
    stable = [c for c in cases.values() if c["vmax_over_vref"] < 0.1]
    assert stable, cases
    best = min(stable, key=lambda c: abs(c["tail_rel_error_pct"]))
    assert best["case"] != "closure_off"
    assert abs(best["tail_rel_error_pct"]) < abs(off["tail_rel_error_pct"]) - 10.0


def test_the_all_donors_variant_is_the_stable_one(ladder):
    """The literal formula (all donors) is the variant that stays quiet.

    The one-sided variant over-pressurises the shell and drives a breathing
    mode; this pin records that, so the choice of default donor set is not made
    silently.
    """
    cases = {c["case"]: c for c in ladder["cases"]}
    if "closure_on_interior_layers2" not in cases:
        pytest.skip("interior variant not in this ladder")
    allv = cases["closure_on_all_donors"]
    interior = cases["closure_on_interior_layers2"]
    assert allv["vmax_over_vref"] < interior["vmax_over_vref"]


def test_verdict_is_withheld_not_victory(ladder):
    """Honest FAIL pin: no run is inside the 15 % band, so Laplace stays WITHHELD.

    If a future change does put a variant inside the band, this test fails and
    forces the verification document (and the capability grading) to be updated
    instead of drifting.
    """
    cases = {c["case"]: c for c in ladder["cases"]}
    stable = [c for c in cases.values() if c["vmax_over_vref"] < 0.1]
    best = min(stable, key=lambda c: abs(c["tail_rel_error_pct"]))
    assert abs(best["tail_rel_error_pct"]) > 15.0, best
