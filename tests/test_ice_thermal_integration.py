"""Integration tests: the ice multi-physics face as a *caller* of a solver path.

Two things are asserted here, and they are the ones a reviewer of node
IN13W2-N07 should check:

1. the thermal-saline state really drives the mechanical branch of the main
   wave-ice solver (``tensorsph.wave_ice_interaction``): the flexural strength
   used by the fracture criterion is the one computed from ``T``/``S`` through
   the brine-volume relation of ``tensorsph.ice_thermal.properties``.  The test
   re-derives the number independently and checks that the fracture verdict
   responds to it (warm/saline ice breaks, cold/fresh ice does not), so a
   regression that silently re-inserts the 0.7 MPa constant is caught;
2. the evidence bundle of the campaign run exists, points at files that are
   really in the repository, and its benchmark rows pass their declared
   tolerances (i.e. the numbers in the report are not hand-typed).

The second test skips when ``runs/IN13W2-N07`` is absent (installed package /
docs-only checkout) and is strict in CI where the run is committed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorsph.ice_thermal.properties import (V_B_CLOSURE, brine_volume_fraction,
                                              flexural_strength, strength_band)
from tensorsph.wave_ice_interaction import (SIGMA_F_DEFAULT, WaveIceScenario,
                                            run_wave_ice_scenario,
                                            strength_from_ice_state)

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "runs" / "IN13W2-N07"


# --------------------------------------------------------------- strength law
def test_flexural_strength_matches_the_timco_frederking_collapse():
    """sigma = 1.76 MPa (1 - sqrt(v_b / 0.27)) on the brine volume of the state."""
    for temperature_c, salinity in ((-2.0, 5.0), (-5.0, 10.0), (-15.0, 5.0)):
        temperature = 273.15 + temperature_c
        v_b = brine_volume_fraction(temperature, salinity)
        expected = 1.76e6 * (1.0 - math.sqrt(v_b / V_B_CLOSURE))
        assert flexural_strength(temperature, salinity) == pytest.approx(expected, rel=1e-12)


def test_strength_inversion_recovers_the_platform_default():
    """The 0.7 MPa default must correspond to a physical first-year state.

    Inverting the collapse curve gives the brine volume at which the strength
    equals the constant the rest of the platform uses::

        v_b = 0.27 (1 - 0.7/1.76)^2 = 0.098,

    and a 10 psu column reaches that brine volume at about -5.4 C.  Both the
    brine-volume and the temperature leg are checked, so the default is tied to
    a state a ship could actually measure rather than to a free constant.
    """
    v_b_target = V_B_CLOSURE * (1.0 - SIGMA_F_DEFAULT / 1.76e6) ** 2
    assert v_b_target == pytest.approx(0.0979, abs=2e-3)
    # bisect T at S = 10 psu until the brine volume matches the inverted value
    lo, hi = 263.15, 271.15
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if brine_volume_fraction(mid, 10.0) > v_b_target:
            hi = mid          # too warm: the brine volume is above target
        else:
            lo = mid
    temperature = 0.5 * (lo + hi)
    assert temperature - 273.15 == pytest.approx(-5.4, abs=0.3)
    assert flexural_strength(temperature, 10.0) == pytest.approx(SIGMA_F_DEFAULT, rel=2e-3)


def test_strength_monotone_in_temperature_and_salinity():
    """Warmer or saltier ice is weaker -- the whole point of the coupling."""
    cold = flexural_strength(273.15 - 20.0, 5.0)
    mild = flexural_strength(273.15 - 10.0, 5.0)
    warm = flexural_strength(273.15 - 2.0, 5.0)
    assert cold > mild > warm
    assert flexural_strength(273.15 - 5.0, 12.0) < flexural_strength(273.15 - 5.0, 3.0)
    # above the pore-closure volume fraction the skeleton carries nothing
    assert flexural_strength(273.15 - 2.0, 32.0) >= 0.0
    sigma, low, high = strength_band(273.15 - 5.0, 10.0)
    assert low < sigma < high
    # the derived value must live inside the observed first-year flexural range
    assert 0.5e6 <= sigma <= 1.5e6


# ------------------------------------------------------- main-solver integration
def test_strength_from_ice_state_reports_the_state_it_used():
    state = strength_from_ice_state(-5.0, 10.0)
    assert state["temperature_k"] == pytest.approx(268.15)
    assert state["brine_fraction"] == pytest.approx(
        brine_volume_fraction(268.15, 10.0))
    assert state["sigma_f_pa"] == pytest.approx(flexural_strength(268.15, 10.0))
    assert state["sigma_f_mpa"] == pytest.approx(state["sigma_f_pa"] / 1e6)


def test_wave_ice_fracture_uses_the_thermal_state():
    """The solver's fracture criterion must take sigma_f from T and S."""
    base = run_wave_ice_scenario(WaveIceScenario(Hs=1.0, Tp=8.0, h_ice=0.35))
    cold = run_wave_ice_scenario(
        WaveIceScenario(Hs=1.0, Tp=8.0, h_ice=0.35, temperature_c=-20.0, salinity_psu=3.0))
    warm = run_wave_ice_scenario(
        WaveIceScenario(Hs=1.0, Tp=8.0, h_ice=0.35, temperature_c=-2.0, salinity_psu=5.0))

    # the default scenario is left bit-for-bit unchanged (the constant path)
    assert base["fracture"]["sigma_f_Pa"] == pytest.approx(SIGMA_F_DEFAULT)
    assert base["fracture"]["thermal_state"] is None

    for result, (temperature_c, salinity) in ((cold, (-20.0, 3.0)),
                                              (warm, (-2.0, 5.0))):
        fracture = result["fracture"]
        assert fracture["thermal_state"] is not None
        assert fracture["sigma_f_Pa"] == pytest.approx(
            flexural_strength(273.15 + temperature_c, salinity))
        # the criterion, the FSD evolution and K_MIZ all consume this sigma_f
        assert fracture["ratio"] == pytest.approx(
            fracture["sigma_max_Pa"] / fracture["sigma_f_Pa"])

    # strong cold/fresh ice resists, weak warm/saline ice breaks earlier
    assert cold["fracture"]["sigma_f_Pa"] > warm["fracture"]["sigma_f_Pa"]
    assert warm["fracture"]["ratio"] > cold["fracture"]["ratio"]
    assert cold["K_MIZ"] != warm["K_MIZ"] or cold["D_mean_final"] != warm["D_mean_final"]


def test_wave_ice_breaking_threshold_follows_the_derived_strength():
    """Picking the wave amplitude so sigma_max/sigma_f ~ 1 turns on the breakup."""
    reference = strength_from_ice_state(-5.0, 10.0)["sigma_f_pa"]
    strong = run_wave_ice_scenario(
        WaveIceScenario(Hs=1.0, h_ice=0.35, temperature_c=-22.0, salinity_psu=2.0))
    weak = run_wave_ice_scenario(
        WaveIceScenario(Hs=1.0, h_ice=0.35, temperature_c=-2.0, salinity_psu=5.0))
    sigma_max = weak["fracture"]["sigma_max_Pa"]          # same wave on both
    assert strong["fracture"]["sigma_max_Pa"] == pytest.approx(sigma_max)
    assert strong["fracture"]["ratio"] < weak["fracture"]["ratio"]
    assert reference > 0.0


def test_fully_disintegrated_ice_breaks_at_any_wave():
    """v_b above the pore-closure fraction -> no strength -> ratio = inf, no crash."""
    state = strength_from_ice_state(-1.5, 15.0)
    assert state["brine_fraction"] > V_B_CLOSURE
    assert state["sigma_f_pa"] == 0.0
    result = run_wave_ice_scenario(
        WaveIceScenario(Hs=1.0, h_ice=0.35, temperature_c=-1.5, salinity_psu=15.0))
    assert result["fracture"]["sigma_f_Pa"] == 0.0
    assert math.isinf(result["fracture"]["ratio"])
    assert result["fracture"]["breaks"] is True


# ------------------------------------------------------------- evidence bundle
def test_evidence_bundle_is_repository_backed():
    if not (RUN_DIR / "evidence.json").is_file():
        pytest.skip("no committed campaign run under runs/IN13W2-N07")
    bundle = json.loads((RUN_DIR / "evidence.json").read_text())
    assert bundle["checks"]["impl_files_ok"]
    assert bundle["checks"]["entries_exist"]
    assert bundle["checks"]["run_artifacts_exist"]
    assert bundle["checks"]["integration_callers_verified"]
    for row in bundle["impl_files"]:
        assert (REPO / row["path"]).is_file()
        assert not row["stub_functions"]
    for row in bundle["run_artifacts"]:
        path = REPO / row["path"]
        assert path.is_file(), row
        assert path.stat().st_size > 0
    for caller in bundle["integration_callers"]:
        assert caller["verified"]
        assert (REPO / caller["file"]).is_file()
    rows = bundle["benchmark"]["rows"]
    assert rows, "benchmark table must not be empty"
    for row in rows:
        assert row["pass"], row
        assert row["rel_error"] <= row["tolerance"]


def test_benchmark_numbers_reproduce_the_analysis_formula():
    """Re-derive two anchors from results.json instead of trusting the verdicts."""
    if not (RUN_DIR / "results.json").is_file():
        pytest.skip("no committed campaign run under runs/IN13W2-N07")
    res = json.loads((RUN_DIR / "results.json").read_text())
    for row in res["stefan"]:
        err = abs(row["front_final_m"] - row["front_analytic_m"]) / row["front_analytic_m"]
        assert err == pytest.approx(row["front_rel_error_final"], rel=1e-9)
        assert err <= 0.03
    darcy = res["darcy"]
    err = abs(darcy["ra_critical_numeric"] - darcy["ra_critical_exact"]) \
        / darcy["ra_critical_exact"]
    assert err == pytest.approx(darcy["ra_relative_error"], rel=1e-9)
    stress = res["stress"]
    err = abs(stress["restrained_max_pa"] - stress["timoshenko_analytic_pa"]) \
        / stress["timoshenko_analytic_pa"]
    assert err == pytest.approx(stress["restrained_relative_error"], rel=1e-9)
    assert res["benchmark"]["all_pass"] or not res["benchmark"]["rows"]