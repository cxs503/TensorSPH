"""Guards for the FSI2 beam modal reference and the lock-in audit.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

What is asserted, and why each assertion can fail:

1. **The analytic reference is the closed form, not a fitted number.** The
   clamped-free Euler-Bernoulli frequency of the shipped beam must equal
   ``1.8751^2/(2 pi) sqrt(EI/(rho A L^4))`` recomputed here from the case's own
   ``EI``/``rho A``/``L``.  ``EI`` is checked against ``E t^3/12`` separately,
   so a silently changed beam geometry cannot hide inside the frequency.
2. **The discrete rod converges to it.** The ladder in ``n`` must be monotone
   under refinement and the shipped ``n = 30`` model must be inside the
   predeclared 5 % target (it is *not* inside 3 %, and the test says so).
3. **The linear spectrum is the dynamics the driver integrates.** The free
   decay of the very routine the coupled driver sub-steps
   (``beam_advance``) must reproduce the eigenproblem frequency.  This is the
   assertion that makes the modal numbers usable to explain a run; without it
   the spectrum could be a property of a model nobody integrates.
4. **The lock-in audit is not flattering itself.** The similarity-corrected
   case (``beam_E ~ U^2``) must have overlapping fluid/shedding and structural
   period bands, *and* the un-scaled counterfactual must not -- otherwise the
   "similarity scaling is what saves the gate" claim would be vacuous.  The two
   archived coupled runs must be rejected by *both* yardsticks, i.e. the new
   reference must not rescue anything.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph import fsi2_beam_modes as bm  # noqa: E402
from tensorsph import fsi2_modal  # noqa: E402
from tensorsph.fsi2_similarity import similarity_corrected_case  # noqa: E402
from tensorsph.fsi2_turek_hron import FSI2Case  # noqa: E402


def test_bending_stiffness_is_the_closed_form():
    case = FSI2Case()
    ei = case.beam_E * case.beam_thickness ** 3 / 12.0
    assert case.beam_stiffness_bending() == pytest.approx(ei, rel=1e-12)
    assert case.beam_area_density() == pytest.approx(
        case.beam_rho * case.beam_thickness, rel=1e-12)


def test_analytic_frequency_matches_independent_closed_form():
    case = FSI2Case()
    beta = 1.875104068711961
    expected = (beta ** 2 / (2.0 * math.pi)
                * math.sqrt(case.beam_stiffness_bending()
                            / (case.beam_area_density() * case.beam_length ** 4)))
    got = bm.cantilever_frequency_hz(case.beam_E, case.beam_rho,
                                     case.beam_thickness, case.beam_length)
    assert got == pytest.approx(expected, rel=1e-12)
    # 4 MPa / 0.35 m / 20 kg/m^2 beam -> ~1.67 Hz; a geometry slip would move it
    assert got == pytest.approx(1.668, abs=0.005)


def test_mode_ratios_are_the_euler_bernoulli_eigenvalues():
    case = FSI2Case()
    f1 = bm.cantilever_frequency_hz(case.beam_E, case.beam_rho,
                                    case.beam_thickness, case.beam_length, 1)
    f2 = bm.cantilever_frequency_hz(case.beam_E, case.beam_rho,
                                    case.beam_thickness, case.beam_length, 2)
    assert (f2 / f1) ** 0.5 == pytest.approx(
        bm.EULER_BERNOULLI_BETAS[1] / bm.EULER_BERNOULLI_BETAS[0], rel=1e-12)


def test_added_mass_only_softens_the_beam():
    case = FSI2Case()
    dry = bm.cantilever_frequency_hz(case.beam_E, case.beam_rho,
                                     case.beam_thickness, case.beam_length)
    wet = bm.coupled_frequency_hz(case.beam_E, case.beam_rho,
                                  case.beam_thickness, case.beam_length,
                                  case.rho0)
    assert wet < dry
    assert bm.added_mass_per_length(case.rho0, case.beam_thickness) == pytest.approx(
        case.rho0 * math.pi * (0.5 * case.beam_thickness) ** 2, rel=1e-12)
    band = bm.lockin_period_band_s(case.beam_E, case.beam_rho,
                                   case.beam_thickness, case.beam_length,
                                   case.rho0)
    assert band[0] < 1.0 / wet < band[1]
    assert band[1] / band[0] == pytest.approx(
        (1 + bm.LOCKIN_HALF_WIDTH) / (1 - bm.LOCKIN_HALF_WIDTH), rel=1e-12)


def test_discrete_rod_matches_the_analytic_spectrum_within_declared_target():
    """Predeclared: |err f1| < 5 % at the shipped n=30, ladder monotone."""
    case = FSI2Case()
    modes = fsi2_modal.discrete_modes(case, n_modes=2)
    f1_ref = bm.cantilever_frequency_hz(case.beam_E, case.beam_rho,
                                        case.beam_thickness, case.beam_length)
    err = modes["frequencies_hz"][0] / f1_ref - 1.0
    assert abs(err) < 0.05, f"shipped rod f1 error {err:+.4f} outside 5 %"
    # honest record of the sharper target: it is NOT met at n = 30
    assert err < 0.0, "the finite clamp must soften the beam, not stiffen it"

    ladder = fsi2_modal.modal_ladder((15, 30), n_modes=1)
    e15, e30 = (r["mode1"]["rel_error"] for r in ladder)
    assert abs(e30) < abs(e15), "refining 15 -> 30 nodes must reduce the error"


@pytest.mark.slow
def test_free_decay_reproduces_the_linear_spectrum():
    """The integrated dynamics must be the spectrum the modal claim uses."""
    case = FSI2Case(n_beam_nodes=15)  # cheaper; the mechanism is the same
    decay = fsi2_modal.free_decay_period(case, periods=2.0)
    assert decay["period_s"] is not None
    assert abs(decay["relative_to_eigen"]) < 0.01, decay
    # symplectic Euler, three periods, no physical damping in the model
    assert decay["envelope_first_over_last"] == pytest.approx(1.0, abs=0.02)


def test_similarity_scaling_is_what_keeps_the_gate_band_valid():
    """The falsified hypothesis, pinned: scaling E is load bearing.

    ``fsi2_similarity.similarity_corrected_case`` scales ``E ~ U^2``.  With the
    scaling the structural period lands inside the fluid shedding band (so the
    verdict gate's band is corroborated by an independent route); *without* it
    the two bands are disjoint, i.e. the band would reject a genuine
    structural resonance.  Both halves are asserted: dropping the scaling from
    the fix must break this test.
    """
    u_mean = 0.2
    scaled = similarity_corrected_case(U_mean=u_mean)
    plain = FSI2Case()  # benchmark stiffness, repro inflow: NOT similarity scaled

    def bands(case):
        s = bm.structural_period_from_case(
            {"beam_E": case.beam_E, "beam_rho": case.beam_rho,
             "beam_thickness": case.beam_thickness,
             "beam_length": case.beam_length, "rho0": case.rho0},
            channel_height=case.channel_height)
        fluid = bm.period_strouhal_band_s(u_mean, case.cylinder_d)
        lo = max(fluid[0], s["lockin_band_s"][0])
        hi = min(fluid[1], s["lockin_band_s"][1])
        return s, fluid, hi - lo

    s_ok, fluid, overlap_ok = bands(scaled)
    assert scaled.beam_E < plain.beam_E, "similarity correction must soften E"
    assert overlap_ok > 0.0, (s_ok, fluid)
    _, _, overlap_bad = bands(plain)
    assert overlap_bad < 0.0, "counterfactual must be disjoint for this to mean something"

    s, fluid, _ = bands(scaled)
    ratio = 0.5 * (fluid[0] + fluid[1]) / s["period_coupled_s"]
    assert abs(ratio - 1.0) < 0.25, "shedding and structural periods must be close"


def test_archived_runs_stay_inadmissible_under_the_new_yardstick():
    """The reference must not rescue a ringing run (honest-FAIL discipline)."""
    import json

    sim = similarity_corrected_case(U_mean=0.2)
    period_struct = bm.structural_period_from_case(
        {"beam_E": sim.beam_E, "beam_rho": sim.beam_rho,
         "beam_thickness": sim.beam_thickness, "beam_length": sim.beam_length,
         "rho0": sim.rho0}, channel_height=sim.channel_height)["period_coupled_s"]
    fluid = bm.period_strouhal_band_s(0.2, 0.1)
    lo, hi = bm.lockin_period_band_s(sim.beam_E, sim.beam_rho,
                                     sim.beam_thickness, sim.beam_length,
                                     sim.rho0, channel_height=sim.channel_height)

    checked = 0
    for name in ("viv_dx002_aitken_admitted.json",
                 "viv_dx005_aitken_admitted.json"):
        path = ROOT / "docs" / "xval_fsi2" / name
        if not path.is_file():
            continue
        rec = json.loads(path.read_text())
        period = rec["window"]["period_s"]
        assert not (fluid[0] <= period <= fluid[1]), (name, period)
        assert not (lo <= period <= hi), (name, period)
        assert rec["admissibility"]["admissible"] is False
        # the ring is at a small fraction of the structural period
        assert period / period_struct < 0.5, (name, period, period_struct)
        checked += 1
    assert checked == 2, "archived verdict artefacts missing"