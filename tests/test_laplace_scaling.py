"""Regression tests for the Laplace-law scaling verification (m2574).

Two jobs:

1. Hold the *instrument* honest: the cut-force readings must be invariant to cut
   direction (hex lattice anisotropy is not allowed to masquerade as physics),
   the two force estimators must be related by exactly the free-surface density
   deficit, and the σ-linearity of the CSF force must be exact.
2. Pin the validity envelope measured in ``benchmarks/laplace_scaling``: the
   amplitude is inside the 15 % platform tolerance at ``R/dx <= 20`` and the
   bias *grows* with refinement.  The second assertion is deliberately written
   as "grows", not "converges", because that is what the harness measures -- if
   a future fix makes the scheme convergent, this test fails and the verification
   record has to be updated rather than quietly re-pinned.

Numbers come from live runs of the harness, not from the committed JSON, so the
JSON cannot rot unnoticed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

RUN = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_scaling" / "run.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("laplace_scaling_run", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _load_harness()


@pytest.fixture(scope="module")
def reference():
    """The reference point of benchmarks/csf_consistent: R = 20 mm, R/dx = 10."""
    return H.cut_force_and_fields(0.02, 0.002, 7.2, h_factor=1.3)


@pytest.fixture(scope="module")
def coarse_and_fine():
    return (H.cut_force_and_fields(0.02, 0.0025, 7.2, h_factor=1.3),   # R/dx = 8
            H.cut_force_and_fields(0.02, 0.00125, 7.2, h_factor=1.3))  # R/dx = 16


def test_force_is_exactly_linear_in_sigma(reference):
    """CSF force ~ sigma by construction: the ratio must not move at all."""
    twice = H.cut_force_and_fields(0.02, 0.002, 14.4, h_factor=1.3)
    assert twice["cut_force"]["ratio_measured_over_theory"] == pytest.approx(
        reference["cut_force"]["ratio_measured_over_theory"], rel=1e-12)
    assert twice["cut_force_geometric"]["ratio_measured_over_theory"] == pytest.approx(
        reference["cut_force_geometric"]["ratio_measured_over_theory"], rel=1e-12)


def test_cut_direction_does_not_change_the_answer(reference):
    """The hex lattice's 6-fold symmetry must not show up as a cut-force spread."""
    aa = reference["angle_average"]
    assert aa["ratio_momentum_spread"] < 0.02        # measured: < 1 %
    for ratio in aa["ratio_momentum"]:
        assert ratio == pytest.approx(aa["ratio_momentum_mean"], rel=0.02)


def test_net_force_vanishes(reference):
    """A closed interface transmits no net force (Newton's third law check)."""
    for net in reference["angle_average"]["net_force_over_2sigma"]:
        assert abs(net) < 0.02


def test_two_estimators_differ_by_the_surface_density_deficit(reference):
    """momentum/geometric must equal rho0/rho on the force-carrying shell."""
    cf = reference["cut_force"]
    geom = reference["cut_force_geometric"]
    rho_ratio = geom["rho_over_rho0_force_weighted"]
    # The identity is exact only for identical weights: the recorded rho/rho0
    # mean is weighted by |m a| over the whole cut half-plane, while the ratio of
    # the two estimators is weighted by the signed x-component.  The measured
    # 0.06 % mismatch is that difference, not a modelling error.
    assert cf["ratio_measured_over_theory"] / geom["ratio_measured_over_theory"] == \
        pytest.approx(1.0 / rho_ratio, rel=0.01)
    # The free-surface deficit is large enough that the two estimators straddle
    # the tolerance: this is the instrument-ambiguity caveat of the report.
    assert 0.70 < rho_ratio < 0.90


def test_laplace_amplitude_within_tolerance_up_to_R_over_dx_20(reference,
                                                              coarse_and_fine):
    """The published claim: <= 15 % error for R/dx in {8, 10, 16}."""
    for case in (reference, *coarse_and_fine):
        ratio = case["cut_force"]["ratio_measured_over_theory"]
        assert abs(ratio - 1.0) < 0.15, case["resolution_R_over_dx"]


def test_bias_grows_with_refinement(coarse_and_fine, reference):
    """Not convergent: the bias at R/dx = 16 exceeds the one at R/dx = 8.

    This pins the limitation of the merged scheme; see
    docs/verification_laplace_scaling.md section 3.3.  A scheme that converges
    would break this test on purpose.
    """
    coarse, fine = coarse_and_fine
    b_coarse = coarse["cut_force"]["ratio_measured_over_theory"] - 1.0
    b_fine = fine["cut_force"]["ratio_measured_over_theory"] - 1.0
    b_ref = reference["cut_force"]["ratio_measured_over_theory"] - 1.0
    assert 0.0 < b_coarse < b_ref < b_fine


def test_delta_s_integral_is_short_of_unity(reference):
    """int delta_s dn is ~0.75, not 1: the force is right only through kappa."""
    int_ds = reference["delta_s"]["integral_dn"]
    assert 0.70 < int_ds < 0.80


def test_curvature_bias_is_inward_and_outer(unambiguous):
    """The interface band is asymmetric: inner half over-, outer half under-curved."""
    inner = unambiguous["curvature"]["inner_half_mean"] * unambiguous["radius_m"]
    outer = unambiguous["curvature"]["outer_half_mean"] * unambiguous["radius_m"]
    assert inner > 1.0 > outer
    assert 1.0 < inner < 1.5 and 0.8 < outer < 1.0


@pytest.fixture(scope="module")
def unambiguous(reference):
    return reference


# sigma = 0 is deliberately not exercised here: the harness normalises by 2*sigma
# (it is a ratio harness), and the zero-force path is already pinned by
# tests/test_laplace_droplet.py::test_zero_sigma_produces_no_force.


def test_committed_results_json_matches_a_fresh_run(reference):
    """The committed artifacts must not drift away from the code."""
    import json
    data = json.loads((RUN.parent / "results.json").read_text())
    committed = next(c for c in data["grid"]
                     if c["radius_m"] == 0.02 and c["sigma_N_per_m"] == 7.2)
    assert committed["cut_force"]["ratio_measured_over_theory"] == pytest.approx(
        reference["cut_force"]["ratio_measured_over_theory"], rel=1e-9)
    assert torch.isfinite(torch.tensor(reference["cut_force"]["measured_N_per_m"]))