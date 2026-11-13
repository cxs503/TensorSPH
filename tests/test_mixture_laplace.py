"""Young-Laplace gate for the mixture surface-tension model (m18642).

The tests assert *invariants*, not just execution:

``test_operator_floor``            the discrete stress divergence, fed an
                                  analytic unit-integral continuous surface
                                  stress, must return ``sigma/R`` within 5 %.
                                  This is the floor below which no colour-field
                                  model can be blamed for a Laplace error.
``test_assembly_normalisation``    ``upstream`` must equal ``consistent`` times
                                  the particle mass exactly -- that is the
                                  factor fixed in m18642 (``upstream`` carries
                                  ``m_i = rho_i dx^2`` too much).
``test_laplace_gate``              the gated configuration (consistent weight,
                                  symmetric stress bracket, hourglass off)
                                  reproduces ``Delta p = sigma/R`` within the
                                  ledger's 15 % at ``R/dx = 16``.
``test_upstream_assembly_fails``   regression lock on the documented defect:
                                  the faithful transcription must *stay* below
                                  -90 % so nobody can silently "fix" it back.
``test_delta_integral_is_unit``    ``int |grad c| dn`` on the water side stays in
                                  [0.9, 1.0] for a 1000:1 mixture.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "surface_tension"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mixture_laplace as ml  # noqa: E402


@pytest.fixture(scope="module")
def res16():
    return 16.0


def test_operator_floor(res16):
    dx = ml.RADIUS / res16
    floor = ml.operator_floor(ml.RADIUS, dx)
    assert abs(floor["err_pct"]) < 5.0, floor
    assert floor["momentum_residual"] < 1e-12, "pair assembly must conserve momentum"


def test_assembly_normalisation(res16):
    """Uniform density: upstream / consistent == m_i exactly.

    In the two-phase mixture the same identity holds per *pair class*: water-water
    pairs carry ``rho_w dx^2``, air-air and cross pairs ``dx^2`` (the pair weight
    of the light end is ``V/rho``), so the global ratio sits between ``dx^2`` and
    ``rho_w dx^2``.  Both bounds are asserted.
    """
    dx = ml.RADIUS / res16
    up = ml.measure(ml.RADIUS, dx, ml.RHO_AIR, "single_phase", "upstream", False)
    co = ml.measure(ml.RADIUS, dx, ml.RHO_AIR, "single_phase", "consistent", False)
    ratio = up["dp_flux_all_pa"] / co["dp_flux_all_pa"]
    mass = co["particle_mass_water"]
    assert abs(ratio - mass) / mass < 1e-10, (ratio, mass)

    up2 = ml.measure(ml.RADIUS, dx, ml.RHO_AIR, "two_phase", "upstream", False)
    co2 = ml.measure(ml.RADIUS, dx, ml.RHO_AIR, "two_phase", "consistent", False)
    ratio2 = up2["dp_flux_all_pa"] / co2["dp_flux_all_pa"]
    assert dx * dx <= ratio2 <= mass * (1 + 1e-12), (ratio2, dx * dx, mass)
    assert abs(ratio2 - mass) / mass < 5e-3, (ratio2, mass)


def test_laplace_gate(res16):
    dx = ml.RADIUS / res16
    row = ml.measure(ml.RADIUS, dx, ml.RHO_AIR, "two_phase", "consistent", False,
                     bracket="symmetric")
    assert abs(row["dp_flux_all_err_pct"]) < ml.GATE_PCT, row
    assert row["momentum_residual"] < 1e-12, row
    assert abs(row["dp_stress_err_pct"]) < ml.GATE_PCT, row


def test_upstream_assembly_fails(res16):
    """The faithful transcription decays as dx^2: the error must be < -90 %."""
    errs = []
    for rdx in (16.0, 32.0):
        dx = ml.RADIUS / rdx
        row = ml.measure(ml.RADIUS, dx, ml.RHO_AIR, "two_phase", "upstream", False)
        errs.append(row["dp_flux_all_err_pct"])
    assert all(e < -90.0 for e in errs), errs
    # and it must get *worse* with refinement (the mass factor shrinks)
    assert errs[1] < errs[0], errs


def test_delta_integral_is_unit(res16):
    dx = ml.RADIUS / res16
    row = ml.measure(ml.RADIUS, dx, ml.RHO_AIR, "two_phase", "consistent", False)
    assert 0.90 < row["delta_integral"] < 1.05, row["delta_integral"]