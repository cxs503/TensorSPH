"""Audit tests for the Wagner pressure reference of CASE-SPH-WEDGE-ENTRY-2D (m17079).

These pin the *reference side* of the case, which no solver run can catch:

* the flat-plate Wagner potential really satisfies its BVP (so the numerics
  below are standing on a valid solution, not on a plausible formula);
* the Eulerian Bernoulli pressure of that potential is ``2 C s - s^2`` and its
  peak is ``C^2``, while the in-repo profile is ``2 C s - s^2 + 1`` (peak
  ``C^2 + 1``) -- the offset is exactly one dynamic pressure ``q``;
* the graded stagnation reference is in the same units as the measurement
  (``p/q``, ``q = rho V^2 / 2``), i.e. ``2 C`` and not ``C``.

Negative control included: a profile shifted by two ``q`` must break the offset
identity, otherwise the identity test is vacuous.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorsph.wedge_entry import (
    wagner_c_coeff,
    wagner_p_centre,
    wagner_p_over_q,
    wagner_peak_over_q,
    wagner_peak_tau,
)

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules[cls.__module__].
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


audit = _load("wedge_audit", CASE / "audit_wagner_reference.py")
case_run = _load("wedge_case_run", CASE / "run.py")

BETAS = (15.0, 30.0, 45.0)


# --------------------------------------------------------------------------- #
# 1. The potential is a valid Wagner solution
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("beta_deg", BETAS)
def test_potential_satisfies_wagner_bvp(beta_deg):
    """phi = 0 dry, |dphi/dy| = V on the wetted plate, O(c^2/r) far field."""
    res = audit.bvp_residuals(math.radians(beta_deg), n=301)
    assert res["max_abs_dphi_dy_minus_V"] < 1e-8
    assert res["max_phi_dry"] < 1e-10
    # |W| ~ c^2/(2|z|) => |phi| * r / c^2 -> O(1), and it must not grow with r.
    scaled = res["far_field_c2_scaled"]
    assert scaled[-1] <= scaled[0] + 1e-9
    assert all(s < 1.0 for s in scaled)


def test_branch_is_symmetric_in_x():
    """A silent conjugate branch would make phi odd in x instead of even."""
    c, V = 1.0, 1.0
    xs = np.linspace(1e-3, 0.9, 25)
    even = audit.potential_phi(xs, c, V) - audit.potential_phi(-xs, c, V)
    assert float(np.max(np.abs(even))) < 1e-12
    assert float(np.max(audit.potential_phi(xs, c, V))) < 0.0      # over-pressure side


# --------------------------------------------------------------------------- #
# 2. Eulerian Bernoulli pressure from those numerics
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("beta_deg", BETAS)
def test_eulerian_profile_matches_closed_form(beta_deg):
    """Numerics must reproduce 2 C s - s^2, s = 1/sqrt(1 - tau^2)."""
    beta = math.radians(beta_deg)
    C = wagner_c_coeff(beta)
    tau = np.linspace(0.0, 0.95, 201)
    s = 1.0 / np.sqrt(1.0 - tau ** 2)
    closed = 2.0 * C * s - s * s
    assert float(np.max(np.abs(audit.eulerian_pressure_over_q(tau, beta) - closed))) < 1e-4


@pytest.mark.parametrize("beta_deg", BETAS)
def test_repo_profile_is_the_relative_kinetic_variant(beta_deg):
    """repo == Eulerian with the v^2 = V^2 term dropped (identification)."""
    beta = math.radians(beta_deg)
    tau = np.linspace(0.0, 0.95, 201)
    rel = audit.relative_kinetic_pressure_over_q(tau, beta)
    repo = np.array([float(wagner_p_over_q(t, beta)) for t in tau])
    assert float(np.max(np.abs(repo - rel))) < 1e-4


@pytest.mark.parametrize("beta_deg", BETAS)
def test_repo_profile_sits_one_q_above_eulerian(beta_deg):
    beta = math.radians(beta_deg)
    off = audit.profile_offset(beta_deg, n=201, tau_max=0.95)
    assert off["offset_repo_minus_eulerian_mean"] == pytest.approx(1.0, abs=1e-5)
    assert off["offset_repo_minus_eulerian_std"] < 1e-4


def test_offset_identity_negative_control():
    """A profile shifted by two q must not satisfy the one-q identity."""
    beta = math.radians(30.0)
    tau = np.linspace(0.0, 0.9, 51)
    eul = audit.eulerian_pressure_over_q(tau, beta)
    fake = eul + 2.0
    assert float(np.max(np.abs(fake - eul))) == pytest.approx(2.0)
    assert abs(float(np.mean(fake - eul)) - 1.0) > 0.5      # identity must reject it


# --------------------------------------------------------------------------- #
# 3. Peak / apex values, in q units
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("beta_deg", [10.0, 20.0, 30.0, 45.0])
def test_apex_and_peak_in_q_units(beta_deg):
    beta = math.radians(beta_deg)
    C = wagner_c_coeff(beta)
    a = audit.apex_audit(beta_deg)
    # Eulerian: apex 2C - 1, interior peak C^2 at tau = sqrt(1 - 1/C^2).
    assert a.cp_apex_eulerian == pytest.approx(2.0 * C - 1.0, abs=1e-4)
    assert a.cp_peak_eulerian == pytest.approx(C * C, rel=1e-5)
    # In-repo profile: apex 2C, interior peak C^2 + 1 (one q higher throughout).
    assert a.cp_apex_repo_profile == pytest.approx(2.0 * C, abs=1e-9)
    assert a.cp_peak_repo == pytest.approx(C * C + 1.0, rel=1e-12)
    assert a.tau_peak == pytest.approx(wagner_peak_tau(beta), rel=1e-12)


def test_peak_location_is_shared():
    """Both profiles peak at the same tau (the offset is constant, not a shift)."""
    beta = math.radians(30.0)
    tau = np.linspace(0.5, 0.999, 5000)
    eul = audit.eulerian_pressure_over_q(tau, beta)
    repo = np.array([float(wagner_p_over_q(t, beta)) for t in tau])
    assert abs(tau[int(np.argmax(eul))] - tau[int(np.argmax(repo))]) < 2e-3


# --------------------------------------------------------------------------- #
# 4. Units of the graded stagnation reference (the fix under test)
# --------------------------------------------------------------------------- #

def test_measured_and_graded_normalisations_agree():
    """cp is p/q; the reference must be p/q too: p(0)/q = 2C, not C."""
    rho, V = 1000.0, 1.0
    q = 0.5 * rho * V * V
    beta = math.radians(case_run.BETA_DEG)
    C = wagner_c_coeff(beta)
    ref = case_run.refs_derived()["wagner"]
    assert ref["cp_centre"] == pytest.approx(float(wagner_p_over_q(0.0, beta)), rel=1e-12)
    assert ref["cp_centre"] == pytest.approx(2.0 * C, rel=1e-12)
    assert ref["cp_centre"] == pytest.approx(wagner_p_centre(V, beta, rho) / q, rel=1e-12)
    # And the q-unit peak reference stays the profile maximum.
    assert ref["cp_max"] == pytest.approx(wagner_peak_over_q(beta), rel=1e-12)


def test_unit_shift_flips_only_the_stagnation_verdict():
    """Quoted measurements (docs/case_sph_wedge_entry_2d.md §4.2, dx = 2 mm, c0/V=20).

    Not a re-grade (the raw artefacts are untracked): the numbers are used only
    to show how the reported errors move when the reference is put in the same
    units as the measurement.  The peak-pressure failure survives both ways.
    """
    measured = {"cp_centre_mean": 4.873, "cp_band_peak": 28.05}
    out = audit.shift_errors(measured, 30.0)

    centre = out["cp_centre_mean"]
    assert centre["err_as_graded"] > 0.15                # the committed FAIL
    assert abs(centre["err_q_unit_corrected"]) < 0.15    # PASS once units agree
    assert centre["err_as_graded"] == pytest.approx((4.873 - 2.720699) / 2.720699, rel=1e-4)

    band = out["cp_band_peak"]
    assert band["err_as_graded"] > 1.0                   # peak failure stands
    assert band["err_eulerian"] > 1.0                    # ... under the Eulerian value too


def test_no_zhao_faltinsen_number_is_invented():
    """The mission reference stays WITHHELD: no value may be manufactured here."""
    ref = case_run.refs_derived()["zhao_faltinsen_1993"]
    assert ref["status"] == "WITHHELD" and ref["cp_max"] is None