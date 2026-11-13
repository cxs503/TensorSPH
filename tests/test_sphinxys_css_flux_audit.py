"""Regression locks for the CSS surface-tension flux-reading audit.

Anchor ``XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION``.  See
``docs/verification_sphinxys_css_flux_audit.md`` for the reasoning; this module
pins the four facts the audit rests on:

1. the conservative pair divergence is a sound operator (O1),
2. the "flux reading" reported by m17909 is dominated by the particle mass
   ``m_i = rho0 dx^2`` (O2/O3),
3. the normalised reading is 27-31 % low - it does **not** meet the 15 % gate
   (documented honest FAIL, locked so a future fix cannot silently regress),
4. an idealised radial CSS stress field produces no flux through that operator
   (O5).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "css_flux_audit", REPO / "benchmarks" / "surface_tension" / "css_flux_audit.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

LEVELS = (16, 32)


@pytest.fixture(scope="module")
def calibration():
    return audit.operator_calibration()


@pytest.fixture(scope="module")
def rows():
    return {res: audit.droplet_controls(res) for res in LEVELS}


def test_operator_calibration_is_sound(calibration):
    """O1: div(S = x^2 I) must be 2x within truncation error."""
    assert calibration["worst_err_pct"] < 5.0, calibration
    for probe in calibration["probes"]:
        if probe["x"] == 0.0:
            assert abs(probe["pair_div"]) < 1e-12
    # monotone as the probe moves away from the axis (no sign flip)
    xs = [p for p in calibration["probes"] if p["x"] > 0]
    assert all(p["pair_div"] > 0 for p in xs)


def test_reported_reading_is_dominated_by_particle_mass(rows):
    """O2/O3: the reported deficit is the particle mass, not a cancellation."""
    for res, row in rows.items():
        mass = row["particle_mass_rho0_vol"]
        # the reported reading must scale like m_i = rho0 dx^2
        assert row["dp_ported_over_mass_ratio"] == pytest.approx(
            row["dp_ported_pa"] / (mass * row["theory_pa"]), rel=1e-12)
        assert 0.60 < row["dp_ported_over_mass_ratio"] < 0.85, (res, row)
    # and the ratio must be resolution independent while m_i is not
    ratios = [rows[r]["dp_ported_over_mass_ratio"] for r in LEVELS]
    masses = [rows[r]["particle_mass_rho0_vol"] for r in LEVELS]
    assert max(masses) / min(masses) > 3.0            # m_i really varies
    assert max(ratios) - min(ratios) < 0.10           # the ratio does not


def test_normalised_reading_is_not_within_gate(rows):
    """Documented honest FAIL: 27-31 % low, so the 15 % gate is not met.

    If a future fix makes the flux reading converge, this test must be updated
    together with the audit document - that is the point of locking it.
    """
    for res, row in rows.items():
        err = row["dp_normalised_err_pct"]
        assert -40.0 < err < -20.0, (res, err)
        assert abs(err) > 15.0, (res, err)


def test_idealised_css_field_carries_no_flux(rows):
    """O5: the exact radial tangential stress ansatz yields ~0 through the operator."""
    for res, row in rows.items():
        assert row["ideal_css_field_worst_ratio"] < 1e-3, (res, row["ideal_css_field_worst_ratio"])
        assert len(row["ideal_css_field_flux"]) == 12


def test_delta_function_normalisation(rows):
    """The ported |grad c| integrates to ~1, so the stress field itself is right."""
    for res, row in rows.items():
        assert 0.90 < row["delta_integral"] < 1.05, (res, row["delta_integral"])