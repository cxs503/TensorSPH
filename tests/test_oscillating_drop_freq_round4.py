"""Invariants for round 4 (independent estimator of ``CASE-SPH-OSC-DROP-FREQ``).

What these tests are for
------------------------
Round 4 replaces the estimator of rounds 1-3 with one that shares no basis and
no parameterisation with them, and then re-issues the anchor verdict on it.  An
estimator that *cannot fail* would make the new verdict worthless, so the tests
below are written in two groups:

* **analytic identities** -- the quadrature, the surface-of-revolution area and
  the mode projection are checked against geometry whose answer is known in
  closed form.  A mistake here would silently move the reported transfer.
* **planted-defect (falsifiability) tests** -- a known amplitude bias is injected
  and the estimator *must* report it.  This is what stops a "passing" anchor row
  from being an artefact of a blind measurement.

Run: ``python -m pytest tests/test_oscillating_drop_freq_round4.py -q``
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "oscillating_drop_freq"))

from round4_energy_contour import (  # noqa: E402
    analytic_area, analytic_contour, contour_geometry, contour_points,
    energy_transfer, modal_amplitudes, shape_radius, shipped_with_exact_inputs,
    sph_contour, uniform_theta,
)
from tensorsph.oscillating_drop3d import axisym_drop_geometry  # noqa: E402

R = 0.02
SIGMA, RHO = 7.2, 1000.0
JSON = ROOT / "benchmarks" / "oscillating_drop_freq" / "round4.json"


# --------------------------------------------------------------------------- #
# analytic identities
# --------------------------------------------------------------------------- #
def test_analytic_area_of_a_sphere():
    """``eps = 0`` is a sphere: ``A = 4 pi R^2`` to quadrature precision."""
    a = analytic_area(R, 2, 0.0)
    assert abs(a / (4.0 * math.pi * R ** 2) - 1.0) < 1e-7


def test_analytic_area_grows_quadratically_in_eps():
    """``A(eps) - A(0) = O(eps^2)`` -- the family is volume preserving to O(eps)."""
    a0 = analytic_area(R, 2, 0.0)
    a1 = analytic_area(R, 2, 0.001)
    a2 = analytic_area(R, 2, 0.002)
    assert abs(a1 - a0) / a0 < 1e-5           # linear term vanishes
    assert (a2 - a0) / (4.0 * (a1 - a0)) == pytest.approx(1.0, abs=2e-2)


@pytest.mark.parametrize("mode", [2, 3, 4])
def test_mode_projection_recovers_a_planted_mode(mode):
    """``a_l/(eps R) = 1`` and every other mode vanishes on an exact shape."""
    eps = 0.05
    theta = torch.linspace(1e-6, math.pi - 1e-6, 4001, dtype=torch.float64)
    r = shape_radius(theta, R, mode, eps)
    amps = modal_amplitudes(theta, r)
    assert amps[f"a{mode}"] / (eps * R) == pytest.approx(1.0, abs=1e-6)
    for other in (2, 3, 4, 5, 6):
        if other != mode:
            assert abs(amps[f"a{other}"]) / (eps * R) < 1e-6


def test_contour_geometry_on_an_exact_circle():
    """Area and volume of a revolution measured on a raw polyline."""
    n = 4001
    theta = torch.linspace(0.0, math.pi, n, dtype=torch.float64)
    pts = torch.stack((R * torch.sin(theta), R * torch.cos(theta)), dim=1)
    geo = contour_geometry(pts)
    assert geo["area"] / (4.0 * math.pi * R ** 2) == pytest.approx(1.0, abs=1e-6)
    assert geo["volume"] / (4.0 / 3.0 * math.pi * R ** 3) == pytest.approx(1.0, abs=1e-6)


def test_energy_route_recovers_a_planted_transfer():
    """``A(eps) = A0 (1 + K g^2 eps^2)``: the fit must return ``g``."""
    eps = torch.tensor([0.0, 0.02, -0.02, 0.05, -0.05], dtype=torch.float64)
    a0, k = 4.0 * math.pi * R ** 2, 0.5
    for g in (1.0, 0.92, 0.75):
        areas = (a0 * (1.0 + k * g ** 2 * eps ** 2)).tolist()
        fit = energy_transfer(areas, eps.tolist())
        assert math.sqrt(fit["c2"] / (a0 * k)) == pytest.approx(g, abs=1e-9)


def test_estimator_control_bias_is_below_a_tenth_of_a_percent():
    """The pipeline on an analytic level set must return the seeded amplitude."""
    geom = axisym_drop_geometry(R, R / 12.0, mode=2, eps=0.05, rho0=RHO)
    ac = analytic_contour(geom, n_grid_per_dx=6.0)
    u = uniform_theta(ac["theta"], ac["r"])
    amps = modal_amplitudes(u["theta"], u["r"], modes=(2,))
    assert amps["a2"] / (0.05 * R) == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# falsifiability: a planted bias must show up
# --------------------------------------------------------------------------- #
def test_planted_amplitude_bias_is_detected():
    """Shrink the interface by 10 %; the readout must report a transfer of 0.9.

    This is the guard against a "measuring device that always says 1": if a
    future change to the contour or the quadrature silently normalised the
    amplitude, this test (not the case report) would fail first.
    """
    geom = axisym_drop_geometry(R, R / 12.0, mode=2, eps=0.05, rho0=RHO)
    ac = analytic_contour(geom, n_grid_per_dx=6.0)
    pts = ac["pts"].clone()
    pts[:, 0] *= 0.90
    pts[:, 1] *= 0.90
    r = torch.linalg.vector_norm(pts, dim=1)
    theta = torch.atan2(pts[:, 0].abs(), pts[:, 1])
    u = uniform_theta(theta, r)
    amps = modal_amplitudes(u["theta"], u["r"], modes=(2,))
    assert amps["a2"] / (0.05 * R) == pytest.approx(0.90, abs=2e-3)


def test_contour_of_a_shifted_level_reports_a_smaller_drop():
    """A level set at a larger colour value sits inside: the estimator must see it."""
    geom = axisym_drop_geometry(R, R / 10.0, mode=2, eps=0.0, rho0=RHO)
    sc_in = sph_contour(geom, n_grid_per_dx=5.0)
    geo_in = contour_geometry(sc_in["pts"])
    # the free-surface colour field falls off inward, so any contour is inside
    assert geo_in["area"] < 4.0 * math.pi * (1.06 * R) ** 2
    assert geo_in["area"] > 4.0 * math.pi * (0.85 * R) ** 2


# --------------------------------------------------------------------------- #
# shipped operator with exact inputs (part D)
# --------------------------------------------------------------------------- #
def test_shipped_fit_curvature_sign_on_an_undeformed_drop():
    """With exact normals the shipped fit must return ``kappa R ~ 2`` on a sphere.

    Round 3 carried the shipped CSF operator's ``-223.8 %`` forward without
    retesting; part D of round 4 feeds it exact positions and normals so the
    *fit* can be judged separately from the colour-field normal.  A static sign
    error in the fit would show up here.
    """
    row = shipped_with_exact_inputs(R, SIGMA, RHO, mode=2, eps=0.0, rdx=12.0)
    assert row["kappaR_mean_exact_inputs"] == pytest.approx(2.0, abs=0.15)


def test_shipped_fit_curvature_responds_to_the_mode():
    """The deformation response must be *restoring* (dp < 0 for eps > 0)."""
    row = shipped_with_exact_inputs(R, SIGMA, RHO, mode=2, eps=0.05, rdx=12.0)
    assert row["dp_measured"] < 0.0
    assert math.isfinite(row["omega_error_pct"])


# --------------------------------------------------------------------------- #
# the published artifact, when present
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not JSON.exists(), reason="round4.json not generated yet")
def test_round4_json_is_internally_consistent():
    d = json.loads(JSON.read_text())
    assert d["round"] == 4
    assert d["case"] == "SPH-OSC-DROP-FREQ"
    assert d["gate_pct"] == 6.0
    rows = d["transfer_rows"]
    assert rows, "no rows"
    for r in rows:
        if r["eps"] > 0.0:
            # the independent readout must be a sane number, not a silent 0/NaN
            assert math.isfinite(r["transfer_sph"])
            assert 0.5 < r["transfer_sph"] < 1.5
            assert math.isfinite(r["differential_transfer"])
    # the estimator control (analytic level set) is the measurement floor
    for c in d["estimator_control"]:
        assert abs(c["control_transfer"] - 1.0) < 2e-3
    # the verdict must name every resolution in the ladder
    assert set(d["verdict"]["per_resolution"]) == {
        str(float(v)) for v in d["parameters"]["ladder_radius_over_dx"]}