"""Lamb-mode (dynamic) cross-validation of the ported SPHinXsys CSS surface tension.

Anchor: ``XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION``.

The static droplet audit could not decide whether the ported CSS *force assembly*
is blind to curvature (a lattice-symmetry cancellation) or merely mis-normalised:
``sphinxys_xval.py`` reported -99.8 %, ``css_flux_audit.py`` -30 % after removing
the ``m_i = rho0 dx^2`` bookkeeping factor, while its own synthetic "ideal CSS
field" control returned ~0.  A *perturbed* interface breaks the lattice symmetry
and answers the question in a physical unit, the Lamb-mode frequency

    omega_n^2 = n (n^2 - 1) sigma / (rho R^3),   n = 2

which is measured here with the same projector for both models:

    dp = 2/L * sum_i (f_i . e_r) V_i cos(n theta_i),   L = 2 pi R.

The Brackbill CSF model (already validated against the same law,
``TOL_LAMB_PCT = 3`` in ``surface_tension_dispersion``) calibrates the projector;
the SPHinXsys port is the subject.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "surface_tension"))

import sphinxys_lamb as L  # noqa: E402

SIGMA = L.SIGMA
RHO0 = L.RHO0
RADIUS = L.RADIUS
GATE_PCT = 15.0          # owner accuracy standard: > 15 % is unacceptable


# --------------------------------------------------------------------------- #
# geometry / projector
# --------------------------------------------------------------------------- #
def test_lobe_geometry_area_and_normals():
    """The lattice must reproduce the perturbed disc area before any physics."""
    mode, eps, radius, dx = 2, 0.10, 0.05, 0.05 / 16
    g = L.lobe_geometry(mode, eps, radius, dx)
    area = float(g["vol"].sum())
    # exact area of r(theta) = R(1 + eps cos n theta)
    exact = math.pi * radius ** 2 * (1.0 + 0.5 * eps ** 2)
    assert abs(area - exact) / exact < 0.02, (area, exact)
    # particle count matches vol = dx^2
    assert abs(float(g["mass"].sum()) / RHO0 - area) < 1e-12


def test_two_phase_geometry_phases_are_disjoint_and_complete():
    mode, eps, radius, dx = 2, 0.10, 0.05, 0.05 / 16
    g = L.two_phase_lobe_geometry(mode, eps, radius, dx, rings=4)
    phase, r, r_surf = g["phase"], g["r"], radius * (1.0 + eps * torch.cos(mode * g["theta"]))
    assert bool(((phase == 0) == (r <= r_surf + 1e-12)).all())
    assert int((phase == 0).sum()) > 100
    assert int((phase == 1).sum()) > 100
    water_area = float(g["vol"][phase == 0].sum())
    exact = math.pi * radius ** 2 * (1.0 + 0.5 * eps ** 2)
    assert abs(water_area - exact) / exact < 0.02


def test_modal_pressure_is_linear_in_amplitude():
    """The projector is linear, so a doubled force gives a doubled reading."""
    mode, eps, radius, dx = 2, 0.10, 0.05, 0.05 / 16
    g = L.lobe_geometry(mode, eps, radius, dx)
    r = torch.linalg.vector_norm(g["pos"], dim=1)
    e_r = g["pos"] / r[:, None]
    f = -torch.cos(mode * g["theta"])[:, None] * e_r
    one = L.modal_pressure(g["pos"], g["theta"], f, g["vol"], mode, radius)
    two = L.modal_pressure(g["pos"], g["theta"], 2.0 * f, g["vol"], mode, radius)
    assert abs(two["dp_measured"] - 2.0 * one["dp_measured"]) < 1e-12


# --------------------------------------------------------------------------- #
# calibration channel: the projector must reproduce the known CSF Lamb result
# --------------------------------------------------------------------------- #
def test_csf_calibration_reproduces_lamb_within_gate():
    row = L.csf_lamb(2, 0.10, RADIUS, RADIUS / 16)
    assert abs(row["omega_error_pct"]) < L.TOL_LAMB_PCT, row


# --------------------------------------------------------------------------- #
# the subject: the ported CSS force
# --------------------------------------------------------------------------- #
def test_css_force_reproduces_lamb_frequency_without_hourglass():
    """The ported stress *flux* is not blind: on a perturbed interface it gives
    the Lamb frequency to the accuracy of the CSF control (gate < 15 %)."""
    g = L.lobe_geometry(2, 0.10, RADIUS, RADIUS / 16)
    row = L.css_single_phase(g, hourglass=False)
    assert abs(row["omega_error_pct"]) < GATE_PCT, row
    # the physical force must be restoring (positive stiffness ratio)
    assert row["stiffness_ratio"] > 0.5, row


def test_css_bookkeeping_identity_upstream_vs_consistent():
    """``a = V div S`` (upstream) and ``a = (1/rho) div S`` (consistent) must give
    the same force density once the matching factor is applied -- this is the
    identity that the -99.8 % static reading was missing."""
    g = L.lobe_geometry(2, 0.10, RADIUS, RADIUS / 16)
    up = L.css_single_phase(g, hourglass=False, assembly="upstream")
    co = L.css_single_phase(g, hourglass=False, assembly="consistent")
    assert abs(up["dp_measured_pa"] - co["dp_measured_pa"]) < 1e-10 * max(
        1.0, abs(up["dp_measured_pa"]))


def test_two_phase_mixture_lamb_frequency_within_gate():
    """The mixture path (contact fraction 1000:1, contact stress flux) must also
    reproduce the Lamb frequency."""
    dx = RADIUS / 16
    g = L.two_phase_lobe_geometry(2, 0.10, RADIUS, dx, rings=6)
    row = L.css_two_phase(g, hourglass=True)
    assert abs(row["omega_error_pct"]) < GATE_PCT, row
    # contact fraction of the water phase in a 1000:1 pair
    assert abs(row["contact_fraction_water"] - 1000.0 / 1001.0) < 1e-12


@pytest.mark.xfail(strict=True, reason="hourglass term injects a spurious mode-2 pressure "
                                       "of order sigma/R on a circle (documented defect)")
def test_hourglass_contribution_vanishes_with_the_perturbation():
    """The hourglass penalty is a *lattice* term and must not act as a physical
    restoring force.  Two independent statements are pinned here:

    * its contribution is not proportional to the perturbation (a physical
      surface-tension term scales exactly with ``eps``);
    * on a perfect circle (``eps = 0``) the mode-2 modal pressure of a physical
      force is zero by symmetry, so anything the port returns there is spurious.

    Locked as ``xfail(strict)``-style evidence in
    ``benchmarks/surface_tension/out_sphinxys_lamb/lamb.json`` and
    ``docs/xval_sphinxys_surface_tension/LAMB.md``; the assertion below states
    what *should* hold, so it will start passing -- and this test will fail --
    as soon as the hourglass term is rescaled or gated to the contact path.
    """
    dx = RADIUS / 16
    circle = L.lobe_geometry(2, 0.0, RADIUS, dx)
    dp_circle = L.css_single_phase(circle, hourglass=True)["dp_measured_pa"]
    spurious = abs(dp_circle) * RADIUS / SIGMA
    assert spurious < 0.15, (
        f"hourglass injects a spurious mode-2 pressure of {spurious:.2f} sigma/R "
        "on an unperturbed circle")




def test_css_hourglass_off_matches_lamb_on_two_resolutions():
    """Headline claim: the ported force (hourglass off) is inside the 15 % gate at
    two resolutions, so the result is not a single-level coincidence."""
    for ppd in (16, 24):
        g = L.lobe_geometry(2, 0.10, RADIUS, RADIUS / ppd)
        row = L.css_single_phase(g, hourglass=False)
        assert abs(row["omega_error_pct"]) < GATE_PCT, (ppd, row)


def test_css_hourglass_off_vanishes_on_a_circle():
    """On a perfect circle the mode-2 modal pressure of a physical force is zero.
    The ported CSS force (hourglass off) satisfies this; the hourglass term does
    not (see the strict-xfail test above)."""
    g = L.lobe_geometry(2, 0.0, RADIUS, RADIUS / 16)
    row = L.css_single_phase(g, hourglass=False)
    spurious = abs(row["dp_measured_pa"]) * RADIUS / SIGMA
    assert spurious < 0.1, (spurious, row)


@pytest.mark.parametrize("eps", [0.02])
@pytest.mark.xfail(strict=True, reason="the inner hourglass part is O(sigma/R) and does not scale "
                                       "with the perturbation (documented defect)")
def test_hourglass_part_scales_with_the_perturbation(eps):
    """A physical surface-tension term is proportional to ``eps``; the hourglass
    part must shrink by the same factor between eps = 0.10 and eps = 0.02."""
    dx = RADIUS / 16

    def dp_hg(e):
        g = L.lobe_geometry(2, e, RADIUS, dx)
        return (L.css_single_phase(g, hourglass=True)["dp_measured_pa"]
                - L.css_single_phase(g, hourglass=False)["dp_measured_pa"])

    assert abs(dp_hg(eps)) < 0.3 * abs(dp_hg(0.10)), (abs(dp_hg(eps)), abs(dp_hg(0.10)))
