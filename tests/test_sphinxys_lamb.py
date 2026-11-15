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


def test_hourglass_scope_contact_removes_the_spurious_circle_force():
    """The inner hourglass penalty is upstream code for *phase* mixing, and on a
    vacuum-reduced single-phase interface it returns an O(sigma/R) modal force
    that does not scale with the perturbation.  ``hourglass_scope="contact"``
    drops that inner penalty; on a circle (``eps = 0``), where the mode-2 modal
    pressure of any physical force is zero by symmetry, nothing is left."""
    for ppd in (16, 24):
        circle = L.lobe_geometry(2, 0.0, RADIUS, RADIUS / ppd)
        row = L.css_single_phase(circle, hourglass=True, hourglass_scope="contact")
        spurious = abs(row["dp_measured_pa"]) * RADIUS / SIGMA
        assert spurious < 0.15, (ppd, spurious, row)


def test_hourglass_scope_contact_is_identical_to_hourglass_off_on_a_free_surface():
    """With no second phase there is no contact loop, so the scoped penalty must
    reduce exactly to the physical (hourglass off) flux."""
    g = L.lobe_geometry(2, 0.10, RADIUS, RADIUS / 24)
    on = L.css_single_phase(g, hourglass=True, hourglass_scope="contact")
    off = L.css_single_phase(g, hourglass=False)
    assert abs(on["dp_measured_pa"] - off["dp_measured_pa"]) < 1e-12 * max(
        1.0, abs(off["dp_measured_pa"]))


def test_hourglass_scope_contact_passes_the_lamb_gate_at_every_level():
    """The fix is a *setting*, not a tuned coefficient: with the inner penalty
    dropped the ported force reproduces the Lamb frequency inside the 15 % gate
    on the whole resolution ladder (R/dx = 16, 24, 32)."""
    for ppd in (16, 24, 32):
        g = L.lobe_geometry(2, 0.10, RADIUS, RADIUS / ppd)
        row = L.css_single_phase(g, hourglass=True, hourglass_scope="contact")
        assert abs(row["omega_error_pct"]) < GATE_PCT, (ppd, row)


def test_hourglass_scope_contact_is_inert_on_the_two_phase_path():
    """The contact (cross-phase) hourglass penalty is the one that upstream
    actually uses -- it is suppressed by ``4 f (1-f)`` -- so restricting the
    scope must leave the two-phase answer unchanged."""
    dx = RADIUS / 16
    g = L.two_phase_lobe_geometry(2, 0.10, RADIUS, dx, rings=6)
    all_ = L.css_two_phase(g, hourglass=True, hourglass_scope="all")
    contact = L.css_two_phase(g, hourglass=True, hourglass_scope="contact")
    assert abs(all_["omega_ratio"] - contact["omega_ratio"]) < 5e-3, (all_, contact)
    assert abs(contact["omega_error_pct"]) < GATE_PCT, contact


def test_upstream_inner_hourglass_defect_stays_pinned():
    """Regression pin on the *literal* upstream transcription (scope="all"): the
    spurious mode-2 pressure on a circle is ~2 sigma/R, which is what
    ``docs/xval_sphinxys_surface_tension/LAMB.md`` reports.  If the ported inner
    hourglass is ever rewritten, this test must be updated together with the
    document -- the record must not drift silently."""
    circle = L.lobe_geometry(2, 0.0, RADIUS, RADIUS / 24)
    row = L.css_single_phase(circle, hourglass=True)
    spurious = abs(row["dp_measured_pa"]) * RADIUS / SIGMA
    assert 1.5 < spurious < 3.0, (spurious, row)


def test_invalid_hourglass_scope_is_rejected():
    g = L.lobe_geometry(2, 0.10, RADIUS, RADIUS / 16)
    with pytest.raises(ValueError):
        L.css_single_phase(g, hourglass=True, hourglass_scope="inner")


def test_css_hourglass_off_matches_lamb_on_two_resolutions():
    """Headline claim: the ported force (hourglass off) is inside the 15 % gate at
    two resolutions, so the result is not a single-level coincidence."""
    for ppd in (16, 24):
        g = L.lobe_geometry(2, 0.10, RADIUS, RADIUS / ppd)
        row = L.css_single_phase(g, hourglass=False)
        assert abs(row["omega_error_pct"]) < GATE_PCT, (ppd, row)


def test_css_hourglass_off_vanishes_on_a_circle():
    """On a perfect circle the mode-2 modal pressure of a physical force is zero.
    The ported CSS force (hourglass off) satisfies this, and so does the fixed
    setting ``hourglass_scope="contact"``; the literal upstream inner penalty
    does not (``test_upstream_inner_hourglass_defect_stays_pinned``)."""
    g = L.lobe_geometry(2, 0.0, RADIUS, RADIUS / 16)
    row = L.css_single_phase(g, hourglass=False)
    spurious = abs(row["dp_measured_pa"]) * RADIUS / SIGMA
    assert spurious < 0.1, (spurious, row)


@pytest.mark.parametrize("eps", [0.02, 0.05, 0.10])
def test_hourglass_scope_contact_passes_the_gate_across_amplitudes(eps):
    """The setting is not tuned to one perturbation: the ported modal force keeps
    the Lamb frequency inside the 15 % gate over an order of magnitude in ``eps``
    (linear theory is only meaningful for small ``eps``; the eps-scan of
    ``lamb.json`` shows the modal force growing with it)."""
    g = L.lobe_geometry(2, eps, RADIUS, RADIUS / 16)
    row = L.css_single_phase(g, hourglass=True, hourglass_scope="contact")
    assert abs(row["omega_error_pct"]) < GATE_PCT, (eps, row)
