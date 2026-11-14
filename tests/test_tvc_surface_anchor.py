"""Free-surface and wall branches of the TVC operator against analytic anchors.

The operator (``tensorsph.transport_velocity_correction.kernel_gradient_integral``,
ported from SPHinXsys ``TransportVelocityCorrection``) had only *invariant*
coverage until now: a perfect lattice is a fixed point, so ``g = 0`` and there
was no non-zero reference to grade against.  These tests supply references:

* plane free surface -> gradient theorem, ``g = +2 n I(h, d)`` with
  ``I(h, d) = int W_2D(s, d) ds`` and ``I(h, 0) = 15 / (7 pi h)`` closed form;
* wall contact -> mirror-image ghost lattice, where fluid + wall is the
  full-space sum and must vanish **exactly** iff ``wall_factor = 2``.

Measured numbers and their honest gaps are recorded in
``docs/xval_tvc_surface_anchor.md``; the gates below are the ones that report
holds itself to.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "xval_sphinx_tvc"))

from surface_term import (  # noqa: E402
    check_normalisation,
    closed_form_I0,
    line_integral,
    measure,
    wall_image_case,
)

HBAR = 1.3  # production h / dx of this repo
H = 0.026


# --------------------------------------------------------------------------- #
# the anchor itself must be right before it grades anything
# --------------------------------------------------------------------------- #
def test_kernel_profile_is_normalised():
    """int W dA = 1 -- otherwise every anchor below is scaled wrong."""
    assert check_normalisation(H, n=12001) == pytest.approx(1.0, abs=1e-6)


def test_line_integral_matches_closed_form():
    """I(h, 0) = 15 / (7 pi h) exactly; quadrature must reproduce it."""
    assert line_integral(H, 0.0) == pytest.approx(closed_form_I0(H), rel=1e-9)


def test_line_integral_decays_to_zero_at_support_edge():
    assert line_integral(H, 2.0 * H) == pytest.approx(0.0, abs=1e-12)
    assert line_integral(H, 0.5 * H) < line_integral(H, 0.0)


# --------------------------------------------------------------------------- #
# free surface: direction, magnitude, convergence
# --------------------------------------------------------------------------- #
def test_surface_term_points_outward_with_no_tangential_component():
    row = measure(H, 8.0, 0)
    assert abs(row["g_x1"]) < 1e-12 * row["g_mag"], "tangential leak on a symmetric surface"
    assert row["g_x2"] > 0.0, "correction must point out of the fluid, not into it"


@pytest.mark.parametrize("h_over_dx", [2.0, 4.0, 8.0])
def test_surface_term_strict_gate_at_refined_lattices(h_over_dx):
    """h/dx >= 4 resolves the surface term to better than 3 %."""
    row = measure(H, h_over_dx, 0)
    gate = 0.03 if h_over_dx >= 4.0 else 0.15
    assert row["rel_err"] < gate, row


def test_surface_term_at_production_resolution_is_within_owner_gate():
    """h/dx = 1.3 (every production run here) is a *coarse* lattice: ~14 %.

    The number is asserted in a band so the test reports the honest size of the
    error instead of silently accepting any value below the gate.
    """
    row = measure(H, HBAR, 0)
    assert 0.05 < row["rel_err"] < 0.15, row
    assert row["rel_err"] < 0.15, "owner gate (15 %)"


def test_surface_term_converges_second_order_in_dx():
    coarse = measure(H, 2.0, 0)["rel_err"]
    fine = measure(H, 8.0, 0)["rel_err"]
    order = math.log(coarse / fine) / math.log(8.0 / 2.0)
    assert 1.5 < order < 2.5, f"observed order {order:.2f}"


def test_linked_cell_sum_equals_independent_brute_force():
    """The measurement must not be an artefact of the neighbour search."""
    for n in (HBAR, 8.0):
        row = measure(H, n, 0)
        assert row["dense_vs_linked_rel"] < 1e-12, row


# --------------------------------------------------------------------------- #
# depth decay: the operator's spatial structure, not just its surface value
# --------------------------------------------------------------------------- #
def test_depth_decay_follows_the_analytic_line_integral():
    rows = [measure(H, 8.0, m) for m in (0, 1, 2, 4)]
    for row in rows:  # d <= 0.5 h: the near-surface region that does the work
        assert row["rel_err"] < 0.10, row
    mags = [row["g_mag"] for row in rows]
    assert mags == sorted(mags, reverse=True), "term must decay with depth"


def test_depth_decay_reaches_zero_at_the_support_edge():
    row = measure(H, 8.0, 16)  # depth = 2 h
    assert row["g_mag"] < 1e-10 * measure(H, 8.0, 0)["g_mag"]


# --------------------------------------------------------------------------- #
# wall contact: exact mirror ghost anchor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("h_over_dx", [HBAR, 4.0])
def test_wall_factor_two_is_the_only_value_that_cancels_the_mirror_ghost(h_over_dx):
    """Ghost-fluid wall: fluid + wall weighting must be equal -> factor 2."""
    ok = wall_image_case(H, h_over_dx, wall_factor=2.0)
    bad = wall_image_case(H, h_over_dx, wall_factor=1.0)
    # exact anchor: vanishes to round-off, ~1e-16 of the surface scale
    assert ok["g_mag"] < 1e-12 * ok["surface_scale"], ok
    # and the test is discriminating: a wrong factor is O(50 %) of the scale
    assert bad["g_mag"] > 0.4 * bad["surface_scale"], bad
    assert ok["surface_scale"] / ok["g_mag"] > 1e10, "anchor does not discriminate"


# --------------------------------------------------------------------------- #
# consequence: the per-application shift is resolution independent in units of h
# --------------------------------------------------------------------------- #
def test_shift_over_h_is_resolution_independent_in_units_of_h():
    """|dr|/h = 2 alpha h I(h,0) = 30 alpha / (7 pi) -- no dx in the limit."""
    predicted = 30.0 * 0.1 / (7.0 * math.pi)
    for n in (HBAR, 4.0, 16.0):
        row = measure(H, n, 0)
        assert row["shift_over_h_alpha_0p1"] == pytest.approx(predicted, rel=0.16), row