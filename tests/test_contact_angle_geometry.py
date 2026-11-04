"""Tests for the contact-angle BC verification instrument.

These hold the *instrument* (analytic 2-D spherical cap) to its own standard and
pin the measured state of ``apply_contact_angle_boundary``, following the
convention used by ``tests/test_laplace_droplet.py``: the numbers below record
what the merged model does today, so a future change shows up as a failure here
rather than passing silently.

The gap being pinned: the BC reproduces the prescribed contact angle to 0.1-2.8
deg on the c=0.5 shell, but the delta_s-weighted curvature at the contact
region is +10..+46 % against the analytic kappa = 1/R, i.e. outside the +-15 %
gate.  ``test_curvature_remains_outside_the_gate`` is the honest record of that
gap -- when the curvature operator is fixed it must fail and be rewritten.

Numbers come from ``benchmarks/contact_angle_geometry/run.py``; the doc is
``docs/verification_contact_angle_geometry.md``.
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

# This file only exercises static operators; batched 2x2 torch.linalg.inv costs
# seconds per call at the default thread count (see the doc), so pin one thread.
# The global default dtype is deliberately NOT touched: the lattices below are
# built with an explicit float64 dtype, and overriding the default would leak
# into the float32 unit tests in tests/test_contact_angle.py.
torch.set_num_threads(1)

_RUN = Path(__file__).resolve().parents[1] / "benchmarks" / "contact_angle_geometry" / "run.py"


def _load():
    spec = importlib.util.spec_from_file_location("contact_angle_run", _RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _load()


@pytest.fixture(scope="module")
def cases():
    return {(60, 20): R.run_case(60.0, 20), (90, 20): R.run_case(90.0, 20),
            (120, 20): R.run_case(120.0, 20), (60, 10): R.run_case(60.0, 10)}


# --------------------------------------------------------------------------- #
# The instrument
# --------------------------------------------------------------------------- #

def test_closed_form_geometry_reproduces_the_contact_angle():
    """The (R, y_c) formula is proved numerically, not asserted by hand."""
    out = R.cap_geometry_check()
    assert out["result"] == "pass"
    assert out["max_abs_error_deg"] < 1e-6


def test_bc_is_idempotent_on_normals_that_already_satisfy_youngs_law():
    """C1: an analytic normal with n.n_wall = cos(theta) must pass through unchanged."""
    out = R.control_exactness()
    assert out["result"] == "pass"
    assert out["max_deviation_deg"] < 1e-6


def test_bc_leaves_the_far_field_bitwise_unchanged(cases):
    """C2: the BC is local (reach = 2h)."""
    for key, case in cases.items():
        assert case["locality_max_abs_delta"] == 0.0, key


def test_local_assembly_matches_the_production_csf_fields(cases):
    """C3: the benchmark measures ``csf_fields``, not a private reimplementation."""
    for key, case in cases.items():
        assert case["assembly_max_abs_error_vs_csf_fields"] < 1e-12, key


# --------------------------------------------------------------------------- #
# The measured state of the model
# --------------------------------------------------------------------------- #

def test_geometric_contact_angle_gate(cases):
    """R/dx >= 20 reproduces the prescribed angle; R/dx = 10 (thin cap) does not."""
    for key in ((60, 20), (90, 20), (120, 20)):
        assert abs(cases[key]["fit"]["theta_error_deg"]) < 3.0, key
    assert abs(cases[(60, 10)]["fit"]["theta_error_deg"]) > 3.0


def test_geometric_angle_converges_under_refinement(cases):
    """Coarser lattice -> larger shape error: the documented O(dx) behaviour."""
    assert (abs(cases[(60, 10)]["fit"]["theta_error_deg"])
            > abs(cases[(60, 20)]["fit"]["theta_error_deg"]))


def test_bc_reduces_the_interface_normal_error(cases):
    """The BC's stated purpose, measured against the analytic normal."""
    for key, case in cases.items():
        off = case["variants"]["off"]["normal_angle_deg_shell_mean"]
        on = case["variants"]["on"]["normal_angle_deg_shell_mean"]
        assert on <= off, key


def test_curvature_gap_is_pinned_at_the_coarse_and_fine_ends(cases):
    """Honest record: kappa error +10..+46 % against 1/R, and it is not monotone.

    The delta_s-weighted curvature error sits *inside* the +-15 % gate at
    theta=90 deg / R/dx=20 (+10.5 %) and outside it at R/dx=10 (+39.8 %) and at
    R/dx=40 (+21.4 %), i.e. the 2-D CSF curvature operator does not converge in
    the contact-angle configuration -- the same non-convergence documented for
    the free droplet in docs/verification_surface_tension_curvature.md.
    """
    k = {(t, r): c["variants"]["on"]["kappa_delta_s_weighted_rel_error"]
         for (t, r), c in cases.items()}
    assert k[(90, 20)] == pytest.approx(0.105, abs=0.02)
    assert k[(90, 20)] < 0.15                     # inside the gate ...
    assert k[(120, 20)] > 0.15                    # ... and outside it nearby
    assert k[(60, 10)] > 0.30


def test_bc_does_not_change_the_shell_curvature_away_from_the_wall(cases):
    """The contact-angle BC acts within 2h of the wall; the shell curvature is not there.

    For a well-resolved semicircular cap (theta = 90 deg, R/dx = 20) the BC
    moves the delta_s-weighted curvature by < 0.5 percentage point, so the
    +10..+46 % kappa gap belongs to the CSF curvature operator, not to the
    contact-angle BC.  Where the cap is thin (theta = 60 deg) most of the shell
    lies inside the 2h band and the BC does move the curvature, by a lot.
    """
    off = cases[(90, 20)]["variants"]["off"]["kappa_delta_s_weighted_rel_error"]
    on = cases[(90, 20)]["variants"]["on"]["kappa_delta_s_weighted_rel_error"]
    assert abs(on - off) < 0.005
    off60 = cases[(60, 20)]["variants"]["off"]["kappa_delta_s_weighted_rel_error"]
    on60 = cases[(60, 20)]["variants"]["on"]["kappa_delta_s_weighted_rel_error"]
    assert abs(on60) < abs(off60)


def test_the_wall_band_carries_no_interface_shell():
    """The colour field is inflated by wall particles, so d<h holds no c~0.5 point.

    This is why the BC cannot deliver the full rotation: there is no interface in
    the band it is allowed to touch.  Recorded so that a change to the colour
    field near the wall is noticed.
    """
    case = R.run_case(90.0, 20)
    assert case["n_shell_near_wall"] == 0
    assert case["rotated_fluid_fraction"] > 0.0


def test_wall_particles_are_rotated_by_the_production_call(cases):
    """C4: the BC is applied to wall rows too (d = 0 -> blend = 1). Implicit behaviour."""
    for case in cases.values():
        assert case["n_rotated_wall"] > 0
        assert case["rotated_wall_fraction"] > 0.5


def test_no_non_finite_values():
    case = R.run_case(120.0, 20)
    assert case["acceptance"]["no_non_finite_values"] == "pass"
    assert case["variants"]["on"]["n_finite"] is True