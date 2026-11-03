"""Verification tests for the wall contact-angle boundary condition.

These tests cover the chain that ``tests/test_contact_angle.py`` and
``tests/test_csf_consistent.py`` do not: those pin the *operator*
(``apply_contact_angle_boundary`` on hand-built gradients) and that
``csf_fields`` *changes* when the BC is switched on.  What is pinned here is
the measurement machinery of ``benchmarks/contact_angle/run.py`` and the two
facts a user needs:

1. the boundary condition survives the model's own normal smoothing, i.e. the
   normal the CSF force actually uses satisfies ``n . n_wall ~ cos(theta)``;
2. the emergent macroscopic contact angle does **not** follow Young's law yet
   for the shipped single-phase scheme.  That is an honest known gap, locked
   here so that it cannot regress silently and so that whoever fixes it has to
   update ``docs/benchmarks/contact-angle.md`` and this test together.

The estimator itself is validated first (synthetic caps of known angle), so a
later disagreement cannot be blamed on the measurement.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

torch.set_num_threads(1)

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "contact_angle"
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run as ca  # noqa: E402  (benchmark harness)

from tensorsph.csf_consistent import csf_fields  # noqa: E402
from tensorsph.neighbor_search import PairListCache  # noqa: E402

DX = 0.005
POOL_WIDTH = 0.30
RADIUS = 0.06


# --------------------------------------------------------------------------- #
# Estimator: calibrated on caps of known angle
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def calibration():
    pos, _mass, area_pp = ca.half_disc_lattice(RADIUS, DX, POOL_WIDTH)
    area = pos.shape[0] * area_pp
    anchor = ca.height_area_ratio(pos, DX, area_pp)
    return {"table": ca.calibration_table(DX, area, POOL_WIDTH, step_deg=2.0,
                                          anchor_ratio=anchor),
            "area": area, "area_pp": area_pp, "pos": pos, "mass": _mass}


@pytest.mark.parametrize("theta_deg", [30.0, 45.0, 60.0, 90.0, 120.0])
def test_estimator_recovers_synthetic_cap_angle(calibration, theta_deg):
    """The H^2/A estimator, calibrated on the same lattice, inverts a cap exactly."""
    pos, _mass, area_pp, _R = ca.cap_lattice(math.radians(theta_deg),
                                             calibration["area"], DX, POOL_WIDTH)
    r = ca.height_area_ratio(pos, DX, area_pp)
    measured = ca.invert_ratio(calibration["table"], r)
    assert abs(measured - theta_deg) < 5.0, (measured, theta_deg)


def test_estimator_above_135_is_weak_and_is_reported_as_such(calibration):
    """Near 180 deg the cap closes onto a disc and H stops discriminating.

    The estimator must still be usable there but only inside the arrangement
    band the benchmark reports for that angle -- this test pins the band rather
    than pretending the number is as sharp as it is at 90 deg.
    """
    unc = ca.estimator_uncertainty(DX, calibration["area"], POOL_WIDTH,
                                   calibration["table"],
                                   angles=(135.0, 150.0))["per_angle_deg"]
    for theta in (135.0, 150.0):
        pos, _mass, area_pp, _R = ca.cap_lattice(math.radians(theta),
                                                 calibration["area"], DX, POOL_WIDTH)
        r = ca.height_area_ratio(pos, DX, area_pp)
        measured = ca.invert_ratio(calibration["table"], r)
        # above ~135 deg the cap closes onto a disc: the height stops
        # discriminating and the estimator is only good to its arrangement band
        assert abs(measured - theta) <= max(6.0, 1.5 * unc[f"{theta:g}"]), (theta, measured, unc)


def test_estimator_zero_point_is_the_half_disc(calibration):
    """The shared initial state must measure 90 deg; otherwise the A/B is void."""
    pos = calibration["pos"]
    r = ca.height_area_ratio(pos, DX, calibration["area_pp"])
    assert abs(ca.invert_ratio(calibration["table"], r) - 90.0) < 3.0


def test_estimator_sensitivity_to_arrangement_is_bounded(calibration):
    unc = ca.estimator_uncertainty(DX, calibration["area"], POOL_WIDTH,
                                   calibration["table"])
    # near 150-180 deg the cap closes onto a disc and the height stops
    # discriminating; the benchmark reports that band per angle
    assert unc["per_angle_deg"]["90"] < 6.0, unc
    assert unc["per_angle_deg"]["60"] < 8.0, unc


def test_circle_fit_cross_check_is_consistent_within_its_known_bias(calibration):
    """Circle fit carries a known inward bias; it must stay a usable cross-check."""
    pos, mass, area_pp = ca.half_disc_lattice(RADIUS, DX, POOL_WIDTH)
    measured = ca.measure_contact_angle(pos, mass, 1.3 * DX, DX, area_pp,
                                        table=calibration["table"])
    assert abs(measured["theta_circle_deg"] - 90.0) < 20.0, measured["theta_circle_deg"]


# --------------------------------------------------------------------------- #
# The boundary condition must survive the model's own normal smoothing
# --------------------------------------------------------------------------- #

def _wall_and_fluid():
    pos, mass, _area_pp = ca.half_disc_lattice(RADIUS, DX, POOL_WIDTH)
    wall = ca.bottom_wall(POOL_WIDTH, DX)
    h = 1.3 * DX
    n = pos.shape[0]
    pairs, _ = PairListCache(2.0 * h, 0.0, fluid_count=n).get(
        torch.cat((pos, wall), dim=0))
    fluid_pairs = pairs[(pairs[:, 0] < n) & (pairs[:, 1] < n)]
    return pos, mass, wall, h, fluid_pairs


def _bottom_row_interface_normal(theta):
    pos, mass, wall, h, fluid_pairs = _wall_and_fluid()
    state = csf_fields(pos, mass, fluid_pairs, h, ca.RHO0, wall=wall,
                       contact_angle=None if theta is None
                       else math.radians(theta))
    dy = DX * math.sqrt(3.0) / 2.0
    y_min = float(pos[:, 1].min())
    bottom = pos[:, 1] < y_min + 1.5 * dy
    interface = bottom & (state.delta_s > 0.05 * float(state.delta_s.max()))
    return state.normal[interface]


def test_bc_survives_the_normal_smoothing():
    """Rotating before the Shepard average erased the BC; rotating after must not.

    With ``theta = 30 deg`` every wetted-base normal that the BC touches has to
    have a positive wall-normal component (it is being rotated towards the
    wall-normal/tangent mix of a wetting contact line), whereas the untouched
    field points into the wall for the whole bottom row (``n . n_wall < 0``).
    """
    dy = torch.tensor([0.0, 1.0], dtype=torch.float64)
    untouched = (_bottom_row_interface_normal(None) * dy).sum(dim=1)
    wetting = (_bottom_row_interface_normal(30.0) * dy).sum(dim=1)
    assert float(untouched.median()) < -0.5      # untouched: into the wall
    assert float(wetting.max()) > 0.3            # BC: rotated towards the fluid
    assert float(wetting.median()) > float(untouched.median()) + 0.2


def test_bc_rotation_direction_tracks_the_prescribed_angle():
    """n . n_wall must increase with the prescribed angle (drying rotates less)."""
    dy = torch.tensor([0.0, 1.0], dtype=torch.float64)
    medians = [float((_bottom_row_interface_normal(t) * dy).sum(dim=1).median())
               for t in (30.0, 90.0, 150.0)]
    assert medians[0] >= medians[1] - 1.0e-9
    assert medians[1] >= medians[2] - 1.0e-9
    assert medians[0] > medians[2] + 0.05


# --------------------------------------------------------------------------- #
# Known gap: the emergent angle does not follow Young's law yet
# --------------------------------------------------------------------------- #

def test_emergent_contact_angle_does_not_yet_follow_youngs_law():
    """Honest known gap, locked so it cannot regress silently.

    Measured on the benchmark (dx = 0.005, 0.05 s of simulated time): every
    prescribed angle ends at the same angle as the no-BC control, and that
    common angle is ~12 deg away from the 90 deg initial state.  So the gap is
    set by the solver's spurious currents at this resolution, not by the BC --
    see docs/benchmarks/contact-angle.md.  When this test starts failing
    because the gaps got small, the capability has improved and the docs and
    the benchmark numbers must be updated with it.
    """
    results = ca.run_case_set([30.0, 150.0, None], DX, 0.05, RADIUS, POOL_WIDTH,
                              0.02, 10.0, quiet=True)
    prescribed = {c["contact_angle_prescribed_deg"]: c for c in results["cases"]}
    control = prescribed[None]
    for target in (30.0, 150.0):
        case = prescribed[target]
        # the BC does not move the emergent angle beyond the control by more
        # than 2 deg, and both miss the prescribed angle by far more than the
        # 15 deg platform tolerance
        assert abs(case["theta_final_deg"] - control["theta_final_deg"]) < 2.0
        assert abs(case["theta_final_deg"] - target) > 15.0
    # the solver itself stays healthy: area is conserved, no particle escapes
    for case in results["cases"]:
        assert abs(case["area_change_percent"]) < 1.0e-6
        assert case["boundary_events"] == 0