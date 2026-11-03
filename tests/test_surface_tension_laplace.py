"""Tests for the Laplace-law verification of the CSF surface-tension model.

Two kinds of test live here, deliberately separated:

1. **Gauge tests** (must pass). They verify the *measurement*: the estimator
   recovers the exact Laplace pressure ``sigma/R`` on a synthetic field whose
   normal integral is known analytically. If these fail, the benchmark's numbers
   are meaningless regardless of what the model does.
2. **Known-defect tests** (expected to fail, ``xfail(strict=True)``). They assert
   that the shipped CSF force satisfies the Laplace law. They do not today; the
   strict marker turns a silent physics failure into a visible, non-blocking
   signal and forces this file and
   ``docs/benchmarks/surface-tension-laplace.md`` to be updated by whoever fixes
   the model (a strict xfail that starts passing fails the suite).
"""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tensorsph import Config
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import compute_surface_tension_force

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "surface_tension" / "laplace.py"
_spec = importlib.util.spec_from_file_location("laplace_bench", _BENCH)
laplace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(laplace)


SIGMA = laplace.SIGMA
RHO0 = laplace.RHO0


def _disc(radius, dx):
    pos = laplace.make_disc(radius, dx)
    r = torch.linalg.vector_norm(pos, dim=1)
    return pos, r


def _synthetic_bump(r, radius, dx):
    """Smooth bump on the fluid side whose normal integral is exactly 1."""
    t = 2.0 * dx
    u = (r - (radius - 0.5 * t)) / t
    return torch.where(u.abs() <= 0.5,
                       (1.0 / t) * (1.0 + torch.cos(2.0 * math.pi * u)),
                       torch.zeros_like(u))


def test_estimator_recovers_laplace_on_analytic_field():
    """Gauge: with f = (sigma/R) * delta(r-R), the estimator must return sigma/R."""
    radius, dx = 0.1, 0.1 / 16
    pos, r = _disc(radius, dx)
    laplace._LAST_RADIUS = r
    inward = (SIGMA / (RHO0 * radius)) * _synthetic_bump(r, radius, dx)
    jump, r_c = laplace.perimeter_jump(RHO0, inward, radius, dx)
    theory = SIGMA / radius
    # The bump spans [R - 2dx, R], so its force centroid sits at R - dx; the
    # identity in perimeter_jump uses r_c, which is why the jump can be exact
    # even though the centroid is a full cell inside the interface.
    assert abs(r_c - (radius - dx)) < 0.5 * dx, f"unexpected centroid {r_c}"
    assert abs(jump - theory) / theory < 0.01, (
        f"gauge error {abs(jump - theory) / theory:.3%} exceeds 1%: "
        "the estimator, not the model, would be responsible for a failure")


def test_estimator_scales_linearly_with_sigma():
    """Gauge: the estimator is linear in the surface-tension coefficient."""
    radius, dx = 0.1, 0.1 / 16
    pos, r = _disc(radius, dx)
    laplace._LAST_RADIUS = r
    bump = _synthetic_bump(r, radius, dx)
    a, _ = laplace.perimeter_jump(RHO0, (SIGMA / (RHO0 * radius)) * bump, radius, dx)
    b, _ = laplace.perimeter_jump(RHO0, (2 * SIGMA / (RHO0 * radius)) * bump, radius, dx)
    assert abs(b / a - 2.0) < 1e-6


def test_estimator_rejects_outward_force():
    """Gauge: an outward force must produce a negative pressure jump (no abs())."""
    radius, dx = 0.1, 0.1 / 16
    pos, r = _disc(radius, dx)
    laplace._LAST_RADIUS = r
    inward = -(SIGMA / (RHO0 * radius)) * _synthetic_bump(r, radius, dx)
    jump, _ = laplace.perimeter_jump(RHO0, inward, radius, dx)
    assert jump < 0.0


def test_detector_covers_the_interface_shell():
    """The shipped neighbour-count detector on a disc at R/dx = 16."""
    radius, dx = 0.1, 0.1 / 16
    cfg = Config(dx=dx, surface_tension_coefficient=SIGMA)
    pos, r = _disc(radius, dx)
    pairs = cell_linked_pairs(pos, 2.0 * cfg.h_factor * dx)
    count = laplace.neighbour_count(pos, pairs)
    detected = count < 0.7 * count.mean()
    geometric = r > radius - dx
    recall = float((detected & geometric).sum()) / float(geometric.sum())
    precision = float((detected & geometric).sum()) / float(detected.sum())
    assert recall > 0.9 and precision > 0.9


@pytest.mark.xfail(strict=True, reason=(
    "known defect: the shipped CSF force does not satisfy the 2-D Laplace law "
    "dp = sigma/R; see docs/benchmarks/surface-tension-laplace.md"))
def test_laplace_law_is_satisfied():
    """Known defect: the model should reproduce Delta p = sigma/R within 15%."""
    worst = 0.0
    for ratio in (16.0, 32.0):
        row = laplace.measure(0.1, 0.1 / ratio, SIGMA)
        worst = max(worst, row["jump_rel_error"])
    assert worst <= 0.15, f"worst Laplace error {worst:.1%}"


@pytest.mark.xfail(strict=True, reason=(
    "known defect: curvature from div(n) on the neighbour-count colour field "
    "has the wrong sign and 4-7x too small a magnitude"))
def test_curvature_matches_exact_interface_curvature():
    """Known defect: median kappa*R should be +1 for the module's convention."""
    row = laplace.measure(0.1, 0.1 / 32, SIGMA)
    assert abs(row["curvature_ratio"] - 1.0) <= 0.15


def test_force_direction_is_mostly_inward():
    """The shipped model does pull inward for the large majority of particles.

    This is the one property the shipped static-droplet benchmark actually
    checked, and it does hold; it is simply not the Laplace law.
    """
    row = laplace.measure(0.1, 0.1 / 16, SIGMA)
    assert row["inward_fraction"] > 0.85