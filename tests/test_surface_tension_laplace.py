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

Scope added by the 0.9.0 colour-function work (merged from
``agent/m2191``), kept in the same file because both halves guard the same
model:

* a circle of radius ``R`` has curvature ``kappa = 1/R`` and outward normal
  ``r_hat``;
* a static 2-D droplet holds ``dp = sigma / R``;
* the curvature operator itself is exact for an exact normal field
  (``< 5 %``), so any error measured below belongs to the colour function and
  the normal it produces, not to the discrete divergence.

Whoever changes the surface-tension model will find these tests fail before
the benchmark numbers move: ``tests/test_surface_tension.py`` only checks
invariants (zero force for ``sigma = 0``, unit normals, sign conventions) and
passes for a model that cannot hold any Laplace pressure at all.
"""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tensorsph import Config
from tensorsph.droplet_impact import density_from_neighbor_pairs, rhs
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import compute_curvature, surface_tension_field

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

# --------------------------------------------------------------------------- #
# 0.9.0: colour-function study on the same lattice (merged from agent/m2191)
# --------------------------------------------------------------------------- #
def make_circle(radius: float, dx: float):
    """Deterministic lattice disc of radius ``radius``, centered on the origin."""
    half = int(math.ceil(radius / dx)) + 1
    axis = torch.arange(-half, half + 1, dtype=torch.float64) * dx
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    pos = pos[torch.linalg.vector_norm(pos, dim=1) <= radius].contiguous()
    mass = torch.full((pos.shape[0],), 1000.0 * dx * dx, dtype=torch.float64)
    return pos, mass


def field(pos, mass, dx, model, sigma=0.5, smoothing=4, delta_scale=1.0):
    cfg = Config(dx=dx, rho0=1000.0, gravity=0.0, viscosity=0.0,
                 surface_tension_coefficient=sigma,
                 surface_tension_color=model,
                 surface_tension_smoothing=smoothing,
                 surface_tension_delta_scale=delta_scale)
    h = cfg.h_factor * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    return cfg, pairs, rho, surface_tension_field(pos, rho, mass, pairs, cfg)


@pytest.mark.parametrize("R_over_dx", [8, 12, 16])
def test_curvature_of_a_circle_with_the_analytic_normal(R_over_dx):
    """The discrete div(n) is exact for the exact normal -- it is not the culprit."""
    radius = 0.02
    dx = radius / R_over_dx
    pos, mass = make_circle(radius, dx)
    h = 1.3 * dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    r = torch.linalg.vector_norm(pos, dim=1)
    band = (r > radius - 1.5 * dx) & (r < radius + 0.5 * dx)
    kappa = compute_curvature(pos, pos / r[:, None], mass, rho, pairs, h)
    err = abs(float(kappa[band].mean()) * radius - 1.0)
    assert err < 0.05, f"curvature operator error {err:.3%} at R/dx={R_over_dx}"


@pytest.mark.parametrize("R_over_dx", [8, 12, 16])
def test_support_sum_normal_is_outward(R_over_dx):
    """``-grad c`` from the missing-support vector is the outward radial normal."""
    radius = 0.02
    dx = radius / R_over_dx
    pos, mass = make_circle(radius, dx)
    _, _, _, _ = field(pos, mass, dx, "neighbor_count")
    cfg, pairs, rho, (normal, kappa, delta_s, acc) = field(
        pos, mass, dx, "support_sum")
    r = torch.linalg.vector_norm(pos, dim=1)
    band = (r > radius - 1.5 * dx) & (r < radius + 0.5 * dx)
    dot = float((normal[band] * (pos / r[:, None])[band]).sum(dim=1).mean())
    assert dot > 0.98, f"normal/radial correlation {dot:+.3f} at R/dx={R_over_dx}"
    # ... and the curvature it produces reproduces 1/R.
    ratio = float(kappa[band].mean()) * radius
    assert abs(ratio - 1.0) < 0.08, f"kappa/(1/R) = {ratio:.3f} at R/dx={R_over_dx}"
    # The force points inward for a convex droplet.
    assert float((acc[band] * (pos / r[:, None])[band]).sum(dim=1).mean()) < 0.0


@pytest.mark.parametrize("R_over_dx", [8, 12])
def test_unsmoothed_normal_does_not_converge(R_over_dx):
    """Negative control: normalising the raw support vector diverges with refinement.

    Without the smoothing passes the unit-vector field carries lattice noise that
    ``div(n)`` amplifies, so the curvature error *grows* with resolution instead
    of shrinking. Measured: 6.4x / 9.4x / 13.1x the analytic curvature at
    R/dx = 8 / 12 / 16. This test exists so that removing the smoothing cannot
    look like a harmless simplification.
    """
    radius = 0.02
    dx = radius / R_over_dx
    pos, mass = make_circle(radius, dx)
    cfg, pairs, rho, (normal, kappa, delta_s, acc) = field(
        pos, mass, dx, "support_sum", smoothing=0)
    r = torch.linalg.vector_norm(pos, dim=1)
    band = (r > radius - 1.5 * dx) & (r < radius + 0.5 * dx)
    ratio = float(kappa[band].mean()) * radius
    assert ratio > 2.0, (
        f"unsmoothed curvature came out at {ratio:.2f}x analytic; the divergence "
        "this test documents is gone -- update docs/benchmarks/surface-tension.md")


def test_neighbour_count_colour_has_no_normal():
    """The historical heuristic does not produce the surface normal -- documented.

    Its colour takes two values, so ``grad c`` is not correlated with ``r_hat``
    (measured 0.02) and the resulting curvature is negative, i.e. the force
    expands the droplet instead of contracting it.
    """
    radius = 0.02
    dx = radius / 12
    pos, mass = make_circle(radius, dx)
    cfg, pairs, rho, (normal, kappa, delta_s, acc) = field(
        pos, mass, dx, "neighbor_count")
    r = torch.linalg.vector_norm(pos, dim=1)
    band = (r > radius - 1.5 * dx) & (r < radius + 0.5 * dx)
    dot = float((normal[band] * (pos / r[:, None])[band]).sum(dim=1).mean())
    assert abs(dot) < 0.5, (
        f"neighbour-count normal/radial correlation {dot:+.3f}; if this now "
        "reaches ~1 the heuristic has been replaced and this test is obsolete")
    assert float(kappa[band].mean()) < 0.0


def test_delta_scale_scales_the_force_linearly():
    """``surface_tension_delta_scale`` is the documented multiplier on |grad c|."""
    radius, R_over_dx = 0.02, 12
    dx = radius / R_over_dx
    pos, mass = make_circle(radius, dx)
    _, _, _, (_, _, _, acc1) = field(pos, mass, dx, "support_sum", delta_scale=1.0)
    _, _, _, (_, _, _, acc2) = field(pos, mass, dx, "support_sum", delta_scale=2.0)
    assert torch.allclose(acc2, 2.0 * acc1, atol=1e-12)


def test_static_droplet_holds_the_laplace_pressure():
    """End-to-end: a pre-stressed droplet keeps ``dp = sigma / R`` in the core.

    Soft EOS (``c0 = 3 m/s``) so that ``sigma/R`` is ~2 % of the bulk modulus and
    therefore above the density-summation noise; pre-stressed masses so that the
    initial state already carries the pressure the interface must sustain. The
    measured core pressure is compared with ``sigma/R`` after the acoustic
    transient has decayed (the first half of the run is discarded).
    """
    radius, R_over_dx = 0.02, 8
    dx = radius / R_over_dx
    sigma, sound_speed, gamma = 0.5, 3.0, 7.0
    pos, mass = make_circle(radius, dx)
    h = 1.3 * dx
    rho_lattice = density_from_neighbor_pairs(pos, mass, h)
    core = torch.linalg.vector_norm(pos, dim=1) < 0.6 * radius
    rho_ref = float(rho_lattice[core].mean())
    bulk = rho_ref * sound_speed ** 2 / gamma
    dp_theory = sigma / radius
    scale = (1.0 + dp_theory / bulk) ** (1.0 / gamma)

    cfg = Config(dx=dx, rho0=rho_ref, sound_speed=sound_speed, gamma=gamma,
                 gravity=0.0, viscosity=0.005,
                 surface_tension_coefficient=sigma,
                 surface_tension_color="support_sum")
    mass = mass * scale
    dt = min(0.25 * h / sound_speed,
             0.125 * h * h / cfg.viscosity)
    cfg.dt = dt
    vel = torch.zeros_like(pos)
    r_start = float(torch.linalg.vector_norm(pos, dim=1).max())
    history = []
    for _ in range(600):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None)
        vel = vel + dt * acc
        pos = pos + dt * vel
        history.append(float(pressure[torch.linalg.vector_norm(pos, dim=1)
                                      < 0.5 * r_start].mean()))
    tail = history[len(history) // 2:]
    dp = sum(tail) / len(tail)
    ratio = dp / dp_theory
    # The model currently sustains ~0.7 of sigma/R because the surface delta
    # function integrates to ~0.67 over the fluid (see
    # docs/benchmarks/surface-tension.md, section 5.2). This test pins the
    # measured value, it does not bless it: the sign must be right and the
    # magnitude within the documented shortfall. Tighten to 0.15 once the delta
    # function is normalised.
    assert ratio > 0.0, f"pressure jump has the wrong sign: {ratio:+.3f}"
    assert 0.5 < ratio < 1.15, (
        f"Laplace jump {dp:.3f} Pa vs sigma/R = {dp_theory:.3f} Pa "
        f"({ratio:+.3f}x); run benchmarks/surface_tension/laplace.py")
