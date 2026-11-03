"""Verification tests for the plane-Poiseuille benchmark.

The benchmark exists to score the viscous operator and the no-slip wall against
an exact solution, so the tests do the same thing at a size that fits in CI:

* the analytic reference is checked against the Navier-Stokes balance it is
  supposed to solve (no solver involved);
* the hand-built periodic images are checked to be a *complete* periodic
  neighbourhood (uniform summation density across the seam), with a negative
  control proving the images are what produce that uniformity;
* one short integration is scored against ``u = g(a^2 - x^2)/(2 nu)``.
"""
import math

import pytest
import torch

from benchmarks.poiseuille.run import (
    analytic_profile,
    analytic_scalars,
    augment,
    build_channel,
    make_config,
    mean_profile,
    periodic_images,
    run_case,
)

torch.set_num_threads(2)

NU = 1.0e-2
BODY = 9.81
RHO0 = 1000.0


# --------------------------------------------------------------------------- #
# 1. the reference itself
# --------------------------------------------------------------------------- #
def test_analytic_profile_solves_the_channel_problem():
    """``u = g (a^2 - x^2) / 2 nu`` is zero at the walls and balances the body force."""
    a, nu, g = 0.01, NU, BODY
    xs = [-a + 0.001 * i for i in range(21)]
    u = analytic_profile(xs, a, nu, g)
    # no-slip at both walls
    assert abs(u[0]) < 1e-15 and abs(u[-1]) < 1e-15
    # nu u'' = -g everywhere (second difference, uniform grid)
    dx = xs[1] - xs[0]
    for k in range(1, len(u) - 1):
        second = (u[k + 1] - 2 * u[k] + u[k - 1]) / (dx * dx)
        assert abs(nu * second + g) < 1e-8
    # u_max and mean velocity of a parabola
    s = analytic_scalars(a, nu, g)
    assert math.isclose(max(u), s["u_max_m_per_s"], rel_tol=1e-12)
    assert math.isclose(s["u_mean_m_per_s"], (2.0 / 3.0) * s["u_max_m_per_s"],
                        rel_tol=1e-12)


# --------------------------------------------------------------------------- #
# 2. the periodic images
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def channel():
    dx = 0.004
    cfg = make_config(dx, NU, BODY, RHO0, 2.0)
    h = cfg.h_factor * dx
    pos, mass, wall = build_channel(dx, 0.02, 16 * dx, RHO0, h=h)
    return cfg, dx, h, pos, mass, wall


def test_image_strip_does_not_duplicate_particles(channel):
    """Images live one period away and the two strips cannot overlap."""
    cfg, dx, h, pos, mass, wall = channel
    idx, shift = periodic_images(pos, None, None, 16 * dx, 2 * h)
    assert idx.numel() > 0
    # every image is a full period away from its parent: no self-interaction
    assert set(shift.abs().tolist()) == {1.0}
    depth = 4.0 * h                        # two support radii, see periodic_images
    y = pos[:, 1]
    lo = y < depth                         # copied up
    hi = y > 16 * dx - depth               # copied down
    n_lo, n_hi = int(lo.sum()), int(hi.sum())
    assert n_lo > 0
    assert n_hi > 0
    assert n_lo + n_hi == int(idx.numel())
    assert int(torch.logical_and(lo, hi).sum()) == 0, (
        "strips overlap: reduce the strip depth")


def _density(pos, mass, wall, cfg, h, use_images):
    from tensorsph.droplet_impact import drop_wall_wall_pairs, rhs
    from tensorsph.neighbor_search import cell_linked_pairs
    vel = torch.zeros_like(pos)
    if use_images:
        ap, av, am, _ = augment(pos, vel, mass, 16.0 * cfg.dx, 2.0 * h)
    else:
        ap, av, am = pos, vel, mass
    pairs = cell_linked_pairs(torch.cat((ap, wall), dim=0), 2.0 * h)
    pairs = drop_wall_wall_pairs(pairs, ap.shape[0])
    rho, _, _ = rhs(ap, av, am, cfg, pairs=pairs, wall=wall, dt=cfg.dt)
    return rho[:pos.shape[0]]


def test_density_is_uniform_across_the_periodic_seam(channel):
    """Summation density must not know where the seam is.

    With a complete image set every fluid particle, including the ones sitting
    on the seam, has the kernel support of an infinite channel.  A strip that is
    too thin (the first version of this benchmark copied only ``2h`` instead of
    ``4h``) leaves ghost densities at 0.3-0.7 rho0 which, through the ``m/rho``
    weight of the viscous term, act as an extra 1.5x viscosity near the seam.
    """
    cfg, dx, h, pos, mass, wall = channel
    rho = _density(pos, mass, wall, cfg, h, use_images=True)
    ratio = rho / RHO0
    assert float(ratio.min()) > 0.995
    assert float(ratio.max()) < 1.005
    # spread between the seam columns and the middle column
    spread = float(ratio.max() - ratio.min())
    assert spread < 0.01


def test_without_images_the_seam_loses_its_neighbours(channel):
    """Negative control: the uniformity above comes from the images."""
    cfg, dx, h, pos, mass, wall = channel
    rho = _density(pos, mass, wall, cfg, h, use_images=False)
    ratio = rho / RHO0
    assert float(ratio.min()) < 0.9, (
        "the raw channel should show a density deficit at the seam; if it does "
        "not, this test no longer controls anything")


# --------------------------------------------------------------------------- #
# 3. one short integration against the exact solution
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def coarse_run():
    return run_case(dx=0.004, nu=NU, body=BODY, sound_speed=2.0, t_end=0.08)


def test_steady_profile_matches_the_parabola(coarse_run):
    p = coarse_run["profile"]
    assert len(p) == 5                      # H/dx = 0.02 / 0.004
    xb = [q["x_m"] for q in p]
    a = 0.5 * coarse_run["setup"]["channel_width_m"]
    nu = coarse_run["setup"]["viscosity_m2_per_s"]
    delta = coarse_run["wall_condition"]["effective_wall_offset_m"]
    ref = analytic_profile(xb, a + delta, nu, BODY)
    num = sum((q["u_mean"] - r) ** 2 for q, r in zip(p, ref))
    den = sum(r * r for r in ref)
    assert math.sqrt(num / den) * 100.0 < 5.0
    # the fitted wall plane must be about half a spacing outside the fluid face
    assert 0.35 < coarse_run["wall_condition"]["effective_wall_offset_over_dx"] < 0.75


def test_recovered_viscosity_is_the_configured_one(coarse_run):
    fit = coarse_run["viscosity_fit"]["effective_wall"]
    assert abs(fit["nu_rel_pct"]) < 3.0, fit
    assert fit["nu_effective_m2_per_s"] > 0.0


def test_run_is_consistent_and_contained(coarse_run):
    g = coarse_run["gates"]
    assert g["mass_drift_rel"] < 1e-12
    assert g["density_min"] > 0.99 and g["density_max"] < 1.01
    # no particle may cross the nominal wall face by more than the half
    # spacing the wall discretisation itself sits outside it
    assert g["wall_penetration_over_dx"] <= 0.0
    # the flow is x-only: no streamwise variation inside a column
    assert g["max_profile_std_over_umax"] < 0.05
    for value in coarse_run["history"]["vmax"]:
        assert math.isfinite(value)


def test_nominal_wall_convention_is_the_biased_one(coarse_run):
    """Documented, quantified bias: scoring against ``x = +-a`` reads ~+29 %.

    This is a *characterisation* test, not an acceptance test: it pins the
    number the docs quote so that a future change to the wall convention cannot
    silently invalidate them.
    """
    g = coarse_run["gates"]
    assert g["u_max_rel_pct"] > 15.0
    assert not g["within_15pct"]
    assert abs(g["flow_rate_rel_pct_effective_wall"]) < 5.0