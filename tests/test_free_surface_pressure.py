"""Unit gates for the free-surface pressure closure (mission m3289).

``tensorsph.droplet_impact.free_surface_pressure`` is the Adami/Hu/Adams (2012,
eq. 11) shell closure: it replaces the Tait value of a free-surface particle
(``rho < 0.99 rho0``, i.e. p = 0 after the clamp) with a Shepard-renormalised,
body-force-continued average of its fluid neighbours' pressure.

These tests pin the three properties the closure is *for* -- exact
reproduction of a constant field, exact hydrostatic continuation, and
non-negativity -- plus the plumbing (flag off is a no-op, sweeps converge, the
dense reference implementation agrees with the production pair path). The
end-to-end Young-Laplace numbers live in
``benchmarks/laplace_pressure_floor/report.json`` and are gated in
``tests/test_laplace_pressure_floor.py``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    cubic_spline_kernel,
    density_from_neighbor_pairs,
    free_surface_pressure,
    rhs,
    rhs_all_pairs,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs

RHO0 = 1000.0
G = 9.81


def _cfg(**kwargs):
    base = dict(dx=0.01, h_factor=1.3, rho0=RHO0, sound_speed=20.0, gamma=7.0,
                gravity=G, viscosity=0.0, device="cpu")
    base.update(kwargs)
    return Config(**base)


def _column(n=26, spacing_factor=0.5, x=0.0):
    """Vertical column of particles; the top one is the free surface."""
    cfg = _cfg()
    h = cfg.h_factor * cfg.dx
    y = torch.arange(n, dtype=torch.float64) * (spacing_factor * h)
    pos = torch.stack([torch.full_like(y, x), y], dim=1)
    return cfg, pos


def _pairs(pos, cfg):
    h = cfg.h_factor * cfg.dx
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    wij = cubic_spline_kernel(torch.linalg.vector_norm(delta, dim=1), h)
    return i, j, wij


def _masked_rho(pos, cfg, n_masked=1, deficit=0.95):
    """rho = rho0 everywhere except the top ``n_masked`` particles."""
    rho = torch.full((pos.shape[0],), RHO0, dtype=pos.dtype)
    rho[-n_masked:] = deficit * RHO0
    return rho


# --------------------------------------------------------------------------- #
# the three properties
# --------------------------------------------------------------------------- #
def test_a_constant_pressure_field_is_reproduced_exactly():
    """Shepard renormalisation: dividing by the *deficient* sum must not scale.

    Gravity is switched off here on purpose: a *uniform* pressure field is not
    hydrostatically consistent, so with g != 0 the body-force continuation term
    is supposed to pull the answer down by rho g times the shell depth. That
    term is tested separately, below.
    """
    cfg, pos = _column()
    cfg = _cfg(gravity=0.0)
    i, j, wij = _pairs(pos, cfg)
    rho = _masked_rho(pos, cfg)
    pressure = torch.full((pos.shape[0],), 1234.5, dtype=pos.dtype)

    out = free_surface_pressure(pos, rho, pressure, cfg, i, j, wij)

    assert out[-1].item() == pytest.approx(1234.5, rel=1e-12)
    # Every particle that is not flagged free-surface is returned untouched.
    torch.testing.assert_close(out[:-1], pressure[:-1], rtol=0.0, atol=0.0)


def test_the_body_force_term_makes_the_closure_hydrostatically_exact():
    """A column at rest must extrapolate to p = 0 at its own surface.

    Without the ``rho_j g.(r_i - r_j)`` term (Adami eq. 11 drops it only for
    a = g = 0) the top particle inherits the *average* pressure of the shell
    below it, i.e. a free surface that pushes itself up. With the term the
    linear hydrostatic field is reproduced exactly, because the weight of the
    2h shell is accounted for.
    """
    cfg, pos = _column()
    cfg = _cfg(gravity=G)
    i, j, wij = _pairs(pos, cfg)
    y_surface = float(pos[-1, 1])            # the column top is the free surface
    inner = pos.shape[0] - 2                # deficient, with a one-sided shell
    rho = torch.full((pos.shape[0],), RHO0, dtype=pos.dtype)
    rho[inner] = 0.95 * RHO0                # flagged, everything around it is not
    pressure = (RHO0 * G * (y_surface - pos[:, 1])).clamp_min(0.0)

    with_g = free_surface_pressure(pos, rho, pressure, cfg, i, j, wij)
    no_g = free_surface_pressure(pos, rho, pressure,
                                 _cfg(gravity=0.0), i, j, wij)

    h = cfg.h_factor * cfg.dx
    analytic = float(RHO0 * G * (y_surface - pos[inner, 1]))
    # The linearly varying hydrostatic field is reproduced at the particle's own
    # height, not at the pressure-weighted centroid of its shell.
    assert with_g[inner].item() == pytest.approx(analytic, rel=0.02)
    # Without the body-force term the closure reads the shell mean instead, which
    # for a particle above its neighbours is *above* the local value.
    assert no_g[inner].item() > analytic
    # Measured offset is 0.35 * rho0 g h (the shell centroid sits below the
    # particle); 0.1 is a margin, not a fit.
    assert no_g[inner].item() - analytic > 0.1 * RHO0 * G * h


def test_the_closure_never_returns_tensile_pressure():
    cfg, pos = _column()
    i, j, wij = _pairs(pos, cfg)
    rho = _masked_rho(pos, cfg, n_masked=6)
    pressure = torch.zeros_like(rho)

    out = free_surface_pressure(pos, rho, pressure, cfg, i, j, wij)
    assert bool((out >= 0.0).all())
    torch.testing.assert_close(out, pressure, rtol=0.0, atol=0.0)


def test_extra_sweeps_only_help_masked_on_masked_neighbourhoods():
    """One sweep moves the value one shell layer inwards.

    The update is simultaneous (all flagged particles are written after the sum
    is formed), so a particle whose whole neighbourhood is flagged still reads
    the *clamped* zeros in sweep 1. It receives a value only in sweep 2, and the
    sequence converges to the interior value from below, one layer per sweep.
    On the Young-Laplace drop the deficient shell is two particles deep, which is
    why ``free_surface_pressure_sweeps = 2``: fewer leaves -22 % (measured in
    benchmarks/laplace_pressure_floor).
    """
    cfg, pos = _column()
    i, j, wij = _pairs(pos, cfg)
    # Three flagged particles at the top: the outermost one has no unflagged
    # neighbour, so it needs a second sweep to receive a value at all.
    rho = _masked_rho(pos, cfg, n_masked=3)
    rho[:-3] = 1.002 * RHO0
    pressure = tait_pressure(rho, cfg)

    one = free_surface_pressure(pos, rho, pressure, cfg, i, j, wij, sweeps=1)
    two = free_surface_pressure(pos, rho, pressure, cfg, i, j, wij, sweeps=2)
    three = free_surface_pressure(pos, rho, pressure, cfg, i, j, wij, sweeps=3)

    interior = float(pressure[:-3].mean())
    # Sweep 1: the outermost flagged particle sees only flagged neighbours, i.e.
    # the clamped zeros -- the closure does not invent a value out of nothing.
    assert one[-1].item() == 0.0
    assert two[-1].item() > 0.0
    # Converging from below towards the interior value, never past it.
    assert one[-1].item() < two[-1].item() < three[-1].item() <= interior
    assert (interior - three[-1].item()) < (interior - two[-1].item())


def test_the_flag_off_path_is_bit_identical():
    cfg, pos = _column()
    i, j, wij = _pairs(pos, cfg)
    rho = _masked_rho(pos, cfg)
    pressure = tait_pressure(rho, cfg)
    off = _cfg(free_surface_pressure=False)

    out = free_surface_pressure(pos, rho, pressure, off, i, j, wij)
    torch.testing.assert_close(out, pressure, rtol=0.0, atol=0.0)


def test_the_partial_neighbourhood_is_renormalised_not_damped():
    """A deficient kernel sum must not be read as a lower pressure.

    With the closure, a shell particle next to a *constant* interior carries the
    interior pressure; without it (clamped Tait) it carries zero and the
    interface loses the jump. This is the failure mode the P0 was about.
    """
    cfg, pos = _column()
    cfg = _cfg(gravity=0.0)
    i, j, wij = _pairs(pos, cfg)
    rho = torch.full((pos.shape[0],), 1.002 * RHO0, dtype=pos.dtype)
    rho[-1] = 0.95 * RHO0
    pressure = tait_pressure(rho, cfg)
    interior = pressure[:-1].mean()

    closed = free_surface_pressure(pos, rho, pressure, cfg, i, j, wij)
    open_ = free_surface_pressure(
        pos, rho, pressure, _cfg(free_surface_pressure=False), i, j, wij)

    assert open_[-1].item() == 0.0
    assert closed[-1].item() > 0.9 * float(interior)


# --------------------------------------------------------------------------- #
# plumbing: the dense reference must agree with the production pair path
# --------------------------------------------------------------------------- #
def _blob(n_side=11, dx=0.006, radius=0.026):
    grid = torch.arange(n_side, dtype=torch.float64) * dx
    ys, xs = torch.meshgrid(grid, grid, indexing="ij")
    pos = torch.stack([xs.flatten(), ys.flatten()], dim=1)
    keep = torch.linalg.vector_norm(pos - 0.5 * (n_side - 1) * dx, dim=1) < radius
    return pos[keep]


def test_a_lattice_blob_has_a_deficient_shell():
    """Precondition for the parity test below: the closure must actually fire."""
    cfg = _cfg(dx=0.006)
    pos = _blob()
    mass = torch.full((pos.shape[0],), RHO0 * cfg.dx ** 2, dtype=pos.dtype)
    rho = density_from_neighbor_pairs(pos, mass, cfg.h_factor * cfg.dx)
    deficit = (rho / RHO0) < cfg.free_surface_density_ratio
    assert bool(deficit.any())
    assert float(deficit.to(torch.float64).mean()) > 0.1


def test_the_pair_and_dense_paths_agree_with_the_closure_on():
    cfg = _cfg(dx=0.006)
    pos = _blob()
    vel = 0.05 * torch.stack([pos[:, 1] - pos[:, 1].mean(),
                             -(pos[:, 0] - pos[:, 0].mean())], dim=1)
    mass = torch.full((pos.shape[0],), RHO0 * cfg.dx ** 2, dtype=pos.dtype)

    dense = rhs_all_pairs(pos, vel, mass, cfg)
    paired = rhs(pos, vel, mass, cfg)
    for actual, expected in zip(paired, dense):
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-9)


def test_the_pressure_returned_by_rhs_is_never_tensile():
    cfg = _cfg(dx=0.006)
    pos = _blob()
    vel = torch.zeros_like(pos)
    mass = torch.full((pos.shape[0],), RHO0 * cfg.dx ** 2, dtype=pos.dtype)
    rho, pressure, _ = rhs(pos, vel, mass, cfg)
    assert bool((pressure >= 0.0).all())
    assert bool(torch.isfinite(pressure).all())
    # The closure must give the shell more pressure than the clamped Tait law.
    open_cfg = _cfg(dx=0.006, free_surface_pressure=False)
    rho_o, pressure_o, _ = rhs(pos, vel, mass, open_cfg)
    assert float(pressure.sum()) > float(pressure_o.sum())
    assert math.isfinite(float(pressure.max()))