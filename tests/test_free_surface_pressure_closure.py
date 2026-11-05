"""Closed-form gates for the free-surface pressure closure.

``tensorsph.droplet_impact.free_surface_pressure`` extrapolates the bulk
pressure onto the shell where density summation under-counts neighbours
(Adami, Ihmsen & Adams 2012, §3.3). The tests below avoid the SPH relaxation
entirely: the closure is a purely algebraic operator on ``(rho, p, pairs)``, so
it can be pinned against exact statements.

The strongest one is exactness on a linear pressure field. In a hydrostatic
field ``p(y) = rho0 g (y_top - y)`` the body term inside the extrapolation,
``rho_j g . (r_i - r_j)``, reproduces the same straight line, so *every* donor
gives the same answer and the kernel weights cancel. A correct implementation
therefore returns the analytic pressure at the receiver to round-off, for any
donor set and any number of layers; a missing or mis-signed body term does not,
which is what the wall-pressure recurrence (0.7.0 P1) failed on.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorsph.droplet_impact import (
    Config,
    free_surface_pressure,
    rhs,
    rhs_all_pairs,
    tait_pressure,
)
from tensorsph.neighbor_search import cell_linked_pairs

RHO0 = 1000.0
GRAV = 9.81


def _lattice(nx: int, ny: int, dx: float = 0.01, x0: float = 0.0,
             y0: float = 0.0, dtype=torch.float64):
    xs = x0 + dx * torch.arange(nx, dtype=dtype)
    ys = y0 + dx * torch.arange(ny, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)


def _pairs(pos, h):
    return cell_linked_pairs(pos, 2.0 * h)


def _weights(pos, pairs, h):
    from tensorsph.droplet_impact import pair_kernels

    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    wij, _ = pair_kernels(delta, h)
    return delta, wij, i, j


def _hydrostatic_case(ny=25, nx=4, dx=0.01, n_free=3, **cfg_kw):
    """Column lattice with an analytic hydrostatic pressure and a free top."""
    pos = _lattice(nx, ny, dx)
    h = 1.3 * dx
    y_top = float(pos[:, 1].max())
    rho = torch.full((pos.shape[0],), RHO0, dtype=pos.dtype)
    p = RHO0 * GRAV * (y_top - pos[:, 1])
    order = torch.argsort(pos[:, 1], descending=True)
    free = torch.zeros(pos.shape[0], dtype=torch.bool)
    free[order[: nx * n_free]] = True
    rho[free] = 0.97 * RHO0           # deficit that triggers the closure
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, gravity=GRAV, **cfg_kw)
    return pos, rho, p, free, h, cfg


# --------------------------------------------------------------------------- #
# exactness on a linear field (the body term is what makes this hold)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("layers", [1, 2, 3])
def test_linear_hydrostatic_field_is_reproduced_exactly(layers):
    """Exactness holds for one-sided ("interior") donors only.

    With ``donors="all"`` the free-surface neighbours enter with their clamped
    p = 0 *and* their deficient density, so the body term ``rho_j g.(r_i-r_j)``
    no longer lies on the analytic line; that variant is a different (diluted)
    operator and is pinned by
    :func:`test_all_donors_is_diluted_by_free_surface_zeros` instead.
    """
    donors = "interior"
    pos, rho, p, free, h, cfg = _hydrostatic_case(
        surface_pressure_closure=True, surface_pressure_donors=donors,
        surface_pressure_layers=layers)
    delta, wij, i, j = _weights(pos, _pairs(pos, h), h)
    out = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg)
    # The top row sits at p = 0 by construction; every other receiver is filled
    # (not left at its clamped value) and equals the analytic field to round-off.
    filled = free & (p > 1.0)
    assert bool(filled.any())
    assert float(out[filled].min()) > 0.0
    rel = (out[filled] - p[filled]) / p[filled]
    assert float(rel.abs().max()) < 1e-10
    # Interior particles are untouched.
    assert torch.equal(out[~free], p[~free])


def test_body_term_sign_matters():
    """Flipping the gravity term must break the exactness above.

    This is the same negative control as the wall-pressure test: it fails if the
    body term is written with the wrong sign, or dropped (gravity = 0).
    """
    pos, rho, p, free, h, cfg = _hydrostatic_case(surface_pressure_closure=True)
    delta, wij, i, j = _weights(pos, _pairs(pos, h), h)
    good = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg)

    cfg_no_g = Config(dx=cfg.dx, h_factor=1.3, rho0=RHO0, gravity=0.0,
                      surface_pressure_closure=True)
    filled = free & (p > 1.0)
    dropped = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg_no_g)
    assert float((good[filled] - p[filled]).abs().max()) < 1e-10
    assert float((dropped[filled] - p[filled]).abs().max()) > 1.0  # Pa


# --------------------------------------------------------------------------- #
# donor semantics
# --------------------------------------------------------------------------- #
def test_all_donors_is_diluted_by_free_surface_zeros():
    """Literal Adami weights in the neighbours' clamped p = 0; "interior" does not.

    On a convex surface the "all" variant therefore returns a lower value. The
    closure used by the Young-Laplace benchmark should be the one-sided one.
    """
    pos, rho, p, free, h, cfg = _hydrostatic_case(
        surface_pressure_closure=True, surface_pressure_donors="interior")
    delta, wij, i, j = _weights(pos, _pairs(pos, h), h)
    interior = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg)
    cfg_all = Config(dx=cfg.dx, h_factor=1.3, rho0=RHO0, gravity=GRAV,
                     surface_pressure_closure=True,
                     surface_pressure_donors="all")
    allv = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg_all)
    assert float(allv[free].mean()) < float(interior[free].mean())


def test_layers_fill_the_outermost_shell():
    """A surface particle with no bulk neighbour needs the layer below filled.

    The column has bulk up to y = 0.02 and free particles at y = 0.03 .. 0.07
    (the support radius is 2h = 0.026). One pass fills y = 0.03 only; each
    further pass extends the filled band by roughly one spacing, and five rows
    are filled after three passes. With a single pass the outermost particles
    keep their own (clamped) pressure -- this is what
    ``surface_pressure_layers`` exists for.
    """
    dx, h = 0.01, 0.013
    pos = _lattice(nx=6, ny=8, dx=dx)
    y = pos[:, 1]
    free = y > 0.025
    rho = torch.where(free, torch.tensor(0.98 * RHO0, dtype=pos.dtype),
                      torch.tensor(RHO0, dtype=pos.dtype))
    p = torch.where(free, torch.zeros_like(rho), torch.full_like(rho, 150.0))
    delta, wij, i, j = _weights(pos, _pairs(pos, h), h)

    def run(layers):
        cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, gravity=0.0,
                     surface_pressure_closure=True,
                     surface_pressure_layers=layers)
        return free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg)

    one, three = run(1), run(3)

    def at(out, y_row):
        return float(out[(y > y_row - 1e-9) & (y < y_row + 1e-9)].mean())

    # One-sided extrapolation approaches the bulk value from below, never above.
    assert at(three, 0.03) == pytest.approx(150.0, rel=1e-6)
    assert float(three.max()) <= 150.0 + 1e-9
    # Pass 1 fills exactly the row adjacent to the bulk; the rest stay clamped.
    assert at(one, 0.03) == pytest.approx(150.0, rel=1e-6)
    assert at(one, 0.05) == 0.0 and at(one, 0.07) == 0.0
    # Three passes reach the outermost row and are monotone in the layer count.
    assert at(three, 0.05) == pytest.approx(150.0, rel=1e-6)
    assert at(three, 0.07) == pytest.approx(150.0, rel=1e-6)
    assert at(one, 0.07) <= at(run(2), 0.07) <= at(three, 0.07)


def test_no_free_surface_is_a_no_op():
    pos = _lattice(5, 5, 0.01)
    h = 0.013
    rho = torch.full((25,), RHO0, dtype=torch.float64)
    p = torch.arange(25, dtype=torch.float64)
    cfg = Config(dx=0.01, h_factor=1.3, rho0=RHO0, surface_pressure_closure=True)
    delta, wij, i, j = _weights(pos, _pairs(pos, h), h)
    out = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg)
    assert torch.equal(out, p)


def test_result_is_clamped_non_negative():
    """Extrapolation with all donors empty must not produce tensile pressure."""
    pos = _lattice(4, 4, 0.01)
    h = 0.013
    rho = torch.full((16,), 0.97 * RHO0, dtype=torch.float64)   # everything free
    p = torch.zeros_like(rho)
    cfg = Config(dx=0.01, h_factor=1.3, rho0=RHO0, gravity=0.0,
                 surface_pressure_closure=True,
                 surface_pressure_donors="all")
    delta, wij, i, j = _weights(pos, _pairs(pos, h), h)
    out = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg)
    assert float(out.min()) >= 0.0
    assert torch.equal(out, torch.zeros_like(out))


# --------------------------------------------------------------------------- #
# wiring: default off, and the two rhs paths agree when it is on
# --------------------------------------------------------------------------- #
def _cloud(n=140, seed=7, radius=0.03):
    gen = torch.Generator().manual_seed(seed)
    r = radius * torch.sqrt(torch.rand(n, generator=gen, dtype=torch.float64))
    t = 2.0 * math.pi * torch.rand(n, generator=gen, dtype=torch.float64)
    return torch.stack((r * torch.cos(t), r * torch.sin(t)), dim=1)


def test_closure_is_off_by_default():
    assert Config().surface_pressure_closure is False


def test_rhs_pair_and_dense_paths_agree_with_closure_on():
    pos = _cloud()
    dx = 0.004
    h = 1.3 * dx
    mass = torch.full((pos.shape[0],), RHO0 * dx * dx, dtype=pos.dtype)
    vel = torch.zeros_like(pos)
    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, gravity=0.0,
                 surface_tension_coefficient=0.0, dt=1e-5,
                 surface_pressure_closure=True, surface_pressure_layers=2)
    pairs = _pairs(pos, h)
    rho_p, p_p, acc_p = rhs(pos, vel, mass, cfg, pairs=pairs)
    rho_d, p_d, acc_d = rhs_all_pairs(pos, vel, mass, cfg)
    assert torch.allclose(rho_p, rho_d, rtol=1e-9, atol=1e-9)
    assert torch.allclose(p_p, p_d, rtol=1e-6, atol=1e-6)
    assert torch.allclose(acc_p, acc_d, rtol=1e-5, atol=1e-6)


def test_closure_changes_only_the_surface_pressure():
    """Inside the drop the closure must be inert: bulk pressure is the Tait value."""
    pos = _cloud()
    dx = 0.004
    h = 1.3 * dx
    mass = torch.full((pos.shape[0],), RHO0 * dx * dx, dtype=pos.dtype)
    vel = torch.zeros_like(pos)
    base = dict(dx=dx, h_factor=1.3, rho0=RHO0, gravity=0.0, dt=1e-5)
    cfg_off = Config(**base, surface_pressure_closure=False)
    cfg_on = Config(**base, surface_pressure_closure=True)
    pairs = _pairs(pos, h)
    rho_off, p_off, _ = rhs(pos, vel, mass, cfg_off, pairs=pairs)
    rho_on, p_on, _ = rhs(pos, vel, mass, cfg_on, pairs=pairs)
    assert torch.equal(rho_off, rho_on)            # density is untouched
    core = rho_on > 0.99 * RHO0
    assert core.any() and (~core).any()
    assert torch.equal(p_on[core], p_off[core])
    assert torch.equal(p_on[core], tait_pressure(rho_on, cfg_on)[core])
    assert float(p_on[~core].mean()) >= float(p_off[~core].mean())
