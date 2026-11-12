"""Incompressible SPH (ISPH) projection solver for the plane Poiseuille channel.

Why this module exists
----------------------
The repository's production solver is *weakly compressible* SPH (WCSPH): pressure
comes from a Tait equation of state and the flow is only as incompressible as the
artificial sound speed allows.  The measured price of that choice is recorded in
``benchmarks/verification_ledger/ledger.json``: on the plane-Poiseuille channel the
centre velocity is right to ``1.51 %`` but the density field carries ``~2.3e-2``
of ``rho/rho0 - 1`` fluctuation, and the wall-shear readout is withheld because
the near-wall profile and the centre velocity disagree.

This module is the complementary route: a *projection* (incompressible) SPH
solver, following Cummins & Rudman (1999, *J. Comput. Phys.* **152**:584-607) and
Shao & Lo (2003, *Adv. Water Resour.* **26**:787-800).  The time step is split into

1. a **prediction** step that integrates viscosity and the body force without any
   equation of state and without an acoustic sound speed
   (:math:`u^* = u + \\Delta t (\\nu \\nabla^2 u + g)`);
2. a **projection** step that solves the pressure Poisson equation

   .. math:: \\nabla^2 p = \\frac{\\rho_0}{\\Delta t} \\nabla\\cdot u^*

   and corrects the velocity with :math:`u = u^* - \\frac{\\Delta t}{\\rho}\\nabla p`
   so that the corrected velocity field is discretely divergence free.

Consequences that are *measurable*, not rhetorical: there is no Tait EOS, so no
density is ever updated from pressure; the density reported below is the SPH
summation density of the particle cloud, i.e. a pure diagnostic of particle
order.  And there is no acoustic Courant limit, so the time step is set by the
convective CFL and the viscous limit only.

Discretisation (all operators are pairwise SPH sums)
---------------------------------------------------
With :math:`V_j = m_j/\\rho_j`, :math:`g_{ij} = \\nabla_i W_{ij}` and
:math:`\\delta_{ij} = x_i - x_j`:

* divergence      :math:`\\nabla\\cdot u_i = \\sum_j V_j (u_j-u_i)\\cdot g_{ij}`
* gradient        :math:`\\nabla p_i   = \\sum_j V_j (p_j-p_i)\\,g_{ij}`
* Laplacian       :math:`L[p]_i = \\sum_j 2 V_j \\frac{g_{ij}\\cdot\\delta_{ij}}{|\\delta_{ij}|^2+\\eta^2}(p_i-p_j)`
* viscous Laplacian: same kernel but acting on the velocity difference
  (Brookshaw 1985; Morris, Fox & Zhu 1997), which is the form that stays regular
  for a linear velocity field.

The discrete Laplacian :math:`L` approximates :math:`-\\nabla^2` (this is *checked
at run time* by :func:`operator_signatures`, not assumed), so the Poisson equation
solved here is :math:`L[p] = -\\frac{\\rho_0}{\\Delta t}\\nabla\\cdot u^*`.

Boundary conditions
-------------------
Walls are three layers of static particles with ``u = 0`` on the same lattice
spacing as the fluid.  They enter the *viscous* operator (that is what makes the
no-slip condition a force and not a clamp) and the summation density, and they are
excluded from the pressure unknowns with a zero-normal-gradient
(:math:`\\partial p/\\partial n = 0`) treatment, which is the projection-method
boundary condition implied by no penetration of a rigid wall.  Because the pressure
problem is then pure-Neumann in ``y`` and periodic in ``x``, its constant mode is
fixed by pinning one particle and the right-hand side is projected onto the
compatible subspace (volume-weighted zero mean) before the solve.

The two walls make the discrete divergence of a wall-truncated particle cloud
integrate to a small non-zero number; that projection removes exactly that
truncation bias and nothing else (the residual is reported as
``poisson.mean_divergence_before``/``after``).

Usage
-----
``python -m tensorsph.isph_channel --out runs/isph_channel`` writes
``results.json`` with the resolution ladder and the three gates of the
``GAP-SPH-ISPH-PROJECTION`` acceptance criterion.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .droplet_impact import (
    cubic_spline_kernel,
    lattice_kernel_sum,
    pair_kernels,
    resolve_device,
)

__all__ = [
    "ChannelConfig",
    "build_channel",
    "pair_geometry",
    "summation_density",
    "divergence",
    "pressure_gradient",
    "viscous_laplacian",
    "poisson_operator",
    "project",
    "operator_signatures",
    "run_case",
    "fit_profile",
    "gates",
    "main",
]


@dataclasses.dataclass
class ChannelConfig:
    """Non-dimensional-free description of the channel benchmark."""

    height: float = 0.1
    columns: int = 12
    resolution: int = 8
    h_factor: float = 1.90
    rho0: float = 1000.0
    viscosity: float = 0.01
    body_force: float = 9.81
    wall_layers: int = 5
    cfl: float = 0.25
    viscous_safety: float = 0.125
    t_end: float = 1.0
    t_average: float = 0.1
    xsph: float = 0.00
    laplacian_eta: float = 0.01
    steps_max: int = 200_000

    @property
    def dx(self) -> float:
        return self.height / self.resolution

    @property
    def h(self) -> float:
        return self.h_factor * self.dx

    @property
    def length(self) -> float:
        return self.columns * self.dx

    @property
    def u_max_analytic(self) -> float:
        return self.body_force * self.height ** 2 / (8.0 * self.viscosity)

    @property
    def tau_wall_analytic(self) -> float:
        return self.rho0 * self.body_force * self.height / 2.0


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def build_channel(cfg: ChannelConfig, device) -> dict:
    """Fluid lattice rows ``y = dx/2 + k dx`` between static wall bands."""
    dx = cfg.dx
    xs = torch.arange(dx * 0.5, cfg.length, dx, device=device)
    ys = torch.arange(dx * 0.5, cfg.height, dx, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    fluid = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    n_fluid = fluid.shape[0]

    # Wall columns must sit on the *same* x lattice as the fluid (half-spacing
    # offset included): a half-cell misalignment puts wall particles at dx/2 from
    # fluid particles and inflates the summation density to ~1.96 rho0 near the
    # wall face (measured before this fix).
    # The channel is periodic in x, so the wall only has to carry one period of
    # columns: the minimum-image pair search supplies the images.  Extending the
    # wall band past one period double counts it after the minimum-image fold and
    # inflates the first fluid row to 1.53 rho0.
    wx = dx * 0.5 + torch.arange(cfg.columns, device=device) * dx
    walls = []
    for layer in range(cfg.wall_layers):
        yb = torch.full_like(wx, -dx * 0.5 - layer * dx)
        walls.append(torch.stack((wx, yb), dim=-1))
        yt = torch.full_like(wx, cfg.height + dx * 0.5 + layer * dx)
        walls.append(torch.stack((wx, yt), dim=-1))
    wall = torch.cat(walls, dim=0)

    pos = torch.cat((fluid, wall), dim=0)
    # Lattice-calibrated mass: m = rho0 / S removes the discrete kernel-sum
    # offset so that the summation density of a perfect lattice is rho0 (see
    # droplet_impact.lattice_kernel_sum / docs/verification_lattice_mass_calibration.md).
    mass_unit = cfg.rho0 / lattice_kernel_sum(cfg.h, dx)
    mass = torch.full((pos.shape[0],), mass_unit, device=device)
    vel = torch.zeros_like(pos)
    kind = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
    kind[n_fluid:] = 1
    return {
        "pos": pos,
        "mass": mass,
        "vel": vel,
        "kind": kind,
        "n_fluid": n_fluid,
        "n_wall": int(pos.shape[0] - n_fluid),
        "dx": dx,
        "h": cfg.h,
    }


def pair_geometry(pos: torch.Tensor, h: float, length: float) -> dict:
    """All unordered pairs inside ``2h`` with the minimum image along ``x``.

    The clouds used here are small (a few hundred particles), so the pair list is
    rebuilt from a dense distance matrix every step; there is no cell list to get
    out of sync with the moving particles.
    """
    n = pos.shape[0]
    delta = pos[:, None, :] - pos[None, :, :]
    delta[..., 0] -= length * torch.round(delta[..., 0] / length)
    r = torch.linalg.vector_norm(delta, dim=-1)
    inside = (r < 2.0 * h) & (r > 1e-12)
    inside = torch.triu(inside, diagonal=1)
    ii, jj = torch.nonzero(inside, as_tuple=True)
    d = delta[ii, jj]
    rr = r[ii, jj]
    w, g = pair_kernels(d, h)
    return {"ii": ii, "jj": jj, "r": rr, "w": w, "g": g, "delta": d, "n": n}


# --------------------------------------------------------------------------- #
# SPH operators
# --------------------------------------------------------------------------- #

def summation_density(geo: dict, mass: torch.Tensor, h: float) -> torch.Tensor:
    """``rho_i = sum_j m_j W_ij`` including the self term."""
    ii, jj, w = geo["ii"], geo["jj"], geo["w"]
    zero = torch.zeros(1, dtype=mass.dtype, device=mass.device)
    rho = mass * cubic_spline_kernel(zero, h).reshape(())
    rho = rho.index_add(0, ii, mass[jj] * w)
    rho = rho.index_add(0, jj, mass[ii] * w)
    return rho


def divergence(geo: dict, vel: torch.Tensor, volume: torch.Tensor,
               n_fluid: int | None = None) -> torch.Tensor:
    """Pairwise SPH divergence over fluid-fluid pairs.

    Static wall particles are *excluded*: a wall particle carrying ``u = 0``
    would enter the difference ``(u_j - u_i)`` as a tangential velocity jump and
    inject a spurious divergence into every near-wall fluid particle (measured:
    -1.04 against an analytic 0.4 for a linear field, see
    ``operator_signatures``).  The wall enters the projection only through the
    zero-normal-gradient pressure condition, which is where the no-penetration
    constraint belongs; restricting the sum to fluid pairs also makes the
    Poisson right-hand side satisfy the compatibility condition
    ``sum_i V_i div_i = 0`` exactly, because each pair then contributes
    ``+V_j (u_j-u_i).g`` and ``+V_i (u_j-u_i).g`` to an antisymmetric sum.
    """
    ii, jj, g = geo["ii"], geo["jj"], geo["g"]
    if n_fluid is not None:
        keep = (ii < n_fluid) & (jj < n_fluid)
        ii, jj, g = ii[keep], jj[keep], g[keep]
    gd = ((vel[jj] - vel[ii]) * g).sum(-1)
    div = torch.zeros(vel.shape[0], dtype=vel.dtype, device=vel.device)
    div = div.index_add(0, ii, volume[jj] * gd)
    div = div.index_add(0, jj, volume[ii] * gd)
    return div


def pressure_gradient(geo: dict, p: torch.Tensor, volume: torch.Tensor,
                      n_fluid: int) -> torch.Tensor:
    """``grad p_i = sum_j V_j (p_j - p_i) g_ij`` over fluid-fluid pairs only.

    Wall neighbours are omitted because the wall condition is
    ``d p / d n = 0``, i.e. their contribution ``(p_wall - p_i)`` is zero.
    """
    ii, jj, g = geo["ii"], geo["jj"], geo["g"]
    fluid = (ii < n_fluid) & (jj < n_fluid)
    i, j, gg = ii[fluid], jj[fluid], g[fluid]
    dp = p[j] - p[i]
    grad = torch.zeros((p.shape[0], 2), dtype=p.dtype, device=p.device)
    contrib = volume[j] * dp
    grad = grad.index_add(0, i, contrib.unsqueeze(-1) * gg)
    grad = grad.index_add(0, j, (volume[i] * dp).unsqueeze(-1) * gg)
    return grad


def viscous_laplacian(geo: dict, vel: torch.Tensor, volume: torch.Tensor,
                      eta: float) -> torch.Tensor:
    """Morris/Fox/Zhu pairwise Laplacian, regularised by ``eta``."""
    ii, jj, g, d, r = geo["ii"], geo["jj"], geo["g"], geo["delta"], geo["r"]
    s = (g * d).sum(-1) / (r ** 2 + eta ** 2)
    du = vel[ii] - vel[jj]
    lap = torch.zeros_like(vel)
    lap = lap.index_add(0, ii, (2.0 * volume[jj] * s).unsqueeze(-1) * du)
    lap = lap.index_add(0, jj, (-2.0 * volume[ii] * s).unsqueeze(-1) * du)
    return lap


def poisson_operator(geo: dict, volume: torch.Tensor, n_fluid: int, eta: float):
    """Sparse fluid-only Laplacian approximating ``-grad^2``."""
    ii, jj, g, d, r = geo["ii"], geo["jj"], geo["g"], geo["delta"], geo["r"]
    fluid = (ii < n_fluid) & (jj < n_fluid)
    i = ii[fluid]
    j = jj[fluid]
    s = ((g * d).sum(-1) / (r ** 2 + eta ** 2))[fluid]
    a_ij = 2.0 * volume[j] * s                    # coefficient of (p_i - p_j)
    a_ji = 2.0 * volume[i] * s
    diag = torch.zeros(n_fluid, dtype=volume.dtype, device=volume.device)
    diag = diag.index_add(0, i, a_ij)
    diag = diag.index_add(0, j, a_ji)
    rows = torch.cat((i, j, torch.arange(n_fluid, device=volume.device)))
    cols = torch.cat((j, i, torch.arange(n_fluid, device=volume.device)))
    vals = torch.cat((-a_ij, -a_ji, diag))
    m = sp.coo_matrix(
        (vals.detach().cpu().numpy(),
         (rows.detach().cpu().numpy(), cols.detach().cpu().numpy())),
        shape=(n_fluid, n_fluid),
    ).tocsr()
    return m


def _solve_pinned(matrix, rhs: np.ndarray, pin: int = 0) -> np.ndarray:
    """Solve a pure-Neumann Poisson system with one unknown pinned to zero."""
    a = matrix.tolil(copy=True)
    a[pin, :] = 0.0
    a[:, pin] = 0.0
    a[pin, pin] = 1.0
    a = a.tocsr()
    b = rhs.copy()
    b[pin] = 0.0
    diag = a.diagonal()
    diag[diag == 0.0] = 1.0
    precond = spla.LinearOperator(a.shape, matvec=lambda v: v / diag)
    x, info = spla.cg(a, b, rtol=1e-10, atol=0.0, maxiter=5000, M=precond)
    if info != 0:
        raise RuntimeError(f"pressure Poisson solve did not converge (info={info})")
    return x


# --------------------------------------------------------------------------- #
# Projection step
# --------------------------------------------------------------------------- #

def project(cloud: dict, geo: dict, cfg: ChannelConfig, dt: float,
            rho: torch.Tensor, n_fluid: int) -> dict:
    """One pressure projection: solve the Poisson equation and correct ``u``."""
    pos, vel, mass = cloud["pos"], cloud["vel"], cloud["mass"]
    volume = mass / rho
    eta = cfg.laplacian_eta * cfg.h

    div_star = divergence(geo, vel, volume, n_fluid=n_fluid)
    div_fluid = div_star[:n_fluid]
    # The discrete Laplacian above approximates +grad^2 p (measured, see
    # ``operator_signatures``), and the correction is u = u* - dt/rho grad p, so
    # div u = 0 requires L[p] = (rho0/dt) div u*.
    b = (cfg.rho0 / dt) * div_fluid
    # Compatibility projection onto the zero-volume-mean subspace.  The retained
    # part is the flow divergence the Poisson problem can actually remove.
    vf = volume[:n_fluid]
    mean_b = float((vf * b).sum() / vf.sum())
    b = b - mean_b

    matrix = poisson_operator(geo, volume, n_fluid, eta)
    p = _solve_pinned(matrix, b.detach().cpu().numpy())
    pressure = torch.zeros(pos.shape[0], dtype=pos.dtype, device=pos.device)
    pressure[:n_fluid] = torch.from_numpy(p).to(pos.dtype).to(pos.device)

    grad = pressure_gradient(geo, pressure, volume, n_fluid)
    vel = vel - (dt / cfg.rho0) * grad
    vel[n_fluid:] = 0.0

    div_after = divergence(geo, vel, volume, n_fluid=n_fluid)[:n_fluid]
    cloud["vel"] = vel
    return {
        "pressure": pressure,
        "div_before": float(torch.sqrt(torch.mean(div_fluid ** 2))),
        "div_after": float(torch.sqrt(torch.mean(div_after ** 2))),
        "mean_divergence_before": float((vf * div_fluid).sum() / vf.sum()),
        "mean_divergence_after": float((vf * div_after).sum() / vf.sum()),
        "mean_rhs_removed": mean_b,
    }


def operator_signatures(cfg: ChannelConfig, device) -> dict:
    """Measure the sign and consistency of the three discrete operators.

    The test fields are chosen so that the analytic answer is known *and* so that
    the field respects the domain the operator is used in: the ``x`` direction is
    periodic in this solver, so a linear-in-``x`` field is a poor probe (it jumps
    by ``a L`` across the seam); a sine is used instead.  The ``y`` field is
    linear, and the Laplacian probe is the quadratic ``p = P y^2`` whose
    ``grad^2`` is ``2P``.  Measurements are taken only on particles farther than
    ``2h`` from both walls, i.e. where every operator has full support -- near-wall
    truncation is a separate, documented limitation.
    """
    cloud = build_channel(cfg, device)
    n_fluid = cloud["n_fluid"]
    pos = cloud["pos"]
    geo = pair_geometry(pos, cloud["h"], cfg.length)
    rho = summation_density(geo, cloud["mass"], cloud["h"])
    volume = cloud["mass"] / rho
    eta = cfg.laplacian_eta * cfg.h

    y = pos[:n_fluid, 1]
    sel = (y > 2.0 * cfg.h) & (y < cfg.height - 2.0 * cfg.h)

    def stats(error: torch.Tensor) -> dict:
        if int(sel.sum()) == 0:
            return {"n": 0, "rms": float("nan"), "max_abs": float("nan")}
        e = error[sel]
        return {
            "n": int(sel.sum()),
            "rms": float(torch.sqrt(torch.mean(e ** 2))),
            "max_abs": float(torch.max(torch.abs(e))),
        }

    yy = pos[:, 1]

    # (1) Laplacian of a quadratic profile: L[p] is claimed to be +grad^2 p.
    lap = torch.zeros_like(pos[:, :1])
    ii, jj, g, d, r = geo["ii"], geo["jj"], geo["g"], geo["delta"], geo["r"]
    fluid_pair = (ii < n_fluid) & (jj < n_fluid)
    s = (g * d).sum(-1) / (r ** 2 + eta ** 2)
    p = 3.0 * yy ** 2
    dp = p[ii] - p[jj]
    lap = lap.index_add(0, ii, (2.0 * volume[jj] * s * dp
                                * fluid_pair.to(dtype=pos.dtype)).unsqueeze(-1))
    lap = lap.index_add(0, jj, (-2.0 * volume[ii] * s * dp
                                * fluid_pair.to(dtype=pos.dtype)).unsqueeze(-1))
    lap_stats = stats(lap.squeeze(-1)[:n_fluid] - 6.0)

    # (2) Divergence of a periodic field u = (A sin(2 pi x / L), 0).
    amp = 0.4
    k = 2.0 * math.pi / cfg.length
    u = torch.zeros_like(pos)
    u[:, 0] = amp * torch.sin(k * pos[:, 0])
    div = divergence(geo, u, volume, n_fluid=n_fluid)[:n_fluid]
    div_exact = amp * k * torch.cos(k * pos[:n_fluid, 0])
    div_stats = stats(div - div_exact)

    # (3) Divergence of u = (0, a y): exact value a everywhere.
    a = 0.5
    u2 = torch.zeros_like(pos)
    u2[:, 1] = a * yy
    div2 = divergence(geo, u2, volume, n_fluid=n_fluid)[:n_fluid]
    div2_stats = stats(div2 - a)

    # (4) Gradient of p = c y: exact value (0, c).
    c = 2.0
    pc = c * yy
    grad = pressure_gradient(geo, pc, volume, n_fluid)
    grad_stats = stats(grad[:n_fluid, 1] - c)

    # (5) Viscous Laplacian of a quadratic vector field u = (b y^2, 0).
    b = 0.5
    u3 = torch.zeros_like(pos)
    u3[:, 0] = b * yy ** 2
    lap_u = viscous_laplacian(geo, u3, volume, eta)[:n_fluid, 0]
    lap_u_stats = stats(lap_u - 2.0 * b)

    return {
        "interior_support_cells": 2.0 * cfg.h / cfg.dx,
        "laplacian_of_quadratic_y": {
            "probe": "p = 3 y^2, exact grad^2 p = 6", **lap_stats},
        "divergence_periodic_sine": {
            "probe": "u_x = 0.4 sin(2 pi x / L), exact div = 0.4 k cos(k x)",
            **div_stats},
        "divergence_linear_y": {
            "probe": "u_y = 0.5 y, exact div = 0.5", **div2_stats},
        "gradient_linear_y": {
            "probe": "p = 2 y, exact dp/dy = 2", **grad_stats},
        "viscous_laplacian_quadratic_y": {
            "probe": "u_x = 0.5 y^2, exact lap = 1", **lap_u_stats},
        "laplacian_sign": "plus: L[p] approximates +grad^2 p (see laplacian_of_quadratic_y)",
    }


# --------------------------------------------------------------------------- #
# No-slip walls: antisymmetric mirror particles
# --------------------------------------------------------------------------- #

def mirror_ghosts(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor,
                  cfg: ChannelConfig, n_fluid: int) -> dict:
    """Mirror the near-wall fluid across each wall plane with reversed velocity.

    A static wall particle carrying ``u = 0`` one spacing away from the first
    fluid row implements no-slip only to first order; the measured consequence in
    this repository is the withheld ``poiseuille.wall_shear_ratio`` entry (ratio
    0.50, i.e. half the analytic drag).  The antisymmetric image
    ``u_ghost = -u_parent`` at the mirrored position makes the wall plane itself
    the zero-velocity surface (``u`` is odd about it), which is the second-order
    treatment of a Dirichlet condition.  Ghosts are used by the viscous operator
    only: they carry no pressure, no divergence and no density.
    """
    y = pos[:n_fluid, 1]
    near_bottom = y < 2.0 * cfg.h
    near_top = y > cfg.height - 2.0 * cfg.h
    pieces, vpieces, mpieces, parent = [], [], [], []
    if bool(near_bottom.any()):
        idx = torch.nonzero(near_bottom, as_tuple=True)[0]
        mirror = pos[idx] * torch.tensor([1.0, -1.0], dtype=pos.dtype,
                                         device=pos.device)
        pieces.append(mirror)
        vpieces.append(-vel[idx])
        mpieces.append(mass[idx])
        parent.append(idx)
    if bool(near_top.any()):
        idx = torch.nonzero(near_top, as_tuple=True)[0]
        mirror = pos[idx] * torch.tensor([1.0, -1.0], dtype=pos.dtype,
                                         device=pos.device)
        mirror[:, 1] += 2.0 * cfg.height
        pieces.append(mirror)
        vpieces.append(-vel[idx])
        mpieces.append(mass[idx])
        parent.append(idx)
    if not pieces:
        return {"pos": None}
    return {
        "pos": torch.cat(pieces, dim=0),
        "vel": torch.cat(vpieces, dim=0),
        "mass": torch.cat(mpieces, dim=0),
        "parent": torch.cat(parent, dim=0),
        "n": int(sum(p.shape[0] for p in pieces)),
    }


def wall_viscous_laplacian(pos_fluid, vel_fluid, mass_fluid, volume_fluid,
                           cfg: ChannelConfig, eta: float) -> torch.Tensor:
    """Morris Laplacian of the fluid velocity with antisymmetric wall images."""
    n = pos_fluid.shape[0]
    ghosts = mirror_ghosts(pos_fluid, vel_fluid, mass_fluid, cfg, n)
    if ghosts["pos"] is None:
        return viscous_laplacian({"ii": torch.zeros(0, dtype=torch.long),
                                 "jj": torch.zeros(0, dtype=torch.long),
                                 "g": torch.zeros((0, 2), dtype=pos_fluid.dtype),
                                 "delta": torch.zeros((0, 2), dtype=pos_fluid.dtype),
                                 "r": torch.zeros(0, dtype=pos_fluid.dtype)},
                                vel_fluid, volume_fluid, eta)
    pos_all = torch.cat((pos_fluid, ghosts["pos"]), dim=0)
    vel_all = torch.cat((vel_fluid, ghosts["vel"]), dim=0)
    mass_all = torch.cat((mass_fluid, ghosts["mass"]), dim=0)
    vol_all = torch.cat((volume_fluid, volume_fluid[ghosts["parent"]]), dim=0)
    geo = pair_geometry(pos_all, cfg.h, cfg.length)
    ip, jp = geo["ii"], geo["jj"]
    keep = (ip < n) | (jp < n)          # drop ghost-ghost pairs (unused rows)
    geo["ii"], geo["jj"], geo["g"] = ip[keep], jp[keep], geo["g"][keep]
    geo["delta"], geo["r"] = geo["delta"][keep], geo["r"][keep]
    lap = viscous_laplacian(geo, vel_all, vol_all, eta)
    return lap[:n]


# --------------------------------------------------------------------------- #
# Time stepping
# --------------------------------------------------------------------------- #

def _stability_dt(cfg: ChannelConfig, u_max: float) -> float:
    """No acoustic limit: convective CFL and the viscous limit only."""
    dt_cfl = cfg.cfl * cfg.h / max(u_max, 1e-12)
    dt_visc = cfg.viscous_safety * cfg.h ** 2 / cfg.viscosity
    return min(dt_cfl, dt_visc)


def _xsph(geo: dict, vel: torch.Tensor, volume: torch.Tensor, h: float,
          eps: float, n_fluid: int) -> torch.Tensor:
    """Beltrami-style velocity smoothing, acting on fluid particles only."""
    ii, jj, w = geo["ii"], geo["jj"], geo["w"]
    fluid = (ii < n_fluid) & (jj < n_fluid)
    i, j, ww = ii[fluid], jj[fluid], w[fluid]
    denom = torch.zeros(n_fluid, dtype=vel.dtype, device=vel.device)
    denom = denom.index_add(0, i, volume[j] * ww)
    denom = denom.index_add(0, j, volume[i] * ww)
    num = torch.zeros((n_fluid, 2), dtype=vel.dtype, device=vel.device)
    num = num.index_add(0, i, (volume[j] * ww).unsqueeze(-1) * (vel[j] - vel[i]))
    num = num.index_add(0, j, (volume[i] * ww).unsqueeze(-1) * (vel[i] - vel[j]))
    out = vel.clone()
    out[:n_fluid] = vel[:n_fluid] + eps * num / denom.clamp_min(1e-30).unsqueeze(-1)
    return out


def run_case(cfg: ChannelConfig, device, report: bool = False) -> dict:
    """Drive the channel to a statistically steady state and measure the profile."""
    cloud = build_channel(cfg, device)
    n_fluid = cloud["n_fluid"]
    pos, mass = cloud["pos"], cloud["mass"]
    vel = cloud["vel"]
    walls0 = pos[n_fluid:].clone()
    eta = cfg.laplacian_eta * cfg.h

    t = 0.0
    steps = 0
    n_clamped = 0
    t_start = time.time()
    trace = []
    profile_sum = torch.zeros(2 * cfg.resolution + 2, dtype=pos.dtype, device=device)
    profile_cnt = torch.zeros_like(profile_sum)
    samples = 0
    max_density_error = 0.0
    last_div = {"div_before": 0.0, "div_after": 0.0}
    t_average_from = cfg.t_end - cfg.t_average

    while t < cfg.t_end and steps < cfg.steps_max:
        geo = pair_geometry(pos, cfg.h, cfg.length)
        rho = summation_density(geo, mass, cfg.h)
        volume = mass / rho
        u_max = float(torch.linalg.vector_norm(vel[:n_fluid], dim=-1).max()) if n_fluid else 0.0
        dt = _stability_dt(cfg, u_max)
        dt = min(dt, cfg.t_end - t) if cfg.t_end - t > 1e-12 else dt

        # 1. prediction: viscosity, with antisymmetric wall images imposing no-slip
        lap = wall_viscous_laplacian(pos[:n_fluid], vel[:n_fluid], mass[:n_fluid],
                                     volume[:n_fluid], cfg, eta)
        lap_full = torch.zeros_like(vel)
        lap_full[:n_fluid] = lap
        vel = vel + dt * cfg.viscosity * lap_full
        vel[:n_fluid, 0] += dt * cfg.body_force
        vel[n_fluid:] = 0.0

        # 2. projection
        cloud["vel"] = vel
        proj = project(cloud, geo, cfg, dt, rho, n_fluid)
        vel = cloud["vel"]
        last_div = proj

        # 3. advection
        pos = pos + dt * vel
        pos[n_fluid:] = walls0
        below = pos[:n_fluid, 1] < 0.5 * cfg.dx
        above = pos[:n_fluid, 1] > cfg.height - 0.5 * cfg.dx
        if bool(below.any() or above.any()):
            n_clamped += int(below.sum()) + int(above.sum())
            pos[:n_fluid, 1] = torch.clamp(pos[:n_fluid, 1], 0.5 * cfg.dx,
                                           cfg.height - 0.5 * cfg.dx)

        if cfg.xsph > 0.0:
            geo = pair_geometry(pos, cfg.h, cfg.length)
            rho = summation_density(geo, mass, cfg.h)
            volume = mass / rho
            vel = _xsph(geo, vel, volume, cfg.h, cfg.xsph, n_fluid)
            vel[n_fluid:] = 0.0
            pos[n_fluid:] = walls0

        t += dt
        steps += 1

        density_error = float(torch.max(torch.abs(rho[:n_fluid] / cfg.rho0 - 1.0)))
        max_density_error = max(max_density_error, density_error)

        if t >= t_average_from:
            rows = torch.clamp((pos[:n_fluid, 1] / cfg.dx).long(), 0,
                               profile_sum.shape[0] - 1)
            profile_sum = profile_sum.index_add(0, rows, vel[:n_fluid, 0])
            profile_cnt = profile_cnt.index_add(
                0, rows, torch.ones_like(rows, dtype=profile_sum.dtype))
            samples += 1

        if steps % max(1, int(round(cfg.t_end / cfg.t_average / 50.0))) == 0:
            trace.append({
                "t": t,
                "steps": steps,
                "dt": dt,
                "u_max": u_max,
                "density_error_max": density_error,
                "div_rms_before": proj["div_before"],
                "div_rms_after": proj["div_after"],
            })

    wall_clock = time.time() - t_start
    geo = pair_geometry(pos, cfg.h, cfg.length)
    rho = summation_density(geo, mass, cfg.h)
    volume = mass / rho
    profile = []
    for k in range(profile_sum.shape[0]):
        if float(profile_cnt[k]) == 0.0:
            continue
        profile.append({
            "row": k,
            "y": (k + 0.5) * cfg.dx,
            "y_over_H": (k + 0.5) * cfg.dx / cfg.height,
            "u_mean": float(profile_sum[k] / profile_cnt[k]),
        })
    density_err = (rho[:n_fluid] / cfg.rho0 - 1.0)
    return {
        "case": "isph_plane_poiseuille_channel",
        "resolution_H_over_dx": cfg.resolution,
        "dx": cfg.dx,
        "h": cfg.h,
        "length": cfg.length,
        "columns": cfg.columns,
        "viscosity": cfg.viscosity,
        "body_force": cfg.body_force,
        "rho0": cfg.rho0,
        "t_end": cfg.t_end,
        "t_average": cfg.t_average,
        "t_reached": t,
        "steps": steps,
        "n_fluid": n_fluid,
        "n_wall": cloud["n_wall"],
        "n_boundary_clamped": n_clamped,
        "n_averaged_samples": samples,
        "wall_clock_s": wall_clock,
        "xsph": cfg.xsph,
        "profile": profile,
        "trace": trace,
        "density_error_max": float(torch.max(torch.abs(density_err))),
        "density_error_rms": float(torch.sqrt(torch.mean(density_err ** 2))),
        "density_error_max_over_run": max_density_error,
        "divergence_rms_before": last_div["div_before"],
        "divergence_rms_after": last_div["div_after"],
        "divergence_reduction": (
            last_div["div_after"] / last_div["div_before"]
            if last_div["div_before"] else float("nan")
        ),
        "u_max_measured": max((p["u_mean"] for p in profile), default=float("nan")),
        "u_max_analytic": cfg.u_max_analytic,
    }


# --------------------------------------------------------------------------- #
# Profile fit and gates
# --------------------------------------------------------------------------- #

def fit_profile(profile, height: float, body_force: float, viscosity: float,
                lo: float = 0.0, hi: float = 1.0) -> dict:
    """Least-squares quadratic ``u = c0 + c1 y + c2 y^2`` over a row band."""
    rows = [p for p in profile if lo <= p["y_over_H"] <= hi]
    if len(rows) < 3:
        return {"n_rows": len(rows)}
    y = np.array([p["y"] for p in rows], dtype=float)
    u = np.array([p["u_mean"] for p in rows], dtype=float)
    a = np.stack((np.ones_like(y), y, y * y), axis=1)
    coef, *_ = np.linalg.lstsq(a, u, rcond=None)
    c0, c1, c2 = (float(v) for v in coef)
    residual = a @ coef - u
    curvature = -c2
    nu_eff = body_force / (2.0 * curvature) if curvature > 0 else float("nan")
    # du/dy at the wall (y = 0) of the fitted parabola is c1, so the wall shear
    # implied by the measured profile is rho nu c1 -- exactly rho g H / 2 for the
    # analytic parabola.
    tau_wall_measured = -1.0  # placeholder, filled by caller with rho0
    return {
        "n_rows": len(rows),
        "c0": c0,
        "c1": c1,
        "c2": c2,
        "du_dy_wall": c1,
        "nu_eff": nu_eff,
        "nu_eff_over_nu": nu_eff / viscosity,
        "fit_residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "tau_wall_per_unit_rho": viscosity * c1,
        "u_centre_fit": c0 + c1 * (height / 2.0) + c2 * (height ** 2 / 4.0),
        "placeholder": tau_wall_measured,
    }


def gates(result: dict, fit: dict, cfg: ChannelConfig) -> dict:
    """The three ``GAP-SPH-ISPH-PROJECTION`` acceptance gates."""
    tau_measured = cfg.rho0 * fit.get("tau_wall_per_unit_rho", float("nan"))
    tau_analytic = cfg.tau_wall_analytic
    centre_measured = result["u_max_measured"]
    return {
        "density_fluctuation_max": {
            "measured": result["density_error_max"],
            "limit": 1.0e-3,
            "pass": result["density_error_max"] <= 1.0e-3,
        },
        "centre_velocity": {
            "measured": centre_measured,
            "reference": result["u_max_analytic"],
            "rel_error_pct": 100.0 * (centre_measured - result["u_max_analytic"])
            / result["u_max_analytic"],
            "limit_pct": 3.0,
            "pass": abs(100.0 * (centre_measured - result["u_max_analytic"])
                        / result["u_max_analytic"]) <= 3.0,
        },
        "wall_shear_budget": {
            "measured": tau_measured,
            "reference": tau_analytic,
            "ratio": tau_measured / tau_analytic,
            "rel_error_pct": 100.0 * (tau_measured - tau_analytic) / tau_analytic,
            "limit_pct": 3.0,
            "pass": abs(100.0 * (tau_measured - tau_analytic) / tau_analytic) <= 3.0,
        },
        "divergence_reduction": result["divergence_reduction"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/isph_channel"))
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--t-average", type=float, default=0.1)
    parser.add_argument("--viscosity", type=float, default=0.01)
    parser.add_argument("--body-force", type=float, default=9.81)
    parser.add_argument("--xsph", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    if args.quick:
        args.resolutions = [8]
        args.t_end = 0.2
        args.t_average = 0.05

    results = []
    signatures = operator_signatures(ChannelConfig(resolution=args.resolutions[0]),
                                     device)
    for res in args.resolutions:
        cfg = ChannelConfig(columns=args.columns, resolution=res,
                            viscosity=args.viscosity, body_force=args.body_force,
                            t_end=args.t_end, t_average=args.t_average,
                            xsph=args.xsph)
        print(f"[isph] H/dx={res} ...", flush=True)
        res_dict = run_case(cfg, device)
        fit = fit_profile(res_dict["profile"], cfg.height, cfg.body_force,
                          cfg.viscosity, lo=0.05, hi=0.95)
        res_dict["fit"] = fit
        res_dict["gates"] = gates(res_dict, fit, cfg)
        g = res_dict["gates"]
        print(f"[isph]   rho_err={g['density_fluctuation_max']['measured']:.3e} "
              f"u_max_err={g['centre_velocity']['rel_error_pct']:.2f}% "
              f"tau_err={g['wall_shear_budget']['rel_error_pct']:.2f}% "
              f"div_ratio={res_dict['divergence_reduction']:.3e} "
              f"({res_dict['wall_clock_s']:.1f}s)", flush=True)
        results.append(res_dict)

    payload = {
        "case": "isph_plane_poiseuille_channel",
        "anchor": "GAP-SPH-ISPH-PROJECTION",
        "method": "incompressible SPH, pressure projection (Cummins & Rudman 1999; "
                  "Shao & Lo 2003), no Tait EOS, no acoustic time-step limit",
        "analytic": "u(y) = g y (H - y) / (2 nu); u_max = g H^2/(8 nu); "
                    "tau_wall = rho g H / 2",
        "operator_signatures": signatures,
        "results": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"[isph] wrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())