"""Fluid-to-solid load transfer: a port of SPHinXsys ``FSI::ForceFromFluid``.

Why this module exists
----------------------
TensorSPH's elastic-gate FSI (``tensorsph.fsi_elastic_gate``) transferred the
fluid load with a kernel-weighted plane fit ("panel") after the Adami pairwise
summation was measured to lose 20-28 % of a hydrostatic load (m16257 audit).
The open-source implementation this repository is benchmarked against,
``Xiangyu-Hu/SPHinXsys``, keeps the one-sided summation and repairs the wall
pressure by extrapolation instead.  To compare the two conventions on the same
particle data, the OSS operator has to exist here in a form that is faithful to
the reference algebra, which is what this module provides.

Provenance (blueprint read 2026-11-13 from ``Xiangyu-Hu/SPHinXsys`` master; the
verbatim files are archived under ``artifacts/xval_fsi2/oss_src/``)

* ``src/shared/shared_ck/particle_dynamics/fluid_structure_interaction/force_on_structure.hpp``
  -- ``PressureForceFromFluid::InteractKernel::interact`` (lines 103-124) and
  ``ViscousForceFromFluid::InteractKernel::interact`` (lines 55-72).
* ``tests/2d_examples/test_2d_fsi2/fsi2.cpp`` -- the benchmark that instantiates
  them: ``PressureForceFromFluid<decltype(density_relaxation)>`` with
  ``density_relaxation = Integration2ndHalfWithWallNoRiemann``.
* ``src/shared/particle_dynamics/fluid_dynamics/fluid_integration.h:206``
  -- that alias fixes ``RiemannSolverType = NoRiemannSolver``.
* ``src/shared/shared_ck/particle_dynamics/fluid_dynamics/acoustic_step_2nd_half.h:197``
  -- ``NoKernelCorrectionCK`` for the wall contact, i.e. the kernel-correction
  matrix is the identity in the fsi2 configuration.
* ``src/shared/materials/riemann_solver.h:93`` -- ``DissipativePJump``.
* ``src/shared/materials/viscosity.h:54`` -- ``OneSideViscosity`` is the
  constant dynamic viscosity ``mu`` of the fluid.

Reduced algebra actually integrated here (``V_i`` solid volume, ``V_j`` fluid
volume, ``e_ij`` the unit vector from fluid ``j`` to solid ``i``)::

    p_wall_j = p_j + rho_j * r_ij * max(0, (a_ext_j - a_i) . e_ij)      (Adami)
    n_face   = -sign(e_ij . n_i) * n_i
    u_jump   = 2 (u_j - u_i) . n_face
    F_i = V_i * sum_j [ -( D(u_jump) * n_face + (p_wall_j + p_j) e_ij )
                        * dW_ij/dr * V_j ]                (pressure branch)
    F_i = V_i * sum_j [ r_ij * mu_j * (u_i - u_j) / (r_ij^2 + 0.01 h^2)
                        * dW_ij/dr * V_j ]                (viscous branch)

``D`` is the acoustic Riemann dissipation, which is exactly zero in the fsi2
configuration (``NoRiemannSolver``); it is kept switchable because the
``ViscousForceFromFluid``/``PressureForceFromFluid`` pair is also used with the
Riemann solver in SPHinXsys' other FSI cases.

Sign conventions were checked against the analytic references in
``benchmarks/xval_fsi2``: a pressure field pushes the solid away from the fluid
and a shear field drags it towards the fluid velocity.

Units: ``F_i`` is a force per unit thickness in 2-D (N/m), i.e. the same
convention as SPHinXsys, where volumes carry ``dx**dim`` and ``rho0 = 1`` makes
``mass == volume`` in the dimensionless benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .neighbor_search import cell_linked_pairs
from .wcsph import kernels

__all__ = [
    "ForceFromFluidConfig",
    "cross_pairs",
    "pressure_force_from_fluid",
    "viscous_force_from_fluid",
    "wall_shear_from_gradient",
    "one_sided_viscous_truncation_factor",
    "panel_pressure_force",
    "pairwise_pressure_force",
    "wall_traction",
    "OSS_PROVENANCE",
]

OSS_PROVENANCE = {
    "repo": "Xiangyu-Hu/SPHinXsys",
    "ref": "master",
    "read_on": "2026-11-13",
    "files": [
        "src/shared/shared_ck/particle_dynamics/fluid_structure_interaction/force_on_structure.hpp:55-72",
        "src/shared/shared_ck/particle_dynamics/fluid_structure_interaction/force_on_structure.hpp:103-124",
        "src/shared/particle_dynamics/fluid_dynamics/fluid_integration.h:206",
        "src/shared/shared_ck/particle_dynamics/fluid_dynamics/acoustic_step_2nd_half.h:197",
        "src/shared/materials/riemann_solver.h:93",
        "src/shared/materials/viscosity.h:54",
        "tests/2d_examples/test_2d_fsi2/fsi2.cpp:155",
        "tests/2d_examples/test_2d_fsi2/fsi2.h:14-35",
    ],
}


@dataclass(frozen=True)
class ForceFromFluidConfig:
    """Discretisation settings of the transfer operator.

    ``riemann=False`` and ``kernel_correction=False`` reproduce the fsi2 case
    exactly (see the provenance notes in the module docstring).
    """

    dim: int = 2
    h: float = 0.13
    kernel: str = "quintic"
    riemann: bool = False
    rho0_f: float = 1.0
    c0_f: float = 10.0
    limiter_coefficient: float = 3.0
    viscous_regularisation: float = 0.01
    kernel_correction: bool = False

    @property
    def support(self) -> float:
        return kernels.SUPPORT_FACTOR[self.kernel] * self.h


def _riemann_limiter(x: torch.Tensor, beta: float) -> torch.Tensor:
    """SPHinXsys ``DissipativePJump`` limiter ``x(3 - x^2)/4`` clipped to 1."""
    xb = (beta * x).clamp_min(0.0)
    return torch.where(xb < 1.0, xb * (3.0 - xb * xb) * 0.25, torch.ones_like(xb))


def cross_pairs(pos_solid: torch.Tensor, pos_fluid: torch.Tensor,
                cutoff: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Solid particle index ``i`` and fluid particle index ``j`` with ``r < cutoff``."""
    n_s = int(pos_solid.shape[0])
    if n_s == 0 or pos_fluid.shape[0] == 0:
        empty = torch.zeros(0, dtype=torch.long, device=pos_solid.device)
        return empty, empty
    pos = torch.cat((pos_solid, pos_fluid), dim=0)
    pairs = cell_linked_pairs(pos, float(cutoff))
    if pairs.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=pos_solid.device)
        return empty, empty
    a, b = pairs.unbind(dim=1)
    # normalise to (solid, fluid)
    lo = torch.where(a < n_s, a, b)
    hi = torch.where(a < n_s, b, a)
    keep = (lo < n_s) & (hi >= n_s)
    return lo[keep], hi[keep] - n_s


def _pair_geometry(pos_solid, pos_fluid, i_s, j_f, cutoff, kernel):
    delta = pos_solid[i_s] - pos_fluid[j_f]
    r = torch.linalg.vector_norm(delta, dim=1)
    active = (r > 1.0e-12) & (r < cutoff)
    delta, r = delta[active], r[active]
    e = delta / r.clamp_min(1.0e-300)[:, None]
    _, grad = kernels.kernel_pairs(delta, cutoff / kernels.SUPPORT_FACTOR[kernel],
                                   delta.shape[1], kernel)
    # grad = dW/dr * e, so a dot with e recovers the scalar derivative
    dwdr = (grad * e).sum(dim=1)
    return i_s[active], j_f[active], r, e, dwdr


def _gather(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return t[idx] if t is not None else None


def pressure_force_from_fluid(
    pos_solid: torch.Tensor,
    vol_solid: torch.Tensor,
    normal_solid: torch.Tensor,
    acc_solid: torch.Tensor,
    vel_solid: torch.Tensor,
    pos_fluid: torch.Tensor,
    vol_fluid: torch.Tensor,
    p_fluid: torch.Tensor,
    rho_fluid: torch.Tensor,
    vel_fluid: torch.Tensor,
    acc_ext_fluid: torch.Tensor,
    cfg: ForceFromFluidConfig,
    correction_fluid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``SPHinXsys::FSI::PressureForceFromFluid`` (force per unit thickness).

    ``acc_solid`` is the solid body's *average* acceleration (SPHinXsys
    ``AverageVelocityAndAcceleration``), used in the wall-pressure
    extrapolation; pass zeros for a static comparison.
    ``acc_ext_fluid`` is the fluid particle's external acceleration
    (``ForcePrior / mass``, i.e. gravity).
    """
    n_s = pos_solid.shape[0]
    out = torch.zeros((n_s, cfg.dim), dtype=pos_solid.dtype, device=pos_solid.device)
    if acc_ext_fluid.dim() == 1:      # a single gravity vector is broadcast
        acc_ext_fluid = acc_ext_fluid.expand(pos_fluid.shape[0], cfg.dim)
    if acc_solid.dim() == 1:
        acc_solid = acc_solid.expand(n_s, cfg.dim)
    i_s, j_f = cross_pairs(pos_solid, pos_fluid, cfg.support)
    if i_s.numel() == 0:
        return out
    i_s, j_f, r, e, dwdr = _pair_geometry(pos_solid, pos_fluid, i_s, j_f,
                                          cfg.support, cfg.kernel)

    rho_j = rho_fluid[j_f]
    p_j = p_fluid[j_f]
    acc_face = ((acc_ext_fluid[j_f] - acc_solid[i_s]) * e).sum(dim=1)
    p_wall = p_j + rho_j * r * acc_face.clamp_min(0.0)

    n_i = normal_solid[i_s]
    if cfg.kernel_correction and correction_fluid is not None:
        corr = correction_fluid[j_f]
        e = torch.bmm(corr, e.unsqueeze(-1)).squeeze(-1)
    n_face = -torch.sign((e * n_i).sum(dim=1, keepdim=True)) * n_i

    u_jump = 2.0 * ((vel_fluid[j_f] - vel_solid[i_s]) * n_face).sum(dim=1)
    if cfg.riemann:
        diss = (cfg.rho0_f * cfg.c0_f * u_jump
                * _riemann_limiter(u_jump / cfg.c0_f, cfg.limiter_coefficient))
    else:
        diss = torch.zeros_like(u_jump)

    contrib = -(diss[:, None] * n_face
                + (p_wall + p_j)[:, None] * e) * (dwdr * vol_fluid[j_f])[:, None]
    out = out.index_add(0, i_s, contrib)
    return out * vol_solid[:, None]


def viscous_force_from_fluid(
    pos_solid: torch.Tensor,
    vol_solid: torch.Tensor,
    vel_solid: torch.Tensor,
    pos_fluid: torch.Tensor,
    vol_fluid: torch.Tensor,
    vel_fluid: torch.Tensor,
    mu_fluid: torch.Tensor,
    cfg: ForceFromFluidConfig,
    correction_fluid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``SPHinXsys::FSI::ViscousForceFromFluid`` (force per unit thickness)."""
    n_s = pos_solid.shape[0]
    out = torch.zeros((n_s, cfg.dim), dtype=pos_solid.dtype, device=pos_solid.device)
    i_s, j_f = cross_pairs(pos_solid, pos_fluid, cfg.support)
    if i_s.numel() == 0:
        return out
    i_s, j_f, r, e, dwdr = _pair_geometry(pos_solid, pos_fluid, i_s, j_f,
                                          cfg.support, cfg.kernel)
    # r_ij . (H_j e_ij); the identity correction makes this plain r_ij
    e_eff = e
    if cfg.kernel_correction and correction_fluid is not None:
        e_eff = torch.bmm(correction_fluid[j_f], e.unsqueeze(-1)).squeeze(-1)
    rdot = r * (e * e_eff).sum(dim=1)
    vel_derivative = ((vel_solid[i_s] - vel_fluid[j_f])
                      / (r.square() + cfg.viscous_regularisation * cfg.h ** 2)[:, None])
    contrib = (rdot * mu_fluid[j_f] * dwdr * vol_fluid[j_f])[:, None] * vel_derivative
    out = out.index_add(0, i_s, contrib)
    return out * vol_solid[:, None]


def one_sided_viscous_truncation_factor(h: float, dim: int = 2,
                                       kernel: str = "quintic",
                                       regularisation: float = 0.0,
                                       spacing: Optional[float] = None,
                                       n_quad: int = 400001) -> float:
    """Continuum response of the one-sided viscous sum to a linear wall shear.

    For a wall at ``y = 0`` wetted by a linear profile ``u_x = gamma y`` the OSS
    branch, summed over a surface layer of spacing ``s``, is

        F_x/L = s * mu gamma * (-2) int_0^R (dW/dr) r / (1 + eps h^2/r^2) dr

    so the *ratio to the exact traction* ``mu gamma`` is

        ratio = eps=0:   -2 s int_0^R (dW/dr) r dr = (7/(3 pi)) (s/h)
                 (Wendland C2: int_0^2 f'(q) q dq = -2/3, alpha_2 = 7/(4 pi)),

    i.e. a *fixed fraction* that carries no ``dx`` beyond the ``s/h`` ratio.  That
    is why the measured deficit of the OSS branch is dx independent
    (``docs/xval_fsi2/X3_VISCOUS_DEFICIT.md``) -- it is a truncation moment, not
    a discretisation error.  ``s`` defaults to ``h`` (so the closed form above
    applies); pass the lattice spacing to get the value of an actual case.

    The number is an independent *prediction* from kernel quadrature (trapezoid
    on ``n_quad`` points), not a fitted constant.  It over-estimates the lattice
    case by ~7 % because the first fluid layer sits at ``s/2`` from the wall and
    the near-wall shell of the integral carries no particle.
    """
    s = h if spacing is None else spacing
    r = torch.linspace(1.0e-9, kernels.SUPPORT_FACTOR[kernel] * h, n_quad,
                       dtype=torch.float64)
    _, df = kernels.shape(r / h, kernel)
    dwdr = kernels.normalisation(dim, kernel) / h ** (dim + 1) * df
    integrand = dwdr * r / (1.0 + regularisation * (h / r) ** 2)
    return float(-2.0 * s * torch.trapezoid(integrand, r))


def wall_shear_from_gradient(
    pos_solid: torch.Tensor,
    vol_solid: torch.Tensor,
    normal_solid: torch.Tensor,
    pos_fluid: torch.Tensor,
    vol_fluid: torch.Tensor,
    vel_fluid: torch.Tensor,
    mu_fluid: torch.Tensor,
    vel_solid: torch.Tensor,
    cfg: ForceFromFluidConfig,
) -> torch.Tensor:
    """Viscous wall load from a *kernel-corrected* velocity-gradient reconstruction.

    The OSS one-sided sum (``viscous_force_from_fluid``) is a truncated
    quadrature: with a support cut by the wall it returns a fixed fraction of
    ``mu * du_t/dn`` (``one_sided_viscous_truncation_factor``, -38 % measured on
    the X3 Couette case) and the deficit does not vanish under refinement.

    This arm instead reconstructs the velocity-gradient tensor at the wall from
    the fluid neighbours and evaluates the traction directly::

        G_km = sum_j V_j (u_j - u_i)_k (dW/dr) e_m             (raw gradient sum)
        B_pm = -sum_j V_j (dW/dr) r_p r_m / r                 (first moment)
        a    = G B^-1                                         (corrected grad u)
        F_i  = A_i mu_i (a . n_i)

    ``B`` is the *measured* first moment of the truncated neighbour set, so the
    correction absorbs both the cut support and the wall curvature; for a linear
    velocity field the reconstruction is exact whatever the truncation, which is
    the property X3 tests.  ``mu_i`` is the kernel-weighted mean of ``mu_fluid``
    over the neighbours, and the tributary face length is
    ``V_i**((dim-1)/dim)`` (= ``dx`` for a lattice particle), the same area
    convention as ``panel_pressure_force``.
    """
    n_s = pos_solid.shape[0]
    dim = cfg.dim
    out = torch.zeros((n_s, dim), dtype=pos_solid.dtype, device=pos_solid.device)
    i_s, j_f = cross_pairs(pos_solid, pos_fluid, cfg.support)
    if i_s.numel() == 0:
        return out
    i_s, j_f, r, e, dwdr = _pair_geometry(pos_solid, pos_fluid, i_s, j_f,
                                          cfg.support, cfg.kernel)
    vol_f = vol_fluid[j_f]
    w = kernels.kernel_pairs(pos_solid[i_s] - pos_fluid[j_f], cfg.h, dim,
                             cfg.kernel)[0] * vol_f
    # first moment of the actual (truncated) neighbour set
    moment = (dwdr * vol_f * r)[:, None, None] * (e[:, :, None] * e[:, None, :])
    b_mat = torch.zeros((n_s, dim, dim), dtype=pos_solid.dtype,
                        device=pos_solid.device).index_add(0, i_s, -moment)
    # raw gradient sum
    du = vel_fluid[j_f] - vel_solid[i_s]
    raw = (dwdr * vol_f)[:, None, None] * (du[:, :, None] * e[:, None, :])
    g_mat = torch.zeros((n_s, dim, dim), dtype=pos_solid.dtype,
                        device=pos_solid.device).index_add(0, i_s, raw)
    trace = b_mat.diagonal(dim1=-2, dim2=-1).sum(dim=1).clamp_min(1.0e-30)
    eye = torch.eye(dim, dtype=pos_solid.dtype,
                    device=pos_solid.device).expand_as(b_mat)
    b_inv = torch.linalg.inv(b_mat + 1.0e-12 * trace[:, None, None] * eye)
    grad_u = torch.bmm(g_mat, b_inv)
    mu_wall = (torch.zeros(n_s, dtype=pos_solid.dtype, device=pos_solid.device)
               .index_add(0, i_s, w * mu_fluid[j_f])
               / torch.zeros(n_s, dtype=pos_solid.dtype, device=pos_solid.device)
               .index_add(0, i_s, w).clamp_min(1.0e-30))
    shear = torch.bmm(grad_u, normal_solid.unsqueeze(-1)).squeeze(-1)
    face_area = vol_solid.pow((dim - 1.0) / dim)
    return face_area[:, None] * mu_wall[:, None] * shear


def panel_pressure_force(

    pos_solid: torch.Tensor,
    vol_solid: torch.Tensor,
    normal_solid: torch.Tensor,
    pos_fluid: torch.Tensor,
    vol_fluid: torch.Tensor,
    p_fluid: torch.Tensor,
    cfg: ForceFromFluidConfig,
) -> torch.Tensor:
    """TensorSPH's kernel-weighted plane fit (``fsi_elastic_gate`` panel mode).

    Included as the *comparison arm*: the pressure at each solid particle is the
    constant term of a weighted least-squares plane through the neighbouring
    fluid pressures, and the load is ``p_face`` times the tributary area
    ``dx**(dim-1)``.  A linear pressure field is reproduced exactly, which is
    the property the m16352 audit measured at +0.1..0.2 %.
    """
    n_s = pos_solid.shape[0]
    dim = cfg.dim
    out = torch.zeros((n_s, dim), dtype=pos_solid.dtype, device=pos_solid.device)
    i_s, j_f = cross_pairs(pos_solid, pos_fluid, cfg.support)
    if i_s.numel() == 0:
        return out
    i_s, j_f, r, e, dwdr = _pair_geometry(pos_solid, pos_fluid, i_s, j_f,
                                          cfg.support, cfg.kernel)
    w = kernels.kernel_pairs(pos_solid[i_s] - pos_fluid[j_f], cfg.h, dim,
                             cfg.kernel)[0] * vol_fluid[j_f]
    delta = pos_solid[i_s] - pos_fluid[j_f]
    n_unknown = 1 + dim
    m = torch.zeros((n_s, n_unknown, n_unknown), dtype=pos_solid.dtype,
                    device=pos_solid.device)
    b = torch.zeros((n_s, n_unknown), dtype=pos_solid.dtype, device=pos_solid.device)
    m[:, 0, 0] = torch.zeros(n_s, dtype=pos_solid.dtype, device=pos_solid.device).index_add(0, i_s, w)
    b[:, 0] = torch.zeros(n_s, dtype=pos_solid.dtype, device=pos_solid.device).index_add(
        0, i_s, w * p_fluid[j_f])
    for a in range(dim):
        m[:, 0, a + 1] = torch.zeros(n_s, dtype=pos_solid.dtype,
                                     device=pos_solid.device).index_add(
            0, i_s, w * delta[:, a])
        m[:, a + 1, 0] = m[:, 0, a + 1]
        b[:, a + 1] = torch.zeros(n_s, dtype=pos_solid.dtype,
                                  device=pos_solid.device).index_add(
            0, i_s, w * delta[:, a] * p_fluid[j_f])
        for c in range(dim):
            m[:, a + 1, c + 1] = torch.zeros(n_s, dtype=pos_solid.dtype,
                                             device=pos_solid.device).index_add(
                0, i_s, w * delta[:, a] * delta[:, c])
    trace = m.diagonal(dim1=-2, dim2=-1).sum(dim=1).clamp_min(1.0e-30)
    eye = torch.eye(n_unknown, dtype=pos_solid.dtype,
                    device=pos_solid.device).expand_as(m)
    sol = torch.linalg.solve(m + 1.0e-10 * trace[:, None, None] * eye,
                             b[:, :, None])[:, :, 0]
    p_face = torch.where(m[:, 0, 0] > 1.0e-12, sol[:, 0],
                         torch.zeros(n_s, dtype=pos_solid.dtype, device=pos_solid.device))
    # tributary face length of a lattice particle: (V)**((dim-1)/dim) == dx
    face_area = vol_solid.pow((dim - 1.0) / dim) if dim > 1 else vol_solid
    return -p_face[:, None] * normal_solid * face_area[:, None]


def pairwise_pressure_force(
    pos_solid: torch.Tensor,
    vol_solid: torch.Tensor,
    pos_fluid: torch.Tensor,
    vol_fluid: torch.Tensor,
    p_fluid: torch.Tensor,
    rho_fluid: torch.Tensor,
    cfg: ForceFromFluidConfig,
    rho0_solid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The legacy symmetric Adami term kept in ``fsi_elastic_gate`` ("pairwise").

    ``f_g = m_g sum_f m_f (p_f/rho_f^2 + p_g/rho_g^2) grad_g W`` with the wall
    pressure ``p_g`` from a kernel-weighted average.  Baseline arm only: the
    m16257 audit measured -20.4 % on a hydrostatic load at ``dx = 10`` mm.
    """
    n_s = pos_solid.shape[0]
    dim = cfg.dim
    out = torch.zeros((n_s, dim), dtype=pos_solid.dtype, device=pos_solid.device)
    i_s, j_f = cross_pairs(pos_solid, pos_fluid, cfg.support)
    if i_s.numel() == 0:
        return out
    delta_all = pos_solid[i_s] - pos_fluid[j_f]
    r_all = torch.linalg.vector_norm(delta_all, dim=1)
    active = (r_all > 1.0e-12) & (r_all < cfg.support)
    i_s, j_f, delta_all, r_all = i_s[active], j_f[active], delta_all[active], r_all[active]
    w = kernels.kernel_pairs(delta_all, cfg.h, dim, cfg.kernel)[0]
    denom = torch.zeros(n_s, dtype=pos_solid.dtype, device=pos_solid.device).index_add(
        0, i_s, w)
    numer = torch.zeros(n_s, dtype=pos_solid.dtype, device=pos_solid.device).index_add(
        0, i_s, w * p_fluid[j_f])
    p_solid = numer / denom.clamp_min(1.0e-30)
    rho_s = rho0_solid if rho0_solid is not None else torch.ones_like(p_solid)
    _, grad = kernels.kernel_pairs(delta_all, cfg.h, dim, cfg.kernel)
    coeff = p_fluid[j_f] / rho_fluid[j_f].square() + p_solid[i_s] / rho_s[i_s].square()
    m_f = rho_fluid[j_f] * vol_fluid[j_f]
    m_s = rho_s[i_s] * vol_solid[i_s]
    contrib = (m_f * m_s * coeff)[:, None] * grad
    return out.index_add(0, i_s, contrib)


def wall_traction(force: torch.Tensor, area: float) -> torch.Tensor:
    """Sum a per-particle load and divide by the wetted area (per unit thickness)."""
    return force.sum(dim=0) / area
