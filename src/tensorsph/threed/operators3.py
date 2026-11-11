"""3-D WCSPH operators: summation density, Tait EOS, pressure and viscous forces.

Every operator here is written so that it can be applied to a *given analytic
field* on a fixed particle set, which is what the operator benchmarks do, and so
that it can also be driven forward in time by :mod:`tensorsph.threed.solver`.

The pressure force is the symmetric form of Monaghan (1992),

    a_i = -sum_j m_j (p_i / rho_i^2 + p_j / rho_j^2) grad_i W_ij ,

which conserves linear momentum exactly for equal-mass pairs, and the viscous
term is the second-derivative (Brookshaw) Laplacian

    a_i = sum_j m_j nu (v_j - v_i) laplacian_i W_ij / rho_ij ,

with ``rho_ij`` the arithmetic mean, so that the two members of a pair get
exactly equal and opposite momentum.  The continuum limit of the second operator
is verified against an analytic field in
``benchmarks/threed_kernel/report.json`` (it is ``+nu laplacian v``, not minus:
the sign follows from ``int r^2 laplacian W dV = 6``).
"""

from __future__ import annotations

import math

import torch

from .kernel3 import cubic_spline_kernel, pair_kernels
from .neighbors3 import minimum_image

__all__ = [
    "pair_geometry",
    "summation_density",
    "tait_pressure",
    "tait_density",
    "pressure_acceleration",
    "viscous_acceleration",
    "wall_pressure_adami",
    "accelerations",
    "laplacian_of_field",
    "laplacian_second_moment",
    "laplacian_fourth_moment",
    "fourth_order_coefficient",
    "gradient_of_field",
]


def pair_geometry(pos: torch.Tensor, pairs: torch.Tensor, h: float,
                  box=None):
    """Return ``(delta, W, grad, laplacian)`` for every pair.

    ``delta = x_i - x_j`` is the *minimum image* when ``box`` is given, so a
    periodic pair that straddles the boundary is handled here once instead of in
    every operator.
    """
    delta = pos[pairs[:, 0]] - pos[pairs[:, 1]]
    if box is not None:
        delta = minimum_image(delta, box)
    weight, grad, lap = pair_kernels(delta, h)
    return delta, weight, grad, lap


def summation_density(pos: torch.Tensor, mass: torch.Tensor, h: float,
                      pairs: torch.Tensor, box=None) -> torch.Tensor:
    """``rho_i = sum_j m_j W_ij`` including the self term ``m_i W(0)``."""
    _, weight, _, _ = pair_geometry(pos, pairs, h, box)
    self_weight = float(cubic_spline_kernel(
        torch.zeros(1, dtype=pos.dtype, device=pos.device), h)[0])
    rho = torch.zeros_like(mass)
    rho = rho.index_add(0, pairs[:, 0], mass[pairs[:, 1]] * weight)
    rho = rho.index_add(0, pairs[:, 1], mass[pairs[:, 0]] * weight)
    return rho + mass * self_weight


def tait_pressure(rho: torch.Tensor, rho0: float, sound_speed: float,
                  gamma: float) -> torch.Tensor:
    """Tait equation of state, clamped at ``p = 0`` (no tensile stress)."""
    stiffness = rho0 * sound_speed ** 2 / gamma
    return (stiffness * ((rho / rho0).pow(gamma) - 1.0)).clamp_min(0.0)


def tait_density(pressure: torch.Tensor, rho0: float, sound_speed: float,
                 gamma: float) -> torch.Tensor:
    """Inverse of :func:`tait_pressure` (used for Tait-consistent wall density)."""
    stiffness = rho0 * sound_speed ** 2 / gamma
    return rho0 * (1.0 + pressure.clamp_min(0.0) / stiffness).pow(1.0 / gamma)


def pressure_acceleration(pos: torch.Tensor, mass: torch.Tensor,
                          rho: torch.Tensor, pressure: torch.Tensor, h: float,
                          pairs: torch.Tensor, box=None) -> torch.Tensor:
    """Symmetric pressure acceleration ``-sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) grad_i W``.

    For a pair ``(i, j)`` the two contributions are exactly equal and opposite
    (``grad_j W = -grad_i W``, same scalar), so linear momentum is conserved to
    round-off whatever the density field is.  A static wall particle must,
    therefore, be filtered out by the caller: it does not receive its recoil.
    """
    _, _, grad, _ = pair_geometry(pos, pairs, h, box)
    i, j = pairs[:, 0], pairs[:, 1]
    sym = pressure[i] / rho[i].square() + pressure[j] / rho[j].square()
    acc = torch.zeros_like(pos)
    contribution = grad * (mass[j] * sym).unsqueeze(-1)
    acc = acc.index_add(0, i, -contribution)
    acc = acc.index_add(0, j, grad * (mass[i] * sym).unsqueeze(-1))
    return acc


def viscous_acceleration(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor,
                         rho: torch.Tensor, nu: float, h: float,
                         pairs: torch.Tensor, box=None,
                         renormalise: bool = True,
                         order: int = 2) -> torch.Tensor:
    """Viscous acceleration ``sum_j m_j nu (v_j - v_i) laplacian_i W_ij / rho_ij``.

    ``renormalise`` divides the pair kernel by the discrete second moment
    ``K_ij / 2`` (:func:`laplacian_second_moment`), which is the difference
    between a viscous operator that over-predicts the decay of a 3-D shear wave
    by 17 % and one that is within a couple of percent
    (``benchmarks/threed_shear/report.json``).  The factor is symmetrised over the
    pair, so the two members still receive exactly equal and opposite momentum
    and the run's momentum gate stays at round-off.
    """
    if nu == 0.0:
        return torch.zeros_like(pos)
    delta, _, _, lap = pair_geometry(pos, pairs, h, box)
    i, j = pairs[:, 0], pairs[:, 1]
    rho_ij = 0.5 * (rho[i] + rho[j])
    weight = mass[j] * lap / rho_ij
    r2 = delta.square().sum(dim=1)
    volume_j = mass[j] / rho_ij
    volume_i = mass[i] / rho_ij
    moment = torch.zeros_like(rho)
    moment = moment.index_add(0, i, r2 * lap * volume_j)
    moment = moment.index_add(0, j, r2 * lap * volume_i)
    moment = moment / 3.0
    if renormalise:
        weight = weight * (2.0 / (0.5 * (moment[i] + moment[j])).clamp_min(1.0e-12))
    acc = _pair_laplacian(vel, i, j, weight)
    if order == 4:
        fourth = torch.zeros_like(rho)
        fourth = fourth.index_add(0, i, r2 * r2 * lap * volume_j)
        fourth = fourth.index_add(0, j, r2 * r2 * lap * volume_i)
        beta = 0.5 * (fourth[i] + fourth[j]) / 120.0
        acc = acc - _pair_laplacian(acc, i, j, weight * beta)
    acc = acc * nu
    return acc


def _pair_laplacian(field: torch.Tensor, i: torch.Tensor, j: torch.Tensor,
                    weight: torch.Tensor) -> torch.Tensor:
    """``sum_j weight_ij (f_j - f_i)`` written as an antisymmetric pair sum.

    ``weight`` must be symmetric under ``i <-> j``: then every pair contributes
    ``+m_i w (f_j - f_i)`` to ``i`` and ``-m_j w (f_j - f_i)`` to ``j``, so
    ``sum_i m_i (out_i) = 0`` exactly -- which is what keeps the shear-wave
    linear-momentum gate at round-off with the fourth-order term switched on.
    """
    dv = (field[j] - field[i]) * weight.unsqueeze(-1)
    out = torch.zeros_like(field)
    out = out.index_add(0, i, dv)
    out = out.index_add(0, j, -dv)
    return out


def wall_pressure_adami(pos: torch.Tensor, mass: torch.Tensor, rho: torch.Tensor,
                        pressure: torch.Tensor, h: float, pairs: torch.Tensor,
                        n_fluid: int, gravity_vec=None, box=None) -> torch.Tensor:
    """Adami et al. (2012) eq. (11) pressure for the static wall particles.

    ``p_w = sum_{f in fluid} (p_f + rho_f g . (r_w - r_f)) W_wf / sum W_wf``,
    returned for the wall particles only.

    Density summation under-counts the neighbours of a wall particle, and the
    Tait law would then give it a pressure far below the fluid it is supposed to
    hold up; the fluid would sink into the wall and the hydrostatic profile
    would be wrong by tens of percent (measured in
    ``benchmarks/threed_hydrostatic/report.json``).
    """
    n = pos.shape[0]
    i, j = pairs[:, 0], pairs[:, 1]
    # Keep only fluid-wall pairs, oriented as (fluid, wall).
    fluid_first = (i < n_fluid) & (j >= n_fluid)
    wall_first = (j < n_fluid) & (i >= n_fluid)
    f = torch.cat((i[fluid_first], j[wall_first]))
    w = torch.cat((j[fluid_first], i[wall_first]))
    weight = cubic_spline_kernel(
        torch.linalg.vector_norm(pos[f] - pos[w], dim=1), h)
    if gravity_vec is not None:
        g = torch.as_tensor(gravity_vec, dtype=pos.dtype, device=pos.device).reshape(1, -1)
        donor = pressure[f] + rho[f] * ((pos[w] - pos[f]) * g).sum(dim=1)
    else:
        donor = pressure[f]
    numerator = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    denominator = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    numerator = numerator.index_add(0, w, donor * weight)
    denominator = denominator.index_add(0, w, weight)
    out = torch.where(denominator > 0.0, numerator / denominator.clamp_min(1.0e-30),
                      torch.zeros_like(numerator))
    # Returned for the *wall* particles only: a wall particle with no fluid
    # neighbour in support gets p = 0, which is the correct "nothing to hold up".
    return out[n_fluid:].clamp_min(0.0)


def accelerations(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor,
                  h: float, pairs: torch.Tensor, *, rho0: float,
                  sound_speed: float, gamma: float = 7.0, viscosity: float = 0.0,
                  gravity=None, box=None, n_fluid: int | None = None,
                  renormalise: bool = True, wall_model: str = "adami",
                  viscous_order: int = 2):
    """Full right-hand side for a fluid (+ optional static wall) particle set.

    Returns ``(acc, rho, pressure)``; ``acc`` rows for wall particles are the
    fluid-like values and are simply ignored by the caller (walls are static).
    """
    n = pos.shape[0]
    n_fluid = n if n_fluid is None else n_fluid
    rho = summation_density(pos, mass, h, pairs, box)
    pressure = tait_pressure(rho, rho0, sound_speed, gamma)
    rho_used = rho
    if n_fluid < n and wall_model == "adami":
        p_wall = wall_pressure_adami(pos, mass, rho, pressure, h, pairs,
                                     n_fluid, gravity, box)
        pressure = torch.cat((pressure[:n_fluid], p_wall))
        rho_wall = tait_density(p_wall, rho0, sound_speed, gamma)
        rho_used = torch.cat((rho[:n_fluid], rho_wall))
    elif n_fluid < n and wall_model != "tait":
        raise ValueError(f"unknown wall_model {wall_model!r}")
    acc = pressure_acceleration(pos, mass, rho_used, pressure, h, pairs, box)
    acc = acc + viscous_acceleration(pos, vel, mass, rho_used, viscosity, h,
                                     pairs, box, renormalise=renormalise,
                                     order=viscous_order)
    if gravity is not None:
        g = torch.as_tensor(gravity, dtype=pos.dtype, device=pos.device).reshape(1, -1)
        acc[:n_fluid] = acc[:n_fluid] + g
    return acc, rho, pressure


def laplacian_of_field(pos: torch.Tensor, field: torch.Tensor, volume: torch.Tensor,
                       h: float, pairs: torch.Tensor, box=None,
                       renormalise: bool = True,
                       order: int = 2) -> torch.Tensor:
    """SPH Laplacian of a scalar field: ``sum_j V_j (f_j - f_i) laplacian W_ij``.

    This is the operator whose convergence order the benchmark measures; it is
    the same discrete operator the viscous term uses, with the field passed in
    instead of a velocity component.
    """
    _, _, _, lap = pair_geometry(pos, pairs, h, box)
    i, j = pairs[:, 0], pairs[:, 1]
    difference = field[j] - field[i]
    out = torch.zeros_like(field)
    out = out.index_add(0, i, difference * volume[j] * lap)
    # At j the same pair reads (f_i - f_j) and the radial Laplacian is even, so
    # the second contribution carries the opposite sign to the gradient case.
    out = out.index_add(0, j, -difference * volume[i] * lap)
    if renormalise:
        moment = laplacian_second_moment(pos, volume, h, pairs, box)
        out = out * (2.0 / moment.clamp_min(1.0e-12))
    if order == 4:
        # Iterated operator with the kernel-lattice fourth moment as coefficient:
        # L4 f = L2 f - beta * L2(L2 f).  ``beta`` carries no field information.
        beta = fourth_order_coefficient(pos, volume, h, pairs, box)
        out = out - beta * laplacian_of_field(pos, out, volume, h, pairs, box,
                                              renormalise=renormalise, order=2)
    return out


def gradient_of_field(pos: torch.Tensor, field: torch.Tensor, volume: torch.Tensor,
                      h: float, pairs: torch.Tensor, box=None) -> torch.Tensor:
    """SPH gradient of a scalar field: ``sum_j V_j (f_j - f_i) grad_i W_ij``."""
    _, _, grad, _ = pair_geometry(pos, pairs, h, box)
    i, j = pairs[:, 0], pairs[:, 1]
    difference = field[j] - field[i]
    out = torch.zeros((pos.shape[0], pos.shape[1]), dtype=pos.dtype, device=pos.device)
    out = out.index_add(0, i, grad * (difference * volume[j]).unsqueeze(-1))
    out = out.index_add(0, j, grad * (difference * volume[i]).unsqueeze(-1))
    return out


def laplacian_second_moment(pos: torch.Tensor, volume: torch.Tensor, h: float,
                            pairs: torch.Tensor, box=None) -> torch.Tensor:
    """Discrete second moment ``K_i = (1/3) sum_j V_j r_ij^2 laplacian_i W_ij``.

    The continuum value is ``2`` (from ``int r^2 laplacian W dV = 6``), which is
    what makes the plain operator consistent.  A *particle* distribution replaces
    the integral by a lattice sum, and the sum is not the integral: on a cubic
    lattice with ``h = 1.3 dx`` the measured ``K = 2.347``, i.e. the raw operator
    is 17 % too strong.  Dividing by ``K / 2`` is the zeroth/second-moment
    renormalisation; both the raw and the corrected error are reported in
    ``benchmarks/threed_kernel/report.json``.
    """
    delta, _, _, lap = pair_geometry(pos, pairs, h, box)
    r2 = delta.square().sum(dim=1)
    i, j = pairs[:, 0], pairs[:, 1]
    out = torch.zeros_like(volume)
    out = out.index_add(0, i, r2 * lap * volume[j])
    out = out.index_add(0, j, r2 * lap * volume[i])
    return out / 3.0


def laplacian_fourth_moment(pos: torch.Tensor, volume: torch.Tensor, h: float,
                            pairs: torch.Tensor, box=None) -> torch.Tensor:
    """Discrete fourth moment ``I4_i = sum_j V_j r_ij^4 laplacian_i W_ij``.

    Continuum value ``18 h^2`` for the 3-D cubic spline (checked by radial
    quadrature in ``docs/verification_laplacian_fourth_order.md``).  It is the
    fourth-moment analogue of :func:`laplacian_second_moment` and is likewise a
    property of the lattice *and* kernel only -- no field, no reference solution.
    """
    delta, _, _, lap = pair_geometry(pos, pairs, h, box)
    r2 = delta.square().sum(dim=1)
    i, j = pairs[:, 0], pairs[:, 1]
    out = torch.zeros_like(volume)
    out = out.index_add(0, i, r2 * r2 * lap * volume[j])
    out = out.index_add(0, j, r2 * r2 * lap * volume[i])
    return out


def fourth_order_coefficient(pos: torch.Tensor, volume: torch.Tensor, h: float,
                             pairs: torch.Tensor, box=None) -> torch.Tensor:
    """``beta_i = I4_i / 120``, the truncation-cancelling coefficient of ``L4``.

    Taylor expansion of the continuum operator gives
    ``L2 f = laplacian f + (I4/120) laplacian(laplacian f) + O(h^4)``; the
    ``I4/120`` comes from ``1/4!`` times the isotropic fifth contraction
    ``I4/15`` times three equal index patterns, i.e. ``2/(4! * 15) = 1/120``.
    Subtracting ``beta * L2(L2 f)`` leaves ``O(h^4)``.
    """
    return laplacian_fourth_moment(pos, volume, h, pairs, box) / 120.0


def analytic_laplacian_of_mode(pos: torch.Tensor, wavenumber: float,
                               mode: str = "sin-sin-sin-prod"):
    """Laplacian of ``f = sin(k x) sin(k y) sin(k z)``: ``-3 k^2 f``."""
    f = (torch.sin(wavenumber * pos[:, 0]) * torch.sin(wavenumber * pos[:, 1])
         * torch.sin(wavenumber * pos[:, 2]))
    return f, -3.0 * wavenumber ** 2 * f


def mode_amplitude(pos: torch.Tensor, vel: torch.Tensor, wavenumber: float) -> float:
    """Projection of ``v_x`` onto ``sin(k y) sin(k z)`` (the shear-wave mode).

    For ``v_x = A sin(k y) sin(k z)`` on a uniform lattice the discrete mean of
    ``sin^2 sin^2`` is 1/4, so ``4 * <v_x sin sin>`` returns ``A``.
    """
    shape = torch.sin(wavenumber * pos[:, 1]) * torch.sin(wavenumber * pos[:, 2])
    return float(4.0 * (vel[:, 0] * shape).mean())


def analytic_mode_decay(nu: float, wavenumber: float) -> float:
    """Exact decay exponent of ``u = A sin(ky) sin(kz)``: ``2 nu k^2``."""
    return 2.0 * nu * wavenumber ** 2


def shear_wave_field(pos: torch.Tensor, amplitude: float, wavenumber: float) -> torch.Tensor:
    """Initial velocity of the 3-D double-shear field (``u_x`` only)."""
    vel = torch.zeros_like(pos)
    vel[:, 0] = amplitude * torch.sin(wavenumber * pos[:, 1]) * torch.sin(
        wavenumber * pos[:, 2])
    return vel


def cell_dx(box: float, n: int) -> float:
    """Uniform lattice spacing that exactly tiles ``[0, box)`` with ``n`` points."""
    return float(box) / float(n)


def wavenumber_for(box: float, periods: int = 1) -> float:
    """``k = 2 pi periods / box``: the lowest mode that fits the periodic box."""
    return 2.0 * math.pi * periods / float(box)
