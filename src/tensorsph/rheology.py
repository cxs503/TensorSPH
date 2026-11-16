"""Generalised-Newtonian rheology: power-law viscosity closure + channel reference.

Why this module exists
----------------------
Every viscous operator in this package (:func:`tensorsph.droplet_impact.rhs`,
its all-pairs twin, the channel and duct cases) takes a *constant* kinematic
viscosity ``cfg.viscosity``.  The case pool therefore only ever verified
Newtonian fluids: plane Poiseuille (``benchmarks/poiseuille*``), the rectangular
duct, Navier slip, the Womersley pulsatile response.  Non-Newtonian rheology was
simply not measurable, so no case could tell whether the solver transports a
shear-dependent stress correctly.

This module supplies the missing half:

* :func:`power_law_viscosity` -- the Ostwald-de Waele (power-law) closure
  ``mu(gd) = K gd**(n-1)`` with the two regularisations a discrete solver
  needs (a floor on the shear rate for ``n < 1``, where ``mu -> inf`` as
  ``gd -> 0``, and an explicit ``mu_max`` cap);
* :func:`velocity_gradient` / :func:`shear_rate` -- the SPH estimate of the
  strain-rate magnitude that feeds that closure.  With mirror/ghost wall
  particles (:mod:`tensorsph.wall_mirror`) the ghost cloud is the *odd
  extension* of a near-wall velocity field, so the estimate stays first-order
  consistent up to the wall row instead of being truncated there.  It carries
  the discrete kernel-sum bias of the estimator (measured <= 2 % in the bulk on
  a linear field; ``tests/test_case_poiseuille_powerlaw.py`` pins the
  proportionality and the row uniformity rather than exactness);
* the closed-form plane-channel solution
  (:func:`power_law_channel_profile`) and its invariants, which are the
  reference this closure is graded against.

Closed form
-----------
Steady plane channel ``0 <= y <= H``, no-slip on both walls, uniform body
force per unit volume ``G = rho g`` along ``x``.  Momentum balance in the
channel direction is rheology independent,

    0 = G + dtau/dy   =>   tau(y) = G (H/2 - y),   tau_w = G H / 2,

and with ``tau = K |du/dy|**(n-1) du/dy`` the profile follows

    u(y) = n/(n+1) (G/K)**(1/n) [ (H/2)**((n+1)/n) - |y - H/2|**((n+1)/n) ],

    u_max  = n/(n+1) (G/K)**(1/n) (H/2)**((n+1)/n),
    gd_w   = (G H / (2K))**(1/n)          (wall shear rate).

``n = 1`` collapses to the Newtonian parabola ``G(H-y)y/(2K)`` bit-exactly
(:func:`power_law_channel_profile` special-cases it), and the wall shear stress
``G H/2`` is *independent of* ``n`` -- a redundant relation the benchmark uses
as an extra gate because a solver that mis-scales the stress still gets the
velocity shape roughly right.

Regularity
----------
``du/dy ~ |y - H/2|**(1/n)``, so for ``n > 1`` (shear thickening) the second
derivative of ``u`` is singular on the centre line and for ``n < 1``
(shear thinning) ``mu`` is unbounded there.  Both facts are physics, not a bug,
and both are handled explicitly (capped ``mu``; quoted centre-line error) rather
than hidden.  See ``docs/verification_case_poiseuille_powerlaw.md``.
"""

from __future__ import annotations

import math

import torch

from .droplet_impact import cubic_spline_gradient, density_from_neighbor_pairs
from .neighbor_search import cell_linked_pairs

__all__ = [
    "power_law_viscosity",
    "velocity_gradient",
    "shear_rate",
    "strain_rate_magnitude",
    "power_law_channel_profile",
    "power_law_channel_umax",
    "power_law_wall_shear_rate",
    "power_law_wall_shear_stress",
    "consistency_for_umax",
    "power_law_channel_residual",
]


# --------------------------------------------------------------------------- #
# Closure
# --------------------------------------------------------------------------- #

def power_law_viscosity(gamma_dot, K: float, n: float,
                        gamma_dot_floor: float | None = None,
                        mu_max: float | None = None):
    """Dynamic viscosity ``mu = K gd**(n-1)`` of an Ostwald-de Waele fluid.

    Parameters
    ----------
    gamma_dot : tensor or float
        Shear-rate magnitude (1/s).  Negative entries are rejected: the closure
        is a function of ``|gd|`` and a negative input means the caller forgot
        to take a norm.
    K : float
        Consistency index (Pa s**n).
    n : float
        Flow-behaviour index.  ``n = 1`` returns exactly ``K`` (no floating
        point round trip through the power), which is what lets the Newtonian
        control be bit-comparable.
    gamma_dot_floor : float, optional
        Floor applied to ``gd`` before the power.  Required for ``n < 1``,
        where the exact closure diverges as ``gd -> 0``; the floor bounds
        ``mu`` by ``K * floor**(n-1)``.  Default 0 for ``n >= 1`` and
        *mandatory* (a ``ValueError`` otherwise) for ``n < 1`` -- silently
        picking one would make the cap an invisible physics parameter.
    mu_max : float, optional
        Hard cap, applied after the floor.  ``None`` means no cap.
    """
    if not math.isfinite(K) or K <= 0.0:
        raise ValueError(f"K must be finite and positive, got {K!r}")
    if not math.isfinite(n) or n <= 0.0:
        raise ValueError(f"n must be finite and positive, got {n!r}")
    if mu_max is not None and (not math.isfinite(mu_max) or mu_max <= 0.0):
        raise ValueError(f"mu_max must be finite and positive, got {mu_max!r}")

    gd = gamma_dot if isinstance(gamma_dot, torch.Tensor) else torch.tensor(
        float(gamma_dot))
    if gd.is_complex():
        raise TypeError("gamma_dot must be real")
    if bool((gd < 0).any()):
        raise ValueError("gamma_dot must be non-negative (it is a magnitude)")
    floor = 0.0 if gamma_dot_floor is None else float(gamma_dot_floor)
    if n < 1.0:
        if not (floor > 0.0):
            raise ValueError(
                "n < 1 makes mu unbounded as gamma_dot -> 0; pass "
                "gamma_dot_floor > 0 (the cap is then K * floor**(n-1))")
    if floor < 0.0 or not math.isfinite(floor):
        raise ValueError(f"gamma_dot_floor must be finite and >= 0, got {floor!r}")

    if n == 1.0:
        mu = torch.full_like(gd, float(K))
    else:
        mu = K * torch.clamp(gd, min=floor).pow(n - 1.0)
    if mu_max is not None:
        mu = torch.clamp(mu, max=float(mu_max))
    return mu


# --------------------------------------------------------------------------- #
# Strain-rate estimate
# --------------------------------------------------------------------------- #

def velocity_gradient(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor,
                      h: float, pairs=None, rho: torch.Tensor | None = None,
                      neumann: torch.Tensor | None = None):
    """SPH estimate of ``grad v`` at every particle, shape ``(N, D, D)``.

    Standard symmetric-support form

        grad v_i = sum_j (m_j / rho_j) (v_j - v_i) (x) grad_i W_ij,

    with the sign convention pinned by
    ``tests/test_case_poiseuille_powerlaw.py`` on an exactly linear field
    (``grad v[i, 0, 1] = du/dy``).  ``rho`` defaults to a density summation over
    the same pairs; ``pairs`` defaults to a linked-cell search at ``2h``.
    """
    if pos.ndim != 2 or vel.shape != pos.shape or mass.shape != (pos.shape[0],):
        raise ValueError("pos and vel must be (N, D), mass must be (N,)")
    n, dim = pos.shape
    if rho is None:
        rho = density_from_neighbor_pairs(pos, mass, h, pairs=pairs)
    if pairs is None:
        pairs = cell_linked_pairs(pos, 2.0 * h)
    grad = torch.zeros((n, dim, dim), dtype=pos.dtype, device=pos.device)
    if pairs.numel() == 0:
        return grad
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    grad_w = cubic_spline_gradient(delta, h)
    dv = vel[j] - vel[i]
    contribution = ((mass[j] / rho[j])[:, None, None]
                    * dv[:, :, None] * grad_w[:, None, :])
    return grad.index_add(0, i, contribution)


def strain_rate_magnitude(grad: torch.Tensor) -> torch.Tensor:
    """``gamma_dot = sqrt(2 S:S)`` with ``S = (grad v + grad v^T)/2``.

    Invariant under a sign flip of the estimate (so the convention question
    above cannot silently invalidate a result), and equal to ``|du/dy|`` for a
    plane channel, where only ``S_xy = S_yx`` is non-zero.
    """
    s = 0.5 * (grad + grad.transpose(-1, -2))
    return (2.0 * (s * s).sum(dim=(-1, -2))).clamp_min(0.0).sqrt()


def shear_rate(pos, vel, mass, h: float, pairs=None, rho=None,
               neumann=None) -> torch.Tensor:
    """Convenience wrapper: strain-rate magnitude from an SPH velocity gradient."""
    return strain_rate_magnitude(
        velocity_gradient(pos, vel, mass, h, pairs=pairs, rho=rho,
                          neumann=neumann))


# --------------------------------------------------------------------------- #
# Analytic plane-channel reference
# --------------------------------------------------------------------------- #

def power_law_channel_profile(y, height: float, body_force_volume: float,
                              K: float, n: float):
    """Exact steady profile of a power-law fluid in a plane channel.

    ``y`` may be a float or a tensor; ``body_force_volume`` is ``G = rho g``
    (force per unit *volume*, Pa/m), not the per-mass acceleration.
    """
    if not (height > 0.0):
        raise ValueError("height must be positive")
    G = float(body_force_volume)
    if G <= 0.0:
        raise ValueError("body_force_volume must be positive")
    if n == 1.0:
        # Newtonian parabola, evaluated in the same form the rest of the
        # package uses so n = 1 is bit-identical to the mirror-wall case.
        gy = G / (2.0 * K)
        if isinstance(y, torch.Tensor):
            return gy * (height - y) * y
        return gy * (height - float(y)) * float(y)
    pref = (G / K) ** (1.0 / n) * (n / (n + 1.0))
    hh = (height / 2.0) ** ((n + 1.0) / n)
    if isinstance(y, torch.Tensor):
        return pref * (hh - (y - height / 2.0).abs().pow((n + 1.0) / n))
    return pref * (hh - abs(float(y) - height / 2.0) ** ((n + 1.0) / n))


def power_law_channel_umax(height: float, body_force_volume: float, K: float,
                           n: float) -> float:
    """Centre-line velocity of :func:`power_law_channel_profile`."""
    G = float(body_force_volume)
    if n == 1.0:
        return G * height * height / (8.0 * K)
    return ((G / K) ** (1.0 / n) * (n / (n + 1.0))
            * (height / 2.0) ** ((n + 1.0) / n))


def power_law_wall_shear_rate(height: float, body_force_volume: float, K: float,
                              n: float) -> float:
    """``du/dy`` magnitude at the wall: ``(G H / (2K))**(1/n)``."""
    return (float(body_force_volume) * height / (2.0 * K)) ** (1.0 / n)


def power_law_wall_shear_stress(height: float, body_force_volume: float) -> float:
    """``tau_w = G H / 2`` -- independent of ``K`` and ``n`` (momentum balance)."""
    return float(body_force_volume) * height / 2.0


def consistency_for_umax(height: float, body_force_volume: float, n: float,
                         umax: float) -> float:
    """Inverse of :func:`power_law_channel_umax`: ``K`` that gives ``u_max``.

    Used by the benchmark to hold the *velocity scale* fixed across the
    ``n`` ladder, so a difference between the regimes is a difference in shape
    and not in Reynolds number.
    """
    if not (umax > 0.0):
        raise ValueError("umax must be positive")
    G = float(body_force_volume)
    if n == 1.0:
        return G * height * height / (8.0 * umax)
    A = (n / (n + 1.0)) * (height / 2.0) ** ((n + 1.0) / n)
    return G * (A / umax) ** n


def power_law_channel_residual(y, height: float, body_force_volume: float,
                               K: float, n: float):
    """Residual of ``d/dy(K |u'|**(n-1) u') + G`` for the closed form.

    Evaluated with :mod:`torch`, so it can be run as an independent numerical
    check of the algebra (the symbolic version lives in the benchmark's
    provenance block).  At ``y = H/2`` the derivative of ``tau`` is one-sided
    discontinuous in ``|u'|**(n-1)u'`` for ``n != 1``; the returned tensor
    therefore excludes the exact centre line and the walls, where the
    expression is not differentiable.
    """
    yt = torch.as_tensor(y, dtype=torch.float64)
    H, G = float(height), float(body_force_volume)

    def tau(yy):
        s = (yy - H / 2.0)
        if n == 1.0:
            up = G * (H / 2.0 - s) / K
            return K * up
        gd = (G / K) ** (1.0 / n) * s.abs().pow(1.0 / n)
        return -K * gd.pow(n) * torch.sign(s)

    eps = 1.0e-6 * H
    return (tau(yt + eps) - tau(yt - eps)) / (2.0 * eps) + G