"""Exact hydrostatic equilibrium of a Tait (WCSPH) fluid column.

Why this module exists
----------------------
Weakly compressible SPH uses the Tait equation of state (EOS) ::

    p(rho) = B ((rho/rho0)**gamma - 1),      B = rho0 * c0**2 / gamma

so the density is *not* constant in a column at rest: it follows the
barotropic balance ``dp/dy = -rho(p) g``.  The incompressible reference
``p(y) = rho0 g (H - y)`` that the hydrostatic-tank case pool uses is therefore
the ``c0 -> infinity`` limit of the model, not its exact solution.  The
difference is a *modelling* bias that no mesh/particle refinement can remove,
because it is set by the acoustic Mach number and not by the resolution.

Solving the balance exactly (``rho_hat = rho/rho0``, ``h = H - y``)::

    dp/drho = c0**2 rho_hat**(gamma-1)
    c0**2 rho_hat**(gamma-1) rho0 drho_hat/dy = -rho0 rho_hat g
    =>  d/dy [ rho_hat**(gamma-1) ] = -(gamma-1) g / c0**2
    =>  rho_hat**(gamma-1) = 1 + (gamma-1) g h / c0**2          (rho_hat(H)=1)
    =>  p(y) = (rho0 c0**2 / gamma) (
            (1 + (gamma-1) g h / c0**2)**(gamma/(gamma-1)) - 1)   (*)

with the ``p(H) = 0`` gauge.  ``(*)`` is closed form -- no quadrature, no table,
no iteration.  Its small-``h`` expansion is uniformly ::

    p(y) = rho0 g h (1 + (1/2) g h / c0**2 + O((g h / c0**2)**2))

i.e. a positive *relative* bias of about ``50 g h / c0**2`` per cent,
``rho0``-independent and ``gamma``-independent to leading order.  Carrying one
more order, which is what the tests assert::

    bias % = 50 x (1 + (2 - gamma) x / 3 + ...),      x = g h / c0**2  For the
``CASE-HYDROSTATIC`` gate configuration (``H = 0.4 m``, ``c0 = 20 m/s``) the
bias is ``+0.45 %`` at the tank bottom -- comparable to that case's whole 1 %
tolerance, which is why it has to be separated from the discretisation error
before the tolerance is interpreted (see
``benchmarks/case_hydrostatic/eos_decomposition.py``).

Relationship to ``droplet_impact.hydrostatic_density``
-----------------------------------------------------
That helper inverts the EOS *at the incompressible pressure*:
``rho = B_inv(rho0 g h)``.  Two consequences worth being explicit about, because
they are easy to get backwards:

* its **density** is a first-order approximation to the equilibrium density:
  the two agree to ``O((g h / c0**2)**2)`` (``< 1e-4`` relative at Mach 0.14), so
  it is a legitimate initial-condition helper;
* its **pressure** (EOS applied back to that density) is *identically* the
  incompressible ``rho0 g h`` -- the inversion and the EOS cancel exactly.  So
  the helper carries the ideal reference and *not* the compressibility bias of
  the equilibrium solution, and it must not be used as a verification reference
  for pressure.

:func:`pointwise_eos_pressure` exposes that identity so it can be asserted
rather than remembered.

Validity
--------
``1 + (gamma-1) g h / c0**2 > 0`` holds for every physical column (``g, h >= 0``),
so ``(*)`` never leaves its branch; the only cost of a small ``c0`` is a larger
bias, not a breakdown.  ``p`` is strictly decreasing in ``y`` and reaches exactly
``0`` at the free surface by construction.
"""
from __future__ import annotations

import math

import torch

GAMMA_WATER = 7.0
RHO0_WATER = 1000.0
SOUND_SPEED_CASE = 20.0
GRAVITY = 9.81


def _as_float_tensor(x):
    if torch.is_tensor(x):
        return x.to(torch.float64) if x.dtype != torch.float64 else x
    return torch.as_tensor(x, dtype=torch.float64)


def _check(gamma: float, sound_speed: float, rho0: float) -> None:
    if gamma <= 1.0:
        raise ValueError(f"gamma must be > 1 for a Tait liquid EOS, got {gamma}")
    if sound_speed <= 0.0:
        raise ValueError(f"sound speed must be positive, got {sound_speed}")
    if rho0 <= 0.0:
        raise ValueError(f"rho0 must be positive, got {rho0}")


def bulk_modulus(rho0: float = RHO0_WATER,
                 sound_speed: float = SOUND_SPEED_CASE,
                 gamma: float = GAMMA_WATER) -> float:
    """Tait stiffness ``B = rho0 c0**2 / gamma`` (the repo's EOS constant)."""
    _check(gamma, sound_speed, rho0)
    return rho0 * sound_speed ** 2 / gamma


def tait_pressure(rho, rho0: float = RHO0_WATER,
                  sound_speed: float = SOUND_SPEED_CASE,
                  gamma: float = GAMMA_WATER):
    """EOS ``p = B ((rho/rho0)**gamma - 1)`` (tensor in, tensor out)."""
    rho = _as_float_tensor(rho)
    b = bulk_modulus(rho0, sound_speed, gamma)
    return b * ((rho / rho0) ** gamma - 1.0)


def tait_density(pressure, rho0: float = RHO0_WATER,
                 sound_speed: float = SOUND_SPEED_CASE,
                 gamma: float = GAMMA_WATER):
    """Inverse EOS ``rho = rho0 (1 + p/B)**(1/gamma)``."""
    pressure = _as_float_tensor(pressure)
    b = bulk_modulus(rho0, sound_speed, gamma)
    return rho0 * (1.0 + pressure / b) ** (1.0 / gamma)


def acoustic_parameter(y, depth: float, sound_speed: float = SOUND_SPEED_CASE,
                       gravity: float = GRAVITY):
    """Compressibility parameter ``x = g (H - y) / c0**2`` (dimensionless)."""
    if sound_speed <= 0.0:
        raise ValueError(f"sound speed must be positive, got {sound_speed}")
    y = _as_float_tensor(y)
    return gravity * (depth - y).clamp_min(0.0) / sound_speed ** 2


def eos_hydrostatic_pressure(y, depth: float, rho0: float = RHO0_WATER,
                             sound_speed: float = SOUND_SPEED_CASE,
                             gamma: float = GAMMA_WATER,
                             gravity: float = GRAVITY):
    """Exact Tait-consistent hydrostatic pressure ``(*)`` at height ``y``.

    ``dp/dy = -rho(p) g`` with the Tait EOS and ``p(depth) = 0``, integrated in
    closed form.  Reduces to ``rho0 g (depth - y)`` as ``sound_speed -> inf``
    (incompressible limit) and is exact for any finite ``c0``.
    """
    _check(gamma, sound_speed, rho0)
    x = acoustic_parameter(y, depth, sound_speed, gravity)
    b = bulk_modulus(rho0, sound_speed, gamma)
    # log1p/expm1 instead of ``(1+a)**s - 1``: at large c0 the bracket is
    # ``1 + O(1e-10)`` and the naive form cancels down to 0 in float64 (the
    # incompressible limit is the *only* regime where the two differ, so this
    # matters for every verification use of the module).
    exponent = gamma / (gamma - 1.0)
    return b * torch.expm1(exponent * torch.log1p((gamma - 1.0) * x))


def eos_hydrostatic_density(y, depth: float, rho0: float = RHO0_WATER,
                            sound_speed: float = SOUND_SPEED_CASE,
                            gamma: float = GAMMA_WATER,
                            gravity: float = GRAVITY):
    """Exact equilibrium density ``rho_hat**(gamma-1) = 1 + (gamma-1) x``."""
    _check(gamma, sound_speed, rho0)
    x = acoustic_parameter(y, depth, sound_speed, gravity)
    return rho0 * torch.exp(torch.log1p((gamma - 1.0) * x) / (gamma - 1.0))


def pointwise_eos_pressure(y, depth: float, rho0: float = RHO0_WATER,
                           sound_speed: float = SOUND_SPEED_CASE,
                           gamma: float = GAMMA_WATER,
                           gravity: float = GRAVITY):
    """First-order approximation used by ``droplet_impact.hydrostatic_density``.

    Inverts the EOS at the *incompressible* pressure ``rho0 g (depth - y)``
    instead of integrating the barotropic balance -- i.e. it reproduces the
    ``droplet_impact.hydrostatic_density`` construction.

    The inversion and the EOS cancel, so this function returns ``rho0 g (depth
    - y)`` *identically*: it is the ideal reference, not an approximation to
    :func:`eos_hydrostatic_pressure`.  Exposed so that the identity can be
    asserted (it is the reason that helper's pressure must not be used as a
    verification reference) rather than rediscovered.
    """
    _check(gamma, sound_speed, rho0)
    y = _as_float_tensor(y)
    ideal = rho0 * gravity * (depth - y).clamp_min(0.0)
    return tait_pressure(tait_density(ideal, rho0, sound_speed, gamma),
                         rho0, sound_speed, gamma)


def compressibility_bias_pct(y, depth: float, rho0: float = RHO0_WATER,
                             sound_speed: float = SOUND_SPEED_CASE,
                             gamma: float = GAMMA_WATER,
                             gravity: float = GRAVITY):
    """Relative bias of the exact EOS solution against ``rho0 g (H - y)``, in %.

    Positive everywhere below the free surface; leading order ``50 g h / c0**2``
    per cent, next order ``1 + (2 - gamma) x / 3`` with ``x = g h / c0**2``
    (``gamma`` enters only there, hence the weak ``gamma`` sensitivity).  This is the floor below which a WCSPH hydrostatic profile cannot
    agree with the incompressible reference *for a perfect discretisation*.
    """
    exact = eos_hydrostatic_pressure(y, depth, rho0, sound_speed, gamma, gravity)
    y = _as_float_tensor(y)
    ideal = rho0 * gravity * (depth - y).clamp_min(0.0)
    safe = torch.where(ideal > 0.0, ideal, torch.ones_like(ideal))
    bias = torch.where(ideal > 0.0, (exact - ideal) / safe * 100.0,
                       torch.zeros_like(ideal))
    return bias


def acoustic_mach(depth: float, sound_speed: float = SOUND_SPEED_CASE,
                  gravity: float = GRAVITY) -> float:
    """Acoustic Mach number ``sqrt(2 g H) / c0`` (the case's own diagnostic)."""
    return math.sqrt(2.0 * gravity * depth) / sound_speed


__all__ = [
    "GAMMA_WATER", "RHO0_WATER", "SOUND_SPEED_CASE", "GRAVITY",
    "acoustic_mach", "acoustic_parameter", "bulk_modulus",
    "compressibility_bias_pct", "eos_hydrostatic_density",
    "eos_hydrostatic_pressure", "pointwise_eos_pressure", "tait_density",
    "tait_pressure",
]