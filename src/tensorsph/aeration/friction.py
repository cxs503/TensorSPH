"""Frictional resistance of an aerated (bubbly) flow.

Two standard closures are implemented and cross-checked in the benchmark:

**Chisholm (1967) two-phase multiplier.**  With the Lockhart-Martinelli parameter

    X^2 = (G_l / G_g)^1.8 (rho_g / rho_l)(mu_l / mu_g)^0.2      (turbulent-turbulent)

the frictional pressure gradient of the mixture is ``Phi_l^2`` times the liquid
gradient, with

    Phi_l^2 = 1 + C / X + 1 / X^2,     C = 20 (turbulent-turbulent).

**Wood (1991)-form log-law closure.**  The wall shear of the mixture follows from
the logarithmic velocity profile ``u+ = (1/kappa_eff) ln y+ + B`` with a
von-Karman constant that the bubbles may modify,
``kappa_eff = kappa_0 (1 - gamma alpha)``.  ``gamma`` is a *declared* model
constant: in this deliverable it is left at ``gamma = 0`` so that the reduction
comes purely from the mixture density and the mixture Reynolds number, because
no drag-reduction calibration dataset is bundled and fitting ``gamma`` to the
Chisholm answer would make the cross-check circular.  The hook exists so that a
future calibration has a single home.

Both routes return a **reduction rate** at the same mixture volumetric flux
``u_m``, referenced to the liquid alone at that same flux:

    R = 1 - (dp/dz)_aerated / (dp/dz)_liquid-only .
"""
from __future__ import annotations

import math

from .bubble_transport import AIR_20C, WATER_20C

CHISHOLM_C = 20.0
"""Chisholm constant for turbulent-turbulent flow."""


def lockhart_martinelli_x(
    liquid_flux: float,
    gas_flux: float,
    rho_l: float = WATER_20C["rho"],
    rho_g: float = AIR_20C["rho"],
    mu_l: float = WATER_20C["mu"],
    mu_g: float = 1.8e-5,
):
    """``X^2 = (G_l/G_g)^1.8 (rho_g/rho_l)(mu_l/mu_g)^0.2`` (turbulent-turbulent)."""
    if liquid_flux <= 0.0 or gas_flux <= 0.0:
        raise ValueError("both phase fluxes must be positive")
    g_l = liquid_flux * rho_l
    g_g = gas_flux * rho_g
    x_sq = (g_l / g_g) ** 1.8 * (rho_g / rho_l) * (mu_l / mu_g) ** 0.2
    return math.sqrt(x_sq)


def chisholm_multiplier(x, c: float = CHISHOLM_C):
    """``Phi_l^2 = 1 + C/X + 1/X^2``."""
    if x <= 0.0:
        raise ValueError("Lockhart-Martinelli parameter must be positive")
    return 1.0 + c / x + 1.0 / (x * x)


def friction_factor_loglaw(reynolds, kappa: float = 0.41, b: float = 5.0, iterations: int = 200):
    """Friction factor from the logarithmic law of the wall.

    Solves ``sqrt(8/f) = (1/kappa)(ln(Re sqrt(f/8)) - 1.5) + B`` by fixed-point
    iteration.  Valid in the turbulent regime (``Re`` well above 4e3).
    """
    if reynolds < 1.0e3:
        raise ValueError("log-law friction factor is not defined below Re = 1e3")
    f = 0.02
    for _ in range(iterations):
        root = math.sqrt(f / 8.0)
        rhs = (math.log(reynolds * root) - 1.5) / kappa + b
        f_new = 8.0 / (rhs * rhs)
        if abs(f_new - f) <= 1.0e-14 * f_new:
            f = f_new
            break
        f = 0.5 * (f + f_new)
    return f


def wood_kappa_effective(kappa0: float, alpha, gamma: float = 0.0):
    """``kappa_eff = kappa_0 (1 - gamma alpha)``."""
    return kappa0 * (1.0 - gamma * float(alpha))


def aerated_friction_factor(
    mixture_flux: float,
    alpha: float,
    diameter: float,
    rho_l: float = WATER_20C["rho"],
    rho_g: float = AIR_20C["rho"],
    mu_l: float = WATER_20C["mu"],
    kappa: float = 0.41,
    b: float = 5.0,
    gamma: float = 0.0,
):
    """Log-law friction factor of the homogeneous bubbly mixture."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("void fraction must lie in [0, 1)")
    rho_m = rho_l * (1.0 - alpha) + rho_g * alpha
    re_m = mixture_flux * diameter * rho_m / mu_l
    kappa_eff = wood_kappa_effective(kappa, alpha, gamma)
    return friction_factor_loglaw(re_m, kappa=kappa_eff, b=b)


def chisholm_reduction_rate(
    alpha: float,
    mixture_flux: float,
    diameter: float,
    rho_l: float = WATER_20C["rho"],
    rho_g: float = AIR_20C["rho"],
    mu_l: float = WATER_20C["mu"],
    mu_g: float = 1.8e-5,
    c: float = CHISHOLM_C,
):
    """Frictional reduction rate implied by the Chisholm multiplier.

    ``R = 1 - Phi_l^2 [f(j_l)/f(u_m)] (1 - alpha)^2``, where ``j_l = (1-alpha) u_m``
    is the liquid superficial velocity and both friction factors are the liquid
    log-law values.
    """
    j_l = (1.0 - alpha) * mixture_flux
    j_g = alpha * mixture_flux
    x = lockhart_martinelli_x(j_l, j_g, rho_l, rho_g, mu_l, mu_g)
    phi_sq = chisholm_multiplier(x, c=c)
    re_l = j_l * diameter * rho_l / mu_l
    re_m = mixture_flux * diameter * rho_l / mu_l
    f_ratio = friction_factor_loglaw(re_l) / friction_factor_loglaw(re_m)
    return 1.0 - phi_sq * f_ratio * (1.0 - alpha) ** 2


def loglaw_reduction_rate(
    alpha: float,
    mixture_flux: float,
    diameter: float,
    rho_l: float = WATER_20C["rho"],
    rho_g: float = AIR_20C["rho"],
    mu_l: float = WATER_20C["mu"],
    kappa: float = 0.41,
    b: float = 5.0,
    gamma: float = 0.0,
):
    """Frictional reduction rate from the mixture log-law closure."""
    f_m = aerated_friction_factor(
        mixture_flux, alpha, diameter, rho_l, rho_g, mu_l, kappa, b, gamma
    )
    re_l = mixture_flux * diameter * rho_l / mu_l
    f_l = friction_factor_loglaw(re_l, kappa=kappa, b=b)
    rho_ratio = (rho_l * (1.0 - alpha) + rho_g * alpha) / rho_l
    return 1.0 - (f_m / f_l) * rho_ratio
