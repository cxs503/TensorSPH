"""Self-aeration inception on a sloping free surface (Chanson 1997 criterion).

The criterion used throughout this package is the dimensionless inception
parameter

    K = V sin(theta) / sqrt(g d cos(theta)),

with self-aeration starting once ``K`` exceeds ``C_crit``.  Chanson's
smooth-chute experiments place the onset at ``C_crit`` of order unity; the value
``C_CRIT = 1.0`` used here is the empirical constant carried by the campaign
specification (Chanson 1997, *Air Bubble Entrainment in Free-Surface Turbulent
Shear Flows*).

Two derived quantities are worth stating because they are what the benchmarks
actually compare:

* the **energy-balance derivation** of the constant.  A surface cavity of depth
  ``c d`` needs potential energy ``rho g cos(theta) c d`` per unit volume to be
  opened against the surface-normal gravity component, and it is fed by the
  surface-normal kinetic energy ``(1/2) rho (V sin theta)^2``.  Balancing the
  two gives ``V sin(theta) = sqrt(2 c) sqrt(g d cos(theta))``, i.e.
  ``C_crit = sqrt(2 c)``.  The closure ``c = 1/2`` (the cavity reaches half the
  flow depth, the classical order-of-magnitude choice for a breaking surface
  eddy) reproduces ``C_crit = 1.0``.  ``derived_crit_constant`` returns that
  value, and this is the only route by which an *independent* number can be put
  next to the empirical constant without a digitized experimental table.

* the **inception Froude number** ``Fr_i = V_i / sqrt(g d)``.  Eliminating ``d``
  from the criterion gives ``Fr_i = C_crit sqrt(cos theta) / sin theta``, which
  is steeply slope dependent: ``Fr_i = 1.19 C_crit`` at ``theta = 45 deg``,
  ``Fr_i = C_crit`` at ``theta = 51.8 deg`` (where ``cos = sin^2``) and
  ``Fr_i = 5.7 C_crit`` at ``theta = 10 deg``.  Any single "typical" inception Froude
  number quoted without a slope therefore carries no information, and this
  package never quotes one.

Honest scope note: no digitized Chanson inception table is bundled here, so the
benchmark compares the criterion's constant with the energy-balance prediction
(a consistency check on the closure, not independent experimental validation).
"""
from __future__ import annotations

import math

import torch

C_CRIT = 1.0
"""Empirical inception constant, ``K > C_crit`` triggers self-aeration."""

CAVITY_FRACTION = 0.5
"""Cavity depth as a fraction of the local flow depth at inception."""

GRAVITY = 9.81


def _t(value):
    return torch.as_tensor(value, dtype=torch.float64)


def _out(value):
    return float(value) if value.ndim == 0 else value


def chanson_parameter(velocity, depth, angle, gravity: float = GRAVITY):
    """``K = V sin(theta) / sqrt(g d cos(theta))``.

    ``angle`` is the chute (bed) angle in radians, measured from the horizontal.
    """
    v, d = _t(velocity), _t(depth)
    sin_a, cos_a = math.sin(angle), math.cos(angle)
    if sin_a <= 0.0:
        raise ValueError("chute angle must be positive: no normal velocity on a flat bed")
    return _out(v * sin_a / torch.sqrt(gravity * d * cos_a))


def inception_velocity(depth, angle, gravity: float = GRAVITY, c_crit: float = C_CRIT):
    """Streamwise velocity at which ``K`` reaches ``C_crit``."""
    d = _t(depth)
    return _out(c_crit * torch.sqrt(gravity * d * math.cos(angle)) / math.sin(angle))


def inception_depth(velocity, angle, gravity: float = GRAVITY, c_crit: float = C_CRIT):
    """Flow depth at which a given velocity reaches inception."""
    v = _t(velocity)
    scale = v * math.sin(angle) / c_crit
    return _out(scale.square() / (gravity * math.cos(angle)))


def inception_froude(angle, c_crit: float = C_CRIT):
    """Inception Froude number ``V/sqrt(g d) = c_crit sqrt(cos)/sin``."""
    return c_crit * math.sqrt(math.cos(angle)) / math.sin(angle)


def inception_froude_normal(angle, c_crit: float = C_CRIT):
    """Inception Froude number formed with the surface-normal gravity."""
    return c_crit / math.sin(angle)


def derived_crit_constant(cavity_fraction: float = CAVITY_FRACTION) -> float:
    """``C_crit = sqrt(2 c)`` from the surface energy balance.

    ``c`` is the inception cavity depth normalised by the flow depth.  With
    ``c = 1/2`` this returns exactly ``1.0``, the empirical constant.
    """
    if not 0.0 < cavity_fraction <= 1.0:
        raise ValueError("cavity_fraction must lie in (0, 1]")
    return math.sqrt(2.0 * cavity_fraction)


def aeration_active(parameter, c_crit: float = C_CRIT):
    """Boolean mask: is the criterion exceeded?"""
    k = _t(parameter)
    out = k > c_crit
    return bool(out) if out.ndim == 0 else out


def entrainment_flux(
    velocity,
    depth,
    angle,
    k_a: float = 0.02,
    gravity: float = GRAVITY,
    c_crit: float = C_CRIT,
):
    """Bulk air flux entrained per unit width [m^2/s].

    Closed with the single declared coefficient ``k_a``: the entrainment rate is
    proportional to the *excess* of the inception parameter over the threshold,

        ``q_a = k_a * max(K - c_crit, 0) * V d``.

    ``k_a`` is a model constant of this package, not an experimental value; the
    benchmarks use it only to locate the onset (where ``q_a`` becomes positive),
    so no accuracy claim rides on its magnitude.
    """
    k = _t(chanson_parameter(velocity, depth, angle, gravity=gravity))
    v, d = _t(velocity), _t(depth)
    excess = (k - c_crit).clamp_min(0.0)
    return _out(k_a * excess * v * d)


def onset_velocity_by_bisection(
    depth,
    angle,
    gravity: float = GRAVITY,
    c_crit: float = C_CRIT,
    bounds=(1.0e-6, 1.0e3),
    iterations: int = 200,
):
    """Locate the inception velocity numerically, independently of the algebra.

    Bisects the sign change of :func:`entrainment_flux`, which is monotone in
    velocity.  Used by the benchmarks to check that the algebraic
    :func:`inception_velocity` and the operational onset detector agree.
    """
    lo, hi = bounds
    if float(entrainment_flux(hi, depth, angle, gravity=gravity, c_crit=c_crit)) <= 0.0:
        raise ValueError("no onset inside the bracket: upper bound does not entrain")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if float(entrainment_flux(mid, depth, angle, gravity=gravity, c_crit=c_crit)) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1.0e-14 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)
