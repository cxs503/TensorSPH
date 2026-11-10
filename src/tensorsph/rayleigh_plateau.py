"""Linear capillary instability of a liquid cylinder (Rayleigh-Plateau theory).

The analytic anchors used to score the SPH liquid-column runs.

Inviscid dispersion (Rayleigh 1878; Lamb 1932 Art. 268)
-------------------------------------------------------
For a cylinder of radius ``R`` in an ambient fluid of density ``rho2`` the
axisymmetric (``m = 0``) mode of axial wavenumber ``k`` grows as
``exp(omega t)`` with

    omega^2(k) = sigma / (rho R^3) * x (1 - x^2) * I1(x) / I0(x),
    x = k R,   rho = rho1 + rho2,

with ``I0``, ``I1`` the modified Bessel functions.  ``x < 1`` is unstable
(capillary pressure in a neck exceeds the pressure in the swells once the
wavelength exceeds the circumference ``2 pi R``); ``x -> 0`` gives the
``sqrt(sigma k^2 / (2 rho R))`` slender-jet limit and ``x -> 1`` the neutral
point.  Maximising ``x (1 - x^2) I1/I0`` gives

    x* = 0.6971...,        lambda* = 2 pi R / x* = 9.0165... R,

the classical most-unstable wavelength (Plateau's ``lambda = 9.02 R``).  Note
that the *Bessel factor* is what moves the maximum away from the naive
``x = 1/sqrt(3) = 0.577`` (which is the maximum of ``x (1 - x^2)`` alone, i.e.
``lambda = 10.88 R``); a benchmark that only quotes ``9.02 R`` cannot tell the
two apart, so :func:`most_unstable_wavenumber` starts on the Bessel branch and
:func:`naive_wavenumber` is kept as the discriminator.

Viscous correction (Tomotika 1935)
----------------------------------
Viscosity damps the growth, ``omega_visc = -sigma / (2 rho nu R) * ...``; the
exact expression is transcendental.  What matters for a water jet is that the
Ohnesorge number is tiny, ``Oh = mu / sqrt(rho sigma R)``; for ``R = 6 mm``
water ``Oh = 1.5e-3``, so the inviscid anchor is the right reference to
``O(Oh)``.  :func:`ohnesorge` is exported so a run can state *why* the inviscid
formula applies instead of assuming it.

Planar control: a 2-D strip is stable
-------------------------------------
The 2-D analogue of a jet is a strip of half-thickness ``a``.  Its varicose
(squeezing) mode satisfies

    omega^2(k) = sigma k^3 tanh(k a) / rho   > 0  for all k,

i.e. the strip *oscillates*: surface tension shortens the interface
(``E = 2 sigma int sqrt(1 + eta'^2) dx``) and there is no azimuthal curvature to
convert a long wavelength into a pressure maximum at the neck.  This is why a
2-D SPH code cannot reproduce ``lambda = 9.02 R`` no matter how good its CSF
model is, and :func:`planar_strip_dispersion` exists so the negative control can
be asserted analytically rather than argued.

References
    Rayleigh, Lord (1878), Proc. London Math. Soc. 10:4.
    Lamb, H. (1932), Hydrodynamics, 6th ed., Art. 268.
    Tomotika, S. (1935), Proc. R. Soc. Lond. A 150:322.
    Eggers, J. & Villermaux, E. (2008), Rep. Prog. Phys. 71:036601 (review).
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "dispersion_rate_squared",
    "dispersion_rate",
    "most_unstable_wavenumber",
    "most_unstable_wavelength",
    "naive_wavenumber",
    "ohnesorge",
    "capillary_time",
    "planar_strip_dispersion",
    "slender_jet_dispersion",
]

#: Plateau's classical most-unstable wavelength in units of the cylinder radius.
LAMBDA_STAR_OVER_R = 9.016530021282


def _bessel(x):
    """``x (1 - x^2) I1(x) / I0(x)`` -- the dimensionless growth-rate shape."""
    if isinstance(x, torch.Tensor):
        xx = x
        return xx * (1.0 - xx.square()) * torch.special.i1(xx) / torch.special.i0(xx)
    x = float(x)
    return x * (1.0 - x * x) * _bessel_scalar(x)


def _bessel_scalar(x: float) -> float:
    """``I1(x)/I0(x)`` in double precision for scalars (no torch overhead)."""
    num, den, term_n, term_d, n = 0.0, 0.0, 1.0, 1.0, 0
    # Series: I0 = sum (x^2/4)^n/(n!)^2, I1 = (x/2) sum (x^2/4)^n/(n!(n+1)!)
    while term_d > 1e-18 * max(den, 1e-300) or n < 4:
        if n > 0:
            term_n *= (x * x) / (4.0 * n * (n + 1))
            term_d *= (x * x) / (4.0 * n * n)
        num += term_n
        den += term_d
        n += 1
        if n > 200:
            break
    return 0.5 * x * num / den


def dispersion_rate_squared(sigma: float, rho: float, radius: float,
                            k) -> torch.Tensor | float:
    """``omega^2`` of the inviscid Rayleigh-Plateau mode (negative = stable).

    ``rho`` is the *total* density ``rho1 + rho2``; for a jet in a gas,
    ``rho2 = 0`` and the density is the liquid one.
    """
    if isinstance(k, torch.Tensor):
        x = k * radius
        return (sigma / (rho * radius ** 3)) * _bessel(x)
    x = float(k) * radius
    return (sigma / (rho * radius ** 3)) * x * (1.0 - x * x) * _bessel_scalar(x)


def dispersion_rate(sigma: float, rho: float, radius: float, k):
    """``omega`` of the inviscid Rayleigh-Plateau mode (``nan`` when stable)."""
    w2 = dispersion_rate_squared(sigma, rho, radius, k)
    if isinstance(w2, torch.Tensor):
        return torch.sqrt(w2.clamp_min(0.0))
    return math.sqrt(w2) if w2 > 0.0 else float("nan")


def naive_wavenumber() -> float:
    """``1/sqrt(3) = 0.577``: the maximum of ``x (1 - x^2)`` without the Bessel
    factor.  Kept as the *discriminator* -- a scheme that ignores the Bessel
    ratio lands here and reports ``lambda = 10.9 R``."""
    return 1.0 / math.sqrt(3.0)


def most_unstable_wavenumber(tolerance: float = 1e-12) -> float:
    """``x* = k R`` maximising ``x (1 - x^2) I1(x)/I0(x)`` on ``(0, 1)``.

    Golden-section search; the Bessel factor is a smooth concave function on
    the unstable interval, so the search converges from a bracket of
    ``[0.4, 0.95]`` that brackets the ``0.697`` reported in the literature.
    """
    lo, hi = 0.4, 0.95
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    fc, fd = _bessel(c), _bessel(d)
    while hi - lo > tolerance:
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - inv_phi * (hi - lo)
            fc = _bessel(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + inv_phi * (hi - lo)
            fd = _bessel(d)
    return 0.5 * (lo + hi)


def most_unstable_wavelength(radius: float = 1.0) -> float:
    """``lambda* = 2 pi R / x*`` -- the breakup wavelength of a liquid cylinder."""
    return 2.0 * math.pi * radius / most_unstable_wavenumber()


def ohnesorge_guard(sigma: float, rho: float, radius: float, mu: float) -> bool:
    """``True`` when the inviscid anchor is valid to ``O(Oh)``."""
    return ohnesorge(sigma, rho, radius, mu) < 0.05


def ohnesorge(sigma: float, rho: float, radius: float, mu: float) -> float:
    """Ohnesorge number ``Oh = mu / sqrt(rho sigma R)`` (viscous/inertial ratio)."""
    return mu / math.sqrt(rho * sigma * radius)


def capillary_time(sigma: float, rho: float, radius: float) -> float:
    """``tau = sqrt(rho R^3 / sigma)`` -- the unit in which ``omega`` is quoted."""
    return math.sqrt(rho * radius ** 3 / sigma)


def planar_strip_dispersion(sigma: float, rho: float, half_thickness: float,
                            k: float) -> float:
    """``omega^2`` of the 2-D strip varicose mode: ``sigma k^3 tanh(k a)/rho``.

    Always positive, i.e. the strip is linearly *stable* -- the negative control
    for "the 2-D solver cannot fake a Rayleigh-Plateau breakup".
    """
    return sigma * k ** 3 * math.tanh(k * half_thickness) / rho


def slender_jet_dispersion(sigma: float, rho: float, radius: float,
                           k: float) -> float:
    """Long-wavelength limit ``omega^2 = sigma k^2 / (2 rho R)`` (``kR -> 0``).

    The ``kR -> 0`` limit of the Bessel branch; used to check that the exact
    dispersion is implemented consistently at the small-``k`` end of a sweep.
    """
    return sigma * k * k / (2.0 * rho * radius)


LAMBDA_STAR_OVER_R = most_unstable_wavelength(1.0)
