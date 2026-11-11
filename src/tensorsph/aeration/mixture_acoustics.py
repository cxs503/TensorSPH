"""Mixture acoustics of an aerated (bubbly) liquid: Wood / van Wijngaarden.

This module is the acoustic core of campaign node ``p457w2-n04`` (aeration buffer
decalibration).  It replaces the single calibration constant ``k_b`` -- which
used to knock the impact pressure down by a fixed factor -- with an
*alpha-resolved* impact-pressure law:

    p_peak(alpha, v) = Z(alpha) * v * [Mach correction] * [wall mismatch]

with the mixture impedance ``Z = rho_m c_m`` taken from two closed forms.

**Wood (1955) homogeneous-mixture equation** (exact for a homogeneous, no-slip,
low-frequency bubbly mixture):

    1 / (rho_m c_m^2) = alpha / (rho_g c_g^2) + (1 - alpha) / (rho_l c_l^2)
    rho_m            = alpha rho_g + (1 - alpha) rho_l

**van Wijngaarden (1972) low-frequency (dilute) limit**: the gas inertia is
negligible (``rho_m -> rho_l``) and the bubble response is the linearised
adiabatic gas law ``rho_g c_g^2 -> gamma_g p``:

    1 / c_m^2 = rho_l [ (1 - alpha) / (rho_l c_l^2) + alpha / (gamma_g p) ]

The two agree to ``O(alpha)`` and both are *low-frequency* results: valid while
``omega r_b / c_m << 1`` (see :func:`low_frequency_ratio`).  ``van_wijngaarden``
is additionally only a *dilute* limit -- at ``alpha -> 1`` it returns
``sqrt(1/(rho_l gamma_g p)) = 11.9 m/s`` instead of the gas sound speed
``c_g = 343 m/s``, because it dropped the gas density.  The domain boundary
``ALPHA_VW_MAX`` is therefore explicit and the function *fails closed* beyond
it instead of returning an unphysical number.

Everything here is closed-form (no iteration, no fitting).  In particular the
minimum-sound-speed point is available analytically: with

    A = 1/(rho_g c_g^2)      gas compressibility
    B = 1/(rho_l c_l^2)      liquid compressibility
    m = rho_l - rho_g        density contrast

the mixture ``c_m^2 = 1 / [(rho_l - m alpha)(B + (A - B) alpha)]`` is extremal
where ``d/dalpha[(rho_l - m alpha)(B + (A-B) alpha)] = 0``:

    alpha*  = [rho_l (A - B) - m B] / [2 m (A - B)]
    c_min   = 2 sqrt(m (A - B)) / [rho_l (A - B) + m B]

For air/water at 20 C, 1 atm: ``alpha* = 0.5006`` and ``c_min = 23.8 m/s``.
"""
from __future__ import annotations

import math

#: Liquid phase (fresh water, 20 C, 1 atm).
WATER = {"rho": 998.2, "c": 1483.0, "gamma": 7.0}
#: Dispersed phase (air, 20 C, 1 atm); ``gamma`` is the adiabatic exponent.
AIR = {"rho": 1.204, "c": 343.0, "gamma": 1.4}
#: Reference ambient pressure [Pa].
P_ATM = 101325.0
#: Steel hull plating (mild steel, longitudinal wave speed).
STEEL = {"rho": 7850.0, "c": 5900.0}

#: Below this void fraction the mixture impedance differs from the clear-water
#: value by less than 1e-5 relative -- i.e. below the resolution of the impact
#: law's own bookkeeping.  Treated as the minimum resolvable void fraction and
#: rejected by the domain gate (fail closed, never silently clamped).
ALPHA_MIN_RESOLVABLE = 1.0e-8
#: Upper end of the dilute domain of the van Wijngaarden limit.
ALPHA_VW_MAX = 0.05


class OutOfDomain(ValueError):
    """Raised when the aeration law is asked for a state outside its domain."""


def _phases(rho_l, rho_g, c_l, c_g):
    if not (rho_l > rho_g > 0.0 and c_l > 0.0 and c_g > 0.0):
        raise ValueError("require rho_l > rho_g > 0 and positive sound speeds")
    return float(rho_l), float(rho_g), float(c_l), float(c_g)


# --------------------------------------------------------------------------- #
# mixture state
# --------------------------------------------------------------------------- #
def mixture_density(alpha, rho_l=WATER["rho"], rho_g=AIR["rho"]):
    """``rho_m = alpha rho_g + (1 - alpha) rho_l`` (volume-averaged, no slip)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("void fraction must lie in [0, 1]")
    return alpha * rho_g + (1.0 - alpha) * rho_l


def mixture_compressibility(
    alpha, rho_l=WATER["rho"], rho_g=AIR["rho"], c_l=WATER["c"], c_g=AIR["c"]
):
    """``1/(rho_m c_m^2) = alpha/(rho_g c_g^2) + (1-alpha)/(rho_l c_l^2)``."""
    rho_l, rho_g, c_l, c_g = _phases(rho_l, rho_g, c_l, c_g)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("void fraction must lie in [0, 1]")
    return alpha / (rho_g * c_g * c_g) + (1.0 - alpha) / (rho_l * c_l * c_l)


def wood_sound_speed(
    alpha, rho_l=WATER["rho"], rho_g=AIR["rho"], c_l=WATER["c"], c_g=AIR["c"]
):
    """Wood (1955) mixture sound speed ``c_m(alpha)`` [m/s], closed form.

    Exact in the limits: ``c_m(0) = c_l`` and ``c_m(1) = c_g`` (to machine
    precision), which is what the acceptance limits C2 pin.
    """
    rho_m = mixture_density(alpha, rho_l, rho_g)
    comp = mixture_compressibility(alpha, rho_l, rho_g, c_l, c_g)
    return 1.0 / math.sqrt(rho_m * comp)


def van_wijngaarden_sound_speed(
    alpha,
    pressure=P_ATM,
    rho_l=WATER["rho"],
    c_l=WATER["c"],
    gamma_g=AIR["gamma"],
    alpha_max=ALPHA_VW_MAX,
):
    """van Wijngaarden (1972) low-frequency *dilute* limit of ``c_m(alpha)``.

    ``1/c^2 = rho_l [(1-alpha)/(rho_l c_l^2) + alpha/(gamma_g p)]``.

    Domain: ``0 <= alpha <= alpha_max`` (default :data:`ALPHA_VW_MAX`).
    Beyond it the limit is meaningless (it drops the gas density and therefore
    cannot tend to ``c_g`` as ``alpha -> 1``); the call fails closed instead.
    """
    if alpha < 0.0:
        raise ValueError("void fraction must be non-negative")
    if alpha > alpha_max:
        raise OutOfDomain(
            "van Wijngaarden low-frequency limit is a dilute-phase result: "
            f"alpha={alpha:g} > alpha_max={alpha_max:g}"
        )
    if pressure <= 0.0:
        raise ValueError("pressure must be positive")
    inv_c2 = rho_l * (
        (1.0 - alpha) / (rho_l * c_l * c_l) + alpha / (gamma_g * pressure)
    )
    return 1.0 / math.sqrt(inv_c2)


def _AB(rho_l, rho_g, c_l, c_g):
    return 1.0 / (rho_g * c_g * c_g), 1.0 / (rho_l * c_l * c_l)


def alpha_at_min_sound_speed(
    rho_l=WATER["rho"], rho_g=AIR["rho"], c_l=WATER["c"], c_g=AIR["c"]
):
    """Void fraction of minimum Wood sound speed, closed form (``dc_m/dalpha=0``)."""
    rho_l, rho_g, c_l, c_g = _phases(rho_l, rho_g, c_l, c_g)
    a_gas, b_liq = _AB(rho_l, rho_g, c_l, c_g)
    x = a_gas - b_liq
    if x <= 0.0:
        raise ValueError("gas phase must be the more compressible one")
    m = rho_l - rho_g
    return (rho_l * x - m * b_liq) / (2.0 * m * x)


def min_sound_speed(
    rho_l=WATER["rho"], rho_g=AIR["rho"], c_l=WATER["c"], c_g=AIR["c"]
):
    """Wood minimum sound speed ``c_m(alpha*)``, closed form.

    ``c_min = 2 sqrt(m x) / (rho_l x + m B)`` with ``x = A - B``.
    """
    rho_l, rho_g, c_l, c_g = _phases(rho_l, rho_g, c_l, c_g)
    a_gas, b_liq = _AB(rho_l, rho_g, c_l, c_g)
    x = a_gas - b_liq
    m = rho_l - rho_g
    return 2.0 * math.sqrt(m * x) / (rho_l * x + m * b_liq)


def acoustic_impedance(
    alpha, rho_l=WATER["rho"], rho_g=AIR["rho"], c_l=WATER["c"], c_g=AIR["c"]
):
    """``Z(alpha) = rho_m c_m`` [Pa s/m]."""
    rho_m = mixture_density(alpha, rho_l, rho_g)
    return rho_m * wood_sound_speed(alpha, rho_l, rho_g, c_l, c_g)


def mixture_gamma(alpha, gamma_l=WATER["gamma"], gamma_g=AIR["gamma"]):
    """Volume-harmonic mixing rule for the effective isentropic exponent.

    ``1/(Gamma-1) = alpha/(gamma_g-1) + (1-alpha)/(gamma_l-1)``; exact at both
    ends (``Gamma(0)=gamma_l``, ``Gamma(1)=gamma_g``).  Used only inside the
    finite-Mach correction, which is a weak-shock expansion.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("void fraction must lie in [0, 1]")
    inv = alpha / (gamma_g - 1.0) + (1.0 - alpha) / (gamma_l - 1.0)
    return 1.0 + 1.0 / inv


# --------------------------------------------------------------------------- #
# domain gate (fail closed)
# --------------------------------------------------------------------------- #
def domain_status(
    alpha,
    rho_l=WATER["rho"],
    rho_g=AIR["rho"],
    c_l=WATER["c"],
    c_g=AIR["c"],
    alpha_min=ALPHA_MIN_RESOLVABLE,
):
    """Classify ``alpha`` for the impact law: ``IN_DOMAIN`` / ``OUT_OF_DOMAIN``.

    The law is only meaningful on ``alpha_min <= alpha <= alpha*``:

    * ``alpha < alpha_min`` -- below the minimum resolvable void fraction
      (impedance indistinguishable from clear water);
    * ``alpha > alpha*`` -- on the descending branch of ``c_m(alpha)`` the
      impedance is *not* monotone in ``alpha``, so a void-fraction field would
      no longer map one-to-one onto a pressure reduction (and the Wood minimum
      sound speed, a resonance-dominated regime, is outside the scope of a
      quasi-static impact law).
    """
    alpha_star = alpha_at_min_sound_speed(rho_l, rho_g, c_l, c_g)
    if not math.isfinite(alpha) or alpha < 0.0:
        return {
            "status": "OUT_OF_DOMAIN",
            "alpha": alpha,
            "reason": "void fraction must be finite and non-negative",
            "alpha_star": alpha_star,
            "alpha_min": alpha_min,
        }
    if alpha < alpha_min:
        return {
            "status": "OUT_OF_DOMAIN",
            "alpha": alpha,
            "reason": "below the minimum resolvable void fraction",
            "alpha_star": alpha_star,
            "alpha_min": alpha_min,
        }
    if alpha > alpha_star:
        return {
            "status": "OUT_OF_DOMAIN",
            "alpha": alpha,
            "reason": "beyond the minimum-sound-speed void fraction alpha*",
            "alpha_star": alpha_star,
            "alpha_min": alpha_min,
        }
    return {
        "status": "IN_DOMAIN",
        "alpha": alpha,
        "reason": "monotone impedance branch, above the resolution floor",
        "alpha_star": alpha_star,
        "alpha_min": alpha_min,
    }


def require_domain(alpha, alpha_min=ALPHA_MIN_RESOLVABLE, **kwargs):
    """Raise :class:`OutOfDomain` unless ``alpha`` is inside the law's domain."""
    status = domain_status(alpha, alpha_min=alpha_min, **kwargs)
    if status["status"] != "IN_DOMAIN":
        raise OutOfDomain(f"{status['reason']} (alpha={alpha!r})")
    return status


# --------------------------------------------------------------------------- #
# impact pressure
# --------------------------------------------------------------------------- #
def water_hammer_pressure(
    alpha, velocity, rho_l=WATER["rho"], rho_g=AIR["rho"], c_l=WATER["c"], c_g=AIR["c"]
):
    """1-D water-hammer (Joukowsky) limit ``p = Z(alpha) v``; rigid wall."""
    if velocity < 0.0:
        raise ValueError("impact velocity must be non-negative")
    return acoustic_impedance(alpha, rho_l, rho_g, c_l, c_g) * velocity


def mach_correction(
    alpha,
    velocity,
    rho_l=WATER["rho"],
    rho_g=AIR["rho"],
    c_l=WATER["c"],
    c_g=AIR["c"],
    gamma_l=WATER["gamma"],
    gamma_g=AIR["gamma"],
):
    """Weak-shock (finite-Mach) correction to the acoustic water hammer.

    For a piston/shock in a medium of isentropic exponent ``Gamma`` the exact
    weak-shock expansion of the pressure jump is

        dp = rho c v [1 + (Gamma + 1)/4 * M + O(M^2)],  M = v / c_m,

    i.e. the acoustic (Joukowsky) value plus a first-order Mach term.  ``Gamma``
    is the mixture exponent of :func:`mixture_gamma`.  At ``alpha -> 0`` this
    tends to ``1 + 2M`` (water, ``Gamma = 7``), i.e. ~1.3% at ``v = 10 m/s`` --
    so the *acoustic* law is kept as the default and this correction is opt-in;
    the acceptance limits C2 are stated on the acoustic limit.
    """
    c_m = wood_sound_speed(alpha, rho_l, rho_g, c_l, c_g)
    mach = velocity / c_m
    gamma_m = mixture_gamma(alpha, gamma_l, gamma_g)
    return 1.0 + 0.25 * (gamma_m + 1.0) * mach


def wall_mismatch_factor(
    alpha,
    rho_s=STEEL["rho"],
    c_s=STEEL["c"],
    rho_l=WATER["rho"],
    rho_g=AIR["rho"],
    c_l=WATER["c"],
    c_g=AIR["c"],
):
    """1-D elastic-impact transmission factor ``Z_s / (Z_s + Z(alpha))``.

    A fluid column of impedance ``Z`` advancing at ``v`` onto a solid at rest
    leaves the interface at velocity ``u_s`` with ``Z (v - u_s) = Z_s u_s``,
    hence ``p = Z v * Z_s/(Z_s + Z)``.  A rigid wall (``Z_s -> inf``) recovers
    the full water hammer exactly; for steel in clear water the factor is
    0.969 (a 3.1% compliance relief).
    """
    z_fluid = acoustic_impedance(alpha, rho_l, rho_g, c_l, c_g)
    z_solid = rho_s * c_s
    return z_solid / (z_solid + z_fluid)


def impact_pressure(
    alpha,
    velocity,
    wall=None,
    finite_mach=False,
    strict=True,
    alpha_min=ALPHA_MIN_RESOLVABLE,
    rho_l=WATER["rho"],
    rho_g=AIR["rho"],
    c_l=WATER["c"],
    c_g=AIR["c"],
):
    """Aeration-buffered impact pressure ``p_peak(alpha, v)`` [Pa].

    ``p = Z(alpha) v`` times the optional finite-Mach correction and, when
    ``wall=(rho_s, c_s)`` is given, the acoustic mismatch factor.  With
    ``wall=None`` and ``finite_mach=False`` this is the rigid-wall acoustic
    water hammer, so ``p(0, v) = rho_l c_l v`` and ``p(1, v) = rho_g c_g v``
    hold to machine precision (acceptance C2).

    ``strict=True`` (default) *fails closed* through :func:`require_domain`: an
    ``alpha`` above ``alpha*`` or below the minimum resolvable void fraction
    raises :class:`OutOfDomain` instead of returning a plausible-looking number.
    """
    if strict:
        require_domain(
            alpha, alpha_min=alpha_min, rho_l=rho_l, rho_g=rho_g, c_l=c_l, c_g=c_g
        )
    p = water_hammer_pressure(alpha, velocity, rho_l, rho_g, c_l, c_g)
    if finite_mach:
        p *= mach_correction(alpha, velocity, rho_l, rho_g, c_l, c_g)
    if wall is not None:
        rho_s, c_s = wall
        p *= wall_mismatch_factor(alpha, rho_s, c_s, rho_l, rho_g, c_l, c_g)
    return p


def pressure_reduction(alpha, velocity=1.0, wall=None, **kwargs):
    """``G_peak(alpha) = 1 - p(alpha, v) / p(0, v)`` (dimensionless).

    This is a *mapping* function used for curves and inversions, so it is
    deliberately non-strict; the domain gate is applied where a number would be
    used as a prediction (:func:`impact_pressure`, ``impact_law.local_impact``).
    """
    p0 = impact_pressure(0.0, velocity, wall=wall, strict=False, **kwargs)
    pa = impact_pressure(alpha, velocity, wall=wall, strict=False, **kwargs)
    return 1.0 - pa / p0


def equivalent_buffer_coefficient(alpha, wall=None, **kwargs):
    """``k_b,eq(alpha) = p(alpha, v)/(rho_l c_l v)``.

    The constant ``k_b`` that the alpha-resolved law is equivalent to at this
    void fraction.  In the acoustic, rigid-wall limit ``k_b,eq = Z(alpha)/Z(0)``
    (independent of ``v``, which is exactly what makes it usable as a
    like-for-like replacement for the legacy constant); ``k_b,eq(0) = 1``.
    """
    k_eq = impact_pressure(alpha, 1.0, wall=wall, strict=False, **kwargs)
    return k_eq / impact_pressure(0.0, 1.0, wall=wall, strict=False, **kwargs)


def alpha_for_equivalent_buffer_coefficient(
    k_eq, lo=None, hi=None, tol=1.0e-14, **kwargs
):
    """Invert ``k_b,eq(alpha) = k_eq`` on the monotone branch (bisection).

    Answers the decalibration question "which void fraction does the legacy
    constant imply?" -- e.g. ``k_eq = 0.65`` corresponds to ``alpha ~ 9.7e-5``.
    """
    if not 0.0 < k_eq < 1.0:
        raise ValueError("equivalent buffer coefficient must lie in (0, 1)")
    if lo is None:
        lo = ALPHA_MIN_RESOLVABLE
    if hi is None:
        hi = alpha_at_min_sound_speed()
    f_lo = equivalent_buffer_coefficient(lo, **kwargs) - k_eq
    f_hi = equivalent_buffer_coefficient(hi, **kwargs) - k_eq
    if f_lo * f_hi > 0.0:
        raise ValueError("bracket does not contain the requested coefficient")
    a, b = lo, hi
    for _ in range(200):
        mid = 0.5 * (a + b)
        if equivalent_buffer_coefficient(mid, **kwargs) > k_eq:
            a = mid
        else:
            b = mid
        if b - a < tol * max(1.0, a):
            break
    return 0.5 * (a + b)


def low_frequency_ratio(alpha, radius, frequency, **kwargs):
    """``k a = omega r_b / c_m`` -- the low-frequency validity parameter.

    Wood and van Wijngaarden both assume the bubbles respond quasi-statically
    (``omega r_b / c_m << 1``).  For a 1 mm bubble in clear water at 1 kHz this
    is 4e-3; in a strongly aerated mixture (``c_m = 24 m/s``) it is 0.26, i.e.
    the assumption is close to its limit -- reported, not silently assumed.
    """
    c_m = wood_sound_speed(alpha, **kwargs)
    return 2.0 * math.pi * frequency * radius / c_m
