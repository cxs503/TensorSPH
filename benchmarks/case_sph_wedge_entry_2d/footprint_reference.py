"""Continuum limit of the finite-area-averaged peak pressure (m21742).

Why this module exists
----------------------
``footprint_peak.py`` (m21677) grades ``Cp_Phi`` = maximum over sliding windows
of width ``Phi`` of the wall pressure coefficient.  Its reference side is built by
*re-sampling the analytic Wagner field on the SPH particle grid* and pushing it
through the same window operator (``estimator_bias.SynthChain``).  That removes
the footprint-definition bias (both sides average over the same finite area) but
it does **not** remove the *sampling* bias: the window mean of a cusp-shaped field
estimated from a finite number of samples under-reads the maximum, and it
under-reads it more the coarser the grid.  So the same-operator residual
``r = obs/synth - 1`` is only asymptotically unbiased in ``dx``.

This module supplies the missing denominator analytically.  The Wagner wall
pressure in self-similar coordinates is

    p/q(tau) = 2 C s - s^2 + 1,      s = 1/sqrt(1 - tau^2),   C = pi/(2 tan beta)

and its mean over a window ``[tau1, tau2]`` has a closed form because
``int s dtau = asin tau`` and ``int s^2 dtau = artanh tau``:

    <p/q>_[t1,t2] = ( 2C (asin t2 - asin t1) - (artanh t2 - artanh t1) + (t2-t1) )
                    / (t2 - t1)                                            (1)

The finite-area reading of the *ideal* field is therefore
``Cp_Phi^inf = max over fully-wet windows of (1)``, and the ratio

    R(eps, beta) = Cp_Phi^inf / Cp_max,        eps = Phi / c                        (2)

is a pure number that depends only on the footprint expressed in similarity
coordinates.  ``R`` is what a perfect (infinitely resolved) simulation would read
through a transducer of width ``Phi``; its deficit from 1 is the part of the
measured gap that is *not* solver error.

Two consequences used by the case:

1. **Metric-consistent acceptance denominator.**  The case is graded against the
   Zhao & Faltinsen (1993) composite point maximum ``Cp_max = 6.92`` (q units, via
   the open-access table of Jain et al. 2022).  A finite-area reading must be
   compared with that point value *through the same footprint*:
   ``Cp_max * R(eps, beta)``.  Comparing ``Cp_Phi`` with the bare point value
   charges the solver for the sensor, which at ``beta = 30 deg``,
   ``Phi = 8 mm``, ``c_end = 50 mm`` is a ``-5.5 %`` shift (``R = 0.945``).
2. **Reference-model spread.**  The same footbed is applied to the Wagner level
   (``C^2+1 = 8.40``) and to the composite level (``6.92``): those two published
   models of the *same* idealisation differ by 21 %, i.e. more than the 15 %
   gate.  Reporting one without the other makes the verdict look sharper than the
   reference allows.

Everything here is closed-form; no solver call and no stored artefact is needed,
so the numbers can be re-derived in a test at machine precision.

Usage
-----
::

    from footprint_reference import reference_denominators
    reference_denominators(beta_deg=30.0, phi_m=8e-3, c_end=0.05)
"""

from __future__ import annotations

import math

Q_UNIT = 2.0                     # p/q = 2 * p/(rho V^2)
TAU_EDGE_GUARD = 1.0e-9          # the window must stay strictly inside tau <= 1


def c_wagner(beta_rad: float) -> float:
    """Wet-width coefficient ``C = pi / (2 tan beta)`` (Wagner 1932)."""
    t = math.tan(beta_rad)
    if t <= 0.0:
        raise ValueError("beta must be in (0, pi/2)")
    return math.pi / (2.0 * t)


def cp_max_point(beta_rad: float) -> float:
    """Point maximum of the Wagner profile in ``q`` units, ``C^2 + 1``."""
    C = c_wagner(beta_rad)
    return C * C + 1.0 if C >= 1.0 else Q_UNIT * C


def tau_peak(beta_rad: float) -> float:
    """Similarity coordinate of the point maximum, ``sqrt(1 - 1/C^2)``."""
    C = c_wagner(beta_rad)
    return math.sqrt(max(0.0, 1.0 - 1.0 / (C * C)))


def profile_q(tau: float, beta_rad: float) -> float:
    """``p/q`` of the Wagner profile at ``tau`` (``inf`` at and beyond ``tau=1``)."""
    t = abs(float(tau))
    if t >= 1.0:
        return float("inf")
    return 2.0 * c_wagner(beta_rad) / math.sqrt(1.0 - t * t) - t * t / (1.0 - t * t)


def _artanh(x: float) -> float:
    return 0.5 * math.log((1.0 + x) / (1.0 - x))


def window_mean_q(tau1: float, tau2: float, beta_rad: float) -> float:
    """Closed form (1): mean of ``p/q`` over ``[tau1, tau2]``, fully wet window."""
    if not (0.0 <= tau1 < tau2):
        raise ValueError("need 0 <= tau1 < tau2")
    t2 = min(tau2, 1.0 - TAU_EDGE_GUARD)
    if t2 <= tau1:
        raise ValueError("window collapsed inside the edge guard")
    C = c_wagner(beta_rad)
    width = t2 - tau1
    num = (2.0 * C * (math.asin(t2) - math.asin(tau1))
           - (_artanh(t2) - _artanh(tau1))
           + width)
    return num / width


def window_mean_q_numeric(tau1: float, tau2: float, beta_rad: float,
                          n: int = 20001) -> float:
    """Same integral by quadrature -- used by the tests to pin the closed form."""
    if n % 2 == 0:
        n += 1
    h = (tau2 - tau1) / (n - 1)
    acc = 0.0
    for k in range(n):
        tau = tau1 + k * h
        w = 1.0 if k in (0, n - 1) else (4.0 if k % 2 else 2.0)
        acc += w * profile_q(min(tau, 1.0 - 1e-12), beta_rad)
    return (acc * h / 3.0) / (tau2 - tau1)


def footprint_ratio(eps: float, beta_rad: float, n_scan: int = 4000) -> float:
    """``R(eps) = Cp_Phi^inf / Cp_max``, eq. (2).

    The maximum over windows fully inside the wetted footprint.  ``eps >= 1`` is
    clamped to the widest admissible window (``eps -> 1``), and windows are
    scanned on ``n_scan + 1`` positions so the reported value is reproducible.
    """
    if not (0.0 < eps):
        raise ValueError("eps must be positive")
    e = min(float(eps), 1.0 - 1e-6)
    best = -float("inf")
    for k in range(n_scan + 1):
        t1 = (1.0 - e) * k / n_scan
        best = max(best, window_mean_q(t1, t1 + e, beta_rad))
    return best / cp_max_point(beta_rad)


def required_eps(target_ratio: float, beta_rad: float,
                 lo: float = 1e-4, hi: float = 0.999) -> float:
    """Footprint ``eps`` at which the ideal field reads ``target_ratio`` of the point max.

    ``R`` is strictly decreasing in ``eps`` on ``(0, 1)`` (a wider window averages a
    sharper cusp), so bisection is well posed.  Raises if ``target_ratio`` is not
    reachable inside ``[lo, hi]`` -- never silently extrapolates.
    """
    r_lo, r_hi = footprint_ratio(lo, beta_rad), footprint_ratio(hi, beta_rad)
    if not (r_hi <= target_ratio <= r_lo):
        raise ValueError(f"target {target_ratio:.4f} outside reachable "
                         f"[{r_hi:.4f}, {r_lo:.4f}] for this beta")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if footprint_ratio(mid, beta_rad) > target_ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def reference_denominators(beta_deg: float, phi_m: float, c_end: float,
                           designated_cp: float, exact_cp: float | None = None,
                           wagner_cp: float | None = None) -> dict:
    """Metric-consistent denominators for a finite-area reading.

    ``designated_cp`` / ``exact_cp`` / ``wagner_cp`` are *point* maxima in ``q``
    units (Zhao & Faltinsen COM, Dobrovol'skaya/Wang & Faltinsen EIM, Wagner
    ``C^2+1``).  Each is multiplied by the same continuum footprint ratio, which is
    the only thing a transducer of width ``phi_m`` can report from that profile
    shape at wetted half-width ``c_end``.
    """
    beta = math.radians(beta_deg)
    eps = phi_m / c_end
    r = footprint_ratio(eps, beta)
    out = {
        "beta_deg": beta_deg,
        "phi_m": phi_m,
        "c_end_m": c_end,
        "eps": eps,
        "footprint_ratio": r,
        "point_wagner": cp_max_point(beta),
        "tau_peak": tau_peak(beta),
        "denominator_designated": designated_cp * r,
        "point_designated": designated_cp,
    }
    if exact_cp is not None:
        out["point_exact"] = exact_cp
        out["denominator_exact"] = exact_cp * r
    if wagner_cp is not None:
        out["point_wagner_reported"] = wagner_cp
        out["denominator_wagner"] = wagner_cp * r
    return out