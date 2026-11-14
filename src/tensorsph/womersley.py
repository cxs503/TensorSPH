"""Pulsatile (Womersley) plane-channel flow: closed forms, series and checks.

Why this module exists
----------------------
The repository verifies the channel viscous operator in three ways that are all
*steady in the mean*: the steady parabola held against a constant body force
(``benchmarks/poiseuille_mirror``), the start-up from rest against the
Boussinesq series (``benchmarks/poiseuille_startup``), and the oscillating
plate against Stokes' second problem (``src/tensorsph/stokes_second.py``,
``benchmarks/stokes_second``).  None of them measures the **frequency
response** of the channel to a time-dependent pressure gradient, which is the
classic pulsatile-flow problem of the arterial/oil-pipe literature.

The case
--------
Fluid in ``0 <= y <= H``, no-slip at both walls, periodic in ``x``, driven by a
uniform body force (pressure gradient) ``g(t) = g0 cos(omega t)`` per unit mass::

    u_t = nu u_yy + g0 cos(omega t),      u(0,t) = u(H,t) = 0              (1)

The periodic state of (1) is *not* the quasi-steady parabola.  With
``alpha = sqrt(i omega / nu)`` and the time dependence ``Re{ Uhat(y) e^{i omega t} }``,

    Uhat(y) = (g0 / (i omega)) [ 1 - cosh(alpha (y - H/2)) / cosh(alpha H/2) ]   (2)

which satisfies ``i omega Uhat = nu Uhat'' + g0`` and both no-slip conditions
identically (:func:`verify_symbolic` proves those three statements with sympy).
The solution is usually written with the Womersley number

    Wo = (H/2) sqrt(omega / nu)                                              (3)

and the Stokes layer thickness ``delta = sqrt(2 nu / omega)``.  For ``Wo -> 0``
(2) collapses to the steady parabola ``u = g0 (H - y) y / (2 nu)``; for
``Wo >> 1`` the amplitude is flat in the core and the wall layer carries the
shear, with a phase that lags the forcing by up to 45 deg.  This module also
provides the from-rest start-up series so the transient part of a run can be
accounted for instead of assumed away.

Reference provenance
--------------------
* Womersley, J. R. (1955) *J. Physiol.* **127**:553-563 (the physiological
  solution); the plane-channel form (2) is the same balance.
* Batchelor, *An Introduction to Fluid Dynamics* (1967) §4.2 (unsteady
  plane Poiseuille flow, eigenfunction expansion).
* Landau & Lifshitz, *Fluid Mechanics* 2nd ed. §24 (Stokes layer scaling).
* Loudon & Tordesillas (1998), *J. Fluid Eng.* (pulsatile channel expansions).

Nothing here is hard-coded to a particular benchmark: every function takes SI
inputs and returns SI outputs, so the case harness in
``benchmarks/case_poiseuille`` fixes the parameters, not this module.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

#: Case-pool identifier serviced by this reference set.
CASE_KEY = "CASE-POISEUILLE-SPH"

__all__ = [
    "CASE_KEY",
    "womersley_number",
    "stokes_layer_thickness",
    "steady_velocity",
    "complex_amplitude",
    "amplitude",
    "phase",
    "periodic_velocity",
    "series_from_rest",
    "startup_velocity",
    "verify_symbolic",
    "verify_numeric",
]


def _require_positive(**kwargs: float) -> None:
    for name, value in kwargs.items():
        if not (value > 0.0):
            raise ValueError(f"{name} must be positive, got {value!r}")


# --------------------------------------------------------------------------- #
# Scaling groups
# --------------------------------------------------------------------------- #

def womersley_number(height: float, viscosity: float, omega: float) -> float:
    """``Wo = (H/2) sqrt(omega / nu)`` -- the unsteadiness of the channel."""
    _require_positive(height=height, viscosity=viscosity, omega=omega)
    return 0.5 * height * math.sqrt(omega / viscosity)


def stokes_layer_thickness(viscosity: float, omega: float) -> float:
    """``delta = sqrt(2 nu / omega)`` -- the wall layer that carries the shear."""
    _require_positive(viscosity=viscosity, omega=omega)
    return math.sqrt(2.0 * viscosity / omega)


# --------------------------------------------------------------------------- #
# Closed forms
# --------------------------------------------------------------------------- #

def steady_velocity(y, height: float, body_force: float, viscosity: float):
    """The ``omega -> 0`` limit of (2): ``u = g (H - y) y / (2 nu)``."""
    _require_positive(height=height, viscosity=viscosity)
    y = np.asarray(y, dtype=float)
    return body_force * (height - y) * y / (2.0 * viscosity)


def complex_amplitude(y, height: float, body_force: float, viscosity: float,
                      omega: float):
    """Complex amplitude ``Uhat(y)`` of the periodic state, equation (2).

    The physical velocity is ``Re{ Uhat(y) exp(i omega t) }``.  The function is
    returned for any real ``y`` vector and is exactly zero at ``y = 0`` and
    ``y = H`` (to round-off).
    """
    _require_positive(height=height, viscosity=viscosity, omega=omega)
    y = np.asarray(y, dtype=float)
    alpha = np.sqrt(1j * omega / viscosity)
    half = 0.5 * height
    bracket = 1.0 - np.cosh(alpha * (y - half)) / np.cosh(alpha * half)
    return (body_force / (1j * omega)) * bracket


def amplitude(y, height: float, body_force: float, viscosity: float,
              omega: float):
    """``|Uhat(y)|`` -- the amplitude the harmonic fit of a run should recover."""
    return np.abs(complex_amplitude(y, height, body_force, viscosity, omega))


def phase(y, height: float, body_force: float, viscosity: float,
          omega: float):
    """Phase ``arg Uhat(y)`` in radians, with ``u ~ A cos(omega t + phase)``.

    This is the same convention as
    :func:`tensorsph.stokes_second.harmonic_fit` (which returns
    ``-atan2(b, a)`` for ``u = a cos + b sin``), so measured and analytic
    phases are directly comparable, modulo ``2 pi``.
    """
    return np.angle(complex_amplitude(y, height, body_force, viscosity, omega))


def periodic_velocity(y, t: float, height: float, body_force: float,
                      viscosity: float, omega: float):
    """The periodic-state velocity ``Re{ Uhat(y) exp(i omega t) }``."""
    uhat = np.atleast_1d(complex_amplitude(y, height, body_force, viscosity, omega))
    t = np.asarray(t, dtype=float)
    if t.ndim == 0:
        return np.real(uhat * np.exp(1j * omega * float(t)))
    # (ny,) x (nt,) -> (ny, nt): one row per ``y``, one column per time
    return np.real(uhat[:, None] * np.exp(1j * omega * t)[None, :])


# --------------------------------------------------------------------------- #
# From-rest series (transient included)
# --------------------------------------------------------------------------- #

def series_from_rest(y, t: float, height: float, body_force: float,
                     viscosity: float, omega: float = 0.0,
                     n_terms: int = 801):
    """Exact solution of (1) started from rest, by eigenfunction expansion.

    With ``lambda_n = nu (n pi / H)^2`` and odd ``n`` only,

        u(y,t) = sum_n (4 g0 / (n pi)) / (lambda_n^2 + omega^2) *
                 [ lambda_n cos(omega t) + omega sin(omega t)
                   - lambda_n exp(-lambda_n t) ] sin(n pi y / H)

    which is the Duhamel solution for the harmonic forcing and reduces at
    ``omega = 0`` to the Boussinesq start-up series
    ``u = u_s (1 - (32/pi^3) sum sin(n pi y/H) exp(-lambda_n t) / n^3)``.
    ``n_terms`` odd modes are summed; convergence is fast once
    ``lambda_1 t >~ 1`` and slow for ``t -> 0`` (a logarithmically singular
    corner in time), which is why the case harness grades late times only.
    """
    _require_positive(height=height, viscosity=viscosity)
    if t < 0.0:
        raise ValueError("t must be non-negative")
    y = np.asarray(y, dtype=float).reshape(-1)
    n = np.arange(1, 2 * int(n_terms) + 1, 2, dtype=float)
    lam = viscosity * (n * math.pi / height) ** 2
    proj = 4.0 * body_force / (n * math.pi)
    denom = lam ** 2 + omega ** 2
    if omega == 0.0:
        time_part = (lam - lam * np.exp(-lam * t)) / denom
    else:
        time_part = (lam * math.cos(omega * t) + omega * math.sin(omega * t)
                     - lam * np.exp(-lam * t)) / denom
    modes = np.sin(np.outer(n, y) * math.pi / height)          # (n_modes, ny)
    return modes.T @ (proj * time_part)


def startup_velocity(y, t: float, height: float, body_force: float,
                     viscosity: float, n_terms: int = 801):
    """Boussinesq start-up from rest -- :func:`series_from_rest` at ``omega=0``."""
    return series_from_rest(y, t, height, body_force, viscosity, 0.0, n_terms)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def verify_symbolic() -> dict:
    """Re-derive (2) from (1) with sympy: ODE residual, both BCs, ``alpha``."""
    try:
        import sympy as sp
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "reason": f"sympy unavailable: {exc}"}

    y, H, g0, nu, om = sp.symbols("y H g0 nu omega", positive=True)
    alpha = sp.sqrt(sp.I * om / nu)
    half = H / 2
    uhat = (g0 / (sp.I * om)) * (1 - sp.cosh(alpha * (y - half)) / sp.cosh(alpha * half))

    ode = sp.simplify(sp.I * om * uhat - (nu * sp.diff(uhat, y, 2) + g0))
    bc0 = sp.simplify(uhat.subs(y, 0))
    bcH = sp.simplify(uhat.subs(y, H))
    centre = sp.simplify(uhat.subs(y, half))
    quasi = sp.limit(uhat, om, 0)                             # om -> 0 limit
    steady = g0 * (H - y) * y / (2 * nu)
    return {
        "available": True,
        "alpha_squared_is_iomega_over_nu": bool(sp.simplify(alpha ** 2 - sp.I * om / nu) == 0),
        "ode_residual": str(ode),
        "ode_residual_is_zero": bool(ode == 0),
        "bc_at_y0_is_zero": bool(bc0 == 0),
        "bc_at_yH_is_zero": bool(bcH == 0),
        "centreline_amplitude_symbolic": str(centre),
        "omega_to_zero_limit_equals_steady_parabola": bool(
            sp.simplify(quasi - steady) == 0),
        "steady_parabola": "g0*(H-y)*y/(2*nu)",
    }


def verify_numeric(height: float, viscosity: float, body_force: float,
                   omega: float, t: float, n_terms: int = 801) -> dict:
    """Numeric cross-checks that need no sympy.

    * the from-rest series minus its transient must approach the periodic state
      ``Re{Uhat e^{i omega t}}`` (they are two independent derivations of it);
    * both representations must vanish at the walls;
    * the ``omega -> 0`` limit must reproduce the steady parabola.
    """
    y = np.linspace(0.0, height, 41)
    periodic = periodic_velocity(y, t, height, body_force, viscosity, omega)
    series = series_from_rest(y, t, height, body_force, viscosity, omega, n_terms)
    steady = steady_velocity(y, height, body_force, viscosity)
    steady_series = startup_velocity(y, 5.0 / (viscosity * (math.pi / height) ** 2),
                                     height, body_force, viscosity, n_terms)
    scale = max(float(np.linalg.norm(steady)), 1e-30)
    return {
        "series_vs_periodic_rel_l2": float(
            np.linalg.norm(series - periodic) / max(np.linalg.norm(periodic), 1e-30)),
        "wall_value_over_scale": float(
            max(abs(periodic[0]), abs(periodic[-1])) / scale),
        "steady_series_vs_closed_form_rel_l2": float(
            np.linalg.norm(steady_series - steady) / scale),
        "womersley_number": womersley_number(height, viscosity, omega),
        "stokes_layer_over_height": stokes_layer_thickness(viscosity, omega) / height,
    }