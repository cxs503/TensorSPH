"""Reference solution for Stokes' second problem (oscillating flat plate).

What the case is
----------------
An incompressible fluid of kinematic viscosity ``nu`` fills the half space above
a flat plate at ``y = 0`` that oscillates in its own plane,

.. math:: u(0, t) = U_0 \\cos(\\omega t), \\qquad u(y\\to\\infty) \\to 0 .

Landau & Lifshitz, *Fluid Mechanics* (2nd English ed., §24 "The motion of a
viscous fluid" — the section on the oscillations of a viscous fluid), give the
exact solution as a transverse (shear) wave that propagates into the fluid while
decaying:

.. math::

   u(y,t) = U_0 \\, e^{-y/\\delta}
            \\cos\\!\\left(\\omega t - \\frac{y}{\\delta}\\right),
   \\qquad \\delta = \\sqrt{\\frac{2\\nu}{\\omega}} .

The quantity this repository grades is the *amplitude* profile

.. math:: A(y) = U_0 e^{-y/\\delta},

i.e. the Stokes penetration depth ``delta`` and the exponential decay of the
amplitude with wall distance.  ``delta`` is the distance over which the amplitude
falls by ``1/e``; the phase lag over the same distance is one radian.  The wave
carries no pressure signature (the motion is solenoidal, ``u = (u(y,t), 0)``), so
the case isolates the **viscous operator and the moving no-slip wall** and
nothing else.

Reference provenance (no hand-copied numbers)
--------------------------------------------
The closed forms below are not transcribed from a table: :func:`verify_symbolic`
re-derives them with sympy from the governing equation and records the
residuals in the artifact — ``u_t - nu u_yy`` must be identically zero,
``u(0,t) = U_0 cos(omega t)`` must hold, ``A(delta)/A(0) = 1/e`` and
``delta = sqrt(2 nu / omega)`` must be identities.  That is the same
"re-derive instead of transcribe" discipline as
``benchmarks/poiseuille_mirror/run.py``; the reference is therefore independent
of the transcription of any table, which is what the case pool asks for.

A second, *finite-depth* solution is also provided
--------------------------------------------------
A numerical channel has to stop somewhere.  If the far wall sits at ``y = H``
and is held at rest, the exact solution of *that* problem is not the
semi-infinite one but

.. math::

   u(y,t) = \\Re\\left\\{ U_0
            \\frac{\\sinh\\!\\big(m (H-y)\\big)}{\\sinh(m H)} e^{i\\omega t}
            \\right\\}, \\qquad m = \\frac{1+i}{\\delta},

which decays to the semi-infinite form with error ``exp(-(H-y)/delta)``.  Both
are offered so a benchmark can separate *discretisation* error (numerics vs
finite-depth exact) from *domain truncation* error (finite-depth exact vs the
semi-infinite referent the case pool specifies).  :func:`finite_depth_amplitude`
returns the complex amplitude, :func:`finite_depth` the real field.

Everything here is dimensionless-free and takes SI inputs; nothing is
hard-coded to a particular benchmark's numbers.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

__all__ = [
    "penetration_depth",
    "semi_infinite",
    "semi_infinite_amplitude",
    "finite_depth_complex_amplitude",
    "finite_depth_amplitude",
    "finite_depth",
    "harmonic_fit",
    "verify_symbolic",
    "verify_numeric",
]


# --------------------------------------------------------------------------- #
# Closed forms
# --------------------------------------------------------------------------- #

def penetration_depth(viscosity: float, omega: float) -> float:
    """Stokes penetration depth ``delta = sqrt(2 nu / omega)`` (m)."""
    if viscosity <= 0.0 or omega <= 0.0:
        raise ValueError("viscosity and omega must be positive")
    return math.sqrt(2.0 * viscosity / omega)


def semi_infinite(y, t, amplitude: float, viscosity: float, omega: float):
    """Exact velocity of Stokes' second problem in a half space (Landau §24).

    ``u(y,t) = U0 exp(-y/delta) cos(omega t - y/delta)``.
    """
    delta = penetration_depth(viscosity, omega)
    y = np.asarray(y, dtype=float)
    return amplitude * np.exp(-y / delta) * np.cos(omega * t - y / delta)


def semi_infinite_amplitude(y, amplitude: float, viscosity: float,
                            omega: float):
    """Amplitude ``U0 exp(-y/delta)`` of the half-space solution."""
    delta = penetration_depth(viscosity, omega)
    return amplitude * np.exp(-np.asarray(y, dtype=float) / delta)


def finite_depth_complex_amplitude(y, height: float, amplitude: float,
                                   viscosity: float, omega: float):
    """Complex amplitude ``U0 sinh(m(H-y)) / sinh(mH)``, ``m = (1+i)/delta``.

    The time dependence is ``Re{ Uhat(y) exp(i omega t) }``.
    """
    delta = penetration_depth(viscosity, omega)
    m = (1.0 + 1.0j) / delta
    y = np.asarray(y, dtype=float)
    return amplitude * np.sinh(m * (height - y)) / np.sinh(m * height)


def finite_depth_amplitude(y, height: float, amplitude: float,
                           viscosity: float, omega: float):
    """Magnitude of :func:`finite_depth_complex_amplitude`."""
    return np.abs(finite_depth_complex_amplitude(y, height, amplitude,
                                                 viscosity, omega))


def finite_depth(y, t, height: float, amplitude: float, viscosity: float,
                 omega: float):
    """Real velocity field of the same problem in a channel of depth ``H``."""
    uhat = finite_depth_complex_amplitude(y, height, amplitude, viscosity,
                                         omega)
    return np.real(uhat * np.exp(1.0j * omega * t))


# --------------------------------------------------------------------------- #
# Estimator: harmonic amplitude and phase of a sampled time series
# --------------------------------------------------------------------------- #

def harmonic_fit(times: Iterable[float], signal: Iterable[float],
                 omega: float, with_mean: bool = True) -> dict:
    """Least-squares fit of ``A cos(omega t + phase) [+ c]`` to a signal.

    Returns ``amplitude``, ``phase`` (radians, ``u ~ A cos(omega t + phase)``),
    the fitted constant ``mean`` and the relative residual ``residual``
    (RMS of the unexplained part over the RMS of the signal).  The design matrix
    is ``[cos, sin]`` (plus ``1``), solved in the normal equations; an integer
    number of periods makes ``cos``/``sin`` orthogonal and the fit is then just
    the pair of projections.
    """
    t = np.asarray(times, dtype=float)
    u = np.asarray(signal, dtype=float)
    if t.shape != u.shape:
        raise ValueError("times and signal must have the same shape")
    if t.size < 3:
        raise ValueError("need at least three samples")
    cols = [np.cos(omega * t), np.sin(omega * t)]
    if with_mean:
        cols.append(np.ones_like(t))
    design = np.stack(cols, axis=1)
    coeff, *_ = np.linalg.lstsq(design, u, rcond=None)
    a, b = coeff[0], coeff[1]
    fitted = design @ coeff
    rms = float(np.sqrt(np.mean(u * u)))
    resid = float(np.sqrt(np.mean((u - fitted) ** 2)))
    return {
        "amplitude": float(math.hypot(a, b)),
        "phase": float(-math.atan2(b, a)),
        "cos_coefficient": float(a),
        "sin_coefficient": float(b),
        "mean": float(coeff[2]) if with_mean else 0.0,
        "residual": resid / rms if rms > 0.0 else float("nan"),
        "rms": rms,
        "n_samples": int(t.size),
    }


# --------------------------------------------------------------------------- #
# Provenance: re-derive the closed forms instead of transcribing them
# --------------------------------------------------------------------------- #

def verify_symbolic() -> dict:
    """Re-derive both closed forms from the governing equation with sympy.

    The half-space form must satisfy ``u_t = nu u_yy`` identically, honour
    ``u(0,t) = U0 cos(omega t)``, and produce ``A(delta)/U0 = 1/e``.  The
    finite-depth amplitude must satisfy ``Uhat'' = m^2 Uhat`` with
    ``Uhat(0) = U0``, ``Uhat(H) = 0`` and ``m = (1+i)/delta``.

    If sympy is unavailable the entry says so; :func:`verify_numeric` then
    carries the verification instead.
    """
    try:
        import sympy as sp
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "reason": f"sympy unavailable: {exc}"}

    y, t, nu, U0, om = sp.symbols("y t nu U0 omega", positive=True)
    delta = sp.sqrt(2 * nu / om)
    u = U0 * sp.exp(-y / delta) * sp.cos(om * t - y / delta)
    residual = sp.simplify(sp.diff(u, t) - nu * sp.diff(u, y, 2))
    bc_plate = sp.simplify(u.subs(y, 0) - U0 * sp.cos(om * t))
    amp = U0 * sp.exp(-y / delta)
    decay_at_delta = sp.simplify(amp.subs(y, delta) / U0)
    delta_identity = sp.simplify(delta - sp.sqrt(2 * nu / om))

    H = sp.symbols("H", positive=True)
    m = (1 + sp.I) / delta
    uhat = U0 * sp.sinh(m * (H - y)) / sp.sinh(m * H)
    ode_fd = sp.simplify(sp.diff(uhat, y, 2) - m ** 2 * uhat)
    bc_fd_0 = sp.simplify(uhat.subs(y, 0) - U0)
    bc_fd_H = sp.simplify(uhat.subs(y, H))

    return {
        "available": True,
        "source": ("Landau & Lifshitz, Fluid Mechanics, 2nd English ed., §24 "
                   "(oscillations of a viscous fluid); re-derived here, not "
                   "transcribed"),
        "half_space_profile": "u = U0 exp(-y/delta) cos(omega t - y/delta)",
        "delta": "sqrt(2 nu / omega)",
        "pde_residual_u_t_minus_nu_u_yy": str(residual),
        "pde_residual_is_zero": bool(sp.simplify(residual) == 0),
        "plate_bc_residual": str(bc_plate),
        "plate_bc_is_satisfied": bool(sp.simplify(bc_plate) == 0),
        "amplitude_at_delta_over_U0": str(decay_at_delta),
        "amplitude_at_delta_is_one_over_e": bool(
            sp.simplify(decay_at_delta - sp.exp(-1)) == 0),
        "delta_identity_is_zero": bool(delta_identity == 0),
        "finite_depth_profile": ("u = Re{U0 sinh(m(H-y))/sinh(mH) exp(i omega t)}, "
                                 "m = (1+i)/delta"),
        "finite_depth_pde_residual": str(ode_fd),
        "finite_depth_pde_residual_is_zero": bool(sp.simplify(ode_fd) == 0),
        "finite_depth_bc_y0_residual": str(bc_fd_0),
        "finite_depth_bc_y0_satisfied": bool(sp.simplify(bc_fd_0) == 0),
        "finite_depth_bc_yH_residual": str(bc_fd_H),
        "finite_depth_bc_yH_satisfied": bool(sp.simplify(bc_fd_H) == 0),
    }


def verify_numeric(viscosity: float = 0.01, omega: float = 2.0 * math.pi,
                   amplitude: float = 1.0, n: int = 4001,
                   height: float = 6.0, tol: float = 5.0e-3) -> dict:
    """Independent numeric check of the same identities (no sympy needed).

    Second-order central differences on a fine grid, plus the boundary
    conditions and the ``1/e`` decay; the numbers are recorded in the artifact
    so a reader can see the reference is checked, not asserted.
    """
    delta = penetration_depth(viscosity, omega)
    y = np.linspace(0.0, 8.0 * delta, n)
    t0 = 0.37 / omega                      # arbitrary phase, not a node
    dt = 1.0e-4 / omega
    dy = y[1] - y[0]

    u = semi_infinite(y, t0, amplitude, viscosity, omega)
    ut = (semi_infinite(y, t0 + dt, amplitude, viscosity, omega)
          - semi_infinite(y, t0 - dt, amplitude, viscosity, omega)) / (2 * dt)
    uyy = np.zeros_like(u)
    uyy[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dy ** 2
    scale = omega * amplitude
    pde_rel = float(np.max(np.abs(ut[1:-1] - viscosity * uyy[1:-1])) / scale)

    plate = float(abs(semi_infinite(np.array([0.0]), t0, amplitude, viscosity,
                                    omega)[0]
                      - amplitude * math.cos(omega * t0)) / amplitude)
    A = semi_infinite_amplitude(y, amplitude, viscosity, omega)
    decay = float(A[np.argmin(np.abs(y - delta))] / amplitude)
    decay_err = float(abs(decay - math.exp(-1.0)))

    # finite depth: the same field must approach the half space as H grows.
    # The sinh form is only the physical field *inside* 0 <= y <= H (outside it
    # is the analytic continuation, which grows), so the comparison is restricted
    # to the channel.
    H = height * delta
    inside = y <= H
    yc = y[inside]
    fd = finite_depth(yc, t0, H, amplitude, viscosity, omega)
    semi = semi_infinite(yc, t0, amplitude, viscosity, omega)
    fd_plate = float(abs(finite_depth(np.array([0.0]), t0, H, amplitude,
                                      viscosity, omega)[0]
                         - amplitude * math.cos(omega * t0)) / amplitude)
    fd_wall = float(abs(finite_depth(np.array([H]), t0, H, amplitude,
                                     viscosity, omega)[0]) / amplitude)
    trunc = float(np.max(np.abs(fd - semi)) / amplitude)

    ok = (pde_rel < tol and plate < 1e-12 and decay_err < 1e-4
          and fd_plate < 1e-12 and fd_wall < 1e-12
          and trunc < math.exp(-height + 1.0))
    return {
        "viscosity": viscosity,
        "omega": omega,
        "delta": delta,
        "grid_points": n,
        "pde_relative_residual": pde_rel,
        "pde_residual_tolerance": tol,
        "plate_bc_relative_error": plate,
        "amplitude_at_delta": decay,
        "amplitude_at_delta_target": math.exp(-1.0),
        "amplitude_at_delta_error": decay_err,
        "finite_depth_plate_bc_relative_error": fd_plate,
        "finite_depth_far_wall_relative_error": fd_wall,
        "finite_depth_vs_half_space_max_diff": trunc,
        "finite_depth_expected_truncation": math.exp(-(height - 1.0)),
        "finite_depth_comparison_domain": "0 <= y <= H (channel interior)",
        "all_checks_pass": bool(ok),
    }