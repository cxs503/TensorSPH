"""Independent reference solver for ``CASE-POISEUILLE-SPH`` (finite differences).

Why a second solver
-------------------
The case grades an SPH channel against the closed forms of
:mod:`tensorsph.womersley`.  A closed form and a numerical run are only
comparable if the closed form actually solves *the problem that was run* -- same
PDE, same forcing, same initial condition, same walls.  This module answers that
question with code that shares nothing with the SPH path: a 1-D Crank-Nicolson
finite-difference solution of

    u_t = nu u_yy + g(t),   u(0,t) = u(H,t) = 0,   u(y,0) = 0

on a fine grid, integrated with the same forcing history the SPH run sees.  The
difference between the FD solution and the closed form is then a property of the
*reference*, not of the SPH run, and it is reported as its own gate.

The scheme
----------
Crank-Nicolson (trapezoidal in time, central in space), second order in both,
with the implicit tridiagonal system factorised **once** and reused every step
(the matrix is time independent).  No scipy: the Thomas sweep is written out so
the reference has no dependency the grading path could accidentally share.

Only used by the benchmark; the library reference is in
:mod:`tensorsph.womersley`.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def _thomas_factor(n: int, r: float):
    """Factorise the constant tridiagonal ``(I - dt/2 A)`` for reuse.

    ``r = nu dt / (2 dy^2)``; interior unknowns ``i = 1..n-1``.  Returns the
    forward-elimination coefficients ``cp[i]`` and the pivot denominators
    ``dp[i]`` such that one solve is a forward sweep plus a back substitution.
    """
    a = -r                     # sub-diagonal
    b = 1.0 + 2.0 * r          # diagonal
    c = -r                     # super-diagonal
    m = n - 1                  # number of interior unknowns
    cp = np.empty(m - 1)
    dp = np.empty(m)
    dp[0] = b
    if m > 1:
        cp[0] = c / dp[0]
    for i in range(1, m):
        dp[i] = b - a * cp[i - 1]
        if i < m - 1:
            cp[i] = c / dp[i]
    return cp, dp


def _thomas_solve(cp, dp, rhs, sub: float):
    """Solve the factorised system for one right-hand side.

    ``sub`` is the (constant) sub-diagonal of the matrix that was factorised;
    it is what the forward sweep needs after the elimination step.
    """
    m = rhs.size
    d = np.empty(m)
    d[0] = rhs[0] / dp[0]
    for i in range(1, m):
        d[i] = (rhs[i] - sub * d[i - 1]) / dp[i]
    x = np.empty(m)
    x[-1] = d[-1]
    for i in range(m - 2, -1, -1):
        x[i] = d[i] - cp[i] * x[i + 1]
    return x


def cn_solve(forcing: Callable[[float], float], height: float, viscosity: float,
             t_end: float, sample_times: Sequence[float], nx: int = 401,
             n_steps: int = 20000) -> dict:
    """Integrate the channel from rest with Crank-Nicolson; sample the profile.

    ``forcing(t)`` is the body force per unit mass along ``x`` (SI).  The
    returned ``samples`` array has one row per requested time, ``y`` is the
    node grid including both walls.
    """
    if height <= 0 or viscosity <= 0 or t_end <= 0:
        raise ValueError("height, viscosity and t_end must be positive")
    dy = height / nx
    dt = t_end / n_steps
    r = viscosity * dt / (2.0 * dy * dy)
    cp, dp = _thomas_factor(nx + 1, r)

    y = np.linspace(0.0, height, nx + 1)
    u = np.zeros(nx + 1)                      # state: full grid, walls stay 0
    interior = slice(1, nx)

    times = sorted(float(s) for s in sample_times)
    out_t, out_u = [], []
    g_prev = float(forcing(0.0))

    for step in range(1, n_steps + 1):
        t_new = step * dt
        g_new = float(forcing(t_new))
        # explicit half: (I + dt/2 A) u^n
        ui = u[interior]
        rhs = (ui + r * (np.concatenate(([u[0]], ui[:-1]))
                         - 2.0 * ui
                         + np.concatenate((ui[1:], [u[-1]])))
               + 0.5 * dt * (g_prev + g_new))
        sol = _thomas_solve(cp, dp, rhs, -r)
        u[interior] = sol
        u[0] = 0.0
        u[-1] = 0.0
        g_prev = g_new

        while times and t_new >= times[0] - 0.5 * dt:
            out_t.append(t_new)
            out_u.append(u.copy())
            times.pop(0)

    return {
        "y": y,
        "times": np.array(out_t),
        "samples": np.array(out_u),
        "dy": dy,
        "dt": dt,
        "r": r,
        "nx": nx,
        "n_steps": n_steps,
        "scheme": "Crank-Nicolson, central differences, Thomas solve",
    }


def interpolate_to_rows(y_ref, values, y_rows):
    """Linear interpolation of a FD profile onto the SPH row centres."""
    return np.interp(np.asarray(y_rows, dtype=float),
                     np.asarray(y_ref, dtype=float),
                     np.asarray(values, dtype=float))


def convergence_check(forcing, height: float, viscosity: float, t_end: float,
                      sample_times, y_rows, refinements=(101, 201, 401, 801)):
    """Self-convergence of the reference: same problem on four grids.

    Returns the successive differences of the profile at the last sample time,
    which is what bounds how much of any measured error can be blamed on the
    reference itself.
    """
    profiles = []
    for nx in refinements:
        sol = cn_solve(forcing, height, viscosity, t_end, sample_times, nx=nx,
                       n_steps=max(2000, 50 * nx))
        profiles.append(interpolate_to_rows(sol["y"], sol["samples"][-1], y_rows))
    steps = []
    for a, b in zip(profiles[:-1], profiles[1:]):
        denom = max(float(np.max(np.abs(b))), 1e-30)
        steps.append(float(np.max(np.abs(a - b)) / denom))
    return {"grids": list(refinements), "successive_max_rel_diff": steps,
            "finest_vs_coarsest": float(
                np.max(np.abs(profiles[0] - profiles[-1]))
                / max(float(np.max(np.abs(profiles[-1]))), 1e-30))}