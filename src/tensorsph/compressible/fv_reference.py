"""Independent finite-volume (HLLC) reference solution for the 1-D Euler equations.

Why this module exists
----------------------
Every error number of ``CASE-SPH-SOD-SHOCK-COMPRESSIBLE`` is measured against
:func:`tensorsph.wcsph.sod_shock.exact_riemann_profile`.  That profile function
is the *yardstick*: if it is wrong, all L1/L2/plateau errors are wrong while the
test suite stays green (the tests compare the yardstick with itself).  A
Godunov-type finite-volume solver is a completely independent route to the same
solution — different discretisation, different code path — so agreement between
the two is evidence that the yardstick itself is correct.

The solver is deliberately simple: first-order upwind (piecewise-constant
interface states), HLLC approximate Riemann flux (Toro 1999, ch. 10), explicit
Euler in time, CFL-limited.  It is *not* meant to be accurate to better than a
fraction of a percent at 10^3 cells; it is meant to be trustworthy: no tuning,
no smoothing, no comparison with the analytic solution inside the solver.

Reference: E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid
Dynamics*, 2nd ed., Springer (1999), ch. 10 (HLLC flux); G. A. Sod, J. Comput.
Phys. 27(1):1-31 (1978) for the initial data.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = ["run_hllc", "compare_with_exact"]


def _primitive(U: np.ndarray, gamma: float):
    rho = U[0]
    u = U[1] / rho
    p = (gamma - 1.0) * (U[2] - 0.5 * rho * u * u)
    c = np.sqrt(gamma * p / rho)
    return rho, u, p, c


def _physical_flux(rho, u, p, E):
    return np.vstack([rho * u, rho * u * u + p, u * (E + p)])


def run_hllc(n_cells: int = 2000, t_end: float = 0.2, cfl: float = 0.4,
             gamma: float = 1.4,
             rho_L: float = 1.0, p_L: float = 1.0, u_L: float = 0.0,
             rho_R: float = 0.125, p_R: float = 0.1, u_R: float = 0.0,
             x_left: float = 0.0, x_right: float = 1.0,
             x_diaphragm: float = 0.5) -> Dict[str, object]:
    """Advance Sod's Riemann problem with a first-order HLLC finite-volume scheme.

    Returns a dict with cell centres ``x``, the primitive fields ``rho/u/p`` at
    ``t_end``, the number of steps taken and the final time (which may be a
    rounding-hair below ``t_end`` because the last step is clipped).
    """
    dx = (x_right - x_left) / n_cells
    xc = x_left + (np.arange(n_cells) + 0.5) * dx
    rho = np.where(xc < x_diaphragm, rho_L, rho_R).astype(float)
    p = np.where(xc < x_diaphragm, p_L, p_R).astype(float)
    u = np.where(xc < x_diaphragm, u_L, u_R).astype(float)
    U = np.vstack([rho, rho * u, p / (gamma - 1.0) + 0.5 * rho * u * u])

    t = 0.0
    steps = 0
    while t < t_end * (1.0 - 1e-12):
        r, uu, pp, c = _primitive(U, gamma)
        dt = cfl * dx / float(np.max(np.abs(uu) + c))
        dt = min(dt, t_end - t)

        # piecewise-constant interface states (left cell i, right cell i+1)
        rL, rR = r[:-1], r[1:]
        uL, uR = uu[:-1], uu[1:]
        pL, pR = pp[:-1], pp[1:]
        EL, ER = U[2, :-1], U[2, 1:]
        cL, cR = c[:-1], c[1:]

        SL = np.minimum(uL - cL, uR - cR)
        SR = np.maximum(uL + cL, uR + cR)
        den = rL * (SL - uL) - rR * (SR - uR)
        Sstar = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / den

        FL = _physical_flux(rL, uL, pL, EL)
        FR = _physical_flux(rR, uR, pR, ER)

        def _star(r_k, u_k, p_k, E_k, S_k):
            d = np.where(np.abs(S_k - Sstar) < 1e-12, 1e-12, S_k - Sstar)
            fac = r_k * (S_k - u_k) / d
            return np.vstack([
                fac,
                fac * Sstar,
                fac * (E_k / r_k + (Sstar - u_k) * (Sstar + p_k / (r_k * (S_k - u_k)))),
            ])

        ULs = _star(rL, uL, pL, EL, SL)
        URs = _star(rR, uR, pR, ER, SR)

        F = np.where(SL >= 0.0, FL,
                     np.where(Sstar >= 0.0, FL + SL * (ULs - U[:, :-1]),
                              np.where(SR >= 0.0, FR + SR * (URs - U[:, 1:]), FR)))

        U = U.copy()
        U[:, 1:-1] = U[:, 1:-1] - dt / dx * (F[:, 1:] - F[:, :-1])
        t += dt
        steps += 1

    r, uu, pp, _ = _primitive(U, gamma)
    return {"x": xc, "rho": r, "u": uu, "p": pp, "steps": steps, "t": t,
            "n_cells": n_cells, "cfl": cfl}


def compare_with_exact(fv: Dict[str, object], t: float = None,
                       gamma: float = 1.4, **kw) -> Dict[str, float]:
    """Relative L1 difference between a ``run_hllc`` result and the analytic profile.

    ``kw`` is forwarded to :func:`tensorsph.wcsph.sod_shock.exact_riemann_profile`
    (the function under test); normalisation is by the mean of the exact field,
    the same convention as :func:`tensorsph.compressible.sod.evaluate_metrics`.
    """
    from ..wcsph.sod_shock import exact_riemann_profile

    tt = fv["t"] if t is None else t
    x = np.asarray(fv["x"])
    rho_ref, u_ref, p_ref = exact_riemann_profile(x, tt, gamma=gamma, **kw)
    out = {}
    for key, ref, denom in (("rho", rho_ref, np.mean(rho_ref)),
                            ("p", p_ref, np.mean(p_ref)),
                            ("u", u_ref, np.mean(np.abs(u_ref)))):
        got = np.asarray(fv[key])
        out[key] = float(np.mean(np.abs(got - ref)) / denom)
    return out