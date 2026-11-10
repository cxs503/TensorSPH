"""Onset of buoyancy-driven convection: the Horton-Rogers-Lapwood eigenvalue.

Brine rejected by freezing makes the water below the ice denser than the mixed
layer above it.  Whether that drives convection at all is decided by the
porous-medium Rayleigh number against a critical value that is *exact* for the
Darcy-Benard problem:

    Ra_c = 4 pi^2 = 39.478...        (free or impermeable boundaries alike,
                                      a peculiarity of Darcy flow: with no
                                      no-slip condition, the mechanical BC does
                                      not change Ra_c)

Instead of quoting the number, :func:`critical_rayleigh` **solves** the linear
stability problem by finite differences and returns the numerical Ra_c, so the
implementation is verified against the analytic value rather than restating it.
The normal-mode system (Nield & Bejan, ch. 6; growth rate ``sigma``, wavenumber
``a``, ``D = d/dz``, perturbation temperature Theta and vertical velocity W)::

    sigma Theta = (D^2 - a^2) Theta - W
    0           = (D^2 - a^2) W - a^2 Ra Theta

Eliminating ``W`` gives ``sigma Theta = A Theta - a^2 Ra A^{-1} Theta`` with
``A = D^2 - a^2``; the marginal curve is the ``Ra`` at which the largest real
eigenvalue of ``A + a^2 Ra A^{-1}`` vanishes, and ``Ra_c = min_a Ra(a)``.
"""

from __future__ import annotations

import math

import numpy as np

from . import properties as props

__all__ = [
    "RA_CRITICAL_EXACT",
    "growth_rate",
    "marginal_rayleigh",
    "critical_rayleigh",
    "porous_rayleigh",
    "mushy_rayleigh",
]

#: exact Horton-Rogers-Lapwood value ``4 pi^2``
RA_CRITICAL_EXACT = 4.0 * math.pi ** 2


def _operator(n: int, a: float) -> np.ndarray:
    """FD matrix ``A = D^2 - a^2`` with Dirichlet (isothermal) walls, unit depth."""
    dz = 1.0 / (n + 1)
    main = np.full(n, -2.0 / dz ** 2 - a * a)
    off = np.full(n - 1, 1.0 / dz ** 2)
    return np.diag(main) + np.diag(off, 1) + np.diag(off, -1)


def growth_rate(rayleigh: float, wavenumber: float, n: int = 80) -> float:
    """Largest real growth rate ``sigma`` of the linearised Darcy-Benard system."""
    a = float(wavenumber)
    amat = _operator(n, a)
    inv = np.linalg.inv(amat)
    mat = amat - a * a * float(rayleigh) * inv
    eig = np.linalg.eigvals(mat)
    return float(np.max(eig.real))


def marginal_rayleigh(wavenumber: float, n: int = 80, tol: float = 1e-5):
    """Rayleigh number at which ``sigma`` changes sign for one wavenumber."""
    lo, hi = 1.0, 4000.0
    if growth_rate(hi, wavenumber, n) < 0.0:
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if growth_rate(mid, wavenumber, n) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * hi:
            break
    return 0.5 * (lo + hi)


def critical_rayleigh(n: int = 80, n_wavenumbers: int = 60):
    """Numerical ``Ra_c`` and its wavenumber: ``(Ra_c, a_c, curve)``.

    ``curve`` is the marginal-stability curve ``(a, Ra(a))`` that the report
    plots against the analytic ``(a^2 + pi^2)^2 / a^2``.
    """
    curve = []
    for k in range(n_wavenumbers):
        a = 0.5 + 4.0 * k / max(n_wavenumbers - 1, 1)   # a in [0.5, 4.5], a = pi inside
        ra = marginal_rayleigh(a, n)
        if not math.isnan(ra):
            curve.append((a, ra))
    best = min(curve, key=lambda item: item[1])
    return best[1], best[0], curve


def analytic_marginal(wavenumber: float) -> float:
    """``Ra(a) = (a^2 + pi^2)^2 / a^2`` -- the free-free (exact) curve."""
    a = float(wavenumber)
    return (a * a + math.pi ** 2) ** 2 / (a * a)


def porous_rayleigh(permeability_m2, delta_rho, height, diffusivity, viscosity=props.MU_BRINE):
    """Darcy-Rayleigh number ``Ra = k g drho h / (mu kappa)``."""
    return (float(permeability_m2) * props.GRAVITY * float(delta_rho) * float(height)
            / (float(viscosity) * float(diffusivity)))


def mushy_rayleigh(temperature, salinity, height, v_irr=0.03):
    """Darcy-Rayleigh number of a mushy layer driven by brine rejection.

    The density contrast is taken between the brine in the layer and the
    interstitial fluid above it, using the haline contraction coefficient
    (thermal buoyancy is neglected: the layer is nearly isothermal at ``Tf``).
    Salt diffusivity in brine ``1e-9 m^2/s`` (ionic).
    """
    t = float(temperature)
    s = float(salinity)
    v_b = props.brine_volume_fraction(t, s)
    k = props.permeability(v_b)
    sb = props.brine_salinity(t)
    delta_rho = max(props.brine_density(sb) - props.brine_density(s), 0.0)
    ra = porous_rayleigh(k, delta_rho, height, 1e-9)
    return {"rayleigh": ra, "ra_critical": RA_CRITICAL_EXACT,
            "supercritical": bool(ra > RA_CRITICAL_EXACT),
            "brine_volume_fraction": v_b, "permeability_m2": k,
            "brine_salinity_psu": sb, "delta_rho": delta_rho}


def nusselt(rayleigh, exponent=1.0):
    """Declared heat-transport closure; **1 below onset**, ``>= 1`` above.

    ``Nu = 1`` for subcritical layers (pure conduction) and
    ``Nu = (Ra/Ra_c)**exponent`` above -- used only as a *qualitative* switch in
    :mod:`tensorsph.ice_thermal.coupled`; the convective velocity there comes
    from a Deardorff scaling, not from this closure.
    """
    ra = float(rayleigh)
    if ra <= RA_CRITICAL_EXACT:
        return 1.0
    return (ra / RA_CRITICAL_EXACT) ** exponent
