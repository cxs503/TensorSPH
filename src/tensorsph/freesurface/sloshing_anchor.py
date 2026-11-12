"""Linear-theory reference for rectangular-tank sloshing, two ways.

The pass/fail quantity of the sloshing case is a frequency, so the reference has
to be at least as trustworthy as the measurement. This module supplies it twice,
by two independent routes, and the test suite checks that they agree:

1. **Closed form** (:func:`omega_n`). The linearised potential-flow problem in a
   rectangular tank of length ``L`` and still-water depth ``H``,

   .. math:: \\nabla^2 \\phi = 0, \\quad
             \\partial_n\\phi = 0 \\text{ on } x=0,L \\text{ and } y=0, \\quad
             \\partial_t^2\\phi + g\\,\\partial_y\\phi = 0 \\text{ on } y=H,

   separates with :math:`\\phi = \\cos(k_n x)\\cosh(k_n y)` and
   :math:`k_n = n\\pi/L`, which gives the free-surface condition
   :math:`\\omega_n^2 = g k_n \\tanh(k_n H)`. This is the classical
   result quoted for rectangular tanks in the sloshing literature
   (Abramson, *The Dynamic Behavior of Liquids in Moving Containers*,
   NASA SP-106 (1966), ch. 2; Faltinsen & Timokha, *Sloshing* (CUP, 2009), the
   natural-frequency section of the rectangular-tank chapter). It reduces to
   :math:`\\omega_n \\to k_n\\sqrt{gH}` for :math:`k_nH \\to 0` (shallow water) and
   :math:`\\omega_n \\to \\sqrt{g k_n}` for :math:`k_nH \\to \\infty` (deep water).
   The value itself is taken from
   :func:`tensorsph.freesurface.sloshing_mode.analytic_omega` -- the reference the
   existing free-surface study already gates on -- and :func:`omega_n` only adds
   the cross-checks, so the benchmark and that study cannot drift apart.

2. **Numerical eigensolve** (:func:`omega_n_fd`). The same boundary-value problem
   is discretised without ever writing ``tanh``: the wall conditions fix the
   horizontal wavenumber, and the vertical structure ``Y(y)`` is solved on a
   finite-difference grid with the free-surface condition
   :math:`Y'(H) = (\\omega^2/g)Y(H)`. The eigenvalue follows from the discrete
   solution, so this route is independent of the closed form; it converges to it
   as :math:`O(\\Delta y^2)`. :func:`omega_n_fd` runs four grids and Richardson-
   extrapolates. A typo in either route is caught rather than believed.

Why a released-mode frequency at all
------------------------------------
For a linearly forced tank the steady-state response peaks where the denominator
:math:`\\omega_n^2 - \\omega^2` vanishes, i.e. at the natural frequency, so
"resonance frequency" and "natural frequency" are the same number in the linear
limit. Measuring it from a released first mode needs one run and no forcing
model; the forced response curve is the job of :mod:`tensorsph.sloshing`.
"""
from __future__ import annotations

import math

import torch

from . import sloshing_mode

__all__ = [
    "omega_from_wavenumber",
    "omega_n",
    "omega_n_fd",
]

GRAVITY = 9.81


def omega_from_wavenumber(k: float, depth: float, gravity: float = GRAVITY) -> float:
    """Linear dispersion relation ``omega^2 = g k tanh(k H)``."""
    if k <= 0.0:
        raise ValueError("wavenumber must be positive")
    if depth <= 0.0:
        raise ValueError("depth must be positive")
    if gravity <= 0.0:
        raise ValueError("gravity must be positive")
    return math.sqrt(gravity * k * math.tanh(k * depth))


def omega_n(tank_length: float, depth: float, mode: int = 1,
            gravity: float = GRAVITY) -> float:
    """Natural (== linear resonance) angular frequency of a rectangular tank.

    ``omega_n = sqrt(g k_n tanh(k_n H))`` with ``k_n = n pi / L``.
    """
    n = int(mode)
    if n < 1:
        raise ValueError("mode number must be >= 1")
    if tank_length <= 0.0 or depth <= 0.0:
        raise ValueError("tank length and depth must be positive")
    k = n * math.pi / tank_length
    omega = omega_from_wavenumber(k, depth, gravity)
    # cross-check against the module the free-surface study already gates on
    reference = sloshing_mode.analytic_omega(tank_length, depth, n, gravity)
    if not math.isclose(omega, reference, rel_tol=1e-12, abs_tol=0.0):
        raise AssertionError(
            f"sloshing_mode.analytic_omega disagrees: {reference!r} vs {omega!r}")
    return omega


def _omega_fd_single(k: float, depth: float, gravity: float, cells: int) -> float:
    """``omega`` from one FD grid of ``Y'' = k^2 Y``, ``Y'(0)=0``, ``Y'(H)=lam Y(H)``.

    The vertical profile is fixed up to a factor, so ``Y(H) = 1`` is imposed and
    the interior rows give the profile; the free-surface condition then *defines*
    ``lam = omega^2/g``. A second-order one-sided form of the derivative is used
    (the ghost node is expanded with ``Y'' = k^2 Y``), so the error is O(dy^2).
    """
    n = int(cells)
    if n < 4:
        raise ValueError("need at least four cells")
    dy = depth / n
    k2 = k * k
    a = torch.zeros((n, n), dtype=torch.float64)   # unknowns Y_0 .. Y_{n-1}
    rhs = torch.zeros(n, dtype=torch.float64)
    inv = 1.0 / (dy * dy)
    # interior rows j = 1 .. n-1 (Y_n = 1 goes to the right-hand side)
    for j in range(1, n):
        a[j, j - 1] += inv
        a[j, j] += -2.0 * inv - k2
        if j + 1 < n:
            a[j, j + 1] += inv
        else:
            rhs[j] -= inv
    # bottom row: ghost Y_{-1} = Y_1  ->  (2 Y_1 - 2 Y_0)/dy^2 = k^2 Y_0
    a[0, 0] += -2.0 * inv - k2
    a[0, 1] += 2.0 * inv
    y = torch.linalg.solve(a, rhs)
    lam = 0.5 * dy * k2 + (1.0 - float(y[n - 1])) / dy
    if lam <= 0.0:
        raise ValueError("FD solution gave a non-positive eigenvalue")
    return math.sqrt(lam * gravity)


def omega_n_fd(tank_length: float, depth: float, mode: int = 1,
               gravity: float = GRAVITY, cells=(32, 64, 128, 256)) -> float:
    """Natural frequency from the FD eigensolve, Richardson-extrapolated.

    Independent of :func:`omega_n`: this route never evaluates ``tanh``. With
    three grids at ratio two, ``f(dy) = f0 + c dy^2 + d dy^4`` is eliminated by
    ``(64 f3 - 20 f2 + f1)/45``.
    """
    n = int(mode)
    if n < 1:
        raise ValueError("mode number must be >= 1")
    k = n * math.pi / tank_length
    values = [_omega_fd_single(k, depth, gravity, c) for c in cells]
    if len(values) >= 3:
        f1, f2, f3 = values[-3], values[-2], values[-1]
        return (64.0 * f3 - 20.0 * f2 + f1) / 45.0
    return values[-1]