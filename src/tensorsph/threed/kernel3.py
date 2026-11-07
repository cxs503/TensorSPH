"""Three-dimensional cubic-spline kernel and the lattice constants it implies.

Why a separate module exists
----------------------------
``tensorsph.droplet_impact.cubic_spline_kernel`` carries the 2-D normalisation
``sigma = 10 / (7 pi h^2)`` and ``rhs`` refuses any cloud whose second axis is
not 2, so a 3-D run cannot be assembled out of the shipped pieces: the kernel,
the lattice calibration and the wall model all have dimension-dependent
constants.  This module adds the 3-D branch without touching the 2-D solver, so
every number already in ``docs/`` stays reproducible bit for bit.

Conventions
-----------
* support radius ``2 h`` (cubic spline, Monaghan 1992),
* ``q = r / h``,
* ``delta_ij = x_i - x_j``, gradients are with respect to ``x_i``.

The normalisation ``1 / (pi h^3)`` and the Laplacian below are verified
numerically against quadrature in ``tests/test_threed_kernel.py``; nothing here
is asserted on faith.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "SUPPORT_FACTOR",
    "sigma3",
    "cubic_spline_kernel",
    "cubic_spline_gradient",
    "cubic_spline_laplacian",
    "pair_kernels",
    "lattice_kernel_sum",
    "lattice_bias",
    "calibrated_particle_mass",
    "kernel_normalisation",
]

#: Kernel support is ``SUPPORT_FACTOR * h`` for the cubic spline.
SUPPORT_FACTOR = 2.0


def sigma3(h: float) -> float:
    """3-D normalisation constant ``1 / (pi h^3)``."""
    return 1.0 / (math.pi * h * h * h)


def _shape(q: torch.Tensor) -> torch.Tensor:
    """Cubic-spline profile ``f(q)`` (unnormalised)."""
    first = 1.0 - 1.5 * q.square() + 0.75 * q.pow(3)
    second = 0.25 * (2.0 - q).clamp_min(0.0).pow(3)
    return torch.where(q < 1.0, first, torch.where(q < 2.0, second, 0.0))


def _shape_prime(q: torch.Tensor) -> torch.Tensor:
    """``df/dq``."""
    return torch.where(
        q < 1.0,
        -3.0 * q + 2.25 * q.square(),
        torch.where(q < 2.0, -0.75 * (2.0 - q).square(), 0.0),
    )


def _shape_second(q: torch.Tensor) -> torch.Tensor:
    """``d2f/dq2``."""
    return torch.where(
        q < 1.0,
        -3.0 + 4.5 * q,
        torch.where(q < 2.0, 1.5 * (2.0 - q), 0.0),
    )


def _twice_prime_over_q(q: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """``2 f'(q) / q`` with its finite ``q -> 0`` limit ``-6``.

    ``f'(0) = 0``, so the naive quotient is ``0 / eps`` and the limit is lost:
    that silently turned the ``r = 0`` Laplacian from ``-9 sigma / h^2`` into
    ``-3 sigma / h^2``.  Coincident particles are rare but legal (symmetry
    planes, duplicated wall particles), so the branch is closed analytically.
    """
    return torch.where(q > 1.0e-10, 2.0 * _shape_prime(q) / q.clamp_min(eps),
                       torch.full_like(q, -6.0))


def cubic_spline_kernel(r: torch.Tensor, h: float) -> torch.Tensor:
    """``W_3D(r, h)`` with ``int W dV = 1`` over the 2h ball.

    ``r`` is a non-negative radius (or array of them).
    """
    q = r / h
    return sigma3(h) * _shape(q).clamp_min(0.0)


def cubic_spline_gradient(delta: torch.Tensor, h: float,
                          eps: float = 1.0e-12) -> torch.Tensor:
    """``grad_{x_i} W`` for ``delta = x_i - x_j``, shape ``(..., D)``."""
    r = torch.linalg.vector_norm(delta, dim=-1)
    q = r / h
    scale = sigma3(h) * _shape_prime(q) / (h * r.clamp_min(eps))
    return delta * scale.unsqueeze(-1)


def cubic_spline_laplacian(delta: torch.Tensor, h: float,
                           eps: float = 1.0e-12) -> torch.Tensor:
    """``laplacian_{x_i} W`` for ``delta = x_i - x_j``.

    In ``D`` dimensions the radial Laplacian is
    ``W'' + (D - 1) W' / r``; with ``D = 3`` the factor is 2, not the 1 the 2-D
    kernel uses.  The finite value at ``r = 0`` (``-9 sigma / h^2``) is obtained
    as the ``q -> 0`` limit of ``f'' + 2 f'/q`` and is why ``eps`` only guards
    the division, never the result.
    """
    r = torch.linalg.vector_norm(delta, dim=-1)
    q = r / h
    radial = _shape_second(q) + _twice_prime_over_q(q, eps)
    inside = q < 2.0
    return torch.where(inside, sigma3(h) * radial / (h * h),
                       torch.zeros_like(radial))


def pair_kernels(delta: torch.Tensor, h: float, eps: float = 1.0e-12):
    """Return ``(W, grad W, laplacian W)`` for a batch of separations.

    One pass over the radius, ``q`` and the polynomial branches; this is the hot
    path used by the operators in :mod:`tensorsph.threed.operators3`.
    """
    r = torch.linalg.vector_norm(delta, dim=-1)
    q = r / h
    sig = sigma3(h)
    weight = sig * _shape(q).clamp_min(0.0)
    scale = sig * _shape_prime(q) / (h * r.clamp_min(eps))
    grad = delta * scale.unsqueeze(-1)
    radial = _shape_second(q) + _twice_prime_over_q(q, eps)
    lap = torch.where(q < 2.0, sig * radial / (h * h), torch.zeros_like(radial))
    return weight, grad, lap


def lattice_kernel_sum(dx: float, h: float, half_width: int | None = None) -> float:
    """``C = dx^3 * sum_lattice W(|n| dx, h)`` over a simple cubic lattice.

    ``C`` is the discrete zeroth moment of the kernel: a particle with
    ``V = dx^3`` sitting in an infinite lattice sees ``rho / rho0 = C`` instead
    of 1.  It is a pure lattice sum in units of ``h`` and therefore independent
    of ``dx`` (pinned by a test), which is what makes the mass calibration below
    a constant rather than a fitted correction.
    """
    if half_width is None:
        half_width = int(math.ceil(SUPPORT_FACTOR * h / dx)) + 1
    n = torch.arange(-half_width, half_width + 1, dtype=torch.float64) * dx
    grid = torch.cartesian_prod(n, n, n)
    r = torch.linalg.vector_norm(grid, dim=1)
    return float(cubic_spline_kernel(r, h).sum() * dx ** 3)


def lattice_bias(dx: float, h: float, half_width: int | None = None) -> float:
    """``C - 1``: the fractional density offset of a full-support particle."""
    return lattice_kernel_sum(dx, h, half_width) - 1.0


def calibrated_particle_mass(dx: float, h: float, rho0: float,
                             half_width: int | None = None) -> float:
    """``m = rho0 dx^3 / C`` so that a full-support particle has ``rho = rho0``.

    Left uncorrected, the bias ``b = C - 1`` is *uniform* and the Tait law turns
    it into a spurious background pressure ``B[(1 + b)^gamma - 1]`` that has
    nothing to do with the case being solved (measured for 3-D in
    ``benchmarks/threed_kernel/report.json``).
    """
    return rho0 * dx ** 3 / lattice_kernel_sum(dx, h, half_width)


def kernel_normalisation(h: float, n_theta: int = 400, n_r: int = 400,
                         device: torch.device | None = None):
    """Numerical check of ``int W dV = 1`` over the 2h ball.

    Returns ``(integral, radial_profile)``: spherical quadrature
    ``4 pi int_0^{2h} W r^2 dr`` on a uniform radial grid (trapezoidal, which is
    spectrally accurate for a C1 kernel), plus the sampled profile so a caller
    can integrate it differently.
    """
    device = device or torch.device("cpu")
    r = torch.linspace(0.0, SUPPORT_FACTOR * h, n_r, dtype=torch.float64, device=device)
    w = cubic_spline_kernel(r, h)
    dr = float(r[1] - r[0])
    integral = float((4.0 * math.pi * w * r.square()).sum() * dr)
    return integral, (r, w)
