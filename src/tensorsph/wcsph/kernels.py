"""Smoothing kernels for the weakly-compressible SPH verification chain.

Two kernels are provided, both in ``d`` dimensions with the dimensionless
argument ``q = r / h``:

``quintic``
    Morris (1997) quintic spline, compact support ``r < 3 h`` and a continuous
    second derivative.  Exactly normalised by quadrature over the analytic
    ``q`` variable (``tests/test_wcsph_kernels.py`` re-derives the constants):

    ========  ==================  ==================================
    d         ``alpha_d``         ``sum_q alpha_q h**-d``
    ========  ==================  ==================================
    1         ``1 / 120``         ``1 / (120 h)``
    2         ``7 / (478 pi)``    ``7 / (478 pi h**2)``
    3         ``1 / (120 pi)``    ``1 / (120 pi h**3)``
    ========  ==================  ==================================

``gaussian``
    ``W = (pi h^2)^(-d/2) exp(-q^2)`` truncated at ``q = 3``.  The truncation
    costs ``exp(-9) ~ 1.2e-4`` of the unit mass in one dimension, so the kernel
    integrates to ``1 - 1.2e-4`` (reported, not hidden, by the test).

Why a second kernel at all: the compact quintic is what the production cases
use, and the Gaussian is kept as an independent quadrature-friendly kernel so
that the *kernel-independent* part of the verification chain (particle-pair
conservation, EOS response) can be re-run without touching the pair logic.

The gradient returned is ``grad_i W(x_i - x_j)``; for the central kernels here
``grad_i W = (dW/dr) * (x_i - x_j) / r``, which is antisymmetric under
``i <-> j`` -- the property the momentum-conservation test hinges on.
"""
from __future__ import annotations

import math

import torch

#: Support radius in units of ``h`` for each kernel.
SUPPORT_FACTOR = {"quintic": 3.0, "gaussian": 3.0}


def normalisation(dim: int, kind: str = "quintic") -> float:
    """``alpha_d`` such that ``W = alpha_d / h**dim * f(r/h)`` integrates to 1."""
    if kind == "quintic":
        if dim == 1:
            return 1.0 / 120.0
        if dim == 2:
            return 7.0 / (478.0 * math.pi)
        if dim == 3:
            return 1.0 / (120.0 * math.pi)
        raise ValueError(f"quintic kernel not tabulated in dim={dim}")
    if kind == "gaussian":
        if dim not in (1, 2, 3):
            raise ValueError(f"gaussian kernel not tabulated in dim={dim}")
        return math.pi ** (-0.5 * dim)
    raise ValueError(f"unknown kernel {kind!r}")


def _quintic_shape(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dimensionless quintic spline ``f(q)`` and ``f'(q)`` (piecewise)."""
    q = q.clamp_min(0.0)
    one_minus = (1.0 - q).clamp_min(0.0)
    two_minus = (2.0 - q).clamp_min(0.0)
    three_minus = (3.0 - q).clamp_min(0.0)

    f = three_minus**5
    df = -5.0 * three_minus**4
    inner2 = q < 2.0
    if bool(inner2.any()):
        f = torch.where(inner2, f - 6.0 * two_minus**5, f)
        df = torch.where(inner2, df + 30.0 * two_minus**4, df)
    inner1 = q < 1.0
    if bool(inner1.any()):
        f = torch.where(inner1, f + 15.0 * one_minus**5, f)
        df = torch.where(inner1, df - 75.0 * one_minus**4, df)
    outside = q >= 3.0
    f = torch.where(outside, torch.zeros_like(f), f)
    df = torch.where(outside, torch.zeros_like(df), df)
    return f, df


def _gaussian_shape(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    f = torch.exp(-q * q)
    df = -2.0 * q * f
    outside = q >= 3.0
    f = torch.where(outside, torch.zeros_like(f), f)
    df = torch.where(outside, torch.zeros_like(df), df)
    return f, df


def shape(q: torch.Tensor, kind: str = "quintic") -> tuple[torch.Tensor, torch.Tensor]:
    """``(f(q), f'(q))`` for the requested kernel; zero outside the support."""
    if kind == "quintic":
        return _quintic_shape(q)
    if kind == "gaussian":
        return _gaussian_shape(q)
    raise ValueError(f"unknown kernel {kind!r}")


def kernel_value(distance: torch.Tensor, h: torch.Tensor | float,
                 dim: int, kind: str = "quintic") -> torch.Tensor:
    """``W(r, h)`` for ``r`` given as a tensor (any shape)."""
    h_t = torch.as_tensor(h, dtype=distance.dtype, device=distance.device)
    q = distance / h_t
    f, _ = shape(q, kind)
    return normalisation(dim, kind) / h_t.pow(dim) * f


def kernel_pairs(delta: torch.Tensor, h: torch.Tensor | float, dim: int,
                 kind: str = "quintic",
                 eta: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel and gradient for pair separations ``delta = x_i - x_j``.

    Parameters
    ----------
    delta : ``(P, d)`` tensor of ``x_i - x_j``
    h : scalar or ``(P,)`` tensor of smoothing lengths
    eta : regularisation added under the square root of ``r`` (rarely needed;
        exact zeros are handled separately)

    Returns
    -------
    ``(w, grad_w)`` with ``w`` of shape ``(P,)`` and ``grad_w`` of ``(P, d)``;
    ``grad_w[p] = grad_{x_i} W(x_i - x_j)``.
    """
    h_t = torch.as_tensor(h, dtype=delta.dtype, device=delta.device)
    h_col = h_t.reshape(-1, 1) if h_t.dim() else h_t
    r = torch.linalg.vector_norm(delta, dim=1) if eta == 0.0 else torch.sqrt(
        delta.square().sum(dim=1) + eta * eta)
    q = r / (h_t if h_t.dim() == 0 else h_t)
    f, df = shape(q, kind)
    alpha = normalisation(dim, kind)
    w = alpha / h_t.pow(dim) * f
    # dW/dr = alpha / h**(d+1) * f'(q);  grad = dW/dr * delta / r
    dwdr = alpha / h_t.pow(dim + 1) * df
    inv_r = torch.where(r > 0.0, 1.0 / r.clamp_min(1.0e-300),
                        torch.zeros_like(r))
    grad = (dwdr * inv_r).reshape(-1, 1) * delta
    del h_col
    return w, grad


def self_contribution(h: torch.Tensor | float, dim: int,
                      kind: str = "quintic") -> torch.Tensor:
    """``W(0, h)`` -- the self term of the summation density."""
    h_t = torch.as_tensor(h)
    zero = torch.zeros_like(h_t)
    return kernel_value(zero, h_t, dim, kind)


def adaptive_h(rho: torch.Tensor, rho0: float, h0: float,
               dim: int, bounds: tuple[float, float] = (0.8, 1.25)) -> torch.Tensor:
    """Benz (1990) smoothing length ``h_i = h0 (rho0 / rho_i)**(1/d)``.

    Clipped to ``bounds * h0`` so that a single under-resolved particle cannot
    stretch the neighbour cutoff for the whole domain (the pair list is built
    with the clipped maximum).
    """
    ratio = (rho0 / rho.clamp_min(1.0e-12)).pow(1.0 / dim)
    return (h0 * ratio).clamp(bounds[0] * h0, bounds[1] * h0)


def convergence_order(errors, spacings) -> float:
    """Observed order ``p`` in ``error ~ spacing**p`` from two rungs.

    ``p = log(e_coarse / e_fine) / log(dx_coarse / dx_fine)``; returns ``inf``
    when the fine error is zero and ``nan`` when either error is zero/negative
    in a way that admits no order.
    """
    (e_c, e_f), (dx_c, dx_f) = list(errors), list(spacings)
    if e_f <= 0.0:
        return float("inf")
    if e_c <= 0.0:
        return float("nan")
    return float(math.log(e_c / e_f) / math.log(dx_c / dx_f))