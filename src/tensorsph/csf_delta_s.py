"""Analytic-profile inversion for the CSF surface delta function.

The CSF delta function has to satisfy

    int_{interface} delta_s dn = 1                                               (1)

(Brackbill, Kothe & Zemach 1992).  In a single-phase particle cloud the colour
function only lives in the fluid: it falls from ``c = 1`` in the bulk to
``c = 1/2`` on the interface, so the in-fluid gradient integral is exactly
``1/2`` and ``delta_s = 2 |grad c|``.

Taking ``2 |grad c|`` with a lattice derivative operator, however, under-reads
the delta function by ~25 % and the deficit does not shrink with resolution
(``docs/verification_surface_tension_delta_s.md``): the transition of the
mollified indicator is about one lattice layer thick, whose derivative is a
spike that the ``2h`` MLS stencil averages away (measured ratio to the analytic
gradient 0.758 / 0.760 / 0.764 at ``R/h = 5.1 / 7.7 / 10.3``).

This module replaces the derivative with the *exact* profile of the smoothed
indicator.  For a planar interface the mollified indicator and its derivative
are known in closed form up to a one-dimensional quadrature of the kernel,

    c(d)      = 1/2 + int_0^d  f'(s) ds ,
    f'(d)     = int_{-2h}^{2h} W(sqrt(d^2 + y^2)) dy ,                           (2)

with ``d`` the distance into the fluid from the interface.  Equation (2) is
monotone and tabulated once per smoothing length.  Given the kernel-sum colour
of a particle, ``c_i``, the sign of the distance follows from the side of the
interface (``c_i > 1/2`` means fluid), and inverting the table gives ``d_i`` and

    delta_s_i = 2 f'(d_i) .                                                      (3)

By construction ``int delta_s dn = 2 [c(+inf) - c(0)] = 1`` holds for the
discrete tabulation, so the estimator cannot lose normalisation.  Its only
approximation is the planar-interface assumption of (2): the exact curvature
correction is ``O(h/R)``, which the benchmark scores explicitly.

References:
    Brackbill, Kothe & Zemach (1992), J. Comput. Phys. 100:335, eq. (6)-(8).
    Morris (2000), Int. J. Numer. Methods Fluids 33:333 (delta-function
        normalisation and its role in the CSF amplitude).
"""
from __future__ import annotations

import torch

from .droplet_impact import cubic_spline_kernel

# d = 0 .. _PROFILE_SPAN*h covers the whole compact support of the cubic spline.
_PROFILE_SPAN = 2.0
_PROFILE_NODES = 512
_QUAD_NODES = 801
_CACHE: dict[tuple[float, torch.dtype], tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor]] = {}


def kernel_profile(h: float, dtype: torch.dtype = torch.float64
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tabulate ``d``, ``c(d)`` and ``dc/dd`` for a planar mollified indicator.

    Returns:
        (d, c, dc_dd): ``(nodes,)`` tensors, ``c`` monotone increasing from 1/2
        at ``d = 0`` to 1 at the edge of the kernel support.
    """
    key = (float(h), dtype)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    d = torch.linspace(0.0, _PROFILE_SPAN * h, _PROFILE_NODES, dtype=dtype,
                       device="cpu")
    y = torch.linspace(-_PROFILE_SPAN * h, _PROFILE_SPAN * h, _QUAD_NODES,
                       dtype=dtype, device="cpu")
    dy = (y[1] - y[0])
    r = torch.sqrt(d[:, None] ** 2 + y[None, :] ** 2)
    dc_dd = (cubic_spline_kernel(r, h) * dy).sum(dim=1)
    # Trapezoidal integral of dc/dd, anchored at c(0) = 1/2.
    dr = d[1] - d[0]
    steps = 0.5 * (dc_dd[1:] + dc_dd[:-1]) * dr
    c = torch.cat([torch.full((1,), 0.5, dtype=dtype, device="cpu"),
                   torch.cumsum(steps, dim=0) + 0.5])
    c = c.clamp(0.0, 1.0)
    _CACHE[key] = (d, c, dc_dd)
    return d, c, dc_dd


def profile_delta_s(color: torch.Tensor, h: float, factor: float = 2.0,
                    ) -> torch.Tensor:
    """Surface delta function from inverting the analytic kernel profile.

    Args:
        color: (N,) kernel-sum colour function, 1 in the bulk, 1/2 on the
            interface (the convention of :func:`kernel_sum_color`).
        h: smoothing length.
        factor: 2.0 keeps the ``int delta_s dn = 1`` normalisation of the
            single-phase CSF model (the void half of the indicator is
            analytically accounted for, see the module docstring).

    Returns:
        delta_s: (N,) and non-negative; zero where ``color`` has reached the
        bulk value.
    """
    d, c, dc_dd = kernel_profile(h, color.dtype)
    dev = color.device
    d, c, dc_dd = d.to(dev), c.to(dev), dc_dd.to(dev)

    # Invert c(d): c is monotone, so searchsorted gives the bracketing nodes.
    clamped = color.clamp(c[0], c[-1])
    idx = torch.searchsorted(c, clamped).clamp(1, c.numel() - 1)
    lo = idx - 1
    c_lo, c_hi = c[lo], c[idx]
    w = (clamped - c_lo) / (c_hi - c_lo).clamp_min(1e-30)
    d_i = d[lo] + w * (d[idx] - d[lo])
    slope = dc_dd[lo] + w * (dc_dd[idx] - dc_dd[lo])
    # Colour below 1/2 puts the particle outside the interface plane (a lattice
    # staircase artefact); it is clamped to the interface, where |grad c| peaks.
    out = torch.where(color < c[0], dc_dd[0].expand_as(color), slope)
    return (factor * out).clamp_min(0.0)