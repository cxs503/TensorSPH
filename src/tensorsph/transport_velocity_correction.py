"""Transport-velocity correction (TVC), ported from SPHinXsys.

Why this module exists
----------------------
TensorSPH's WCSPH solver transports particles with the *fluid* velocity.  On a
stretched or compressive flow the particle distribution loses its lattice
regularity and the SPH quadrature degrades -- the ``grad-1`` inconsistency
``sum_j grad_i W_ij V_j`` stops being zero, which shows up as pressure noise,
particle clumping and (at free surfaces) tensile instability.  The
transport-velocity formulation (Adami, Hu & Adams 2013, JCP 241:292-307)
advects the particles with a *regularised* velocity instead.  SPHinXsys
implements that regularisation as a **position correction** whose driving term
is exactly the ``grad-1`` inconsistency; this module ports that
implementation.

Provenance (zero hand-copying: the reference implementation was fetched from
the upstream repository, not reproduced from memory).  Files are stored
verbatim under ``docs/xval_sphinx_tvc/oss_source/`` with their sha256 in
``docs/xval_sphinx_tvc/oss_source/SOURCE_SHA256.txt``:

===========================  ========================================  ========
upstream path                 sha256 (first 12)                         note
===========================  ========================================  ========
``src/shared/particle_dynamics/fluid_dynamics/
transport_velocity_correction.hpp``                                   da3979...
  the algorithm: ``interaction()`` fills ``kernel_gradient_integral_``,
  ``update()`` moves the particle
``.../transport_velocity_correction.h``                               348ff1...
  constructor: ``correction_scaling_ = coefficient * h_ref**2``, default
  ``coefficient = 0.2`` (dual time criteria; ~1/4 of that for a single
  acoustic step), default ``NoLimiter``/``NoKernelCorrection``
``.../Dambreak.cpp``                                                  39db69...
  reference test case and time-stepping context (no TVC block in the current
  master example -- see the difference table in the report)
===========================  ========================================  ========

Sign mapping (derivation, not a quotation)
------------------------------------------
Upstream builds (``e_ij`` unit vector, ``dW_ij = dW/dr <= 0``)::

    inconsistency_i = -sum_j (C_i + C_j(i)) dW_ij V_j e_ij            (inner)
    inconsistency_i += -2 C_i sum_j dW_ij V_j^wall e_ij               (wall)

The orientation of ``dW_ij e_ij`` is fixed by the same header's pressure
gradient (``pressure_gradient.hpp``)::

    Vecd nablaW_ijV_j = dW_ij * Vol_j * e_ij;
    pressure_grad -= (p_i - p_j) * nablaW_ijV_j;      // pressure_grad = grad p

which is the standard identity ``grad p_i = sum_j V_j (p_j - p_i) grad_i W_ij``
only if ``dW_ij e_ij == grad_i W_ij``.  With our kernel convention
``cubic_spline_gradient(x_i - x_j, h) = grad_i W_ij`` (which points from ``i``
towards ``j``, because moving ``i`` towards ``j`` increases ``W``), upstream's
expression is therefore::

    inconsistency_i = -[ sum_j (C_i + C_j) grad_i W_ij V_j
                       + 2 C_i sum_j grad_i W_ij V_j^wall ]

i.e. **minus** the weighted ``grad-1`` integral, and ``pos_i += alpha h_ref**2
* inconsistency_i`` (times the limiter).  The minus sign is what makes the
correction act down the particle-concentration gradient: at a free surface the
``grad-1`` integral of a particle points *into* the fluid
(``int_V grad_i W dV = -n_out int_plane W dA`` by the gradient theorem), so the
correction moves the surface particle *outwards*, and a particle in a compressed
cluster *away* from it -- de-clumping, the opposite of a compressive drift.
The sign is not cosmetic: flipping it compresses the fluid and blows the run up
(measured, see ``docs/xval_sphinx_tvc``).

What this module does *not* do
------------------------------
* No velocity is modified: only positions move, so mass and momentum of the
  discrete system are untouched by the correction itself (the momentum effect
  enters through the subsequent density/pressure evaluation, which is the
  point of the formulation).
* The adaptive-smoothing-length rescaling ``h_ratio`` of upstream is not
  ported (TensorSPH has a single ``h``); the ratio is 1 in every run here.
* The ``TruncatedLinear`` limiter variants of upstream are exposed as a
  pluggable callable but not shipped: the upstream default is ``NoLimiter``
  for the complex types in ``transport_velocity_correction.h`` and that is
  what the cross-validation runs use.

References
    Adami, S., Hu, X.Y. & Adams, N.A. (2013), "A transport-velocity
    formulation for smoothed particle hydrodynamics", J. Comput. Phys.
    241:292-307.
    Zhang, C., Rezavand, M. & Hu, X. (2020), "Dual-criteria time stepping for
    weakly compressible smoothed particle hydrodynamics", J. Comput. Phys.
    404:109135 (the coefficient default of 0.2 belongs to this scheme).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

from .droplet_impact import cell_linked_pairs, cubic_spline_gradient

__all__ = [
    "TVCResult",
    "kernel_gradient_integral",
    "no_limiter",
    "transport_velocity_correction",
    "SPHINXSYS_DEFAULT_COEFFICIENT",
    "SPHINXSYS_SINGLE_STEP_FRACTION",
]

#: ``TransportVelocityCorrection(..., Real coefficient = 0.2)`` in the upstream
#: header; valid for the *dual* time criteria scheme.
SPHINXSYS_DEFAULT_COEFFICIENT = 0.2

#: Upstream header: "If single (acoustic) time step is used, the coefficient
#: should be decreased to about 1/4 of the default value."
SPHINXSYS_SINGLE_STEP_FRACTION = 0.25


@dataclass
class TVCResult:
    """Diagnostics of one TVC application (all measured, none assumed)."""

    gradient_integral: torch.Tensor   # (N, D) inconsistency vector, 1/length
    shift: torch.Tensor               # (N, D) applied displacement delta r
    squared_norm: torch.Tensor        # (N,)   |grad-1|^2
    limiter: torch.Tensor             # (N,)   limiter(|grad-1|^2 h_ref^2)
    alpha: float                      # coefficient actually used
    h_ref: float
    clipped: float = 0.0              # fraction of particles scaled by max_shift
    extras: dict = field(default_factory=dict)

    @property
    def max_shift(self) -> float:
        if self.shift.numel() == 0:
            return 0.0
        return float(torch.linalg.vector_norm(self.shift, dim=-1).max())

    @property
    def mean_shift(self) -> float:
        if self.shift.numel() == 0:
            return 0.0
        return float(torch.linalg.vector_norm(self.shift, dim=-1).mean())

    @property
    def max_gradient_integral(self) -> float:
        if self.gradient_integral.numel() == 0:
            return 0.0
        return float(torch.linalg.vector_norm(self.gradient_integral, dim=-1).max())


def no_limiter(squared_norm: torch.Tensor) -> torch.Tensor:
    """``NoLimiter`` of upstream: the correction is never damped."""
    return torch.ones_like(squared_norm)


def kernel_gradient_integral(
    positions: torch.Tensor,
    volume: torch.Tensor,
    h: float,
    pairs: torch.Tensor | None = None,
    wall_count: int = 0,
    wall_factor: float = 2.0,
    correction: torch.Tensor | None = None,
) -> torch.Tensor:
    """Upstream's (negated, weighted) ``grad-1`` integral for fluid particles.

    Returns ``-[sum_j (C_i + C_j) grad_i W_ij V_j + 2 C_i sum_j grad_i W_ij
    V_j^wall]``, i.e. the ``inconsistency`` vector of ``interaction()`` in
    ``transport_velocity_correction.hpp`` -- not the plain ``grad-1`` sum.

    Parameters
    ----------
    positions : (N + M, D) tensor
        Fluid particles first (``N`` of them), wall particles last (``M``).
    volume : (N + M,) tensor
        ``m / rho`` of every particle.
    h : float
        Smoothing length; the kernel is the 2-D cubic spline of
        :mod:`tensorsph.droplet_impact` (``cubic_spline_gradient``).
    pairs : (K, 2) int tensor, optional
        Unordered unique neighbour pairs ``(i, j)``, ``i < j``, into
        ``positions`` (the convention of
        :func:`tensorsph.neighbor_search.cell_linked_pairs`).  Both directions
        are accumulated internally.  Computed for the support ``2 h`` when
        omitted.
    wall_count : int
        Number of trailing wall particles.  Pairs with ``j >= N`` are the wall
        contact and carry the upstream factor ``wall_factor * C_i`` (2.0 with
        ``NoKernelCorrection``); pairs with ``j < N`` carry ``C_i + C_j``.
    correction : (N + M,) tensor, optional
        Kernel correction factors ``C``; ``None`` is upstream's
        ``NoKernelCorrection`` (all ones).
    """
    n_total = positions.shape[0]
    if volume.shape != (n_total,):
        raise ValueError("volume must have one entry per position")
    if not math.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    n_fluid = n_total - int(wall_count)
    if n_fluid <= 0:
        raise ValueError("need at least one fluid particle")
    if pairs is None:
        pairs = cell_linked_pairs(positions, 2.0 * h)
    grad = torch.zeros((n_fluid, positions.shape[1]),
                       dtype=positions.dtype, device=positions.device)
    if pairs.numel() == 0:
        return grad
    c = torch.ones(n_total, dtype=positions.dtype, device=positions.device) \
        if correction is None else correction

    # ``cell_linked_pairs`` returns *unordered* unique pairs ``i < j``; both
    # directions of every couple have to be accumulated explicitly, otherwise
    # a symmetric lattice would not cancel and the correction would push every
    # particle the same way.
    i, j = pairs.unbind(dim=1)
    gw = cubic_spline_gradient(positions[i] - positions[j], h)  # grad_i W_ij
    fluid_i = i < n_fluid
    fluid_j = j < n_fluid
    ff = fluid_i & fluid_j
    fw = fluid_i & ~fluid_j
    wf = ~fluid_i & fluid_j

    targets = [i[ff], j[ff], i[fw], j[wf]]
    values = [
        ((c[i[ff]] + c[j[ff]]) * volume[j[ff]])[:, None] * gw[ff],
        -((c[i[ff]] + c[j[ff]]) * volume[i[ff]])[:, None] * gw[ff],
        (wall_factor * c[i[fw]] * volume[j[fw]])[:, None] * gw[fw],
        -(wall_factor * c[j[wf]] * volume[i[wf]])[:, None] * gw[wf],
    ]
    target = torch.cat(targets)
    value = torch.cat(values)
    keep = target < n_fluid
    grad = grad.index_add(0, target[keep], value[keep])
    return -grad          # upstream minus sign, see the module docstring


def transport_velocity_correction(
    positions: torch.Tensor,
    volume: torch.Tensor,
    h: float,
    alpha: float | None = None,
    pairs: torch.Tensor | None = None,
    wall_count: int = 0,
    h_ref: float | None = None,
    limiter: Callable[[torch.Tensor], torch.Tensor] | None = None,
    max_shift: float | None = None,
    active: torch.Tensor | None = None,
) -> TVCResult:
    """One upstream ``TransportVelocityCorrection`` application.

    ``delta r_i = alpha * h_ref**2 * limiter(h_ref**2 |g_i|^2) * g_i`` with
    ``g_i`` the :func:`kernel_gradient_integral` of particle ``i``.

    ``alpha=None`` selects the upstream default for a *single acoustic* step
    (``0.2 / 4 = 0.05``); pass ``SPHINXSYS_DEFAULT_COEFFICIENT`` to reproduce
    the literal constructor default of the dual-criteria scheme.

    ``active`` (bool, (N,)) restricts the correction to a particle scope, the
    analogue of upstream's ``ParticleScope`` template parameter
    (``AllParticles`` / ``InnerParticles``).
    """
    if alpha is None:
        alpha = SPHINXSYS_DEFAULT_COEFFICIENT * SPHINXSYS_SINGLE_STEP_FRACTION
    if h_ref is None:
        h_ref = h
    if limiter is None:
        limiter = no_limiter
    g = kernel_gradient_integral(positions, volume, h, pairs=pairs,
                                 wall_count=wall_count)
    squared = torch.sum(g * g, dim=1)
    fac = limiter(h_ref * h_ref * squared)
    shift = (alpha * h_ref * h_ref) * fac[:, None] * g
    if active is not None:
        if active.shape != squared.shape:
            raise ValueError("active must be (N,) over the fluid particles")
        shift = shift * active[:, None].to(shift.dtype)
    clipped = 0.0
    if max_shift is not None:
        norm = torch.linalg.vector_norm(shift, dim=1)
        over = norm > max_shift
        clipped = float(over.to(norm.dtype).mean()) if norm.numel() else 0.0
        scale = torch.where(over, max_shift / norm.clamp_min(1.0e-30),
                            torch.ones_like(norm))
        shift = shift * scale[:, None]
    return TVCResult(gradient_integral=g, shift=shift, squared_norm=squared,
                     limiter=fac, alpha=float(alpha), h_ref=float(h_ref),
                     clipped=clipped,
                     extras={"max_shift": max_shift,
                             "scope_active_fraction": (
                                 1.0 if active is None else
                                 float(active.to(torch.float64).mean()))})