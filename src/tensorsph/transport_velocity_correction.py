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
``src/shared/common/common_functors.h``                               c6ccd39c...
  the limiter objects ``NoLimiter`` / ``TruncatedLinear`` (transcribed in
  :func:`no_limiter` / :func:`truncated_linear`)
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
* The adaptive-smoothing-length rescaling ``h_ratio`` of upstream **is**
  ported (``h_ratio=`` on both entry points, upstream's
  ``AdaptiveSmoothingLength`` branch): the kernel is evaluated at the local
  smoothing length ``h_local_i = h_ref / h_ratio_i`` (upstream's ratio is
  *reference over local*, ``adaptation.h``) and the displacement carries
  upstream's ``1/h_ratio_i**2`` factor from ``update()``.  TensorSPH itself
  still runs a single ``h``, so every solver-level run sets ``h_ratio = 1``;
  the branch is verified against upstream's own formulas in
  ``docs/xval_sphinx_tvc/ADAPTIVE_H_RATIO.md`` and
  ``tests/test_tvc_adaptive_h_ratio.py``.
* Two of upstream's limiter objects are shipped -- ``no_limiter``
  (``NoLimiter``) and ``truncated_linear`` (``TruncatedLinear``, slope 100) --
  because both are in use in upstream's own examples: ``NoLimiter`` is the
  limiter of ``TransportVelocityCorrectionComplex`` (and its ``Corrected`` /
  ``Adaptive`` siblings), ``TruncatedLinear`` of the three
  ``TransportVelocityLimitedCorrection*Complex`` aliases used by the corrected
  lid-driven cavity, the non-Newtonian lid-driven cavity and the throat
  example.  Upstream's ``common_functors.h`` in this revision defines exactly
  those two limiters and nothing else (there is no ``ModifiedTruncatedLinear``
  in it -- an earlier revision of this docstring said there was, from memory;
  the fetched source is the authority).  Effect of the limiter, measured:
  ``docs/xval_sphinx_tvc/LIMITER_LEG.md``.

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
    "truncated_linear",
    "transport_velocity_correction",
    "SPHINXSYS_DEFAULT_COEFFICIENT",
    "SPHINXSYS_SINGLE_STEP_FRACTION",
    "TRUNCATED_LINEAR_DEFAULT_SLOPE",
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


#: ``TruncatedLinear(Real slope = 100.0)`` -- ``common_functors.h``.
TRUNCATED_LINEAR_DEFAULT_SLOPE = 100.0


def truncated_linear(
    slope: float = TRUNCATED_LINEAR_DEFAULT_SLOPE,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Upstream ``TruncatedLinear`` limiter, as a callable factory.

    ``common_functors.h``::

        class TruncatedLinear : public Limiter
        {
            Real slope_;
          public:
            TruncatedLinear(Real slope = 100.0) : Limiter(), slope_(slope) {};
            Real operator()(Real dimensionless_measure) const
            { return SMIN(slope_ * dimensionless_measure, Real(1)); };
        };

    with ``SMIN(a, b) = (a <= b ? a : b)`` (``scalar_functions.h``).  The
    argument is ``h_ref**2 |inconsistency_i|**2`` (``|grad-1|`` squared times
    ``h_ref`` squared, dimensionless), so the limiter is

        ``min(slope * h_ref**2 |g_i|**2, 1)``

    -- it is *not* a cap on large inconsistencies: for ``h_ref |g_i| >=
    sqrt(1/slope) = 0.1`` (slope 100) it returns 1 and leaves upstream's
    uncorrected displacement, while below that threshold it *attenuates* the
    correction quadratically, i.e. it is a noise floor that keeps the
    correction from acting on barely-inconsistent (well ordered) regions.
    The free-surface branch of the operator sits far above the threshold
    (``|g| h_ref = 30/(7 pi) ~ 1.364`` on the production lattice, see
    ``docs/xval_sphinx_tvc/SURFACE_ANCHOR.md``), so a free-surface run is
    limiter-insensitive by construction.

    ``slope <= 0`` or a non-finite slope is rejected: upstream would return a
    negative or NaN damping factor there, which is outside its valid domain.
    """
    if not math.isfinite(slope) or slope <= 0.0:
        raise ValueError("TruncatedLinear slope must be finite and positive")

    def limiter(squared_norm: torch.Tensor) -> torch.Tensor:
        return torch.minimum(slope * squared_norm,
                             torch.ones_like(squared_norm))

    limiter.__name__ = f"truncated_linear({slope:g})"
    return limiter


def _as_ratio(h_ratio, n: int, like: torch.Tensor) -> torch.Tensor:
    """Broadcast ``h_ratio`` to ``n`` particles on ``like``'s dtype/device.

    Upstream stores one ratio per particle (``DiscreteVariable<Real>
    "SmoothingLengthRatio"``); a shorter vector is padded with ones, which is
    the wall-particle convention (walls keep the reference smoothing length).
    """
    if isinstance(h_ratio, torch.Tensor):
        r = h_ratio.to(dtype=like.dtype, device=like.device).reshape(-1)
        if r.numel() == n:
            return r
        if r.numel() < n:
            return torch.cat([r, torch.ones(n - r.numel(), dtype=like.dtype,
                                            device=like.device)])
        raise ValueError("h_ratio has more entries than particles")
    return torch.full((n,), float(h_ratio), dtype=like.dtype, device=like.device)


def kernel_gradient_integral(
    positions: torch.Tensor,
    volume: torch.Tensor,
    h: float,
    pairs: torch.Tensor | None = None,
    h_ratio: float | torch.Tensor | None = None,
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
        # The search radius must cover the largest support in the body: with
        # adaptive smoothing lengths a coarse particle (h_ratio < 1) has
        # h_local > h, and upstream's cell linked list uses each particle's own
        # cut-off (``SearchDepthAdaptive``: ``CutOffRadius(h_ratio_[i])``).
        # Searching at ``2 h`` would silently drop the outer shell of such a
        # neighbour list -- measured: 42 % of the displacement at
        # ``h_ratio = 0.5`` before this fix (docs/xval_sphinx_tvc/
        # ADAPTIVE_H_RATIO.md, C1).  ``h_ratio = None`` keeps ``2 h``.
        search_h = h
        if h_ratio is not None:
            ratio_t = _as_ratio(h_ratio, positions.shape[0], positions)
            search_h = h / float(ratio_t.min().clamp_min(1.0e-30))
        pairs = cell_linked_pairs(positions, 2.0 * search_h)
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
    delta = positions[i] - positions[j]
    if h_ratio is None:
        gw_i = cubic_spline_gradient(delta, h)      # grad_i W_ij at h
        gw_j = -gw_i                                # grad_j W_ij at h
    else:
        # Upstream ``Kernel::dW(h_ratio, r_ij)`` (``base_kernel.cpp``)
        # evaluates the profile at ``q = r/h_ref * h_ratio`` with the
        # derivative normalisation scaled by ``h_ratio**(dim+1)``, and
        # ``h_ratio_i = h_ref/h_local_i`` (``adaptation.h``), so the result is
        # exactly the gradient of the same kernel at ``h_local_i``.  The
        # cut-off follows the owner particle's ratio (``SearchDepthAdaptive``
        # uses ``CutOffRadius(h_ratio_[i])``), which the profile enforces by
        # returning zero for ``q >= 2``.  The two directions of a pair are
        # therefore evaluated separately: with per-particle ``h`` there is no
        # ``gw_j = -gw_i`` symmetry.
        ratio = _as_ratio(h_ratio, n_total, positions)
        h_local = h / ratio
        gw_i = cubic_spline_gradient(delta, h_local[i])
        gw_j = cubic_spline_gradient(-delta, h_local[j])
    fluid_i = i < n_fluid
    fluid_j = j < n_fluid
    ff = fluid_i & fluid_j
    fw = fluid_i & ~fluid_j
    wf = ~fluid_i & fluid_j

    targets = [i[ff], j[ff], i[fw], j[wf]]
    values = [
        ((c[i[ff]] + c[j[ff]]) * volume[j[ff]])[:, None] * gw_i[ff],
        ((c[i[ff]] + c[j[ff]]) * volume[i[ff]])[:, None] * gw_j[ff],
        (wall_factor * c[i[fw]] * volume[j[fw]])[:, None] * gw_i[fw],
        (wall_factor * c[j[wf]] * volume[i[wf]])[:, None] * gw_j[wf],
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
    h_ratio: float | torch.Tensor | None = None,
    wall_count: int = 0,
    h_ref: float | None = None,
    limiter: Callable[[torch.Tensor], torch.Tensor] | None = None,
    max_shift: float | None = None,
    active: torch.Tensor | None = None,
    scope_tol: float | None = None,
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

    ``scope_tol`` is the *first-moment* scope criterion:
    ``|g_i| h_ref <= scope_tol`` keeps a particle, everything above it is
    masked out.  Use it when the scope must mean "particles whose kernel
    support is complete": the zeroth moment (``sum_j V_j W_ij``, the particle
    concentration ``C_i``) reaches its full-support value while the kernel
    support is still truncated, because ``grad W`` is far more sensitive to a
    missing neighbour shell than ``W`` is.  Measured consequence in the
    hydrostatic column of ``tests/test_tvc_hydrostatic.py``: a ``C_i >= 0.95
    c_ref`` criterion admits particles an order of magnitude closer to the
    free surface than full support allows, and their ``|g| h`` is ~10^2 times
    the true interior value (``benchmarks/xval_sphinx_tvc/bulk_scope_criterion.py``).
    """
    if alpha is None:
        alpha = SPHINXSYS_DEFAULT_COEFFICIENT * SPHINXSYS_SINGLE_STEP_FRACTION
    if h_ref is None:
        h_ref = h
    if limiter is None:
        limiter = no_limiter
    # ``h_ratio`` is upstream's *reference over local* ratio, so the local
    # smoothing length this module's kernel must see is ``h / ratio``; when
    # ``h_ref`` differs from ``h`` the ratio is rescaled to the same base.
    g = kernel_gradient_integral(
        positions, volume, h, pairs=pairs, wall_count=wall_count,
        h_ratio=(None if h_ratio is None
                 else _as_ratio(h_ratio, positions.shape[0], positions)
                 * (h / h_ref)))
    squared = torch.sum(g * g, dim=1)
    fac = limiter(h_ref * h_ref * squared)
    shift = (alpha * h_ref * h_ref) * fac[:, None] * g
    if h_ratio is not None:
        # upstream ``update()``: ``pos += correction_scaling * limiter * g *
        # inv_h_ratio**2`` with ``inv_h_ratio = 1/h_ratio = h_local/h_ref``.
        # Combined with ``|g| ~ 1/h_local`` this makes the relative
        # displacement ``|dr|/h_local`` independent of the local refinement.
        inv_ratio = 1.0 / _as_ratio(h_ratio, g.shape[0], positions)
        shift = shift * inv_ratio.square()[:, None]
    if active is not None:
        if active.shape != squared.shape:
            raise ValueError("active must be (N,) over the fluid particles")
        shift = shift * active[:, None].to(shift.dtype)
    kept_fraction = None
    if scope_tol is not None:
        if not math.isfinite(scope_tol) or scope_tol < 0.0:
            raise ValueError("scope_tol must be finite and non-negative")
        keep = torch.linalg.vector_norm(g, dim=1) * h_ref <= scope_tol
        kept_fraction = float(keep.to(shift.dtype).mean()) if keep.numel() else 0.0
        shift = shift * keep[:, None].to(shift.dtype)
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
                             "h_ratio": None if h_ratio is None else (
                                 float(min(_as_ratio(h_ratio, g.shape[0], positions).min(),
                                           torch.tensor(float("inf")))),
                                 float(_as_ratio(h_ratio, g.shape[0], positions).max())),
                             "scope_tol": scope_tol,
                             "scope_tol_kept_fraction": kept_fraction,
                             "scope_active_fraction": (
                                 1.0 if active is None else
                                 float(active.to(torch.float64).mean()))})