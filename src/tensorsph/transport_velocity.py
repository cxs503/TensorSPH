"""Velocity-form particle-shifting regulariser (Lind et al. 2012, velocity form).

**Attribution (corrected 2026-11-13, XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION).**
This module is *not* the operator SPHinXsys implements.  It writes the
particle-shifting displacement of Lind et al. (2012) as a velocity
``dX/dt = v + dv`` and advects with it; the SPHinXsys
``TransportVelocityCorrection`` is a *position* update driven by the ``grad-1``
inconsistency ``sum_j grad_i W_ij V_j`` with coefficient ``alpha * h_ref**2`` and
no reference velocity at all -- that operator is
:mod:`tensorsph.transport_velocity_correction`, whose docstring records the
line-by-line provenance (verbatim upstream files with sha256 under
``docs/xval_sphinx_tvc/oss_source/``).  An earlier revision of this docstring
claimed the ``dv = -A grad C / C`` form was "in SPHinXsys"; the vendored source
refutes it (``transport_velocity_correction.hpp`` has no concentration term and
``update()`` moves ``pos_``, not ``v_``), and
``tests/test_tvc_attribution.py`` now pins the distinction.  Both modules are
kept because they are different schemes -- ``droplet_impact`` uses this one,
``dam_break`` uses the upstream port -- but they must never be quoted
interchangeably.

Why this module exists
----------------------
WCSPH on a regular lattice develops particle disorder as the simulation
progresses: the quadrature assumes an isotropic neighbour distribution, but
advection by the physical velocity alone lets particles clump and void,
degrading accuracy.  The transport-velocity formulation (Zhang, Hu & Adams
2017, *J. Comput. Phys.* 337:1-20) decouples the advection velocity from the
momentum-carrying velocity:

    dX/dt = V = v + δv,

where the correction ``δv`` drives particles toward a locally uniform
distribution.  In this module (the Lind et al. 2012 form, *not* the upstream
SPHinXsys operator -- see the attribution note above) the correction is

    δv = -A · ∇C / C,

with ``C`` the particle concentration (eq. 1 of Lind et al. 2012) and
``A = C_d · h · v_ref · dt`` the diffusion coefficient.  This is the same
mathematical form as the particle-shifting displacement of Lind et al. (2012),
but it is applied as a *velocity correction* during the advection step rather
than as a post-hoc position shift.  The practical difference is that the
transport-velocity formulation is embedded in the time integrator: the
correction is part of the ODE right-hand side, not a separate pass.

What this module provides
-------------------------
* ``concentration`` and ``concentration_gradient`` — the same operators as
  :mod:`tensorsph.particle_shifting`, re-exported for clarity.
* ``transport_velocity_correction`` — computes ``δv`` for every particle.
* ``TransportVelocityState`` — diagnostics of one correction application.

The correction is **zero on a perfect lattice** (``∇C = 0`` by symmetry),
which is the key invariant tested in :mod:`tests.test_transport_velocity`.

At a free surface the correction is projected onto the tangent plane
(``δv ← δv - (δv·n) n``) to prevent particles from leaving the fluid domain,
exactly as in Lind et al. (2012) §2.2.

References
    Zhang, C., Hu, X.Y., Adams, N.A. (2017), "A generalized transport-velocity
    formulation for smoothed particle hydrodynamics", J. Comput. Phys. 337:1-20.
    Lind, S.J., Xu, R., Stansby, P.K., Rogers, B.D. (2012), "Incompressible-
    compressible flows with a new boundary condition", Comput. Fluids 49:121.
    SPHinXsys, Xiangyu-Hu/SPHinXsys, Apache-2.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = [
    "TransportVelocityState",
    "concentration",
    "concentration_gradient",
    "estimate_reference_concentration",
    "transport_velocity_correction",
]


def _kernels():
    """Lazy import to break the circular dependency with droplet_impact."""
    from .droplet_impact import (cell_linked_pairs, cubic_spline_gradient,
                                 cubic_spline_kernel)
    return cell_linked_pairs, cubic_spline_gradient, cubic_spline_kernel


# ---------------------------------------------------------------------------
# Concentration operators (same as particle_shifting, re-exported for clarity)
# ---------------------------------------------------------------------------

def concentration(pos: torch.Tensor, volume: torch.Tensor, h: float,
                  pairs: torch.Tensor | None = None) -> torch.Tensor:
    """``C_i = sum_j V_j W_ij`` including the self term.

    On a full lattice ``C`` tends to ``int W dV = 1`` (in whatever dimension
    the kernel is normalised for).  Only the *ratio* of ``C`` to a reference
    value is used below, so a kernel with a different normalisation is handled
    by :func:`estimate_reference_concentration`.
    """
    if pos.ndim != 2 or volume.shape != pos.shape[:1]:
        raise ValueError("pos must be (N, D) and volume must be (N,)")
    if not math.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    _, _, cubic_spline_kernel = _kernels()
    n = pos.shape[0]
    c = volume * cubic_spline_kernel(torch.zeros(n, dtype=pos.dtype,
                                                 device=pos.device), h)
    if pairs is None:
        cell_linked, _, _ = _kernels()
        pairs = cell_linked(pos, 2.0 * h)
    if pairs.numel() == 0:
        return c
    i, j = pairs.unbind(dim=1)
    r = torch.linalg.vector_norm(pos[i] - pos[j], dim=1)
    w = cubic_spline_kernel(r, h)
    c = c.index_add(0, i, volume[j] * w)
    c = c.index_add(0, j, volume[i] * w)
    return c


def concentration_gradient(pos: torch.Tensor, volume: torch.Tensor,
                           c: torch.Tensor, h: float,
                           pairs: torch.Tensor | None = None) -> torch.Tensor:
    """Antisymmetric SPH gradient ``grad C_i = sum_j V_j (C_j - C_i) grad_i W_ij``.

    The antisymmetric form is identically zero for a constant field, which is
    what makes a perfect lattice a fixed point of the correction.
    """
    n = pos.shape[0]
    grad = torch.zeros_like(pos)
    if pairs is None:
        cell_linked, _, _ = _kernels()
        pairs = cell_linked(pos, 2.0 * h)
    if pairs.numel() == 0:
        return grad
    _, cubic_spline_gradient, _ = _kernels()
    i, j = pairs.unbind(dim=1)
    gw = cubic_spline_gradient(pos[i] - pos[j], h)
    diff = (c[j] - c[i])[:, None] * gw
    grad = grad.index_add(0, i, volume[j, None] * diff)
    grad = grad.index_add(0, j, volume[i, None] * diff)
    return grad


def estimate_reference_concentration(c: torch.Tensor,
                                     percentile: float = 95.0) -> float:
    """Concentration of a fully supported particle, taken from the cloud itself."""
    if c.numel() == 0:
        return 0.0
    q = min(max(percentile, 0.0), 100.0)
    return float(torch.quantile(c.detach().double(), q / 100.0))


# ---------------------------------------------------------------------------
# Transport-velocity correction
# ---------------------------------------------------------------------------

@dataclass
class TransportVelocityState:
    """Diagnostics of one transport-velocity correction application."""

    concentration: torch.Tensor          # (N,) C_i
    grad_c: torch.Tensor                 # (N, D) unprojected grad C
    normal: torch.Tensor                 # (N, D) outward normal -grad C/|grad C|
    delta_v: torch.Tensor                # (N, D) velocity correction δv
    free_surface: torch.Tensor           # (N,) bool
    c_reference: float                   # c_ref used for the detector
    coefficient: float                   # A = C_d h v_ref dt
    extras: dict = field(default_factory=dict)

    @property
    def max_correction(self) -> float:
        return float(torch.linalg.vector_norm(self.delta_v, dim=-1).max()) \
            if self.delta_v.numel() else 0.0


def transport_velocity_correction(
        pos: torch.Tensor,
        volume: torch.Tensor,
        h: float,
        *,
        dt: float,
        v_ref: float,
        strength: float = 1.0,
        free_surface_threshold: float = 0.95,
        pairs: torch.Tensor | None = None,
        c_reference: float | None = None,
        return_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, TransportVelocityState]:
    """Compute the transport-velocity correction ``δv``.

    The correction is

        δv = -A · ∇C / C,    A = strength · h · v_ref · dt,

    where ``C`` is the particle concentration.  At a free surface the
    correction is projected onto the tangent plane to prevent particles from
    leaving the fluid domain.

    Parameters
    ----------
    pos : (N, D) particle positions.
    volume : (N,) particle volumes (``m / rho``).
    h : smoothing length.
    dt : time step the correction is paired with.
    v_ref : reference speed for the coefficient ``A``.
    strength : ``C_d`` in the coefficient; 0 disables the correction.
    free_surface_threshold : detector threshold, ``C_i < threshold * c_ref``.
    pairs : optional pre-computed pair list.
    c_reference : optional reference concentration (computed from ``c`` if None).
    return_state : if True, also return a :class:`TransportVelocityState`.

    Returns
    -------
    delta_v : (N, D) velocity correction to add to the physical velocity
              for advection.
    """
    if pos.ndim != 2 or volume.shape != pos.shape[:1]:
        raise ValueError("pos must be (N, D) and volume must be (N,)")
    if dt <= 0 or not math.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    n = pos.shape[0]

    c = concentration(pos, volume, h, pairs=pairs)
    grad = concentration_gradient(pos, volume, c, h, pairs=pairs)
    norm = torch.linalg.vector_norm(grad, dim=-1)
    normal = torch.where(norm[:, None] > 1.0e-30,
                         -grad / norm.clamp_min(1.0e-30)[:, None],
                         torch.zeros_like(grad))
    c_ref = (estimate_reference_concentration(c) if c_reference is None
             else float(c_reference))
    fs_mask = c < free_surface_threshold * c_ref if c_ref > 0 else torch.zeros(
        n, dtype=torch.bool, device=pos.device)

    # δv = -A · ∇C / C  (Zhang, Hu & Adams 2017, eq. 12)
    coefficient = float(strength) * h * float(v_ref) * float(dt)
    # Avoid division by zero: where C is tiny, the correction is meaningless.
    c_safe = c.clamp_min(1.0e-30)
    delta_v = -coefficient * grad / c_safe[:, None]

    if fs_mask.any():
        # Tangential projection at the free surface (Lind et al. 2012 §2.2).
        n_hat = normal[fs_mask]
        proj = (delta_v[fs_mask] * n_hat).sum(dim=-1, keepdim=True) * n_hat
        delta_v = delta_v.index_copy(
            0, torch.nonzero(fs_mask, as_tuple=False).squeeze(-1),
            delta_v[fs_mask] - proj)

    if not return_state:
        return delta_v
    return delta_v, TransportVelocityState(
        concentration=c, grad_c=grad, normal=normal, delta_v=delta_v,
        free_surface=fs_mask, c_reference=c_ref, coefficient=coefficient,
    )
