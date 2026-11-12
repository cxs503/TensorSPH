"""Particle shifting for the WCSPH solver (Lind, Xu, Stansby & Rogers 2012).

Why this module exists
----------------------
TensorSPH's open defect register has one entry that no operator fix has closed:
**F1, the resting-flume noise** -- a column at rest reads ``v_late_rms =
0.0798 m/s`` against a ``0.06 m/s`` ceiling (``docs/verification_startup_quiescence.md``
§3.3/§5), i.e. 60 % of the orbital velocity of the smallest wave the flume is
asked to carry.  The ablation recorded there shows the noise is not produced by
the wall model: with ``density_diffusion = 0`` and ``viscosity = 0.01`` the same
column reads ``0.0072 m/s``.  The residual is the *lattice*: a square lattice
is not the isotropic particle distribution the SPH quadrature assumes, and the
odd-even mode of the summation density is a persistent, unphysical excitation
(``docs/verification_ledger.md``: ``threed_hydrostatic`` +43 % core pressure
error on that mode).

Lind et al. (2012) remove exactly this failure mode by letting the particle
positions relax towards a locally uniform distribution: the particle
concentration

    C_i = sum_j V_j W_ij                                      (1)

is 1 for a complete kernel support (the kernel is normalised, so the sum
approximates its integral) and *decreases* where the support is truncated or
where the local packing is sparse.  Diffusing the particles down the gradient
of ``C``

    delta r_i = -A grad C_i,   A = C_d h |v|_ref dt          (2)

is a Fickian flux from dense to sparse packing, and it leaves a perfect
lattice untouched (``grad C = 0`` there to round-off -- asserted in
``tests/test_particle_shifting.py``).  At a free surface the shift is projected
onto the surface tangent, ``delta r_i <- delta r_i - (delta r_i . n_i) n_i``,
with ``n = -grad C / |grad C|``: shifting there would otherwise let particles
leave the free surface and erode it (Lind et al. 2012 §2.2).

What this module deliberately does *not* do
-------------------------------------------
* It moves positions only.  Mass, density and velocity are untouched, so the
  scheme conserves mass exactly; it does **not** conserve momentum exactly
  (the shift is not derived from a pair force), which is the standard trade-off
  of the method and is measured rather than assumed in the demo benchmark.
* It is not a substitute for particle *splitting/merging* and does not change
  the resolution.
* The shift stays below ``limit * h`` (default ``0.25 h``) by construction, and
  the caller re-applies its own boundary clamp afterwards (this module never
  writes to wall particles: they are not in ``pos``).

References
    Lind, S.J., Xu, R., Stansby, P.K. & Rogers, B.D. (2012), "Incompressible-
    compressible flows with a new boundary condition", Comput. Fluids 49:121.
    Skillen, A. et al. (2013), "Accuracy and conservation properties of a new
    particle shifting scheme", J. Comput. Phys. 249:161.
    Adami, S., Ihmsen, M. & Adams, N.A. (2012), J. Comput. Phys. 231:7057
    (spurious-current admissibility gate ``v_max/v_ref < 0.1``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .droplet_impact import (cell_linked_pairs, cubic_spline_gradient,
                             cubic_spline_kernel)

__all__ = [
    "ShiftingState", "concentration", "concentration_gradient",
    "estimate_reference_concentration", "shift_positions",
]


@dataclass
class ShiftingState:
    """Diagnostics of one shift application (all measured, none assumed)."""

    concentration: torch.Tensor          # (N,) C_i, eq. (1)
    grad_c: torch.Tensor                 # (N, D) unprojected grad C
    normal: torch.Tensor                 # (N, D) outward normal -grad C/|grad C|
    shift: torch.Tensor                  # (N, D) applied displacement delta r
    free_surface: torch.Tensor           # (N,) bool, C_i < free_surface * c_ref
    c_reference: float                   # c_ref used for the detector
    magnitude: float                     # coefficient A = C_d h |v|_ref dt
    clipped: float = 0.0                 # fraction of particles scaled by the cap
    extras: dict = field(default_factory=dict)

    @property
    def max_shift(self) -> float:
        return float(torch.linalg.vector_norm(self.shift, dim=-1).max()) \
            if self.shift.numel() else 0.0


def concentration(pos: torch.Tensor, volume: torch.Tensor, h: float,
                  pairs: torch.Tensor | None = None) -> torch.Tensor:
    """``C_i = sum_j V_j W_ij`` including the self term (eq. 1).

    ``V_j`` is the volume of particle ``j`` (``m_j / rho_j``), so on a full
    lattice ``C`` tends to ``int W dV = 1`` in whatever dimension the kernel is
    normalised for.  Only the *ratio* of ``C`` to a reference value is used
    below, so a kernel with a different normalisation is handled by
    :func:`estimate_reference_concentration` rather than by assuming 1.
    """
    if pos.ndim != 2 or volume.shape != pos.shape[:1]:
        raise ValueError("pos must be (N, D) and volume must be (N,)")
    if not math.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    n = pos.shape[0]
    c = volume * cubic_spline_kernel(torch.zeros(n, dtype=pos.dtype,
                                                 device=pos.device), h)
    if pairs is None:
        pairs = cell_linked_pairs(pos, 2.0 * h)
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
    what makes a perfect lattice a fixed point of the shift (invariant 1 of the
    test suite).
    """
    n = pos.shape[0]
    grad = torch.zeros_like(pos)
    if pairs is None:
        pairs = cell_linked_pairs(pos, 2.0 * h)
    if pairs.numel() == 0:
        return grad
    i, j = pairs.unbind(dim=1)
    gw = cubic_spline_gradient(pos[i] - pos[j], h)
    # The volume and the sign belong to the *partner*: pair ``(i, j)`` carries
    # ``V_j (C_j - C_i) grad_i W`` to ``i`` and ``V_i (C_j - C_i) grad_i W`` to
    # ``j`` (because ``grad_j W_ji = -grad_i W_ij`` and the field difference
    # flips too).  Writing ``-contrib`` here instead of ``+contrib`` leaves the
    # lattice fixed point intact but mirrors ``grad C`` on every particle that
    # is not fully symmetric, which biases the diffusion direction.
    diff = (c[j] - c[i])[:, None] * gw
    grad = grad.index_add(0, i, volume[j, None] * diff)
    grad = grad.index_add(0, j, volume[i, None] * diff)
    return grad


def estimate_reference_concentration(c: torch.Tensor,
                                     percentile: float = 95.0) -> float:
    """Concentration of a fully supported particle, taken from the cloud itself.

    Using the ``percentile`` of ``C`` instead of the analytic ``int W dV`` makes
    the free-surface detector independent of the kernel's dimensional
    normalisation (the shipped cubic spline is normalised per dimension, but a
    caller may pass another kernel) and of a globally compressed packing.
    """
    if c.numel() == 0:
        return 0.0
    q = min(max(percentile, 0.0), 100.0)
    return float(torch.quantile(c.detach().double(), q / 100.0))


def shift_positions(pos: torch.Tensor, volume: torch.Tensor, h: float, *,
                    dt: float, v_ref: float, strength: float = 1.0,
                    limit: float = 0.25, free_surface: float = 0.95,
                    pairs: torch.Tensor | None = None,
                    c_reference: float | None = None,
                    return_state: bool = False):
    """One shifting pass.  Returns the shifted positions (and the state).

    Parameters
    ----------
    dt : time step the shift is paired with, seconds.
    v_ref : reference speed in the coefficient ``A = strength h v_ref dt``
        (eq. 2).  Lind et al. use the instantaneous maximum speed; a resting
        column has ``v_max -> 0``, so a caller that wants the lattice to relax
        at rest must pass a floor (the demo uses ``0.05 c0``).
    strength : ``C_d`` in eq. (2); 0 disables the shift.
    limit : cap on ``|delta r|`` in units of ``h`` (0.25 h by default, the
        value used by Lind et al.).
    free_surface : the detector threshold, ``C_i < free_surface * c_ref``.
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
    fs_mask = c < free_surface * c_ref if c_ref > 0 else torch.zeros(
        n, dtype=torch.bool, device=pos.device)

    magnitude = float(strength) * h * float(v_ref) * float(dt)
    shift = -magnitude * grad
    if fs_mask.any():
        # Tangential projection at the free surface (Lind et al. 2012 §2.2).
        n_hat = normal[fs_mask]
        proj = (shift[fs_mask] * n_hat).sum(dim=-1, keepdim=True) * n_hat
        shift = shift.index_copy(0, torch.nonzero(fs_mask, as_tuple=False).squeeze(-1),
                                 shift[fs_mask] - proj)

    cap = float(limit) * h
    mag = torch.linalg.vector_norm(shift, dim=-1)
    clipped = mag > cap
    if bool(clipped.any()):
        scale = torch.where(clipped, cap / mag.clamp_min(1.0e-300),
                            torch.ones_like(mag))
        shift = shift * scale[:, None]
    out = pos + shift
    if not return_state:
        return out
    return out, ShiftingState(
        concentration=c, grad_c=grad, normal=normal, shift=shift,
        free_surface=fs_mask, c_reference=c_ref, magnitude=magnitude,
        clipped=float(clipped.double().mean()) if n else 0.0,
    )