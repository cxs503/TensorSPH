"""Initial-condition consistency: the discrete zeroth moment of the kernel.

A WCSPH summation density is ``rho_i = sum_j m_j W_ij``.  On a *uniform*
lattice with mass ``m_j = rho0 * V`` every interior particle shares the same
value ``rho_i = rho0 * C``, where

    C = sum_j V * W(|r_i - r_j|)   (the discrete zeroth moment, ``C -> 1`` as
                                    the lattice is refined)

``C`` is **not** a free parameter to be tuned per case: it is a fixed property
of the lattice geometry together with ``h / dx``.  For the 2-D cubic-spline
kernel used here, ``|C - 1|`` is ``5.3e-5`` on a square lattice at
``h = 1.3 dx`` but ``5.9e-3`` on a hexagonal lattice at the same ratio -- a
factor 111 apart for the same kernel and the same smoothing length.

Because the Tait equation of state converts that bias straight into pressure,

    p = (rho0 c0^2 / gamma) * ((rho / rho0)^gamma - 1),

a ``0.59 %`` density bias becomes ``~2.4 kPa`` of *spurious* pressure at
``c0 = 20 m/s`` -- the same order as the dynamic pressure ``rho U^2`` of a
moderate impact, or 6.7x the ``sigma / R`` signal a surface-tension
verification is trying to measure.  Rescaling the particle mass by ``1 / C``
removes it exactly (``C`` is uniform in the interior by construction), at the
price of a ``0.59 %`` mass error on a hex lattice -- which is why the rescaling
is opt-in here and never applied silently by :func:`tensorsph.make_particles`.

Scope and limits:

* The correction is defined on *interior* particles (further than ``2 h`` from
  the domain boundary).  Free-surface particles are kernel-deficient by
  definition (``C_i < 1``); rescaling their mass would be wrong, so
  :func:`calibrate_mass` only reads the interior and applies one global factor.
* It is an initial-condition fix.  It neither changes the SPH operators nor
  makes the scheme conservative; it removes a constant offset that would
  otherwise be confused with physics.
* Non-uniform lattices have no single ``C``; :func:`kernel_sum_factor` still
  reports the per-particle field so the spread can be inspected.
"""

from __future__ import annotations

import math

import torch

from .droplet_impact import cell_linked_pairs, cubic_spline_kernel

__all__ = [
    "kernel_sum_factor",
    "interior_mask",
    "radial_interior_mask",
    "calibrate_mass",
    "tait_pressure",
    "spurious_pressure",
]


def kernel_sum_factor(pos: torch.Tensor, volume: torch.Tensor, h: float,
                      pairs: torch.Tensor | None = None) -> torch.Tensor:
    """Per-particle discrete zeroth moment ``C_i = sum_j V_j W_ij`` (self included).

    ``volume`` is the volume carried by each particle (``mass / rho0`` for a
    single-phase case).  On a uniform lattice away from the boundary the result
    is a constant; :func:`interior_mask` tells you where to read it.
    """
    if pos.ndim != 2 or volume.shape != (pos.shape[0],):
        raise ValueError("pos must be (N, D) and volume must be (N,)")
    if pos.shape[0] == 0:
        raise ValueError("pos must not be empty")
    if not (math.isfinite(float(h)) and h > 0.0):
        raise ValueError("h must be finite and positive")
    n = pos.shape[0]
    zero = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    total = volume * cubic_spline_kernel(zero, h)
    if pairs is None:
        pairs = cell_linked_pairs(pos, 2.0 * h)
    if pairs.numel():
        i, j = pairs.unbind(dim=1)
        rij = torch.linalg.vector_norm(pos[i] - pos[j], dim=1)
        wij = cubic_spline_kernel(rij, h)
        total = total.index_add(0, i, volume[j] * wij)
        total = total.index_add(0, j, volume[i] * wij)
    return total


def interior_mask(pos: torch.Tensor, h: float, margin: float | None = None) -> torch.Tensor:
    """Boolean mask of particles further than ``2 h`` from every axis-aligned face."""
    if pos.ndim != 2:
        raise ValueError("pos must be (N, D)")
    margin = 2.0 * h if margin is None else float(margin)
    if margin <= 0.0:
        raise ValueError("margin must be positive")
    lo = pos.amin(dim=0) + margin
    hi = pos.amax(dim=0) - margin
    return ((pos > lo) & (pos < hi)).all(dim=1)


def radial_interior_mask(pos: torch.Tensor, center, radius: float, h: float,
                         margin: float | None = None) -> torch.Tensor:
    """Interior mask for a *disc*: further than ``2 h`` from the circular free surface.

    :func:`interior_mask` bounds an axis-aligned box, which is right for a pool
    or a tank but wrong for a disc: the diagonal corners of that box lie inside
    the box and outside the circle, i.e. on the free surface.  Using the box
    mask on the ``laplace_plateau`` hex disc leaves the calibration factor
    contaminated by depleted free-surface particles (measured: ``b`` only drops
    from ``+5.88e-3`` to ``+4.8e-3`` instead of to machine zero).  For a disc the
    distance to the surface is radial, so use this mask instead.
    """
    if pos.ndim != 2:
        raise ValueError("pos must be (N, D)")
    if not (math.isfinite(float(radius)) and radius > 0.0):
        raise ValueError("radius must be finite and positive")
    margin = 2.0 * h if margin is None else float(margin)
    c = torch.as_tensor(center, dtype=pos.dtype, device=pos.device).reshape(1, -1)
    return torch.linalg.vector_norm(pos - c, dim=1) < (float(radius) - margin)


def calibrate_mass(pos: torch.Tensor, mass: torch.Tensor, h: float, rho0: float,
                   margin: float | None = None,
                   interior: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """Rescale mass by ``1 / C_interior`` so the interior starts at exactly ``rho0``.

    ``interior`` overrides the default axis-aligned mask; pass
    :func:`radial_interior_mask` for a disc.  Reading ``C`` from a mask that
    still contains free-surface particles makes the correction wrong by
    construction, so the mask is the physically important argument here.

    Returns ``(mass_calibrated, diagnostics)``.  ``diagnostics["kernel_sum"]`` is
    the mean ``C`` measured *before* the rescaling, so the size of the correction
    is auditable; ``diagnostics["applied"]`` is ``False`` when no interior
    particle exists (the mass is then returned unchanged).
    """
    if pos.shape[0] != mass.shape[0]:
        raise ValueError("pos and mass must agree on N")
    volume = mass / rho0
    c = kernel_sum_factor(pos, volume, h)
    mask = interior_mask(pos, h, margin) if interior is None else interior.to(torch.bool)
    if mask.shape != (pos.shape[0],):
        raise ValueError("interior mask must have shape (N,)")
    if not bool(mask.any()):
        return mass.clone(), {"kernel_sum": float("nan"), "applied": False,
                              "interior_count": 0}
    c_mean = float(c[mask].mean())
    if not (math.isfinite(c_mean) and c_mean > 0.0):
        raise ValueError("interior kernel sum is not positive and finite")
    out = mass / c_mean
    return out, {
        "kernel_sum": c_mean,
        "density_bias": c_mean - 1.0,
        "interior_count": int(mask.sum()),
        "interior_spread": float(c[mask].std()),
        "applied": True,
    }


def tait_pressure(rho: torch.Tensor, rho0: float, c0: float, gamma: float) -> torch.Tensor:
    """Tait equation of state used by ``rhs``, clamped at zero on the tension side."""
    return ((rho0 * c0 * c0 / gamma) * ((rho / rho0) ** gamma - 1.0)).clamp_min(0.0)


def spurious_pressure(density_bias: float, rho0: float, c0: float,
                      gamma: float) -> float:
    """Pressure a *uniform* density bias ``b`` generates at ``t = 0`` (Tait, unclamped).

    Negative values mean the bias is on the tension side, where the solver
    clamps the pressure to zero -- so a negative bias is harmless at rest,
    whereas a positive one is a live source term.
    """
    return float((rho0 * c0 * c0 / gamma) * ((1.0 + density_bias) ** gamma - 1.0))
