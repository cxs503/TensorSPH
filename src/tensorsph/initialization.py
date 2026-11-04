"""Initial-state mass calibration for wcSPH lattices (error source C2).

The discrete SPH density sum on a regular lattice does not reproduce the
analytic density exactly.  With the lattice-constant mass ``m = rho0 * dA``
the interior sum

    rho_i = sum_j m_j W(r_ij, h)

equals ``rho0 * (1 + b)`` where ``b`` is a lattice+kernel constant that is
*not* zero: for the 2-D cubic spline at ``h = 1.3 dx`` it measures
``b ~ +5.9e-3`` on a hexagonal lattice (the Laplace / oscillating-droplet
line) and ``b ~ -2.2e-3`` on the square lattice of the impact case.  The
Tait law then assigns a spurious *initial* pressure

    p = B [(1 + b)^gamma - 1]  ~  gamma * B * b

to every interior particle -- 2.4 kPa at ``c0 = 20 m/s`` on the hexagonal
lattice, i.e. 6.7x the Young-Laplace signal of the ``R = 2 cm``,
``sigma = 7.2 N/m`` droplet used in the verification ladder.  On the square
lattice the bias is negative, so the clamp hides it as a density deficit
instead of a pressure.

:func:`calibrate_masses` removes the offset with one pass
``m -> m / (1 + b)`` where ``b`` is measured in the *core* of the cloud
(particles with complete kernel support), so the calibrated lattice sums to
``rho0`` to machine precision.  The rescale is uniform, so it preserves the
relative mass distribution (a hydrostatic profile stays hydrostatic) and the
total mass changes by exactly the factor ``1 / (1 + b)`` -- the lattice
constant mass was wrong, not the physics.

Nothing here touches the time integration; the calibration is a one-shot
initial-condition correction.
"""
from __future__ import annotations

import math

import torch

from .droplet_impact import cubic_spline_kernel
from .neighbor_search import cell_linked_pairs


def _sum_density(pos: torch.Tensor, mass: torch.Tensor, h: float,
                 pairs: torch.Tensor | None = None) -> torch.Tensor:
    """Summation density including each particle's self term."""
    n = pos.shape[0]
    if pairs is None:
        pairs = cell_linked_pairs(pos, 2.0 * h)
    zero = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    rho = mass * cubic_spline_kernel(zero, h)
    if pairs.numel() == 0:
        return rho
    i, j = pairs.unbind(dim=1)
    rij = torch.linalg.vector_norm(pos[i] - pos[j], dim=1)
    wij = cubic_spline_kernel(rij, h)
    rho = rho.index_add(0, i, mass[j] * wij)
    rho = rho.index_add(0, j, mass[i] * wij)
    return rho


def calibrate_masses(pos: torch.Tensor, mass: torch.Tensor, h: float,
                     rho0: float, core_mask: torch.Tensor,
                     pairs: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, float]:
    """Rescale ``mass`` so the core sum-density is exactly ``rho0``.

    ``core_mask`` selects the particles whose kernel support is complete
    (interior of the cloud, away from any free surface or wall).  The bias
    ``b = mean(rho_core / rho0) - 1`` is a lattice+kernel property, so the
    same factor is applied to every particle.  Returns
    ``(mass_calibrated, b)``.

    Raises if the core is empty or the measured bias is not small: a large
    ``|b|`` means the core mask is wrong (it caught a surface) or the
    lattice/kernel pair is not the one this module was tuned for, and
    silently rescaling would corrupt the run.
    """
    if pos.ndim != 2 or mass.ndim != 1 or mass.shape[0] != pos.shape[0]:
        raise ValueError("pos must be (N, D) and mass must be (N,)")
    if core_mask.shape[0] != pos.shape[0]:
        raise ValueError("core_mask must have one entry per particle")
    if not bool(core_mask.any()):
        raise ValueError("core_mask selects no particles")
    if not math.isfinite(float(h)) or h <= 0:
        raise ValueError("h must be finite and positive")
    rho = _sum_density(pos, mass, h, pairs)
    b = float(rho[core_mask].mean() / rho0) - 1.0
    if abs(b) > 0.05:
        raise ValueError(
            f"measured initialisation bias b={b:.4g} is not small: the core "
            "mask probably includes a free surface, or the lattice spacing "
            "is inconsistent with h")
    return mass / (1.0 + b), b


def rectangular_core_mask(pos: torch.Tensor, width: float, depth: float,
                          h: float) -> torch.Tensor:
    """Interior of a rectangular pool: distance to every wall > ``h``."""
    x, y = pos[:, 0], pos[:, 1]
    return (x > h) & (x < width - h) & (y > h) & (y < depth - h)


def circular_core_mask(pos: torch.Tensor, center: torch.Tensor, radius: float,
                       h: float) -> torch.Tensor:
    """Interior of a circular drop: distance to the centre < ``radius - h``."""
    r = torch.linalg.vector_norm(pos - center, dim=1)
    return r < (radius - h)


def calibrate_droplet(pos: torch.Tensor, mass: torch.Tensor, h: float,
                      rho0: float, center: torch.Tensor, radius: float,
                      pairs: torch.Tensor | None = None
                      ) -> tuple[torch.Tensor, float]:
    """Convenience wrapper: calibrate a free drop centred on ``center``."""
    # 2h margin: the particle needs its full kernel support (radius 2h)
    # inside the drop for the sum to be the lattice constant.
    core = circular_core_mask(pos, center, radius, 2.0 * h)
    return calibrate_masses(pos, mass, h, rho0, core, pairs=pairs)
