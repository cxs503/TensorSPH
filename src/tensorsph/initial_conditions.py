"""Initial conditioning for summation-density SPH lattices.

Why this module exists
----------------------
A case starts from a lattice whose particle mass is ``m = rho0 * V0``
(``V0 = dx^2`` in 2-D). The summation density
``rho_i = sum_j m_j W_ij`` then reproduces ``rho0`` only when the *discrete
kernel sum* ``C = sum_j W_ij V0`` equals one for a fully supported particle.
``C`` depends on nothing but the lattice x smoothing-length geometry, so the
initial density carries a constant offset ``b = C - 1`` that is independent of
the particle spacing:

    lattice      h / dx     b = rho_init / rho0 - 1
    hex          1.3        +0.588 %
    hex          1.5        +0.172 %
    square       1.3        -0.005 %

Through the Tait law ``p = B[(rho/rho0)^gamma - 1]``, ``B = rho0 c0^2 / gamma``,
a ``+0.588 %`` density offset at ``c0 = 20 m/s`` becomes a **2395 Pa** spurious
pressure - 6.7 x the ``sigma / R = 360 Pa`` Young-Laplace signal of the
verification droplet (``backlog_update_m2971.md`` item C2 / P1). Because ``b``
does not shrink with ``dx`` it is not a discretization error that refinement
removes; it is a bias that has to be calibrated away at initialisation.

``calibrate_lattice_mass`` does that by measuring ``b`` on the fully supported
particles and rescaling ``m -> m / (1 + b)``. The rescaling is *exact* in one
pass: :func:`tensorsph.droplet_impact.density_from_neighbor_pairs` is linear in
the masses, so scaling every ``m`` by ``alpha`` scales every ``rho`` by
``alpha``. The cost is exact mass conservation against the analytic
``rho0 * Area``: the total mass moves by ``-b / (1 + b)`` (0.58 % for the hex
lattice), which is the documented trade-off, see
``docs/verification_initial_pressure_offset.md``.

Measurement, acceptance criterion and resulting numbers live in
``benchmarks/init_conditioning/``; the pinned facts are asserted by
``tests/test_init_conditioning.py``.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "supported_mask",
    "measure_density_bias",
    "calibrate_lattice_mass",
    "spurious_pressure",
    "TAIT_ACCEPTANCE_LIMIT",
]

#: Acceptance criterion from ``backlog_update_m2971.md`` (P1): the initial
#: spurious pressure must satisfy ``max(p) / (rho0 c0^2) < 1e-4``.
TAIT_ACCEPTANCE_LIMIT = 1.0e-4


def _density(pos, mass, h, pairs):
    """Summation density, imported lazily to avoid a circular import."""
    from .droplet_impact import density_from_neighbor_pairs

    return density_from_neighbor_pairs(pos, mass, h, pairs)


def supported_mask(pos: torch.Tensor, h: float, pairs: torch.Tensor | None = None,
                   tolerance: int = 0) -> torch.Tensor:
    """Select the particles whose kernel neighbourhood is saturated.

    A particle is "fully supported" when its neighbour count is within
    ``tolerance`` of the maximum count in the configuration. With the default
    ``tolerance = 0`` this is exactly the interior of a lattice; particles next
    to a free surface or a wall have fewer neighbours and are excluded, because
    their kernel sum is legitimately below one (:func:`measure_density_bias`
    must not average that in).

    ``tolerance > 0`` widens the set to particles that are one or two
    neighbours short. Those carry a *different* kernel sum, so a calibration
    fitted on the mixed set leaves the strictly interior particles slightly
    over-dense (measured: +7e-4 in ``rho/rho0`` at both spacings). The default
    is therefore 0; the wider set is only a fallback for configurations without
    a saturated interior.
    """
    from .neighbor_search import cell_linked_pairs

    if pos.ndim != 2 or pos.shape[0] == 0:
        raise ValueError("pos must be a non-empty (N, D) tensor")
    if not math.isfinite(float(h)) or h <= 0:
        raise ValueError("h must be finite and positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if pairs is None:
        pairs = cell_linked_pairs(pos, 2.0 * h)
    count = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    if pairs.numel():
        i, j = pairs.unbind(dim=1)
        count = count.index_add(0, i, torch.ones_like(i))
        count = count.index_add(0, j, torch.ones_like(j))
    return count >= count.max() - tolerance


def measure_density_bias(pos: torch.Tensor, mass: torch.Tensor, h: float,
                         rho0: float, mask: torch.Tensor | None = None,
                         pairs: torch.Tensor | None = None) -> dict:
    """Measure the lattice density offset ``b = mean(rho/rho0) - 1``.

    Only the fully supported particles take part (see :func:`supported_mask`).
    Returns a dict with ``bias``, ``bias_percent``, ``n_samples``,
    ``rho_ratio_min`` / ``rho_ratio_max`` over the mask.
    """
    if rho0 <= 0:
        raise ValueError("rho0 must be positive")
    if mask is None:
        mask = supported_mask(pos, h, pairs)
    if int(mask.sum()) == 0:
        raise ValueError("no fully supported particle to measure the bias on")
    rho = _density(pos, mass, h, pairs)
    ratio = (rho[mask] / rho0).double()
    bias = float(ratio.mean() - 1.0)
    return {
        "bias": bias,
        "bias_percent": 100.0 * bias,
        "n_samples": int(mask.sum()),
        "rho_ratio_min": float(ratio.min()),
        "rho_ratio_max": float(ratio.max()),
    }


def calibrate_lattice_mass(pos: torch.Tensor, mass: torch.Tensor, h: float,
                           rho0: float, mask: torch.Tensor | None = None,
                           pairs: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, dict]:
    """Rescale the masses so that the supported particles carry ``rho0``.

    Returns ``(mass_calibrated, info)``. One pass is exact, because the
    summation density is linear in the masses. ``info`` carries the measured
    ``bias``, the applied ``scale`` and the resulting total-mass change
    ``mass_change`` (relative), which is the documented trade-off.
    """
    stats = measure_density_bias(pos, mass, h, rho0, mask=mask, pairs=pairs)
    scale = 1.0 / (1.0 + stats["bias"])
    mass_calibrated = mass * scale
    after = measure_density_bias(pos, mass_calibrated, h, rho0, mask=mask, pairs=pairs)
    info = {
        "bias": stats["bias"],
        "bias_percent": stats["bias_percent"],
        "bias_after": after["bias"],
        "scale": scale,
        "mass_change": float((mass_calibrated.sum() - mass.sum()) / mass.sum()),
        "n_samples": stats["n_samples"],
    }
    return mass_calibrated, info


def spurious_pressure(pos: torch.Tensor, mass: torch.Tensor, h: float,
                      rho0: float, sound_speed: float, gamma: float = 7.0,
                      mask: torch.Tensor | None = None,
                      pairs: torch.Tensor | None = None) -> dict:
    """Initial spurious pressure implied by the summation density.

    Uses the same Tait law as the solver (``p >= 0``, ``p =
    rho0 c0^2/gamma [(rho/rho0)^gamma - 1]``), so ``p_max`` is the pressure the
    case would see on its first step. ``p_max_over_rho0c0sq`` is the quantity
    the acceptance criterion bounds.
    """
    from .droplet_impact import Config, tait_pressure

    if sound_speed <= 0:
        raise ValueError("sound_speed must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    rho = _density(pos, mass, h, pairs)
    # The solver's own equation of state, clamped at p >= 0; a density deficit
    # (free surface, or a square lattice at b < 0) therefore reads as p = 0.
    p = tait_pressure(rho, Config(rho0=rho0, sound_speed=sound_speed, gamma=gamma))
    scale = rho0 * sound_speed ** 2
    p_max = float(p.max())
    p_abs = float(p.abs().max())
    return {
        "p_max": p_max,
        "p_abs_max": p_abs,
        "p_max_over_rho0c0sq": p_abs / scale,
        "rho_max_ratio": float(rho.max() / rho0),
        "acceptance_limit": TAIT_ACCEPTANCE_LIMIT,
        "passes_acceptance": p_abs / scale < TAIT_ACCEPTANCE_LIMIT,
    }