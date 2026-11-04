"""Free-surface pressure closure ("shell pressure") for WCSPH.

Why this module exists
----------------------
Summation density under-counts neighbours inside ``2h`` of a free surface, so
``rho < rho0`` there and the Tait law clamped at ``p >= 0``
(:func:`tensorsph.droplet_impact.tait_pressure`) puts every particle in that
shell at exactly zero pressure.  The discrete pressure gradient then sees the
interior value ``p_core`` fall to zero across a single particle spacing, which
is not the continuum field the momentum equation was discretised against.

Adami, Hu & Adams (2012, J. Comput. Phys. 231, 7057), Sec. 3.3, remove the
defect by extrapolating the pressure onto the deficient particles from their
*complete-kernel* neighbours::

    p_i = max( 0,  sum_j p_j W_ij / sum_j W_ij ),   i free surface, j not free

This is the same interpolation the solver already applies to static wall
particles (:func:`tensorsph.droplet_impact.rhs`, Adami eq. 11, body-force term
included); here it is applied to the *fluid* free-surface particles, which
previously received no correction at all.

What is deliberately **not** implemented
----------------------------------------
The kernel-weighted average carries no body-force term: for a static drop in
vacuum ``g = 0``, and for a hydrostatic free surface the pressure there is zero
by definition, so extrapolating it would be wrong.  The closure therefore only
helps configurations where the free surface carries a genuine pressure offset —
capillary pressure being the case of interest here.

The function is pure and pair-based so it can be unit tested without a
simulation; ``rhs`` calls it when ``Config.free_surface_pressure == "shell"``.
"""
from __future__ import annotations

import torch

__all__ = ["shell_pressure", "free_surface_mask"]


def free_surface_mask(rho: torch.Tensor, rho0: float,
                      threshold: float = 0.99) -> torch.Tensor:
    """Particles whose summation density is deficient by more than ``threshold``.

    ``threshold=0.99`` is the density analogue of Adami's colour function test
    ``c_i < 0.99``: a complete kernel sums to ``rho0`` to within the lattice
    bias (~6e-3 measured at ``h = 1.3 dx``), so a 1 % deficit can only come from
    a truncated support.
    """
    if rho.ndim != 1:
        raise ValueError(f"rho must be 1-D, got shape {tuple(rho.shape)}")
    if threshold >= 1.0:
        raise ValueError("threshold must be < 1 (it marks density *deficit*)")
    return (rho / rho0) < threshold


def shell_pressure(rho: torch.Tensor, pressure: torch.Tensor,
                   pairs: torch.Tensor, wij: torch.Tensor, n: int, *,
                   rho0: float, threshold: float = 0.99,
                   wall_pressure: torch.Tensor | None = None,
                   ) -> tuple[torch.Tensor, dict]:
    """Replace the pressure of free-surface fluid particles by the shell average.

    Parameters
    ----------
    rho, pressure : ``(n,)`` fluid density and Tait pressure.
    pairs : ``(P, 2)`` long tensor, indices into ``fluid + wall`` (the layout
        ``rhs`` uses, so fluid indices are ``< n`` and wall indices ``>= n``).
    wij : ``(P,)`` kernel values for exactly those pairs.
    n : number of fluid particles.
    rho0 : reference density, used only for the free-surface test.
    threshold : density-deficit fraction that marks a free-surface particle.
    wall_pressure : ``(nw,)`` optional extrapolated wall pressure.  Wall
        particles are never free surface and are valid extrapolation sources.

    Returns
    -------
    ``(pressure_out, stats)``.  ``pressure_out`` equals ``pressure`` outside
    the free-surface set, and is clamped to ``>= 0`` inside it.  ``stats``
    records how many particles moved, the mean lift, and how many free-surface
    particles had no complete-kernel neighbour (they keep their clamp value).
    """
    for name, tensor in (("rho", rho), ("pressure", pressure)):
        if tensor.ndim != 1 or tensor.shape[0] != n:
            raise ValueError(f"{name} must be 1-D with {n} entries")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be (P, 2)")
    if pairs.dtype != torch.long:
        raise ValueError("pairs must be a long tensor")
    if wij.ndim != 1 or wij.shape[0] != pairs.shape[0]:
        raise ValueError("wij must be 1-D with one value per pair")

    nw = 0 if wall_pressure is None else int(wall_pressure.shape[0])
    if nw:
        if wall_pressure.ndim != 1:
            raise ValueError("wall_pressure must be 1-D")
        p_all = torch.cat((pressure, wall_pressure))
    else:
        p_all = pressure

    free_fluid = free_surface_mask(rho, rho0, threshold)
    empty = {"n_free": int(free_fluid.sum()),
             "n_updated": 0, "n_starved": 0,
             "mean_lift_Pa": 0.0, "max_lift_Pa": 0.0}
    if not bool(free_fluid.any()):
        return pressure, empty

    # Wall particles are never free surface, so they are always valid sources.
    free_all = torch.cat((
        free_fluid,
        torch.zeros(nw, dtype=torch.bool, device=rho.device)))

    i, j = pairs.unbind(dim=1)
    # A contribution exists only on pairs that straddle the free-surface set:
    # free-surface particles are excluded as *sources* (their pressure is the
    # quantity being reconstructed) and as *joint targets* (two deficient
    # neighbours carry no information).
    crossing = torch.logical_xor(free_all[i], free_all[j])
    i_is_free = free_all[i]
    target = torch.where(i_is_free, i, j)   # the free-surface partner
    source = torch.where(i_is_free, j, i)   # its complete-kernel neighbour
    weight = torch.where(crossing, wij, torch.zeros_like(wij))

    numerator = torch.zeros(n + nw, dtype=rho.dtype, device=rho.device)
    denominator = torch.zeros(n + nw, dtype=rho.dtype, device=rho.device)
    numerator = numerator.index_add(0, target, p_all[source] * weight)
    denominator = denominator.index_add(0, target, weight)

    usable = free_all & (denominator > 0.0)
    p_shell = numerator / denominator.clamp_min(torch.finfo(rho.dtype).tiny)
    pressure_out = torch.where(usable[:n], p_shell[:n].clamp_min(0.0), pressure)

    updated = usable[:n]
    n_updated = int(updated.sum())
    lift = pressure_out - pressure
    stats = {
        "n_free": int(free_fluid.sum()),
        "n_updated": n_updated,
        "n_starved": int(free_fluid.sum()) - n_updated,
        "mean_lift_Pa": float(lift[updated].mean()) if n_updated else 0.0,
        "max_lift_Pa": float(lift[updated].max()) if n_updated else 0.0,
    }
    return pressure_out, stats