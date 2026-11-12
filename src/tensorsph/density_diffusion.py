"""Diffusive density terms for the weakly compressible summation-density solver.

Why a second form exists
------------------------
``tensorsph.droplet_impact`` obtains density by kernel summation and adds a
diffusive term to keep the pressure field smooth.  The term that has been
shipped so far is the Molteni & Colagrossi form

    D rho_i / Dt = 2 delta c0 h sum_j V_j (rho_i - rho_j) (r_i - r_j).grad W_ij / r_ij^2
                 = -2 delta c0 h sum_j V_j (rho_i - rho_j) s_ij / r_ij ,
    s_ij = -dW/dr > 0 ,

which is symmetric in ``(i, j)`` and therefore conserves ``sum_i V_i rho_i``.

It has a property that matters for a *resting* body of water: for a linear
density field it vanishes only when every particle's neighbour distribution is
centrally symmetric.  Cut the support open -- a free surface, a wall -- and the
sum no longer cancels, so an exact hydrostatic density profile
``rho(y) = rho0 + (d rho/dy) y`` produces a steady diffusive flux even though
the physical state is at rest.  That flux perturbs the density (hence the Tait
pressure) and drives spurious currents: the resting-flume probe of
``benchmarks/startup_quiescence`` reads 0.0798 m/s, and switching this term off
halves it (see ``docs/verification_startup_quiescence.md`` section 5).

The corrected form subtracts the local density gradient from the pair
difference (Antuono, Colagrossi & Marrone 2010, the ``psi`` formulation; the
renormalised gradient that makes it hydrostatic-compatible is the correction of
Fourtakas et al. 2019 -- the citation identifies the shape of the term, the
evidence for the claims here is the self-contained test
``tests/test_density_diffusion_hydrostatic.py``, which needs no reference):

    psi_ij = 2 (rho_i - rho_j) e_ij / r_ij - (<grad rho>_i + <grad rho>_j)
    D rho_i / Dt = delta c0 h sum_j V_j psi_ij . grad W_ij

with ``<grad rho>`` the first-moment-renormalised gradient

    <grad rho>_a = M_a^-1 G_a ,
    G_a = sum_b V_b (rho_b - rho_a) grad_a W_ab ,
    M_a = sum_b V_b (r_b - r_a) (x) grad_a W_ab   (symmetric, -> I on a full support),

so ``psi_ij`` is **identically zero for any linear density field whose gradient
the renormalised estimator reproduces**, truncated support included.  ``M_a`` is
inverted with its eigenvalues floored at ``eigenvalue_floor * trace(M)/D``: where
the support is so incomplete that a direction is (nearly) missing, the
correction is switched off in that direction instead of being amplified.

Notes
-----
* Both forms return a *rate* (density per unit time) to be integrated over one
  step by the summation-density solver; neither is delta-SPH proper (the solver
  never integrates the continuity equation).
* Both are applied pairwise with the antisymmetric split used by the shipped
  term, so ``sum_i V_i drho_i/dt = 0`` holds exactly for both: the volume
  weighted mean density is conserved.
* ``molteni`` stays bit-for-bit what it was; ``antuono`` is opt-in through
  ``Config.density_diffusion_form``.
"""
from __future__ import annotations

import torch

FORMS = ("molteni", "antuono")


def _ends(pairs):
    """Pair indices as a tuple, accepting both ``(i, j)`` and a ``(P, 2)`` tensor."""
    if isinstance(pairs, tuple):
        return pairs
    return pairs.unbind(dim=1)


def _pair_volumes(rho: torch.Tensor, mass: torch.Tensor, pairs):
    """Volumes ``V_i = m_i / rho_i`` of the two ends of every pair."""
    i, j = _ends(pairs)
    return mass[i] / rho[i], mass[j] / rho[j]


def molteni_colagrossi_rate(rho: torch.Tensor, mass: torch.Tensor,
                            radial: torch.Tensor, distance_sq: torch.Tensor,
                            pairs: tuple, scale: float) -> torch.Tensor:
    """Shipped Molteni & Colagrossi diffusive rate (unchanged arithmetic).

    ``radial`` is ``-(r_i - r_j).grad_i W_ij = r |dW/dr| > 0`` per pair and
    ``distance_sq`` is ``|r_i - r_j|^2``.
    """
    i, j = _ends(pairs)
    v_i, v_j = _pair_volumes(rho, mass, pairs)
    coupling = (rho[i] - rho[j]) * radial / (distance_sq + 1.0e-12)
    rate = torch.zeros_like(rho)
    rate = rate.index_add(0, i, -scale * v_j * coupling)
    rate = rate.index_add(0, j, scale * v_i * coupling)
    return rate


def renormalized_density_gradient(pos: torch.Tensor, rho: torch.Tensor,
                                  mass: torch.Tensor, pairs: tuple,
                                  delta: torch.Tensor, grad_w: torch.Tensor,
                                  eigenvalue_floor: float = 0.1):
    """First-moment-renormalised density gradient ``<grad rho>``.

    ``delta = pos[i] - pos[j]`` and ``grad_w = grad_i W_ij`` are the pair arrays
    the solver already has.  Returns ``(grad_rho, completeness)`` with
    ``completeness = lambda_min / (trace(M)/D)`` in ``[0, 1]`` (1 = the support
    resolves every direction at the lattice level).  Particles with no pairs get
    a zero gradient and zero completeness.
    """
    i, j = _ends(pairs)
    n, dim = pos.shape
    v_i, v_j = _pair_volumes(rho, mass, pairs)
    diff = (rho[j] - rho[i]).unsqueeze(-1)

    grad_rho = torch.zeros((n, dim), dtype=pos.dtype, device=pos.device)
    grad_rho = grad_rho.index_add(0, i, (v_j.unsqueeze(-1) * diff) * grad_w)
    # For the mirrored pair the two sign flips cancel: (rho_i - rho_j) and
    # grad_j W_ji = -grad_i W_ij, so the j row reads the same gradient.
    grad_rho = grad_rho.index_add(0, j, (v_i.unsqueeze(-1) * diff) * grad_w)

    # M_a += -V_b delta (x) grad_w for both ends (both factors flip sign).
    outer = -(delta.unsqueeze(-1) * grad_w.unsqueeze(1))
    moment = torch.zeros((n, dim, dim), dtype=pos.dtype, device=pos.device)
    moment = moment.index_add(0, i, v_j.reshape(-1, 1, 1) * outer)
    moment = moment.index_add(0, j, v_i.reshape(-1, 1, 1) * outer)

    values, vectors = torch.linalg.eigh(moment)
    trace = values.sum(dim=-1)
    mean = (trace / dim).clamp_min(1.0e-30)
    floor = (eigenvalue_floor * mean).unsqueeze(-1)
    clamped = values.clamp_min(floor)
    projected = torch.einsum("pka,pk->pa", vectors, grad_rho)
    corrected = torch.einsum("pak,pk->pa", vectors, projected / clamped)
    completeness = ((values[:, 0] / mean).clamp(0.0, 1.0)
                    if dim > 0 else torch.zeros(n, dtype=pos.dtype,
                                                device=pos.device))
    # A particle with no neighbours has M = 0 and G = 0: report the raw zero,
    # not 0/floor scaled quantities.
    empty = trace <= 0.0
    if bool(empty.any()):
        corrected = torch.where(empty.unsqueeze(-1), torch.zeros_like(corrected),
                                corrected)
        completeness = torch.where(empty, torch.zeros_like(completeness),
                                   completeness)
    return corrected, completeness


def hydrostatic_corrected_rate(rho: torch.Tensor, mass: torch.Tensor,
                               delta: torch.Tensor, radial: torch.Tensor,
                               distance_sq: torch.Tensor, pairs: tuple,
                               scale: float, grad_rho: torch.Tensor,
                               weights: torch.Tensor | None = None) -> torch.Tensor:
    """``psi``-form diffusive rate (Antuono et al. 2010 shape, renormalised).

    With ``c_ij = (rho_i - rho_j) - 0.5 (<grad rho>_i + <grad rho>_j).(r_i - r_j)``
    the pair contribution is ``-scale V_j L_ij c_ij`` to ``i`` and
    ``+scale V_i L_ij c_ij`` to ``j``, ``L_ij = radial / r^2 > 0``.  For a linear
    density field ``c_ij = 0`` by construction, which is the whole point: a
    resting hydrostatic column carries no diffusive flux, truncated support or
    not.

    ``weights`` (optional, per pair) multiplies the whole contribution; it is the
    fluid/fluid mask the solver uses so wall density is never diffused into.
    """
    i, j = _ends(pairs)
    v_i, v_j = _pair_volumes(rho, mass, pairs)
    projected = 0.5 * ((grad_rho[i] + grad_rho[j]) * delta).sum(dim=1)
    contrast = (rho[i] - rho[j]) - projected
    coupling = contrast * radial / (distance_sq + 1.0e-12)
    if weights is not None:
        coupling = coupling * weights
    rate = torch.zeros_like(rho)
    rate = rate.index_add(0, i, -scale * v_j * coupling)
    rate = rate.index_add(0, j, scale * v_i * coupling)
    return rate


def rate_from_positions(pos: torch.Tensor, rho: torch.Tensor,
                        mass: torch.Tensor, delta: torch.Tensor,
                        grad_w: torch.Tensor, radial: torch.Tensor,
                        distance_sq: torch.Tensor, pairs: tuple, scale: float,
                        form: str = "molteni",
                        eigenvalue_floor: float = 0.1) -> torch.Tensor:
    """Same dispatcher with the positions the renormalisation needs.

    ``density_diffusion_rate`` cannot infer the particle count from the pair
    arrays alone, so the solver path (which has ``positions``) uses this one and
    the three-argument ``density_diffusion_rate`` stays a thin wrapper for the
    old public signature.
    """
    if form == "molteni":
        return molteni_colagrossi_rate(rho, mass, radial, distance_sq, pairs,
                                       scale)
    if form != "antuono":
        raise ValueError(f"unknown density_diffusion_form {form!r}; "
                         f"expected one of {FORMS}")
    grad_rho, _ = renormalized_density_gradient(pos, rho, mass, pairs, delta,
                                                grad_w,
                                                eigenvalue_floor=eigenvalue_floor)
    return hydrostatic_corrected_rate(rho, mass, delta, radial, distance_sq,
                                      pairs, scale, grad_rho)