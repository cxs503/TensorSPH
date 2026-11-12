"""Initial-state mass calibration: the lattice kernel-sum offset ``b``.

Why this module exists
----------------------
Density *summation* does not reproduce ``rho0`` on a regular particle lattice.
For a particle whose support is complete

    rho_i = sum_j m_j W(r_ij, h) = m * S,      S = sum over the lattice of W

so with the lattice-constant mass (``m = rho0 dx^2`` in 2-D, ``rho0 dx dy`` for
hex packing) the fluid starts at

    rho_i / rho0 = 1 + b,      b = m * S / rho0 - 1.

The offset depends on ``h/dx`` *and* on the lattice geometry, and the two
lattices this repo builds sit on opposite sides of zero at the production
``h = 1.3 dx``: the square lattice gives ``b = -5.3e-5`` (measured; the Tait
clamp then hides it) while the hexagonal lattice of the Laplace benchmark gives
``b = +5.9e-3``.  Only the over-dense side is visible: the Tait law turns it
into a uniform spurious pressure

    p_spurious = B [(1+b)^gamma - 1],   B = rho0 c0^2 / gamma,

which at ``c0 = 20 m/s`` (``B = 5.71e4 Pa``) is **2394 Pa** for the hex drop --
the number ``backlog_update_m2971.md`` reports (2395 Pa), and 6.7x the
Young-Laplace signal of a 2 cm water drop.  ``backlog_update_m3211.md`` corrects
the older claim that this is universal: it is a property of the *pair*
``(h/dx, geometry)``, so a case run away from ``h/dx = 1.3`` (or on the hex
lattice) carries it and a case run at the square production default does not.
Calibration is therefore applied uniformly and costs nothing on the default.

``b`` depends only on ``h/dx`` and on the lattice geometry -- not on ``dx``,
``rho0``, ``c0``, the body shape or the resolution.  A single scalar therefore
divides it out:

    m_calibrated = m / (1 + b),

after which the sum-density of every complete-support particle is ``rho0`` to
machine precision, and the initial spurious pressure of those particles is
exactly zero (they all share the same lattice sum, so the correction is not an
average that leaves scatter behind).

The correction is a *mass* correction, so total mass changes by ``1/(1+b)``.
That is the point: ``rho0`` is the density the fluid is supposed to have, and
the lattice mass ``rho0 dx^2`` is the wrong mass to represent it.  For a pure
bulk fluid the two statements are indistinguishable in the continuum limit; for
a 0.59 % effect measured against a 360 Pa signal they are not.

What is *not* fixed by calibration
----------------------------------
The free surface: partially supported particles have ``rho < rho0`` and their
pressure is clamped to zero.  That deficit is a different defect and is
measured elsewhere (``benchmarks/laplace_pressure_floor``).

Usage
-----
    >>> from tensorsph.initial_density import lattice_bias, calibrate_mass
    >>> b = lattice_bias(dx=0.002, h=1.3 * 0.002, kind="square")
    >>> mass = calibrate_mass(mass, dx=0.002, h=1.3 * 0.002)      # doctest: +SKIP
"""
from __future__ import annotations

import math

import torch

from .droplet_impact import cubic_spline_kernel

__all__ = [
    "SUPPORTED_LATTICES",
    "lattice_offsets",
    "lattice_kernel_sum",
    "lattice_bias",
    "calibration_factor",
    "calibrate_mass",
    "measure_complete_support_bias",
    "spurious_pressure",
    "spurious_pressure_ratio",
]

SUPPORTED_LATTICES = ("square", "hex")


def _n_half(dx: float, h: float) -> int:
    """Half-width (in cells) that encloses the kernel support plus a margin."""
    return int(math.ceil(2.0 * h / dx)) + 2


def lattice_offsets(dx: float, h: float, kind: str = "square",
                    n_half: int | None = None,
                    dtype: torch.dtype = torch.float64,
                    device: torch.device | str = "cpu") -> torch.Tensor:
    """Offsets ``r_ij`` of one lattice particle to every other particle in range.

    The returned ``(M, 2)`` tensor contains *all* lattice sites inside the
    square ``|x|, |y| <= n_half * dx`` except the origin, so ``sum W`` over it is
    the discrete kernel sum of a complete-support particle (the kernel vanishes
    beyond ``2h``; ``n_half`` is chosen to enclose that support with margin).

    ``kind="square"``  spacing ``dx`` in both directions (the lattice
        ``make_particles`` builds).
    ``kind="hex"``     rows every ``dx sqrt(3)/2``, alternate rows offset by
        ``dx/2``, one particle per ``dx * dx sqrt(3)/2`` of area -- the packing
        ``benchmarks/laplace_law`` uses.
    """
    if kind not in SUPPORTED_LATTICES:
        raise ValueError(f"kind must be one of {SUPPORTED_LATTICES}, got {kind!r}")
    if not (math.isfinite(dx) and dx > 0.0):
        raise ValueError("dx must be finite and positive")
    if not (math.isfinite(h) and h > 0.0):
        raise ValueError("h must be finite and positive")
    n = _n_half(dx, h) if n_half is None else int(n_half)
    if n < 1:
        raise ValueError("n_half must be at least 1")

    dy = dx * math.sqrt(3.0) / 2.0 if kind == "hex" else dx
    offsets = []
    # Row index k is anchored at y = 0 (not at the bottom of the cut-out), so the
    # reference particle at the origin *is* a lattice site and the alternating
    # half-offset of hex packing has a fixed parity.
    for k in range(-n, n + 1):
        y = k * dy
        shift = 0.5 * dx if (kind == "hex" and k % 2) else 0.0
        xs = torch.arange(-n, n + 1, dtype=dtype) * dx + shift
        gx, gy = torch.meshgrid(xs, torch.tensor([y], dtype=dtype),
                                indexing="xy")
        offsets.append(torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1))
    out = torch.cat(offsets, dim=0)
    keep = torch.linalg.vector_norm(out, dim=1) > 0.0
    return out[keep].to(device)


def lattice_kernel_sum(dx: float, h: float, kind: str = "square",
                       n_half: int | None = None) -> float:
    """``S = W(0) + sum_{j != i} W(r_ij, h)`` for one complete-support particle."""
    zero = torch.zeros(1, 2, dtype=torch.float64)
    total = cubic_spline_kernel(torch.linalg.vector_norm(zero, dim=1), h)
    offs = lattice_offsets(dx, h, kind=kind, n_half=n_half)
    if offs.numel():
        total = total + cubic_spline_kernel(
            torch.linalg.vector_norm(offs, dim=1), h).sum()
    return float(total[0])


def lattice_bias(dx: float, h: float, kind: str = "square",
                 n_half: int | None = None) -> float:
    """``b`` with ``rho_i/rho0 = 1 + b`` for the lattice-constant mass.

    ``kind="square"`` uses ``m = rho0 dx^2``, ``kind="hex"`` uses
    ``m = rho0 dx dy`` with ``dy = dx sqrt(3)/2``, matching the lattices the
    cases build.  Positive ``b`` means the lattice starts over-dense.  The 3-D
    cubic lattice is *not* handled here: it needs the 3-D kernel and lives in
    :func:`tensorsph.threed.kernel3.lattice_bias`.
    """
    area = dx * dx
    if kind == "hex":
        area = dx * dx * math.sqrt(3.0) / 2.0
    return area * lattice_kernel_sum(dx, h, kind=kind, n_half=n_half) - 1.0


def calibration_factor(dx: float, h: float, kind: str = "square",
                       n_half: int | None = None) -> float:
    """``1 / (1 + b)`` -- the factor that makes the sum-density equal ``rho0``."""
    return 1.0 / (1.0 + lattice_bias(dx, h, kind=kind, n_half=n_half))


def calibrate_mass(mass: torch.Tensor, dx: float, h: float,
                   kind: str = "square") -> torch.Tensor:
    """Divide out the lattice bias: returns ``mass / (1 + b)``.

    A single scalar, so the *shape* of the mass distribution (uniform, or the
    hydrostatic profile of ``hydrostatic_density``) is preserved and the target
    density it encodes is what the summation then actually produces.
    """
    if not isinstance(mass, torch.Tensor):
        raise TypeError("mass must be a torch.Tensor")
    factor = calibration_factor(dx, h, kind=kind)
    return mass * mass.new_tensor(factor)


def measure_complete_support_bias(pos: torch.Tensor, mass: torch.Tensor,
                                  h: float, rho0: float, pairs=None,
                                  rel_tol: float = 1.0e-9) -> dict:
    """Measure ``b`` on an actual particle set, not on an idealised lattice.

    Complete-support particles all have the *same* sum-density on a regular
    lattice (the sum is translation invariant), and no other particle can reach
    it, so the maximum of ``rho`` is the complete-support value::

        core = { i : rho_i >= (1 - rel_tol) * max(rho) }

    Returns ``dict(bias, rho_core_over_rho0, count, count_fraction)``.  Raises
    when no particle has complete support (every particle then sees a
    surface/wall deficit, and ``b`` is not observable from the state).
    """
    from .droplet_impact import density_from_neighbor_pairs

    if pos.ndim != 2:
        raise ValueError("pos must be (N, D)")
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    peak = float(rho.max())
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("sum-density is not positive; check mass and h")
    core = rho >= (1.0 - rel_tol) * peak
    count = int(core.sum())
    if count == 0:  # pragma: no cover - max() always selects itself
        raise ValueError("no complete-support particle found")
    core_mean = float(rho[core].mean())
    return {
        "bias": core_mean / rho0 - 1.0,
        "rho_core_over_rho0": core_mean / rho0,
        "count": count,
        "count_fraction": count / int(rho.shape[0]),
    }


def spurious_pressure(rho: torch.Tensor, rho0: float, sound_speed: float,
                      gamma: float = 7.0) -> torch.Tensor:
    """Tait pressure of the offset ``rho/rho0 - 1`` (clamped at zero, as in rhs)."""
    bulk = rho0 * sound_speed ** 2 / gamma
    return bulk * ((rho / rho0).pow(gamma) - 1.0).clamp_min(0.0)


def spurious_pressure_ratio(rho: torch.Tensor, rho0: float,
                            sound_speed: float, gamma: float = 7.0) -> float:
    """``max(p) / (rho0 c0^2)`` -- dimensionless size of the initial offset.

    The gate used by ``backlog_update_m2971.md`` is ``< 1e-4``, i.e. the
    spurious pressure must be under 0.01 % of the EOS scale ``rho0 c0^2``.
    """
    pressure = spurious_pressure(rho, rho0, sound_speed, gamma)
    return float(pressure.max()) / (rho0 * sound_speed ** 2)
