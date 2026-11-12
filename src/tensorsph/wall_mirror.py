"""Mirror (ghost) particle boundary condition for planar walls.

Why this module exists
----------------------
``rhs`` in :mod:`tensorsph.droplet_impact` implements a *blend* no-slip
condition: viscosity acts, with weight ``cfg.wall_friction``, between fluid
particles and **static** wall particles that sit one spacing outside the
boundary.  That drags the fluid towards zero at ``y = -dx/2`` rather than at
``y = 0``, so the effective no-slip plane lands ~0.9 dx *inside* the flow and
the fluid slips at the wall (measured in
``docs/verification_poiseuille_channel.md``: slip 31 % of the centre-line
speed at ``H/dx = 16``, wall shear 50 % of ``rho g H / 2``).

The classical remedy is a mirror image of the near-wall fluid across the
boundary plane: the ghost particle carries the *negated* velocity, so the
antisymmetric pair has zero mean velocity on the plane itself and the discrete
viscous operator sees ``u = 0`` **on the boundary**, at second order, without
dragging the fluid over a full spacing.  This is the standard SPH wall
treatment (mirror/ghost particles; see e.g. Colagrossi & Landrini 2003, and
Adami et al. 2012 whose extrapolated wall *pressure* ``rhs`` already uses).

Convention
----------
``side = +1``: fluid lives above the plane, ``y > plane_y`` (bottom wall).
``side = -1``: fluid lives below the plane, ``y < plane_y`` (top wall).

``slip = "no_slip"``: ghost velocity ``-v`` (tangential and normal reversed).
``slip = "free_slip"``: ghost velocity ``(vx, -vy)`` for a ``y``-normal plane,
i.e. only the normal component is reversed -- the mirror-image free-slip
control, which leaves the tangential shear at zero.

The ghosts are rebuilt from the current fluid state on every step, so they
follow the flow; ``rhs`` treats them as moving wall particles through its
``wall_vel`` argument (which enters the wall/fluid viscosity term).
"""

from __future__ import annotations

from typing import Sequence

import torch

__all__ = ["mirror_plane", "plane_walls"]


def mirror_plane(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor,
                 plane_y: float, support: float, side: int = 1,
                 slip: str = "no_slip"):
    """Mirror the near-plane fluid across ``y = plane_y``.

    Only fluid particles within ``support`` of the plane are mirrored: a ghost
    further than that is beyond the kernel support of every fluid particle and
    would only cost work.

    Returns ``(ghost_pos, ghost_vel, ghost_mass)``.
    """
    if slip not in ("no_slip", "free_slip"):
        raise ValueError(f"unknown slip type {slip!r}")
    if side not in (1, -1):
        raise ValueError("side must be +1 (fluid above) or -1 (fluid below)")
    if support <= 0.0:
        raise ValueError("support must be positive")

    dist = (pos[:, 1] - float(plane_y)) * side
    keep = (dist > 0.0) & (dist <= float(support))
    if not bool(keep.any()):
        empty_p = pos.new_zeros((0, pos.shape[1]))
        empty_m = mass.new_zeros((0,))
        return empty_p, empty_p.clone(), empty_m

    ghost_pos = pos[keep].clone()
    ghost_pos[:, 1] = 2.0 * float(plane_y) - pos[keep, 1]

    ghost_vel = vel[keep].clone()
    if slip == "no_slip":
        ghost_vel = -ghost_vel
    else:  # free_slip
        ghost_vel[:, 1] = -ghost_vel[:, 1]

    return ghost_pos, ghost_vel, mass[keep].clone()


def plane_walls(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor,
                planes: Sequence[tuple[float, int]], support: float,
                slip: str = "no_slip"):
    """Mirror across several planar walls; returns stacked ghosts.

    ``planes`` is a sequence of ``(plane_y, side)`` pairs.
    """
    gp, gv, gm = [], [], []
    for plane_y, side in planes:
        p, v, m = mirror_plane(pos, vel, mass, plane_y, support, side, slip)
        if p.shape[0]:
            gp.append(p)
            gv.append(v)
            gm.append(m)
    if not gp:
        empty_p = pos.new_zeros((0, pos.shape[1]))
        return empty_p, empty_p.clone(), mass.new_zeros((0,))
    return torch.cat(gp, dim=0), torch.cat(gv, dim=0), torch.cat(gm, dim=0)
