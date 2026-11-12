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

Moving (prescribed-velocity) planes
-----------------------------------
``plane_vel`` gives the plane itself a velocity ``w`` (default zero, which is
every result quoted above).  The mirror is then taken *through the wall's own
motion* instead of through rest:

``no_slip``     ghost ``2 w - v``  -- the antisymmetric pair has mean ``w``, so
                the discrete viscous operator sees ``u = w`` on the plane, and
                an oscillating wall can drive the fluid (:mod:`tensorsph.
                stokes_second` and ``benchmarks/stokes_second``).
``free_slip``   ghost ``(vx, 2 wy - vy)`` -- only the normal component is
                mirrored, so the tangential velocity difference across the plane
                is *zero* and no tangential momentum is exchanged; a tangentially
                moving free-slip wall therefore cannot drive the flow, which is
                the negative control for the moving-wall path.

Both reduce to the static-wall forms above at ``w = 0``, so every previously
measured number is unchanged.
"""

from __future__ import annotations

from typing import Sequence

import torch

__all__ = ["mirror_plane", "plane_walls"]


def mirror_plane(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor,
                 plane_y: float, support: float, side: int = 1,
                 slip: str = "no_slip", plane_vel=None):
    """Mirror the near-plane fluid across ``y = plane_y``.

    Only fluid particles within ``support`` of the plane are mirrored: a ghost
    further than that is beyond the kernel support of every fluid particle and
    would only cost work.

    ``plane_vel`` is the prescribed velocity ``(wx, wy)`` of the plane itself
    (sequence or tensor, default zero).  With ``slip="no_slip"`` the ghost is
    ``2 w - v``, so the wall/fluid viscous pair has mean ``w`` and the no-slip
    plane moves at ``w``; with ``slip="free_slip"`` only the normal component is
    mirrored about ``w``, so no tangential shear is exchanged.

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
    wall_v = torch.zeros(ghost_vel.shape[1], dtype=ghost_vel.dtype,
                         device=ghost_vel.device)
    if plane_vel is not None:
        wall_v = torch.as_tensor(plane_vel, dtype=ghost_vel.dtype,
                                 device=ghost_vel.device).reshape(-1)
        if wall_v.numel() != ghost_vel.shape[1]:
            raise ValueError(
                f"plane_vel must have {ghost_vel.shape[1]} components, "
                f"got {wall_v.numel()}")
    if slip == "no_slip":
        # mirror through the wall's own motion: mean of (v, 2w - v) is w
        ghost_vel = 2.0 * wall_v - ghost_vel
    else:  # free_slip: only the normal component is mirrored about w
        ghost_vel[:, 1] = 2.0 * wall_v[1] - ghost_vel[:, 1]
        # the tangential component is left untouched: zero tangential jump,
        # hence zero tangential momentum exchange even for a moving wall.

    return ghost_pos, ghost_vel, mass[keep].clone()


def plane_walls(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor,
                planes: Sequence[tuple], support: float,
                slip: str = "no_slip"):
    """Mirror across several planar walls; returns stacked ghosts.

    ``planes`` is a sequence of ``(plane_y, side)`` or ``(plane_y, side,
    plane_vel)`` entries; a per-plane velocity lets one wall move while the
    other is at rest (Stokes' second problem, an oscillating wavemaker floor).
    """
    gp, gv, gm = [], [], []
    for plane in planes:
        plane_y, side = plane[0], plane[1]
        plane_vel = plane[2] if len(plane) > 2 else None
        p, v, m = mirror_plane(pos, vel, mass, plane_y, support, side, slip,
                               plane_vel=plane_vel)
        if p.shape[0]:
            gp.append(p)
            gv.append(v)
            gm.append(m)
    if not gp:
        empty_p = pos.new_zeros((0, pos.shape[1]))
        return empty_p, empty_p.clone(), mass.new_zeros((0,))
    return torch.cat(gp, dim=0), torch.cat(gv, dim=0), torch.cat(gm, dim=0)
