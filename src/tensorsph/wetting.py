"""Wall contact-angle (wetting) boundary condition for the CSF surface-tension
model.

``docs/contact_angle_boundary.md`` documented this boundary condition and
``tests/test_contact_angle.py`` imports ``apply_contact_angle_boundary`` from
``tensorsph.surface_tension``, but the function did not exist: the test module
failed at collection and the documented ``Config.contact_angle`` knob was a
silent no-op.  This module is the implementation the documentation describes.

Geometry and sign convention
----------------------------
At a three-phase contact line Young's relation fixes the angle ``theta``
between the *interface tangent* and the *wall*, measured inside the liquid.
For a numerical model it is easier to state the same condition on the interface
normal ``n`` (unit vector pointing out of the liquid, i.e. ``n = -grad(c)/|grad(c)|``
for a colour field that is 1 inside the liquid).  With ``w`` the wall normal
pointing *into* the liquid, the contact angle satisfies

    n . w = cos(theta)                                            (1)

Check it on a sessile drop (a circular cap of radius ``R`` whose centre sits at
height ``y_c`` above the wall): at the contact point
``n = (x0/R, -y_c/R)`` and ``w = (0, 1)``, so ``n . w = -y_c/R``; a hemisphere
(``theta = 90 deg``) has ``y_c = 0``, a flat puddle (``theta -> 0``) has
``y_c -> -infty`` with ``R ~ |y_c|`` and a non-wetting sphere (``theta -> 180``)
has ``y_c ~ R``.  Every limit matches ``cos(theta)``, so (1) is the condition
this module enforces.

The implementation replaces the near-wall colour gradient by

    grad(c)_new = -|grad(c)| * (cos(theta) * w + sin(theta) * t_hat)   (2)

where ``t_hat`` is the current outward normal projected onto the wall plane and
re-normalised.  Magnitude is preserved exactly, ``theta = pi/2`` reproduces the
tangential component of the incoming normal, and the original tangential
direction is kept, so the model does not invent an azimuth around the wall
normal.

Everything is exact geometry: no calibration constant enters.  The residual
error of a single projection is floating-point round-off, which is what
``benchmarks/contact_angle/run.py`` measures.
"""
from __future__ import annotations

import math

import torch

from .droplet_impact import cubic_spline_kernel
from .neighbor_search import cell_linked_pairs

# Band, in smoothing lengths, in which the wall rotates the interface normal.
# Two support radii is where the cubic-spline kernel vanishes, so beyond it a
# wall particle cannot influence the colour field at all.
NEAR_WALL_SUPPORT = 2.0


def wall_normals(
    pos: torch.Tensor,
    wall: torch.Tensor,
    h: float,
    support: float = NEAR_WALL_SUPPORT,
):
    """Outward wall normal and wall distance for every fluid particle.

    Returns ``(normal, distance, near)``:

    * ``normal``   -- ``(N, 2)`` unit vector pointing *into* the fluid, zero
      where no wall particle is inside the support;
    * ``distance`` -- ``(N,)`` distance to the closest wall particle, ``inf``
      outside the support;
    * ``near``     -- ``(N,)`` boolean, wall particle inside the support.

    The normal is the kernel-weighted mean of the fluid-minus-wall offsets,
    ``w_i = normalise(sum_j W_ij (x_i - x_j))``.  For a flat wall the tangential
    components cancel and ``w`` is the wall normal; using all wall particles
    inside the support instead of only the closest one keeps ``w`` smooth when
    the lattice is irregular.  A wall that surrounds a particle from opposite
    sides (a slot narrower than the support) cancels to zero and is reported as
    "no normal" rather than as an arbitrary direction.
    """
    n = pos.shape[0]
    dtype, device = pos.dtype, pos.device
    normal = torch.zeros((n, pos.shape[1]), dtype=dtype, device=device)
    distance = torch.full((n,), float("inf"), dtype=dtype, device=device)
    near = torch.zeros((n,), dtype=torch.bool, device=device)
    if n == 0 or wall is None or wall.numel() == 0 or wall.shape[0] == 0:
        return normal, distance, near

    cutoff = support * h
    all_pos = torch.cat((pos, wall), dim=0)
    pairs = cell_linked_pairs(all_pos, cutoff)
    if pairs.numel() == 0:
        return normal, distance, near

    i, j = pairs.unbind(dim=1)
    # Fluid particles are stored first, and pairs are i < j, so a fluid-wall
    # pair is exactly "i is fluid and j is wall".
    is_fluid_wall = (i < n) & (j >= n)
    if not bool(is_fluid_wall.any()):
        return normal, distance, near
    i = i[is_fluid_wall]
    j = j[is_fluid_wall]

    delta = pos[i] - wall[j - n]
    r = torch.linalg.vector_norm(delta, dim=1)
    weight = cubic_spline_kernel(r, h)

    acc = torch.zeros((n, pos.shape[1]), dtype=dtype, device=device)
    acc = acc.index_add(0, i, weight[:, None] * delta)
    # Nearest wall-particle distance, vectorised: scatter the pair distances to
    # the owning fluid particle and reduce with "amin".
    closest = torch.full((n,), float("inf"), dtype=dtype, device=device)
    closest = closest.scatter_reduce(0, i, r, reduce="amin", include_self=True)

    norm = torch.linalg.vector_norm(acc, dim=1)
    ok = norm > 1.0e-12
    normal[ok] = acc[ok] / norm[ok, None]
    distance = closest
    near = torch.isfinite(closest)
    return normal, distance, near


def apply_contact_angle_boundary(
    grad_c: torch.Tensor,
    pos: torch.Tensor,
    wall: torch.Tensor,
    h: float,
    theta: float,
    support: float = NEAR_WALL_SUPPORT,
) -> torch.Tensor:
    """Rotate the colour gradient of near-wall particles to enforce ``theta``.

    Args:
        grad_c: ``(N, 2)`` colour gradient; it points from the vacuum into the
            liquid, so ``-grad_c/|grad_c|`` is the outward interface normal.
        pos: ``(N, 2)`` fluid particle positions.
        wall: ``(M, 2)`` wall particle positions (static).
        h: smoothing length.
        theta: contact angle in radians, measured inside the liquid between the
            interface and the wall.  ``theta < pi/2`` is wetting,
            ``theta = pi/2`` neutral, ``theta > pi/2`` non-wetting.
        support: band width in smoothing lengths around the wall.

    Returns:
        ``(N, 2)`` gradient with the same magnitude as the input.  Particles
        outside the wall support, particles with a vanishing gradient, and
        particles whose normal is parallel to the wall normal (no tangential
        direction to rotate about) are returned unchanged -- the boundary
        condition is undefined there, and silently inventing a direction would
        be a fake.
    """
    out = grad_c.clone()
    if grad_c.numel() == 0 or wall is None or wall.numel() == 0:
        return out

    normal_wall, _distance, near = wall_normals(pos, wall, h, support)
    active = near & (torch.linalg.vector_norm(normal_wall, dim=1) > 0.0)
    if not bool(active.any()):
        return out

    magnitude = torch.linalg.vector_norm(grad_c, dim=1)
    active = active & (magnitude > 1.0e-12)
    if not bool(active.any()):
        return out

    index = active.nonzero(as_tuple=False).squeeze(1)
    mag = magnitude[index]
    outward = -grad_c[index] / mag[:, None]          # unit outward normal
    w = normal_wall[index]

    tangential = outward - (outward * w).sum(dim=1, keepdim=True) * w
    t_norm = torch.linalg.vector_norm(tangential, dim=1)
    usable = t_norm > 1.0e-6
    index = index[usable]
    if index.numel() == 0:
        return out
    mag = mag[usable]
    w = w[usable]
    t_hat = tangential[usable] / t_norm[usable, None]

    cos_t = torch.as_tensor(math.cos(float(theta)), dtype=grad_c.dtype,
                            device=grad_c.device)
    sin_t = torch.as_tensor(math.sin(float(theta)), dtype=grad_c.dtype,
                            device=grad_c.device)
    # cos/sin of the same angle and an orthonormal (w, t_hat) pair make this
    # unit already; the explicit normalisation removes the float32 round-off so
    # |grad c| is preserved to round-off as the boundary condition promises.
    new_outward = cos_t * w + sin_t * t_hat
    new_outward = new_outward / torch.linalg.vector_norm(
        new_outward, dim=1, keepdim=True).clamp_min(1.0e-30)
    out = out.index_copy(0, index, -mag[:, None] * new_outward)
    return out


def measure_contact_angle(
    grad_c: torch.Tensor,
    normal_wall: torch.Tensor,
) -> torch.Tensor:
    """Contact angle implied by a gradient, ``acos(-n . w)`` in radians.

    ``n = -grad_c/|grad_c|`` is the outward interface normal and ``w`` the wall
    normal pointing into the fluid, so equation (1) gives
    ``theta = acos(n . w)``.  Used by the tests and the benchmark to score the
    construction instead of trusting it.
    """
    mag = torch.linalg.vector_norm(grad_c, dim=-1)
    outward = -grad_c / mag.clamp_min(1.0e-30).unsqueeze(-1)
    cos_theta = (outward * normal_wall).sum(dim=-1)
    return torch.arccos(cos_theta.clamp(-1.0, 1.0))