"""Interface curvature from a weighted local surface fit (CSF helper).

``csf_fields`` originally obtained the CSF curvature as the MLS divergence of the
unit normal, ``kappa = div n``.  That is the textbook CSF definition, but its
truncation error on a particle shell is a *second* derivative of a quantity that
is itself discretised noise: the normal is piecewise constant inside a particle
layer, so ``div n`` pays an ``O(1)`` lattice error rather than the ``O(h^2)`` of
an interpolated field.

On a circle of radius ``R`` the measured error was a systematic **+25 %**
(``docs/verification_surface_tension_consistent.md``), i.e. outside the platform
15 % gate even though the force *amplitude* (cut-force identity) and the force
*direction* were already inside it.

This module replaces ``div n`` by a geometric fit of the interface surface in
the neighbourhood of every interface particle, which is the standard cure in
the level-set / front-tracking literature and is second-order accurate for a
smooth interface.  On a hexagonal lattice the error is resolution-dependent
due to staircase effects:

| R/dx | fit_quadratic κ error | <15 %? |
|------|----------------------|--------|
| 5    | +37 %                | no     |
| 6.7  | +18 %                | no     |
| 10   | +5 %                 | yes    |
| 13.3 | +1 %                 | yes    |

The operator needs R/dx ≥ 10 to meet the 15 % platform gate on a hex lattice.
For exact (non-lattice) circle data the quadratic fit is algebraically accurate
to within a few percent even at coarse resolution.

``fit_quadratic`` (default)
    Rotate into the local frame ``(t, n)`` of particle ``i`` and fit the graph
    ``v = c + b u + a u^2`` to the nearby interface particles, weighted by
    ``delta_s * W(r)/W(0)`` (the CSF surface weight times a normalised kernel,
    so the stencil is local and only the shell contributes).  Then

        kappa = -2 a / (1 + b^2)^{3/2},

    the signed curvature of the graph at ``u = 0`` with the convention
    ``kappa > 0`` when the interface bends *away* from the outward normal
    (convex droplet) -- identical to the convention of ``div n`` used by the
    rest of the code.

``fit_circle``
    Algebraic (Pratt / Kasa) weighted circle fit of the same stencil in the
    frame centred on particle ``i``, ``min sum_j w_j (|y_j - c|^2 - R^2)^2``,
    then ``kappa = -sign(c . n_i) / R``.

Particles whose stencil does not contain enough interface partners keep the MLS
curvature supplied by the caller, so the operator degrades to the previous
behaviour instead of producing NaNs (lattice defects, isolated pairs, 3-D).

References:
    Brackbill, Kothe & Zemach (1992), J. Comput. Phys. 100:335 (CSF).
    Cummins, Francois & Kothe (2005), Comput. Struct. 83:425 (curvature from
        interface fits in particle level-set methods).
    Marchandise et al. (2007), J. Comput. Phys. 226:1906 (quadratic interface
        fit for curvature in particle methods).
"""
from __future__ import annotations

import torch

__all__ = ["fit_curvature", "SCHEMES"]

SCHEMES = ("mls", "fit_quadratic", "fit_circle")


def _scatter(feats: list[torch.Tensor], w: torch.Tensor, dst: torch.Tensor,
             n: int) -> torch.Tensor:
    """Accumulate ``sum w * feat`` for each feature at index ``dst``."""
    stack = torch.stack(feats, dim=0) * w
    out = torch.zeros((len(feats), n), dtype=stack.dtype, device=stack.device)
    return out.index_add(1, dst, stack)


def fit_curvature(
    pos: torch.Tensor,
    normal: torch.Tensor,
    delta_s: torch.Tensor,
    pairs: torch.Tensor,
    h: float,
    fallback: torch.Tensor | None = None,
    method: str = "fit_quadratic",
    stencil: float = 4.0,
    min_weight_fraction: float = 0.05,
) -> torch.Tensor:
    """Signed curvature of the colour-function interface, per particle.

    Args:
        pos: (N, dim) particle positions.
        normal: (N, dim) outward unit normals (zero away from the interface).
        delta_s: (N,) surface delta function, used as the interface weight.
        pairs: (P, 2) neighbour pairs, cutoff at least ``stencil * h``.
        h: smoothing length.
        fallback: (N,) curvature used where the stencil is too poor, usually the
            MLS ``div n``; defaults to zeros.
        method: ``"fit_quadratic"``, ``"fit_circle"``, or ``"mls"`` (returns
            ``fallback`` unchanged, kept for A/B runs).
        stencil: fit radius in units of ``h``.  ``pairs`` must be built with at least
            this cutoff; 4h is the validated default (2h, the kernel support, is
            too short to resolve a lattice staircase).
        min_weight_fraction: particles whose ``delta_s`` is below this fraction
            of the maximum are bulk and keep the fallback curvature.

    Returns:
        (N,) signed curvature in 1/m.
    """
    if fallback is None:
        fallback = torch.zeros(pos.shape[0], dtype=pos.dtype, device=pos.device)
    if method == "mls":
        return fallback
    if method not in SCHEMES:
        raise ValueError(f"unknown curvature method {method!r}; expected {SCHEMES}")
    if pos.shape[1] != 2:
        # A 3-D interface fit (quadric / sphere) is a separate operator; keep the
        # existing behaviour rather than silently fitting a 2-D frame in 3-D.
        return fallback
    if pairs.numel() == 0:
        return fallback

    n = pos.shape[0]
    dtype, device = pos.dtype, pos.device
    ds_max = delta_s.max().clamp_min(1e-30)
    w_node = (delta_s / ds_max).clamp(0.0, 1.0)

    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    r = torch.linalg.vector_norm(delta, dim=1)
    inside = r <= stencil * h

    # Flat weights inside the stencil, scaled by the partner's interface weight.
    # A kernel weight is *wrong* here: the cubic spline support is 2h, so a
    # stencil wider than that would silently collapse back to the 2h shell --
    # which on a lattice carries only 3-4 staircase points and gives +55 %
    # curvature error with 40 % sign flips.  The interface shell is a single
    # particle layer thick, so the stencil, not the kernel, sets the arc length.
    w_ij = w_node[j].masked_fill(~inside, 0.0)
    w_ji = w_node[i].masked_fill(~inside, 0.0)
    w_self = w_node
    active = w_node > min_weight_fraction

    if method == "fit_quadratic":
        tangent = torch.stack([-normal[:, 1], normal[:, 0]], dim=1)
        # Offset of the partner from the target, in the target's local frame
        # (u along the tangent, v along the outward normal).
        u_i = (-delta * tangent[i]).sum(dim=1)
        v_i = (-delta * normal[i]).sum(dim=1)
        u_j = (delta * tangent[j]).sum(dim=1)
        v_j = (delta * normal[j]).sum(dim=1)

        def quad_moments(u, v, w, dst):
            return _scatter([torch.ones_like(u), u, v, u * u, u * v,
                             u * u * u, u * u * v, u * u * u * u], w, dst, n)

        s = quad_moments(u_i, v_i, w_ij, i) + quad_moments(u_j, v_j, w_ji, j)
        s[0] = s[0] + w_self          # the target sits at (u, v) = (0, 0)
        s0, s1, s2, s3, s4, s5, s6, s7 = s
        # Normal equations for v = c + b u + a u^2:
        #   [s0 s1 s3][c]   [s2]      (s2 = sum w v)
        #   [s1 s3 s5][b] = [s4]      (s4 = sum w u v)
        #   [s3 s5 s7][a]   [s6]      (s6 = sum w u^2 v)
        mat = torch.zeros((n, 3, 3), dtype=dtype, device=device)
        mat[:, 0, 0], mat[:, 0, 1], mat[:, 0, 2] = s0, s1, s3
        mat[:, 1, 0], mat[:, 1, 1], mat[:, 1, 2] = s1, s3, s5
        mat[:, 2, 0], mat[:, 2, 1], mat[:, 2, 2] = s3, s5, s7
        rhs = torch.stack([s2, s4, s6], dim=1)
        det = torch.linalg.det(mat)
        scale = det.abs().max().clamp_min(1e-30)
        ok = active & (det.abs() > 1e-6 * scale)
        eye = torch.eye(3, dtype=dtype, device=device)
        mat = torch.where(ok[:, None, None], mat, eye.expand_as(mat))
        sol = torch.linalg.solve(mat, rhs.unsqueeze(-1)).squeeze(-1)
        b, a = sol[:, 1], sol[:, 2]
        kappa_fit = -2.0 * a / (1.0 + b * b) ** 1.5
        return torch.where(ok, kappa_fit, fallback)

    # fit_circle: Pratt algebraic fit in the frame centred on the target.
    def circle_moments(x, y, w, dst):
        f = x * x + y * y
        return _scatter([torch.ones_like(x), x, y, x * x, x * y, y * y,
                         x * f, y * f, f], w, dst, n)

    m = (circle_moments(-delta[:, 0], -delta[:, 1], w_ij, i)
         + circle_moments(delta[:, 0], delta[:, 1], w_ji, j))
    z0, z1, z2, z3, z4, z5, z6, z7, z8 = m
    z0 = z0 + w_self      # self point at the origin of the local frame
    # Pratt: (x^2 + y^2) + D x + E y + F = 0
    #   [z3 z4 z1][D]   [-z6]
    #   [z4 z5 z2][E] = [-z7]
    #   [z1 z2 z0][F]   [-z8]
    mat = torch.zeros((n, 3, 3), dtype=dtype, device=device)
    mat[:, 0, 0], mat[:, 0, 1], mat[:, 0, 2] = z3, z4, z1
    mat[:, 1, 0], mat[:, 1, 1], mat[:, 1, 2] = z4, z5, z2
    mat[:, 2, 0], mat[:, 2, 1], mat[:, 2, 2] = z1, z2, z0
    rhs = torch.stack([-z6, -z7, -z8], dim=1)
    det = torch.linalg.det(mat)
    scale = det.abs().max().clamp_min(1e-30)
    ok = active & (det.abs() > 1e-6 * scale)
    eye = torch.eye(3, dtype=dtype, device=device)
    mat = torch.where(ok[:, None, None], mat, eye.expand_as(mat))
    sol = torch.linalg.solve(mat, rhs.unsqueeze(-1)).squeeze(-1)
    cx, cy = -0.5 * sol[:, 0], -0.5 * sol[:, 1]
    rad = torch.sqrt((cx * cx + cy * cy - sol[:, 2]).clamp_min(1e-30))
    # Convex droplet: the centre lies opposite the outward normal -> kappa > 0.
    sign = torch.sign(cx * normal[:, 0] + cy * normal[:, 1])
    kappa_fit = -sign / rad.clamp_min(1e-30)
    ok = ok & (sign != 0.0)
    return torch.where(ok, kappa_fit, fallback)
