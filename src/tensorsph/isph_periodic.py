"""Periodic incompressible-projection (ISPH) operators for a torus.

Why this module exists
----------------------
The repository's projection machinery (``tensorsph.projection_adjoint``) is
built for **wall-bounded** clouds: the Poisson problem is regularised by pinning
one pressure particle and every operator evaluates the plain separation
``r_i - r_j``.  The method-selection registry therefore only carries a
*channel* record for ``isph_projection``, and the selector is (correctly) not
allowed to extrapolate it to the periodic vortex class
(``smooth_inviscid_vortex`` / Gresho-Taylor-Green), which is exactly the hole
``XVAL-SPH-METHOD-SELECTION-2026-11-12`` still names as residual debt.

This module supplies the *periodic* counterpart, so that a vortex-class case can
be computed at all and its error measured:

* pairs under the minimum-image convention (no ghost particles, no walls);
* the ISPH operator triple of Cummins & Rudman (1999) / Shao & Lo (2003),
  each normalised by its own kernel moment so that it is first-order consistent
  on a **disordered** cloud (an un-normalised pair operator loses O(1) accuracy
  as soon as the lattice jitters -- measured in
  ``tests/test_isph_periodic.py``):

      D u|_i  = 1/b_i  sum_j V_j (u_j - u_i) . grad_i W_ij        (~ div u)
      G p|_i  = 1/b_i  sum_j V_j (p_j - p_i)  grad_i W_ij         (~ grad p)
      L p|_i  = 1/a_i  sum_j c_ij (p_i - p_j),  c_ij = -2 V_j (r.grad W)/(r^2+eta^2)

  with ``b_i = sum_j (-V_j r_ij . grad W_ij)`` and ``a_i = sum_j c_ij``; both
  reduce to the kernel moments (2 in 2D, 3 in 3D) on a lattice and to the
  local Shepard factor on a disordered cloud.  ``L ~ -laplacian`` is a graph
  (M-matrix) Laplacian: symmetric positive definite up to a constant, i.e. free
  of the *checkerboard* null mode that makes the strictly-adjoint pair
  ``D D^T`` singular on a collocated lattice (measured: CG on ``D D^T`` blows
  the pressure up to +-1e19 on a 20x20 lattice, see
  ``docs/xval_isph_projection_taylor_green.md``).
* the pressure Poisson equation in its standard ISPH form
  ``L p = -(rho/dt) D u*``, solved with a matrix-free CG; on a torus the
  constant mode is removed by projecting the iterate to zero mean instead of
  pinning a particle (a pin would make one particle special, which is wrong in
  a periodic cell).
* one projection step and one viscous (pair-Laplacian) step, plus a helper
  :func:`divergence_norm` used as the case-level gate on how solenoidal the
  result is.

Sign conventions are stated next to every operator because a flipped sign in a
projection does not produce a crash -- it produces a run that doubles its own
divergence every step, which is what the first draft of this module did.

References
    Cummins & Rudman (1999), J. Comput. Phys. 152:584  (projection ISPH).
    Shao & Lo (2003), Adv. Water Resour. 26:787       (pressure Poisson, source term).
    Monaghan (1992), Annu. Rev. Astron. Astrophys. 30:543 (pair Laplacian).
"""
from __future__ import annotations

import torch

from .csf_consistent import cubic_spline_gradient

__all__ = [
    "periodic_pairs",
    "pair_geometry",
    "divergence",
    "gradient",
    "laplacian",
    "kernel_moments",
    "poisson",
    "project",
    "viscous_laplacian",
    "divergence_norm",
]


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def minimum_image(delta: torch.Tensor, box: float) -> torch.Tensor:
    """Shortest periodic image of ``delta`` in a box of side ``box``."""
    return delta - box * torch.round(delta / box)


def periodic_pairs(pos: torch.Tensor, cutoff: float, box: float, chunk: int = 1024):
    """Unordered ``i < j`` pairs under the minimum-image convention.

    Returns a ``(P, 2)`` long tensor.  Separations are *not* cached: they are
    recomputed from the current positions by :func:`pair_geometry`, because a
    cached separation silently freezes the geometry.
    """
    n = pos.shape[0]
    index = torch.arange(n, device=pos.device)
    rows_i, rows_j = [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = pos[start:stop, None, :] - pos[None, :, :]
        block = minimum_image(block, box)
        r2 = block.square().sum(dim=-1)
        ii = index[start:stop, None].expand(stop - start, n)
        jj = index[None, :].expand(stop - start, n)
        mask = (r2 < cutoff * cutoff) & (jj > ii)
        if not bool(mask.any()):
            continue
        rows_i.append(ii[mask])
        rows_j.append(jj[mask])
    if not rows_i:
        empty = torch.zeros(0, dtype=torch.long, device=pos.device)
        return torch.stack((empty, empty), dim=1)
    return torch.stack((torch.cat(rows_i), torch.cat(rows_j)), dim=1)


def pair_geometry(pos, pairs, h, box):
    """``(i, j, r_ij, grad_i W_ij)`` with ``r_ij`` the minimum image of ``r_i-r_j``."""
    i, j = pairs.unbind(dim=1)
    delta = minimum_image(pos[i] - pos[j], box)
    return i, j, delta, cubic_spline_gradient(delta, h)


def _lap_coeffs(pos, vol, pairs, h, box):
    """Symmetric graph-Laplacian coefficients ``c_ij > 0`` and their diagonal.

    ``c_ij = -2 Vbar_ij (r . grad W) / (r^2 + eta^2)`` with ``Vbar_ij`` the mean
    of the two volumes: the mean keeps the operator exactly symmetric (a
    particle-dependent ``V_j`` would make it a non-symmetric weighting, on which
    CG is not applicable).
    """
    i, j, delta, gw = pair_geometry(pos, pairs, h, box)
    r2 = (delta * delta).sum(dim=1)
    rdot = (delta * gw).sum(dim=1)
    eta = 0.1 * h
    coef = -2.0 * (0.5 * (vol[i] + vol[j])) * rdot / (r2 + eta * eta)
    n = pos.shape[0]
    diag = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    diag = diag.index_add(0, i, coef).index_add(0, j, coef)
    return i, j, coef, diag.clamp_min(1e-30)


def kernel_moments(pos, vol, pairs, h, box):
    """Per-particle normalisers ``(b_i, a_i)`` of the D/G and L operators.

    ``b_i`` is the first moment of the kernel gradient (``~2``/``~3`` on a
    2D/3D lattice, Shepard-like on a disordered cloud) and ``a_i`` the diagonal
    of the graph Laplacian.  Dividing by them is what keeps the operators
    first-order consistent when the particles stop being a lattice.
    """
    i, j, delta, gw = pair_geometry(pos, pairs, h, box)
    n = pos.shape[0]
    rdot = (delta * gw).sum(dim=1)
    dim = pos.shape[1]
    b = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    b = b.index_add(0, i, -(rdot * vol[j])).index_add(0, j, -(rdot * vol[i]))
    _i, _j, _c, a = _lap_coeffs(pos, vol, pairs, h, box)
    # ``sum_j -V_j r_ij . grad W_ij`` is the *trace* of the first moment
    # tensor, i.e. ``dim`` times the per-component normaliser of the gradient.
    return (b / dim).clamp_min(1e-30), a


# --------------------------------------------------------------------------- #
# the operator triple
# --------------------------------------------------------------------------- #
def divergence(pos, u, vol, pairs, h, box, moments=None):
    """``div u|_i = (1/b_i) sum_j V_j (u_j - u_i) . grad_i W_ij``  (~ div u)."""
    i, j, delta, gw = pair_geometry(pos, pairs, h, box)
    b, _a = kernel_moments(pos, vol, pairs, h, box) if moments is None else moments
    coef = ((u[j] - u[i]) * gw).sum(dim=1)
    out = torch.zeros(pos.shape[0], dtype=u.dtype, device=u.device)
    out = out.index_add(0, i, vol[j] * coef).index_add(0, j, vol[i] * coef)
    return out / b


def gradient(pos, p, vol, pairs, h, box, moments=None):
    """``grad p|_i = (1/b_i) sum_j V_j (p_j - p_i) grad_i W_ij``  (~ grad p)."""
    i, j, delta, gw = pair_geometry(pos, pairs, h, box)
    b, _a = kernel_moments(pos, vol, pairs, h, box) if moments is None else moments
    coef = p[j] - p[i]
    out = torch.zeros((pos.shape[0], pos.shape[1]), dtype=p.dtype, device=p.device)
    out = out.index_add(0, i, (vol[j] * coef)[:, None] * gw)
    out = out.index_add(0, j, (vol[i] * coef)[:, None] * gw)
    return out / b[:, None]


def laplacian(pos, p, vol, pairs, h, box, moments=None):
    """``L p|_i = sum_j c_ij (p_i - p_j)``  (~ -laplacian p): graph Laplacian.

    Symmetric positive semi-definite (``c_ij > 0``), constants are its only null
    space on a clean lattice, so CG needs no pinned particle -- only a zero-mean
    iterate on a torus.

    **No diagonal division**: the raw pair sum is already the consistent ISPH
    Laplacian (Cummins & Rudman 1999).  Dividing by ``diag(A)`` was measured to
    cancel the second-order term the Laplacian lives on and return ~0 for a
    smooth Fourier mode (ratio -1e-3, ``tests/test_isph_periodic.py``).
    """
    i, j, coef, _a = _lap_coeffs(pos, vol, pairs, h, box)
    val = coef * (p[i] - p[j])
    out = torch.zeros(pos.shape[0], dtype=p.dtype, device=p.device)
    out = out.index_add(0, i, val).index_add(0, j, -val)
    return out


def divergence_norm(pos, u, vol, pairs, h, box):
    """RMS of :func:`divergence`, i.e. how compressible the discrete field is."""
    d = divergence(pos, u, vol, pairs, h, box)
    return float(torch.sqrt((d * d).mean()))


# --------------------------------------------------------------------------- #
# Poisson solve and projection
# --------------------------------------------------------------------------- #
def poisson(pos, source, vol, pairs, h, box, rel_tol: float = 1e-10,
            max_iter: int = 4000):
    """Jacobi-preconditioned CG for ``L p = source`` (``L`` = :func:`laplacian`).

    ``L`` is the symmetric graph Laplacian with ``diag(a)``; the diagonal is used
    as the preconditioner.  Its only null space on a torus is the constant mode,
    removed by projecting residuals and iterates to zero mean (no particle is
    pinned -- that would single out one particle in a periodic cell).
    """
    i, j, coef, a = _lap_coeffs(pos, vol, pairs, h, box)

    def apply(q):
        q = q - q.mean()
        val = coef * (q[i] - q[j])
        out = torch.zeros(q.shape[0], dtype=q.dtype, device=q.device)
        out = out.index_add(0, i, val).index_add(0, j, -val)
        return out - out.mean()

    b = source - source.mean()
    p = torch.zeros_like(b)
    r = b.clone()
    z = r / a
    d = z.clone()
    rz = (r * z).sum()
    norm0 = torch.linalg.vector_norm(b).clamp_min(1e-300)
    it = 0
    for it in range(1, max_iter + 1):
        ad = apply(d)
        den = (d * ad).sum()
        if float(den.abs()) <= 1e-300:
            break
        alpha = rz / den
        p = p + alpha * d
        r = r - alpha * ad
        if float(torch.linalg.vector_norm(r) / norm0) < rel_tol:
            break
        z = r / a
        rz_new = (r * z).sum()
        if float(rz.abs()) <= 1e-300:
            break
        d = z + (rz_new / rz) * d
        rz = rz_new
    return p - p.mean(), it


def project(pos, u_star, vol, pairs, h, box, rho: float, dt: float,
            rel_tol: float = 1e-10, max_iter: int = 4000):
    """One ISPH projection step; returns ``(u, p, info)``.

    Solves ``L p = -(rho/dt) D u*`` (with ``L ~ -laplacian`` this is the textbook
    ``laplacian p = (rho/dt) div u*``) and corrects ``u = u* - (dt/rho) G p``.
    Because ``L`` and ``-D G`` are two different stencils of the same operator,
    the discrete divergence is *reduced*, not annihilated: the leftover is the
    truncation mismatch and is reported as ``div_after_over_div_before``.
    """
    moments = kernel_moments(pos, vol, pairs, h, box)
    div_star = divergence(pos, u_star, vol, pairs, h, box, moments)
    p, it = poisson(pos, -(rho / dt) * div_star, vol, pairs, h, box,
                    rel_tol=rel_tol, max_iter=max_iter)
    u = u_star - (dt / rho) * gradient(pos, p, vol, pairs, h, box, moments)
    div_new = divergence(pos, u, vol, pairs, h, box, moments)
    scale = torch.linalg.vector_norm(div_star).clamp_min(1e-300)
    return u, p, {
        "cg_iterations": int(it),
        "div_star_rms": float(torch.sqrt((div_star * div_star).mean())),
        "div_after_over_div_before": float(torch.linalg.vector_norm(div_new) / scale),
    }


def viscous_laplacian(pos, u, vol, pairs, h, box, nu: float):
    """``nu * laplacian(u)`` component-wise with the pair (graph) Laplacian.

    ``laplacian`` returns ``-nabla^2`` up to the per-particle normaliser, so the
    sign is flipped here to keep this function a *diffusion* (positive ``nu``
    must decay the field, which the tests pin on a single Fourier mode).
    """
    if nu <= 0.0:
        return torch.zeros_like(u)
    return -nu * torch.stack(
        [laplacian(pos, u[:, 0], vol, pairs, h, box),
         laplacian(pos, u[:, 1], vol, pairs, h, box)], dim=1)