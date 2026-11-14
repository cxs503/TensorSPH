"""Young-Laplace pressure jump ``Delta p = 2 sigma / R`` (3-D sphere) with projection SPH.

Why this module exists (the documented gap it closes)
-----------------------------------------------------
``laplace_isph`` verifies the projection/incompressible-SPH Young-Laplace route
on a **2-D** droplet, where the analytic law is ``dp = sigma / R``.  The 3-D
counterpart (``dp = 2 sigma / R``) was never scored, and the one number the
repository did measure on the shipped 3-D CSF operator was a failure:
``kappa R = 0.77`` instead of the exact ``2`` on a static lattice sphere at
``R/dx = 8..16``, resolution independent (``tensorsph.oscillating_drop3d``,
``docs/case_sph_osc_drop_freq.md``).  ``tensorsph.threed.csf3`` documents the
reason and the missing piece in one line: *"A correct implementation would fit
the surface in 3-D"* -- the shipped estimator is ``div n`` with ``n`` built from
a lattice-staircase colour gradient, and in 3-D the staircase noise of a
two-parameter interface does not average out the way it does for a 2-D curve.

What is implemented here
------------------------
1. ``fit_curvature3`` -- the 3-D local quadratic surface fit that closes it.
   For an interface particle the tangent frame ``(t1, t2)`` is built from its
   own normal, the neighbours are expressed in local coordinates
   ``(u, v, w) = ((x - x_i).t1, (x - x_i).t2, (x - x_i).n_i)``, and the weighted
   least-squares problem

       w(u, v) = a u + b v + (A/2) u^2 + B u v + (C/2) v^2

   is solved, giving the mean curvature ``kappa = -(A + C)`` with the sign
   convention of the 2-D module (positive where the interface bends away from
   the outward normal, i.e. a convex droplet).  Only neighbours inside the
   interface *slab* ``|w| <= band * h`` enter the fit: the CSF surface is a
   shell of thickness ``~2h`` and a band-unrestricted height fit sees the shell
   thickness as curvature (this is the failure mode measured for the
   axisymmetric projection of the 3-D shell in ``csf3.curvature_axisymmetric``,
   ``kappa R = -0.35`` on a cylinder).
2. ``pressure_poisson3`` -- the static projection step: the Cummins & Rudman
   (1999) pair Laplacian in 3-D, Dirichlet ``p = 0`` on the outer lattice layer
   (ambient gauge), matrix-free Jacobi-preconditioned CG.
3. ``laplace_isph3d_case`` -- the case: colour -> normal -> curvature -> CSF
   traction ``a = -sigma kappa delta_s n / rho0`` -> Poisson source
   ``-rho0 div a`` -> ``dp = <p>_core - <p>_band``, with the independent
   traction-flux cross-check ``-int rho a_r dr`` (an axisymmetric flux through a
   sphere, ``-sum_i a_r,i m_i / (4 pi R^2)``) which never touches the pressure
   solve.  A ``sigma = 0`` negative control must return ``dp ~ 0``.

Every reported number is a returned dict entry; nothing is typed by hand into
the report.
"""
from __future__ import annotations

import math

import torch

from .threed.kernel3 import cubic_spline_gradient, cubic_spline_kernel
from .threed.neighbors3 import box_pairs

__all__ = [
    "sphere_block", "mls_gradient3", "mls_divergence3", "shepard_color3",
    "shepard_smooth3", "fit_curvature3", "curvature_color_quadric3",
    "csf_state3_two_phase",
    "pressure_poisson3", "spurious_acceleration3", "laplace_isph3d_case",
]


# --------------------------------------------------------------------------- #
# carrier
# --------------------------------------------------------------------------- #
def sphere_block(radius: float = 0.02, dx: float = 0.002, outer: float = 0.012,
                 rho0: float = 1000.0):
    """Two-phase carrier: simple-cubic lattice block with a liquid sphere inside.

    The lattice spacing is ``dx``, so the rest volume is exactly ``V_i = dx^3``
    and the discrete kernel sum over a complete lattice is ``~1`` (checked in
    the tests through the colour field, ``c -> 1`` in the liquid).  The block
    half-width is ``radius + outer``; the outer ``1.6 dx`` layer is the
    Dirichlet gauge.

    Returns ``(pos, mass, liquid)`` with ``liquid = |x| < radius``.
    """
    half = radius + outer
    n = int(round(2.0 * half / dx))
    axis = (torch.arange(n, dtype=torch.float64) + 0.5) * dx - half
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    pos = grid.reshape(-1, 3).contiguous()
    liquid = torch.linalg.vector_norm(pos, dim=1) < radius
    mass = torch.full((pos.shape[0],), rho0 * dx ** 3, dtype=torch.float64)
    return pos, mass, liquid


# --------------------------------------------------------------------------- #
# MLS operators in 3-D (same conventions as tensorsph.csf_consistent, dim 3)
# --------------------------------------------------------------------------- #
def _mls_operators3(pos, vol, pairs, h):
    """``M_i = sum_j V_j (x_i - x_j) (x) grad_i W_ij`` and its validity mask."""
    n = pos.shape[0]
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    gw = cubic_spline_gradient(delta, h)
    outer = -torch.einsum("pa,pb->pab", delta, gw)
    big_m = torch.zeros((n, 3, 3), dtype=pos.dtype)
    big_m = big_m.index_add(0, i, vol[j, None, None] * outer)
    big_m = big_m.index_add(0, j, vol[i, None, None] * outer)
    det = torch.linalg.det(big_m)
    scale = det.abs().median().clamp_min(1e-30)
    ok = det.abs() > 1e-4 * scale
    eye = torch.eye(3, dtype=pos.dtype).expand_as(big_m)
    big_m = torch.where(ok[:, None, None], big_m, eye)
    return big_m, ok


def mls_gradient3(pos, field, vol, pairs, h):
    """First-order consistent ``grad(field)`` with volume weights ``V_j``."""
    n = pos.shape[0]
    if pairs.numel() == 0:
        return torch.zeros((n, 3), dtype=pos.dtype)
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    gw = cubic_spline_gradient(delta, h)
    df = field[j] - field[i]
    rhs = torch.zeros((n, 3), dtype=pos.dtype)
    rhs = rhs.index_add(0, i, vol[j, None] * (df[:, None] * gw))
    rhs = rhs.index_add(0, j, vol[i, None] * (df[:, None] * gw))
    big_m, _ = _mls_operators3(pos, vol, pairs, h)
    return torch.linalg.solve(big_m.transpose(-1, -2), rhs.unsqueeze(-1)).squeeze(-1)


def mls_divergence3(pos, field, vol, pairs, h):
    """First-order consistent ``div(field)`` with volume weights ``V_j``."""
    n = pos.shape[0]
    if pairs.numel() == 0:
        return torch.zeros(n, dtype=pos.dtype)
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    gw = cubic_spline_gradient(delta, h)
    df = field[j] - field[i]
    contrib = torch.einsum("pa,pb->pab", df, gw)
    s = torch.zeros((n, 3, 3), dtype=pos.dtype)
    s = s.index_add(0, i, vol[j, None, None] * contrib)
    s = s.index_add(0, j, vol[i, None, None] * contrib)
    big_m, _ = _mls_operators3(pos, vol, pairs, h)
    m_inv = torch.linalg.inv(big_m)
    return (s * m_inv.transpose(-1, -2)).sum(dim=(-2, -1))


# --------------------------------------------------------------------------- #
# colour field (two-phase, Shepard-normalised)
# --------------------------------------------------------------------------- #
def shepard_color3(pos, liquid, vol, pairs, h):
    """``c_i = sum_j V_j W_ij c_j / sum_j V_j W_ij`` with ``c = 1`` in the liquid.

    The pair list is unordered (``i < j``, stored once), so every pair
    contributes ``V_j W_ij`` to particle ``i`` *and* ``V_i W_ij`` to particle
    ``j``; the self term ``V_i W(0)`` is added explicitly.  On this lattice
    ``V_i W(0) = 0.145`` of the unit kernel sum, so dropping it would put the
    interior colour at 0.86 instead of 1 and bias ``|grad c|`` by ~14 %.
    """
    n = pos.shape[0]
    w0 = cubic_spline_kernel(torch.zeros((), dtype=pos.dtype), h)
    num = vol * w0 * liquid.to(pos.dtype)
    den = vol * w0
    if pairs.numel() == 0:
        return liquid.to(pos.dtype)
    i, j = pairs.unbind(dim=1)
    r = torch.linalg.vector_norm(pos[i] - pos[j], dim=1)
    w = cubic_spline_kernel(r, h)
    num = num.index_add(0, i, w * vol[j] * liquid[j].to(pos.dtype))
    den = den.index_add(0, i, w * vol[j])
    num = num.index_add(0, j, w * vol[i] * liquid[i].to(pos.dtype))
    den = den.index_add(0, j, w * vol[i])
    return torch.where(den > 1e-12, num / den.clamp_min(1e-30),
                       liquid.to(pos.dtype))


def shepard_smooth3(field, pos, vol, pairs, h, passes: int = 2):
    """Kernel (Shepard) smoothing of a particle field, ``passes`` times."""
    if passes <= 0 or pairs.numel() == 0:
        return field
    n = pos.shape[0]
    i, j = pairs.unbind(dim=1)
    r = torch.linalg.vector_norm(pos[i] - pos[j], dim=1)
    w = cubic_spline_kernel(r, h)
    w0 = cubic_spline_kernel(torch.zeros((), dtype=pos.dtype), h)
    out = field
    for _ in range(passes):
        num = (vol * w0)[:, None] * out
        den = torch.zeros(n, dtype=pos.dtype) + vol * w0
        num = num.index_add(0, i, (vol[j] * w)[:, None] * out[j])
        den = den.index_add(0, i, vol[j] * w)
        num = num.index_add(0, j, (vol[i] * w)[:, None] * out[i])
        den = den.index_add(0, j, vol[i] * w)
        out = num / den.clamp_min(1e-30)[:, None]
    return out


# --------------------------------------------------------------------------- #
# the 3-D curvature estimator (the piece the repository was missing)
# --------------------------------------------------------------------------- #
def _tangent_basis(normal: torch.Tensor):
    """Orthonormal ``(t1, t2)`` with ``t1 x t2 = normal`` for every particle."""
    helper = torch.zeros_like(normal)
    pick = normal.abs().argmin(dim=1)
    helper[torch.arange(normal.shape[0]), pick] = 1.0
    t1 = torch.cross(helper, normal, dim=1)
    t1 = t1 / torch.linalg.vector_norm(t1, dim=1, keepdim=True).clamp_min(1e-30)
    t2 = torch.cross(normal, t1, dim=1)
    return t1, t2


def fit_curvature3(pos, normal, delta_s, pairs, h, *, band: float = 0.6,
                   stencil: float = 2.5, fallback: torch.Tensor | None = None,
                   min_points: int = 12):
    """Local quadratic surface fit -> mean curvature ``kappa = 2H`` of a 3-D interface.

    Args:
        pos: ``(N, 3)`` positions.
        normal: ``(N, 3)`` outward unit normals (zero away from the interface).
        delta_s: ``(N,)`` surface delta function; only particles with
            ``delta_s > 0`` are fitted, everything else takes ``fallback``.
        pairs: neighbour pairs built with a cutoff of at least ``stencil * h``.
        h: smoothing length.
        band: half-thickness of the fitted slab in units of ``h``.  The CSF
            interface is a shell of thickness ``~2h``; points outside
            ``|w| <= band * h`` are the bulk of the two phases, not the surface.
        stencil: lateral fit radius in units of ``h``.
        fallback: value for degenerate stencils (default 0).  Typical choice is
            the MLS ``div n``.

    Sign convention: ``kappa > 0`` where the interface bends *away* from the
    outward normal (convex droplet), i.e. ``kappa = 2/R`` on a liquid sphere of
    radius ``R`` -- the same convention as ``csf_curvature.fit_curvature`` and
    the same one the CSF force ``a = -sigma kappa delta_s n / rho`` needs.
    """
    n = pos.shape[0]
    kappa = torch.zeros(n, dtype=pos.dtype) if fallback is None else fallback.clone()
    if pairs.numel() == 0:
        return kappa
    t1, t2 = _tangent_basis(normal)

    def _local(centre, partner):
        rel = pos[partner] - pos[centre]
        return ((rel * t1[centre]).sum(dim=1),
                (rel * t2[centre]).sum(dim=1),
                (rel * normal[centre]).sum(dim=1))

    # The pair list is unordered, so each pair is expanded into both orientations:
    # particle i sees j through i's tangent frame, particle j sees i through j's.
    first, second = pairs[:, 0], pairs[:, 1]
    u_a, v_a, w_a = _local(first, second)
    u_b, v_b, w_b = _local(second, first)
    index = torch.cat((first, second))
    u = torch.cat((u_a, u_b))
    v = torch.cat((v_a, v_b))
    w = torch.cat((w_a, w_b))
    r_lat = torch.sqrt(u * u + v * v)
    keep = (w.abs() <= band * h) & (r_lat <= stencil * h)
    keep = keep & (delta_s[index] > 0.0)
    if not bool(keep.any()):
        return kappa
    i = index[keep]
    u, v, w = u[keep], v[keep], w[keep]
    # distance from the centre to the partner (kernel weight)
    partner = torch.cat((second, first))[keep]
    r = torch.linalg.vector_norm(pos[partner] - pos[i], dim=1)
    weight = cubic_spline_kernel(r, h)
    rows = torch.stack((u, v, 0.5 * u * u, u * v, 0.5 * v * v), dim=1)
    lhs = torch.zeros((n, 5, 5), dtype=pos.dtype)
    rhs = torch.zeros((n, 5), dtype=pos.dtype)
    outer = weight[:, None, None] * torch.einsum("pa,pb->pab", rows, rows)
    lhs = lhs.index_add(0, i, outer)
    rhs = rhs.index_add(0, i, weight[:, None] * (rows * w[:, None]))

    # Normal equations are solved with a Tikhonov floor on the diagonal so that
    # a degenerate (planar/short) stencil returns a finite curvature instead of
    # NaN; the stencil count decides whether the result is kept.
    trace = lhs.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    reg = 1e-8 * trace.clamp_min(1e-30)
    eye = torch.eye(5, dtype=pos.dtype).expand_as(lhs)
    lhs = lhs + reg[:, None, None] * eye
    solution = torch.linalg.solve(lhs, rhs.unsqueeze(-1)).squeeze(-1)

    count = torch.zeros(n, dtype=pos.dtype)
    count = count.index_add(0, i, torch.ones_like(w))
    k_fit = -(solution[:, 2] + solution[:, 4])
    ok = count >= min_points
    kappa = torch.where(ok, k_fit, kappa)
    # the fit is only meaningful where the particle actually carries interface
    kappa = torch.where(delta_s > 0.0, kappa, torch.zeros_like(kappa))
    return kappa


def curvature_color_quadric3(pos, color, delta_s, normal, vol, pairs, h, *,
                             fallback: torch.Tensor | None = None,
                             min_points: int = 16):
    """Mean curvature from a local second-order fit of the colour *field*.

    This is the level-set route that survives the CSF shell thickness.  With
    ``n = -grad c / |grad c|`` (outward), the mean curvature of the level set is

        kappa = div n = -[ trace(H) - n^T H n ] / |grad c| ,
        H = Hessian of the colour field,  |grad c| = |g| ,

    which for a radial profile ``c(r)`` reads ``2 c' / (r |c'|) = 2/r`` -- exact
    for a sphere at *any* point of the shell, because the radial profile drops
    out of the ratio.  The 3-D lattice staircase enters only through the local
    polynomial fit of ``(c_j - c_i)`` over the kernel support, not through a
    differenced normal, which is what the shipped ``div n`` route pays.

    A height-function fit in the tangent frame (``fit_curvature3``) cannot do
    this: the shell is ~2h thick, so the fitted ``w(u, v)`` is dominated by the
    shell thickness instead of the curvature sag (measured: ``kappa ~ 0``).
    Both arms are kept so the report can quote the comparison.
    """
    n = pos.shape[0]
    kappa = torch.zeros(n, dtype=pos.dtype) if fallback is None else fallback.clone()
    if pairs.numel() == 0:
        return kappa
    first, second = pairs[:, 0], pairs[:, 1]
    index = torch.cat((first, second))
    partner = torch.cat((second, first))

    rel = (pos[partner] - pos[index]) / h
    dc = color[partner] - color[index]
    r = torch.linalg.vector_norm(pos[partner] - pos[index], dim=1)
    weight = cubic_spline_kernel(r, h) * vol[partner]

    x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]
    # basis of the second-order local model, 0.5 already folded into the
    # quadratic columns so the solution vector is (g, H_xx, H_xy, ...) in
    # units of 1/h and 1/h^2
    rows = torch.stack((x, y, z,
                        0.5 * x * x, x * y, x * z,
                        0.5 * y * y, y * z, 0.5 * z * z), dim=1)
    m = rows.shape[1]
    lhs = torch.zeros((n, m, m), dtype=pos.dtype)
    rhs = torch.zeros((n, m), dtype=pos.dtype)
    outer = weight[:, None, None] * torch.einsum("pa,pb->pab", rows, rows)
    lhs = lhs.index_add(0, index, outer)
    rhs = rhs.index_add(0, index, weight[:, None] * (rows * dc[:, None]))

    trace = lhs.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    reg = 1e-8 * trace.clamp_min(1e-30)
    eye = torch.eye(m, dtype=pos.dtype).expand_as(lhs)
    lhs = lhs + reg[:, None, None] * eye
    sol = torch.linalg.solve(lhs, rhs.unsqueeze(-1)).squeeze(-1)

    grad = sol[:, 0:3] / h
    hess = torch.zeros((n, 3, 3), dtype=pos.dtype)
    hess[:, 0, 0] = sol[:, 3] / h ** 2
    hess[:, 0, 1] = hess[:, 1, 0] = sol[:, 4] / h ** 2
    hess[:, 0, 2] = hess[:, 2, 0] = sol[:, 5] / h ** 2
    hess[:, 1, 1] = sol[:, 6] / h ** 2
    hess[:, 1, 2] = hess[:, 2, 1] = sol[:, 7] / h ** 2
    hess[:, 2, 2] = sol[:, 8] / h ** 2

    g_mag = torch.linalg.vector_norm(grad, dim=1)
    hnn = torch.einsum("pa,pab,pb->p", normal, hess, normal)
    tr = hess[:, 0, 0] + hess[:, 1, 1] + hess[:, 2, 2]
    k = -(tr - hnn) / g_mag.clamp_min(1e-30)

    count = torch.zeros(n, dtype=pos.dtype)
    count = count.index_add(0, index, torch.ones(index.shape[0], dtype=pos.dtype))
    ok = (count >= min_points) & (delta_s > 0.0) & torch.isfinite(k)
    return torch.where(ok, k, kappa)


# --------------------------------------------------------------------------- #
# CSF state on the two-phase 3-D carrier
# --------------------------------------------------------------------------- #
def csf_state3_two_phase(pos, mass, liquid, h, rho0, sigma,
                         curvature_method: str = "fit3",
                         curvature_stencil: float = 2.5,
                         fit_band: float = 0.6,
                         mask_fraction: float = 0.05):
    """Colour / normal / curvature / delta_s / CSF acceleration (3-D)."""
    vol = mass / rho0
    pairs = box_pairs(pos, 2.0 * h)
    color = shepard_color3(pos, liquid, vol, pairs, h)
    grad_c = mls_gradient3(pos, color, vol, pairs, h)
    mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)
    tiny = 1e-14
    normal_raw = torch.where(mag > tiny, -grad_c / mag.clamp_min(tiny),
                             torch.zeros_like(grad_c))
    normal = shepard_smooth3(normal_raw, pos, vol, pairs, h, passes=2)
    norm = torch.linalg.vector_norm(normal, dim=1, keepdim=True)
    normal = torch.where(norm > 1e-8, normal / norm.clamp_min(1e-30), normal_raw)
    delta_s = mag.squeeze(-1)
    active = delta_s > mask_fraction * delta_s.max()

    kappa_mls = mls_divergence3(pos, normal, vol, pairs, h)
    if curvature_method == "mls":
        kappa = kappa_mls
        pairs_fit = pairs
    elif curvature_method == "color_quadric":
        kappa = curvature_color_quadric3(pos, color, delta_s, normal, vol,
                                         pairs, h, fallback=kappa_mls)
        pairs_fit = pairs
    elif curvature_method == "fit3":
        pairs_fit = (pairs if curvature_stencil <= 2.0
                     else box_pairs(pos, curvature_stencil * h))
        kappa = fit_curvature3(pos, normal, delta_s, pairs_fit, h,
                               band=fit_band, stencil=curvature_stencil,
                               fallback=kappa_mls)
    else:
        raise ValueError(f"unknown curvature_method {curvature_method!r}")
    kappa = torch.where(active, kappa, torch.zeros_like(kappa))
    delta_s = torch.where(active, delta_s, torch.zeros_like(delta_s))
    acc = (-sigma * kappa * delta_s)[:, None] * normal / rho0
    return {"color": color, "grad_c": grad_c, "normal": normal,
            "kappa": kappa, "delta_s": delta_s, "acc": acc,
            "pairs": pairs, "pairs_fit": pairs_fit, "vol": vol, "active": active,
            "kappa_mls": kappa_mls}


# --------------------------------------------------------------------------- #
# static projection (pressure Poisson)
# --------------------------------------------------------------------------- #
def _pair_coeffs3(pos, pairs, vol, h, eta):
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    r2 = (delta * delta).sum(dim=1)
    gw = cubic_spline_gradient(delta, h)
    rdot = (delta * gw).sum(dim=1)
    return i, j, -2.0 * vol[j] * rdot / (r2 + eta * eta)


def pressure_poisson3(pos, source, pairs, vol, h, dirichlet,
                      rel_tol: float = 1e-10, max_iter: int = 4000):
    """Solve ``-grad^2 p = source`` in 3-D with ``p = 0`` on ``dirichlet`` (CG)."""
    n = pos.shape[0]
    eta = 0.1 * h
    i, j, coef = _pair_coeffs3(pos, pairs, vol, h, eta)
    fixed = dirichlet

    def apply(p):
        val = coef * (p[i] - p[j])
        out = torch.zeros(n, dtype=p.dtype)
        out = out.index_add(0, i, val)
        out = out.index_add(0, j, -val)
        return torch.where(fixed, p, out)

    rhs = torch.where(fixed, torch.zeros_like(source), source.to(pos.dtype))
    diag = torch.zeros(n, dtype=pos.dtype)
    diag = diag.index_add(0, i, coef)
    diag = diag.index_add(0, j, coef)
    diag = torch.where(fixed, torch.ones_like(diag), diag.clamp_min(1e-30))
    inv = 1.0 / diag

    p = torch.zeros(n, dtype=pos.dtype)
    r = rhs.clone()
    z = inv * r
    d = z.clone()
    rz = (r * z).sum()
    rhs_norm = torch.linalg.vector_norm(rhs).clamp_min(1e-300)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        ad = apply(d)
        denom = (d * ad).sum()
        if float(denom.abs()) <= 1e-300:
            break
        alpha = rz / denom
        p = p + alpha * d
        r = r - alpha * ad
        if float(torch.linalg.vector_norm(r) / rhs_norm) < rel_tol:
            break
        z = inv * r
        rz_new = (r * z).sum()
        d = z + (rz_new / rz) * d
        rz = rz_new
    return p, n_iter


def spurious_acceleration3(pos, acc_csf, p, rho0, pairs, vol, h):
    """Momentum-balance residual ``a_CSF - grad p / rho`` per particle."""
    grad_p = mls_gradient3(pos, p, vol, pairs, h)
    return acc_csf - grad_p / rho0


# --------------------------------------------------------------------------- #
# the case
# --------------------------------------------------------------------------- #
def laplace_isph3d_case(radius: float = 0.02, dx: float = 0.002,
                        sigma: float = 0.072, rho0: float = 1000.0,
                        outer: float = 0.012, h_factor: float = 1.3,
                        curvature_method: str = "fit3",
                        curvature_stencil: float = 2.5, fit_band: float = 0.6,
                        variant: str = "model", core_frac: float = 0.5,
                        band=(1.25, 1.45), rel_tol: float = 1e-10,
                        max_iter: int = 4000, return_fields: bool = False):
    """Projection-SPH Young-Laplace jump of a 3-D droplet against ``dp = 2 sigma/R``.

    ``variant``:
        ``model``   measured curvature, delta function ``|grad c|``
        ``kappa_exact``  analytic ``kappa = 2/R`` (instrument check: isolates the
                    curvature error from the pressure solve)
        ``delta_unit``   measured curvature with ``delta_s`` rescaled so that
                    ``int delta_s dV = 4 pi R^2`` (isolates the delta error)
        ``gauge``   both replaced (instrument check: must be within a few %)
    """
    h = h_factor * dx
    pos, mass, liquid = sphere_block(radius, dx, outer, rho0)
    st = csf_state3_two_phase(pos, mass, liquid, h, rho0, sigma,
                             curvature_method=curvature_method,
                             curvature_stencil=curvature_stencil,
                             fit_band=fit_band)
    vol, pairs, active = st["vol"], st["pairs"], st["active"]
    delta_s, kappa, normal = st["delta_s"], st["kappa"], st["normal"]

    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r.clamp_min(1e-30)[:, None]
    sign = torch.sign((normal * rhat).sum(dim=1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    kappa_eff = 2.0 * sign / radius          # dp = 2 sigma / R

    surface = 4.0 * math.pi * radius ** 2
    ds_integral = float((delta_s * vol).sum() / surface)

    if variant in ("kappa_exact", "gauge"):
        kappa_use = torch.where(active, kappa_eff, torch.zeros_like(kappa))
    else:
        kappa_use = kappa
    if variant == "delta_unit":
        delta_use = delta_s * (1.0 / max(ds_integral, 1e-12))
    elif variant == "gauge":
        w = delta_s * vol
        delta_use = torch.where(w > 0.0, w / w.sum() * surface / vol,
                                torch.zeros_like(delta_s))
    else:
        delta_use = delta_s

    acc = (-sigma * kappa_use * delta_use)[:, None] * normal / rho0
    div_a = mls_divergence3(pos, acc, vol, pairs, h)
    source = -rho0 * div_a

    half = radius + outer
    edge = 1.6 * dx
    dirichlet = ((pos[:, 0].abs() > half - edge) | (pos[:, 1].abs() > half - edge)
                 | (pos[:, 2].abs() > half - edge))
    p, n_cg = pressure_poisson3(pos, source, pairs, vol, h, dirichlet,
                                rel_tol=rel_tol, max_iter=max_iter)

    core = r < core_frac * radius
    out_band = (r > band[0] * radius) & (r < band[1] * radius)
    p_core = float(p[core].mean())
    p_out = float(p[out_band].mean()) if bool(out_band.any()) else 0.0
    dp = p_core - p_out

    resid = spurious_acceleration3(pos, acc, p, rho0, pairs, vol, h)
    resid_norm = torch.linalg.vector_norm(resid[active], dim=1)
    a_scale = float(torch.linalg.vector_norm(acc[active], dim=1).mean())

    a_r = (acc * rhat).sum(dim=1)
    flux = float((-a_r * mass).sum() / surface)
    v_ref = math.sqrt(sigma / (rho0 * radius))

    dp_theory = 2.0 * sigma / radius
    safe = dp_theory if dp_theory > 0.0 else float("nan")
    kappa_w = float((kappa_use * delta_use * vol).sum()
                    / (delta_use * vol).sum())
    return {
        "variant": variant,
        "dx_m": dx, "h_m": h, "R_over_h": radius / h, "R_over_dx": radius / dx,
        "n_particles": int(pos.shape[0]), "n_liquid": int(liquid.sum()),
        "n_active": int(active.sum()),
        "curvature_method": curvature_method,
        "fit_band": fit_band, "curvature_stencil": curvature_stencil,
        "delta_s_integral_over_4pir2": ds_integral,
        "kappa_delta_weighted": kappa_w,
        "kappa_analytic": 2.0 / radius,
        "kappaR_delta_weighted": kappa_w * radius,
        "kappa_mlsR_delta_weighted": float(
            (st["kappa_mls"] * st["delta_s"] * vol).sum()
            / (st["delta_s"] * vol).sum() * radius),
        "p_core_Pa": p_core, "p_vapour_Pa": p_out,
        "dp_measured_Pa": dp, "dp_theory_Pa": dp_theory,
        "dp_over_theory": dp / safe,
        "dp_error_pct": (dp / safe - 1.0) * 100.0,
        "dp_flux_Pa": flux, "dp_flux_over_theory": flux / safe,
        "residual_median_over_mean_csf_acc": float(resid_norm.median() / a_scale),
        "residual_bulk_median_over_mean_csf_acc": float(
            torch.linalg.vector_norm(resid[~active], dim=1).median() / a_scale)
        if bool((~active).any()) else 0.0,
        "v_ref_m_per_s": v_ref,
        "cg_iterations": int(n_cg),
        **({"fields": {"pos": pos, "p": p, "kappa": kappa_use,
                       "delta_s": delta_use, "active": active,
                       "normal": normal}}
           if return_fields else {}),
    }