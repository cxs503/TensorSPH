"""Adjoint-consistent projection (ISPH) operators and a stable Young-Laplace relaxation.

Why this module exists
----------------------
``laplace_isph.py`` (m17086) measured the static Young-Laplace jump with a
projection-SPH pressure and got it to **< 1 %** on a resolution ladder, but its
*time marching* diverged after 13 steps: the recorded honest FAIL in
``docs/verification_laplace_sph_isph.md`` §5.1.  The module header there names
the cure that was missing.  This module supplies it.

The defect, stated exactly
--------------------------
A projection step is only a projection if the divergence it measures and the
gradient it corrects with are **the same operator**:

    u^{n+1} = u* - (dt/rho) G p ,      A p = -(rho/dt) D u* ,   A = D D^T .

If ``G`` is not ``-D^T`` (a *discrete* adjoint pair, Euclidean transpose), then
``D u^{n+1} = D u* + (dt/rho) D G p`` does **not** vanish for any ``p``, and the
leftover divergence is re-injected every step.  Measured for the shipped pair
of MLS operators (``csf_consistent.mls_divergence`` / ``mls_gradient``, which
are first-order *consistent* but not mutually adjoint) on the R/dx = 10 drop:

    <D u, q> = +1.3005e-03   vs   -<u, G q> = +2.0231e-02   (off by a factor 15.6)

So the pressure solve of ``laplace_isph.relax_projection`` can only remove a
small part of the divergence it created, and the rest accumulates in ``u``
until the lattice breaks (the 13-step divergence).  ``tests/test_projection_adjoint.py``
pins the corrected pair: the adjoint defect is **1e-16** (round-off), and the
post-projection discrete divergence is at the same level.

Construction (no new discretisation, only a new pairing)
--------------------------------------------------------
Difference divergence on the unordered pair list (``i < j``), ``g_ij = grad_i W_ij``:

    D u|_i = sum_j V_j (u_j - u_i) . g_ij

The Euclidean transpose is *derived*, not guessed (``tests`` re-derive it by
finite differences on the quadratic form ``<D u, p>``):

    (D^T p)|_i = sum_{pairs (i,j)}  (-1 if i is the first index else +1)
                 * (V_j p_i + V_i p_j) g_ij

and the gradient used by the correction is ``G = -D^T`` (verified to be aligned
with the analytic ``grad p`` on an interior stencil).  ``A = D D^T`` is
symmetric positive semi-definite by construction; one particle is pinned
(``p = 0``) to remove the constant null mode, which is legitimate because only
``grad p`` enters the velocity correction.

Second ingredient: damping.  Even a *consistent* projection cannot remove the
solenoidal part of the lattice CSF force, which is what drives spurious
currents.  A physical (not artificial) viscosity ``nu`` is therefore available
as a knob, discretised with the same pair Laplacian, and every run reports
``max|u| / sqrt(sigma/(rho R))`` against Adami et al. (2012)'s ``< 0.1``
admissibility line, plus the measured radius drift (a spurious inward force
destroys the area first).

References
    Cummins & Rudman (1999), J. Comput. Phys. 152:584  (projection ISPH).
    Shao & Lo (2003), Adv. Water Resour. 26:787       (ISPH pressure BC).
    Adami, Hu & Adams (2012), J. Comput. Phys. 231:7057 (spurious currents).
    Monaghan (1992), Annu. Rev. Astron. Astrophys. 30:543 (viscosity operator).
"""
from __future__ import annotations

import math

import torch

from .csf_consistent import cubic_spline_gradient
from .laplace_isph import csf_state_two_phase, hex_block
from .neighbor_search import cell_linked_pairs

__all__ = [
    "pair_divergence",
    "adjoint_gradient",
    "adjoint_laplacian",
    "adjointness_defect",
    "poisson_adjoint",
    "project_velocity",
    "relax_projection_consistent",
]


# --------------------------------------------------------------------------- #
# the operator triple: D, D^T, A = D D^T
# --------------------------------------------------------------------------- #
def _pair_geometry(pos, pairs, h):
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    gw = cubic_spline_gradient(delta, h)
    return i, j, gw


def pair_divergence(pos, u, vol, pairs, h):
    """``D u|_i = sum_j V_j (u_j - u_i) . grad_i W_ij`` on unordered pairs.

    A pair ``(i, j)`` contributes to ``i`` with ``V_j`` and to ``j`` with
    ``V_i`` -- the partner's volume weights the partner's velocity.
    """
    n = pos.shape[0]
    if pairs.numel() == 0:
        return torch.zeros(n, dtype=pos.dtype, device=pos.device)
    i, j, gw = _pair_geometry(pos, pairs, h)
    du = ((u[j] - u[i]) * gw).sum(dim=1)
    out = torch.zeros(n, dtype=u.dtype, device=u.device)
    out = out.index_add(0, i, vol[j] * du)
    out = out.index_add(0, j, vol[i] * du)
    return out


def adjoint_gradient(pos, p, vol, pairs, h):
    """``G p = -D^T p`` -- the exact Euclidean adjoint of :func:`pair_divergence`.

    Derived from ``<D u, p> = sum_pairs (V_j p_i + V_i p_j) (u_j - u_i) . g_ij``
    by differentiating w.r.t. ``u``, so ``<D u, p> = <u, D^T p>`` holds to
    round-off (asserted in the tests).  Sign convention: ``G p ~ grad p``.
    """
    n = pos.shape[0]
    if pairs.numel() == 0:
        return torch.zeros((n, pos.shape[1]), dtype=pos.dtype, device=pos.device)
    i, j, gw = _pair_geometry(pos, pairs, h)
    coef = vol[j] * p[i] + vol[i] * p[j]
    contrib = coef[:, None] * gw
    out = torch.zeros((n, pos.shape[1]), dtype=p.dtype, device=p.device)
    # d/du_j = +contrib ; d/du_i = -contrib ; G = -D^T flips both signs.
    out = out.index_add(0, i, contrib)
    out = out.index_add(0, j, -contrib)
    return out


def adjoint_laplacian(pos, p, vol, pairs, h):
    """``(D G) p`` with ``G = -D^T``: symmetric **negative** semi-definite.

    ``-adjoint_laplacian`` is ``D D^T``, the positive operator the projection
    solves against; measured sign on a random field is pinned by the tests
    (``<p, lap p> < 0``), because a flipped sign is exactly the failure mode
    that makes a projection double the divergence instead of removing it.
    """
    return pair_divergence(pos, adjoint_gradient(pos, p, vol, pairs, h),
                           vol, pairs, h)


def adjointness_defect(pos, vol, pairs, h, seed: int = 0) -> float:
    """``|<D u, q> - <u, D^T q>| / |<D u, q>|`` -- instrument self-test.

    The shipped MLS pair returns O(1) here (measured 15.6 on the R/dx = 10
    drop); the pair in this module returns round-off.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = pos.shape[0]
    u = torch.randn((n, pos.shape[1]), generator=g, dtype=pos.dtype)
    q = torch.randn((n,), generator=g, dtype=pos.dtype)
    lhs = (pair_divergence(pos, u, vol, pairs, h) * q).sum()
    rhs = (u * (-adjoint_gradient(pos, q, vol, pairs, h))).sum()
    return float((lhs - rhs).abs() / lhs.abs().clamp_min(1e-300))


def poisson_adjoint(pos, rhs, vol, pairs, h, gauge: int = 0,
                    rel_tol: float = 1e-10, max_iter: int = 20000):
    """Conjugate gradient on ``A p = rhs`` with ``A = D D^T`` and ``p[gauge] = 0``.

    The pinned particle is implemented with a Jacobi preconditioner whose
    inverse is zeroed there and a projected residual, so the iterate never
    leaves the gauge slice; ``A`` is SPD on that slice.
    """
    n = pos.shape[0]
    fixed = torch.zeros(n, dtype=torch.bool)
    fixed[gauge] = True

    def apply(q):
        # A = -D G = D D^T is the positive (Laplacian-like) operator; the CG
        # below needs positive definiteness, so the sign is flipped here.
        return -adjoint_laplacian(pos, q, vol, pairs, h)

    # Jacobi diagonal by probing the operator on the unit basis is too costly;
    # use the pair-coefficient estimate instead: diag ~ sum_j (V_j+V_i)|g|^2-like
    i, j, gw = _pair_geometry(pos, pairs, h)
    w = (gw * gw).sum(dim=1)
    diag = torch.zeros(n, dtype=pos.dtype)
    diag = diag.index_add(0, i, (vol[i] + vol[j]) * w)
    diag = diag.index_add(0, j, (vol[i] + vol[j]) * w)
    diag = diag.clamp_min(1e-30)
    inv = torch.where(fixed, torch.zeros_like(diag), 1.0 / diag)

    rhs = torch.where(fixed, torch.zeros_like(rhs), rhs)
    p = torch.zeros(n, dtype=pos.dtype)
    r = rhs.clone()
    z = inv * r
    d = z.clone()
    rz = (r * z).sum()
    norm0 = torch.linalg.vector_norm(rhs).clamp_min(1e-300)
    it = 0
    for it in range(1, max_iter + 1):
        ad = torch.where(fixed, torch.zeros_like(d), apply(d))
        denom = (d * ad).sum()
        if float(denom.abs()) <= 0.0:
            break
        alpha = rz / denom
        p = p + alpha * d
        r = r - alpha * ad
        if float(torch.linalg.vector_norm(r) / norm0) < rel_tol:
            break
        z = inv * r
        rz_new = (r * z).sum()
        if float(rz.abs()) <= 0.0:
            break
        d = z + (rz_new / rz) * d
        rz = rz_new
    return p, it


def project_velocity(pos, u_star, vol, pairs, h, rho, dt, gauge: int = 0,
                     rel_tol: float = 1e-10, max_iter: int = 20000):
    """One exact projection step: return ``(u, p, info)`` with ``D u ~ 0``."""
    div_star = pair_divergence(pos, u_star, vol, pairs, h)
    rhs = -(rho / dt) * div_star
    p, it = poisson_adjoint(pos, rhs, vol, pairs, h, gauge=gauge,
                            rel_tol=rel_tol, max_iter=max_iter)
    u = u_star - (dt / rho) * adjoint_gradient(pos, p, vol, pairs, h)
    div_new = pair_divergence(pos, u, vol, pairs, h)
    # The gauge particle is the one row the pinned system does not enforce, so
    # it is excluded from the ratio (it is also a velocity-pinned particle in
    # ``relax_projection_consistent``).  Keeping it in would report a 1e-2
    # "leftover divergence" that belongs to a single fixed particle.
    keep = torch.ones(pos.shape[0], dtype=torch.bool)
    keep[gauge] = False
    scale = torch.linalg.vector_norm(div_star[keep]).clamp_min(1e-300)
    return u, p, {
        "cg_iterations": int(it),
        "gauge_index": int(gauge),
        "div_star_rms": float(torch.sqrt((div_star ** 2).mean())),
        "div_after_over_div_before": float(
            torch.linalg.vector_norm(div_new[keep]) / scale),
    }


# --------------------------------------------------------------------------- #
# time marching
# --------------------------------------------------------------------------- #
def _viscous_acceleration(pos, u, vol, pairs, h, nu):
    """``nu * lap(u)`` with the pair Laplacian (component-wise)."""
    if nu <= 0.0:
        return torch.zeros_like(u)
    lap = adjoint_laplacian(pos, u[:, 0], vol, pairs, h)
    lap_y = adjoint_laplacian(pos, u[:, 1], vol, pairs, h)
    return nu * torch.stack([lap, lap_y], dim=1)


def relax_projection_consistent(radius: float = 0.02, dx: float = 0.002,
                               sigma: float = 0.072, rho0: float = 1000.0,
                               outer: float = 0.012, h_factor: float = 1.3,
                               curvature_method: str = "mls",
                               steps: int = 200, dt_factor: float = 0.1,
                               sample_every: int = 25,
                               nu_factor: float = 0.0,
                               pin_outer: bool = True,
                               rel_tol: float = 1e-6,
                               cg_max_iter: int = 4000):
    """March projection SPH on a static drop; a correct equilibrium must not move.

    ``nu_factor`` sets ``nu = nu_factor * sqrt(sigma R / rho)`` (a *physical*
    viscosity; ``nu_factor = 0`` is inviscid).  ``pin_outer`` pins the velocity
    of the outer carrier layer to zero: the carrier block has no ghost particles,
    so its boundary layer sees a truncated kernel and would otherwise act as a
    momentum source -- the same reason ``laplace_isph`` put the Dirichlet gauge
    there.

    Reported per sample: ``max|u|/v_ref`` and ``mean|u|/v_ref`` with
    ``v_ref = sqrt(sigma/(rho R))`` (Adami et al. 2012 gate 0.1), the area ratio
    of the liquid label, the measured ``Delta p`` and the post-projection
    discrete divergence ratio.  ``rel_tol = 1e-6`` on the CG residual is the
    default: it leaves a residual divergence of 1e-6 of the pre-projection one,
    five decades below the solenoidal CSF residual that actually limits the run,
    for about a third of the cost of 1e-10.  ``cg_max_iter`` caps the CG so that
    a run whose lattice is already broken (the inviscid control) degrades into a
    reported leftover divergence instead of hanging in a 20 000-iteration
    solve; the per-sample ``div_after_over_div_before`` makes that visible.
    """
    h = h_factor * dx
    pos, mass, liquid, half = hex_block(radius, dx, outer, rho0)
    vol = mass / rho0
    n = pos.shape[0]
    t_sigma = math.sqrt(rho0 * h ** 3 / max(sigma, 1e-30))
    dt = dt_factor * t_sigma
    v_ref = math.sqrt(sigma / (rho0 * radius))
    nu = nu_factor * math.sqrt(sigma * radius / rho0)
    edge = 1.6 * dx

    r0 = torch.linalg.vector_norm(pos, dim=1)
    p0 = torch.linalg.vector_norm(pos, dim=1)
    area_scale = dx * dx * math.sqrt(3.0) / 2.0
    area0 = float((r0 < radius).sum()) * area_scale

    u = torch.zeros_like(pos)
    history = []
    pinned = torch.zeros(n, dtype=torch.bool)
    if pin_outer:
        pinned = ((pos[:, 0].abs() > half - edge)
                  | (pos[:, 1].abs() > half - edge))
    # Gauge particle: a pinned carrier corner (p = 0 there is the ambient gauge).
    gauge = int(torch.nonzero(pinned)[0, 0]) if bool(pinned.any()) else 0
    diverged = False

    for step in range(1, steps + 1):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        st = csf_state_two_phase(pos, mass, liquid, h, rho0, sigma,
                                 curvature_method=curvature_method)
        a = st["acc"] + _viscous_acceleration(pos, u, vol, pairs, h, nu)
        u_star = u + dt * a
        u, p, info = project_velocity(pos, u_star, vol, pairs, h, rho0, dt,
                                      gauge=gauge, rel_tol=rel_tol,
                                      max_iter=cg_max_iter)
        if bool(pinned.any()):
            u = torch.where(pinned[:, None], torch.zeros_like(u), u)
        pos = pos + dt * u
        vmax = float(torch.linalg.vector_norm(u, dim=1).max())
        if (not bool(torch.isfinite(u).all())) or vmax > 1.0e3 * v_ref:
            history.append({"step": step, "diverged": True,
                            "umax_over_vref": vmax / v_ref})
            diverged = True
            break
        if step % sample_every == 0 or step == steps:
            speed = torch.linalg.vector_norm(u, dim=1)
            r_now = torch.linalg.vector_norm(pos, dim=1)
            area = float((r_now < radius).sum()) * area_scale
            core = r_now < 0.5 * radius
            band = (r_now > 1.25 * radius) & (r_now < 1.45 * radius)
            history.append({
                "step": step,
                "t_over_t_sigma": step * dt / t_sigma,
                "umax_over_vref": float(speed.max() / v_ref),
                "umean_over_vref": float(speed.mean() / v_ref),
                "dp_Pa": float(p[core].mean() - p[band].mean()) if bool(band.any())
                          else float("nan"),
                "area_ratio": area / area0,
                "div_after_over_div_before": info["div_after_over_div_before"],
                "cg_iterations": info["cg_iterations"],
            })

    last = history[-1] if history else {}
    dp_theory = sigma / radius
    dp = float(last.get("dp_Pa", float("nan")))
    return {
        "radius_m": radius, "dx_m": dx, "h_m": h, "dt_s": dt,
        "R_over_dx": radius / dx, "R_over_h": radius / h,
        "n_particles": int(n), "n_liquid": int(liquid.sum()),
        "sigma": sigma, "rho0": rho0, "nu_m2_per_s": nu,
        "nu_factor": nu_factor, "pin_outer": bool(pin_outer),
        "curvature_method": curvature_method,
        "v_ref_m_per_s": v_ref, "diverged": diverged,
        "steps_completed": len([x for x in history if not x.get("diverged")]),
        "umax_over_vref_final": float(last.get("umax_over_vref", float("nan"))),
        "umean_over_vref_final": float(last.get("umean_over_vref", float("nan"))),
        "area_ratio_final": float(last.get("area_ratio", float("nan"))),
        "dp_final_Pa": dp,
        "dp_theory_Pa": dp_theory,
        "dp_rel_error_pct": (dp / dp_theory - 1.0) * 100.0,
        "history": history,
    }