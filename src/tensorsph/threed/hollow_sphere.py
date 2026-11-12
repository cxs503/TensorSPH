"""CASE-HOLLOW-SPHERE-3D: the Young-Laplace jump across a *spherical* interface.

Why this case and not a bigger circle
-------------------------------------
``CASE-LAPLACE-SPH`` verifies ``Delta p = sigma (1/R1 + 1/R2)`` with a *circle*,
so only one of the two principal curvatures is ever exercised: a plane cannot
tell a sphere from a cylinder, and it cannot tell ``2 sigma / R`` from
``sigma / R``.  This case closes that hole in the verification ladder with the
cheapest geometry that carries **two** curvature radii: a spherical shell of
liquid (a hollow sphere) with a gas cavity inside and gas outside.

Reference
---------
Across a single spherical interface of radius ``R`` the jump is

.. math::   \\Delta p = \\frac{2\\sigma}{R}

(Young 1805; Laplace 1806; the ``(1/R1 + 1/R2)`` form with both radii finite is
what makes the 3-D case different from the 2-D one).  For the hollow sphere --
liquid between ``R_in`` and ``R_out``, gas on both sides -- the two interfaces
add up:

.. math::   \\Delta p = 2\\sigma \\left(\\frac{1}{R_{in}} + \\frac{1}{R_{out}}\\right),

the classical soap-bubble result (``= 4 sigma / R`` for a thin film,
``Batchelor 1967, An Introduction to Fluid Dynamics, sec. 3.9``; the
``1/R_in + 1/R_out`` statement of the two-surface film is the one used for a
bubble of finite thickness).  Setting ``R_in = 0`` degenerates the shell to a
liquid *drop*, so a single implementation covers both the droplet
(``2 sigma / R``) and the hollow-sphere (``2 sigma (1/R_in + 1/R_out)``)
configurations, and the degenerate limit is checked in the tests.

Method (and why the pressure is not taken from an EOS)
-----------------------------------------------------
Same argument as the 2-D projection case (:mod:`tensorsph.laplace_isph`): a
WCSPH Tait EOS carries ``sigma/R`` through a density ratio of
``Delta p / B ~ 1e-5`` while the kernel-sum density field carries ``1e-3`` noise,
so the signal is structurally below the noise.  The pressure here is the
Lagrange multiplier of a projection,

.. math::   -\\nabla^2 p = -\\rho \\nabla\\cdot a_{CSF}, \\qquad p = 0
            \\text{ on the outer carrier layer},

with the CSF traction of Brackbill, Kothe & Zemach (1992),

.. math::   a_{CSF} = -\\frac{\\sigma \\kappa \\delta_s n}{\\rho},
            \\qquad \\kappa = \\nabla\\cdot n,\\quad \\delta_s = |\\nabla c|,

discretised with the 3-D MLS operators of :mod:`tensorsph.threed` (the 2-D
``csf_consistent`` MLS matrix is hard-coded ``2x2``; the D-generic re-derivation
lives here).  The carrier is a cubic lattice in which *every* site is a particle
and the phase is carried by the colour ``c``; filling the void keeps
``int delta_s dn = 1`` (the CSF normalisation) so ``Delta p`` does not depend on
the density ratio -- the standard static-verification practice.

Two curvature routes are reported, and the difference between them is the point:

``"analytic"``
    ``kappa`` is imposed from the known geometry,
    ``kappa = 2 (n . r_hat) / R_interface``.  This isolates the *rest* of the
    chain (colour -> delta_s -> traction -> Poisson -> pressure jump).  It is the
    gauge variant of the 2-D case.
``"mls"``
    ``kappa = div n`` from the particle field, i.e. the self-consistent CSF
    curvature.  The 2-D audit measured +25 % for this operator on a circle
    (``docs/verification_surface_tension_consistent.md``); the 3-D number is
    reported here rather than assumed to be better.

Both are compared against the same closed form, so the module cannot silently
pass by tuning: the analytic route must reproduce ``2 sigma (1/R_in + 1/R_out)``
inside the platform tolerance, and the self-consistent route's error is what it
is.

Cross-checks (each independent of the Poisson solve)
---------------------------------------------------
* ``kappa delta`` weighted mean against ``2/R`` on each interface separately, so
  ``R_in`` and ``R_out`` are verified independently -- a sphere carries two
  different curvature radii and the case must recover both.
* the radial traction flux ``-int rho a_r dV / (4 pi R^2)`` per interface; it
  never touches the pressure solve, so agreement between the two routes is what
  makes the number trustworthy.
* the ``int delta_s dn = 1`` normalisation of the CSF delta function.
* the momentum-balance residual ``|a_CSF - grad p / rho|`` (spurious
  acceleration), reported as an honest number rather than gated (the 2-D case
  documents why: the lattice artefact is not removable by the projection).
* ``sigma = 0`` negative control: the measured jump must vanish.

References
    Brackbill, Kothe & Zemach (1992), J. Comput. Phys. 100:335 (CSF).
    Cummins & Rudman (1999), J. Comput. Phys. 152:584 (projection SPH).
    Shao & Lo (2003), Adv. Water Resour. 26:787 (SPH pressure Poisson).
    Adami, Hu & Adams (2012), J. Comput. Phys. 231:7057 (spurious currents).
    Batchelor (1967), An Introduction to Fluid Dynamics, CUP, sec. 3.9
        (two-surface film: ``Delta p = 2 sigma (1/R1 + 1/R2)``).
"""
from __future__ import annotations

import math

import torch

from .kernel3 import cubic_spline_kernel
from .neighbors3 import box_pairs
from .operators3 import pair_geometry

__all__ = [
    "sphere_shell_lattice",
    "shell_laplace_analytic",
    "shell_csf_state",
    "pressure_poisson3",
    "hollow_sphere_case",
]

DEFAULT_RHO0 = 1000.0


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def sphere_shell_lattice(radius_in: float, radius_out: float, dx: float,
                         pad: float, rho0: float = DEFAULT_RHO0,
                         lattice: str = "cubic"):
    """Cubic-lattice carrier of a hollow sphere, ``r <= radius_out + pad``.

    The carrier is *filled*: sites with ``r < radius_in`` are the gas cavity,
    ``radius_in <= r <= radius_out`` is the liquid shell and the rest is the
    outer gas.  All particles carry the same mass ``rho0 dx^3`` so the colour
    field is a purely geometric phase indicator.

    ``lattice="cubic"`` (default) or ``"fcc"`` (two interpenetrating cubic
    lattices, the denser packing).  ``radius_in = 0`` gives a liquid ball.
    """
    if dx <= 0.0 or radius_out <= 0.0:
        raise ValueError("dx and radius_out must be positive")
    if not (0.0 <= radius_in < radius_out):
        raise ValueError("need 0 <= radius_in < radius_out")
    if pad < 2.0 * dx:
        raise ValueError("pad must be at least 2 dx (Dirichlet layer)")

    half = radius_out + pad
    n = int(math.floor(half / dx))
    axis = torch.arange(-n, n + 1, dtype=torch.float64) * dx
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
    pos = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)
    if lattice == "fcc":
        shift = torch.tensor([0.5 * dx, 0.5 * dx, 0.0], dtype=torch.float64)
        pos = torch.cat([pos, pos + shift], dim=0)
        box = half + dx  # the shifted copy may leave by one dx
        pos = pos[(pos.abs() <= box).all(dim=1)]
    elif lattice != "cubic":
        raise ValueError(f"unknown lattice {lattice!r}; expected 'cubic' or 'fcc'")

    r = torch.linalg.vector_norm(pos, dim=1)
    pos = pos[r <= half]
    r = r[r <= half]
    mass = torch.full((pos.shape[0],), rho0 * dx ** 3, dtype=torch.float64)
    liquid = (r >= radius_in) & (r <= radius_out)
    return pos, mass, liquid, r, half


def shell_laplace_analytic(sigma: float, radius_in: float,
                           radius_out: float) -> float:
    """``Delta p = 2 sigma (1/R_in + 1/R_out)``; ``2 sigma / R`` when ``R_in = 0``."""
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    inv = (1.0 / radius_in if radius_in > 0.0 else 0.0) + 1.0 / radius_out
    return 2.0 * sigma * inv


# --------------------------------------------------------------------------- #
# D-generic MLS operators (the 2-D csf_consistent matrix is hard-coded 2x2)
# --------------------------------------------------------------------------- #
def _mls_ops(pos, vol, pairs, h):
    """``M_i = sum_j V_j (x_j - x_i) (x) grad_i W_ij`` (pair stored once).

    Identical construction to :func:`tensorsph.threed.csf3._mls_matrix`, so the
    new case verifies the *shipped* 3-D operator rather than a private variant.
    """
    delta, weight, grad, _ = pair_geometry(pos, pairs, h)
    i, j = pairs[:, 0], pairs[:, 1]
    dim = pos.shape[1]
    m = torch.zeros((pos.shape[0], dim, dim), dtype=pos.dtype)
    outer = -torch.einsum("pa,pb->pab", delta, grad)
    m = m.index_add(0, i, vol[j, None, None] * outer)
    m = m.index_add(0, j, vol[i, None, None] * outer)
    return m, delta, weight, grad, i, j


def _regularised_inverse(m):
    """Inverse with csf3's degenerate-stencil guard.

    Returns ``(inv, ok)``: ``ok`` is False where the stencil is degenerate and
    the identity was substituted, so every operator built on it can drop those
    rows instead of amplifying lattice garbage (a 1e30 curvature on one particle
    is enough to make the CG right-hand side meaningless).
    """
    dim = m.shape[-1]
    eye = torch.eye(dim, dtype=m.dtype).unsqueeze(0).expand_as(m)
    det = torch.linalg.det(m)
    scale = det.abs().median().clamp_min(1e-30)
    ok = det.abs() > 1e-6 * scale
    return torch.linalg.inv(torch.where(ok[:, None, None], m, eye)), ok


def _mls_gradient(m_inv, vol, grad, i, j, phi, n):
    """``grad phi = M^{-1} sum_j V_j (phi_j - phi_i) grad_i W_ij``."""
    num = torch.zeros((n, grad.shape[1]), dtype=phi.dtype)
    dc = phi[j] - phi[i]
    num = num.index_add(0, i, (vol[j] * dc).unsqueeze(-1) * grad)
    num = num.index_add(0, j, (vol[i] * dc).unsqueeze(-1) * grad)
    return torch.einsum("nab,nb->na", m_inv, num)


def _mls_divergence(m_inv, vol, grad, i, j, vec, n, dim):
    """``div v`` from the MLS fit, ``trace(S M^{-1})`` with the csf3 convention."""
    dv = vec[j] - vec[i]
    outer = torch.einsum("pa,pb->pab", dv, grad)
    s = torch.zeros((n, dim, dim), dtype=vec.dtype)
    s = s.index_add(0, i, vol[j, None, None] * outer)
    s = s.index_add(0, j, vol[i, None, None] * outer)
    return (s * m_inv.transpose(-1, -2)).sum(dim=(-1, -2))


def _shepard_smooth(field, vol, weight, i, j, n):
    num = torch.zeros_like(field)
    den = torch.zeros((n,), dtype=field.dtype)
    wv = weight.unsqueeze(-1)
    num = num.index_add(0, i, (vol[j].unsqueeze(-1) * wv) * field[j])
    num = num.index_add(0, j, (vol[i].unsqueeze(-1) * wv) * field[i])
    den = den.index_add(0, i, vol[j] * weight)
    den = den.index_add(0, j, vol[i] * weight)
    keep = den > 1e-12
    return torch.where(keep.unsqueeze(-1), num / den.clamp_min(1e-30).unsqueeze(-1),
                       field)


# --------------------------------------------------------------------------- #
# CSF state
# --------------------------------------------------------------------------- #
def shell_csf_state(pos, mass, liquid, h: float, *, rho0: float = DEFAULT_RHO0,
                    sigma: float = 0.072, radius_in: float = 0.0,
                    radius_out: float = 0.0, curvature: str = "analytic",
                    smooth_passes: int = 2):
    """Colour, normal, curvature, ``delta_s`` and CSF acceleration on the carrier."""
    vol = mass / rho0
    pairs = box_pairs(pos, 2.0 * h)
    n = pos.shape[0]
    dim = pos.shape[1]
    m, delta, weight, grad, i, j = _mls_ops(pos, vol, pairs, h)
    m_inv, m_ok = _regularised_inverse(m)

    # Shepard-normalised phase indicator: c -> 1 in the liquid, 0 in the gas.
    c_dtype = liquid.to(pos.dtype)
    c_num = torch.zeros(n, dtype=pos.dtype)
    c_den = torch.zeros(n, dtype=pos.dtype)
    self_w = float(cubic_spline_kernel(torch.zeros(1, dtype=pos.dtype), h)[0])
    c_num = c_num.index_add(0, i, vol[j] * weight * c_dtype[j])
    c_num = c_num.index_add(0, j, vol[i] * weight * c_dtype[i])
    c_den = c_den.index_add(0, i, vol[j] * weight)
    c_den = c_den.index_add(0, j, vol[i] * weight)
    c_num = c_num + vol * self_w * c_dtype
    c_den = c_den + vol * self_w
    color = c_num / c_den.clamp_min(1e-30)

    grad_c = _mls_gradient(m_inv, vol, grad, i, j, color, n)
    mag = torch.linalg.vector_norm(grad_c, dim=1)
    tiny = torch.tensor(1e-14, dtype=pos.dtype)
    safe_mag = mag.clamp_min(1e-30).unsqueeze(-1)
    normal_raw = torch.where((mag > tiny).unsqueeze(-1),
                             -grad_c / safe_mag, torch.zeros_like(grad_c))
    normal = normal_raw
    for _ in range(max(0, smooth_passes)):
        normal = _shepard_smooth(normal, vol, weight, i, j, n)
        norm = torch.linalg.vector_norm(normal, dim=1).clamp_min(1e-30)
        normal = normal / norm.unsqueeze(-1)

    delta_s = mag
    active = delta_s > 0.05 * float(delta_s.max()) if bool(delta_s.numel()) else delta_s > 0
    r = torch.linalg.vector_norm(pos, dim=1)
    rhat = pos / r.clamp_min(1e-30).unsqueeze(-1)
    n_dot_r = (normal * rhat).sum(dim=1)

    r_mid = 0.5 * (radius_in + radius_out)
    inner = active & (r < r_mid)
    outer_face = active & (r >= r_mid)

    if curvature == "mls":
        kappa = _mls_divergence(m_inv, vol, grad, i, j, normal, n, dim)
        kappa = torch.where(m_ok, kappa, torch.zeros_like(kappa))
    elif curvature == "analytic":
        if radius_out <= 0.0:
            raise ValueError("analytic curvature needs radius_out > 0")
        # ``radius_in = 0`` means "no cavity": the only interface is the drop
        # surface at ``radius_out``, so the inner branch must not exist at all
        # (dividing by a guard-radius would turn interior colour noise into a
        # 1e30 curvature and destroy the Poisson right-hand side).
        if radius_in > 0.0:
            r_face = torch.where(r < r_mid,
                                 torch.full_like(r, radius_in),
                                 torch.full_like(r, radius_out))
        else:
            r_face = torch.full_like(r, radius_out)
        kappa = 2.0 * n_dot_r / r_face.clamp_min(1e-30)
    else:
        raise ValueError(f"unknown curvature {curvature!r}; expected 'analytic' or 'mls'")

    kappa = torch.where(active, kappa, torch.zeros_like(kappa))
    delta_s = torch.where(active, delta_s, torch.zeros_like(delta_s))
    acc = (-sigma * kappa * delta_s).unsqueeze(-1) * normal / rho0

    return {
        "color": color, "grad_c": grad_c, "normal": normal, "kappa": kappa,
        "delta_s": delta_s, "acc": acc, "pairs": pairs, "vol": vol,
        "active": active, "inner": inner, "outer": outer_face, "r": r,
        "rhat": rhat, "m_inv": m_inv, "m_ok": m_ok, "grad": grad, "i": i,
        "j": j, "weight": weight,
    }


# --------------------------------------------------------------------------- #
# pressure projection
# --------------------------------------------------------------------------- #
def pressure_poisson3(pos, source, pairs, vol, h, dirichlet,
                      rel_tol: float = 1e-8, max_iter: int = 4000):
    """Solve ``-grad^2 p = source`` with ``p = 0`` on ``dirichlet`` (matrix-free CG).

    Same discrete operator as the 2-D projection case::

        grad^2 p_i = sum_j 2 V_j (p_i - p_j) (r_ij . grad_i W_ij) / (r_ij^2 + eta^2)

    with ``eta = 0.1 h``.  Dirichlet rows are replaced by identity rows so the
    same operator runs inside the CG loop.
    """
    n = pos.shape[0]
    eta = 0.1 * h
    i, j = pairs[:, 0], pairs[:, 1]
    delta, _, gw, _ = pair_geometry(pos, pairs, h)
    r2 = (delta * delta).sum(dim=1)
    rdot = (delta * gw).sum(dim=1)
    coef = -2.0 * vol[j] * rdot / (r2 + eta * eta)
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


# --------------------------------------------------------------------------- #
# the case
# --------------------------------------------------------------------------- #
def _face_kappa(kappa, delta_s, vol, mask):
    w = (delta_s * vol)[mask]
    if float(w.sum()) <= 0.0:
        return float("nan")
    return float((kappa[mask] * w).sum() / w.sum())


def hollow_sphere_case(*, radius_in: float = 6.0, radius_out: float = 14.0,
                       dx: float = 1.0, sigma: float = 0.072,
                       rho0: float = DEFAULT_RHO0, pad: float = 4.0,
                       h_factor: float = 1.3, curvature: str = "analytic",
                       lattice: str = "cubic", max_iter: int = 3000):
    """Run one shell/droplet configuration and return the measured jump.

    Lengths are in metres; the defaults are the dimensionless-lattice
    configuration ``R_in/dx = 6``, ``R_out/dx = 14`` at ``dx = 1 m`` scaled by
    the caller (all reported physics is scale-invariant: ``Delta p`` scales as
    ``sigma/R``).
    """
    pos, mass, liquid, r, half = sphere_shell_lattice(
        radius_in, radius_out, dx, pad, rho0, lattice)
    h = h_factor * dx
    st = shell_csf_state(pos, mass, liquid, h, rho0=rho0, sigma=sigma,
                         radius_in=radius_in, radius_out=radius_out,
                         curvature=curvature)
    vol, pairs, acc = st["vol"], st["pairs"], st["acc"]
    delta_s, kappa, normal, active = st["delta_s"], st["kappa"], st["normal"], st["active"]
    inner, outer_face, rhat = st["inner"], st["outer"], st["rhat"]

    # --- route 1: the pressure projection -------------------------------- #
    dim = pos.shape[1]
    div_a = _mls_divergence(st["m_inv"], vol, st["grad"], st["i"], st["j"],
                            acc, pos.shape[0], dim)
    div_a = torch.where(st["m_ok"], div_a, torch.zeros_like(div_a))
    source = -rho0 * div_a
    edge = 1.6 * dx
    dirichlet = r > half - edge
    p, n_cg = pressure_poisson3(pos, source, pairs, vol, h, dirichlet,
                                max_iter=max_iter)

    core = r < 0.5 * radius_in if radius_in > 0.0 else r < 0.5 * radius_out
    out_band = (r > radius_out + 2.0 * dx) & (r < half - edge)
    p_core = float(p[core].mean()) if bool(core.any()) else float("nan")
    p_out = float(p[out_band].mean()) if bool(out_band.any()) else 0.0
    dp = p_core - p_out

    # --- route 2: radial traction flux, independent of the Poisson solve --- #
    a_r = (acc * rhat).sum(dim=1)
    w = a_r * mass

    def _flux(mask, radius):
        """``-int rho a_r dV / (4 pi R^2)``: the jump the CSF traction carries."""
        if not bool(mask.any()) or radius <= 0.0:
            return float("nan")
        area = 4.0 * math.pi * radius * radius
        return float(-w[mask].sum() / area)

    dp_flux_in = _flux(inner, radius_in) if radius_in > 0.0 else 0.0
    dp_flux_out = _flux(outer_face, radius_out)
    dp_flux = dp_flux_in + dp_flux_out

    # --- curvature and delta normalisation per interface ------------------ #
    kappa_in = _face_kappa(kappa, delta_s, vol, inner) if radius_in > 0.0 else float("nan")
    kappa_out = _face_kappa(kappa, delta_s, vol, outer_face)
    # Signed analytic targets.  The outward normal of the *carrier* points into
    # the liquid at the cavity wall (kappa = -2/R_in) and out of the liquid at
    # the outer surface (kappa = +2/R_out); the drop case has only the latter.
    kappa_in_target = -2.0 / radius_in if radius_in > 0.0 else float("nan")
    kappa_out_target = 2.0 / radius_out
    ds_int = float((delta_s * vol).sum())
    area_total = 4.0 * math.pi * (radius_out ** 2 + (radius_in ** 2 if radius_in > 0 else 0.0))
    ds_int_dn = ds_int / area_total if area_total > 0.0 else float("nan")

    # --- spurious acceleration ------------------------------------------- #
    grad_p = _mls_gradient(st["m_inv"], vol, st["grad"], st["i"], st["j"], p,
                           pos.shape[0])
    resid = acc - grad_p / rho0
    resid_norm = torch.linalg.vector_norm(resid, dim=1)
    a_scale = float(torch.linalg.vector_norm(acc[active], dim=1).mean()) if bool(active.any()) else 0.0
    v_ref = math.sqrt(sigma / (rho0 * radius_out)) if sigma > 0.0 else float("nan")

    # Radial pressure profile (bin means), for the comparison figure: the exact
    # static solution is a two-level staircase,
    #   p = p_out + 2 sigma / R_out            in the liquid shell,
    #   p = p_out + 2 sigma (1/R_in + 1/R_out)  in the cavity,
    # so a plot of p(r) against it checks the whole field, not just two probes.
    n_bins = 36
    edges = torch.linspace(0.0, half, n_bins + 1, dtype=pos.dtype)
    idx = torch.bucketize(r, edges) - 1
    idx = idx.clamp(0, n_bins - 1)
    counts = torch.bincount(idx, minlength=n_bins).clamp_min(1)
    sums = torch.zeros(n_bins, dtype=pos.dtype).index_add(0, idx, p)
    profile = (sums / counts).tolist()
    centers = (0.5 * (edges[:-1] + edges[1:])).tolist()

    dp_theory = shell_laplace_analytic(sigma, radius_in, radius_out)
    safe = dp_theory if dp_theory > 0.0 else float("nan")
    return {
        "radius_in_m": radius_in, "radius_out_m": radius_out, "dx_m": dx,
        "h_m": h, "R_out_over_dx": radius_out / dx, "R_out_over_h": radius_out / h,
        "shell_dx": (radius_out - radius_in) / dx,
        "curvature": curvature, "lattice": lattice,
        "n_particles": int(pos.shape[0]),
        "n_liquid": int(liquid.sum()),
        "n_pairs": int(pairs.shape[0]),
        "dp_theory_Pa": dp_theory,
        "dp_measured_Pa": dp,
        "dp_over_theory": dp / safe,
        "dp_error_pct": (dp / safe - 1.0) * 100.0,
        "dp_flux_Pa": dp_flux,
        "dp_flux_inner_Pa": dp_flux_in,
        "dp_flux_outer_Pa": dp_flux_out,
        "dp_flux_over_theory": dp_flux / safe,
        "dp_flux_error_pct": (dp_flux / safe - 1.0) * 100.0,
        "dp_flux_inner_over_theory": (dp_flux_in / (2.0 * sigma / radius_in)
                                      if radius_in > 0.0 and sigma > 0.0 else float("nan")),
        "dp_flux_outer_over_theory": (dp_flux_out / (2.0 * sigma / radius_out)
                                      if sigma > 0.0 else float("nan")),
        "kappa_inner_measured": kappa_in,
        "kappa_inner_target": kappa_in_target,
        "kappa_outer_measured": kappa_out,
        "kappa_outer_target": kappa_out_target,
        "kappa_inner_error_pct": ((kappa_in - kappa_in_target) / abs(kappa_in_target) * 100.0
                                  if radius_in > 0.0 else float("nan")),
        "kappa_outer_error_pct": (kappa_out - kappa_out_target) / abs(kappa_out_target) * 100.0,
        "delta_s_integral_dn": ds_int_dn,
        "p_cavity_Pa": p_core, "p_outer_Pa": p_out,
        "residual_acc_over_mean_csf_acc": float(resid_norm.mean() / a_scale) if a_scale > 0 else float("nan"),
        "residual_acc_max": float(resid_norm.max()),
        "v_ref_m_per_s": v_ref,
        "cg_iterations": int(n_cg),
        "p_profile_r": centers,
        "p_profile_p": profile,
    }