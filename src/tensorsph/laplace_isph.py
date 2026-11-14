"""Young-Laplace pressure jump ``Delta p = sigma / R`` with projection (incompressible) SPH.

Why yet another module for the same case
----------------------------------------
The shipped TensorSPH droplet is a *weakly compressible* (WCSPH) code: the
pressure is a function of a density that is itself a kernel sum,

    p = B ((rho/rho0)**gamma - 1),       B = rho0 c0**2 / gamma .

For the Laplace jump that is a structural blocker, not a tuning problem.  The
verification audit in ``benchmarks/laplace_droplet/`` already recorded the
numbers (``resolvability_audit``, m13600): at ``sigma = 0.072 N/m``,
``R = 0.1 m``, ``c0 = 20 m/s``

    Delta p / B = 1.26e-5 ,      density ratio signal = 1.8e-6 ,

while the density field of a WCSPH particle cloud carries ~1e-3 relative noise.
The signal is *three orders of magnitude* below the noise, so no amount of
kernel/curvature fixing lets a Tait EOS carry ``sigma/R``; the relaxed runs
report -78.6 % (``benchmarks/csf_consistent/validation_report.json``) and the
best free-surface closure variant -20.1 %
(``docs/verification_laplace_pressure_closure.md``).  That is what killed the
three earlier rounds of this case.

The standard cure in the literature is to stop asking the EOS for the pressure.
In *incompressible* SPH (ISPH; Cummins & Rudman 1999, Shao & Lo 2003) the
pressure is a Lagrange multiplier obtained from a Poisson equation

    div( (1/rho) grad p ) = div(u*) / dt ,      u* = u_n + dt a_CSF ,

so the jump across an interface is carried by the *source* (the CSF force) to
the accuracy of the pressure solve, not by a 1e-6 density difference.  This
module implements exactly the static equilibrium of that formulation: it solves
for the pressure field that cancels the CSF traction, and measures

    Delta p = <p>_liquid - <p>_vapour

together with the momentum-balance residual (the discrete spurious
acceleration) so that the number cannot be a diagnostic artefact: a pressure
field that does not cancel the force is reported as such.

Discretisation (all operators are the ones already shipped in this repository)
------------------------------------------------------------------------------
colour function   ``c = sum_j V_j W_ij c_j / sum_j V_j W_ij``  (Shepard
                  smoothing of the phase indicator, ``V_j = m_j/rho0``), so
                  ``int |grad c| dn = 1`` across the interface.
normal            ``n = -grad c / |grad c|`` (outward), Shepard-smoothed twice.
gradient          first-order consistent MLS gradient (``csf_consistent``).
curvature         ``mls`` ``div n`` (default, validated on this two-phase
                  carrier: ``kappa R -> 1``, jump -0.2 %); the ``fit_curvature``
                  local quadratic surface fit (``csf_curvature``) is the
                  operator that removed the +25 % bias of ``div n`` on a *sharp*
                  free surface, but on the smooth multi-layer colour shell of
                  the filled carrier it measures ``kappa`` ~30 % low and is a
                  documented live defect (-24.5 %, fails the 15 % gate).
delta function    ``delta_s = |grad c|`` (Brackbill, Kothe & Zemach 1992).
force             ``a = -sigma kappa delta_s n / rho`` (CSF).
pressure          ``-grad^2 p = -rho div(a)`` on the pair-Laplacian of Cummins
                  & Rudman (1999),
                  ``grad^2 p_i = sum_j 2 V_j (p_i - p_j) (r_ij . grad_i W_ij) /
                  (r_ij^2 + eta^2)``, Dirichlet ``p = 0`` on the outer boundary
                  layer (the ambient pressure gauge), matrix-free CG.

The two-phase carrier
---------------------
The vapour phase is filled with particles of the *same* density as the liquid.
That is deliberate and standard practice for a *static* verification: the
Young-Laplace law contains ``sigma`` and ``R`` only, the density ratio never
enters it, and filling the void is what makes the colour function complete
``1 -> 0`` (``docs/verification_effective_sigma.md``).  It is not a multiphase
flow model and is not claimed as one.

References
    Brackbill, Kothe & Zemach (1992), J. Comput. Phys. 100:335.
    Cummins & Rudman (1999), J. Comput. Phys. 152:584.
    Shao & Lo (2003), Adv. Water Resour. 26:787.
    Adami, Hu & Adams (2010), J. Comput. Phys. 229:5011 (pressure boundary).
    Adami, Hu & Adams (2012), J. Comput. Phys. 231:7057 (spurious currents).
"""
from __future__ import annotations

import math

import torch

from .csf_consistent import mls_divergence, mls_gradient, shepard_smooth
from .csf_curvature import fit_curvature
from .droplet_impact import cubic_spline_gradient, cubic_spline_kernel
from .neighbor_search import cell_linked_pairs

__all__ = [
    "hex_block",
    "smooth_indicator",
    "csf_state_two_phase",
    "pressure_poisson",
    "spurious_acceleration",
    "laplace_isph_case",
]


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def hex_block(radius: float, dx: float, outer: float, rho0: float = 1000.0,
              center=None, angle: float = 0.0):
    """Hexagonal lattice filling a square block ``[-L, L]**2``, ``L = R + outer``.

    Every lattice site is kept: sites with ``|x - center| < R`` are liquid, the
    rest is the (equal-density) vapour carrier.  The outer 1.6 dx layer of the
    block is the Dirichlet boundary of the pressure solve.

    ``center`` (``(2,)`` sequence, default ``(0, 0)``) moves the *interface*
    relative to the *lattice*, which is how the lattice-alignment (translation)
    invariance of the Young-Laplace measurement is probed: a static drop is a
    physical solution, so the measured jump must not depend on where the
    interface falls inside a lattice cell.
    """
    dy = dx * math.sqrt(3.0) / 2.0
    half = radius + outer
    if angle == 0.0:
        # exact historical path (bit-identical): rows of the triangular lattice
        ny = int(math.floor(2.0 * half / dy))
        points = []
        for j in range(ny + 1):
            y = -half + j * dy
            shift = 0.5 * dx if (j % 2) else 0.0
            nx = int(math.floor((2.0 * half - shift) / dx))
            x = -half + shift + torch.arange(nx + 1, dtype=torch.float64) * dx
            yy = torch.full_like(x, y)
            points.append(torch.stack([x, yy], dim=1))
        pos = torch.cat(points, dim=0)
    else:
        # same triangular lattice, basis vectors rotated by ``angle``; used only
        # to probe the *orientation* (anisotropy) of the interface measurement.
        c, s_ = math.cos(angle), math.sin(angle)
        c2 = math.cos(angle + math.pi / 3.0)
        s2 = math.sin(angle + math.pi / 3.0)
        a1 = torch.tensor([dx * c, dx * s_], dtype=torch.float64)
        a2 = torch.tensor([dx * c2, dx * s2], dtype=torch.float64)
        n_side = int(math.ceil(2.0 * half / dx)) + 2
        idx = torch.arange(-n_side, n_side + 1, dtype=torch.float64)
        gi, gj = torch.meshgrid(idx, idx, indexing="ij")
        pts = (gi.reshape(-1, 1) * a1 + gj.reshape(-1, 1) * a2)
        pos = pts[(pts[:, 0].abs() <= half) & (pts[:, 1].abs() <= half)]
    pos = pos[(pos[:, 0].abs() <= half) & (pos[:, 1].abs() <= half)]
    area = dx * dy
    mass = torch.full((pos.shape[0],), rho0 * area, dtype=torch.float64)
    c = (torch.zeros(2, dtype=torch.float64) if center is None
         else torch.as_tensor(center, dtype=torch.float64))
    liquid = torch.linalg.vector_norm(pos - c, dim=1) < radius
    return pos, mass, liquid, half


def smooth_indicator(pos, liquid, vol, pairs, h):
    """Shepard-normalised smoothing of the phase indicator (``c = 1`` in liquid)."""
    n = pos.shape[0]
    i, j = pairs.unbind(dim=1)
    r = torch.linalg.vector_norm(pos[i] - pos[j], dim=1)
    w = cubic_spline_kernel(r, h)
    zero = torch.zeros(n, dtype=pos.dtype)
    num = zero.index_add(0, i, w * vol[j] * liquid[j].to(pos.dtype))
    den = zero.index_add(0, i, w * vol[j])
    # Non-uniform lattices leave sites with an empty support; keep them at their
    # own indicator value instead of dividing by ~0.
    return torch.where(den > 1e-12, num / den.clamp_min(1e-30),
                       liquid.to(pos.dtype))


# --------------------------------------------------------------------------- #
# CSF state on the two-phase carrier
# --------------------------------------------------------------------------- #
def csf_state_two_phase(pos, mass, liquid, h, rho0, sigma,
                        # ``mls`` (div n) is the *validated* default for the
                        # two-phase carrier: ``fit_quadratic`` fits the smooth
                        # multi-layer colour shell (not the sharp free-surface
                        # layer it was written for) and measures kappa ~30 %
                        # low, a -24.5 % jump that fails the 15 % gate
                        # (docs/xval_projection_laplace_oss.md, m17736 audit).
                        curvature_method: str = "mls",
                        curvature_stencil: float = 4.0):
    """Colour / normal / curvature / delta_s / CSF acceleration."""
    vol = mass / rho0
    pairs = cell_linked_pairs(pos, 2.0 * h)
    color = smooth_indicator(pos, liquid, vol, pairs, h)
    grad_c = mls_gradient(pos, color, vol, pairs, h)
    mag = torch.linalg.vector_norm(grad_c, dim=1, keepdim=True)
    tiny = 1e-14
    normal_raw = torch.where(mag > tiny, -grad_c / mag.clamp_min(tiny),
                             torch.zeros_like(grad_c))
    normal = shepard_smooth(normal_raw, pos, vol, pairs, h, passes=2)
    norm = torch.linalg.vector_norm(normal, dim=1, keepdim=True)
    normal = torch.where(norm > 1e-8, normal / norm.clamp_min(1e-30),
                         normal_raw)

    delta_s = mag.squeeze(-1)
    # Interface-active mask: without it the flat interior contributes numerical
    # dust to kappa and to the Poisson source.
    active = delta_s > 0.05 * delta_s.max()

    kappa_mls = mls_divergence(pos, normal, vol, pairs, h)
    if curvature_method == "mls":
        kappa = kappa_mls
    else:
        wide = cell_linked_pairs(pos, curvature_stencil * h)
        kappa = fit_curvature(pos, normal, delta_s, wide, h,
                              fallback=kappa_mls,
                              method=curvature_method,
                              stencil=curvature_stencil)
    kappa = torch.where(active, kappa, torch.zeros_like(kappa))
    delta_s = torch.where(active, delta_s, torch.zeros_like(delta_s))

    acc = (-sigma * kappa * delta_s)[:, None] * normal / rho0
    return {
        "color": color, "grad_c": grad_c, "normal": normal,
        "kappa": kappa, "delta_s": delta_s, "acc": acc,
        "pairs": pairs, "vol": vol, "active": active,
    }


# --------------------------------------------------------------------------- #
# pressure projection
# --------------------------------------------------------------------------- #
def _pair_coeffs(pos, pairs, vol, h, eta):
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    r2 = (delta * delta).sum(dim=1)
    gw = cubic_spline_gradient(delta, h)
    rdot = (delta * gw).sum(dim=1)
    coef = -2.0 * vol[j] * rdot / (r2 + eta * eta)
    return i, j, coef


def pressure_poisson(pos, source, pairs, vol, h, dirichlet,
                     rel_tol=1e-10, max_iter=4000):
    """Solve ``-grad^2 p = source`` with ``p = 0`` on ``dirichlet`` (matrix-free CG).

    ``dirichlet=True`` rows are eliminated as identity rows, so the same
    operator can be used inside the conjugate-gradient loop.
    """
    n = pos.shape[0]
    eta = 0.1 * h
    i, j, coef = _pair_coeffs(pos, pairs, vol, h, eta)
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
        if denom.abs() <= 1e-300:
            break
        alpha = rz / denom
        p = p + alpha * d
        r = r - alpha * ad
        if torch.linalg.vector_norm(r) / rhs_norm < rel_tol:
            break
        z = inv * r
        rz_new = (r * z).sum()
        d = z + (rz_new / rz) * d
        rz = rz_new
    return p, n_iter


def spurious_acceleration(pos, acc_csf, p, rho0, pairs, vol, h):
    """Residual of the momentum balance ``a_CSF - grad p / rho`` (per particle)."""
    grad_p = mls_gradient(pos, p, vol, pairs, h)
    return acc_csf - grad_p / rho0


# --------------------------------------------------------------------------- #
# the case
# --------------------------------------------------------------------------- #
def laplace_isph_case(radius: float = 0.02, dx: float = 0.002,
                      sigma: float = 0.072, rho0: float = 1000.0,
                      outer: float = 0.012, h_factor: float = 1.3,
                      # ``mls`` is the validated default (see
                      # csf_state_two_phase); ``fit_quadratic`` is a documented
                      # live defect on this two-phase carrier (-24.5 %).
                      curvature_method: str = "mls",
                      variant: str = "model",
                      curvature_stencil: float = 4.0,
                      out_band=(1.25, 1.45), core_frac: float = 0.5,
                      center=None, angle: float = 0.0):
    """One static equilibrium of the projection-SPH droplet.

    ``variant``
        ``model``       model curvature and model ``delta_s`` (the scheme under test)
        ``kappa_exact`` ``kappa = 1/R`` with the model ``delta_s``
        ``delta_unit``  model curvature with ``delta_s`` rescaled so that its
                        measured normal integral is 1
        ``gauge``       both replaced (instrument check: must return 1.000)
    """
    h = h_factor * dx
    pos, mass, liquid, half = hex_block(radius, dx, outer, rho0, center=center,
                                        angle=angle)
    st = csf_state_two_phase(pos, mass, liquid, h, rho0, sigma,
                             curvature_method=curvature_method,
                             curvature_stencil=curvature_stencil)
    vol, pairs, active = st["vol"], st["pairs"], st["active"]
    delta_s, kappa, normal = st["delta_s"], st["kappa"], st["normal"]

    c = (torch.zeros(2, dtype=pos.dtype) if center is None
         else torch.as_tensor(center, dtype=pos.dtype))
    rel = pos - c
    r = torch.linalg.vector_norm(rel, dim=1)
    rhat = rel / r.clamp_min(1e-30)[:, None]
    # sign of the model curvature along the radial direction: +1 when the
    # interface bends away from the outward normal (convex droplet).
    sign = torch.sign((normal * rhat).sum(dim=1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    kappa_eff = sign / radius

    # measured normal integral of delta_s (int delta_s dn), the Brackbill
    # normalisation.  Two-phase carrier => should be 1.
    ds_integral = float((delta_s * vol / (2.0 * math.pi * r.clamp_min(1e-30))).sum())

    if variant in ("kappa_exact", "gauge"):
        kappa_use = torch.where(active, kappa_eff, torch.zeros_like(kappa))
    else:
        kappa_use = kappa
    if variant == "delta_unit":
        scale = 1.0 / max(ds_integral, 1e-12)
        delta_use = delta_s * scale
    elif variant == "gauge":
        # exact analytic delta: distribute unit normal integral over the shell
        w = delta_s * vol
        delta_use = torch.where(w > 0, w / w.sum() * (2.0 * math.pi * radius)
                                / vol, torch.zeros_like(delta_s))
    else:
        delta_use = delta_s

    acc = (-sigma * kappa_use * delta_use)[:, None] * normal / rho0

    # source of the pressure Poisson: -grad^2 p = -rho div(a)
    div_a = mls_divergence(pos, acc, vol, pairs, h)
    source = -rho0 * div_a

    # Dirichlet on the outer layer of the block (ambient pressure gauge).
    edge = 1.6 * dx
    dirichlet = ((pos[:, 0].abs() > half - edge) | (pos[:, 1].abs() > half - edge))
    p, n_cg = pressure_poisson(pos, source, pairs, vol, h, dirichlet)

    core = r < core_frac * radius
    out_band = (r > out_band[0] * radius) & (r < out_band[1] * radius)
    p_core = float(p[core].mean())
    p_out = float(p[out_band].mean()) if bool(out_band.any()) else 0.0
    dp = p_core - p_out

    resid = spurious_acceleration(pos, acc, p, rho0, pairs, vol, h)
    resid_norm = torch.linalg.vector_norm(resid[active], dim=1)
    a_scale = float(torch.linalg.vector_norm(acc[active], dim=1).mean())

    # Independent cross-check of the jump: the radial integral of the CSF
    # traction, -int rho a_r dr.  It never touches the pressure solve, so
    # agreement between the two routes is what makes the number trustworthy.
    a_r = (acc * rhat).sum(dim=1)
    flux = float((-a_r * mass).sum() / (2.0 * math.pi * radius))
    v_ref = math.sqrt(sigma / (rho0 * radius))

    dp_theory = sigma / radius
    # sigma = 0 is the negative control: the ratio is undefined there, so the
    # reported error is the absolute jump (must be ~0).
    safe = dp_theory if dp_theory > 0.0 else float("nan")
    kappa_w = float((kappa_use * delta_use * vol / (delta_use * vol).sum()).sum())
    return {
        "variant": variant,
        "center_m": [float(c[0]), float(c[1])],
        "lattice_angle_deg": math.degrees(angle),
        "dx_m": dx, "h_m": h,
        "R_over_h": radius / h, "R_over_dx": radius / dx,
        "n_particles": int(pos.shape[0]),
        "n_liquid": int(liquid.sum()),
        "curvature_method": curvature_method,
        "delta_s_integral_dn": ds_integral,
        "kappa_interface_mean": float(kappa_use[active].mean()),
        "kappa_analytic": 1.0 / radius,
        "kappa_error_pct": (float(kappa_use[active].mean()) / (1.0 / radius) - 1.0)
        * 100.0,
        "kappa_delta_weighted": kappa_w,
        "kappaR_delta_weighted": kappa_w * radius,
        "p_core_Pa": p_core,
        "p_vapour_Pa": p_out,
        "dp_measured_Pa": dp,
        "dp_theory_Pa": dp_theory,
        "dp_over_theory": dp / safe,
        "dp_error_pct": (dp / safe - 1.0) * 100.0,
        "dp_flux_Pa": flux,
        "dp_flux_over_theory": flux / safe,
        "residual_acc_over_mean_csf_acc": float(resid_norm.mean() / a_scale),
        "residual_acc_median_over_mean_csf_acc": float(resid_norm.median() / a_scale),
        "residual_bulk_median_over_mean_csf_acc": float(
            torch.linalg.vector_norm(resid[~active], dim=1).median() / a_scale
        ) if bool((~active).any()) else 0.0,
        "residual_acc_max": float(resid_norm.max()),
        "v_ref_m_per_s": v_ref,
        "cg_iterations": int(n_cg),
    }

# --------------------------------------------------------------------------- #
# dynamic projection loop (is the static equilibrium actually static?)
# --------------------------------------------------------------------------- #
def relax_projection(radius: float = 0.02, dx: float = 0.002,
                     sigma: float = 0.072, rho0: float = 1000.0,
                     outer: float = 0.012, h_factor: float = 1.3,
                     curvature_method: str = "mls",
                     steps: int = 200, dt_factor: float = 0.1,
                     sample_every: int = 25):
    # KNOWN LIMIT (measured, see docs/verification_laplace_sph_isph.md): this
    # time integration is not yet stable.  The projection removes only the
    # *divergent* part of the CSF acceleration; its divergence-free part is a
    # lattice artefact of order the force itself, and it accumulates in ``u``.
    # The loop therefore stops on the first non-finite velocity and reports
    # ``diverged``.  It is shipped to *measure* that defect, not as a solver.
    """March the projection scheme; a correct equilibrium must not move.

    ``u* = u + dt a_CSF``, ``-grad^2 p = rho div(u*)/dt`` (Dirichlet ``p = 0`` on
    the outer layer), ``u = u* - dt grad p / rho``, ``x += dt u``.  Reported:
    ``max|u| / sqrt(sigma/(rho R))`` (Adami et al. 2012 accept < 0.1) and the
    drift of the equivalent radius, which is what a spurious inward force would
    destroy first.
    """
    import math as _math

    h = h_factor * dx
    pos, mass, liquid, half = hex_block(radius, dx, outer, rho0)
    vol0 = mass / rho0
    dt = dt_factor * _math.sqrt(rho0 * h ** 3 / max(sigma, 1e-30))
    v_ref = _math.sqrt(sigma / (rho0 * radius))
    edge = 1.6 * dx
    u = torch.zeros_like(pos)
    history = []
    r0 = torch.linalg.vector_norm(pos, dim=1)
    area0 = float((r0 < radius).sum()) * dx * dx * _math.sqrt(3.0) / 2.0

    for step in range(1, steps + 1):
        pairs = cell_linked_pairs(pos, 2.0 * h)
        st = csf_state_two_phase(pos, mass, liquid, h, rho0, sigma,
                                 curvature_method=curvature_method)
        vol = vol0
        u_star = u + dt * st["acc"]
        div_u = mls_divergence(pos, u_star, vol, pairs, h)
        source = rho0 * div_u / dt
        dirichlet = ((pos[:, 0].abs() > half - edge)
                     | (pos[:, 1].abs() > half - edge))
        p, _ = pressure_poisson(pos, source, pairs, vol, h, dirichlet,
                                rel_tol=1e-8, max_iter=2000)
        grad_p = mls_gradient(pos, p, vol, pairs, h)
        u = u_star - dt * grad_p / rho0
        pos = pos + dt * u
        _vmax = float(torch.linalg.vector_norm(u, dim=1).max())
        if (not bool(torch.isfinite(u).all())) or _vmax > 1.0e3 * v_ref:
            history.append({"step": step, "diverged": True})
            return {"radius_m": radius, "dx_m": dx, "h_m": h, "dt_s": dt,
                    "R_over_dx": radius / dx, "v_ref_m_per_s": v_ref,
                    "curvature_method": curvature_method, "sigma": sigma,
                    "history": history, "diverged": True,
                    "steps_to_divergence": step,
                    "umax_over_vref_final": float("nan"),
                    "area_ratio_final": float("nan"),
                    "dp_final_Pa": float("nan"),
                    "dp_theory_Pa": sigma / radius,
                    "dp_rel_error_pct": float("nan")}
        if step % sample_every == 0 or step == steps:
            speed = torch.linalg.vector_norm(u, dim=1)
            r_now = torch.linalg.vector_norm(pos, dim=1)
            area = float((r_now < radius).sum()) * dx * dx * _math.sqrt(3.0) / 2.0
            core = r_now < 0.5 * radius
            out = (r_now > 1.25 * radius) & (r_now < 1.45 * radius)
            history.append({
                "step": step,
                "t_s": step * dt,
                "t_over_t_sigma": step * dt / _math.sqrt(rho0 * h ** 3 / sigma),
                "umax_over_vref": float(speed.max() / v_ref),
                "umean_over_vref": float(speed.mean() / v_ref),
                "dp_Pa": float(p[core].mean() - p[out].mean()),
                "area_ratio": area / area0,
            })
    return {
        "radius_m": radius, "dx_m": dx, "h_m": h, "dt_s": dt,
        "R_over_dx": radius / dx,
        "v_ref_m_per_s": v_ref,
        "curvature_method": curvature_method,
        "sigma": sigma,
        "history": history,
        "diverged": False,
        "umax_over_vref_final": history[-1]["umax_over_vref"],
        "area_ratio_final": history[-1]["area_ratio"],
        "dp_final_Pa": history[-1]["dp_Pa"],
        "dp_theory_Pa": sigma / radius,
        "dp_rel_error_pct": (history[-1]["dp_Pa"] / (sigma / radius) - 1.0) * 100.0,
    }
