"""Boundary-integral wall treatment for the closed-tank lake at rest.

Case-pool anchor ``CASE-SPH-LAKE-AT-REST`` (still water, ``u = 0``, flat free
surface, analytic oracle ``p = rho0 g (eta - y)``, tolerance = machine
precision).  The anchor is met by the laterally-periodic tank
(``docs/verification_lake_at_rest_periodic.md``) and is an honest FAIL in the
closed tank (``docs/verification_lake_at_rest_wall_gap.md``).

Two studies have already walked the wall gap down to a single open sentence:

* ``docs/verification_lake_at_rest_wall_gap.md`` -- the residual is not a
  discretisation error that a mass field can cancel (no symmetric mass field is
  admissible), and
* ``docs/verification_lake_at_rest_wb.md`` -- four consistent / well-balanced
  *pressure-gradient* operators (MLS moment correction, free-surface own-pressure
  reconstruction, well-balanced difference form, least-squares plane gradient)
  do not close it either; best ``max|a|/g = 9.285e-02``.

That study ends with an explicit limitation of its own scope:

    "算子族限于「压力梯度 + 质量场」两类自由度；壁面积分（boundary-integral）
    重构成**未**实现，因此「必需壁面积分」仍是**推断**而非本单的测量结论。"

This module implements that last branch -- the wall treated as a *surface
integral* instead of a set of static wall particles -- so that the sentence
"a boundary-integral wall treatment is required" becomes a measurement instead
of an inference.

The wall term
-------------
The static wall particles of the shipped solver are replaced by the boundary of
the fluid domain, discretised into segments of length ``A_s`` (the 2-D surface
measure per unit depth) with outward normals pointing into the fluid.  The
momentum equation of particle ``i`` becomes

    a_i = a_i^fluid + a_i^wall,   a_i^wall = -coef_s sum_s A_s (.) grad_i W_is,

with three standard choices of the coefficient (all measured, none asserted):

``form="dS"``
    ``a_i^wall = -(1/rho_i) sum_s p_w,s A_s grad_i W_is`` -- the plain boundary
    integral of the wall pressure (Ferrand et al. 2013 form without the
    renormalisation factor).
``form="dS2"``
    the same with the factor 2, i.e. the pressure of the mirrored image counted
    on both sides of the face (the classic "mirror pressure" boundary integral
    of Kulasegaram et al. / Feldman & Bonet).
``form="symmetrised"``
    ``-sum_s (p_i/rho_i^2 + p_w/rho_w^2) rho_w A_s grad_i W_is`` -- the wall
    segment carries the same symmetrised coefficient as a fluid neighbour, so
    the only difference from the shipped operator is *where* the wall mass sits
    (on the face, with the face pressure) and that its density comes from the
    same Adami closure.

and three choices of the face pressure ``p_w``:

``wall_mode="adami"``
    Adami et al. (2012) extrapolation from the fluid side -- data-driven, no
    analytic pressure injected;
``wall_mode="hydrostatic"``
    the exact hydrostatic face pressure ``rho0 g (eta - y_s)`` -- an *oracle*
    upper bound: it measures whether an *exact* wall pressure would suffice.  It
    is a diagnostic, never a candidate closure;
``wall_mode="zero"``
    no wall at all (the control that shows the wall term is what is being
    measured).

The fluid side keeps the shipped symmetrised estimate, restricted to
fluid/fluid pairs; ``fluid_term=False`` removes it, ``wall_term=False`` removes
the wall, and both-off reproduces a free-flying fluid blob (used as a control).

Everything is measured on two mass fields: the *shipped* one (closed to vertical
balance by :func:`tensorsph.lake_at_rest.reconstruct`) and one re-solved by the
same Gauss-Newton protocol *with this operator* (:func:`solve_mass`), so a
variant is never credited with a vertical balance it does not hold by itself.
"""
from __future__ import annotations

import math

import torch

from .droplet_impact import cubic_spline_kernel, pair_kernels
from .lake_at_rest import (LakeConfig, analytic_profile, build_state, reconstruct)
from .lake_at_rest_wb import _signed_side, consistent_gradient, wb_fields

__all__ = [
    "ANCHOR_TOLERANCE",
    "wall_segments",
    "wall_face_pressure",
    "bi_accel",
    "band_stats",
    "solve_mass",
    "variant_spec",
    "VARIANTS",
]

#: The anchor's tolerance: a machine-precision identity (``1e-9`` relative to g).
ANCHOR_TOLERANCE = 1.0e-9


# --------------------------------------------------------------------------- #
# The wall surface
# --------------------------------------------------------------------------- #
def wall_segments(cfg: LakeConfig, sub: int = 1) -> dict:
    """Discretised boundary of the fluid domain, oriented into the fluid.

    The fluid (and the first wall layer of the shipped lattice) is centred at
    ``(k + 1/2) dx``, so the faces are the planes ``x = 0``, ``x = width`` and
    ``y = 0``.  Segments are laid out on the same half-offset lattice as the
    particle columns, which is what makes the segment quadrature and the shipped
    wall-particle sum directly comparable: a segment centre then sits exactly at
    the wall-particle position of the same column.

    ``sub`` subdivides every segment (``sub = 2`` halves the quadrature length
    without moving the face); it is the quadrature-refinement knob.

    Returns ``{"pos": (S, 2), "normal": (S, 2), "length": (S,), "labels":
    list[str], "perimeter": float}``.
    """
    if cfg.periodic_x:
        raise ValueError("the closed tank has walls; set periodic_x=False")
    sub = int(sub)
    if sub < 1:
        raise ValueError("sub must be >= 1")
    dx = float(cfg.dx) / sub
    device = torch.device(cfg.device)
    dtype = cfg.torch_dtype
    pos, normal, labels = [], [], []
    # bottom: y = 0, x in [0, width], normal (0, +1)
    nx = int(round(cfg.width / dx))
    x = (torch.arange(nx, device=device, dtype=dtype) + 0.5) * dx
    pos.append(torch.stack((x, torch.zeros_like(x)), dim=1))
    normal.append(torch.tensor([0.0, 1.0], dtype=dtype, device=device).expand(nx, 2))
    labels += ["bottom"] * nx
    # side walls: x = 0 (normal +x) and x = width (normal -x), y in [0, depth]
    ny = int(round(cfg.depth / dx))
    y = (torch.arange(ny, device=device, dtype=dtype) + 0.5) * dx
    for xw, sign, name in ((0.0, 1.0, "left"), (float(cfg.width), -1.0, "right")):
        pos.append(torch.stack((torch.full_like(y, xw), y), dim=1))
        normal.append(torch.tensor([sign, 0.0], dtype=dtype,
                                   device=device).expand(ny, 2))
        labels += [name] * ny
    pos = torch.cat(pos, dim=0)
    normal = torch.cat(normal, dim=0)
    length = torch.full((pos.shape[0],), dx, dtype=dtype, device=device)
    perimeter = 2.0 * float(cfg.depth) + float(cfg.width)
    return {"pos": pos, "normal": normal, "length": length, "labels": labels,
            "perimeter": perimeter,
            "quadrature_perimeter": float(length.sum())}


def wall_face_pressure(seg: dict, fields: dict, mode: str = "adami") -> dict:
    """Face pressure and face density on every wall segment.

    ``adami`` extrapolates from the fluid side exactly as the shipped wall
    particles do (``p_w = [sum_f (p_f + rho_f g (y_f - y_s)) W_sf] / sum_f W_sf``,
    Adami et al. 2012 eq. for a static wall), so the only thing that changes
    between the shipped operator and the boundary integral is *where* the wall is
    sampled, not what pressure it carries.

    ``hydrostatic`` is the analytic face pressure ``rho0 g (eta - y_s)``, clipped
    at zero above the free surface.  It is an oracle: it answers "would an exact
    wall pressure be enough?" and is never a candidate physical closure.

    ``zero`` returns zero pressure (no wall).
    """
    cfg = fields["cfg"]
    dtype, device = seg["pos"].dtype, seg["pos"].device
    bulk = cfg.rho0 * cfg.sound_speed ** 2 / cfg.gamma
    if mode == "hydrostatic":
        p_w = (cfg.rho0 * cfg.gravity * (cfg.depth - seg["pos"][:, 1])).clamp_min(0.0)
    elif mode == "zero":
        p_w = torch.zeros(seg["pos"].shape[0], dtype=dtype, device=device)
    elif mode == "adami":
        # wb_fields carries the *wall* entries of rho/pressure in the same
        # tensors; the extrapolation is over fluid particles only
        n = fields["n"]
        pos, rho, pressure = fields["pos"], fields["rho"][:n], fields["pressure"][:n]
        # d = x_wall - x_fluid, so d[..., 1] = y_s - y_f
        d = seg["pos"][:, None, :] - pos[None, :, :]
        r = torch.linalg.vector_norm(d, dim=-1)
        W = cubic_spline_kernel(r.reshape(-1), fields["h"]).reshape_as(r)
        source = pressure[None, :] + rho[None, :] * (-cfg.gravity) * d[..., 1]
        num = (source * W).sum(dim=1)
        den = W.sum(dim=1)
        p_w = (num / den.clamp_min(1e-30)).clamp_min(0.0)
        non_empty = den > 0.0
    else:
        raise ValueError(f"unknown wall_mode {mode!r}")
    rho_w = (cfg.rho0 * (p_w / bulk + 1.0).pow(1.0 / cfg.gamma)).clamp_min(cfg.rho0)
    if mode == "adami":
        # a segment with no fluid in support carries no wall traction; keep its
        # density at rho0 so that rho_w never divides by a made-up number
        rho_w = torch.where(non_empty, rho_w, torch.full_like(rho_w, cfg.rho0))
    return {"p_w": p_w, "rho_w": rho_w, "mode": mode}


# --------------------------------------------------------------------------- #
# The momentum operator
# --------------------------------------------------------------------------- #
def bi_accel(cfg: LakeConfig, mass: torch.Tensor, wall_mode: str = "adami",
             form: str = "dS", own_pressure: str = "eos",
             gravity_on: bool = True, pressure_scale: float = 1.0,
             sub: int = 1, wall_term: bool = True,
             fluid_term: bool = True,
             fluid_form: str = "symmetrised") -> torch.Tensor:
    """Acceleration of every fluid particle under the boundary-integral wall.

    ``form`` in ``{"dS", "dS2", "symmetrised"}`` and ``wall_mode`` in
    ``{"adami", "hydrostatic", "zero"}`` -- see the module docstring.

    ``own_pressure="reconstructed"`` replaces the particle's own (Tait-clamped)
    pressure in the fluid/fluid term by the least-squares hydrostatic value of
    ``tensorsph.lake_at_rest_wb`` -- the ingredient that was worth ~15 % in that
    study; kept available so that the two effects can be separated.
    """
    fields = wb_fields(cfg, mass)
    if pressure_scale != 1.0:
        # one place only: every pressure the operator sees is scaled, so the
        # control measures sensitivity to the *field*, not to one term of it
        fields = dict(fields)
        fields["pressure"] = fields["pressure"] * pressure_scale
        fields["p_all"] = fields["p_all"] * pressure_scale
        fields["p_wall"] = fields["p_wall"] * pressure_scale
    n = fields["n"]
    pos = fields["pos"]
    rho, p = fields["rho"][:n], fields["pressure"][:n]
    dtype, device = pos.dtype, pos.device
    acc = torch.zeros((n, 2), dtype=dtype, device=device)

    if fluid_term:
        tgt, other, gx, gy, _, _ = _signed_side(fields)
        is_fluid = other < n
        p_own = p if own_pressure == "eos" else consistent_gradient(fields)["p_own"]
        V = fields["V_all"][other]
        if fluid_form == "symmetrised":
            coef = -(p_own[tgt] / rho[tgt].square()
                     + fields["p_all"][other] / fields["rho_all"][other].square()) \
                * fields["all_mass"][other]
            coef = torch.where(is_fluid, coef, torch.zeros_like(coef))
            acc[:, 0] = acc[:, 0].index_add(0, tgt, coef * gx)
            acc[:, 1] = acc[:, 1].index_add(0, tgt, coef * gy)
        elif fluid_form == "difference_mls":
            # first-order consistent (MLS) gradient of the *fluid-only* support,
            # in difference form: -M_i^-1 sum_j V_j (p_j - p_i) grad_i W_ij / rho_i
            ddx, ddy = _raw_offset(fields, tgt, other)
            keep = is_fluid.to(V.dtype)
            Vk = V * keep
            m00 = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, Vk * ddx * gx)
            m01 = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, Vk * ddx * gy)
            m10 = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, Vk * ddy * gx)
            m11 = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, Vk * ddy * gy)
            diff = torch.where(is_fluid, fields["p_all"][other] - p_own[tgt],
                               torch.zeros_like(gx))
            rx = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, Vk * diff * gx)
            ry = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, Vk * diff * gy)
            det = m00 * m11 - m01 * m10
            floor = 1e-6 * (m00.abs() * m11.abs()).clamp_min(1e-30)
            det = torch.where(det.abs() < floor,
                              torch.where(det < 0, -floor, floor), det)
            gpx = (m11 * rx - m01 * ry) / det
            gpy = (-m10 * rx + m00 * ry) / det
            acc[:, 0] = acc[:, 0] - gpx / rho
            acc[:, 1] = acc[:, 1] - gpy / rho
        else:
            raise ValueError(f"unknown fluid_form {fluid_form!r}")

    if wall_term:
        seg = wall_segments(cfg, sub=sub)
        face = wall_face_pressure(seg, fields, mode=wall_mode)
        p_w, rho_w, area = face["p_w"], face["rho_w"], seg["length"]
        if form in ("dS", "dS2"):
            # naive surface forms: keep them, they are measured and *rejected*
            # (a 2-D gradient of the 2-D kernel carries 1/length^3, so the plain
            # surface integral is short by one power of the spacing)
            d = pos[:, None, :] - seg["pos"][None, :, :]          # x_i - x_s
            _, grad = pair_kernels(d.reshape(-1, 2), fields["h"])  # grad_i W_is
            grad = grad.reshape(n, seg["pos"].shape[0], 2)
            scale = 1.0 if form == "dS" else 2.0
            coef = -scale * (p_w * area)[None, :] / rho[:, None]
            acc[:, 0] = acc[:, 0] + (coef * grad[..., 0]).sum(dim=1)
            acc[:, 1] = acc[:, 1] + (coef * grad[..., 1]).sum(dim=1)
        elif form == "surface_w":
            # divergence-theorem (semi-analytical) boundary integral: the wall
            # side of the kernel is transported to the face with W, not grad W
            #   -int_{Omega_w} rho c grad_i W dA
            #     = + n int_S rho_w c_w W dS + int_{Omega_w} W grad(rho c) dA
            # with c_w = p_i/rho_i^2 + p_w/rho_w^2 and grad(rho c) = -g yhat for
            # a hydrostatic fluid.  int_{Omega_w} W dA is a *kernel weight*: it is
            # taken from the shipped wall particles (a smooth quadrature, no
            # pressure involved), which is the semi-analytical split.
            phi_w = (rho_w[None, :] / rho.square()[:, None]) * p[:, None] \
                + (p_w / rho_w)[None, :]
            d = pos[:, None, :] - seg["pos"][None, :, :]
            r = torch.linalg.vector_norm(d, dim=-1)
            W = cubic_spline_kernel(r.reshape(-1), fields["h"]).reshape_as(r)
            coef = phi_w * (W * area[None, :])
            acc[:, 0] = acc[:, 0] + (coef * seg["normal"][None, :, 0]).sum(dim=1)
            acc[:, 1] = acc[:, 1] + (coef * seg["normal"][None, :, 1]).sum(dim=1)
            acc[:, 1] = acc[:, 1] + cfg.gravity * wall_kernel_weight(fields)
        else:
            raise ValueError(f"unknown form {form!r}")

    if gravity_on:
        acc = acc + torch.tensor([0.0, -float(cfg.gravity)], dtype=dtype, device=device)
    return acc


def _raw_offset(fields: dict, tgt: torch.Tensor, other: torch.Tensor):
    """``x_j - x_i`` on the directed pair list, with the periodic seam wrapped.

    The moment matrix of the MLS gradient is built on the same (wrapped)
    separation the kernel was evaluated on; using the raw difference would
    corrupt it by a domain width at the seam, which is exactly the defect the
    ``B1`` control of ``lake_at_rest_wb`` caught.
    """
    pos, all_pos = fields["pos"], fields["all_pos"]
    dd = all_pos[other] - pos[tgt]
    wrap = getattr(fields["cfg"], "wrap_width", None)
    if wrap is not None:
        dd = torch.stack((dd[:, 0] - wrap * torch.round(dd[:, 0] / wrap), dd[:, 1]), dim=1)
    return dd[:, 0], dd[:, 1]


def wall_kernel_weight(fields: dict) -> torch.Tensor:
    """``int_{Omega_wall} W dA`` per fluid particle, from the shipped wall lattice.

    The wall-side kernel weight is a *smooth* quantity (no pressure in it), so the
    shipped wall particles are a good quadrature for it even where their pressure
    extrapolation is not.  Getting this weight from the lattice and the pressure
    traction from the face is the semi-analytical split used by the
    boundary-integral wall conditions.
    """
    n, i, j = fields["n"], fields["i"], fields["j"]
    w, V = fields["wij"], fields["V_all"]
    out = torch.zeros(n, dtype=w.dtype, device=w.device)
    ij = (i < n) & (j >= n)
    if bool(ij.any()):
        out = out.index_add(0, i[ij], V[j[ij]] * w[ij])
    ji = (j < n) & (i >= n)
    if bool(ji.any()):
        out = out.index_add(0, j[ji], V[i[ji]] * w[ji])
    return out


def band_stats(cfg: LakeConfig, pos: torch.Tensor, acc: torch.Tensor,
               h: float) -> dict:
    """Residual max-norms, overall / wall band / interior, and where they live.

    The ``top_row`` fields are the localisation diagnostic: ``wall_gap.md``
    found the shipped operator's residual to sit on the free-surface particle of
    the wall column, and the rest state is only re-opened if a *new* operator
    moves it somewhere else.  Reporting the top-row share makes that visible for
    every variant instead of being inferred from one maximum.
    """
    band = (pos[:, 0] < 2.0 * h) | (pos[:, 0] > cfg.width - 2.0 * h)
    ratio = torch.linalg.vector_norm(acc, dim=1) / cfg.gravity
    top = pos[:, 1] >= pos[:, 1].max() - 1e-9 * max(cfg.depth, 1.0)
    worst = float(ratio.max())
    return {
        "max_a_over_g": worst,
        "max_ax_over_g": float(acc[:, 0].abs().max() / cfg.gravity),
        "max_ay_over_g": float(acc[:, 1].abs().max() / cfg.gravity),
        "max_a_over_g_wall_band": float(ratio[band].max()) if bool(band.any()) else 0.0,
        "max_a_over_g_interior": float(ratio[~band].max()) if bool((~band).any()) else 0.0,
        "max_a_over_g_below_top_row": (float(ratio[~top].max())
                                       if bool((~top).any()) else 0.0),
        "top_row_share": (float(ratio[top].max() / worst) if worst > 0.0
                          and bool(top.any()) else 0.0),
        "worst_particle_xy": [float(pos[ratio.argmax(), 0]),
                              float(pos[ratio.argmax(), 1])],
        "wall_band_particles": int(band.sum()),
        "top_row_particles": int(top.sum()),
        "particles": int(pos.shape[0]),
        "anchors_over_tolerance": worst / ANCHOR_TOLERANCE,
    }


# --------------------------------------------------------------------------- #
# Mass field re-solved with the boundary-integral operator
# --------------------------------------------------------------------------- #
def solve_mass(cfg: LakeConfig, max_iter: int = 25, tol: float = 1.0e-11,
               prior_weight: float | None = None, verbose: bool = False,
               **accel_kwargs) -> dict:
    """Damped Gauss-Newton on the mirror-pair mass field, for this operator.

    Same protocol as :func:`tensorsph.lake_at_rest.reconstruct` and
    :func:`tensorsph.lake_at_rest_wb.wb_reconstruct`: unknown ``u = log(m/m_unit)``
    per mirror pair, residual the *vertical* acceleration (the balance a mass
    field is allowed to close -- the horizontal one provably is not, see
    ``wall_gap.md``), Tikhonov pull towards the analytic profile in the null
    space, backtracking line search.
    """
    state = build_state(cfg)
    pos = state["pos"]
    mapping = state["columns"]
    unique = int(mapping.max().item()) + 1
    m_unit = state["m_unit"]
    dtype, device = pos.dtype, pos.device
    prior = analytic_profile(state, cfg)["mass"]
    u_prior = torch.log(prior / m_unit)
    sums = torch.zeros(unique, dtype=dtype, device=device).index_add(0, mapping, u_prior)
    counts = torch.zeros(unique, dtype=dtype, device=device).index_add(
        0, mapping, torch.ones_like(u_prior))
    u = sums / counts
    u_prior_unique = u.clone()
    lam = float(cfg.prior_weight if prior_weight is None else prior_weight)
    g = torch.tensor([0.0, -float(cfg.gravity)], dtype=dtype, device=device)

    def residual(uu: torch.Tensor) -> torch.Tensor:
        acc = bi_accel(cfg, m_unit * torch.exp(uu[mapping]), gravity_on=False,
                       **accel_kwargs)
        return (acc[:, 1] + g[1]).reshape(-1) / cfg.gravity

    history = []
    for it in range(max_iter):
        r = residual(u)
        worst = float(r.abs().max())
        history.append(worst)
        if verbose:
            print(f"  iter {it:2d}  max|a_y|/g = {worst:.3e}")
        if worst < tol:
            break
        jac = torch.autograd.functional.jacobian(residual, u)
        step = torch.linalg.lstsq(
            torch.cat((jac, lam * torch.eye(unique, dtype=dtype, device=device)), dim=0),
            torch.cat((-r, lam * (u_prior_unique - u)), dim=0)).solution
        merit = float(torch.linalg.vector_norm(r)) / math.sqrt(max(r.numel(), 1))
        alpha, accepted = 1.0, False
        for _ in range(30):
            candidate = u + alpha * step
            rc = residual(candidate)
            if float(torch.linalg.vector_norm(rc)) / math.sqrt(
                    max(rc.numel(), 1)) < merit:
                u, accepted = candidate, True
                break
            alpha *= 0.5
        if not accepted:
            break
    mass = m_unit * torch.exp(u[mapping])
    acc = bi_accel(cfg, mass, **accel_kwargs)
    stats = band_stats(cfg, pos, acc, cfg.h())
    return {"mass": mass, "accel": acc, "stats": stats,
            "iterations": len(history), "residual_history": history,
            "converged": bool(history and history[-1] < tol),
            "mass_modulation": float((mass / m_unit).max() / (mass / m_unit).min())}


# --------------------------------------------------------------------------- #
# The variant table the studies measure
# --------------------------------------------------------------------------- #
#: ``(name, accel kwargs, description)``.  The shipped wall-particle operator is
#: *not* in this table: it is :func:`tensorsph.lake_at_rest_wb.wb_accel` with
#: ``corrected_gradient=False`` and is measured as the baseline.
VARIANTS = (
    ("bi_surface_adami", dict(form="surface_w", wall_mode="adami"),
     "divergence-theorem (semi-analytical) boundary integral, Adami face pressure"),
    ("bi_surface_hydrostatic", dict(form="surface_w", wall_mode="hydrostatic"),
     "the same with the *exact* hydrostatic face pressure (oracle upper bound)"),
    ("bi_surface_mls_fluid",
     dict(form="surface_w", wall_mode="adami", fluid_form="difference_mls"),
     "boundary integral + first-order consistent (MLS) difference-form fluid term"),
    ("bi_naive_dS_adami", dict(form="dS", wall_mode="adami"),
     "naive surface integral of grad W (measured and rejected: short by one length)"),
    ("bi_naive_dS2_adami", dict(form="dS2", wall_mode="adami"),
     "naive surface integral with the mirror factor 2 (also rejected)"),
    ("fluid_only", dict(wall_term=False),
     "no wall at all: the control that the wall term is what is being measured"),
)


def variant_spec(name: str) -> dict:
    """Accel kwargs of a named variant (raises for unknown names)."""
    for variant, kwargs, _ in VARIANTS:
        if variant == name:
            return dict(kwargs)
    raise KeyError(f"unknown variant {name!r}; known: "
                   + ", ".join(v for v, _, _ in VARIANTS))


# --------------------------------------------------------------------------- #
# Implementation check: does the surface form reproduce the volume quadrature?
# --------------------------------------------------------------------------- #
def wall_integral_check(cfg: LakeConfig, mass: torch.Tensor,
                        wall_mode: str = "adami", sub: int = 1) -> dict:
    """Surface form vs the wall-particle volume quadrature of the same integral.

    The shipped operator evaluates the wall side of the kernel as a *volume* sum
    ``sum_{j in wall} V_j phi_j grad_i W_ij``; the boundary integral replaces it
    by ``-n int_S phi_w W dS + int_{Omega_w} W grad(phi) dA``.  For a smooth
    field the two must agree up to quadrature and up to the closure assumption
    ``grad(phi) = -g yhat`` -- which is the only non-exact piece.  The returned
    relative difference therefore measures *how much of the wall integral the
    surface form actually captures*; it is the honest error bar on every
    boundary-integral number in this module, and an implementation check that
    fails loudly if the sign of the surface term is ever flipped.
    """
    fields = wb_fields(cfg, mass)
    n = fields["n"]
    pos, rho, p = fields["pos"], fields["rho"][:n], fields["pressure"][:n]
    i, j, w = fields["i"], fields["j"], fields["wij"]
    _, grad = pair_kernels(fields["delta"], fields["h"])
    V, all_pos = fields["V_all"], fields["all_pos"]
    num_x = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    num_y = torch.zeros_like(num_x)
    for mask, tgt_idx, nb_idx, sign in (((i < n) & (j >= n), i, j, 1.0),
                                        ((j < n) & (i >= n), j, i, -1.0)):
        tgt, nb = tgt_idx[mask], nb_idx[mask]
        phi = (fields["rho_all"][nb] / rho[tgt].square()) * p[tgt] \
            + fields["p_all"][nb] / fields["rho_all"][nb]
        num_x = num_x.index_add(0, tgt, V[nb] * phi * sign * grad[mask, 0])
        num_y = num_y.index_add(0, tgt, V[nb] * phi * sign * grad[mask, 1])
    volume = torch.stack((num_x, num_y), dim=1)

    seg = wall_segments(cfg, sub=sub)
    face = wall_face_pressure(seg, fields, mode=wall_mode)
    p_w, rho_w, area = face["p_w"], face["rho_w"], seg["length"]
    d = pos[:, None, :] - seg["pos"][None, :, :]
    r = torch.linalg.vector_norm(d, dim=-1)
    W = cubic_spline_kernel(r.reshape(-1), fields["h"]).reshape_as(r)
    phi_s = (rho_w[None, :] / rho.square()[:, None]) * p[:, None] + (p_w / rho_w)[None, :]
    surface = torch.stack(((phi_s * W * area[None, :] * seg["normal"][None, :, 0]).sum(1),
                           (phi_s * W * area[None, :] * seg["normal"][None, :, 1]).sum(1)),
                          dim=1)
    grad_phi = torch.stack((torch.zeros_like(pos[:, 0]),
                            -cfg.gravity * wall_kernel_weight(fields)), dim=1)
    approx = -surface + grad_phi          # -int_{Omega_w} phi grad W dA
    scale = float(volume.abs().max())
    return {"relative_error": float((approx - volume).abs().max() / scale),
            "volume_scale": scale, "mode": wall_mode, "sub": int(sub),
            "wall_band": float(volume[(pos[:, 0] < 2.0 * cfg.h())].abs().max())}
