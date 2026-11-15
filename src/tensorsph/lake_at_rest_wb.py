"""Lake at rest with a *consistent* (well-balanced) momentum operator.

Case-pool anchor ``CASE-SPH-LAKE-AT-REST`` (still water, ``u = 0``, ``eta =
const``, analytic oracle ``p = rho0 g (eta - y)``, tolerance = machine
precision).  The shipped operator (:func:`tensorsph.droplet_impact.rhs`) passes
on the laterally periodic tank (``max|a|/g ~ 1e-13``,
``docs/verification_lake_at_rest_periodic.md``) and **fails** in the closed tank
(``max|a_x|/g ~ 0.12``, resolution invariant,
``docs/verification_lake_at_rest_wall_gap.md``).  That document ends with the
reason and the open item:

    "A well-balanced/consistent momentum operator (kernel gradient correction or
    a boundary-integral wall treatment) is required; that is a solver change, not
    a case change."

This module is that operator, measured rather than asserted.

Why the shipped form cannot hold the closed tank at rest
-------------------------------------------------------
It uses the symmetrised pressure-gradient estimate

    a_i = -sum_j m_j (p_i / rho_i^2 + p_j / rho_j^2) grad_i W_ij.

On an *asymmetric* (wall-truncated) support the kernel sum ``sum_j V_j grad_i
W_ij`` is not zero, so a pressure field that is perfectly hydrostatic still
produces a net force -- the wall-band residual is 100 % the *neighbour* term of
the free-surface particle of the wall column, where ``p_i`` is clamped to zero
by the Tait floor while its sub-surface neighbours still push it (`wall_gap.md`
sections 4.2-4.3).  No mass field can cancel it, and refinement cannot reduce it.

What is done here
-----------------
Two ingredients, both standard and both *data-driven* (no analytic pressure is
injected; the analytic profile is only the reconstruction prior and the oracle):

1. **First-order consistent (MLS) gradient.**  Every particle carries its local
   moment matrix ``M_i = sum_j V_j (x_j - x_i) (x) grad_i W_ij`` and the estimate

       grad^_i p = M_i^-1 sum_j V_j (p_j - p_i) grad_i W_ij

   which reproduces *any* linear pressure field exactly, on *any* support --
   truncated by a wall or by the free surface.  The difference form makes a
   constant field produce exactly zero, so the zeroth-order inconsistency of the
   truncated kernel is annihilated by construction.

2. **Hydrostatic reconstruction of the particle's own pressure.**  The Tait floor
   puts ``p_i = 0`` on the free-surface row although the hydrostatic value there
   is ``rho g dx/2``.  With ``p_i`` wrong, the difference form is biased by
   ``p_i^err / h`` (~38 % of g for that row).  The value used at the particle is
   therefore the *reconstructed* one from its own support,
   ``p^_i = pbar - G . (xbar - x_i)`` (weighted linear least squares over the
   neighbours' EOS pressures).  This is the SPH reading of the hydrostatic
   reconstruction of Audusse et al. (2004) / Xing & Shu (2005): the interface
   value is taken from the well-balanced steady state implied by the data, not
   from the (truncated) local EOS.  ``own_pressure="eos"`` keeps the raw clamped
   value and is measured too, so the cost of ingredient 2 is visible.

Both variants are reported.  Momentum conservation is *not* exact in the
difference form (it is in the shipped symmetrised form); that trade is measured
and stated, not hidden.

Controls
--------
``B1`` flat periodic tank through the same code path (the shipped operator's
PASS must survive the new operator); ``N1`` gravity switched off (the residual
must be the free pressure field, i.e. non-zero); ``N2`` pressure field scaled by
``(1 + eps)`` (a wrong hydrostatic field must *not* be annihilated: the operator
must stay sensitive to the data it is given).  A rest state that is produced by
discarding the pressure data would pass N0 and fail N2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import torch

from .droplet_impact import (cubic_spline_kernel, pair_kernels, tait_pressure,
                             wall_mass)
from .lake_at_rest import (LakeConfig, analytic_profile, build_state,
                           periodic_pairs)

__all__ = [
    "WBOptions",
    "cancellation_pressure",
    "wb_fields",
    "consistent_gradient",
    "wb_accel",
    "wb_reconstruct",
    "wb_run_case",
    "ANCHOR_TOLERANCE",
]

ANCHOR_TOLERANCE = 1.0e-9


@dataclass
class WBOptions:
    """Knobs of the consistent operator (both settings are measured)."""

    own_pressure: str = "reconstructed"   # "reconstructed" | "eos"
    regularisation: float = 0.0           # Tikhonov on M_i, in units of (dx * |M|)
    wall_pressure: str = "adami"


def _wall_pressure_adami(fields: dict) -> torch.Tensor:
    """Adami et al. (2012) wall-face pressure extrapolation, as shipped."""
    cfg, n, nw = fields["cfg"], fields["n"], fields["nw"]
    pressure, rho = fields["pressure"], fields["rho"]
    i, j, delta, wij = fields["i"], fields["j"], fields["delta"], fields["wij"]
    i_fluid = i < n
    crossing = i_fluid != (j < n)
    fluid_idx = torch.where(i_fluid, i, j)
    wall_idx = (torch.where(i_fluid, j, i) - n).clamp_(0, nw - 1)
    dy_wall = torch.where(i_fluid, -delta[:, 1], delta[:, 1])
    weight = torch.where(crossing, wij, torch.zeros_like(wij))
    source = pressure[fluid_idx] + rho[fluid_idx] * (-cfg.gravity * dy_wall)
    num = torch.zeros(nw, dtype=pressure.dtype, device=pressure.device)
    num = num.index_add(0, wall_idx, source * weight)
    den = torch.zeros(nw, dtype=pressure.dtype, device=pressure.device)
    den = den.index_add(0, wall_idx, weight)
    return (num / den.clamp_min(1e-30)).clamp_min(0.0)


def wb_fields(cfg: LakeConfig, mass: torch.Tensor) -> dict:
    """Summation density, Tait pressure, wall pressure and pair geometry."""
    state = build_state(cfg)
    pos, wall = state["pos"], state["wall"]
    n, nw = pos.shape[0], wall.shape[0]
    h = cfg.h()
    i, j = state["pairs"][:, 0], state["pairs"][:, 1]
    all_pos = torch.cat((pos, wall), dim=0)
    all_mass = torch.cat((mass, wall_mass(state["config"], nw, pos.device)))
    delta = all_pos[i] - all_pos[j]
    wrap = cfg.wrap_width
    if wrap is not None:
        delta = torch.stack((delta[:, 0] - wrap * torch.round(delta[:, 0] / wrap),
                             delta[:, 1]), dim=1)
    wij, grad_w = pair_kernels(delta, h)
    zero = torch.zeros(n, dtype=pos.dtype, device=pos.device)
    rho_all = torch.zeros(n + nw, dtype=pos.dtype, device=pos.device)
    rho_all[:n] = mass * cubic_spline_kernel(zero, h)
    rho_all = rho_all.index_add(0, i, all_mass[j] * wij)
    rho_all = rho_all.index_add(0, j, all_mass[i] * wij)
    rho = rho_all.clamp_min(0.1 * cfg.rho0)
    pressure = tait_pressure(rho, cfg)
    fields = dict(cfg=cfg, state=state, pos=pos, wall=wall, n=n, nw=nw, i=i, j=j,
                  all_pos=all_pos, all_mass=all_mass, delta=delta, wij=wij,
                  grad_w=grad_w, rho=rho, pressure=pressure, h=h)
    p_wall = _wall_pressure_adami(fields)
    bulk = cfg.rho0 * cfg.sound_speed ** 2 / cfg.gamma
    rho_wall = (cfg.rho0 * (p_wall / bulk + 1.0).pow(1.0 / cfg.gamma)).clamp_min(cfg.rho0)
    fields["p_wall"] = p_wall
    fields["rho_all"] = torch.cat((rho[:n], rho_wall))
    fields["p_all"] = torch.cat((pressure[:n], p_wall))
    fields["V_all"] = all_mass / fields["rho_all"]
    return fields


def _signed_side(fields: dict):
    """Directed pair list: target, neighbour, signed ``grad_i W`` and ``x_j - x_i``.

    The displacement is taken from the *wrapped* pair separation the kernel was
    evaluated on (``fields["delta"]``).  Using the raw ``x_j - x_i`` instead is
    only correct away from the periodic seam, and there it corrupts the moment
    matrix by the width of the domain: the seam columns then carry a diagonal of
    ``-0.055`` instead of ``-1``, and the horizontal residual appears where the
    geometry is periodic (found by the ``B1`` control, not by inspection).
    """
    n, i, j, grad = fields["n"], fields["i"], fields["j"], fields["grad_w"]
    delta = fields["delta"]
    total = i.numel()
    ar = torch.arange(total, device=i.device)
    # directed: fluid target i keeps +grad, fluid target j gets -grad (antisymmetry)
    tgt = torch.cat((i, j))
    other = torch.cat((j, i))
    gx = torch.cat((grad[:, 0], -grad[:, 0]))
    gy = torch.cat((grad[:, 1], -grad[:, 1]))
    ddx = torch.cat((-delta[:, 0], delta[:, 0]))
    ddy = torch.cat((-delta[:, 1], delta[:, 1]))
    src = torch.cat((ar, ar))
    keep = tgt < n
    return tgt[keep], other[keep], gx[keep], gy[keep], ddx[keep], ddy[keep]


def consistent_gradient(fields: dict, own_pressure: str = "reconstructed",
                        regularisation: float = 0.0) -> dict:
    """MLS/consistent pressure gradient, moment matrix and reconstructed p_i."""
    cfg, n = fields["cfg"], fields["n"]
    pos = fields["pos"]
    dtype, device = pos.dtype, pos.device
    tgt, other, gx, gy, dx, dy = _signed_side(fields)
    V = fields["V_all"][other]

    # moment matrix M_i = sum_j V_j (x_j - x_i) (x) grad_i W_ij
    m00 = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, V * dx * gx)
    m01 = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, V * dx * gy)
    m10 = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, V * dy * gx)
    m11 = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, V * dy * gy)

    p_all = fields["p_all"]
    # weighted least squares over the support: value and slope at x_i
    w = (V * torch.sqrt(gx.square() + gy.square())).clamp_min(0.0)
    s0 = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, w)
    sx = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, w * dx)
    sy = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, w * dy)
    sp = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, w * p_all[other])
    sxx = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, w * dx * dx)
    sxy = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, w * dx * dy)
    syy = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, w * dy * dy)
    sxp = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, w * dx * p_all[other])
    syp = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, w * dy * p_all[other])

    # normal equations for the plane p(x) = p_i + a dx + b dy, centred on x_i
    # (the p_i column is the constant one; solve for (p_i, a, b) jointly)
    A = torch.stack((torch.stack((s0, sx, sy), dim=1),
                     torch.stack((sx, sxx, sxy), dim=1),
                     torch.stack((sy, sxy, syy), dim=1)), dim=1)
    rhs = torch.stack((sp, sxp, syp), dim=1)
    eye = torch.eye(3, dtype=dtype, device=device).expand(n, 3, 3)
    scale = A.diagonal(dim1=1, dim2=2).abs().amax(dim=1).clamp_min(1e-30)
    reg = (regularisation if regularisation > 0.0 else 1e-12 * float(cfg.dx) ** -2)
    solution = torch.linalg.solve(A + reg * scale[:, None, None] * eye, rhs)
    p_fit = solution[:, 0]
    p_own = fields["pressure"][:n] if own_pressure == "eos" else p_fit

    # consistent gradient of the *reconstructed* field:
    # grad p = M^-1 sum_j V_j (p_j - p_own) grad_i W_ij
    diff = p_all[other] - p_own[tgt]
    rx = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, V * diff * gx)
    ry = torch.zeros(n, dtype=dtype, device=device).index_add(0, tgt, V * diff * gy)
    det = (m00 * m11 - m01 * m10)
    dscale = (m00.abs() * m11.abs()).clamp_min(1e-30)
    floor = 1e-6 * dscale
    det = torch.where(det.abs() < floor, torch.sign(det) * floor + (det == 0) * floor, det)
    gpx = (m11 * rx - m01 * ry) / det
    gpy = (-m10 * rx + m00 * ry) / det
    return {"grad_p": torch.stack((gpx, gpy), dim=1), "p_fit": p_fit,
            "grad_fit": solution[:, 1:3],
            "p_own": p_own, "moment": torch.stack((torch.stack((m00, m01), 1),
                                                   torch.stack((m10, m11), 1)), 1),
            "det": det}


def wb_accel(fields: dict, own_pressure: str = "reconstructed",
             form: str = "symmetrised", corrected_gradient: bool = True,
             gravity_on: bool = True, pressure_scale: float = 1.0,
             regularisation: float = 0.0) -> torch.Tensor:
    """Momentum with a reconstructed own-pressure and/or a consistent gradient.

    ``form="symmetrised"``
        the shipped conservative estimate
        ``-sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) grad_i W_ij``; the only thing
        that changes is the value used for ``p_i`` (``own_pressure``) and,
        optionally, the gradient it is multiplied with (``corrected_gradient``).
    ``form="difference"``
        ``-M_i^-1 sum_j V_j (p_j - p_i) grad_i W_ij``, i.e. the difference form
        with the first-order consistent (MLS) gradient -- the textbook
        well-balanced gradient, not momentum conserving.

    Every combination is measured in the study; none is asserted.
    """
    cfg, n = fields["cfg"], fields["n"]
    f = fields
    if pressure_scale != 1.0:
        f = dict(fields)
        f["p_all"] = fields["p_all"] * pressure_scale
    geo = consistent_gradient(f, own_pressure=own_pressure,
                              regularisation=regularisation)
    p_own = geo["p_own"]
    tgt, other, gx, gy, dx, dy = _signed_side(f)
    m_other = f["all_mass"][other]
    rho_all = f["rho_all"]
    if form == "difference":
        grad_x, grad_y = geo["grad_p"][:, 0], geo["grad_p"][:, 1]
    elif form == "lsq":
        # the least-squares plane fit's own slope: exact for any *linear* field on
        # any non-degenerate support, truncated by a wall or by the free surface
        grad_x, grad_y = geo["grad_fit"][:, 0], geo["grad_fit"][:, 1]
    elif form == "symmetrised":
        c_i = p_own / rho_all[:n].square()
        c_j = f["p_all"][other] / rho_all[other].square()
        coef = -(c_i[tgt] + c_j) * m_other
        flux_x, flux_y = coef * gx, coef * gy
        if corrected_gradient:
            moment, det = geo["moment"][tgt], geo["det"][tgt]
            flux_x, flux_y = ((moment[:, 1, 1] * flux_x - moment[:, 0, 1] * flux_y)
                              / det,
                              (-moment[:, 1, 0] * flux_x + moment[:, 0, 0] * flux_y)
                              / det)
        grad_x = torch.zeros(n, dtype=f["pos"].dtype, device=f["pos"].device)
        grad_y = torch.zeros(n, dtype=f["pos"].dtype, device=f["pos"].device)
        grad_x = grad_x.index_add(0, tgt, flux_x)
        grad_y = grad_y.index_add(0, tgt, flux_y)
    else:
        raise ValueError(f"unknown form {form!r}")
    acc = torch.stack((grad_x, grad_y), dim=1)
    if form in ("difference", "lsq"):
        # -grad p / rho; the symmetrised form already carries 1/rho^2 per term
        acc = acc / rho_all[:n, None]
    if gravity_on:
        acc = acc + torch.tensor([0.0, -float(cfg.gravity)],
                                 dtype=acc.dtype, device=acc.device)
    return acc


def cancellation_pressure(fields: dict) -> dict:
    """The own-pressure that *would* zero the horizontal residual, vs the physical one.

    The shipped form is
    ``a_x = -(p_i/rho_i^2) sum_j m_j dW/dx - sum_j m_j (p_j/rho_j^2) dW/dx``,
    so setting ``a_x = 0`` gives a unique demand on the particle's own pressure,

        p_i^req = -rho_i^2 sum_j m_j (p_j/rho_j^2) dW/dx  /  sum_j m_j dW/dx .

    Comparing that with the hydrostatic value ``rho0 g (eta - y_i)`` says whether
    the gap can be closed by *any* pressure closure, or only by a support /
    boundary-integral change: if ``p_i^req`` is several times the physical value,
    no physical pressure can cancel the truncated wall-side flux.
    """
    cfg, n = fields["cfg"], fields["n"]
    tgt, other, gx, gy, _dx, _dy = _signed_side(fields)
    m_other = fields["all_mass"][other]
    rho_all = fields["rho_all"]
    demand = torch.zeros(n, dtype=m_other.dtype, device=m_other.device).index_add(
        0, tgt, m_other * (fields["p_all"][other] / rho_all[other].square()) * gx)
    deficiency = torch.zeros(n, dtype=m_other.dtype, device=m_other.device).index_add(
        0, tgt, m_other * gx)
    p_req = -(rho_all[:n].square() * demand) / deficiency
    y = fields["pos"][:, 1]
    p_phys = (cfg.rho0 * cfg.gravity * (cfg.depth - y)).clamp_min(0.0)
    band = (fields["pos"][:, 0] < 2.0 * fields["h"]) | (
        fields["pos"][:, 0] > cfg.width - 2.0 * fields["h"])
    ratio = p_req / p_phys.clamp_min(1e-12)
    # the diagnostic belongs to the *shipped* baseline residual (the one the
    # anchor's FAIL is stated on), not to a corrected variant
    acc = wb_accel(fields, own_pressure="eos", corrected_gradient=False)
    band_idx = torch.where(band)[0]
    worst = int(band_idx[acc[band, 0].abs().argmax()])
    return {
        "deficiency": deficiency, "p_required": p_req, "p_physical": p_phys,
        "p_eos": fields["pressure"][:n], "wall_band": band,
        "worst_particle": worst,
        "worst_position": [float(v) for v in fields["pos"][worst]],
        "worst_p_required": float(p_req[worst]),
        "worst_p_physical": float(p_phys[worst]),
        "worst_p_eos": float(fields["pressure"][worst]),
        "worst_ax_over_g": float(acc[worst, 0] / cfg.gravity),
        "worst_deficiency": float(deficiency[worst]),
        "worst_required_over_physical": float(ratio[worst]),
        "wall_band_residual_ax_max_over_g": float(acc[band, 0].abs().max() / cfg.gravity),
        "interior_residual_ax_max_over_g": float(acc[~band, 0].abs().max() / cfg.gravity)
        if bool((~band).any()) else None,
    }


def _tie_mapping(cfg: LakeConfig, state: dict) -> torch.Tensor:
    """The reconstruction's unknown map, taken from the case itself.

    Rows under lateral periodicity, mirror pairs in the closed tank -- grouped
    per *row* in both cases, because the equilibrium mass field is hydrostatic:
    tying a whole column to one mass (which is what "column" first suggests, and
    what a first version of this module did) destroys the profile and the
    reconstruction starts at ``max|a|/g = 26`` instead of ``3e-2``.
    """
    return state["columns"]


def wb_reconstruct(cfg: LakeConfig, own_pressure: str = "reconstructed",
                   form: str = "symmetrised", corrected_gradient: bool = True,
                   prior: torch.Tensor | None = None, tol: float = 1e-12,
                   max_iter: int = 40, target: str = "vertical",
                   verbose: bool = False) -> dict:
    """Gauss-Newton on the masses, closing the *vertical* balance.

    ``target="vertical"`` (the case's own convention, `lake_at_rest.reconstruct`)
    solves for the mass field that makes ``a_y = 0``; the horizontal residual is
    then the operator's number, not the mass field's.  ``target="both"`` tries to
    zero both components at once -- measured and reported, but it is not a fair
    comparison: masses provably cannot cancel a horizontal residual that lives in
    the wall-column surface row (`wall_gap.md` 4.3), so the solver spends the
    vertical balance trying to buy something that is not for sale.
    """
    state = build_state(cfg)
    pos, device, dtype = state["pos"], state["pos"].device, state["pos"].dtype
    mapping = _tie_mapping(cfg, state)
    unique = int(mapping.max().item()) + 1
    m_unit = state["m_unit"]
    prior = analytic_profile(state, cfg)["mass"] if prior is None else prior
    u_prior = torch.log(prior / m_unit)
    sums = torch.zeros(unique, dtype=dtype, device=device).index_add(0, mapping, u_prior)
    counts = torch.zeros(unique, dtype=dtype, device=device).index_add(
        0, mapping, torch.ones_like(u_prior))
    u = sums / counts
    u_prior_unique = u.clone()
    g = torch.tensor([0.0, -float(cfg.gravity)], dtype=dtype, device=device)

    def residual(uu: torch.Tensor) -> torch.Tensor:
        acc = wb_accel(wb_fields(cfg, m_unit * torch.exp(uu[mapping])),
                       own_pressure=own_pressure, form=form,
                       corrected_gradient=corrected_gradient, gravity_on=False)
        out = acc + g
        if target == "vertical":
            out = out[:, 1]
        return out.reshape(-1) / cfg.gravity

    history = []
    lam = float(cfg.prior_weight)
    for it in range(max_iter):
        r = residual(u)
        worst = float(r.abs().max())
        merit = float(torch.linalg.vector_norm(r)) / math.sqrt(max(r.numel(), 1))
        history.append(worst)
        if verbose:
            print(f"  iter {it:2d}  max|a|/g = {worst:.3e}  rms = {merit:.3e}")
        if worst < tol:
            break
        jac = torch.autograd.functional.jacobian(residual, u)
        rows = [jac, lam * torch.eye(unique, dtype=dtype, device=device)]
        rhs_vec = [-r, lam * (u_prior_unique - u)]
        step = torch.linalg.lstsq(torch.cat(rows, dim=0),
                                  torch.cat(rhs_vec, dim=0)).solution
        alpha, accepted = 1.0, False
        for _ in range(30):
            candidate = u + alpha * step
            if float(torch.linalg.vector_norm(residual(candidate))) < merit * math.sqrt(
                    max(r.numel(), 1)):
                u, accepted = candidate, True
                break
            alpha *= 0.5
        if not accepted:
            break
    mass = m_unit * torch.exp(u[mapping])
    fields = wb_fields(cfg, mass)
    acc = wb_accel(fields, own_pressure=own_pressure, form=form,
                   corrected_gradient=corrected_gradient)
    return {"mass": mass, "fields": fields, "accel": acc,
            "iterations": len(history), "residual_history": history,
            "converged": bool(history and history[-1] < tol),
            "max_a_over_g": float(torch.linalg.vector_norm(acc, dim=1).max() / cfg.gravity),
            "max_ax_over_g": float(acc[:, 0].abs().max() / cfg.gravity),
            "max_ay_over_g": float(acc[:, 1].abs().max() / cfg.gravity),
            "u_unique": u}


def _band_stats(cfg: LakeConfig, pos: torch.Tensor, acc: torch.Tensor,
                h: float) -> dict:
    band = (pos[:, 0] < 2.0 * h) | (pos[:, 0] > cfg.width - 2.0 * h)
    ratio = torch.linalg.vector_norm(acc, dim=1) / cfg.gravity
    return {
        "max_a_over_g": float(ratio.max()),
        "max_ax_over_g": float(acc[:, 0].abs().max() / cfg.gravity),
        "max_ay_over_g": float(acc[:, 1].abs().max() / cfg.gravity),
        "max_a_over_g_wall_band": float(ratio[band].max()) if bool(band.any()) else 0.0,
        "max_a_over_g_interior": float(ratio[~band].max()) if bool((~band).any()) else 0.0,
        "wall_band_particles": int(band.sum()),
        "particles": int(pos.shape[0]),
    }


def wb_run_case(cfg: LakeConfig, own_pressure: str = "reconstructed",
                reconstruct_mass: bool = True) -> dict:
    """Certify (or refute) rest for one configuration with the consistent operator."""
    state = build_state(cfg)
    h = cfg.h()
    m_unit = state["m_unit"]
    lattice = wb_fields(cfg, torch.full_like(state["pos"][:, 0], m_unit))
    lattice_acc = wb_accel(lattice, own_pressure=own_pressure)
    prior = analytic_profile(state, cfg)["mass"]
    out = {
        "geometry": {"periodic_x": bool(cfg.periodic_x), "dx": cfg.dx,
                     "depth": cfg.depth, "width": cfg.width,
                     "c0": cfg.sound_speed, "gravity": cfg.gravity,
                     "h_over_dx": h / cfg.dx, "rho0": cfg.rho0},
        "own_pressure": own_pressure,
        "lattice_mass": _band_stats(cfg, state["pos"], lattice_acc, h),
        "prior_mass": _band_stats(cfg, state["pos"],
                                  wb_accel(wb_fields(cfg, prior),
                                           own_pressure=own_pressure), h),
        "controls": {
            "gravity_off": _band_stats(
                cfg, state["pos"],
                wb_accel(lattice, own_pressure=own_pressure, gravity_on=False), h),
            "pressure_scaled_1pct": _band_stats(
                cfg, state["pos"],
                wb_accel(lattice, own_pressure=own_pressure, pressure_scale=1.01), h),
            "density_20pct_low": _band_stats(
                cfg, state["pos"],
                wb_accel(wb_fields(cfg, 0.8 * m_unit * torch.ones_like(state["pos"][:, 0])),
                         own_pressure=own_pressure), h),
        },
    }
    if reconstruct_mass:
        rec = wb_reconstruct(cfg, own_pressure=own_pressure)
        out["reconstruct"] = {key: rec[key] for key in
                              ("iterations", "converged", "residual_history",
                               "max_a_over_g", "max_ax_over_g", "max_ay_over_g")}
        out["reconstruct"].update(_band_stats(cfg, state["pos"], rec["accel"], h))
        out["reconstruct"]["mass_modulation"] = float(
            (rec["mass"] / m_unit).max() / (rec["mass"] / m_unit).min())
    return out