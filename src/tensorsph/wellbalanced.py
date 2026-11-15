"""First-order-consistent pressure gradient: the closed-tank rest gap, closed.

Why this module exists
----------------------
``docs/verification_lake_at_rest_wall_gap.md`` measured the failure this module
targets.  The laterally periodic lake meets ``CASE-SPH-LAKE-AT-REST`` at machine
precision; the *closed* tank does not: ``max|a_x|/g = 0.10 … 0.12`` inside the
wall band, unchanged over a resolution ladder ``dx = 0.050 … 0.020`` (spread
1.23, no trend), two orders of magnitude above the tank centre, and eight orders
of magnitude away from the tolerance.  That audit also localised the defect: at
the worst particle (free-surface row of the two wall columns) the residual is
100 % the *neighbour* pressure sum, because the Tait clamp puts the truncated
surface density below ``rho0`` and therefore ``p_i = 0``; with
``c_i = p_i / rho_i^2 = 0`` the only unknown of the reconstruction -- the mass
field -- has **no handle at all** on that particle's horizontal force.  Four
wall-face pressure fields, three wall-mass conventions and every shipped
operator option were measured there; none closes the gap.

Its conclusion was that "a well-balanced / consistent momentum operator is
required; that is a solver change, not a case change".  This module is that
solver change.

The estimator
-------------
The shipped pair form (``tensorsph.droplet_impact.rhs``, ``formulation="symmetric"``)

.. math::  a_i = -\\sum_j m_j \\,(c_i + c_j)\\, \\nabla_i W_{ij}, \\qquad c = p/\\rho^2

is exact for a constant pressure field on a *complete* support, and the
``difference`` variant ``-(1/rho_i) \\sum_j V_j (p_j - p_i) \\nabla_i W_{ij}`` is
exact for a *linear* pressure field on a complete support -- no better here,
because the free-surface row's support is truncated both laterally (side wall)
and vertically (free surface), and the truncation breaks the symmetry the
cancellation relies on.  Both forms therefore carry an ``O(1)`` spurious
horizontal force at the wall column.

The first-order consistent (kernel-gradient-corrected / moving-least-squares)
estimator instead reconstructs the gradient from the moment matrix,

.. math::  A_i = \\sum_j V_j \\, \\nabla_i W_{ij} \\otimes (x_j - x_i)

.. math::  \\nabla p_i = A_i^{-1} \\sum_j V_j \\, (p_j - p_i)\\, \\nabla_i W_{ij},
           \\qquad a_i = -\\nabla p_i / \\rho_i + g

which returns the **exact** gradient of any linear pressure field at every
particle whose moment matrix is invertible, *independently of how the support is
truncated*: truncation changes ``A_i`` (it can make it anisotropic and
ill-conditioned) but it does not bias ``A_i^{-1} S_i`` for a linear field,
because both sides are built from the same truncated sum.  Since the discrete
equilibrium pressure of this case is (to the measured mirror antisymmetry
``<= 1e-11``) a function of ``y`` alone, its ``x``-gradient is exactly zero and
the wall-band horizontal force has nothing left to act on.  That is the whole
argument, and it is what the measurement in
``docs/verification_lake_at_rest_wellbalanced.md`` tests.

Honest limitations (measured, not asserted away)
-----------------------------------------------
* The corrected operator is **not** momentum-conserving: the pair antisymmetry
  of the symmetric form is destroyed by the particle-local matrix inverse, so
  ``sum_i m_i a_i^x`` is no longer identically zero for a general flow.  In this
  geometry it still vanishes by mirror symmetry, which is why the trade-off is
  invisible here -- the report measures it rather than hiding it, and the
  conservative symmetric form stays the default of the shipped solver.
* The inversion needs a non-singular moment matrix.  A strongly one-sided
  support can approach rank deficiency; ``WellBalancedOptions`` therefore
  regularises the diagonal, refuses to invert above ``max_condition`` and falls
  back to the ``difference`` form there, counting the fallbacks.
* First-order consistency fixes *gradients*; it says nothing about the density
  summation, which stays exactly as shipped (same ``rho``, same Tait pressure,
  same Adami wall-pressure extrapolation).  The vertical balance is therefore
  still closed by the same mass-field reconstruction -- unchanged physics, one
  operator replaced.

Nothing here is specific to the lake case except the wall-pressure convention it
reuses from the shipped solver; the estimator itself is a general SPH momentum
operator.
"""
from __future__ import annotations

import dataclasses

import torch

from tensorsph.droplet_impact import (cubic_spline_kernel, pair_kernels,
                                      tait_pressure, wall_mass)

FORMULATIONS = ("symmetric", "difference", "moment")


@dataclasses.dataclass
class WellBalancedOptions:
    """Knobs of the consistent pressure-gradient operator.

    ``formulation``: ``symmetric`` reproduces the shipped pair form (used to
    validate this re-implementation against ``droplet_impact.rhs``),
    ``difference`` is the uncorrected linear-field form (the control that shows
    the *form* alone does not cure the gap), ``moment`` applies the first-order
    consistent inverse.
    """

    formulation: str = "moment"
    regularization: float = 1.0e-6   # eps * tr(A)/2 added to the diagonal
    max_condition: float = 1.0e12     # above this, fall back to "difference"
    wall_neighbours: bool = True      # keep static wall particles in the sums

    def __post_init__(self) -> None:
        if self.formulation not in FORMULATIONS:
            raise ValueError(f"unknown formulation {self.formulation!r}; "
                             f"expected one of {FORMULATIONS}")


# --------------------------------------------------------------------------- #
# fields
# --------------------------------------------------------------------------- #
def build_fields(state, cfg, mass: torch.Tensor,
                 options: WellBalancedOptions | None = None) -> dict:
    """Densities, pressures and pair geometry, identical to the shipped operator.

    Density is the summation of ``droplet_impact`` (self term included), the
    fluid pressure is the shipped Tait law, and the wall-face pressure is the
    shipped Adami extrapolation
    ``p_w = sum (p_f + rho_f g (y_w - y_f)) w / sum w``.  Nothing is redefined
    here -- :func:`pressure_acceleration` with ``formulation="symmetric"`` is
    checked against the shipped residual by
    ``tests/test_lake_at_rest_wellbalanced.py``.
    """
    options = options or WellBalancedOptions()
    pos, wall = state["pos"], state["wall"]
    n, nw = pos.shape[0], wall.shape[0]
    h = cfg.h()
    device, dtype = pos.device, pos.dtype
    i, j = state["pairs"][:, 0], state["pairs"][:, 1]
    all_pos = torch.cat((pos, wall), dim=0)
    all_mass = torch.cat(
        (mass, wall_mass(state["config"], nw, device).to(dtype)))
    delta = all_pos[i] - all_pos[j]
    wij, grad_w = pair_kernels(delta, h)

    zero = torch.zeros(n, dtype=dtype, device=device)
    rho_all = torch.zeros(n + nw, dtype=dtype, device=device)
    rho_all[:n] = mass * cubic_spline_kernel(zero, h)
    rho_all = rho_all.index_add(0, i, all_mass[j] * wij)
    rho_all = rho_all.index_add(0, j, all_mass[i] * wij)
    rho = rho_all.clamp_min(0.1 * cfg.rho0)
    pressure = tait_pressure(rho, cfg)

    # Adami wall-face pressure from the crossing pairs.
    i_fluid = i < n
    crossing = i_fluid != (j < n)
    fluid_idx = torch.where(i_fluid, i, j)
    wall_idx = (torch.where(i_fluid, j, i) - n).clamp_(0, nw - 1)
    dy_wall = torch.where(i_fluid, -delta[:, 1], delta[:, 1])
    weight = torch.where(crossing, wij, torch.zeros_like(wij))
    source = pressure[fluid_idx] + rho[fluid_idx] * (-cfg.gravity * dy_wall)
    num = torch.zeros(nw, dtype=dtype, device=device).index_add(
        0, wall_idx, source * weight)
    den = torch.zeros(nw, dtype=dtype, device=device).index_add(
        0, wall_idx, weight)
    p_wall = (num / den.clamp_min(1e-30)).clamp_min(0.0)

    # Wall density implied by the extrapolated pressure (shipped convention).
    bulk = cfg.rho0 * cfg.sound_speed ** 2 / cfg.gamma
    rho_wall = (cfg.rho0 * (p_wall / bulk + 1.0).pow(1.0 / cfg.gamma)
                ).clamp_min(cfg.rho0)
    rho_full = torch.cat((rho[:n], rho_wall))
    p_full = torch.cat((pressure[:n], p_wall))
    return {"cfg": cfg, "options": options, "n": n, "nw": nw, "i": i, "j": j,
            "pos": pos, "wall": wall, "all_pos": all_pos, "all_mass": all_mass,
            "h": h, "wij": wij, "grad_w": grad_w, "rho": rho,
            "pressure": pressure, "p_wall": p_wall,
            "rho_full": rho_full, "p_full": p_full,
            "crossing": crossing}


def _pair_selection(f: dict) -> torch.Tensor:
    """Boolean mask of the pairs that enter the fluid acceleration."""
    n = f["n"]
    mask = (f["i"] < n) | (f["j"] < n)
    if not f["options"].wall_neighbours:
        mask = mask & (f["i"] < n) & (f["j"] < n)
    return mask


def _scatter(array: torch.Tensor, index: torch.Tensor,
             values: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(array)
    return out.index_add(0, index, values)


def moment_operator(f: dict) -> dict:
    """Moment matrix ``A``, right-hand side ``S`` and the gradient ``A^-1 S``.

    ``A_i = sum_j V_j grad_i W_ij (x_j - x_i)^T`` and
    ``S_i = sum_j V_j (p_j - p_i) grad_i W_ij``; a pair contributes the same
    outer product to both of its endpoints, weighted by the *other* particle's
    volume, and the same right-hand-side term.
    """
    n, nw = f["n"], f["nw"]
    total = n + nw
    i, j, grad = f["i"], f["j"], f["grad_w"]
    mask = _pair_selection(f)
    i, j, grad = i[mask], j[mask], grad[mask]
    volume = f["all_mass"] / f["rho_full"]
    dx = f["all_pos"][j] - f["all_pos"][i]           # x_j - x_i, shape (P, 2)
    p = f["p_full"]
    dp = p[j] - p[i]

    outer = grad.unsqueeze(2) * dx.unsqueeze(1)      # (P, 2, 2): grad (x) dx
    A = torch.zeros(total, 2, 2, dtype=grad.dtype, device=grad.device)
    A = A.index_add(0, i, volume[j].view(-1, 1, 1) * outer)
    A = A.index_add(0, j, volume[i].view(-1, 1, 1) * outer)
    S = torch.zeros(total, 2, dtype=grad.dtype, device=grad.device)
    S = S.index_add(0, i, (volume[j] * dp).view(-1, 1) * grad)
    S = S.index_add(0, j, (volume[i] * dp).view(-1, 1) * grad)

    return {"A": A, "S": S, "volume": volume}


def _invert_moments(A: torch.Tensor, trace_scale: torch.Tensor,
                    options: WellBalancedOptions) -> tuple:
    """Regularised 2x2 inverse plus the condition number it was taken at."""
    eps = options.regularization
    A = A + (eps * trace_scale).view(-1, 1, 1) * torch.eye(
        2, dtype=A.dtype, device=A.device)
    a, b = A[:, 0, 0], A[:, 0, 1]
    c, d = A[:, 1, 0], A[:, 1, 1]
    det = a * d - b * c
    scale = (a.abs() + d.abs()).clamp_min(1e-300)
    condition = scale.square() / det.abs().clamp_min(1e-300)
    inv = torch.empty_like(A)
    safe = det.abs() > 1e-300
    det_safe = torch.where(safe, det, torch.ones_like(det))
    inv[:, 0, 0] = d / det_safe
    inv[:, 0, 1] = -b / det_safe
    inv[:, 1, 0] = -c / det_safe
    inv[:, 1, 1] = a / det_safe
    usable = safe & (condition < options.max_condition)
    return inv, condition, usable


def moment_anisotropy(state, cfg, mass: torch.Tensor,
                      options: WellBalancedOptions | None = None) -> dict:
    """Truncation signature of the support: the diagonal moment ratios.

    ``A_xx = A_yy = 1`` is a complete support (the discrete kernel integral).
    A free surface cuts the ``yy`` moment roughly in half and a wall adds an
    ``xy`` coupling, which is the geometry that biases the pair form.
    """
    options = options or WellBalancedOptions()
    f = build_fields(state, cfg, mass, options)
    n = f["n"]
    moments = moment_operator(f)
    A = moments["A"][:n]
    pos = state["pos"]
    band = 2.0 * cfg.h()
    wall_band = (pos[:, 0] < band) | (pos[:, 0] > cfg.width - band)
    interior = (pos[:, 1] > band) & (pos[:, 1] < cfg.depth - band)
    a_xx, a_yy, a_xy = A[:, 0, 0], A[:, 1, 1], 0.5 * (A[:, 0, 1] + A[:, 1, 0])
    worst = int(a_yy.abs().argmin())
    return {
        "A_xx_min": float(a_xx.min()), "A_yy_min": float(a_yy.abs().min()),
        "A_yy_over_A_xx_worst": float(a_yy[worst].abs() / a_xx[worst].abs()),
        "A_xy_max_over_A_xx": float(a_xy.abs().max() / a_xx.abs().max()),
        "A_yy_min_wall_band": float(a_yy[wall_band].abs().min()) if bool(wall_band.any()) else None,
        "A_yy_min_interior": float(a_yy[interior].abs().min()) if bool(interior.any()) else None,
        "worst_particle": {"x": float(pos[worst, 0]), "y": float(pos[worst, 1])},
        "per_column_A_yy_min": {
            f"{c:.4f}": float(a_yy[(pos[:, 0] - c).abs() < 0.25 * cfg.dx].abs().min())
            for c in sorted(set(round(float(v), 6) for v in pos[:, 0]))},
    }


# --------------------------------------------------------------------------- #
# pressure acceleration
# --------------------------------------------------------------------------- #
def pressure_acceleration(state, cfg, mass: torch.Tensor,
                          options: WellBalancedOptions | None = None,
                          fields: dict | None = None) -> dict:
    """Total acceleration of the fluid particles under one operator formulation.

    Returns ``{"acceleration", "pressure_acceleration", "horizontal", ...}``
    with ``acceleration`` ordered ``(n_fluid, 2)`` and including the body force
    ``(0, -g)``, i.e. comparable to ``lake_at_rest.residual_report``.
    """
    options = options or WellBalancedOptions()
    f = build_fields(state, cfg, mass, options)
    if fields is not None:
        # Synthetic (p, rho) field on the same pairs: used to measure whether an
        # operator is exactly well-balanced for a *linear* pressure field, which
        # is a property of the estimator and not of the discretisation.
        f = dict(f)
        f.update(fields)
    n, total = f["n"], f["n"] + f["nw"]
    i, j, grad = f["i"], f["j"], f["grad_w"]
    mask = _pair_selection(f)
    i, j, grad = i[mask], j[mask], grad[mask]
    m_full, rho_full, p_full = f["all_mass"], f["rho_full"], f["p_full"]
    volume = m_full / rho_full

    formulation = options.formulation
    diagnostics = {"fallbacks": 0, "formulation": formulation}

    if formulation == "symmetric":
        coef = (p_full / rho_full.square())
        force = torch.zeros(total, 2, dtype=grad.dtype, device=grad.device)
        force = force.index_add(0, i, -m_full[j].view(-1, 1)
                                * grad * (coef[i] + coef[j]).view(-1, 1))
        force = force.index_add(0, j, m_full[i].view(-1, 1)
                                * grad * (coef[i] + coef[j]).view(-1, 1))
        # -sum_j m_j (c_i + c_j) grad_i W_ij is already an acceleration: the
        # volume weight is inside m_j.  (Verified against the shipped operator
        # to 1.4e-14 in tests/test_lake_at_rest_wellbalanced.py.)
        pressure_acc = force
    else:
        # -(1/rho_i) sum_j V_j (p_j - p_i) grad_i W_ij, optionally moment-inverted
        S = torch.zeros(total, 2, dtype=grad.dtype, device=grad.device)
        S = S.index_add(0, i, (volume[j] * (p_full[j] - p_full[i])).view(-1, 1) * grad)
        S = S.index_add(0, j, (volume[i] * (p_full[j] - p_full[i])).view(-1, 1) * grad)
        gradient = S
        if formulation == "moment":
            moments = moment_operator(f)
            A, S_m = moments["A"], moments["S"]
            trace_scale = (A[:, 0, 0].abs() + A[:, 1, 1].abs()).clamp_min(1e-300)
            inv, condition, usable = _invert_moments(A, trace_scale, options)
            gradient = torch.bmm(inv, S_m.unsqueeze(2)).squeeze(2)
            # particles whose support is too one-sided keep the uncorrected sum
            gradient = torch.where(usable.view(-1, 1), gradient, S_m)
            diagnostics["fallbacks"] = int((~usable).sum())
            diagnostics["condition_max"] = float(condition.max())
            diagnostics["condition_median"] = float(condition.median())
        pressure_acc = -gradient / rho_full.view(-1, 1)

    body = torch.zeros(total, 2, dtype=grad.dtype, device=grad.device)
    body[:, 1] = -cfg.gravity
    acc = pressure_acc + body
    diagnostics.update({
        "acceleration": acc[:n],
        "pressure_acceleration": pressure_acc[:n],
        "net_force": (f["all_mass"].view(-1, 1) * acc).sum(dim=0),
    })
    diagnostics["weight"] = float(f["all_mass"][:n].sum() * cfg.gravity)
    return diagnostics


def band_report(state, cfg, mass: torch.Tensor,
                options: WellBalancedOptions | None = None,
                fields: dict | None = None) -> dict:
    """Horizontal/vertical residual split by band, in the anchor's own units."""
    diag = pressure_acceleration(state, cfg, mass, options, fields)
    acc = diag["acceleration"]
    pos = state["pos"]
    vertical = acc[:, 1].abs() / cfg.gravity
    horizontal = acc[:, 0].abs() / cfg.gravity
    band = 2.0 * cfg.h()
    wall_band = (pos[:, 0] < band) | (pos[:, 0] > cfg.width - band)
    interior = (pos[:, 1] > band) & (pos[:, 1] < cfg.depth - band)
    worst = int(horizontal.argmax())
    total_weight = float(mass.sum() * cfg.gravity)
    return {
        "formulation": options.formulation if options else "moment",
        "a_y_max_over_g": float(vertical.max()),
        "a_x_max_over_g": float(horizontal.max()),
        "a_x_wall_band": float(horizontal[wall_band].max()) if bool(wall_band.any()) else None,
        "a_x_interior": float(horizontal[~wall_band].max()) if bool((~wall_band).any()) else None,
        "a_y_interior": float(vertical[interior].max()) if bool(interior.any()) else None,
        "worst_particle": {"x": float(pos[worst, 0]), "y": float(pos[worst, 1])},
        "net_horizontal_force_over_weight": float(
            (mass * acc[:, 0]).sum().abs() / max(total_weight, 1e-30)),
        "condition_max": diag.get("condition_max"),
        "fallbacks": diag.get("fallbacks", 0),
        "acceleration": acc,
    }


# --------------------------------------------------------------------------- #
# mass field that puts *this* operator at rest
# --------------------------------------------------------------------------- #
def _tie_groups(state) -> torch.Tensor:
    columns = state["columns"]
    if columns.dim() == 2:                       # periodic: (n, width) map
        return columns @ torch.arange(
            columns.shape[1], device=columns.device).to(columns.dtype)
    return columns


def fit_mass(state, cfg, options: WellBalancedOptions | None = None,
             prior: dict | None = None, iterations: int = 40,
             tol: float = 1.0e-12, prior_weight: float = 1.0e-8,
             step: float = 1.0e-6, verbose: bool = False) -> dict:
    """Damped Gauss-Newton on ``u = log(m / m_unit)`` for a chosen operator.

    Same statement as :func:`tensorsph.lake_at_rest.reconstruct` -- the mass
    field is the unknown, the *vertical* fluid acceleration is the residual, the
    mirror-pair tie map is the reduction and a Tikhonov term toward the analytic
    profile fills the near-null space -- but the residual is evaluated with this
    module's operator instead of the shipped pair form.  A Jacobian by central
    differences is affordable because the tie map leaves only a few dozen
    unknowns.
    """
    from tensorsph.lake_at_rest import analytic_profile

    options = options or WellBalancedOptions()
    prior = prior or analytic_profile(state, cfg)
    groups = _tie_groups(state).long()
    count = int(groups.max()) + 1
    unit = float(state["m_unit"])
    prior_mass = prior["mass"]
    # group-wise prior: mean of the analytic masses inside each tie group
    prior_u = torch.zeros(count, dtype=prior_mass.dtype, device=prior_mass.device)
    prior_u = prior_u.index_add(0, groups, torch.log(prior_mass / unit))
    prior_u = prior_u / torch.zeros(count, dtype=prior_mass.dtype,
                                    device=prior_mass.device).index_add(
        0, groups, torch.ones_like(prior_mass))

    def masses(u: torch.Tensor) -> torch.Tensor:
        return unit * torch.exp(u[groups])

    def residual(u: torch.Tensor) -> torch.Tensor:
        return pressure_acceleration(state, cfg, masses(u), options)[
            "acceleration"][:, 1]

    u = prior_u.clone()
    history = []
    eye = torch.eye(count, dtype=u.dtype, device=u.device)
    for iteration in range(iterations):
        r = residual(u)
        metric = float(r.abs().max() / cfg.gravity)
        history.append(metric)
        if verbose:
            print(f"    fit[{options.formulation}] it={iteration:2d} "
                  f"|a_y|/g={metric:.3e}", flush=True)
        if metric < tol:
            break
        columns = []
        for k in range(count):
            bump = torch.zeros_like(u)
            bump[k] = step
            columns.append((residual(u + bump) - residual(u - bump)) / (2 * step))
        J = torch.stack(columns, dim=1)                       # (n, count)
        lam = prior_weight ** 0.5
        block = torch.cat((J, lam * eye), dim=0)
        rhs = torch.cat((-r, lam * (prior_u - u)), dim=0)
        delta = torch.linalg.lstsq(block, rhs.unsqueeze(1)).solution.squeeze(1)
        # backtracking line search on |a_y|_inf
        alpha = 1.0
        best = metric
        accepted = None
        for _ in range(24):
            candidate = u + alpha * delta
            trial = float(residual(candidate).abs().max() / cfg.gravity)
            if trial < best:
                accepted, best = candidate, trial
                break
            alpha *= 0.5
        if accepted is None:
            break
        u = accepted
    report = band_report(state, cfg, masses(u), options)
    report.update({"converged": bool(history[-1] < max(tol, 1e-9)),
                   "iterations": len(history), "history": history,
                   "u": u, "mass": masses(u)})
    return report