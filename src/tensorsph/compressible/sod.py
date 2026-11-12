"""Compressible SPH for the Sod (1978) shock tube.

Physics
-------
1-D Euler equations for an ideal gas (gamma = 1.4), Sod's Riemann problem:

    (rho, u, p) = (1.0,   0.0, 1.0)   for x < 0.5
    (rho, u, p) = (0.125, 0.0, 0.1)   for x > 0.5

Reference: G. A. Sod, "A survey of several finite difference methods for
systems of nonlinear hyperbolic conservation laws", J. Comput. Phys. 27(1):1-31
(1978), doi:10.1016/0021-9991(78)90023-2.

Why not WCSPH
-------------
The weakly compressible machinery in :mod:`tensorsph.wcsph.sod_shock` uses a
Tait (barotropic) equation of state, i.e. p = p(rho).  A shock tube is *not*
barotropic: entropy is generated at the shock.  Its own validation report
(docs/validation/wcsph_validation/REPORT.md) records 64% L2 density error, a
104% shock-position error and a negative convergence order for exactly this
case.  This module solves the genuinely compressible system instead, carrying
the specific internal energy as an independent variable.

Numerics
--------
Lagrangian SPH, ideal-gas EOS p = (gamma-1) rho e:

    d rho_i/dt = sum_j m_j (v_i - v_j) . grad_i W_ij                   (continuity)
    d v_i/dt   = -sum_j m_j Q_ij grad_i W_ij                          (momentum)
    d e_i/dt   = 1/2 sum_j m_j Q_ij (v_i - v_j) . grad_i W_ij          (energy)

    Q_ij = (p_i + p_j)/(rho_i rho_j) + Pi_ij                          (pressure)

Two points that matter for accuracy and are *measured* in
docs/validation/compressible_sod/REPORT.md:

1. ``Q`` uses the arithmetic pressure average ``(p_i+p_j)/(rho_i rho_j)``, not
   ``p_i/rho_i^2 + p_j/rho_j^2``.  The latter is zeroth-order inconsistent at a
   contact discontinuity: for a *static* 1:8 density jump at uniform pressure it
   produces a spurious acceleration of 8.8e3 (measured,
   ``tests/test_compressible_sod.py::test_contact_discontinuity_equilibrium``),
   which destroys pressure equilibrium across the contact.  The arithmetic form
   gives exactly zero and is the non-Riemannian limit of the Godunov-SPH
   pressure ``2 p*/(rho_i rho_j)`` (Inutsuka 2002, J. Comput. Phys. 179:238,
   doi:10.1006/jcph.2002.7064).

2. Monaghan's artificial viscosity (Monaghan & Gingold 1983; Monaghan 1992)

       Pi_ij = (-alpha c_bar_ij mu_ij + beta mu_ij^2)/rho_bar_ij ,  v_ij.x_ij < 0
       mu_ij = h (v_i - v_j).(x_i - x_j) / (|x_ij|^2 + eta^2 h^2)

   is active in compression only, and the pair sum is antisymmetric, so mass,
   momentum and (for alpha_u = 0) total energy are conserved by construction.

3. The smoothing length is kept *fixed* relative to the initial lattice, but the
   ratio h/dx must be large enough to survive the expansion of the star-left
   plateau (density drops by 2.3x, spacing grows by 2.3x).  With h = 1.3 dx only
   1-2 neighbours per particle remain in that plateau and the solution does not
   converge (measured: rho*_L error +25%, p*_L error +40%, independent of N);
   h = 2.5 dx restores the neighbour count and gives <1.5% errors.  See the
   ``h_factor`` sensitivity study in the report.

4. Time integration: second-order midpoint (RK2) with a CFL step
   dt = cfl min_i h_i/(c_i + |v_i|), additionally limited by the viscous
   stability condition dt <= 0.25 h^2/nu.

5. Walls: mirror ghost particles (state mirrored, velocity reversed), exact for
   a uniform state, and placed far enough away that no wave reaches them before
   t = 0.2.

6. SPH cannot represent a true discontinuity: the diaphragm is smeared over
   ``init_smoothing*dx`` with a tanh profile shared by rho, p and u (isentropic
   smearing).  A *sharp* start is unstable (measured: overflow within ~10 steps)
   because two adjacent particles receive O(1e3) opposite accelerations; the
   smearing width is reported as a sensitivity parameter.

Kernels are the 1-D cubic B-spline of :mod:`tensorsph.wcsph.sod_shock`
(re-used, normalised to unity in 1-D, support radius 2h).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

from ..wcsph.sod_shock import (
    _cubic_w_1d,
    _cubic_dw_1d,
    exact_riemann_solution,
    exact_riemann_profile,
)

__all__ = [
    "SodSPHConfig",
    "run_sod_sph",
    "exact_star_state",
    "exact_wave_positions",
    "evaluate_metrics",
    "contact_force_check",
    "WAVE_WINDOW_FRACTION",
]

#: Fraction of each plateau width trimmed at both ends when averaging a plateau
#: value.  Predeclared and *not* tuned: SPH smears a discontinuity over ~3-4 h,
#: i.e. < 5% of the narrowest plateau (star-right, width ~0.165).
WAVE_WINDOW_FRACTION = 0.10


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class SodSPHConfig:
    """Configuration of the 1-D compressible SPH Sod shock tube."""

    n_particles: int = 800
    x_left: float = 0.0
    x_right: float = 1.0
    x_diaphragm: float = 0.5
    gamma: float = 1.4
    rho_L: float = 1.0
    p_L: float = 1.0
    u_L: float = 0.0
    rho_R: float = 0.125
    p_R: float = 0.1
    u_R: float = 0.0
    t_end: float = 0.2
    h_factor: float = 2.5          # h = h_factor * dx  (see module docstring, item 3)
    init_smoothing: float = 1.5    # diaphragm smeared over init_smoothing*dx
    alpha_av: float = 1.0          # Monaghan artificial viscosity (linear)
    beta_av: float = 2.0           # Monaghan artificial viscosity (quadratic)
    eta2: float = 0.01             # eta^2 = eta2 * h^2 (softening inside mu_ij)
    alpha_u: float = 0.0           # artificial thermal conduction (Price 2008)
    cfl: float = 0.15
    pressure_form: str = "arithmetic"   # "arithmetic" | "p_over_rho2"
    max_steps: int = 400000

    @property
    def dx(self) -> float:
        return (self.x_right - self.x_left) / self.n_particles

    @property
    def h(self) -> float:
        return self.h_factor * self.dx

    @property
    def support(self) -> float:
        return 2.0 * self.h

    def as_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d.update({"dx": self.dx, "h": self.h, "support": self.support})
        return d


# ---------------------------------------------------------------------------
# exact solution
# ---------------------------------------------------------------------------
def exact_star_state(gamma: float = 1.4,
                     rho_L: float = 1.0, p_L: float = 1.0, u_L: float = 0.0,
                     rho_R: float = 0.125, p_R: float = 0.1,
                     u_R: float = 0.0) -> Dict[str, float]:
    """Exact star-region state and wave speeds of the Sod Riemann problem.

    Star pressure/velocity come from the two-shock/two-rarefaction iterative
    solver of :mod:`tensorsph.wcsph.sod_shock`; star densities and wave speeds
    are the closed forms of Toro (1999) ch. 4 for gamma-law gases.
    """
    p_star, u_star = exact_riemann_solution(gamma, rho_L, p_L, u_L, rho_R, p_R, u_R)
    c_L = math.sqrt(gamma * p_L / rho_L)
    c_R = math.sqrt(gamma * p_R / rho_R)
    gm1, gp1 = gamma - 1.0, gamma + 1.0

    rho_star_L = rho_L * (p_star / p_L) ** (1.0 / gamma)          # isentropic rarefaction
    rho_star_R = rho_R * ((p_star / p_R + gm1 / gp1)
                          / (gm1 / gp1 * p_star / p_R + 1.0))     # Rankine-Hugoniot
    c_star_L = c_L * (p_star / p_L) ** (gm1 / (2.0 * gamma))
    shock_speed = u_R + c_R * math.sqrt(gp1 / (2.0 * gamma) * p_star / p_R
                                       + gm1 / (2.0 * gamma))
    return {
        "p_star": p_star, "u_star": u_star,
        "rho_star_L": rho_star_L, "rho_star_R": rho_star_R,
        "c_L": c_L, "c_R": c_R, "c_star_L": c_star_L,
        "rarefaction_head_speed": u_L - c_L,
        "rarefaction_tail_speed": u_star - c_star_L,
        "contact_speed": u_star,
        "shock_speed": shock_speed,
    }


def exact_wave_positions(t: float, x0: float = 0.5, **kw) -> Dict[str, float]:
    """Positions of rarefaction head/tail, contact discontinuity and shock at t."""
    s = exact_star_state(**kw)
    return {
        "t": t,
        "x_rarefaction_head": x0 + s["rarefaction_head_speed"] * t,
        "x_rarefaction_tail": x0 + s["rarefaction_tail_speed"] * t,
        "x_contact": x0 + s["contact_speed"] * t,
        "x_shock": x0 + s["shock_speed"] * t,
        "p_star": s["p_star"],
        "u_star": s["u_star"],
        "rho_star_L": s["rho_star_L"],
        "rho_star_R": s["rho_star_R"],
    }


# ---------------------------------------------------------------------------
# SPH machinery
# ---------------------------------------------------------------------------
def _extend_with_ghosts(x, rho, v, e, m, x_L, x_R, support, h):
    """Mirror ghost particles at both walls.

    Ghosts copy the thermodynamic state of their mirror image and reverse the
    velocity, so a uniform state produces zero net acceleration (exact wall
    symmetry).  Returned arrays are (X, RHO, V, E, M, n_real, H).
    """
    n_real = len(x)
    gl = np.where(x - x_L < support)[0]
    gr = np.where(x_R - x < support)[0]
    X = np.concatenate([x, 2.0 * x_L - x[gl], 2.0 * x_R - x[gr]])
    RHO = np.concatenate([rho, rho[gl], rho[gr]])
    V = np.concatenate([v, -v[gl], -v[gr]])
    E = np.concatenate([e, e[gl], e[gr]])
    M = np.concatenate([m, m[gl], m[gr]])
    H = np.concatenate([h, h[gl], h[gr]])
    return X, RHO, V, E, M, n_real, H


def _bc(idx, val, n):
    """Segment sum out[idx] += val (fast, exact replacement of np.add.at)."""
    return np.bincount(idx, weights=val, minlength=n)


def _pairs(X, h, support):
    """Neighbour pairs (i<j) inside the pair-averaged kernel support."""
    tree = cKDTree(X[:, None])
    pairs = tree.query_pairs(support, output_type="ndarray")
    if pairs.size == 0:
        return None
    i, j = pairs[:, 0], pairs[:, 1]
    dxij = X[i] - X[j]
    r = np.abs(dxij)
    h_ij = 0.5 * (h[i] + h[j])
    inside = r < 2.0 * h_ij
    return i[inside], j[inside], dxij[inside], r[inside], h_ij[inside]


def _derivs(x, rho, v, e, m, cfg: SodSPHConfig):
    """SPH rates (drho, dv, de) plus pressure and sound speed of the real particles."""
    gamma = cfg.gamma
    h = np.full(len(x), cfg.h)
    eta2 = cfg.eta2

    p = np.maximum((gamma - 1.0) * rho * e, 1e-12)
    cs = np.sqrt(gamma * p / rho)

    X, RHO, V, E, M, n_real, H = _extend_with_ghosts(
        x, rho, v, e, m, cfg.x_left, cfg.x_right, cfg.support, h)
    P = np.maximum((gamma - 1.0) * RHO * E, 1e-12)
    CS = np.sqrt(gamma * P / RHO)

    pr = _pairs(X, H, 2.0 * H.max())
    if pr is None:
        zero = np.zeros(n_real)
        return zero, zero.copy(), zero.copy(), p, cs
    i, j, dxij, r, h_ij = pr

    # dW/dx_i = (dW/dr) sign(x_i - x_j); the kernel helper carries the sign, so
    # it *is* grad_i W_ij in 1-D, and is antisymmetric under i <-> j as required.
    grad = _cubic_dw_1d(dxij, h_ij)
    vij = V[i] - V[j]
    vdotr = vij * dxij
    mu = h_ij * vdotr / (r * r + eta2 * h_ij * h_ij)
    cs_bar = 0.5 * (CS[i] + CS[j])
    rho_bar = 0.5 * (RHO[i] + RHO[j])
    Pi = np.where(vdotr < 0.0,
                 (-cfg.alpha_av * cs_bar * mu + cfg.beta_av * mu * mu) / rho_bar,
                 0.0)

    if cfg.pressure_form == "arithmetic":
        Q = (P[i] + P[j]) / (RHO[i] * RHO[j]) + Pi
    else:
        Q = P[i] / RHO[i] ** 2 + P[j] / RHO[j] ** 2 + Pi

    uij_grad = vij * grad
    n_ext = len(X)
    a_ext = _bc(j, M[i] * Q * grad, n_ext) - _bc(i, M[j] * Q * grad, n_ext)
    half = 0.5 * uij_grad
    de_ext = (_bc(i, M[j] * Q * half, n_ext)
              + _bc(j, M[i] * Q * half, n_ext))

    if cfg.alpha_u > 0.0:
        # artificial thermal conduction (Price 2008 eq. 33, 1-D form)
        mu_u = h_ij * dxij * grad / (r * r + eta2 * h_ij * h_ij)
        cond = cfg.alpha_u * cs_bar * (E[i] - E[j]) / rho_bar * mu_u
        de_ext = de_ext + _bc(i, M[j] * cond, n_ext) + _bc(j, M[i] * cond, n_ext)

    drho = (_bc(i, M[j] * uij_grad, n_ext) + _bc(j, M[i] * uij_grad, n_ext))[:n_real]
    return drho, a_ext[:n_real], de_ext[:n_real], p, cs


def _timestep(cfg: SodSPHConfig, rho, v, e) -> float:
    p = np.maximum((cfg.gamma - 1.0) * rho * e, 1e-12)
    cs = np.sqrt(cfg.gamma * p / rho)
    dt_cfl = cfg.cfl * cfg.h / np.max(cs + np.abs(v))
    nu_max = cfg.alpha_av * np.max(cs) * cfg.h
    dt_visc = 0.25 * cfg.h * cfg.h / (nu_max + 1e-30)
    return float(min(dt_cfl, dt_visc))


def contact_force_check(cfg: Optional[SodSPHConfig] = None,
                        rho_L: float = 1.0, rho_R: float = 0.125,
                        p: float = 1.0) -> Dict[str, float]:
    """Spurious acceleration on a *static* contact discontinuity (exact: 0).

    Uniform pressure, 1:8 density jump, zero velocity.  This is the diagnostic
    that discriminates the two momentum discretisations; it is asserted in the
    test-suite.
    """
    cfg = cfg or SodSPHConfig(n_particles=400)
    dx = cfg.dx
    x = cfg.x_left + (np.arange(cfg.n_particles) + 0.5) * dx
    rho = np.where(x < cfg.x_diaphragm, rho_L, rho_R)
    pp = np.full(cfg.n_particles, p)
    e = pp / ((cfg.gamma - 1.0) * rho)
    m = rho * dx
    out = {}
    for form in ("p_over_rho2", "arithmetic"):
        cfg2 = replace(cfg, pressure_form=form)
        acc = _derivs(x, rho, np.zeros_like(x), e, m, cfg2)[1]
        out[form] = float(np.max(np.abs(acc)))
    return out


def run_sod_sph(cfg: SodSPHConfig, output_times: Optional[List[float]] = None) -> dict:
    """Run the compressible SPH Sod shock tube; return snapshots + diagnostics."""
    gamma = cfg.gamma
    dx = cfg.dx
    N = cfg.n_particles
    x_L, x_R = cfg.x_left, cfg.x_right

    x = x_L + (np.arange(N) + 0.5) * dx
    if cfg.init_smoothing > 0.0:
        f = 0.5 * (1.0 - np.tanh((x - cfg.x_diaphragm) / (cfg.init_smoothing * dx)))
        rho = (cfg.rho_R + (cfg.rho_L - cfg.rho_R) * f).astype(np.float64)
        p0 = (cfg.p_R + (cfg.p_L - cfg.p_R) * f).astype(np.float64)
        v = (cfg.u_R + (cfg.u_L - cfg.u_R) * f).astype(np.float64)
    else:
        rho = np.where(x < cfg.x_diaphragm, cfg.rho_L, cfg.rho_R).astype(np.float64)
        p0 = np.where(x < cfg.x_diaphragm, cfg.p_L, cfg.p_R).astype(np.float64)
        v = np.where(x < cfg.x_diaphragm, cfg.u_L, cfg.u_R).astype(np.float64)
    e = p0 / ((gamma - 1.0) * rho)
    m = rho * dx                       # 1-D SPH: mass per unit area (frozen)
    m0 = m.copy()

    energy_0 = float(np.sum(m * (0.5 * v * v + e)))
    if output_times is None:
        output_times = [cfg.t_end]
    output_times = sorted(float(t) for t in output_times)
    out_idx = 0
    snapshots: List[dict] = []

    t = 0.0
    step = 0
    while t < cfg.t_end - 1e-15 and step < cfg.max_steps:
        dt = min(_timestep(cfg, rho, v, e), cfg.t_end - t)

        # --- RK2 (midpoint) ---
        drho1, a1, de1, _, _ = _derivs(x, rho, v, e, m, cfg)
        rho_m = np.maximum(rho + 0.5 * dt * drho1, 1e-12)
        v_m = v + 0.5 * dt * a1
        e_m = np.maximum(e + 0.5 * dt * de1, 1e-12)
        x_m = x + 0.5 * dt * v_m

        drho2, a2, de2, _, _ = _derivs(x_m, rho_m, v_m, e_m, m, cfg)

        rho = np.maximum(rho + dt * drho2, 1e-12)
        v = v + dt * a2
        e = np.maximum(e + dt * de2, 1e-12)
        x = x + dt * v_m
        t += dt
        step += 1

        while out_idx < len(output_times) and t >= output_times[out_idx] - 1e-15:
            snapshots.append(_snapshot(x, rho, v, e, m, t, output_times[out_idx], cfg))
            out_idx += 1

    if out_idx < len(output_times):        # always store the final state
        snapshots.append(_snapshot(x, rho, v, e, m, t, output_times[-1], cfg))

    energy_1 = float(np.sum(m * (0.5 * v * v + e)))
    return {
        "config": cfg.as_dict(),
        "steps": step,
        "t_final": t,
        "snapshots": snapshots,
        "mass_initial": float(np.sum(m0)),
        "mass_final": float(np.sum(m)),
        "momentum_final": float(np.sum(m * v)),
        "energy_initial": energy_0,
        "energy_final": energy_1,
        "energy_rel_change": (energy_1 - energy_0) / energy_0,
        "neighbours": _neighbour_stats(x, m, cfg),
    }


def _neighbour_stats(x, m, cfg: SodSPHConfig) -> dict:
    X, RHO, V, E, M, n_real, H = _extend_with_ghosts(
        x, np.ones_like(x), x, x, m, cfg.x_left, cfg.x_right, cfg.support,
        np.full(len(x), cfg.h))
    tree = cKDTree(X[:, None])
    counts = np.array([len(ii) - 1 for ii in tree.query_ball_point(X[:n_real, None],
                                                                  2.0 * cfg.h)],
                      dtype=float)
    return {"min": float(counts.min()), "mean": float(counts.mean()),
            "max": float(counts.max())}


def _snapshot(x, rho, v, e, m, t_actual, t_label, cfg: SodSPHConfig) -> dict:
    p = np.maximum((cfg.gamma - 1.0) * rho * e, 1e-12)
    order = np.argsort(x)
    return {
        "t": float(t_label),
        "t_actual": float(t_actual),
        "x": x[order].copy(),
        "rho": rho[order].copy(),
        "u": v[order].copy(),
        "p": p[order].copy(),
        "e": e[order].copy(),
    }


# ---------------------------------------------------------------------------
# metrics vs the exact Riemann solution
# ---------------------------------------------------------------------------
def _avg(x, q, lo, hi):
    sel = (x >= lo) & (x <= hi)
    if not np.any(sel):
        return float("nan")
    return float(np.mean(q[sel]))


def evaluate_metrics(snap: dict, cfg: SodSPHConfig) -> dict:
    """Plateau / wave metrics of one SPH snapshot versus the exact Sod solution.

    Measurement windows are derived from the *exact* wave positions, never from
    the SPH result: each plateau is trimmed by ``WAVE_WINDOW_FRACTION`` of its
    own width at both ends, so the windows are predeclared and not tuned.
    """
    t = snap["t"]
    x = np.asarray(snap["x"], dtype=float)
    rho = np.asarray(snap["rho"], dtype=float)
    p = np.asarray(snap["p"], dtype=float)
    u = np.asarray(snap["u"], dtype=float)

    kw = dict(gamma=cfg.gamma, rho_L=cfg.rho_L, p_L=cfg.p_L, u_L=cfg.u_L,
              rho_R=cfg.rho_R, p_R=cfg.p_R, u_R=cfg.u_R)
    wp = exact_wave_positions(t, cfg.x_diaphragm, **kw)
    x_head, x_tail = wp["x_rarefaction_head"], wp["x_rarefaction_tail"]
    x_con, x_sho = wp["x_contact"], wp["x_shock"]

    f = WAVE_WINDOW_FRACTION
    wl_lo, wl_hi = cfg.x_left + 0.02, x_head - f * (x_head - cfg.x_left)
    sl_w = x_con - x_tail
    sl_lo, sl_hi = x_tail + f * sl_w, x_con - f * sl_w
    sr_w = x_sho - x_con
    sr_lo, sr_hi = x_con + f * sr_w, x_sho - f * sr_w
    wr_lo, wr_hi = x_sho + 0.02, cfg.x_right - 0.02

    def _steepest(lo, hi, q):
        sel = (x >= lo) & (x <= hi)
        if np.count_nonzero(sel) < 5:
            return float("nan")
        xs, qs = x[sel], q[sel]
        g = np.abs(np.gradient(qs, xs))
        return float(xs[int(np.argmax(g))])

    x_sho_meas = _steepest(x_con + 0.3 * (x_sho - x_con), cfg.x_right - 0.02, rho)
    x_con_meas = _steepest(x_tail + 0.5 * (x_con - x_tail),
                           x_con + 0.7 * (x_sho - x_con), rho)

    def rel_err(got, ref):
        return float((got - ref) / ref) if ref != 0 else float("nan")

    metrics = {
        "t": t,
        "windows": {"left_plateau": [wl_lo, wl_hi], "star_left": [sl_lo, sl_hi],
                    "star_right": [sr_lo, sr_hi], "right_plateau": [wr_lo, wr_hi]},
        "exact": {"rho_L": cfg.rho_L, "rho_star_L": wp["rho_star_L"],
                  "rho_star_R": wp["rho_star_R"], "rho_R": cfg.rho_R,
                  "p_L": cfg.p_L, "p_star": wp["p_star"], "p_R": cfg.p_R,
                  "u_star": wp["u_star"],
                  "x_contact": x_con, "x_shock": x_sho,
                  "x_rarefaction_head": x_head, "x_rarefaction_tail": x_tail},
        "got": {"rho_L": _avg(x, rho, wl_lo, wl_hi),
                "rho_star_L": _avg(x, rho, sl_lo, sl_hi),
                "rho_star_R": _avg(x, rho, sr_lo, sr_hi),
                "rho_R": _avg(x, rho, wr_lo, wr_hi),
                "p_L": _avg(x, p, wl_lo, wl_hi),
                "p_star": _avg(x, p, sl_lo, sr_hi),
                "p_star_L": _avg(x, p, sl_lo, sl_hi),
                "p_star_R": _avg(x, p, sr_lo, sr_hi),
                "p_R": _avg(x, p, wr_lo, wr_hi),
                "u_star": _avg(x, u, sl_lo, sr_hi),
                "x_contact": x_con_meas,
                "x_shock": x_sho_meas},
    }
    got, ex = metrics["got"], metrics["exact"]
    metrics["rel_err"] = {
        "rho_L": rel_err(got["rho_L"], ex["rho_L"]),
        "rho_star_L": rel_err(got["rho_star_L"], ex["rho_star_L"]),
        "rho_star_R": rel_err(got["rho_star_R"], ex["rho_star_R"]),
        "rho_R": rel_err(got["rho_R"], ex["rho_R"]),
        "p_L": rel_err(got["p_L"], ex["p_L"]),
        "p_star_L": rel_err(got["p_star_L"], ex["p_star"]),
        "p_star_R": rel_err(got["p_star_R"], ex["p_star"]),
        "p_R": rel_err(got["p_R"], ex["p_R"]),
        "u_star": rel_err(got["u_star"], ex["u_star"]),
        "x_contact": rel_err(got["x_contact"], ex["x_contact"]),
        "x_shock": rel_err(got["x_shock"], ex["x_shock"]),
    }

    x_ex = np.linspace(cfg.x_left, cfg.x_right, 4001)
    rho_ref, u_ref, p_ref = exact_riemann_profile(
        x_ex, t, cfg.gamma, cfg.rho_L, cfg.p_L, cfg.u_L,
        cfg.rho_R, cfg.p_R, cfg.u_R, cfg.x_diaphragm)
    rho_i = np.interp(x_ex, x, rho)
    p_i = np.interp(x_ex, x, p)
    u_i = np.interp(x_ex, x, u)
    metrics["l1"] = {
        "rho": float(np.mean(np.abs(rho_i - rho_ref)) / np.mean(rho_ref)),
        "p": float(np.mean(np.abs(p_i - p_ref)) / np.mean(p_ref)),
        "u": float(np.mean(np.abs(u_i - u_ref)) / (np.max(np.abs(u_ref)) + 1e-30)),
    }
    metrics["l2"] = {
        "rho": float(np.sqrt(np.mean((rho_i - rho_ref) ** 2))
                     / np.sqrt(np.mean(rho_ref ** 2))),
        "p": float(np.sqrt(np.mean((p_i - p_ref) ** 2))
                   / np.sqrt(np.mean(p_ref ** 2))),
    }
    # pressure jump across the contact (should be continuous: exact |dp| = 0)
    p_sl, p_sr = got["p_star_L"], got["p_star_R"]
    metrics["contact_pressure_jump"] = float(abs(p_sl - p_sr) / ex["p_star"])

    gated = ["rho_star_L", "rho_star_R", "p_star_L", "p_star_R",
             "x_contact", "x_shock", "rho_L", "p_L", "rho_R", "p_R"]
    metrics["gated"] = gated
    metrics["max_abs_rel_err"] = float(max(abs(metrics["rel_err"][k]) for k in gated))
    metrics["pass_5pct"] = bool(metrics["max_abs_rel_err"] < 0.05)
    metrics["pass_3pct"] = bool(metrics["max_abs_rel_err"] < 0.03)
    return metrics