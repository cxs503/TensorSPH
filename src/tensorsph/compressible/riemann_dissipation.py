"""Mach-limited Riemann dissipation for compressible SPH.

Physical idea (Ji et al. 2019, MNRAS 489:1052; Godunov-SPH after Parshikov et
al. 2000 / Inutsuka 2002): the numerical dissipation that an SPH pair receives
should scale with the *local Mach number of that pair*, not be a fixed
fraction of the acoustic scale everywhere.  Then

- a shock (M_ij = O(1)) gets the full acoustic dissipation it needs, and
- a smooth subsonic field (M_ij << 1) is left essentially undamped,
- a uniformly translating gas (M_ij = 0) gets no dissipation at all,

whereas classic Monaghan artificial viscosity damps a smooth low-Mach field by
nu ~ alpha c h / 8 regardless of the flow.

Two implementation points that this module gets right and round 1 did not (see
docs/xval_sph_mach_limited_riemann_dissipation.md, section 6):

1. The particles are *advected* (Lagrangian SPH).  Integrating the
   continuity/momentum pair on frozen positions makes the run collapse within
   ~16 steps.
2. The Mach number used by the limiter is the pair *relative* normal Mach
   number M_ij = max(-(v_i - v_j).e_ij, 0) / c_bar.  A single-particle |v|/c is
   zero in the Sod initial state, i.e. it switches dissipation off exactly where
   the shock has to be captured.

Dissipation models available in ``RiemannDissipationConfig``:

- ``scheme="monaghan"`` (default): Pi = (-alpha_eff c_bar mu + beta mu^2)/rho_bar
- ``scheme="riemann"``: acoustic interface pressure
  p* = p_bar - rho_bar c_bar u_rel / 2 (Godunov-SPH)

with ``mach_limit_mode`` = "relative" (default) | "absolute" (legacy).

Verification: Sod (1978) shock tube against the exact Riemann solution
(``run_sod_riemann``), and the damping of a linear standing acoustic wave in a
closed box (``run_acoustic_pulse``), which is the diagnostic that separates a
Mach-limited operator from an ordinary artificial viscosity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from .sod import (
    exact_star_state,
    exact_wave_positions,
    _cubic_w_1d,
    _cubic_dw_1d,
    WAVE_WINDOW_FRACTION,
    WINDOW_TRIM_H,
)

__all__ = [
    "RiemannDissipationConfig",
    "run_sod_riemann",
    "hllc_star_pressure",
    "roe_average",
    "evaluate_metrics",
    "run_acoustic_pulse",
]


@dataclass
class RiemannDissipationConfig:
    """Configuration for Mach-limited Riemann dissipation SPH."""
    n_particles: int = 800
    h_factor: float = 2.5
    alpha_riemann: float = 1.0
    mach_limit: bool = True
    mach_limit_mode: str = "relative"   # "relative" (pair: normal rel. Mach) | "absolute" (round-1 legacy)
    mach_floor: float = 0.01
    scheme: str = "monaghan"            # "monaghan" (alpha AV) | "riemann" (acoustic Godunov-SPH pressure)
    alpha_av: float = 1.0
    beta_av: float = 2.0                # Monaghan quadratic term (required for strong shocks)
    eta2: float = 0.01
    alpha_u: float = 0.0
    cfl: float = 0.15
    gamma: float = 1.4
    t_end: float = 0.2
    x_left: float = 0.0
    x_right: float = 1.0
    x_diaphragm: float = 0.5
    rho_L: float = 1.0
    p_L: float = 1.0
    u_L: float = 0.0
    rho_R: float = 0.125
    p_R: float = 0.1
    u_R: float = 0.0
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


def roe_average(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma):
    """Roe-averaged state."""
    c_L = math.sqrt(gamma * p_L / max(rho_L, 1e-12))
    c_R = math.sqrt(gamma * p_R / max(rho_R, 1e-12))
    H_L = 0.5 * u_L**2 + gamma / (gamma - 1) * p_L / max(rho_L, 1e-12)
    H_R = 0.5 * u_R**2 + gamma / (gamma - 1) * p_R / max(rho_R, 1e-12)
    sqrt_rho_L = math.sqrt(max(rho_L, 1e-12))
    sqrt_rho_R = math.sqrt(max(rho_R, 1e-12))
    denom = sqrt_rho_L + sqrt_rho_R
    u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / denom
    H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / denom
    c_roe = math.sqrt(max((gamma - 1) * (H_roe - 0.5 * u_roe**2), 1e-12))
    return math.sqrt(rho_L * rho_R), u_roe, H_roe, c_roe


def hllc_star_pressure(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma):
    """TRRS approximation for star pressure (Toro 1999, eq. 10.24)."""
    c_L = math.sqrt(gamma * p_L / max(rho_L, 1e-12))
    c_R = math.sqrt(gamma * p_R / max(rho_R, 1e-12))
    gm1 = gamma - 1.0
    num = c_L + c_R - 0.5 * gm1 * (u_R - u_L)
    den = c_L / (p_L ** (gm1 / (2.0 * gamma))) + c_R / (p_R ** (gm1 / (2.0 * gamma)))
    if den <= 0 or num <= 0:
        p_avg = 0.5 * (p_L + p_R)
        c_avg = 0.5 * (c_L + c_R)
        rho_avg = 0.5 * (rho_L + rho_R)
        p_star = p_avg - 0.5 * (u_R - u_L) * rho_avg * c_avg
    else:
        p_star = (num / den) ** (2.0 * gamma / gm1)
    return max(p_star, 1e-12)


def _extend_with_ghosts(x, rho, v, e, m, x_L, x_R, support, h):
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
    return np.bincount(idx, weights=val, minlength=n)


def _pairs(X, h, support):
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


def _derivs_riemann(x, rho, v, e, m, cfg):
    """SPH rates with Mach-limited Riemann dissipation.

    Uses the standard SPH formulation with Mach-limited viscosity.
    The viscosity coefficient is scaled by min(M, 1) to reduce
    dissipation in smooth subsonic regions.
    """
    gamma = cfg.gamma
    h = np.full(len(x), cfg.h)
    eta2 = cfg.eta2

    p = np.maximum((gamma - 1.0) * rho * e, 1e-12)
    cs = np.sqrt(gamma * p / np.maximum(rho, 1e-12))
    mach = np.abs(v) / np.maximum(cs, 1e-12)

    X, RHO, V, E, M, n_real, H = _extend_with_ghosts(
        x, rho, v, e, m, cfg.x_left, cfg.x_right, cfg.support, h)
    P = np.maximum((gamma - 1.0) * RHO * E, 1e-12)
    CS = np.sqrt(gamma * P / np.maximum(RHO, 1e-12))
    MACH = np.abs(V) / np.maximum(CS, 1e-12)

    pr = _pairs(X, H, 2.0 * H.max())
    if pr is None:
        zero = np.zeros(n_real)
        return zero, zero.copy(), zero.copy(), p, cs
    i, j, dxij, r, h_ij = pr

    grad = _cubic_dw_1d(dxij, h_ij)
    vij = V[i] - V[j]
    vdotr = vij * dxij
    mu = h_ij * vdotr / (r * r + eta2 * h_ij * h_ij)
    cs_bar = 0.5 * (CS[i] + CS[j])
    rho_bar = 0.5 * (RHO[i] + RHO[j])

    # --- Mach limiter ---------------------------------------------------
    # M_ij is the compressive *relative* normal Mach number of the pair:
    # a uniformly translating gas (any speed) gives M_ij = 0 -> no numerical
    # dissipation, a shock front gives M_ij = O(1) -> dissipation saturates at
    # the full acoustic value.  (Round 1 used the single-particle |v|/c, which
    # is 0 in the Sod initial state and therefore switched viscosity *off*
    # exactly where the shock has to be captured -- one of the two reasons the
    # round-1 run died; see the report, section 4.)
    u_rel = vdotr / np.maximum(r, 1e-30)          # < 0 for an approaching pair
    mach_pair = np.maximum(-u_rel, 0.0) / np.maximum(cs_bar, 1e-30)
    if not cfg.mach_limit:
        mach_factor = np.ones_like(mach_pair)
    elif cfg.mach_limit_mode == "absolute":
        mach_factor = np.minimum(np.minimum(MACH[i], MACH[j]), 1.0)
        mach_factor = np.maximum(mach_factor, cfg.mach_floor)
    else:                                          # "relative"
        mach_factor = np.minimum(mach_pair, 1.0)
        mach_factor = np.maximum(mach_factor, cfg.mach_floor)
    alpha_eff = cfg.alpha_av * mach_factor

    if cfg.scheme == "riemann":
        # Acoustic (Godunov-SPH) interface pressure, p* = pbar - rho_bar*c_bar*u_rel/2.
        # The dissipative part is a pressure: 0.5*rho_bar*c_bar*|u_rel|, i.e.
        # Q-units (-0.5*c_bar*u_rel/rho_bar).  No alpha is involved; the pair
        # Mach number is the only limiter.
        Pi = np.where(vdotr < 0.0,
                      -0.5 * mach_factor * cs_bar * u_rel / np.maximum(rho_bar, 1e-12),
                      0.0)
    else:
        # Monaghan artificial viscosity (linear + quadratic), Mach-limited alpha
        Pi = np.where(vdotr < 0.0,
                      (-alpha_eff * cs_bar * mu + cfg.beta_av * mu * mu)
                      / np.maximum(rho_bar, 1e-12),
                      0.0)

    # Pressure term (arithmetic mean form)
    Q = (P[i] + P[j]) / (np.maximum(RHO[i], 1e-12) * np.maximum(RHO[j], 1e-12)) + Pi

    uij_grad = vij * grad
    n_ext = len(X)
    a_ext = _bc(j, M[i] * Q * grad, n_ext) - _bc(i, M[j] * Q * grad, n_ext)

    half = 0.5 * uij_grad
    de_ext = (_bc(i, M[j] * Q * half, n_ext)
              + _bc(j, M[i] * Q * half, n_ext))

    if cfg.alpha_u > 0.0:
        mu_u = h_ij * dxij * grad / (r * r + eta2 * h_ij * h_ij)
        cond = cfg.alpha_u * cs_bar * (E[i] - E[j]) / np.maximum(rho_bar, 1e-12) * mu_u
        de_ext = de_ext + _bc(i, M[j] * cond, n_ext) + _bc(j, M[i] * cond, n_ext)

    drho = (_bc(i, M[j] * uij_grad, n_ext) + _bc(j, M[i] * uij_grad, n_ext))[:n_real]
    return drho, a_ext[:n_real], de_ext[:n_real], p, cs


def _timestep_riemann(cfg, rho, v, e):
    p = np.maximum((cfg.gamma - 1.0) * rho * e, 1e-12)
    cs = np.sqrt(cfg.gamma * p / np.maximum(rho, 1e-12))
    dt_cfl = cfg.cfl * cfg.h / np.max(cs + np.abs(v))
    nu_max = cfg.alpha_av * np.max(cs) * cfg.h
    dt_visc = 0.25 * cfg.h * cfg.h / (nu_max + 1e-30)
    return float(min(dt_cfl, dt_visc))


def run_sod_riemann(cfg, output_times=None):
    """Run the Sod shock tube with Mach-limited Riemann dissipation."""
    N = cfg.n_particles
    dx = cfg.dx
    h = cfg.h
    gamma = cfg.gamma

    x = cfg.x_left + (np.arange(N) + 0.5) * dx
    smooth_width = 1.5 * dx
    tanh_profile = 0.5 * (1 + np.tanh((cfg.x_diaphragm - x) / smooth_width))
    rho = tanh_profile * cfg.rho_L + (1 - tanh_profile) * cfg.rho_R
    p = tanh_profile * cfg.p_L + (1 - tanh_profile) * cfg.p_R
    u = np.zeros(N)
    e = p / ((gamma - 1.0) * rho)
    m = rho * dx

    t = 0.0
    step = 0

    if output_times is None:
        output_times = [cfg.t_end]
    output_times = sorted(output_times)
    snapshots = []
    output_idx = 0

    total_mass_0 = np.sum(m)
    total_energy_0 = np.sum(m * (e + 0.5 * u ** 2))
    dissipation_stats = []

    dt_min = float("inf")
    mach_pair_max = 0.0
    while t < cfg.t_end * (1.0 - 1e-12) and step < cfg.max_steps:
        dt = min(_timestep_riemann(cfg, rho, u, e), cfg.t_end - t)
        if dt < 1e-14:
            break
        dt_min = min(dt_min, dt)

        # RK2 (midpoint).  The particles are advected with the flow: SPH is a
        # Lagrangian method, so freezing x makes the continuity/momentum pair
        # non-conservative and the run collapses within ~16 steps (round-1
        # honest-FAIL, reproduced in the report).
        drho, acc, de, _, _ = _derivs_riemann(x, rho, u, e, m, cfg)
        rho_half = np.maximum(rho + 0.5 * dt * drho, 1e-10)
        u_half = u + 0.5 * dt * acc
        e_half = np.maximum(e + 0.5 * dt * de, 1e-12)
        x_half = x + 0.5 * dt * u_half

        drho2, acc2, de2, _, _ = _derivs_riemann(x_half, rho_half, u_half, e_half, m, cfg)
        rho = np.maximum(rho + dt * drho2, 1e-10)
        u = u + dt * acc2
        e = np.maximum(e + dt * de2, 1e-12)
        x = x + dt * u_half

        cs_now = np.sqrt(gamma * np.maximum((gamma - 1.0) * rho * e, 1e-12)
                         / np.maximum(rho, 1e-12))
        mach_pair_max = max(mach_pair_max, float(np.max(np.abs(u) / cs_now)))

        t += dt
        step += 1

        if step % 100 == 0:
            mach = np.abs(u) / np.maximum(cs_now, 1e-12)
            dissipation_stats.append({
                "t": t, "step": step,
                "max_mach": float(np.max(mach)),
                "mean_mach": float(np.mean(mach)),
                "min_rho": float(np.min(rho)),
            })

        while output_idx < len(output_times) and t >= output_times[output_idx] - 1e-12:
            t_out = output_times[output_idx]
            snapshots.append({
                "t": t_out, "x": x.copy(), "rho": rho.copy(),
                "u": u.copy(),
                "p": np.maximum((gamma - 1.0) * rho * e, 1e-12),
                "e": e.copy(),
            })
            output_idx += 1

    total_mass_final = np.sum(m)
    total_energy_final = np.sum(m * (e + 0.5 * u ** 2))

    tree = cKDTree(x[:, None])
    neighbours = tree.query_pairs(2.0 * h, output_type="ndarray")
    if neighbours.size > 0:
        counts = np.bincount(neighbours[:, 0], minlength=N) + np.bincount(neighbours[:, 1], minlength=N)
        min_neighbours = int(np.min(counts))
        mean_neighbours = float(np.mean(counts))
    else:
        min_neighbours = 0
        mean_neighbours = 0.0

    return {
        "snapshots": snapshots,
        "t_final": t,
        "steps": step,
        "mass_initial": total_mass_0,
        "mass_final": total_mass_final,
        "energy_initial": total_energy_0,
        "energy_final": total_energy_final,
        "energy_rel_change": float((total_energy_final - total_energy_0) / total_energy_0),
        "neighbours": {"min": min_neighbours, "mean": mean_neighbours},
        "dissipation_stats": dissipation_stats,
        "dt_min": dt_min,
        "mach_max": mach_pair_max,
        "x_span_final": float(np.max(x) - np.min(x)),
        "config": cfg.as_dict(),
    }


def run_acoustic_pulse(cfg, amplitude: float = 0.02, n_periods: float = 6.0,
                       mode: int = 1):
    """Damping of a linear standing acoustic wave in a closed 1-D box.

    This diagnostic separates a *Mach-limited* dissipation from an ordinary
    artificial viscosity: both capture the Sod shock (Sod is dominated by
    compressive motion, M_ij = O(1)), but only the Mach-limited operator leaves
    a smooth, low-Mach acoustic field nearly undamped.

    Initial condition: the linear standing-wave eigenfunction of the closed box
    (reflecting walls at x_left / x_right),

        u = A c0 cos(k pi s),  rho = rho0 (1 + A cos(k pi s)),
        p = p0 (1 + gamma A cos(k pi s)),   s = (x - x_left) / L,

    with c0 = sqrt(gamma p0 / rho0), k = mode*pi/L.  The mode amplitude is
    obtained by projecting u and (rho - rho0) on the eigenfunction, and the
    measured decay rate is converted to an effective viscosity
    nu = -2 * slope / k^2, to be compared with the artificial-viscosity estimate
    nu ~ alpha c h / 8.
    """
    gamma = cfg.gamma
    rho0, p0 = 1.0, 1.0
    c0 = math.sqrt(gamma * p0 / rho0)
    L = cfg.x_right - cfg.x_left
    k = mode * math.pi / L

    N = cfg.n_particles
    dx = cfg.dx
    x = cfg.x_left + (np.arange(N) + 0.5) * dx
    shape = np.cos(k * (x - cfg.x_left))
    # t = 0 slice of the standing-wave eigenmode: pure cos quadrature in u,
    # rho = rho0.  (u and rho in phase at t=0 would be a *superposition* of two
    # quadratures and the modal energy would then oscillate instead of decay.)
    u = amplitude * c0 * shape
    rho = np.full(N, rho0)
    p = np.full(N, p0)
    e = p / ((gamma - 1.0) * rho)
    m = (rho0 * dx) * np.ones(N)          # frozen Lagrangian masses

    period = 2.0 * L / c0
    t_end = n_periods * period
    out_times = np.linspace(0.0, t_end, 401)[1:]
    out_idx = 0
    series = {"t": [], "amp_u": [], "amp_rho": [], "energy": [], "mach_max": [],
              "e_acoustic": [], "e_kin": []}

    t = 0.0
    step = 0
    dt_min = float("inf")
    while t < t_end * (1.0 - 1e-12) and step < cfg.max_steps:
        dt = min(_timestep_riemann(cfg, rho, u, e), t_end - t)
        if dt < 1e-14:
            break
        dt_min = min(dt_min, dt)

        drho, acc, de, _, _ = _derivs_riemann(x, rho, u, e, m, cfg)
        rho_h = np.maximum(rho + 0.5 * dt * drho, 1e-12)
        u_h = u + 0.5 * dt * acc
        e_h = np.maximum(e + 0.5 * dt * de, 1e-14)
        x_h = x + 0.5 * dt * u_h

        drho2, acc2, de2, _, _ = _derivs_riemann(x_h, rho_h, u_h, e_h, m, cfg)
        rho = np.maximum(rho + dt * drho2, 1e-12)
        u = u + dt * acc2
        e = np.maximum(e + dt * de2, 1e-14)
        x = x + dt * u_h
        t += dt
        step += 1

        while out_idx < len(out_times) and t >= out_times[out_idx] - 1e-15:
            amp_u = 2.0 * float(np.mean(u * shape)) / (amplitude * c0)
            amp_rho = 2.0 * float(np.mean((rho - rho0) * shape)) / amplitude
            series["t"].append(float(out_times[out_idx]))
            series["amp_u"].append(amp_u)
            series["amp_rho"].append(amp_rho)
            p_i = (gamma - 1.0) * rho * e
            dp = p_i - p0
            # modal acoustic energy per unit length (kinetic + compressional).
            # For a single undamped eigenmode this is exactly constant in time,
            # so its log-slope gives the damping rate without oscillation bias.
            e_ac = float(np.sum(m * (0.5 * u ** 2 + 0.5 * dp ** 2 / (rho0 * c0 ** 2))) / rho0 / L)
            series["energy"].append(float(np.sum(m * (0.5 * u ** 2 + e))))
            series["e_acoustic"].append(e_ac)
            series["e_kin"].append(float(np.sum(m * 0.5 * u ** 2) / rho0 / L))
            cs = np.sqrt(gamma * np.maximum((gamma - 1.0) * rho * e, 1e-12) / rho)
            series["mach_max"].append(float(np.max(np.abs(u) / cs)))
            out_idx += 1

    S = {key: np.asarray(val) for key, val in series.items()}
    tt, eac = S["t"], S["e_acoustic"]
    keep = eac > 0.0
    if np.count_nonzero(keep) >= 3:
        slope_e = float(np.polyfit(tt[keep], np.log(eac[keep]), 1)[0])
    else:
        slope_e = float("nan")
    # E_ac ~ exp(-nu k^2 t)  =>  nu = -slope_E / k^2
    nu_meas = -slope_e / (k ** 2) if np.isfinite(slope_e) else float("nan")
    slope = 0.5 * slope_e if np.isfinite(slope_e) else float("nan")
    e_ac0 = float(eac[0]) if len(eac) else float("nan")
    e_ac_end = float(eac[-1]) if len(eac) else float("nan")
    nu_theory = cfg.alpha_av * c0 * cfg.h / 8.0

    return {
        "series": S,
        "period": period,
        "t_end": t_end,
        "steps": step,
        "dt_min": dt_min,
        "damping_rate": slope,
        "nu_measured": nu_meas,
        "nu_av_theory": nu_theory,
        "e_acoustic_0": e_ac0,
        "e_acoustic_end": e_ac_end,
        "e_acoustic_ratio": (e_ac_end / e_ac0) if e_ac0 else float("nan"),
        "amp_u_end": float(S["amp_u"][-1]) if len(S["amp_u"]) else float("nan"),
        "amp_rho_end": float(S["amp_rho"][-1]) if len(S["amp_rho"]) else float("nan"),
        "config": cfg.as_dict(),
    }


def plateau_windows(x, cfg):
    star = exact_star_state(gamma=cfg.gamma,
                            rho_L=cfg.rho_L, p_L=cfg.p_L, u_L=cfg.u_L,
                            rho_R=cfg.rho_R, p_R=cfg.p_R, u_R=cfg.u_R)
    waves = exact_wave_positions(cfg.t_end, x0=cfg.x_diaphragm,
                                 gamma=cfg.gamma,
                                 rho_L=cfg.rho_L, p_L=cfg.p_L, u_L=cfg.u_L,
                                 rho_R=cfg.rho_R, p_R=cfg.p_R, u_R=cfg.u_R)
    h = cfg.h

    def _trim(x_lo, x_hi):
        w = x_hi - x_lo
        t = max(WAVE_WINDOW_FRACTION * w, WINDOW_TRIM_H * h)
        return x_lo + t, x_hi - t

    return {
        "rho_L": (cfg.x_left, _trim(cfg.x_left, waves["x_rarefaction_head"])[0]),
        "rho_star_L": _trim(waves["x_rarefaction_tail"], waves["x_contact"]),
        "rho_star_R": _trim(waves["x_contact"], waves["x_shock"]),
        "rho_R": (_trim(waves["x_shock"], cfg.x_right)[1], cfg.x_right),
        "p_star_L": _trim(waves["x_rarefaction_tail"], waves["x_contact"]),
        "p_star_R": _trim(waves["x_contact"], waves["x_shock"]),
        "u_star": _trim(waves["x_rarefaction_tail"], waves["x_shock"]),
    }


def evaluate_metrics(snapshot, cfg):
    from tensorsph.wcsph.sod_shock import exact_riemann_profile

    x = snapshot["x"]
    t = snapshot["t"]
    rho_ex, u_ex, p_ex = exact_riemann_profile(
        x, t, gamma=cfg.gamma,
        rho_L=cfg.rho_L, p_L=cfg.p_L, u_L=cfg.u_L,
        rho_R=cfg.rho_R, p_R=cfg.p_R, u_R=cfg.u_R,
        x0=cfg.x_diaphragm,
    )

    rel_err = {}
    regions = plateau_windows(x, cfg)

    for key, (lo, hi) in regions.items():
        mask = (x >= lo) & (x <= hi)
        if not np.any(mask):
            rel_err[key] = float("nan")
            continue

        if key.startswith("rho"):
            ref_key = key.replace("rho_", "")
            if ref_key == "L":
                ref = cfg.rho_L
            elif ref_key == "R":
                ref = cfg.rho_R
            elif ref_key == "star_L":
                ref = exact_star_state(gamma=cfg.gamma,
                                       rho_L=cfg.rho_L, p_L=cfg.p_L, u_L=cfg.u_L,
                                       rho_R=cfg.rho_R, p_R=cfg.p_R, u_R=cfg.u_R)["rho_star_L"]
            elif ref_key == "star_R":
                ref = exact_star_state(gamma=cfg.gamma,
                                       rho_L=cfg.rho_L, p_L=cfg.p_L, u_L=cfg.u_L,
                                       rho_R=cfg.rho_R, p_R=cfg.p_R, u_R=cfg.u_R)["rho_star_R"]
            else:
                ref = np.mean(rho_ex[mask])
            got = np.mean(snapshot["rho"][mask])
        elif key.startswith("p"):
            ref_key = key.replace("p_", "")
            if ref_key == "star_L":
                ref = exact_star_state(gamma=cfg.gamma,
                                       rho_L=cfg.rho_L, p_L=cfg.p_L, u_L=cfg.u_L,
                                       rho_R=cfg.rho_R, p_R=cfg.p_R, u_R=cfg.u_R)["p_star"]
            elif ref_key == "star_R":
                ref = exact_star_state(gamma=cfg.gamma,
                                       rho_L=cfg.rho_L, p_L=cfg.p_L, u_L=cfg.u_L,
                                       rho_R=cfg.rho_R, p_R=cfg.p_R, u_R=cfg.u_R)["p_star"]
            else:
                ref = np.mean(p_ex[mask])
            got = np.mean(snapshot["p"][mask])
        elif key.startswith("u"):
            ref_key = key.replace("u_", "")
            if ref_key == "star":
                ref = exact_star_state(gamma=cfg.gamma,
                                       rho_L=cfg.rho_L, p_L=cfg.p_L, u_L=cfg.u_L,
                                       rho_R=cfg.rho_R, p_R=cfg.p_R, u_R=cfg.u_R)["u_star"]
            else:
                ref = np.mean(u_ex[mask])
            got = np.mean(snapshot["u"][mask])
        else:
            ref = np.mean(rho_ex[mask])
            got = np.mean(snapshot["rho"][mask])

        denom = abs(ref) if abs(ref) > 1e-10 else 1.0
        rel_err[key] = float((got - ref) / denom)

    l1_rho = float(np.mean(np.abs(snapshot["rho"] - rho_ex)) / np.mean(rho_ex))
    l1_p = float(np.mean(np.abs(snapshot["p"] - p_ex)) / np.mean(p_ex))
    l1_u = float(np.mean(np.abs(snapshot["u"] - u_ex)) / (np.mean(np.abs(u_ex)) + 1e-10))

    valid_errs = [abs(v) for v in rel_err.values() if not np.isnan(v)]
    max_abs_rel_err = max(valid_errs) if valid_errs else float("nan")

    star_L_mask = (x >= regions["p_star_L"][0]) & (x <= regions["p_star_L"][1])
    star_R_mask = (x >= regions["p_star_R"][0]) & (x <= regions["p_star_R"][1])
    if np.any(star_L_mask) and np.any(star_R_mask):
        p_star_L = np.mean(snapshot["p"][star_L_mask])
        p_star_R = np.mean(snapshot["p"][star_R_mask])
        p_star_ref = exact_star_state(gamma=cfg.gamma,
                                      rho_L=cfg.rho_L, p_L=cfg.p_L, u_L=cfg.u_L,
                                      rho_R=cfg.rho_R, p_R=cfg.p_R, u_R=cfg.u_R)["p_star"]
        contact_jump = float(abs(p_star_L - p_star_R) / p_star_ref)
    else:
        contact_jump = float("nan")

    return {
        "rel_err": rel_err,
        "l1": {"rho": l1_rho, "p": l1_p, "u": l1_u},
        "max_abs_rel_err": max_abs_rel_err,
        "contact_pressure_jump": contact_jump,
        "pass_5pct": max_abs_rel_err < 0.05,
        "pass_3pct": max_abs_rel_err < 0.03,
    }
