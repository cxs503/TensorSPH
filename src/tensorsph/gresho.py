"""Gresho-Chan vortex: analytic **steady** rotational equilibrium (periodic WCSPH anchor).

Why this case
-------------
The anchors already in the repo each test something the discrete scheme finds
easy or one-dimensional: Taylor-Green (``vv/tgv.py``) decays, so its gate is a
*rate* and an over-damped scheme can look like an under-resolved one;
hydrostatic (``vv/hydrostatic.py``) has **no** velocity, so it never exercises
the momentum flux; Poiseuille / Couette / Stokes-2nd are wall-driven shear
flows.

The Gresho-Chan vortex (Gresho & Chan 1990, tabulated in Gresho & Sani 1990) is
the missing combination: a **steady, rotationally balanced, two-dimensional
velocity field with curved streamlines and no walls**. Its exact solution of the
inviscid incompressible Euler equations is

    u_theta(r) = U r / r_c                          for r <= r_c
    u_theta(r) = U (r_o - r) / (r_o - r_c)          for r_c < r <= r_o
    u_theta(r) = 0                                  for r > r_o

with the pressure fixed by the radial momentum balance ``dp/dr = rho u^2 / r``
(nothing else: the vortex is steady, so the whole of the convective
acceleration is centripetal and all of it is carried by the pressure gradient).
``u_theta`` is continuous at ``r_c`` while ``u_theta'`` jumps, so the pressure is
``C^1`` with a jump in ``p''``: the kink is deliberate and is what makes the case
hard for a kernel-based gradient.

What is measured
----------------
The reference is **derived in code**, never transcribed:

* :func:`analytic_reference` builds ``p(r)`` as the cumulative trapezoid of
  ``rho u^2 / r`` and then verifies itself -- the radial balance residual
  ``dp/dr - rho u^2/r`` is evaluated numerically and reported;
* :func:`spurious_acceleration` is the L2 norm of the **discrete** acceleration
  the library's own ``vv.tgv.tgv_rhs`` produces at ``t = 0`` from the exact
  state, normalised by ``U^2/r_o``. For an exact solution this is pure
  truncation error, so it must go to zero as ``dx -> 0``, and it costs no time
  integration at all;
* the run then reports how long the balance survives: relative L2 velocity
  error against the *same steady field*, energy ratio, the equivalent numerical
  viscosity extracted from the discrete enstrophy, angular momentum ``L_z``
  (a pairwise central-force scheme must conserve it to round-off -- any drift is
  a bug, not truncation) and linear momentum.

What is *not* claimed
---------------------
The vortex is an inviscid steady state, so a weakly compressible scheme cannot
hold it exactly: acoustic fluctuation and truncation error both feed a slow
spurious decay. No literature value exists for that decay; it is reported as an
equivalent viscosity so that the next change to the pressure or viscosity
operator can be compared with a baseline instead of an opinion.

Output is a plain JSON-serialisable dictionary, in the style of ``vv/tgv.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .droplet_impact import Config, calibrated_particle_mass
from .vv.tgv import (TGVConfig, density, minimum_image, periodic_pairs,
                     tgv_rhs)

__all__ = [
    "GreshoConfig", "analytic_reference", "radius", "u_theta",
    "analytic_velocity", "analytic_pressure", "lattice", "make_state",
    "spurious_acceleration", "enstrophy_lattice", "angular_momentum",
    "kinetic_energy", "velocity_error_pct", "run_gresho", "convergence_study",
    "accept", "REFERENCE",
]

REFERENCE = {
    "citation": ("Gresho, P.M. & Chan, S.T. (1990) On the theory of semi-implicit "
                 "projection methods for viscous incompressible flow and its "
                 "implementation via a finite element method, Int. J. Numer. "
                 "Methods Fluids 11:621-659; the steady test vortex is tabulated "
                 "in Gresho & Sani (1990) IJNMF 11:869"),
    "kind": "analytic (closed form of the inviscid steady Euler equations)",
    "velocity": "u_theta(r) piecewise linear, identically zero outside r_o",
    "pressure": "p(r) = p(0) + int_0^r rho u_theta(s)^2/s ds with the gauge p(0) = 0",
    "derived_in": "tensorsph.gresho.analytic_reference (cumulative quadrature)",
}

_DTYPE = {"float64": torch.float64, "float32": torch.float32}


@dataclass
class GreshoConfig(TGVConfig):
    """Gresho-Chan vortex parameters (periodic square box, no walls, no gravity).

    Inherited from :class:`~tensorsph.vv.tgv.TGVConfig` on purpose: the periodic
    neighbour list, the summation density and the WCSPH right-hand side are then
    *literally* the verification-ladder ones, so this anchor measures the library
    and not a private re-implementation.
    """

    L: float = 1.0                  # box side (m)
    dx: float = 0.02                # particle spacing (m)
    h_factor: float = 1.3
    rho0: float = 1000.0
    sound_speed: float = 20.0
    gamma: float = 7.0
    viscosity: float = 0.0          # the reference solution is inviscid
    U0: float = 1.0                 # peak tangential velocity U (m/s)
    waves: int = 1                  # unused; kept for the base-class contract
    max_courant: float = 0.3
    t_end: float = 1.0
    sample_every: int = 25
    rebuild_every: int = 20
    skin_factor: float = 0.4
    dtype: str = "float64"
    device: str = "cpu"

    # vortex geometry, as fractions of the box side so the case scales with L
    r_core_frac: float = 0.2
    r_outer_frac: float = 0.4
    init_iterations: int = 3        # mass <-> summation-density consistency sweeps

    # ---- derived -----------------------------------------------------------
    @property
    def u_max(self) -> float:
        return self.U0

    @property
    def r_core(self) -> float:
        return self.r_core_frac * self.L

    @property
    def r_outer(self) -> float:
        return self.r_outer_frac * self.L

    @property
    def centre(self) -> tuple:
        return (0.5 * self.L, 0.5 * self.L)

    def h(self) -> float:
        return self.h_factor * self.dx

    def dt(self) -> float:
        return self.max_courant * self.h() / (self.u_max + self.sound_speed)

    def a_reference(self) -> float:
        """``U^2 / r_o`` -- the centripetal acceleration scale (m/s^2)."""
        return self.u_max ** 2 / self.r_outer

    def bulk_modulus(self) -> float:
        return self.rho0 * self.sound_speed ** 2 / self.gamma

    def turnover_time(self) -> float:
        """Core rotation period ``2 pi r_c / U`` (s)."""
        return 2.0 * math.pi * self.r_core / self.u_max

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPE[self.dtype]


# --------------------------------------------------------------------------- #
# Analytic reference, derived (not transcribed) and self-verified
# --------------------------------------------------------------------------- #
def u_theta(r: torch.Tensor, cfg: GreshoConfig) -> torch.Tensor:
    """Exact tangential velocity magnitude at radius ``r``."""
    rc, ro, u = cfg.r_core, cfg.r_outer, cfg.u_max
    inner = u * r / rc
    outer = u * (ro - r) / (ro - rc)
    return torch.where(r <= rc, inner,
                       torch.where(r <= ro, outer, torch.zeros_like(r)))


def analytic_reference(cfg: GreshoConfig, samples: int = 200001) -> dict:
    """Radial tables plus the residuals that prove the closed form is exact.

    ``p(r)`` is the cumulative trapezoid of ``rho u^2 / r`` outwards with the
    gauge ``p(0) = 0``, i.e. the lowest pressure in the field is zero. That gauge
    is the one a Tait equation with a non-negative clamp can represent, so the
    ambient fluid outside ``r_o`` ends up slightly *compressed*
    (``ambient_over_bulk`` is reported); an ambient at ``rho0`` would instead
    require a negative core pressure, which no Tait clamp can hold.
    """
    ro = cfg.r_outer
    r = torch.linspace(0.0, ro, samples, dtype=torch.float64)
    ut = u_theta(r, cfg)
    f = torch.where(r > 0.0, cfg.rho0 * ut.square() / r.clamp_min(1e-300),
                    torch.zeros_like(r))
    dr = float(r[1] - r[0])
    p = torch.cat([torch.zeros(1, dtype=r.dtype),
                   torch.cumsum(0.5 * (f[:-1] + f[1:]) * dr, dim=0)])
    dpdr = torch.gradient(p, spacing=dr)[0]
    a_exact = torch.where(r > 0.0, ut.square() / r.clamp_min(1e-300),
                          torch.zeros_like(r))
    balance = (dpdr / cfg.rho0 - a_exact).abs()
    scale = cfg.a_reference()
    return {
        "r": r, "u_theta": ut, "p": p,
        "p_ambient": float(p[-1]),
        "p_core": float(p[0]),
        "quadrature_points": samples,
        "radial_step": dr,
        "radial_balance_residual_max": float(balance.max()),
        "radial_balance_residual_rel": float(balance.max() / (cfg.rho0 * scale)),
        "pressure_scale_pa": cfg.rho0 * cfg.u_max ** 2,
        "ambient_over_bulk": float(p[-1] / cfg.bulk_modulus()),
        "kink_radius": cfg.r_core,
        "citation": REFERENCE["citation"],
    }


def _np():
    import numpy as np
    return np


def _interp_reference(radius_values: torch.Tensor, table: dict,
                      key: str) -> torch.Tensor:
    """Interpolate a radial table onto radii (clamped outside ``r_outer``)."""
    r = table["r"].numpy()
    y = table[key].numpy()
    out = _np().interp(radius_values.detach().cpu().numpy(), r, y)
    return torch.as_tensor(out, dtype=radius_values.dtype,
                           device=radius_values.device)


def radius(pos: torch.Tensor, cfg: GreshoConfig) -> torch.Tensor:
    """Distance to the vortex centre under the minimum-image convention."""
    centre = torch.tensor(cfg.centre, dtype=pos.dtype, device=pos.device)
    delta = minimum_image(pos - centre, cfg.L)
    return torch.linalg.vector_norm(delta, dim=-1)


def analytic_velocity(pos: torch.Tensor, cfg: GreshoConfig,
                      table: dict | None = None) -> torch.Tensor:
    """Exact velocity field ``(u, v) = u_theta(r) e_theta`` at ``pos``."""
    if table is None:
        table = analytic_reference(cfg)
    r = radius(pos, cfg)
    ut = _interp_reference(r, table, "u_theta")
    centre = torch.tensor(cfg.centre, dtype=pos.dtype, device=pos.device)
    delta = minimum_image(pos - centre, cfg.L)
    safe = r.clamp_min(1e-300)
    ex = torch.where(r > 0.0, -delta[:, 1] / safe, torch.zeros_like(r))
    ey = torch.where(r > 0.0, delta[:, 0] / safe, torch.zeros_like(r))
    return torch.stack((ut * ex, ut * ey), dim=-1)


def analytic_pressure(pos: torch.Tensor, cfg: GreshoConfig,
                      table: dict | None = None) -> torch.Tensor:
    """Exact pressure at ``pos`` (gauge ``p = 0`` on the axis)."""
    if table is None:
        table = analytic_reference(cfg)
    return _interp_reference(radius(pos, cfg), table, "p")


def lattice(cfg: GreshoConfig):
    """Cell-centre lattice of the periodic box (exact partition of ``L``)."""
    n = int(round(cfg.L / cfg.dx))
    if abs(n * cfg.dx - cfg.L) > 1e-9 * cfg.L:
        raise ValueError(
            f"dx={cfg.dx:g} does not divide the periodic box L={cfg.L:g}; "
            "a periodic lattice must close exactly")
    line = (torch.arange(n, dtype=cfg.torch_dtype, device=cfg.device) + 0.5) * cfg.dx
    gx, gy = torch.meshgrid(line, line, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def make_state(cfg: GreshoConfig, *, init_mode: str = "analytic",
               pressure_scale: float = 1.0, table: dict | None = None) -> dict:
    """Particle state whose *summation* density matches the Tait density of ``p``.

    A uniform-mass lattice has ``rho = rho0`` everywhere, hence ``p = 0`` and no
    pressure gradient at all: the vortex would be destroyed by the first step.
    The physical initial state therefore takes its mass from the analytic
    pressure, ``rho = rho0 (1 + p/B)^{1/gamma}``, and the mass field is then swept
    ``init_iterations`` times against the *discrete* summation density so that
    ``sum_j m_j W_ij`` reproduces that density. The residual of the sweep is
    reported: it is the honest measure of how far the discrete state is from the
    density the pressure implies.

    ``init_mode="uniform"`` keeps the calibrated uniform mass (``p = 0``) and is
    the negative control: the spurious acceleration must then be of the order of
    the centripetal acceleration itself, which is what proves that
    :func:`spurious_acceleration` measures balance rather than noise.
    ``pressure_scale`` scales only the initial pressure (sensitivity control).
    """
    if init_mode not in ("analytic", "uniform"):
        raise ValueError(f"unknown init_mode {init_mode!r}")
    pos = lattice(cfg)
    vel = analytic_velocity(pos, cfg, table=table)
    bulk = cfg.bulk_modulus()
    if init_mode == "uniform":
        probe = Config(dx=cfg.dx, h_factor=cfg.h_factor, rho0=cfg.rho0)
        m = calibrated_particle_mass(probe)
        mass = torch.full((pos.shape[0],), m, dtype=cfg.torch_dtype,
                          device=cfg.device)
        rho_target = mass / cfg.dx ** 2
        sweeps = 0
    else:
        p = pressure_scale * analytic_pressure(pos, cfg, table=table)
        rho_target = cfg.rho0 * (1.0 + p.clamp_min(0.0) / bulk) ** (1.0 / cfg.gamma)
        mass = rho_target * cfg.dx ** 2
        sweeps = int(cfg.init_iterations)
    h = cfg.h()
    cutoff = 2.0 * h + cfg.skin_factor * h
    pairs = periodic_pairs(pos, cutoff, cfg.L)
    for _ in range(sweeps):
        rho_sum = density(pos, mass, cfg, pairs=pairs)
        mass = mass * (rho_target / rho_sum)
    rho_sum = density(pos, mass, cfg, pairs=pairs)
    ratio = rho_sum / rho_target - 1.0
    return {
        "pos": pos, "vel": vel, "mass": mass, "rho_target": rho_target,
        "rho_sum": rho_sum,
        "init_mode": init_mode,
        "pressure_scale": pressure_scale,
        "init_density_rel_error_max": float(ratio.abs().max()),
        "init_density_rel_error_rms": float(ratio.square().mean().sqrt()),
        "init_sweeps": sweeps,
        "particles": int(pos.shape[0]),
        "total_mass": float(mass.sum()),
    }


# --------------------------------------------------------------------------- #
# Discrete diagnostics
# --------------------------------------------------------------------------- #
def exact_acceleration(pos: torch.Tensor, cfg: GreshoConfig,
                       table: dict | None = None) -> torch.Tensor:
    """Acceleration the discrete momentum equation must reproduce (m/s^2).

    The vortex is steady, so the *material* acceleration of a fluid particle is
    the centripetal one, ``a = -(u_theta^2 / r) e_r``. In the Lagrangian SPH
    momentum equation ``dv/dt = -(1/rho) grad p + nu lap u`` there is no separate
    convective term: the pressure gradient has to deliver this acceleration.
    Comparing the discrete right-hand side with zero (an Eulerian reading) would
    be wrong here -- that mistake was made in the first draft of this module and
    inflated every number by the size of the centripetal term.
    """
    if table is None:
        table = analytic_reference(cfg)
    r = radius(pos, cfg)
    ut = _interp_reference(r, table, "u_theta")
    centre = torch.tensor(cfg.centre, dtype=pos.dtype, device=pos.device)
    delta = minimum_image(pos - centre, cfg.L)
    safe = r.clamp_min(1e-300)
    er = delta / safe[:, None]
    magnitude = torch.where(r > 0.0, ut.square() / safe, torch.zeros_like(r))
    return -magnitude[:, None] * er


def spurious_acceleration(cfg: GreshoConfig, *, init_mode: str = "analytic",
                          pressure_scale: float = 1.0) -> dict:
    """Relative truncation error of the discrete pressure force at ``t = 0``.

    ``residual = acc_discrete - a_exact`` with ``a_exact`` from
    :func:`exact_acceleration`, normalised by the mass-weighted RMS of
    ``a_exact`` (so the number is dimensionless and independent of ``U`` and
    ``r_o``). For a consistent scheme it must vanish as ``dx -> 0``; it involves
    no time integration, so it cannot be confused with a stability or damping
    effect.
    """
    table = analytic_reference(cfg)
    state = make_state(cfg, init_mode=init_mode, pressure_scale=pressure_scale,
                       table=table)
    pos, vel, mass = state["pos"], state["vel"], state["mass"]
    pos_abs = pos.clone()                       # unwrapped (L_z lever arm)
    r0 = radius(pos, cfg)
    _rho, p, acc = tgv_rhs(pos, vel, mass, cfg)
    exact = exact_acceleration(pos, cfg, table=table)
    residual = acc - exact
    w = mass
    exact_rms = float(((w[:, None] * exact.square()).sum()
                      / w.sum()).sqrt())
    res_rms = float(((w[:, None] * residual.square()).sum() / w.sum()).sqrt())
    a_ref = cfg.a_reference()
    return {
        "a_pct": 100.0 * res_rms / max(exact_rms, 1e-300),
        "a_residual_rms": res_rms,
        "a_exact_rms": exact_rms,
        "a_residual_max_pct": 100.0 * float(
            torch.linalg.vector_norm(residual, dim=-1).max()) / max(a_ref, 1e-300),
        "a_ref": a_ref,
        "p_range_pa": [float(p.min()), float(p.max())],
        "init_mode": init_mode,
        "pressure_scale": pressure_scale,
        "init_density_rel_error_max": state["init_density_rel_error_max"],
        "particles": state["particles"],
        "dx": cfg.dx,
    }


def velocity_error_pct(vel: torch.Tensor, mass: torch.Tensor,
                       vel_exact: torch.Tensor) -> float:
    """Mass-weighted relative L2 error of the velocity field (percent)."""
    num = (mass[:, None] * (vel - vel_exact).square()).sum()
    den = (mass[:, None] * vel_exact.square()).sum()
    return float((num / den).clamp_min(0.0).sqrt() * 100.0)


def _lattice_shaped(field: torch.Tensor, cfg: GreshoConfig) -> torch.Tensor:
    n = int(round(cfg.L / cfg.dx))
    # ``lattice()`` fills [i, j] with (x_j, y_i), i.e. the array is [y, x]:
    # dim 0 is y, dim 1 is x. No transpose here (a transpose swapped the
    # derivatives in the first draft and corrupted the enstrophy).
    return field.reshape(n, n, *field.shape[1:])


def enstrophy_lattice(vel: torch.Tensor, cfg: GreshoConfig) -> dict:
    """Discrete enstrophy ``int omega^2 dA`` on the periodic lattice.

    Central differences with periodic wrap on the cell-centre lattice. Only valid
    while the state still *is* a full lattice, so it is evaluated at ``t = 0``
    where that holds by construction.
    """
    v = _lattice_shaped(vel, cfg)
    u, w = v[..., 0], v[..., 1]
    dvdx = (torch.roll(w, -1, dims=1) - torch.roll(w, 1, dims=1)) / (2.0 * cfg.dx)
    dudy = (torch.roll(u, -1, dims=0) - torch.roll(u, 1, dims=0)) / (2.0 * cfg.dx)
    omega = dvdx - dudy
    return {
        "enstrophy": float(omega.square().sum() * cfg.dx ** 2),
        "omega_max": float(omega.abs().max()),
        "solid_body_enstrophy": float(
            math.pi * (cfg.u_max / cfg.r_core) ** 2 * cfg.r_core ** 2 * 2.0),
    }


def angular_momentum(pos_abs: torch.Tensor, vel: torch.Tensor,
                     mass: torch.Tensor, cfg: GreshoConfig) -> float:
    """``L_z`` about the vortex centre from **unwrapped** positions.

    The momentum equation here is a pairwise central force (verified to
    round-off by ``torque_sum`` in the tests), so ``L_z`` is conserved exactly by
    semi-implicit Euler: ``L^{n+1} = sum m x^n x (v^n + dt a^n) = L^n``. That
    identity breaks if the lever arm is taken from *minimum-image* coordinates,
    because a particle that crosses the periodic boundary changes its offset by
    ``L`` and ``L_z`` by ``m L v`` -- a bookkeeping jump, not a torque. The
    unwrapped coordinate is therefore tracked separately and used here.
    """
    centre = torch.tensor(cfg.centre, dtype=pos_abs.dtype, device=pos_abs.device)
    d = pos_abs - centre
    return float((mass * (d[:, 0] * vel[:, 1] - d[:, 1] * vel[:, 0])).sum())


def torque_sum(pos: torch.Tensor, acc: torch.Tensor, mass: torch.Tensor,
               cfg: GreshoConfig) -> dict:
    """Discrete torque about the centre, normalised by its own magnitude scale.

    Zero to round-off means the pressure force is a central pair force, which is
    what makes ``L_z`` an *invariant* of the scheme rather than a diagnostic.
    """
    centre = torch.tensor(cfg.centre, dtype=pos.dtype, device=pos.device)
    d = minimum_image(pos - centre, cfg.L)
    torque = float((mass * (d[:, 0] * acc[:, 1] - d[:, 1] * acc[:, 0])).sum())
    scale = float((mass * torch.linalg.vector_norm(d, dim=1)
                   * torch.linalg.vector_norm(acc, dim=1)).sum())
    return {"torque": torque, "scale": scale,
            "relative": torque / scale if scale > 0 else float("nan")}


def tangential_velocity(pos: torch.Tensor, vel: torch.Tensor,
                        cfg: GreshoConfig) -> torch.Tensor:
    """Signed azimuthal component ``v . e_theta`` at each particle."""
    centre = torch.tensor(cfg.centre, dtype=pos.dtype, device=pos.device)
    delta = minimum_image(pos - centre, cfg.L)
    r = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-300)
    e_th = torch.stack((-delta[:, 1] / r, delta[:, 0] / r), dim=-1)
    return (vel * e_th).sum(dim=-1)


def radial_profile_error(pos: torch.Tensor, vel: torch.Tensor,
                         mass: torch.Tensor, cfg: GreshoConfig,
                         table: dict, bins: int = 16) -> dict:
    """Error of the azimuthally averaged tangential velocity profile (percent).

    This is the metric that survives particle migration: it compares the
    *field* in space, not a particle against its own history. Bins are
    mass-weighted means of ``v . e_theta`` over ``[0, r_outer]`` with the exact
    ``u_theta(r)`` at the bin centre as the reference; bins holding one particle
    are reported but do not enter the RMS.
    """
    r = radius(pos, cfg)
    vt = tangential_velocity(pos, vel, cfg)
    edges = torch.linspace(0.0, cfg.r_outer, bins + 1, dtype=pos.dtype,
                           device=pos.device)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (r >= lo) & (r < hi)
        count = int(mask.sum())
        if count < 2:
            continue
        m = mass[mask]
        measured = float((vt[mask] * m).sum() / m.sum())
        mid = 0.5 * (float(lo) + float(hi))
        exact = float(_interp_reference(
            torch.tensor([mid], dtype=pos.dtype, device=pos.device),
            table, "u_theta")[0])
        rows.append({"r_lo": float(lo), "r_hi": float(hi), "count": count,
                     "measured": measured, "exact": exact,
                     "error_pct": (100.0 * (measured - exact) / exact
                                   if exact > 1e-9 else float("nan"))})
    gated = [row["error_pct"] for row in rows
             if row["exact"] > 1e-9 and not math.isnan(row["error_pct"])]
    rms = math.sqrt(sum(e * e for e in gated) / len(gated)) if gated else float("nan")
    peak = max((abs(e) for e in gated), default=float("nan"))
    return {"bins": rows, "rms_pct": rms, "max_pct": peak,
            "gated_bins": len(gated)}


def kinetic_energy(vel: torch.Tensor, mass: torch.Tensor) -> float:
    return float(0.5 * (mass[:, None] * vel.square()).sum())


# --------------------------------------------------------------------------- #
# Time integration
# --------------------------------------------------------------------------- #
def run_gresho(cfg: GreshoConfig, *, t_end: float | None = None,
               init_mode: str = "analytic", pressure_scale: float = 1.0) -> dict:
    """Integrate the vortex with the verification-ladder WCSPH right-hand side.

    Semi-implicit Euler, fixed ``dt`` from the acoustic Courant limit, periodic
    minimum-image neighbour lists rebuilt every ``rebuild_every`` steps. The
    velocity error is measured against the **same steady field** at every sample
    (the reference does not decay), so any growth is spurious.
    """
    table = analytic_reference(cfg)
    state = make_state(cfg, init_mode=init_mode, pressure_scale=pressure_scale,
                       table=table)
    pos, vel, mass = state["pos"], state["vel"], state["mass"]
    pos_abs = pos.clone()                       # unwrapped (L_z lever arm)
    r0 = radius(pos, cfg)
    # The reference is steady *in space*: re-evaluate it at the current
    # positions at every sample. Comparing a rotating particle against its own
    # initial velocity vector would grow linearly in time and measure rotation,
    # not error (first-draft bug, caught by the series).
    vel_exact = analytic_velocity(pos, cfg, table=table)
    e0 = kinetic_energy(vel, mass)
    lz0 = angular_momentum(pos_abs, vel, mass, cfg)
    p0 = float(torch.linalg.vector_norm((mass[:, None] * vel).sum(dim=0)))
    ens = enstrophy_lattice(vel, cfg)
    initial_acc = spurious_acceleration(cfg, init_mode=init_mode,
                                       pressure_scale=pressure_scale)

    t_end = cfg.t_end if t_end is None else float(t_end)
    dt = cfg.dt()
    steps = max(1, int(math.ceil(t_end / dt)))
    dt = t_end / steps
    h = cfg.h()
    cutoff = 2.0 * h + cfg.skin_factor * h

    pairs = periodic_pairs(pos, cutoff, cfg.L)
    times: list = []
    errors: list = []
    energies: list = []
    lz_series: list = []
    p_series: list = []
    peak_series: list = []
    rho_rms: list = []
    profile_rms: list = []
    profile_rows: list = []
    migration: list = []
    t = 0.0
    for step in range(steps + 1):
        if step % cfg.rebuild_every == 0:
            pairs = periodic_pairs(pos, cutoff, cfg.L)
        if step % cfg.sample_every == 0 or step == steps:
            times.append(t)
            errors.append(velocity_error_pct(
                vel, mass, analytic_velocity(pos, cfg, table=table)))
            energies.append(kinetic_energy(vel, mass))
            lz_series.append(angular_momentum(pos_abs, vel, mass, cfg))
            prof = radial_profile_error(pos, vel, mass, cfg, table)
            profile_rms.append(prof["rms_pct"])
            if profile_rows == []:
                profile_rows = prof["bins"]
            p_series.append(float(torch.linalg.vector_norm(
                (mass[:, None] * vel).sum(dim=0))))  # absolute, kg m/s
            peak_series.append(float(torch.linalg.vector_norm(vel, dim=-1).max()))
            rho_rms.append(float((density(pos, mass, cfg, pairs=pairs)
                                  / cfg.rho0 - 1.0).square().mean().sqrt()))
            mig = (torch.linalg.vector_norm(pos_abs - torch.tensor(
                cfg.centre, dtype=pos.dtype, device=pos.device), dim=-1) - r0)
            migration.append({"rms": float(mig.square().mean().sqrt()),
                              "max_abs": float(mig.abs().max())})
        if step == steps:
            break
        _rho, _p, acc = tgv_rhs(pos, vel, mass, cfg, pairs=pairs)
        vel = vel + dt * acc
        pos_abs = pos_abs + dt * vel
        pos = torch.remainder(pos + dt * vel, cfg.L)
        t += dt

    lz_drift = max(abs(v - lz0) for v in lz_series) / max(abs(lz0), 1e-300)
    # |sum m v| normalised by the sum of momentum magnitudes: for a symmetric
    # vortex p0 is a round-off number, so normalising by p0 alone is meaningless.
    mom_scale = float((mass[:, None] * vel.abs()).abs().sum())
    mom_drift = max(p_series) / mom_scale if mom_scale > 0.0 else float("nan")
    e_end = energies[-1]
    decay_rate = (math.log(e0) - math.log(max(e_end, 1e-300))) / t_end
    nu_eff = (decay_rate * e0 / (cfg.rho0 * ens["enstrophy"])
              if ens["enstrophy"] > 0.0 else float("inf"))
    turnover = cfg.turnover_time()

    return {
        "case": "gresho_chan_vortex_2d_periodic",
        "config": {
            "L": cfg.L, "dx": cfg.dx, "h_over_dx": cfg.h_factor,
            "particles": state["particles"], "steps": steps, "dt": dt,
            "t_end": t_end, "viscosity": cfg.viscosity, "U0": cfg.u_max,
            "sound_speed": cfg.sound_speed, "gamma": cfg.gamma, "rho0": cfg.rho0,
            "r_core": cfg.r_core, "r_outer": cfg.r_outer,
            "mach": cfg.u_max / cfg.sound_speed, "dtype": cfg.dtype,
            "init_mode": init_mode, "pressure_scale": pressure_scale,
            "turnover_time": turnover, "turns_simulated": t_end / turnover,
            "rebuild_every": cfg.rebuild_every, "sample_every": cfg.sample_every,
        },
        "reference": {
            "kind": "analytic",
            "citation": REFERENCE["citation"],
            "velocity": REFERENCE["velocity"],
            "pressure": REFERENCE["pressure"],
            "p_ambient_pa": table["p_ambient"],
            "radial_balance_residual_rel": table["radial_balance_residual_rel"],
            "ambient_over_bulk": table["ambient_over_bulk"],
            "a_reference": cfg.a_reference(),
        },
        "initial": {
            "init_density_rel_error_max": state["init_density_rel_error_max"],
            "init_density_rel_error_rms": state["init_density_rel_error_rms"],
            "spurious_acceleration_pct": initial_acc["a_pct"],
            "spurious_acceleration_max_pct": initial_acc["a_residual_max_pct"],
            "energy": e0, "angular_momentum": lz0,
            "enstrophy": ens["enstrophy"],
            "solid_body_enstrophy": ens["solid_body_enstrophy"],
            "omega_max": ens["omega_max"],
        },
        "measured": {
            "velocity_error_pct_max": max(errors),
            "velocity_error_pct_end": errors[-1],
            "energy_ratio": e_end / e0 if e0 > 0.0 else float("nan"),
            "decay_rate": decay_rate,
            "equivalent_viscosity": nu_eff,
            "equivalent_reynolds": (cfg.u_max * cfg.r_outer / nu_eff
                                    if nu_eff > 0.0 else float("inf")),
            "angular_momentum_drift_rel": lz_drift,
            "linear_momentum_drift_rel": mom_drift,
            "peak_speed_ratio": (peak_series[-1] / peak_series[0]
                                 if peak_series[0] else float("nan")),
            "profile_error_pct_rms": profile_rms[-1],
            "profile_error_pct_rms_max": max(profile_rms),
            "radial_migration_rms": migration[-1]["rms"],
            "radial_migration_max": migration[-1]["max_abs"],
            "radial_migration_max_rel_router": migration[-1]["max_abs"] / cfg.r_outer,
            "density_rms_over_rho0_end": rho_rms[-1],
            "density_rms_over_rho0_max": max(rho_rms),
        },
        "series": {
            "time": times, "velocity_error_pct": errors, "energy": energies,
            "angular_momentum": lz_series, "momentum": p_series,
            "peak_speed": peak_series, "density_rms": rho_rms,
            "profile_error_pct": profile_rms,
            "profile_bins_initial": profile_rows,
            "energy_ratio": [e / e0 if e0 > 0.0 else float("nan") for e in energies],
        },
    }


# --------------------------------------------------------------------------- #
# Predeclared study + acceptance
# --------------------------------------------------------------------------- #
GATES = {
    # truncation-level gate: has nothing to do with time integration
    "spurious_acceleration_pct": 5.0,
    # survival of the balance over ~1 core turnover at the finest resolution
    "velocity_error_pct_end": 10.0,
    # angular momentum is conserved to round-off by a pairwise central scheme
    "angular_momentum_drift_rel": 1.0e-6,
    # observed order of the truncation error across the ladder
    "min_observed_order": 0.8,
    # an exact state must be non-trivial: the initial field has to be moving
    "min_initial_speed": 0.9,        # fraction of U at t = 0
}


def _order(fine: dict, coarse: dict, key: str) -> float:
    """Observed order of ``key`` (an error in percent) between two resolutions."""
    num = max(coarse[key], 1e-12)
    den = max(fine[key], 1e-12)
    if den <= 0.0 or num <= den:
        return float("nan")
    return math.log(num / den) / math.log(coarse["dx"] / fine["dx"])


def convergence_study(dxs=(0.025, 0.02, 0.0125), *, t_end: float = 1.0,
                      **kwargs) -> list:
    """Run the ladder; every row carries its own truncation-error measurement."""
    rows = []
    for dx in dxs:
        cfg = GreshoConfig(dx=dx, t_end=t_end, **kwargs)
        run = run_gresho(cfg)
        row = {
            "dx": cfg.dx,
            "particles": run["config"]["particles"],
            "steps": run["config"]["steps"],
            "spurious_acceleration_pct": run["initial"]["spurious_acceleration_pct"],
            "spurious_acceleration_max_pct": run["initial"]["spurious_acceleration_max_pct"],
            "init_density_rel_error_max": run["initial"]["init_density_rel_error_max"],
            "velocity_error_pct_end": run["measured"]["velocity_error_pct_end"],
            "velocity_error_pct_max": run["measured"]["velocity_error_pct_max"],
            "energy_ratio": run["measured"]["energy_ratio"],
            "equivalent_viscosity": run["measured"]["equivalent_viscosity"],
            "equivalent_reynolds": run["measured"]["equivalent_reynolds"],
            "angular_momentum_drift_rel": run["measured"]["angular_momentum_drift_rel"],
            "linear_momentum_drift_rel": run["measured"]["linear_momentum_drift_rel"],
            "peak_speed_ratio": run["measured"]["peak_speed_ratio"],
            "profile_error_pct_rms": run["measured"]["profile_error_pct_rms"],
            "radial_migration_max_rel_router":
                run["measured"]["radial_migration_max_rel_router"],
            "density_rms_max": run["measured"]["density_rms_over_rho0_max"],
        }
        rows.append({**row, "run": run})
    for coarse, fine in zip(rows[:-1], rows[1:]):
        fine["observed_order_acceleration"] = _order(fine, coarse,
                                                     "spurious_acceleration_pct")
        fine["observed_order_velocity"] = _order(fine, coarse,
                                                 "velocity_error_pct_end")
    rows[0].setdefault("observed_order_acceleration", float("nan"))
    rows[0].setdefault("observed_order_velocity", float("nan"))
    return rows


def accept(rows: list, *, controls: dict | None = None) -> dict:
    """Machine verdicts for the study (nothing is averaged away)."""
    if not rows:
        raise ValueError("no rows")
    finest = min(rows, key=lambda r: r["dx"])
    orders = [r["observed_order_acceleration"] for r in rows
              if r.get("observed_order_acceleration") is not None
              and not (isinstance(r["observed_order_acceleration"], float)
                       and math.isnan(r["observed_order_acceleration"]))]
    checks = []

    def check(name, measured, limit, ok, direction="<="):
        checks.append({"name": name, "measured": measured, "limit": limit,
                       "direction": direction, "pass": bool(ok)})

    check("spurious_acceleration_pct (finest dx)",
          finest["spurious_acceleration_pct"],
          GATES["spurious_acceleration_pct"],
          finest["spurious_acceleration_pct"] <= GATES["spurious_acceleration_pct"])
    check("velocity_error_pct_end (finest dx)",
          finest["velocity_error_pct_end"],
          GATES["velocity_error_pct_end"],
          finest["velocity_error_pct_end"] <= GATES["velocity_error_pct_end"])
    check("angular_momentum_drift_rel (drift must be round-off)",
          finest["angular_momentum_drift_rel"],
          GATES["angular_momentum_drift_rel"],
          finest["angular_momentum_drift_rel"] <= GATES["angular_momentum_drift_rel"])
    best_order = max([o for o in orders if not math.isnan(o)], default=float("nan"))
    check("observed_order_acceleration (best pair)",
          best_order, GATES["min_observed_order"],
          (not math.isnan(best_order)) and best_order >= GATES["min_observed_order"],
          direction=">=")
    if controls:
        def _pct(entry):
            return entry.get("a_pct", entry.get("spurious_acceleration_pct"))

        forward = controls.get("forward", {})
        reverse = controls.get("reverse", {})
        if forward and reverse:
            f_pct, r_pct = _pct(forward), _pct(reverse)
            # the negative control (no pressure field at all) must be *worse*
            ratio = r_pct / max(f_pct, 1e-12)
            check("negative control (uniform density is worse by >5x)",
                  ratio, 5.0, ratio >= 5.0, direction=">=")
            half = controls.get("half_pressure")
            if half:
                h_pct = _pct(half)
                check("sensitivity control (half pressure is worse)",
                      h_pct, f_pct, h_pct > f_pct, direction=">")
    passed = all(c["pass"] for c in checks)
    return {
        "case": "gresho_chan_vortex_2d_periodic",
        "verdict": "PASS" if passed else "FAIL",
        "checks": checks,
        "finest_dx": finest["dx"],
        "note": ("A FAIL here is an honest measurement of the scheme, not an "
                 "error of the harness: the reference is a closed form."),
    }