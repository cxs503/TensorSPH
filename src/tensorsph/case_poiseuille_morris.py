"""CASE-SPH-POISEUILLE-MORRIS -- low-Reynolds-number plane Poiseuille channel.

The case
--------
An incompressible fluid of density ``rho0`` and dynamic viscosity ``mu`` fills a
2-D channel of height ``H`` with no-slip walls at ``y = 0`` and ``y = H``,
periodic in ``x``, driven by a uniform body force per unit mass ``g`` along
``x``.  At low Reynolds number the flow is a steady parabola and its start-up
from rest is the classical Boussinesq series.  This is the channel problem
Morris, Fox & Zhu (1997, *J. Comput. Phys.* **136**:214-226) used to measure
their SPH viscous operator, run here as an anchored verification case: steady
profile, wall shear, flow rate and start-up history against the exact solution.

Reference solution (zero transcription)
---------------------------------------
Every reference number this module writes into a table is *computed at run
time* from the exact solution, never copied from a text::

    u_s(y)  = g (H - y) y / (2 nu)                        steady parabola
    u_max   = g H^2 / (8 nu),        u_mean = 2 u_max / 3
    tau_w   = rho g H / 2                                 wall shear / momentum balance
    Q       = g H^3 / (12 nu)                             flow rate per unit width
    u(y,t)  = u_s(y) - (4 g H^2 / (nu pi^3)) *
                  sum_{n odd} sin(n pi y / H) exp(-nu n^2 pi^2 t / H^2) / n^3

``derive_reference_symbolically`` re-derives the ODE residual, both boundary
conditions, ``u_max``, ``tau_w`` and ``Q`` with sympy and requires them to be
identically satisfied.  ``verify_series`` certifies the start-up series
numerically: ``u(y,0) = 0``, ``u_t = nu u_yy + g`` by finite differences, the
steady limit ``u(y, 10 tau) -> u_s``, and insensitivity to the number of
retained terms.  A transcription error would surface as a non-zero residual in
the artifact (``reference.symbolic`` / ``reference.series``).

The series is the same closed form as
:func:`tensorsph.isph_channel.analytic_startup_profile`; the local copy exists
only because ``isph_channel`` imports scipy, which is not a dependency of this
package (``tests/test_case_poiseuille_morris.py`` asserts the two agree bit for
bit whenever scipy is importable).

Numerics
--------
The production WCSPH path, unmodified: :func:`tensorsph.droplet_impact.rhs`
(Tait EOS, summation density, Morris/Brookshaw viscous Laplacian with the
calibrated factor 2), with the streamwise body force added to the fluid
acceleration.  The no-slip walls are the mirror (ghost) boundary condition of
:mod:`tensorsph.wall_mirror`: near-wall fluid is mirrored across each plane with
the negated velocity and handed to ``rhs`` as a *moving* wall, so the discrete
viscous operator sees ``u = 0`` on the plane instead of at ``-dx/2``.  ``x``
periodicity is exact through two periodic images.  No solver constant is tuned
for this case; only geometry, drive and acceptance criteria are set here.

Acceptance (predeclared; the case-pool tolerance is ``< 5 %``)
-------------------------------------------------------------
===  ==========================================================================
A    centre line  ``|u_max / (g H^2 / 8 nu) - 1|``
B    profile shape  row-RMS of ``(u_num - u_ref)/u_ref`` over ``0.1 <= y/H <= 0.9``
C    wall shear   ``|tau_fit / (rho g H / 2) - 1|`` at both walls (parabola fit)
D    wall slip    parabola intercept at each plane, as a fraction of ``u_max``
E    flow rate    ``|Q_num / (g H^3 / 12 nu) - 1|``, ``Q`` from the row profile
F    start-up     ``max_y |u_num - u_series| / u_max,inf`` at ``t/tau = 0.1,
                 0.2, 0.4``, ``tau = H^2/(pi^2 nu)``; the centre-row relative
                 error is reported with it
G    steady state ``u_max`` drift over the averaging window ``<= 1 %``
H    convergence  B must not grow with ``H/dx``; the observed order is reported
===  ==========================================================================

Usage
-----
``python -m tensorsph.case_poiseuille_morris --out runs/case_poiseuille_morris``
writes ``results.json``, figures and a summary table (``--quick`` is the CI-sized
configuration).  :func:`main` is also what ``python -m tensorsph --case
poiseuille_morris`` calls.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .droplet_impact import Config, __version__, resolve_device, rhs
from .wall_mirror import plane_walls

__all__ = [
    "CASE_KEY",
    "REFERENCE",
    "MorrisConfig",
    "steady_profile",
    "startup_profile",
    "averaged_profile",
    "flow_rate",
    "derive_reference_symbolically",
    "verify_series",
    "build_channel",
    "run_case",
    "evaluate",
    "run_ladder",
    "run_sensitivity",
    "make_figures",
    "main",
]

CASE_KEY = "CASE-SPH-POISEUILLE-MORRIS"

#: Bibliographic identity of the reference case.  The DOI is Crossref-verified
#: (``api.crossref.org``) and written into ``results.json`` with a timestamp, so
#: the citation is checkable rather than asserted.
REFERENCE = {
    "authors": "Morris, J. P.; Fox, P. J.; Zhu, Y.",
    "title": "Modeling Low Reynolds Number Incompressible Flows Using SPH",
    "journal": "Journal of Computational Physics",
    "volume": 136,
    "issue": 1,
    "pages": "214-226",
    "year": 1997,
    "doi": "10.1006/jcph.1997.5776",
    "role": ("the no-slip low-Reynolds plane channel is this paper's viscous-"
             "operator test; the numbers compared against here are that "
             "problem's exact solution, re-derived in code, not copied"),
}


# --------------------------------------------------------------------------- #
# Case definition
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class MorrisConfig:
    """Geometry, fluid properties and run control of the Morris channel case.

    The state is fixed by dimensional groups, so ``body_force`` is *derived*
    from the requested ``reynolds`` (``Re = rho0 ubar H / mu``, ``ubar`` the
    cross-section mean velocity) rather than being a free number: change ``mu``
    or ``H`` and the drive follows.
    """

    height: float = 1.0e-3           # H [m]
    length_over_height: float = 4.0  # periodic streamwise extent L/H
    rho0: float = 1000.0             # [kg/m^3]
    mu: float = 1.0                  # dynamic viscosity [Pa s]
    reynolds: float = 0.1            # Re = rho0 ubar H / mu
    resolution: int = 16             # H/dx
    h_factor: float = 1.3            # kernel support h/dx
    sound_speed: float = 20.0        # Tait c0 [m/s]
    t_end_over_tau: float = 8.0      # run length in diffusive times
    average_fraction: float = 0.25   # last fraction of t_end that is averaged
    sample_fractions: tuple = (0.1, 0.2, 0.4)   # of tau, start-up snapshots
    wall_model: str = "mirror_no_slip"
    slip: str = "no_slip"
    device: str = "cpu"

    # -- derived ---------------------------------------------------------- #
    @property
    def nu(self) -> float:
        """Kinematic viscosity ``mu/rho0`` [m^2/s]."""
        return self.mu / self.rho0

    @property
    def dx(self) -> float:
        return self.height / self.resolution

    @property
    def h(self) -> float:
        return self.h_factor * self.dx

    @property
    def length(self) -> float:
        return self.length_over_height * self.height

    @property
    def u_mean(self) -> float:
        """Cross-section mean velocity fixed by the Reynolds number."""
        return self.reynolds * self.mu / (self.rho0 * self.height)

    @property
    def body_force(self) -> float:
        """Drive that produces ``u_mean`` at steady state: ``12 nu ubar / H^2``."""
        return 12.0 * self.nu * self.u_mean / self.height ** 2

    @property
    def u_max(self) -> float:
        """``g H^2 / (8 nu)``, identical to ``3 ubar / 2``."""
        return self.body_force * self.height ** 2 / (8.0 * self.nu)

    @property
    def tau_wall(self) -> float:
        return self.rho0 * self.body_force * self.height / 2.0

    @property
    def q_ref(self) -> float:
        return self.body_force * self.height ** 3 / (12.0 * self.nu)

    @property
    def tau_diffusive(self) -> float:
        """Slowest diffusive time ``H^2/(pi^2 nu)`` -- the start-up clock."""
        return self.height ** 2 / (self.nu * math.pi ** 2)

    @property
    def t_end(self) -> float:
        return self.t_end_over_tau * self.tau_diffusive

    @property
    def dt(self) -> float:
        """Fixed step: acoustic limit at 30 % Courant, or the viscous limit."""
        return min(0.3 * self.h / self.sound_speed,
                   0.125 * self.h ** 2 / self.nu)

    @property
    def steps(self) -> int:
        return max(1, int(round(self.t_end / self.dt)))

    def dimensionless(self) -> dict:
        """Every dimensionless number that fixes the run -- reported, not assumed."""
        dt = self.dt
        return {
            "Re_mean": self.reynolds,
            "Re_max": self.rho0 * self.u_max * self.height / self.mu,
            "mach_max": self.u_max / self.sound_speed,
            "courant_acoustic": self.sound_speed * dt / self.h,
            "viscous_number": self.nu * dt / self.h ** 2,
            "H_over_dx": self.resolution,
            "h_over_dx": self.h_factor,
            "L_over_H": self.length_over_height,
            "h_over_H": self.h / self.height,
            "t_end_over_tau": self.t_end_over_tau,
            "tau_diffusive_s": self.tau_diffusive,
            "steps": self.steps,
        }


# --------------------------------------------------------------------------- #
# Exact solution
# --------------------------------------------------------------------------- #

def steady_profile(y, cfg: MorrisConfig):
    """Steady parabola ``u_s(y) = g (H - y) y / (2 nu)`` (float64 numpy)."""
    y = np.asarray(y, dtype=np.float64)
    return cfg.body_force * y * (cfg.height - y) / (2.0 * cfg.nu)


def startup_profile(y, t: float, cfg: MorrisConfig, n_terms: int = 200):
    """Start-up from rest: steady parabola minus the Boussinesq sine series."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    k = np.arange(1, 2 * n_terms, 2, dtype=np.float64).reshape(-1, 1)
    decay = np.exp(-cfg.nu * k ** 2 * math.pi ** 2 * float(t) / cfg.height ** 2)
    series = (np.sin(k * math.pi * y.reshape(1, -1) / cfg.height) * decay
              / k ** 3).sum(axis=0)
    return steady_profile(y, cfg) - (4.0 * cfg.body_force * cfg.height ** 2
                                     / (cfg.nu * math.pi ** 3)) * series


def averaged_profile(y, t1: float, t2: float, cfg: MorrisConfig,
                     n_terms: int = 200):
    """Exact time average of the start-up solution over ``[t1, t2]``.

    The measured profile is averaged over a sampling window, so the reference
    for it must be averaged over the *same* window or the residual transient
    (``exp(-t/tau)``) is charged to the discretisation as a bias.  Each series
    term integrates in closed form::

        (1/W) int_{t1}^{t2} exp(-n^2 t/tau) dt
            = (tau / (n^2 W)) (exp(-n^2 t1/tau) - exp(-n^2 t2/tau))

    with ``tau = H^2/(pi^2 nu)``.  For ``t1 >> tau`` this reduces to the steady
    parabola, which is what makes the two references interchangeable once the
    channel has settled.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    window = max(float(t2) - float(t1), 1e-300)
    tau = cfg.tau_diffusive
    k = np.arange(1, 2 * n_terms, 2, dtype=np.float64).reshape(-1, 1)
    integral = (tau / k ** 2) * (np.exp(-k ** 2 * float(t1) / tau)
                                 - np.exp(-k ** 2 * float(t2) / tau))
    series = (np.sin(k * math.pi * y.reshape(1, -1) / cfg.height)
              * integral / k ** 3).sum(axis=0)
    return steady_profile(y, cfg) - (4.0 * cfg.body_force * cfg.height ** 2
                                     / (cfg.nu * math.pi ** 3)) * series / window


def flow_rate(cfg: MorrisConfig) -> float:
    """``Q = int_0^H u_s dy = g H^3 / (12 nu)`` per unit width."""
    return cfg.body_force * cfg.height ** 3 / (12.0 * cfg.nu)


def derive_reference_symbolically() -> dict:
    """Re-derive the steady solution with sympy; every residual must be 0."""
    try:
        import sympy as sp
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "reason": f"sympy unavailable: {exc}"}

    y, g, nu, height, rho = sp.symbols("y g nu H rho", positive=True)
    u = g * (height - y) * y / (2 * nu)
    residual = sp.simplify(nu * sp.diff(u, y, 2) + g)
    u_max = sp.simplify(u.subs(y, height / 2))
    du_wall = sp.simplify(sp.diff(u, y).subs(y, 0))
    tau_wall = sp.simplify(rho * nu * du_wall)
    q = sp.simplify(sp.integrate(u, (y, 0, height)))
    return {
        "available": True,
        "profile": "u(y) = g*(H-y)*y/(2*nu)",
        "ode_residual_nu_uyy_plus_g": str(residual),
        "ode_residual_is_zero": bool(sp.simplify(residual) == 0),
        "bc_y0": str(sp.simplify(u.subs(y, 0))),
        "bc_yH": str(sp.simplify(u.subs(y, height))),
        "bc_both_zero": bool(sp.simplify(u.subs(y, 0)) == 0
                             and sp.simplify(u.subs(y, height)) == 0),
        "u_max": str(u_max),
        "u_max_matches_gH2_over_8nu": bool(
            sp.simplify(u_max - g * height ** 2 / (8 * nu)) == 0),
        "du_dy_at_wall": str(du_wall),
        "tau_wall": str(tau_wall),
        "tau_wall_matches_rho_gH_over_2": bool(
            sp.simplify(tau_wall - rho * g * height / 2) == 0),
        "flow_rate": str(q),
        "flow_rate_matches_gH3_over_12nu": bool(
            sp.simplify(q - g * height ** 3 / (12 * nu)) == 0),
    }


def verify_series(cfg: MorrisConfig, n_terms: int = 400) -> dict:
    """Certify numerically that the start-up series is the exact solution.

    Three independent statements with their own residual: the initial
    condition, the PDE ``u_t = nu u_yy + g`` (central differences in ``y``, a
    one-sided difference in ``t`` at ``t = 0.3 tau``), and the steady limit
    ``u(y, 10 tau) -> u_s``.  Residuals are dimensionless (divided by ``g`` or
    ``u_max``).  ``truncation_sensitivity`` re-runs the series with a quarter of
    the terms.
    """
    height = cfg.height
    y = np.linspace(0.05 * height, 0.95 * height, 181)

    ic_res = float(np.max(np.abs(startup_profile(y, 0.0, cfg, n_terms))) / cfg.u_max)

    tau = cfg.tau_diffusive
    t_mid = 0.3 * tau
    dt = 1.0e-4 * tau
    dy = float(y[1] - y[0])
    u = startup_profile(y, t_mid, cfg, n_terms)
    u_t = (startup_profile(y, t_mid + dt, cfg, n_terms)
           - startup_profile(y, t_mid - dt, cfg, n_terms)) / (2.0 * dt)
    u_yy = (startup_profile(y + dy, t_mid, cfg, n_terms) - 2.0 * u
            + startup_profile(y - dy, t_mid, cfg, n_terms)) / dy ** 2
    pde_res = float(np.max(np.abs(u_t - cfg.nu * u_yy - cfg.body_force))
                    / cfg.body_force)

    late = startup_profile(y, 10.0 * tau, cfg, n_terms)
    steady_res = float(np.max(np.abs(late - steady_profile(y, cfg))) / cfg.u_max)
    # The leading mode is still exp(-10) = 4.5e-5 alive at t = 10 tau, so that
    # (and not 1e-16) is the residual a correct series must show here.

    few = startup_profile(y, t_mid, cfg, max(1, n_terms // 4))
    trunc_res = float(np.max(np.abs(few - u)) / cfg.u_max)

    return {
        "n_terms": int(n_terms),
        "initial_condition_residual_over_umax": ic_res,
        "pde_residual_over_g": pde_res,
        "steady_limit_residual_over_umax": steady_res,
        "truncation_sensitivity_over_umax": trunc_res,
        "pass": bool(ic_res < 1e-6 and pde_res < 1e-3
                     and steady_res < 1e-4 and trunc_res < 1e-3),
    }


# --------------------------------------------------------------------------- #
# Geometry and time integration
# --------------------------------------------------------------------------- #

def build_channel(cfg: MorrisConfig, device):
    """Fluid lattice on rows ``y = dx/2 + k dx`` with ``x`` in ``[0, L)``."""
    dx = cfg.dx
    xs = torch.arange(dx * 0.5, cfg.length, dx, dtype=torch.float64, device=device)
    ys = torch.arange(dx * 0.5, cfg.height, dx, dtype=torch.float64, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), cfg.rho0 * dx * dx,
                      dtype=torch.float64, device=device)
    return pos, mass


def _solver_config(cfg: MorrisConfig, device) -> Config:
    """Production WCSPH configuration: no gravity, wall blend off (mirror BC)."""
    return Config(
        dx=cfg.dx,
        h_factor=cfg.h_factor,
        rho0=cfg.rho0,
        sound_speed=cfg.sound_speed,
        gamma=7.0,
        gravity=0.0,        # the drive is a streamwise body force, added by hand
        viscosity=cfg.nu,
        dt=cfg.dt,
        steps=0,
        output_every=10 ** 9,
        boundary="wall",
        wall_layers=3,
        # Full wall coupling: the mirror ghosts carry the negated velocity, so
        # the wall/fluid viscous pair is antisymmetric and its mean velocity is
        # zero *on* the plane.  With wall_friction = 0 the ghosts would only
        # enter density/pressure and the fluid would accelerate as a rigid body
        # (measured: flat profile, u = g t) -- the failure mode this constant
        # fixes.
        wall_friction=1.0,
        fail_fast=False,
        device=str(device),
    )


def _periodic_images(pos, length: float):
    """Real cloud plus one image on each side: exact for support < L/2."""
    shift = torch.tensor([length, 0.0], dtype=pos.dtype, device=pos.device)
    return torch.cat((pos, pos + shift, pos - shift), dim=0)


def _bin_rows(pos, u_mean, dx: float, height: float) -> list:
    """Average ``u_x`` on the lattice rows (the flow is invariant in ``x``)."""
    y = np.asarray(pos[:, 1].detach().cpu().numpy(), dtype=np.float64)
    u = np.asarray(u_mean[:, 0].detach().cpu().numpy(), dtype=np.float64)
    row = np.rint((y - 0.5 * dx) / dx).astype(int)
    rows = []
    for k in range(int(round(height / dx))):
        sel = row == k
        if not np.any(sel):
            continue
        rows.append({
            "row": int(k),
            "y": float(0.5 * dx + k * dx),
            "y_over_H": float((0.5 * dx + k * dx) / height),
            "u_mean": float(u[sel].mean()),
            "u_std_x": float(u[sel].std()),
            "count": int(sel.sum()),
        })
    return rows


def run_case(cfg: MorrisConfig, device=None, verbose: bool = False) -> dict:
    """Integrate the Morris channel from rest and return the sampled fields.

    Standard explicit WCSPH update; the case-specific parts are the streamwise
    body force, the two mirror planes and the sampling schedule (start-up
    snapshots at ``t = f tau``, then a time average over the last
    ``average_fraction`` of ``t_end``).
    """
    device = resolve_device(cfg.device if device is None else device)
    torch.manual_seed(0)
    pos, mass = build_channel(cfg, device)
    n = int(pos.shape[0])
    vel = torch.zeros_like(pos)

    scfg = _solver_config(cfg, device)
    dt = cfg.dt
    steps = cfg.steps
    support = 2.0 * cfg.h
    planes = [(0.0, +1), (cfg.height, -1)]
    avg_from = cfg.t_end * (1.0 - cfg.average_fraction)
    targets = [(float(f) * cfg.tau_diffusive, float(f)) for f in cfg.sample_fractions]
    targets.sort()
    taken = set()
    snapshots: list = []
    trace: list = []

    acc_sum = torch.zeros_like(pos)
    n_avg = 0
    wall_clock = time.time()
    n_ghost = 0
    for step in range(steps):
        pos_ext = _periodic_images(pos, cfg.length)
        vel_ext = torch.cat((vel, vel, vel), dim=0)
        mass_ext = torch.cat((mass, mass, mass), dim=0)
        ghost_pos, ghost_vel, _ = plane_walls(pos_ext, vel_ext, mass_ext,
                                             planes, support, slip=cfg.slip)
        n_ghost = int(ghost_pos.shape[0])
        _, _, acc = rhs(pos_ext, vel_ext, mass_ext, scfg,
                        wall=ghost_pos if n_ghost else None,
                        wall_vel=ghost_vel if n_ghost else None)
        acc = acc[:n].clone()
        acc[:, 0] += cfg.body_force

        t = step * dt
        # The state (pos, vel) at this point is the state at time t, so the
        # start-up snapshots are taken *before* the update and labelled with t;
        # sampling after the update would charge one time step (up to 2 % of tau
        # at the coarsest rung) to the discretisation error.
        for target, frac in targets:
            if frac not in taken and t >= target:
                taken.add(frac)
                snapshots.append({
                    "fraction_of_tau": frac,
                    "t": float(t),
                    "t_over_tau": float(t / cfg.tau_diffusive),
                    "rows": _bin_rows(pos, vel, cfg.dx, cfg.height),
                })
        if t >= avg_from:
            acc_sum += vel
            n_avg += 1

        vel = vel + dt * acc
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], cfg.length)

        if step % max(1, steps // 8) == 0 or step == steps - 1:
            ux = vel[:, 0]
            trace.append({
                "t": float(t),
                "t_over_tau": float(t / cfg.tau_diffusive),
                "u_max": float(ux.max()),
                "u_mean": float(ux.mean()),
                "mach_max": float(ux.abs().max()) / cfg.sound_speed,
                "wall_clock_s": time.time() - wall_clock,
            })
        if verbose and step % max(1, steps // 8) == 0:
            print(f"    step {step + 1}/{steps} t/tau={t / cfg.tau_diffusive:.2f} "
                  f"u_max={float(vel[:, 0].max()):.4e}", flush=True)

    u_mean = acc_sum / max(n_avg, 1)
    profile = _bin_rows(pos, u_mean, cfg.dx, cfg.height)

    tail = [p for p in trace if p["t"] >= cfg.t_end * 0.9]
    drift = 0.0
    if len(tail) >= 2 and abs(tail[0]["u_max"]) > 0.0:
        drift = abs(tail[-1]["u_max"] - tail[0]["u_max"]) / abs(tail[0]["u_max"])

    return {
        "n_fluid": n,
        "n_ghost": n_ghost,
        "dt": dt,
        "steps": steps,
        "n_averaged_samples": n_avg,
        "t_end": cfg.t_end,
        "avg_window": [float(avg_from), float(cfg.t_end)],
        "profile": profile,
        "snapshots": snapshots,
        "trace": trace,
        "steady_drift": drift,
        "wall_clock_s": time.time() - wall_clock,
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _fit_quadratic(rows) -> tuple:
    ys = np.array([r["y"] for r in rows], dtype=np.float64)
    us = np.array([r["u_mean"] for r in rows], dtype=np.float64)
    a = np.vstack([np.ones_like(ys), ys, ys ** 2]).T
    coef, *_ = np.linalg.lstsq(a, us, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def _trapz(us, ys) -> float:
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(fn(us, ys))


def evaluate(case: dict, cfg: MorrisConfig, limit_pct: float = 5.0) -> dict:
    """Turn a sampled channel into metrics and gates A..H (5 % predeclared)."""
    rows = case["profile"]
    if len(rows) < 4:
        raise ValueError(f"need at least four rows to fit, got {len(rows)}")
    win = case.get("avg_window") or [0.0, float(cfg.t_end)]
    y_all = [r["y"] for r in rows]
    u_exact = averaged_profile(y_all, win[0], win[1], cfg)
    # Reference peak = the peak of the exact solution averaged over the *same*
    # window as the measurement (equals g H^2/(8 nu) up to the residual
    # transient, 0.03 % at t_end = 8 tau).
    u_max_ref = float(np.max(u_exact))
    u_max_meas = max(r["u_mean"] for r in rows)
    per_row = []
    for r, u_ref in zip(rows, u_exact):
        u_ref = float(u_ref)
        per_row.append({**r, "u_ref": u_ref,
                        "rel_error_pct": 100.0 * (r["u_mean"] - u_ref) / u_ref})
    interior = [r for r in per_row if 0.1 <= r["y_over_H"] <= 0.9]
    rms_interior = math.sqrt(sum(r["rel_error_pct"] ** 2 for r in interior)
                             / len(interior))
    rms_all = math.sqrt(sum(r["rel_error_pct"] ** 2 for r in per_row) / len(per_row))

    ys = np.array([r["y"] for r in rows])
    us = np.array([r["u_mean"] for r in rows])
    c0, c1, c2 = _fit_quadratic(rows)
    height = cfg.height
    tau_bottom = cfg.rho0 * cfg.nu * abs(c1)
    tau_top = cfg.rho0 * cfg.nu * abs(c1 + 2.0 * c2 * height)
    # Same-estimator reference: the parabola fit is applied to the exact
    # (window-averaged) profile on the same rows, so gate C measures the
    # discretisation and not the width of the averaging window.
    ref_rows = [{"y": float(yy), "u_mean": float(uu)}
                for yy, uu in zip(ys, np.asarray(u_exact, dtype=np.float64))]
    rc0, rc1, rc2 = _fit_quadratic(ref_rows)
    tau_ref_bottom = cfg.rho0 * cfg.nu * abs(rc1)
    tau_ref_top = cfg.rho0 * cfg.nu * abs(rc1 + 2.0 * rc2 * height)
    tau_ref = cfg.tau_wall
    slip_bottom = abs(c0) / u_max_meas
    slip_top = abs(c0 + c1 * height + c2 * height ** 2) / u_max_meas

    q_num = _trapz(us, ys)
    # Same-estimator reference: the rows only span [dx/2, H-dx/2], so a
    # trapezoid there is ~0.6 % short of the closed-form Q = g H^3/(12 nu) even
    # on exact data.  Gate E therefore compares the measured trapezoid with the
    # trapezoid of the exact profile on the *same rows*; the estimator's own
    # bias against the closed form is reported separately.
    q_ref_est = _trapz(np.asarray(u_exact, dtype=np.float64), ys)

    centre_err = 100.0 * (u_max_meas / u_max_ref - 1.0)
    tau_err = 100.0 * max(abs(tau_bottom / tau_ref_bottom - 1.0),
                          abs(tau_top / tau_ref_top - 1.0))
    tau_err_closed = 100.0 * max(abs(tau_bottom / tau_ref - 1.0),
                                 abs(tau_top / tau_ref - 1.0))
    q_err = 100.0 * (q_num / q_ref_est - 1.0)
    q_err_closed = 100.0 * (q_num / cfg.q_ref - 1.0)
    slip_pct = 100.0 * max(slip_bottom, slip_top)

    gates = {
        "A_centre_line": {"value_pct": centre_err, "limit_pct": limit_pct,
                          "pass": bool(abs(centre_err) <= limit_pct)},
        "B_profile_shape": {"value_pct": rms_interior, "limit_pct": limit_pct,
                            "pass": bool(rms_interior <= limit_pct)},
        "C_wall_shear": {"value_pct": tau_err, "limit_pct": limit_pct,
                         "pass": bool(tau_err <= limit_pct)},
        "D_wall_slip": {"value_pct": slip_pct, "limit_pct": limit_pct,
                        "pass": bool(slip_pct <= limit_pct)},
        "E_flow_rate": {"value_pct": q_err, "limit_pct": limit_pct,
                        "pass": bool(abs(q_err) <= limit_pct)},
    }

    transient = []
    for snap in case.get("snapshots", []):
        t = snap["t"]
        y_rows = [r["y"] for r in snap["rows"]]
        exact = startup_profile(y_rows, t, cfg)
        u_num = np.array([r["u_mean"] for r in snap["rows"]])
        u_max_t = float(startup_profile([0.5 * height], t, cfg)[0])
        scale = max(abs(u_max_t), 1e-30)
        err = 100.0 * float(np.max(np.abs(u_num - exact))) / scale
        # Normalisation.  A *relative* error is meaningless at start-up: the
        # exact profile vanishes at t = 0, so |u_num-u_exact|/u_max(t) diverges
        # for reasons that have nothing to do with the discretisation.  The
        # start-up gate is therefore the absolute error over the case velocity
        # scale u_max,inf; the centre-row relative error is reported alongside
        # because there the denominator is the largest and it stays meaningful.
        err_scale = 100.0 * float(np.max(np.abs(u_num - exact))) / cfg.u_max
        i_c = int(np.argmin(np.abs(np.array(y_rows) - 0.5 * height)))
        rel_c = 100.0 * (u_num[i_c] - exact[i_c]) / max(abs(exact[i_c]), 1e-30)
        worst_y = float(y_rows[int(np.argmax(np.abs(u_num - exact)))]) / height
        transient.append({
            "fraction_of_tau": snap["fraction_of_tau"],
            "t_over_tau": snap["t_over_tau"],
            "u_max_exact": u_max_t,
            "u_max_measured": float(u_num.max()),
            "max_abs_error_over_umax_inf_pct": err_scale,
            "max_abs_error_over_umax_t_pct": err,
            "centre_row_rel_error_pct": rel_c,
            "worst_row_y_over_H": worst_y,
            "limit_pct": limit_pct,
            "pass": bool(err_scale <= limit_pct),
        })
    if transient:
        f_err = max(s["max_abs_error_over_umax_inf_pct"] for s in transient)
        gates["F_startup"] = {"value_pct": f_err, "limit_pct": limit_pct,
                              "pass": bool(f_err <= limit_pct)}
    gates["G_steady_state"] = {"value_pct": 100.0 * case["steady_drift"],
                               "limit_pct": 1.0,
                               "pass": bool(case["steady_drift"] <= 0.01)}

    return {
        "resolution_H_over_dx": cfg.resolution,
        "dx": cfg.dx,
        "dt": case["dt"],
        "steps": case["steps"],
        "n_fluid": case["n_fluid"],
        "n_ghost": case["n_ghost"],
        "n_averaged_samples": case["n_averaged_samples"],
        "u_max_measured": u_max_meas,
        "u_max_reference": u_max_ref,
        "wall_shear_bottom": tau_bottom,
        "wall_shear_top": tau_top,
        "wall_shear_reference": tau_ref,
        "wall_shear_reference_estimator": [tau_ref_bottom, tau_ref_top],
        "wall_shear_error_vs_closed_form_pct": tau_err_closed,
        "flow_rate_measured": q_num,
        "flow_rate_reference": q_ref_est,
        "flow_rate_closed_form": cfg.q_ref,
        "flow_rate_error_vs_closed_form_pct": q_err_closed,
        "flow_rate_estimator_bias_pct": 100.0 * (q_ref_est / cfg.q_ref - 1.0),
        "slip_bottom_over_umax": slip_bottom,
        "slip_top_over_umax": slip_top,
        "profile_rms_interior_pct": rms_interior,
        "profile_rms_all_rows_pct": rms_all,
        "fit": {"c0": c0, "c1": c1, "c2": c2},
        "steady_drift": case["steady_drift"],
        "wall_clock_s": case["wall_clock_s"],
        "rows": per_row,
        "transient": transient,
        "gates": gates,
        "pass": bool(all(g["pass"] for g in gates.values())),
    }


def nodes_table(evaluation: dict, cfg: MorrisConfig) -> list:
    """Flat ref-vs-got table -- the evidence unit the case report quotes."""
    tau_ref = cfg.tau_wall
    tau_est = evaluation.get("wall_shear_reference_estimator") or [tau_ref, tau_ref]
    return [
        {"name": "u_max", "symbol": "u_max", "unit": "m/s",
         "reference": cfg.u_max, "got": evaluation["u_max_measured"],
         "rel_error_pct": 100.0 * (evaluation["u_max_measured"] / cfg.u_max - 1.0)},
        {"name": "profile_rms_interior", "symbol": "RMS(u)", "unit": "%",
         "reference": 0.0, "got": evaluation["profile_rms_interior_pct"],
         "rel_error_pct": evaluation["profile_rms_interior_pct"]},
        {"name": "wall_shear_bottom", "symbol": "tau_w(0)", "unit": "Pa",
         "reference": tau_est[0], "got": evaluation["wall_shear_bottom"],
         "rel_error_pct": 100.0 * (evaluation["wall_shear_bottom"]
                                  / tau_est[0] - 1.0)},
        {"name": "wall_shear_top", "symbol": "tau_w(H)", "unit": "Pa",
         "reference": tau_est[1], "got": evaluation["wall_shear_top"],
         "rel_error_pct": 100.0 * (evaluation["wall_shear_top"]
                                  / tau_est[1] - 1.0)},
        {"name": "wall_shear_vs_closed_form", "symbol": "tau_w (rho g H/2)",
         "unit": "Pa", "reference": tau_ref,
         "got": evaluation["wall_shear_bottom"],
         "rel_error_pct": evaluation["wall_shear_error_vs_closed_form_pct"]},
        {"name": "flow_rate", "symbol": "Q", "unit": "m^2/s",
         "reference": evaluation["flow_rate_reference"],
         "got": evaluation["flow_rate_measured"],
         "rel_error_pct": 100.0 * (evaluation["flow_rate_measured"]
                                  / evaluation["flow_rate_reference"] - 1.0)},
        {"name": "flow_rate_vs_closed_form", "symbol": "Q (closed form)",
         "unit": "m^2/s", "reference": cfg.q_ref,
         "got": evaluation["flow_rate_measured"],
         "rel_error_pct": evaluation["flow_rate_error_vs_closed_form_pct"]},
        {"name": "wall_slip", "symbol": "u_wall/u_max", "unit": "-",
         "reference": 0.0, "got": evaluation["slip_bottom_over_umax"],
         "rel_error_pct": 100.0 * evaluation["slip_bottom_over_umax"]},
    ]


# --------------------------------------------------------------------------- #
# Ladder, sensitivity and the artifact payload
# --------------------------------------------------------------------------- #

def _config_record(cfg: MorrisConfig) -> dict:
    return {
        "case_key": CASE_KEY,
        "height_m": cfg.height,
        "length_m": cfg.length,
        "rho0": cfg.rho0,
        "mu": cfg.mu,
        "nu": cfg.nu,
        "reynolds_mean": cfg.reynolds,
        "body_force_m_per_s2": cfg.body_force,
        "u_mean_reference": cfg.u_mean,
        "u_max_reference": cfg.u_max,
        "tau_wall_reference": cfg.tau_wall,
        "flow_rate_reference": cfg.q_ref,
        "tau_diffusive_s": cfg.tau_diffusive,
        "t_end_s": cfg.t_end,
        "dt_s": cfg.dt,
        "steps": cfg.steps,
        "resolution_H_over_dx": cfg.resolution,
        "dx_m": cfg.dx,
        "h_m": cfg.h,
        "h_factor": cfg.h_factor,
        "sound_speed": cfg.sound_speed,
        "wall_model": cfg.wall_model,
        "slip": cfg.slip,
        "dimensionless": cfg.dimensionless(),
    }


def run_ladder(cfg: MorrisConfig, resolutions=(8, 12, 16, 24), device=None,
               verbose: bool = False) -> dict:
    """Resolution ladder: one evaluation per ``H/dx`` plus the observed order."""
    device = resolve_device(cfg.device if device is None else device)
    entries = []
    for res in resolutions:
        run_cfg = dataclasses.replace(cfg, resolution=int(res))
        case = run_case(run_cfg, device=device, verbose=verbose)
        evaluation = evaluate(case, run_cfg)
        evaluation["nodes"] = nodes_table(evaluation, run_cfg)
        if verbose:
            g = evaluation["gates"]
            print(f"  [ladder] H/dx={res:3d} u_max={g['A_centre_line']['value_pct']:+.2f}% "
                  f"RMS={g['B_profile_shape']['value_pct']:.2f}% "
                  f"tau={g['C_wall_shear']['value_pct']:.2f}% "
                  f"Q={g['E_flow_rate']['value_pct']:+.2f}% "
                  f"({case['wall_clock_s']:.1f}s)", flush=True)
        entries.append({"config": _config_record(run_cfg),
                        "trace": case["trace"],
                        "evaluation": evaluation})

    order = None
    if len(entries) >= 2:
        fine, coarse = entries[-1], entries[-2]
        e_f = fine["evaluation"]["profile_rms_interior_pct"]
        e_c = coarse["evaluation"]["profile_rms_interior_pct"]
        if e_f > 0.0 and e_c > 0.0:
            order = float(math.log(e_c / e_f)
                          / math.log(coarse["config"]["dx_m"] / fine["config"]["dx_m"]))
    monotone = all(entries[i + 1]["evaluation"]["profile_rms_interior_pct"]
                   <= entries[i]["evaluation"]["profile_rms_interior_pct"] * 1.02
                   for i in range(len(entries) - 1))
    tolerance_ok = all(e["evaluation"]["pass"] for e in entries)
    return {
        "resolutions": [int(r) for r in resolutions],
        "entries": entries,
        "observed_order": order,
        "profile_rms_monotone": bool(monotone),
        "pass": bool(tolerance_ok and monotone),
    }


def run_sensitivity(cfg: MorrisConfig, device=None, resolution: int = 12,
                    verbose: bool = False) -> dict:
    """One-at-a-time sensitivity: sound speed, run length, kernel support.

    Each axis is a numerical/physical knob the acceptance numbers depend on and
    each variation is a full run.  They are run one rung coarser than the
    accuracy ladder (default ``H/dx = 12``): this is a *spread* study -- how
    much the answer moves when a knob moves -- not the accuracy measurement,
    and the spread is quoted with the rung it was measured on.
    """
    device = resolve_device(cfg.device if device is None else device)
    cfg = dataclasses.replace(cfg, resolution=int(resolution))
    axes = {
        "sound_speed": [20.0, 40.0, 80.0],
        "t_end_over_tau": [1.0, 2.0, 3.0],
        "h_factor": [1.2, 1.3, 1.5],
    }
    out = {}
    for axis, values in axes.items():
        runs = []
        for value in values:
            run_cfg = dataclasses.replace(cfg, **{axis: value})
            case = run_case(run_cfg, device=device)
            evaluation = evaluate(case, run_cfg)
            runs.append({
                "value": value,
                "u_max_error_pct": evaluation["gates"]["A_centre_line"]["value_pct"],
                "profile_rms_interior_pct": evaluation["profile_rms_interior_pct"],
                "wall_shear_error_pct": evaluation["gates"]["C_wall_shear"]["value_pct"],
                "flow_rate_error_pct": evaluation["gates"]["E_flow_rate"]["value_pct"],
                "mach_max": run_cfg.dimensionless()["mach_max"],
                "steps": run_cfg.steps,
                "pass": evaluation["pass"],
            })
            if verbose:
                print(f"  [sensitivity] {axis}={value:g} "
                      f"RMS={runs[-1]['profile_rms_interior_pct']:.2f}% "
                      f"tau={runs[-1]['wall_shear_error_pct']:.2f}%", flush=True)
        spread = (max(r["profile_rms_interior_pct"] for r in runs)
                  - min(r["profile_rms_interior_pct"] for r in runs))
        out[axis] = {"values": values, "runs": runs,
                     "resolution_H_over_dx": int(resolution),
                     "profile_rms_spread_pct": float(spread)}
    return out


def _environment() -> dict:
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "tensorsph": __version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
    }
    try:
        import subprocess
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=10, check=False).stdout.strip() or None
    except Exception:  # pragma: no cover - no git in a wheel
        env["git_commit"] = None
    return env


def build_payload(cfg: MorrisConfig, ladder: dict, sensitivity: dict,
                  device=None, figures=None) -> dict:
    """Assemble the run artifact: reference provenance, config, ladder, verdict."""
    finest = ladder["entries"][-1]["evaluation"]
    return {
        "case_key": CASE_KEY,
        "anchor": CASE_KEY,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "reference": {
            "bibliography": REFERENCE,
            "provenance": ("reference values are computed from the exact solution "
                           "at run time; the symbolic and series certificates "
                           "below are the anti-transcription evidence"),
            "parameter_provenance": (
                "the case is defined by dimensionless groups (Re_mean, h/H, L/H); "
                "the drive is derived from Re, not tabulated. The Morris et al. "
                "paper's own tabulated parameters were not re-read offline, so no "
                "literal table values are claimed (WITHHELD); the exact solution "
                "used for grading holds for any parameter set."),
            "symbolic": derive_reference_symbolically(),
            "series": verify_series(cfg),
            "values": {
                "u_mean": cfg.u_mean,
                "u_max": cfg.u_max,
                "tau_wall": cfg.tau_wall,
                "flow_rate": cfg.q_ref,
                "tau_diffusive_s": cfg.tau_diffusive,
            },
            "startup_centreline": [
                {"t_over_tau": float(f),
                 "u_max_exact": float(startup_profile([0.5 * cfg.height],
                                                      f * cfg.tau_diffusive, cfg)[0])}
                for f in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0)
            ],
        },
        "config": _config_record(cfg),
        "ladder": ladder,
        "sensitivity": sensitivity,
        "finest": {
            "resolution_H_over_dx": finest["resolution_H_over_dx"],
            "nodes": finest["nodes"],
            "gates": finest["gates"],
            "pass": finest["pass"],
        },
        "convergence": {
            "observed_order": ladder["observed_order"],
            "profile_rms_monotone": ladder["profile_rms_monotone"],
            "profile_rms_pct_by_resolution": {
                str(e["config"]["resolution_H_over_dx"]):
                    e["evaluation"]["profile_rms_interior_pct"]
                for e in ladder["entries"]},
        },
        "environment": _environment(),
        "figures": figures or [],
        "repro_cmd": ("PYTHONPATH=src python -m tensorsph.case_poiseuille_morris "
                      "--out runs/case_poiseuille_morris "
                      "--resolutions 8 12 16 24"),
        "verdict": {
            "tolerance_pct": 5.0,
            "pass": bool(ladder["pass"]),
            "gates": {k: bool(v["pass"]) for k, v in finest["gates"].items()},
        },
    }


def make_figures(payload: dict, outdir: Path) -> list:
    """Profile vs exact, per-row error, convergence and start-up history."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        return [{"error": f"matplotlib unavailable: {exc}"}]

    outdir.mkdir(parents=True, exist_ok=True)
    figs = []
    entries = payload["ladder"]["entries"]
    cfg = entries[-1]["config"]
    height = cfg["height_m"]
    u_max = cfg["u_max_reference"]
    y_ref = np.linspace(0.0, height, 200)
    u_ref = (cfg["body_force_m_per_s2"] * (height - y_ref) * y_ref
             / (2.0 * cfg["nu"]))

    fig, (ax, axe) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax.plot(y_ref / height, u_ref / u_max, "k-", lw=1.6,
            label="exact (steady parabola)")
    for entry in entries:
        rows = entry["evaluation"]["rows"]
        ax.plot([r["y_over_H"] for r in rows],
                [r["u_mean"] / u_max for r in rows], "o", ms=3.5,
                label=f"SPH H/dx={entry['config']['resolution_H_over_dx']}")
        axe.plot([r["y_over_H"] for r in rows],
                 [r["rel_error_pct"] for r in rows], "o-", ms=3.0, lw=0.8,
                 label=f"H/dx={entry['config']['resolution_H_over_dx']}")
    ax.set_xlabel("y / H")
    ax.set_ylabel("u / u_max")
    ax.set_title("Morris low-Re channel: steady profile")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    axe.axhspan(-5.0, 5.0, color="0.85", zorder=0, label="5 % band")
    axe.axhline(0.0, color="k", lw=0.6)
    axe.set_xlabel("y / H")
    axe.set_ylabel("(u_num - u_exact) / u_exact  [%]")
    axe.set_title("per-row relative error")
    axe.legend(fontsize=7)
    axe.grid(alpha=0.3)
    fig.tight_layout()
    path = outdir / "poiseuille_morris_profile.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": path.name,
                 "caption": ("steady streamwise velocity vs the exact parabola "
                             "(left) and the per-row relative error against the "
                             "5 % band (right)")})

    fig, (axc, axt) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    dxs = [e["config"]["dx_m"] for e in entries]
    rms = [e["evaluation"]["profile_rms_interior_pct"] for e in entries]
    axc.loglog(dxs, rms, "o-", label="measured interior row RMS")
    axc.loglog(dxs, [rms[0] * (d / dxs[0]) ** 2 for d in dxs], "k--",
               label="slope 2 reference")
    axc.set_xlabel("dx [m]")
    axc.set_ylabel("profile RMS [%]")
    order = payload["ladder"]["observed_order"]
    axc.set_title(f"convergence (observed order {order:.2f})" if order is not None
                  else "convergence")
    axc.legend(fontsize=7)
    axc.grid(alpha=0.3, which="both")

    exact = payload["reference"]["startup_centreline"]
    axt.plot([s["t_over_tau"] for s in exact],
             [s["u_max_exact"] / u_max for s in exact], "k-", lw=1.5,
             label="exact start-up series")
    finest = entries[-1]["evaluation"]
    axt.plot([s["t_over_tau"] for s in finest["transient"]],
             [s["u_max_measured"] / u_max for s in finest["transient"]],
             "s", ms=6, label=f"SPH H/dx={finest['resolution_H_over_dx']}")
    axt.set_xlabel("t / tau   (tau = H^2 / (pi^2 nu))")
    axt.set_ylabel("u_centre / u_max")
    axt.set_title("start-up from rest")
    axt.legend(fontsize=7)
    axt.grid(alpha=0.3)
    fig.tight_layout()
    path = outdir / "poiseuille_morris_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": path.name,
                 "caption": ("profile error vs dx with a slope-2 reference "
                             "(left) and the measured centre-line start-up "
                             "against the exact series (right)")})
    return figs


def write_artifacts(payload: dict, outdir: Path) -> dict:
    """Write ``results.json`` and ``summary.md`` under ``outdir``."""
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "results.json"
    json_path.write_text(json.dumps(payload, indent=1, default=float))

    lines = [f"# {CASE_KEY} -- ref vs got", ""]
    lines.append("| H/dx | u_max [m/s] | err % | RMS % | tau_w err % "
                 "| Q err % | slip % | pass |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for entry in payload["ladder"]["entries"]:
        ev = entry["evaluation"]
        gates = ev["gates"]
        lines.append(
            f"| {entry['config']['resolution_H_over_dx']} "
            f"| ref {ev['u_max_reference']:.6e} / got {ev['u_max_measured']:.6e} "
            f"| {gates['A_centre_line']['value_pct']:+.2f} "
            f"| {gates['B_profile_shape']['value_pct']:.2f} "
            f"| {gates['C_wall_shear']['value_pct']:.2f} "
            f"| {gates['E_flow_rate']['value_pct']:+.2f} "
            f"| {gates['D_wall_slip']['value_pct']:.2f} "
            f"| {'PASS' if ev['pass'] else 'FAIL'} |")
    lines.append("")
    lines.append(f"- observed order (interior RMS): {payload['ladder']['observed_order']}")
    lines.append(f"- repro: `{payload['repro_cmd']}`")
    summary_path = outdir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    return {"results_json": str(json_path), "summary_md": str(summary_path)}


def main(argv=None) -> int:
    """CLI entry point: run the ladder, write artifacts, return 0 when passing."""
    parser = argparse.ArgumentParser(
        prog="python -m tensorsph.case_poiseuille_morris",
        description=f"{CASE_KEY}: low-Re plane Poiseuille channel vs the exact "
                    "Boussinesq/parabolic solution (tolerance 5 %).")
    parser.add_argument("--out", type=Path, default=Path("runs/case_poiseuille_morris"))
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16, 24])
    parser.add_argument("--reynolds", type=float, default=0.1)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=1.0e-3)
    parser.add_argument("--sound-speed", type=float, default=20.0)
    parser.add_argument("--t-end-over-tau", type=float, default=3.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="CI-sized: one rung, short run, no sensitivity")
    parser.add_argument("--print-table", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    cfg = MorrisConfig(height=args.height, mu=args.mu, reynolds=args.reynolds,
                       sound_speed=args.sound_speed,
                       t_end_over_tau=args.t_end_over_tau,
                       device=args.device)
    if args.quick:
        cfg = dataclasses.replace(cfg, resolution=8, t_end_over_tau=8.0)
        args.resolutions = [8]
        args.no_sensitivity = True

    t0 = time.time()
    print(f"[{CASE_KEY}] Re={cfg.reynolds:g} mu={cfg.mu:g} Pa s "
          f"nu={cfg.nu:.3e} m2/s H={cfg.height:g} m dx={cfg.dx:.3e} m "
          f"g={cfg.body_force:.4e} m/s2 u_max={cfg.u_max:.4e} m/s "
          f"tau={cfg.tau_diffusive:.3e} s", flush=True)
    ladder = run_ladder(cfg, resolutions=args.resolutions, device=device,
                        verbose=True)
    sensitivity = {} if args.no_sensitivity else run_sensitivity(cfg, device=device,
                                                                verbose=True)
    payload = build_payload(cfg, ladder, sensitivity, device=device)
    figures = [] if args.no_figures else make_figures(payload, args.out / "figures")
    payload["figures"] = figures
    paths = write_artifacts(payload, args.out)

    if args.print_table:
        print((args.out / "summary.md").read_text())
    verdict = payload["verdict"]
    print(f"[{CASE_KEY}] pass={verdict['pass']} gates={verdict['gates']} "
          f"order={payload['ladder']['observed_order']} "
          f"({time.time() - t0:.1f}s) -> {paths['results_json']}", flush=True)
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
