"""Rung 2 of the verification ladder: Taylor-Green vortex viscous decay (periodic).

Why this case is a *closed* anchor
----------------------------------
The two-dimensional Taylor-Green vortex

.. math::

    u(x,y,t) = U_0 e^{-2\\nu k^2 t}\\sin(kx)\\cos(ky),\\\\
    v(x,y,t) = -U_0 e^{-2\\nu k^2 t}\\cos(kx)\\sin(ky),\\\\
    p(x,y,t) = \\frac{\\rho U_0^2}{4} e^{-4\\nu k^2 t}
               \\bigl(\\cos 2kx + \\cos 2ky\\bigr)

is an **exact, unsteady** solution of the incompressible Navier-Stokes
equations, at any Reynolds number. Two identities make it exact:

* the convective term is a pure gradient,
  ``(u.grad)u = -grad[ (U0^2/4) e^{-4 nu k^2 t} (cos 2kx + cos 2ky) ]``, so the
  pressure field absorbs it at every instant (nothing is projected away);
* ``lap u = -2 k^2 u`` for this field, so viscosity only rescales the amplitude.

Because the *shape* is an exact invariant, the reference is not a numerical
solution and not a fit: the kinetic energy must decay as

.. math::  E(t) = E(0) e^{-4\\nu k^2 t}

so the measured decay rate ``s = -d ln E / dt`` can be compared with the
analytic exponent ``4 nu k^2`` directly. That is the quantity gated here.

Discretisation notes
--------------------
* Periodicity is enforced by the **minimum image** convention on a square box of
  side ``L``; the neighbour list is rebuilt every ``rebuild_every`` steps from
  the current (wrapped) positions with a skin of ``0.4 h``.
* Particle masses use :func:`tensorsph.droplet_impact.calibrated_particle_mass`,
  so the lattice summation density is ``rho0`` to machine precision at ``t = 0``
  and there is no spurious startup pressure.
* The viscous operator is the Morris et al. (1997) discrete Laplacian with the
  library's ``VISCOUS_LAPLACIAN_FACTOR = 2`` -- the *same* operator the
  production :func:`tensorsph.droplet_impact.rhs` applies, so this anchor tests
  the library's viscosity, not a private implementation of it. The pressure
  term is the same symmetric ``p_i/rho_i^2 + p_j/rho_j^2`` form and the Tait
  equation of state is the library's ``tait_pressure``.
* Nothing is randomised: the initial condition is a lattice evaluated
  analytically.

Output is a plain dictionary (see :func:`run_tgv`), suitable for the ladder JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from ..droplet_impact import (VISCOUS_LAPLACIAN_FACTOR, Config,
                              calibrated_particle_mass, cubic_spline_kernel,
                              pair_kernels, tait_pressure)


@dataclass
class TGVConfig:
    """Taylor-Green case parameters (periodic square, one wavelength per side)."""

    L: float = 1.0                  # box side (m); the field is periodic here
    dx: float = 0.025               # particle spacing (m)
    h_factor: float = 1.3           # smoothing length / dx
    rho0: float = 1000.0
    sound_speed: float = 20.0
    gamma: float = 7.0
    viscosity: float = 1.0 / (10.0 * 2.0 * math.pi)  # nu = U0/(Re k), Re = 10
    U0: float = 1.0                 # peak initial velocity (m/s)
    waves: int = 1                  # k = 2 pi waves / L
    max_courant: float = 0.3
    t_end: float = 1.0              # simulated time (s)
    sample_every: int = 10
    rebuild_every: int = 20
    skin_factor: float = 0.4
    dtype: str = "float64"
    device: str = "cpu"

    # ---- derived -----------------------------------------------------------
    @property
    def k(self) -> float:
        return 2.0 * math.pi * self.waves / self.L

    def h(self) -> float:
        return self.h_factor * self.dx

    def dt(self) -> float:
        velocity = self.U0 + self.sound_speed
        return self.max_courant * self.h() / velocity

    def analytic_decay_rate(self) -> float:
        """``4 nu k^2``: the exponent in ``E(t) = E(0) exp(-rate t)``."""
        return 4.0 * self.viscosity * self.k ** 2

    def analytic_amplitude_rate(self) -> float:
        """``2 nu k^2``: the exponent in ``|u|(t) = U0 exp(-rate t)``."""
        return 2.0 * self.viscosity * self.k ** 2

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float64": torch.float64}[self.dtype]


def make_lattice(cfg: TGVConfig):
    """Square lattice of cell centres plus calibrated masses (periodic box)."""
    n = int(round(cfg.L / cfg.dx))
    if abs(n * cfg.dx - cfg.L) > 1e-9 * cfg.L:
        raise ValueError(
            f"dx={cfg.dx:g} does not divide the periodic box L={cfg.L:g}; "
            "a periodic lattice must close exactly")
    line = (torch.arange(n, dtype=cfg.torch_dtype, device=cfg.device) + 0.5) * cfg.dx
    gx, gy = torch.meshgrid(line, line, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    probe = Config(dx=cfg.dx, h_factor=cfg.h_factor, rho0=cfg.rho0)
    m = calibrated_particle_mass(probe)
    mass = torch.full((pos.shape[0],), m, dtype=cfg.torch_dtype, device=cfg.device)
    return pos, mass


def analytic_field(pos: torch.Tensor, t: float, cfg: TGVConfig):
    """Exact velocity field ``(u, v)`` at time ``t`` on positions ``pos``."""
    a = cfg.U0 * math.exp(-cfg.analytic_amplitude_rate() * t)
    k = cfg.k
    u = a * torch.sin(k * pos[:, 0]) * torch.cos(k * pos[:, 1])
    v = -a * torch.cos(k * pos[:, 0]) * torch.sin(k * pos[:, 1])
    return torch.stack((u, v), dim=-1)


def analytic_pressure(pos: torch.Tensor, t: float, cfg: TGVConfig) -> torch.Tensor:
    a2 = (cfg.rho0 * cfg.U0 ** 2 / 4.0
          * math.exp(-cfg.analytic_decay_rate() * t))
    k = cfg.k
    return a2 * (torch.cos(2 * k * pos[:, 0]) + torch.cos(2 * k * pos[:, 1]))


def minimum_image(delta: torch.Tensor, box: float) -> torch.Tensor:
    """Shortest periodic image of ``delta`` in a cubic box of side ``box``."""
    return delta - box * torch.round(delta / box)


def periodic_pairs(pos: torch.Tensor, cutoff: float, box: float,
                   chunk: int = 512):
    """Unordered ``i < j`` pair list under the minimum-image convention.

    Only indices are returned: the separation has to be re-evaluated from the
    *current* positions on every step (a cached separation silently freezes the
    geometry and turns the scheme unstable -- measured on the first draft of
    this module, where the energy grew by five orders of magnitude).
    """
    n = pos.shape[0]
    rows_i, rows_j = [], []
    index = torch.arange(n, device=pos.device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = pos[start:stop, None, :] - pos[None, :, :]
        block = block - box * torch.round(block / box)
        r2 = block.square().sum(dim=-1)
        ii = index[start:stop, None].expand(stop - start, n)
        jj = index[None, :].expand(stop - start, n)
        mask = (r2 < cutoff * cutoff) & (jj > ii)
        if not bool(mask.any()):
            continue
        rows_i.append(ii[mask])
        rows_j.append(jj[mask])
    if not rows_i:
        empty = torch.zeros(0, dtype=torch.long, device=pos.device)
        return empty, empty
    return torch.cat(rows_i), torch.cat(rows_j)


def tgv_rhs(pos, vel, mass, cfg: TGVConfig, pairs=None):
    """Periodic WCSPH right-hand side (no gravity, no walls, no surface tension).

    Mirrors :func:`tensorsph.droplet_impact.rhs` for the fluid-fluid terms so the
    anchor measures the production discretisation.
    """
    h = cfg.h()
    if pairs is None:
        pairs = periodic_pairs(pos, 2.0 * h, cfg.L)
    i, j = pairs
    delta = minimum_image(pos[i] - pos[j], cfg.L)
    n = pos.shape[0]
    zero = torch.zeros(1, dtype=pos.dtype, device=pos.device)
    w0 = cubic_spline_kernel(zero, h).squeeze()

    rho = mass * w0
    if i.numel() == 0:
        return rho, tait_pressure(rho, cfg), torch.zeros_like(pos)

    distance = torch.linalg.vector_norm(delta, dim=1)
    wij, grad_w = pair_kernels(delta, h)
    rho = rho.index_add(0, i, mass[j] * wij).index_add(0, j, mass[i] * wij)

    pressure = tait_pressure(rho, cfg)
    term = pressure[i] / rho[i].square() + pressure[j] / rho[j].square()
    pair_pressure = term[:, None] * grad_w
    acc = torch.zeros_like(pos)
    acc = acc.index_add(0, i, -mass[j, None] * pair_pressure)
    acc = acc.index_add(0, j, mass[i, None] * pair_pressure)

    acc_visc = viscous_acceleration(pos, vel, mass, cfg, pairs=(i, j), rho=rho,
                                    delta=delta, grad_w=grad_w)
    return rho, pressure, acc + acc_visc


def viscous_acceleration(pos, vel, mass, cfg: TGVConfig, pairs=None,
                         rho=None, delta=None, grad_w=None):
    """Morris et al. (1997) discrete Laplacian term alone.

    Split out so the operator can be pinned on its own: on the Taylor-Green
    velocity field the exact answer is ``nu * lap(u) = -2 nu k^2 u``, and
    ``tests/test_vv_ladder.py::test_viscous_operator_reproduces_the_laplacian``
    measures the ratio. Uses the library's ``VISCOUS_LAPLACIAN_FACTOR``, so a
    change to the production viscosity changes this number too.
    """
    h = cfg.h()
    if pairs is None:
        pairs = periodic_pairs(pos, 2.4 * h, cfg.L)
    i, j = pairs
    if delta is None:
        delta = minimum_image(pos[i] - pos[j], cfg.L)
    distance = torch.linalg.vector_norm(delta, dim=1)
    if grad_w is None:
        _w, grad_w = pair_kernels(delta, h)
    if rho is None:
        rho = density(pos, mass, cfg, pairs=pairs)
    dv = vel[i] - vel[j]
    denom = distance.square() + 0.01 * h * h
    laplace_weight = torch.relu(-(delta * grad_w).sum(dim=1)) / denom
    scale = VISCOUS_LAPLACIAN_FACTOR * cfg.viscosity
    visc_i = -scale * dv * (mass[j] / rho[j] * laplace_weight)[:, None]
    visc_j = scale * dv * (mass[i] / rho[i] * laplace_weight)[:, None]
    acc_visc = torch.zeros_like(pos)
    acc_visc = acc_visc.index_add(0, i, visc_i)
    acc_visc = acc_visc.index_add(0, j, visc_j)
    return acc_visc


def density(pos, mass, cfg: TGVConfig, pairs=None):
    """Summation density on the periodic lattice (self term included)."""
    h = cfg.h()
    if pairs is None:
        pairs = periodic_pairs(pos, 2.4 * h, cfg.L)
    i, j = pairs
    zero = torch.zeros(1, dtype=pos.dtype, device=pos.device)
    w0 = cubic_spline_kernel(zero, h).squeeze()
    rho = mass * w0
    delta = minimum_image(pos[i] - pos[j], cfg.L)
    _w, _g = pair_kernels(delta, h)
    return rho.index_add(0, i, mass[j] * _w).index_add(0, j, mass[i] * _w)


def viscous_operator_error(cfg: TGVConfig) -> float:
    """Relative L2 error of the discrete Laplacian against ``-2 k^2 u``.

    Holds the pressure term out of the way, so it isolates the discretisation of
    viscosity: the decay-rate error of the rung cannot be blamed on the operator
    if this number is much smaller.
    """
    pos, mass = make_lattice(cfg)
    vel = analytic_field(pos, 0.0, cfg)
    acc = viscous_acceleration(pos, vel, mass, cfg)
    exact = -2.0 * cfg.viscosity * cfg.k ** 2 * vel
    return float((acc - exact).square().sum().sqrt() / exact.square().sum().sqrt()
                 * 100.0)


def kinetic_energy(vel: torch.Tensor, mass: torch.Tensor) -> float:
    return float(0.5 * (mass[:, None] * vel.square()).sum())


def fit_decay_rate(times, energies, t_lo: float, t_hi: float):
    """Least-squares slope of ``ln E`` over ``[t_lo, t_hi]`` (returns ``-slope``)."""
    xs = [t for t, e in zip(times, energies) if t_lo <= t <= t_hi and e > 0.0]
    ys = [math.log(e) for t, e in zip(times, energies)
          if t_lo <= t <= t_hi and e > 0.0]
    if len(xs) < 3:
        raise ValueError("decay-rate fit needs at least three samples in the window")
    count = float(len(xs))
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    slope = num / den
    # R^2 of the log-linear fit, reported as a fit-quality diagnostic.
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (mean_y + slope * (x - mean_x))) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return -slope, r2


def run_tgv(cfg: TGVConfig, viscosity_override: float | None = None,
            gravity_free: bool = True) -> dict:
    """Integrate the Taylor-Green vortex and measure its decay rate.

    ``viscosity_override`` exists for the negative control (``0.0`` must leave
    the energy flat); everything else is taken from ``cfg``.
    """
    if viscosity_override is not None:
        cfg = TGVConfig(**{**cfg.__dict__, "viscosity": viscosity_override})
    pos, mass = make_lattice(cfg)
    vel = analytic_field(pos, 0.0, cfg)
    e0 = kinetic_energy(vel, mass)
    dt = cfg.dt()
    steps = int(math.ceil(cfg.t_end / dt))
    dt = cfg.t_end / steps
    h = cfg.h()
    cutoff = 2.0 * h + cfg.skin_factor * h

    pairs = periodic_pairs(pos, cutoff, cfg.L)
    times: list[float] = []
    energies: list[float] = []
    peak_speed: list[float] = []
    t = 0.0
    for step in range(steps + 1):
        if step % cfg.rebuild_every == 0:
            pairs = periodic_pairs(pos, cutoff, cfg.L)
        if step % cfg.sample_every == 0 or step == steps:
            times.append(t)
            energies.append(kinetic_energy(vel, mass))
            peak_speed.append(float(torch.linalg.vector_norm(vel, dim=-1).max()))
        if step == steps:
            break
        rho, _p, acc = tgv_rhs(pos, vel, mass, cfg, pairs=pairs)
        vel = vel + dt * acc
        pos = torch.remainder(pos + dt * vel, cfg.L)
        t += dt

    # Fit window: skip the first 10 % (acoustic settling) and stop at 85 % of
    # the run, where the energy has decayed but has not reached the noise floor
    # (E/E0 ~ 0.15 for the defaults).
    t_lo = 0.10 * cfg.t_end
    t_hi = 0.85 * cfg.t_end
    rate, r2 = fit_decay_rate(times, energies, t_lo, t_hi)
    rate_analytic = cfg.analytic_decay_rate()

    e_end = energies[-1]
    e_start_window = min(e for tt, e in zip(times, energies) if tt >= t_lo)
    t_start_window = min(tt for tt in times if tt >= t_lo)
    rate_from_ratio = (math.log(e_start_window) - math.log(e_end)) / (cfg.t_end - t_start_window)

    amplitude_ratio = peak_speed[-1] / max(peak_speed[0], 1e-300)
    amplitude_analytic = math.exp(-cfg.analytic_amplitude_rate() * cfg.t_end)

    if rate_analytic > 0.0:
        rel_err = abs(rate - rate_analytic) / rate_analytic * 100.0
    else:
        # Negative control (nu = 0): the analytic reference is a flat energy.
        rel_err = float("inf") if abs(rate) > 0.0 else 0.0
    return {
        "case": "taylor_green_vortex_2d_periodic",
        "config": {
            "L": cfg.L, "dx": cfg.dx, "h_over_dx": cfg.h_factor,
            "particles": int(pos.shape[0]), "steps": steps, "dt": dt,
            "t_end": cfg.t_end, "viscosity": cfg.viscosity, "U0": cfg.U0,
            "sound_speed": cfg.sound_speed, "gamma": cfg.gamma,
            "rho0": cfg.rho0, "waves": cfg.waves,
            "reynolds": (cfg.U0 / (cfg.viscosity * cfg.k)
                         if cfg.viscosity > 0.0 else float("inf")),
            "mach": cfg.U0 / cfg.sound_speed, "dtype": cfg.dtype,
        },
        "reference": {
            "kind": "analytic",
            "solution": "u = U0 exp(-2 nu k^2 t) sin(kx) cos(ky), "
                        "v = -U0 exp(-2 nu k^2 t) cos(kx) sin(ky)",
            "energy_decay_rate": rate_analytic,
            "amplitude_decay_rate": cfg.analytic_amplitude_rate(),
            "energy_ratio_analytic": math.exp(-rate_analytic * cfg.t_end),
        },
        "measured": {
            "energy_decay_rate": rate,
            "energy_decay_rate_r2": r2,
            "energy_decay_rate_ratio_estimate": rate_from_ratio,
            "energy_ratio": e_end / e0,
            "energy_ratio_analytic": math.exp(-rate_analytic * cfg.t_end),
            "amplitude_ratio": amplitude_ratio,
            "amplitude_ratio_analytic": amplitude_analytic,
            "fit_window": [t_lo, t_hi],
        },
        "error_pct": rel_err,
        "series": {"time": times, "energy": energies,
                   "energy_analytic": [e0 * math.exp(-rate_analytic * tt) for tt in times]},
        "gate": {"quantity": "energy decay rate vs analytic 4 nu k^2",
                 "limit_pct": 3.0, "pass": bool(rel_err < 3.0)},
    }