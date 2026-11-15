"""CASE-DUCT-POISEUILLE-3D -- pressure-driven flow in a rectangular duct.

Why this case exists
--------------------
The case pool carries plane Poiseuille in 2-D (:mod:`tensorsph.case_poiseuille_morris`,
``benchmarks/case_poiseuille``) and three 3-D anchors: periodic shear-wave decay
(``benchmarks/threed_shear``, no wall), the hydrostatic column
(``benchmarks/threed_hydrostatic``, static wall pressure) and the kernel/operator
sweep (``benchmarks/threed_kernel``).  **No 3-D shear flow with no-slip walls has
ever been compared with an analytic solution.**  A rectangular duct is the
smallest geometry that closes that gap: four no-slip planes, one body force, a
closed-form double-Fourier solution, and a failure mode a 2-D case cannot see --
a boundary condition that acts only on the first pair of walls returns the plane
parabola, whose centre velocity is **1.70x** the square-duct value.

The case
--------
Duct ``0 <= y <= H``, ``0 <= z <= W``, periodic in ``x``, uniform body force
``g0`` along ``x``, no-slip on all four walls::

    nu (u_yy + u_zz) = -g0,        u = 0 on the boundary

    u(y,z) = (16 g0 H^2 W^2 / (nu pi^4)) sum_{m,n odd}
             sin(m pi y/H) sin(n pi z/W) / (m n (m^2 W^2 + n^2 H^2))

Limits used as reference self-checks: ``W/H -> inf`` reproduces the plane
parabola ``g0 (H-y) y / (2 nu)``; ``W = H`` gives the classical square-duct
centre velocity ``u_c = 0.5894 g0 H^2 / (8 nu)``.  An independent finite
difference Poisson solve (:func:`fd_reference`) and a truncation study grade the
reference itself, so a reference bug cannot be laundered as physics error.

Boundary condition
------------------
Mirror (ghost) particles on each of the four planes, rebuilt every step from the
current fluid state with the *negated* velocity (``no_slip``): the antisymmetric
pair has zero mean velocity **on** the plane, so the discrete viscous operator
sees ``u = 0`` on the boundary instead of one spacing inside it -- the mechanism
verified in 2-D by :mod:`tensorsph.wall_mirror`, here exercised for walls normal
to ``y`` *and* ``z`` through its ``axis`` argument.
``slip="free_slip"`` on all four planes is the negative control: no tangential
shear is exchanged, so the fluid accelerates as a rigid body and never reaches
the parabola.

Nothing in the comparison is fitted: ``H``, ``W``, ``nu``, ``g0`` and
``sound_speed`` are inputs, the reference has no free parameter.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch

from .threed.kernel3 import calibrated_particle_mass
from .threed.neighbors3 import box_pairs
from .threed.operators3 import accelerations
from .threed.solver import cubic_lattice
from .wall_mirror import plane_walls

__all__ = [
    "DuctConfig", "duct_series", "fd_reference", "cfg_mean", "plane_limit_table",
    "truncation_table", "build_duct", "run_case", "run_ladder", "evaluate",
    "build_payload", "make_figures", "write_note", "main",
]

#: analytic square-duct centre velocity over the plane-channel value
#: ``g0 H^2 / (8 nu)``: the shape effect a 2-D-only wall treatment cannot show.
SQUARE_TO_PLANE = 0.5894


@dataclass
class DuctConfig:
    """Physical and numerical parameters, fixed before any run."""

    height: float = 0.1          # H [m], y extent
    aspect: float = 1.0          # W / H
    columns: int = 4             # x columns (x periodic, flow invariant)
    h_factor: float = 1.3
    rho0: float = 1000.0
    sound_speed: float = 20.0
    gamma: float = 7.0
    nu: float = 0.01
    g0: float = 9.81
    courant: float = 0.25
    delta: float = 0.1           # density diffusion strength
    slip: str = "no_slip"
    settle_times: float = 3.0    # t_end in units of tau = H^2 / (nu pi^2)
    average_fraction: float = 0.25
    device: str = "cpu"

    @property
    def width(self) -> float:
        return self.aspect * self.height

    @property
    def tau(self) -> float:
        return self.height ** 2 / (self.nu * math.pi ** 2)

    @property
    def t_end(self) -> float:
        return self.settle_times * self.tau

    @property
    def u_plane(self) -> float:
        """Plane-channel centre velocity at the same ``H`` and ``g0``."""
        return self.g0 * self.height ** 2 / (8.0 * self.nu)

    def dx(self, resolution: int) -> float:
        return self.height / float(resolution)

    def h(self, resolution: int) -> float:
        return self.h_factor * self.dx(resolution)

    def counts(self, resolution: int) -> tuple:
        n_z = int(round(self.aspect * resolution))
        if abs(self.aspect * resolution - n_z) > 1e-9:
            raise ValueError("aspect * resolution must be an integer")
        return int(self.columns), int(resolution), n_z

    def dt(self, resolution: int) -> float:
        return self.courant * self.h(resolution) / self.sound_speed

    def steps(self, resolution: int) -> int:
        return int(round(self.t_end / self.dt(resolution)))

    def as_record(self) -> dict:
        d = asdict(self)
        d.update({"width": self.width, "tau": self.tau, "t_end": self.t_end,
                  "u_plane_2d_limit": self.u_plane,
                  "square_to_plane_ratio_reference": SQUARE_TO_PLANE})
        return d


# --------------------------------------------------------------------------- #
# Analytic reference
# --------------------------------------------------------------------------- #

def default_nmax(aspect: float) -> int:
    """Odd-mode cutoff of the double series.

    The ``z`` direction needs ~ ``aspect`` times more modes than ``y`` for the
    same absolute wavenumber, so the cutoff cannot be a constant: at ``W/H = 16``
    a fixed ``nmax = 81`` is 2.4e-4 high at the centre, which is the same order
    as the physics being measured.
    """
    return int(max(81, round(81 * max(1.0, float(aspect)))))


def duct_series(cfg: DuctConfig, y, z, nmax: int | None = None) -> np.ndarray:
    """Closed-form duct velocity at ``(y, z)``; ``nmax`` = largest odd mode."""
    if nmax is None:
        nmax = default_nmax(cfg.aspect)
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    H, W = cfg.height, cfg.width
    modes = np.arange(1, int(nmax) + 1, 2)
    m = modes[:, None, None]
    n = modes[None, :, None]
    coeff = (16.0 * cfg.g0 * H ** 2 * W ** 2 / (cfg.nu * math.pi ** 4)
             / (m * n * (m ** 2 * W ** 2 + n ** 2 * H ** 2)))
    sy = np.sin(m * math.pi * y.reshape(1, 1, -1) / H)
    sz = np.sin(n * math.pi * z.reshape(1, 1, -1) / W)
    return (coeff * sy * sz).sum(axis=(0, 1))


def cfg_mean(cfg: DuctConfig, nmax: int = 81) -> float:
    """Analytic bulk velocity ``Q / (H W)``; the sines integrate in closed form."""
    H, W = cfg.height, cfg.width
    total = 0.0
    for m in range(1, nmax + 1, 2):
        for n in range(1, nmax + 1, 2):
            total += 1.0 / (m ** 2 * n ** 2 * (m ** 2 * W ** 2 + n ** 2 * H ** 2))
    return 64.0 * cfg.g0 * H ** 2 * W ** 2 / (cfg.nu * math.pi ** 6) * total


def fd_reference(cfg: DuctConfig, n_y: int = 96, n_z: int = 96) -> dict:
    """Independent finite-difference Poisson solve (cell-centred, Dirichlet walls).

    The wall lies on the cell face, so the first interior node sees it through
    the image condition ``u_{-1} = -u_0`` (second order).  Sparse direct solve;
    nothing is shared with :func:`duct_series`.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    H, W = cfg.height, cfg.width
    dy, dz = H / n_y, W / n_z

    def idx(j, k):
        return j * n_z + k

    N = n_y * n_z
    rows, cols, vals = [], [], []
    rhs = np.full(N, -cfg.g0 / cfg.nu, dtype=np.float64)
    for j in range(n_y):
        for k in range(n_z):
            i = idx(j, k)
            # u_{-1} = -u_0 at a wall (Dirichlet at the face, second order), so a
            # node next to the wall carries one extra -1/d**2 on its diagonal.
            diag = -2.0 / dy ** 2 - 2.0 / dz ** 2
            if j == 0 or j == n_y - 1:
                diag -= 1.0 / dy ** 2
            if k == 0 or k == n_z - 1:
                diag -= 1.0 / dz ** 2
            rows.append(i); cols.append(i); vals.append(diag)
            if j == 0:
                rows.append(i); cols.append(idx(j + 1, k)); vals.append(1.0 / dy ** 2)
            elif j == n_y - 1:
                rows.append(i); cols.append(idx(j - 1, k)); vals.append(1.0 / dy ** 2)
            else:
                for jj in (j - 1, j + 1):
                    rows.append(i); cols.append(idx(jj, k)); vals.append(1.0 / dy ** 2)
            if k == 0:
                rows.append(i); cols.append(idx(j, k + 1)); vals.append(1.0 / dz ** 2)
            elif k == n_z - 1:
                rows.append(i); cols.append(idx(j, k - 1)); vals.append(1.0 / dz ** 2)
            else:
                for kk in (k - 1, k + 1):
                    rows.append(i); cols.append(idx(j, kk)); vals.append(1.0 / dz ** 2)
    A = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    u = spsolve(A, rhs).reshape(n_y, n_z)
    y = (np.arange(n_y) + 0.5) * dy
    z = (np.arange(n_z) + 0.5) * dz
    Y, Z = np.meshgrid(y, z, indexing="ij")
    ana = duct_series(cfg, Y.ravel(), Z.ravel()).reshape(n_y, n_z)
    ana_c = float(ana[n_y // 2, n_z // 2])
    return {
        "n_y": int(n_y), "n_z": int(n_z), "dy": dy, "dz": dz,
        "u_centre_fd": float(u[n_y // 2, n_z // 2]),
        "u_centre_series": ana_c,
        "centre_error_pct": 100.0 * (float(u[n_y // 2, n_z // 2]) - ana_c) / ana_c,
        "l2_error_pct": 100.0 * float(np.linalg.norm(u - ana) / np.linalg.norm(ana)),
    }


def plane_limit_table(aspects=(1.0, 2.0, 4.0, 8.0)) -> list:
    """Series vs the plane-Poiseuille limit as the duct is widened."""
    rows = []
    for aspect in aspects:
        cfg = DuctConfig(aspect=aspect)
        u_c = float(duct_series(cfg, [cfg.height / 2], [cfg.width / 2],
                                nmax=default_nmax(aspect))[0])
        rows.append({"aspect": float(aspect), "u_centre": u_c,
                     "u_plane_2d": cfg.u_plane, "ratio_to_plane": u_c / cfg.u_plane,
                     "deviation_pct": 100.0 * (u_c - cfg.u_plane) / cfg.u_plane})
    return rows


def truncation_table(cfg: DuctConfig, nmax_list=(21, 41, 81, 161)) -> list:
    """Convergence of the double series at the duct centre."""
    yc, zc = cfg.height / 2, cfg.width / 2
    ref = float(duct_series(cfg, [yc], [zc], nmax=321)[0])
    rows = []
    for nmax in nmax_list:
        val = float(duct_series(cfg, [yc], [zc], nmax=nmax)[0])
        rows.append({"nmax": int(nmax), "u_centre": val,
                     "error_vs_nmax321_pct": 100.0 * (val - ref) / ref})
    return rows


# --------------------------------------------------------------------------- #
# Geometry and integration
# --------------------------------------------------------------------------- #

def build_duct(cfg: DuctConfig, resolution: int, device=None):
    """Fluid lattice at cell centres, ``Nx x Ny x Nz`` at pitch ``dx``."""
    device = device or torch.device(cfg.device)
    dx = cfg.dx(resolution)
    nx, ny, nz = cfg.counts(resolution)
    pos = cubic_lattice((nx, ny, nz), dx, origin=(0.0, 0.5 * dx, 0.5 * dx),
                        dtype=torch.float64).to(device)
    mass = torch.full((pos.shape[0],),
                      calibrated_particle_mass(dx, cfg.h(resolution), cfg.rho0),
                      dtype=torch.float64, device=device)
    return pos, mass, (nx, ny, nz), dx


def _planes(cfg: DuctConfig) -> list:
    """Four no-slip planes as ``(coordinate, side, plane_vel, axis)``."""
    return [(0.0, +1, None, 1), (cfg.height, -1, None, 1),
            (0.0, +1, None, 2), (cfg.width, -1, None, 2)]


def _bin_cells(cfg: DuctConfig, pos: torch.Tensor, u_mean: torch.Tensor,
               counts, dx: float) -> dict:
    """Average ``u_x`` on the ``(y, z)`` lattice cells and attach the reference."""
    nx, ny, nz = counts
    y = pos[:, 1].detach().cpu().numpy()
    z = pos[:, 2].detach().cpu().numpy()
    u = u_mean[:, 0].detach().cpu().numpy()
    iy = np.clip(np.rint((y - 0.5 * dx) / dx).astype(int), 0, ny - 1)
    iz = np.clip(np.rint((z - 0.5 * dx) / dx).astype(int), 0, nz - 1)
    key = iy * nz + iz
    order = np.argsort(key, kind="stable")
    key_s, u_s, y_s, z_s = key[order], u[order], y[order], z[order]
    uniq, starts = np.unique(key_s, return_index=True)
    bounds = np.append(starts, len(key_s))
    rows = []
    for i, cell in enumerate(uniq):
        sl = slice(bounds[i], bounds[i + 1])
        j, k = int(cell // nz), int(cell % nz)
        yc, zc = float(y_s[sl].mean()), float(z_s[sl].mean())
        ana = float(duct_series(cfg, [yc], [zc])[0])
        rows.append({"iy": j, "iz": k, "y": yc, "z": zc,
                     "y_over_H": yc / cfg.height, "z_over_W": zc / cfg.width,
                     "u_mean": float(u_s[sl].mean()), "u_std": float(u_s[sl].std()),
                     "count": int(sl.stop - sl.start), "u_analytic": ana,
                     "error_pct": (100.0 * (float(u_s[sl].mean()) - ana) / ana)
                     if ana > 1e-12 else None})
    centre = min(rows, key=lambda r: (r["y"] - cfg.height / 2) ** 2
                 + (r["z"] - cfg.width / 2) ** 2)
    core = [r for r in rows if 0.1 <= r["y_over_H"] <= 0.9 and 0.1 <= r["z_over_W"] <= 0.9]
    band = [r for r in rows if r not in core]
    def l2(rs):
        num = math.sqrt(sum((r["u_mean"] - r["u_analytic"]) ** 2 for r in rs))
        den = math.sqrt(sum(r["u_analytic"] ** 2 for r in rs)) or 1.0
        return 100.0 * num / den
    return {"cells": len(rows), "core_cells": len(core), "wall_band_cells": len(band),
            "u_centre": centre["u_mean"], "u_centre_analytic": centre["u_analytic"],
            "u_centre_error_pct": 100.0 * (centre["u_mean"] - centre["u_analytic"])
            / centre["u_analytic"],
            "core_l2_error_pct": l2(core), "wall_band_l2_error_pct": l2(band),
            "rows": rows}


def run_case(cfg: DuctConfig, resolution: int, device=None, verbose: bool = False,
             steps: int | None = None) -> dict:
    """Integrate the duct from rest and return the time-averaged velocity field.

    ``x`` is periodic and the flow is ``x``-invariant, so the wrap is completed
    with two images of the cloud; the real copy is the one whose neighbourhood is
    complete (the quadrature used by the 2-D Morris case).
    """
    device = device or torch.device(cfg.device)
    torch.manual_seed(0)
    pos, mass, (nx, ny, nz), dx = build_duct(cfg, resolution, device)
    n_real = int(pos.shape[0])
    vel = torch.zeros_like(pos)
    h = cfg.h(resolution)
    support = 2.0 * h
    length = nx * dx
    planes = _planes(cfg)
    dt = cfg.dt(resolution)
    n_steps = int(steps) if steps is not None else cfg.steps(resolution)
    t_end = n_steps * dt
    avg_from = cfg.settle_times * (1.0 - cfg.average_fraction) * cfg.tau
    shift = torch.tensor([length, 0.0, 0.0], dtype=pos.dtype, device=device)
    body = torch.zeros(3, dtype=pos.dtype, device=device)
    body[0] = cfg.g0

    def _acc(pp, vv):
        pos_ext = torch.cat((pp, pp + shift, pp - shift), dim=0)
        vel_ext = torch.cat((vv, vv, vv), dim=0)
        mass_ext = mass.repeat(3)
        gp, gv, gm = plane_walls(pos_ext, vel_ext, mass_ext, planes, support,
                                 slip=cfg.slip)
        pos_all = torch.cat((pos_ext, gp), dim=0)
        vel_all = torch.cat((vel_ext, gv), dim=0)
        mass_all = torch.cat((mass_ext, gm), dim=0)
        pairs = box_pairs(pos_all, support)
        acc, rho, _ = accelerations(
            pos_all, vel_all, mass_all, h, pairs, rho0=cfg.rho0,
            sound_speed=cfg.sound_speed, gamma=cfg.gamma, viscosity=cfg.nu,
            gravity=None, n_fluid=int(pos_ext.shape[0]), renormalise=True,
            wall_model="adami", viscous_order=2, density_diffusion=cfg.delta,
            dt=dt)
        return acc[:n_real] + body, rho[:n_real], int(gp.shape[0])

    wall_clock = time.time()
    acc, rho, n_ghost = _acc(pos, vel)
    u_sum = torch.zeros_like(vel)
    n_avg = 0
    rho_lo, rho_hi = float("inf"), -float("inf")
    trace = []
    centre_idx = ((0 * ny) + (ny // 2 if ny % 2 else ny // 2 - 1)) * nz + \
                 (nz // 2 if nz % 2 else nz // 2 - 1)
    for step in range(n_steps):
        t = step * dt
        if t >= avg_from:
            u_sum += vel
            n_avg += 1
        vel = vel + dt * acc
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], length)
        acc, rho, n_ghost = _acc(pos, vel)
        rho_lo = min(rho_lo, float(rho.min()))
        rho_hi = max(rho_hi, float(rho.max()))
        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            ux = vel[:, 0]
            trace.append({"step": int(step), "t": float(t),
                          "t_over_tau": float(t / cfg.tau),
                          "u_centre_now": float(vel[centre_idx, 0]),
                          "u_max": float(ux.max()), "u_mean": float(ux.mean()),
                          "mach_max": float(ux.abs().max()) / cfg.sound_speed,
                          "wall_clock_s": time.time() - wall_clock})
            if verbose:
                print(f"    step {step + 1}/{n_steps} t/tau={t / cfg.tau:.2f} "
                      f"u_max={float(ux.max()):.4e}", flush=True)

    u_mean = u_sum / max(n_avg, 1)
    profile = _bin_cells(cfg, pos, u_mean, (nx, ny, nz), dx)
    u_np = u_mean[:, 0].detach().cpu().numpy()
    return {"resolution": int(resolution), "dx": dx, "h": h,
            "h_over_dx": cfg.h_factor, "counts": [int(nx), int(ny), int(nz)],
            "n_fluid": n_real, "n_ghost": int(n_ghost), "slip": cfg.slip,
            "aspect": cfg.aspect, "dt": dt, "steps": n_steps,
            "t_end": float(t_end), "t_end_over_tau": float(t_end / cfg.tau),
            "avg_window": [float(avg_from), float(t_end)],
            "n_averaged_samples": int(n_avg),
            "u_centre_measured": float(profile["u_centre"]),
            "u_bulk_measured": float(u_np.mean()),
            "u_max_particle": float(u_np.max()),
            "density_ratio_range": [rho_lo / cfg.rho0, rho_hi / cfg.rho0],
            "profile": {k: v for k, v in profile.items() if k != "rows"},
            "rows": profile["rows"], "trace": trace,
            "wall_clock_s": time.time() - wall_clock}


def run_ladder(cfg: DuctConfig, resolutions=(8, 12, 16), device=None,
               verbose: bool = False) -> dict:
    runs = []
    for res in resolutions:
        t0 = time.time()
        out = run_case(cfg, res, device=device, verbose=verbose)
        out["wall_clock_s"] = time.time() - t0
        runs.append(out)
        print(f"  aspect={cfg.aspect:g} H/dx={res:3d} u_c={out['u_centre_measured']:.5f} "
              f"(analytic {out['u_centre_analytic']:.5f}) "
              f"coreL2={out['profile']['core_l2_error_pct']:.2f}% "
              f"ghosts={out['n_ghost']} {out['wall_clock_s']:.1f}s", flush=True)
    return {"aspect": cfg.aspect, "slip": cfg.slip, "resolutions": list(resolutions),
            "runs": runs, "config": cfg.as_record()}


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #

def evaluate(ladder: dict, controls: dict, reference: dict, cfg_square: DuctConfig,
             cfg_wide: DuctConfig, limit_pct: float = 5.0) -> dict:
    """Turn the ladder and controls into the predeclared gates D1-D9."""
    gates = []

    def gate(gid, what, measured, threshold, passed, unit="%"):
        gates.append({"gate": gid, "what": what, "measured": measured,
                      "threshold": threshold, "unit": unit, "pass": bool(passed)})

    fd = reference["fd"]
    gate("D1", "series vs independent FD, centre velocity", fd["centre_error_pct"],
         limit_pct / 5.0, abs(fd["centre_error_pct"]) <= limit_pct / 5.0)
    gate("D1b", "series vs independent FD, L2 over the duct", fd["l2_error_pct"],
         limit_pct / 5.0, abs(fd["l2_error_pct"]) <= limit_pct / 5.0)
    worst = max(abs(r["error_vs_nmax321_pct"]) for r in reference["truncation"])
    gate("D2", "series truncation at nmax=81", worst, 0.1, worst <= 0.1)

    for name, cfg in (("square", cfg_square), ("wide", cfg_wide)):
        runs = ladder[name]
        finest = runs["runs"][-1]
        gate(f"D3-{name}", "centre velocity at finest rung",
             finest["u_centre_error_pct"], limit_pct,
             abs(finest["u_centre_error_pct"]) <= limit_pct)
        gate(f"D4-{name}", "interior (y,z core) L2 at finest rung",
             finest["profile"]["core_l2_error_pct"], limit_pct,
             finest["profile"]["core_l2_error_pct"] <= limit_pct)
        bulk = 100.0 * (finest["u_bulk_measured"] - cfg_mean(cfg)) / cfg_mean(cfg)
        gate(f"D5-{name}", "bulk velocity (flow rate) at finest rung", bulk,
             limit_pct, abs(bulk) <= limit_pct)
        errs = [r["profile"]["core_l2_error_pct"] for r in runs["runs"]]
        gate(f"D6-{name}", "interior L2 at every rung <= tol and non-increasing",
             {"rungs": runs["resolutions"], "core_l2_pct": errs}, limit_pct,
             all(e <= limit_pct for e in errs)
             and all(b <= a + 1e-9 for a, b in zip(errs, errs[1:])))

    square_c = ladder["square"]["runs"][-1]["u_centre_measured"]
    ratio = square_c / cfg_square.u_plane
    gate("D7", "square-duct centre / plane-channel centre (shape effect)",
         100.0 * (ratio - SQUARE_TO_PLANE) / SQUARE_TO_PLANE, limit_pct,
         abs(ratio - SQUARE_TO_PLANE) / SQUARE_TO_PLANE * 100.0 <= limit_pct)
    wide_c = ladder["wide"]["runs"][-1]["u_centre_measured"]
    ana_ratio = (float(duct_series(cfg_wide, [cfg_wide.height / 2], [cfg_wide.width / 2])[0])
                 / float(duct_series(cfg_square, [cfg_square.height / 2],
                                     [cfg_square.width / 2])[0]))
    gate("D8", "aspect discrimination: wide/square centre velocity",
         100.0 * ((wide_c / square_c) - ana_ratio) / ana_ratio, limit_pct,
         abs((wide_c / square_c) - ana_ratio) / ana_ratio * 100.0 <= limit_pct)

    free = controls.get("free_slip")
    if free is not None:
        target = 2.0 * float(duct_series(cfg_square, [cfg_square.height / 2],
                                         [cfg_square.width / 2])[0])
        gate("D9", "free-slip control: no steady state (centre > 2x no-slip)",
             free["u_centre_measured"], target,
             free["u_centre_measured"] > target, unit="m/s")

    return {"gates": gates, "all_gates_pass": all(g["pass"] for g in gates),
            "failed": [g["gate"] for g in gates if not g["pass"]],
            "tolerance_pct": limit_pct}


def build_payload(square: dict, wide: dict, reference: dict, controls: dict,
                  cfg_square: DuctConfig, cfg_wide: DuctConfig,
                  tol: float = 5.0) -> dict:
    evaluation = evaluate({"square": square, "wide": wide}, controls, reference,
                          cfg_square, cfg_wide, limit_pct=tol)
    return {"case": "case_duct_poiseuille_3d",
            "anchor": "CASE-POISEUILLE-SPH extension -> CASE-DUCT-POISEUILLE-3D",
            "geometry": {"H": cfg_square.height,
                         "aspects": [cfg_square.aspect, cfg_wide.aspect]},
            "parameters": {"nu": cfg_square.nu, "g0": cfg_square.g0,
                           "rho0": cfg_square.rho0, "sound_speed": cfg_square.sound_speed,
                           "h_factor": cfg_square.h_factor, "delta": cfg_square.delta,
                           "settle_times": cfg_square.settle_times,
                           "average_fraction": cfg_square.average_fraction,
                           "courant": cfg_square.courant,
                           "square_to_plane_ratio_reference": SQUARE_TO_PLANE},
            "reference": reference, "ladder": {"square": square, "wide": wide},
            "controls": controls, "evaluation": evaluation,
            "all_gates_pass": evaluation["all_gates_pass"],
            "generated_by": "tensorsph.case_poiseuille_duct"}


# --------------------------------------------------------------------------- #
# Figures and note
# --------------------------------------------------------------------------- #

def make_figures(payload: dict, outdir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figs = []
    ladder = payload["ladder"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, name in zip(axes, ("square", "wide")):
        runs = ladder[name]
        finest = runs["runs"][-1]
        rows = [r for r in finest["rows"] if abs(r["z_over_W"] - 0.5) <= 0.5 / finest["counts"][2]]
        rows.sort(key=lambda r: r["y"])
        ys = [r["y_over_H"] for r in rows]
        ax.plot([r["u_mean"] for r in rows], ys, "o", ms=4, label="SPH (measured)")
        ax.plot([r["u_analytic"] for r in rows], ys, "-", lw=1.6,
                label="series (analytic)")
        ax.set_xlabel(r"$u_x$ [m/s]")
        ax.set_ylabel("y / H")
        ax.set_title(f"aspect W/H = {runs['aspect']:g}, H/dx = {finest['resolution']}"
                     f"\ncentre err {finest['u_centre_error_pct']:+.2f}%")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Duct Poiseuille: centre-line profile (z = W/2) vs analytic series")
    fig.tight_layout()
    p1 = outdir / "duct_profile.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    figs.append({"name": "duct_profile", "path": str(p1),
                 "caption": "Centre-plane velocity profile, SPH vs the double-Fourier series."})

    runs = ladder["square"]["runs"]
    finest = runs[-1]
    ny, nz = finest["counts"][1], finest["counts"][2]
    grid_m = np.full((ny, nz), np.nan)
    grid_a = np.full((ny, nz), np.nan)
    grid_e = np.full((ny, nz), np.nan)
    for r in finest["rows"]:
        grid_m[r["iy"], r["iz"]] = r["u_mean"]
        grid_a[r["iy"], r["iz"]] = r["u_analytic"]
        grid_e[r["iy"], r["iz"]] = r["error_pct"] if r["error_pct"] is not None else np.nan
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ext = [0, 1, 0, 1]
    for ax, data, title in zip(axes, (grid_m, grid_a, grid_e),
                               ("SPH u_x [m/s]", "series u_x [m/s]", "error [%]")):
        im = ax.imshow(data.T, origin="lower", extent=ext, aspect="auto",
                       cmap="viridis" if title != "error [%]" else "coolwarm")
        ax.set_xlabel("y / H")
        ax.set_ylabel("z / W")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Square duct, H/dx = %d: field and pointwise error" % finest["resolution"])
    fig.tight_layout()
    p2 = outdir / "duct_field_square.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    figs.append({"name": "duct_field_square", "path": str(p2),
                 "caption": "Square-duct field (measured / analytic) and pointwise error map."})

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for name, style in (("square", "o-"), ("wide", "s-")):
        rr = ladder[name]["runs"]
        ax.plot([r["resolution"] for r in rr],
                [r["profile"]["core_l2_error_pct"] for r in rr], style,
                label=f"aspect {ladder[name]['aspect']:g} interior L2")
    ax.axhline(payload["evaluation"]["tolerance_pct"], color="k", ls="--", lw=1,
               label="acceptance 5%")
    ax.set_xlabel("H/dx")
    ax.set_ylabel("interior L2 error [%]")
    ax.set_title("Resolution ladder")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p3 = outdir / "duct_ladder.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    figs.append({"name": "duct_ladder", "path": str(p3),
                 "caption": "Interior L2 error vs H/dx for both aspect ratios against the 5% gate."})
    return figs


def write_note(payload: dict, figures: list, path: Path) -> Path:
    ev = payload["evaluation"]
    L = payload["ladder"]
    lines = ["# CASE-DUCT-POISEUILLE-3D — verification note", "",
             "Generated by `src/tensorsph/case_poiseuille_duct.py` (`--merge`); "
             "every number below is read from "
             "`benchmarks/case_poiseuille_duct/out/result.json`.", "",
             "## 1. Case", "",
             "Rectangular duct `0<=y<=H`, `0<=z<=W`, periodic in x, uniform body "
             "force `g0`, no-slip on all four walls:", "",
             "```", "nu (u_yy + u_zz) = -g0,  u = 0 on the boundary", "```", "",
             "| H [m] | aspects W/H | nu [m2/s] | g0 [m/s2] | rho0 | c_s [m/s] |",
             "|---|---|---|---|---|---|",
             "| {H} | {a} | {nu} | {g} | {rho} | {cs} |".format(
                 H=payload["geometry"]["H"], a=payload["geometry"]["aspects"],
                 nu=payload["parameters"]["nu"], g=payload["parameters"]["g0"],
                 rho=payload["parameters"]["rho0"],
                 cs=payload["parameters"]["sound_speed"]), "",
             "## 2. Reference hygiene", "",
             "| check | value | gate |", "|---|---|---|",
             "| series vs independent FD, centre | {:.4f} % | D1 |".format(
                 payload["reference"]["fd"]["centre_error_pct"]),
             "| series vs independent FD, L2 | {:.4f} % | D1b |".format(
                 payload["reference"]["fd"]["l2_error_pct"]),
             "| truncation nmax=81 vs 321 | {:.4f} % | D2 |".format(
                 max(abs(r["error_vs_nmax321_pct"]) for r in payload["reference"]["truncation"])),
             "", "## 3. Ladder", "",
             "| aspect | H/dx | u_centre [m/s] | analytic | err % | interior L2 % | bulk err % | s |",
             "|---|---|---|---|---|---|---|---|"]
    for name in ("square", "wide"):
        cfg_rows = L[name]
        for r in cfg_rows["runs"]:
            lines.append("| {a:g} | {n} | {u:.5f} | {ua:.5f} | {e:+.2f} | {l2:.2f} | {b:+.2f} | {w:.0f} |".format(
                a=cfg_rows["aspect"], n=r["resolution"], u=r["u_centre_measured"],
                ua=r["u_centre_analytic"], e=r["u_centre_error_pct"],
                l2=r["profile"]["core_l2_error_pct"],
                b=100.0 * (r["u_bulk_measured"] - cfg_mean(DuctConfig(aspect=r["aspect"])))
                / cfg_mean(DuctConfig(aspect=r["aspect"])),
                w=r["wall_clock_s"]))
    lines += ["", "## 4. Gates", "", "| gate | what | measured | threshold | pass |",
              "|---|---|---|---|---|"]
    for g in ev["gates"]:
        lines.append("| {} | {} | {} | {} | {} |".format(
            g["gate"], g["what"], g["measured"], g["threshold"], "PASS" if g["pass"] else "FAIL"))
    lines += ["", "**All gates pass: {}**".format(ev["all_gates_pass"]),
              "", "## 5. Figures", ""]
    for f in figures:
        lines.append("- `{}` — {}".format(f["name"], f["caption"]))
    lines += ["", "## 6. Limits", "",
              "- The duct is periodic in `x` with 4 columns; streamwise development "
              "(entrance length) is out of scope by construction.",
              "- `W/H=4` is the widest aspect run; the plane-Poiseuille limit is "
              "checked analytically (`plane_limit_table`), not simulated.",
              "- WCSPH numerical dissipation is not removed; it appears as the "
              "residual error in the tables above and is not extrapolated away.", ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _merge(outdir: Path, repo: Path, resolutions, tol: float = 5.0) -> int:
    def load(tag):
        return json.loads((outdir / f"{tag}.json").read_text())

    cfg_s = DuctConfig(aspect=1.0)
    cfg_w = DuctConfig(aspect=4.0)
    square = load("square")
    wide = load("wide")
    reference = {"fd": fd_reference(cfg_s, 96, 96),
                 "fd_wide": fd_reference(cfg_w, 96, 96),
                 "truncation": truncation_table(cfg_s),
                 "plane_limit": plane_limit_table(),
                 "provenance": {
                     "series": "separation of variables, odd/odd double Fourier "
                               "series; White, Viscous Fluid Flow 3rd ed. §3-3; "
                               "Bruus, Theoretical Microfluidics §3.3",
                     "fd": "cell-centred 5-point Poisson, sparse direct solve, "
                           "independent of the series",
                     "square_to_plane_ratio_reference": SQUARE_TO_PLANE}}
    controls = {}
    cpath = outdir / "control.json"
    if cpath.exists():
        controls["free_slip"] = load("control")
    payload = build_payload(square, wide, reference, controls, cfg_s, cfg_w, tol=tol)
    (outdir / "result.json").write_text(json.dumps(payload, indent=1, sort_keys=True,
                                                  default=float))
    figs = make_figures(payload, outdir)
    note = write_note(payload, figs, repo / "docs" / "verification_case_poiseuille_duct.md")
    print(f"result.json written; note: {note}; gates pass = {payload['all_gates_pass']}")
    print("failed:", payload["evaluation"]["failed"])
    return 0 if payload["all_gates_pass"] else 1


def main(argv=None) -> int:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="CASE-DUCT-POISEUILLE-3D")
    ap.add_argument("--out", default=str(repo / "benchmarks" / "case_poiseuille_duct" / "out"))
    ap.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--only", default=None, help="tag: square-N | wide-N | control-N")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="override step count (probe)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        return _merge(outdir, repo, args.resolutions)
    if args.only is None:
        for tag in [f"square-{r}" for r in args.resolutions] + \
                   [f"wide-{r}" for r in args.resolutions] + \
                   [f"control-{args.resolutions[-1]}"]:
            rc = main(["--only", tag, "--out", str(outdir), "--device", args.device])
            if rc:
                return rc
        return _merge(outdir, repo, args.resolutions)
    kind, res = args.only.rsplit("-", 1)
    aspect = 1.0 if kind in ("square", "control") else 4.0
    cfg = DuctConfig(aspect=aspect, slip="free_slip" if kind == "control" else "no_slip",
                     device=args.device)
    t0 = time.time()
    out = run_case(cfg, int(res), device=args.device, verbose=True, steps=args.steps)
    out["wall_clock_s"] = time.time() - t0
    out["tag"] = args.only
    (outdir / f"{kind}.json").write_text(json.dumps(out, indent=1, sort_keys=True,
                                                    default=float))
    print(f"{args.only}: u_centre={out['u_centre_measured']:.5f} "
          f"(analytic {out['u_centre_analytic']:.5f}) "
          f"coreL2={out['profile']['core_l2_error_pct']:.2f}% {out['wall_clock_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())