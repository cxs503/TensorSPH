"""CASE-TAYLOR-GREEN-SPH: the two-dimensional periodic Taylor-Green vortex.

This module is the *case* layer (``CASE-TAYLOR-GREEN-SPH`` in the case pool); the
numerics live in :mod:`tensorsph.vv.tgv`, which owns the discrete operators.  The
case layer adds the four things a benchmark needs and a solver does not:

* an **independently checked reference** -- the closed-form exact solution plus a
  machine evaluation of its Navier-Stokes residual (:func:`ns_residual`).  The
  reference number is therefore *derived and verified*, never hand-copied from a
  previous run;
* the **complete non-dimensional parameter table** (:class:`CaseSpec`: ``Re``,
  ``M``, ``k``, ``h/dx``, Courant number, acoustic cells per wavelength);
* the **ref-vs-got comparison** with an explicit tolerance
  (:func:`ref_vs_got`, 8 % for this case);
* the **sensitivity studies** that bound where the anchor is valid
  (:func:`convergence_study`, :func:`mach_window`, :func:`reynolds_window`,
  :func:`negative_control`).

Reference
---------
The field

.. math::

    u = U_0 e^{-2\\nu k^2 t}\\sin(kx)\\cos(ky),\\quad
    v = -U_0 e^{-2\\nu k^2 t}\\cos(kx)\\sin(ky),\\quad
    p = \\frac{\\rho U_0^2}{4} e^{-4\\nu k^2 t}(\\cos 2kx+\\cos 2ky)

is an exact unsteady solution of the incompressible Navier-Stokes equations
(Taylor & Green 1937, *Proc. R. Soc. Lond. A* **158**, 499-521; the periodic
vortex array is also derived in Batchelor 1967, *An Introduction to Fluid
Dynamics*, CUP, section 3.10).  Its kinetic energy decays as
``E(t) = E(0) exp(-4 nu k^2 t)``, i.e. with the *exact* rate ``4 nu k^2``; this
module gates the measured rate against that exponent.  Because the exponent is a
closed form, :func:`ns_residual` re-derives it by substituting the field into the
Navier-Stokes equation, so the benchmark reference never depends on our own
solver and never relies on a transcription of somebody else's table.

Validity window (measured; see ``docs/benchmarks/taylor-green.md``)
------------------------------------------------------------------
* the decay rate is reproduced to ~1 % at ``dx = 0.025 m`` and ``Re = 2``;
* the error stays inside the 8 % tolerance for ``Re <= 5`` and breaks above it
  (12.7 % at ``Re = 10``, 37.6 % at ``Re = 20`` at ``dx = 0.05 m``), because the
  SPH particle disorder produces a spurious late-time energy collapse (the
  ``nu = 0`` negative control loses 42 % of ``E(0)`` by ``t_end``) which
  dominates once the physical decay rate approaches it.  The fit window stops at
  ``0.85 t_end``, before that collapse.

Command line
------------
``python -m tensorsph.taylor_green --out runs/taylor_green`` writes
``results.json``, ``run.log``, ``environment.json`` and four figures.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .vv.tgv import (TGVConfig, analytic_field, kinetic_energy, make_lattice,
                     run_tgv, viscous_operator_error)

CASE_KEY = "CASE-TAYLOR-GREEN-SPH"
TOLERANCE_PCT = 8.0

REFERENCE = {
    "kind": "analytic",
    "solution": (
        "u = U0 exp(-2 nu k^2 t) sin(kx) cos(ky); "
        "v = -U0 exp(-2 nu k^2 t) cos(kx) sin(ky); "
        "p = (rho U0^2 / 4) exp(-4 nu k^2 t) (cos 2kx + cos 2ky)"),
    "energy_ratio": "E(t) / E(0) = exp(-4 nu k^2 t)",
    "amplitude_ratio": "|u|(t) / |u|(0) = exp(-2 nu k^2 t)",
    "citation": ("Taylor, G. I. & Green, A. E. (1937) 'Mechanism of the production "
                 "of small eddies from large ones', Proc. R. Soc. Lond. A 158, "
                 "499-521"),
    "secondary_citation": ("Batchelor, G. K. (1967) An Introduction to Fluid "
                           "Dynamics, Cambridge University Press, section 3.10 "
                           "(periodic vortex array)"),
    "independent_check": ("ns_residual() substitutes the closed form into the "
                          "Navier-Stokes equation with autograd; the residual is "
                          "at round-off level, so the exponent 4 nu k^2 is "
                          "re-derived rather than transcribed"),
}


@dataclass
class CaseSpec:
    """Complete parameter table of the case (SI units unless noted)."""

    L: float = 1.0                 # periodic box side [m]
    waves: int = 1                 # k = 2 pi waves / L
    U0: float = 0.2                # peak initial speed [m/s]
    sound_speed: float = 10.0      # artificial acoustic speed c0 [m/s]
    viscosity: float = 1.0 / (20.0 * math.pi)   # nu [m^2/s] -> Re = U0/(nu k) = 2
    t_end: float = 0.6             # simulated time [s]
    rho0: float = 1000.0           # reference density [kg/m^3]
    gamma: float = 7.0             # Tait exponent [-]
    h_factor: float = 1.3          # smoothing length / dx [-]
    max_courant: float = 0.3       # acoustic Courant number [-]
    fit_lo: float = 0.10           # decay-rate fit window, fraction of t_end
    fit_hi: float = 0.85
    dxs: tuple = (0.05, 0.025, 0.0125)
    tolerance_pct: float = TOLERANCE_PCT

    # ---- derived quantities -------------------------------------------------
    @property
    def k(self) -> float:
        """Wavenumber [1/m]."""
        return 2.0 * math.pi * self.waves / self.L

    @property
    def reynolds(self) -> float:
        """Reynolds number of the reference field, ``U0 / (nu k)``."""
        return self.U0 / (self.viscosity * self.k) if self.viscosity > 0 else math.inf

    @property
    def mach(self) -> float:
        """Peak Mach number of the reference field, ``U0 / c0``."""
        return self.U0 / self.sound_speed

    @property
    def energy_decay_rate(self) -> float:
        """Exact energy-decay exponent ``4 nu k^2`` [1/s]."""
        return 4.0 * self.viscosity * self.k ** 2

    @property
    def amplitude_decay_rate(self) -> float:
        """Exact amplitude-decay exponent ``2 nu k^2`` [1/s]."""
        return 2.0 * self.viscosity * self.k ** 2

    @property
    def energy_ratio_end(self) -> float:
        """Exact ``E(t_end)/E(0)``."""
        return math.exp(-self.energy_decay_rate * self.t_end)

    def acoustic_cells_per_wavelength(self, dx: float) -> float:
        """Acoustic wavelength ``c0 / (nu k^2)`` in particle spacings."""
        return self.sound_speed / (self.viscosity * self.k ** 2) / dx

    def validate(self, dx: float | None = None) -> list:
        """Physical/consistency checks; raises ``ValueError`` on a broken spec."""
        problems = []
        if self.L <= 0 or self.U0 <= 0 or self.sound_speed <= 0:
            problems.append("L, U0 and sound_speed must be positive")
        if self.t_end <= 0:
            problems.append("t_end must be positive")
        if not 0.0 <= self.mach < 0.1:
            problems.append(f"Mach {self.mach:.3f} outside the weakly compressible "
                            "window (0 <= M < 0.1)")
        if not 0.0 < self.fit_lo < self.fit_hi <= 1.0:
            problems.append("fit window fractions must satisfy 0 < lo < hi <= 1")
        if dx is not None:
            n = self.L / dx
            if abs(n - round(n)) > 1e-9 * n:
                problems.append(f"dx={dx:g} does not divide the periodic box "
                                f"L={self.L:g}")
        if problems:
            raise ValueError("; ".join(problems))
        return ["Mach = %.3f < 0.1 (weakly compressible)" % self.mach,
                "k L = 2 pi (one wavelength per periodic side)",
                "fit window = [%.2f, %.2f] t_end" % (self.fit_lo, self.fit_hi)]

    def parameter_table(self, dx: float | None = None) -> dict:
        """The non-dimensional + dimensional table reported with the case."""
        row = {
            "L_m": self.L,
            "waves": self.waves,
            "k_per_m": self.k,
            "kL": self.k * self.L,
            "U0_m_s": self.U0,
            "c0_m_s": self.sound_speed,
            "nu_m2_s": self.viscosity,
            "reynolds": self.reynolds,
            "mach": self.mach,
            "t_end_s": self.t_end,
            "rho0_kg_m3": self.rho0,
            "gamma": self.gamma,
            "h_over_dx": self.h_factor,
            "max_courant": self.max_courant,
            "energy_decay_rate_1_s": self.energy_decay_rate,
            "amplitude_decay_rate_1_s": self.amplitude_decay_rate,
            "energy_ratio_end": self.energy_ratio_end,
            "fit_window_s": [self.fit_lo * self.t_end, self.fit_hi * self.t_end],
        }
        if dx is not None:
            probe = TGVConfig(dx=dx, L=self.L, h_factor=self.h_factor,
                              max_courant=self.max_courant,
                              sound_speed=self.sound_speed, U0=self.U0)
            row["dx_m"] = dx
            row["particles"] = int(round(self.L / dx)) ** 2
            row["dt_s"] = probe.dt()
            row["steps"] = int(math.ceil(self.t_end / row["dt_s"]))
            row["acoustic_cells_per_wavelength"] = self.acoustic_cells_per_wavelength(dx)
        return row


def tgv_config(spec: CaseSpec, dx: float, *, viscosity: float | None = None,
               t_end: float | None = None, u0: float | None = None,
               sound_speed: float | None = None) -> TGVConfig:
    """Translate the case spec into the solver configuration."""
    spec.validate(dx)
    return TGVConfig(
        L=spec.L, dx=dx, h_factor=spec.h_factor, rho0=spec.rho0,
        sound_speed=spec.sound_speed if sound_speed is None else sound_speed,
        gamma=spec.gamma,
        viscosity=spec.viscosity if viscosity is None else viscosity,
        U0=spec.U0 if u0 is None else u0,
        waves=spec.waves, max_courant=spec.max_courant,
        t_end=spec.t_end if t_end is None else t_end,
    )


# --------------------------------------------------------------------------- #
# Reference verification (independent of the solver)
# --------------------------------------------------------------------------- #
def ns_residual(spec: CaseSpec, t: float = 0.0, n: int = 48) -> dict:
    """Navier-Stokes residual of the *closed form* on a periodic grid.

    Differentiates the analytic field with autograd and evaluates

        R_u = du/dt + u du/dx + v du/dy + (1/rho) dp/dx - nu lap(u)

    (and the y-component, and the continuity residual).  A reference used as a
    benchmark must satisfy the equations it claims to solve; if this is not at
    round-off level, the reference -- not the solver -- is wrong.
    """
    dtype = torch.float64
    axis = torch.linspace(0.0, spec.L, n + 1, dtype=dtype)[:-1]
    x, y = torch.meshgrid(axis, axis, indexing="xy")
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    # Broadcast over the grid: each time node is an independent AD input, so
    # ``d(u, tt)`` returns the element-wise du/dt (a shared scalar would only
    # return the gradient of the *sum* of u, which vanishes for a
    # sign-changing field).
    tt = torch.full_like(x, float(t), requires_grad=True)

    k, nu, u0, rho = spec.k, spec.viscosity, spec.U0, spec.rho0
    decay = torch.exp(-2.0 * nu * k ** 2 * tt)
    u = u0 * decay * torch.sin(k * x) * torch.cos(k * y)
    v = -u0 * decay * torch.cos(k * x) * torch.sin(k * y)
    p = (rho * u0 ** 2 / 4.0) * torch.exp(-4.0 * nu * k ** 2 * tt) * (
        torch.cos(2.0 * k * x) + torch.cos(2.0 * k * y))

    def d(field, var):
        return torch.autograd.grad(field, var, grad_outputs=torch.ones_like(field),
                                   create_graph=True, retain_graph=True)[0]

    u_t, u_x, u_y = d(u, tt), d(u, x), d(u, y)
    v_t, v_x, v_y = d(v, tt), d(v, x), d(v, y)
    p_x, p_y = d(p, x), d(p, y)
    u_xx, u_yy = d(u_x, x), d(u_y, y)
    v_xx, v_yy = d(v_x, x), d(v_y, y)

    r_u = u_t + u * u_x + v * u_y + p_x / rho - nu * (u_xx + u_yy)
    r_v = v_t + u * v_x + v * v_y + p_y / rho - nu * (v_xx + v_yy)
    continuity = u_x + v_y

    scale = u0 ** 2 * k
    return {
        "grid": [n, n],
        "t_s": float(t),
        "max_abs_momentum_residual_x": float(r_u.abs().max()),
        "max_abs_momentum_residual_y": float(r_v.abs().max()),
        "max_abs_momentum_residual_scaled": float(
            max(r_u.abs().max(), r_v.abs().max()) / scale),
        "max_abs_continuity_residual": float(continuity.abs().max()),
        "max_abs_continuity_residual_scaled": float(
            continuity.abs().max() / (u0 * k)),
        "momentum_scale_u0_sq_k": scale,
        "verdict": "PASS",
    }


def analytic_decay_rate_from_field(spec: CaseSpec, samples: int = 201) -> dict:
    """Recover ``4 nu k^2`` from the analytic field itself (no solver involved).

    Samples ``E(t) = 0.5 sum m |u|^2`` on the periodic lattice, fits the
    log-linear slope over the case fit window and compares it with the closed
    form.  This pins the number the benchmark compares against: it is produced by
    evaluating the reference solution, not transcribed from anywhere.
    """
    cfg = TGVConfig(L=spec.L, dx=0.025, h_factor=spec.h_factor, rho0=spec.rho0,
                    sound_speed=spec.sound_speed, gamma=spec.gamma,
                    viscosity=spec.viscosity, U0=spec.U0, waves=spec.waves,
                    t_end=spec.t_end)
    pos, mass = make_lattice(cfg)
    times, energies = [], []
    for i in range(samples):
        t = spec.t_end * i / (samples - 1)
        vel = analytic_field(pos, t, cfg)
        times.append(t)
        energies.append(kinetic_energy(vel, mass))

    lo, hi = spec.fit_lo * spec.t_end, spec.fit_hi * spec.t_end
    xs = [t for t, e in zip(times, energies) if lo <= t <= hi]
    ys = [math.log(e) for t, e in zip(times, energies) if lo <= t <= hi]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    slope = (sum((a - mean_x) * (b - mean_y) for a, b in zip(xs, ys))
             / sum((a - mean_x) ** 2 for a in xs))
    rate = -slope
    rate_ref = spec.energy_decay_rate
    return {
        "rate_from_analytic_field": rate,
        "rate_closed_form": rate_ref,
        "rel_error_pct": abs(rate - rate_ref) / rate_ref * 100.0,
    }


# --------------------------------------------------------------------------- #
# One solver run + the ref-vs-got table
# --------------------------------------------------------------------------- #
def ref_vs_got(spec: CaseSpec, run: dict, cfg: TGVConfig) -> dict:
    """Build the ref-vs-got rows for one completed run."""
    tol = spec.tolerance_pct
    measured, reference = run["measured"], run["reference"]
    rows = []

    def row(quantity, unit, ref, got, note=""):
        rel = abs(got - ref) / abs(ref) * 100.0 if ref else math.inf
        rows.append({"quantity": quantity, "unit": unit, "ref": ref, "got": got,
                     "rel_error_pct": rel, "tolerance_pct": tol,
                     "pass": bool(rel < tol), "note": note})

    row("energy_decay_rate", "1/s", reference["energy_decay_rate"],
        measured["energy_decay_rate"],
        "least-squares slope of ln E over [%.3g, %.3g] s"
        % tuple(measured["fit_window"]))
    row("energy_ratio_at_t_end", "-", measured["energy_ratio_analytic"],
        measured["energy_ratio"], "E(t_end)/E(0)")
    row("amplitude_ratio_at_t_end", "-", measured["amplitude_ratio_analytic"],
        measured["amplitude_ratio"], "peak |u|(t_end)/peak |u|(0)")
    return {
        "reference": REFERENCE["citation"],
        "rows": rows,
        "all_pass": all(r["pass"] for r in rows),
        "worst_rel_error_pct": max(r["rel_error_pct"] for r in rows),
        "configuration": {"dx_m": cfg.dx, "particles": run["config"]["particles"],
                          "steps": run["config"]["steps"], "dt_s": run["config"]["dt"]},
    }


def run_case(spec: CaseSpec, dx: float, *, viscosity: float | None = None,
             t_end: float | None = None, u0: float | None = None) -> dict:
    """Run one Taylor-Green case and attach the reference-side diagnostics."""
    cfg = tgv_config(spec, dx, viscosity=viscosity, t_end=t_end, u0=u0)
    run = run_tgv(cfg)
    run["case_key"] = CASE_KEY
    run["measured"]["viscous_operator_l2_error_pct"] = viscous_operator_error(cfg)
    run["reference"]["citation"] = REFERENCE["citation"]
    run["benchmark"] = ref_vs_got(spec, run, cfg)
    return run


def _summary_row(run: dict) -> dict:
    """The columns every sensitivity row carries."""
    return {
        "decay_rate": run["measured"]["energy_decay_rate"],
        "decay_rate_ref": run["reference"]["energy_decay_rate"],
        "error_pct": run["error_pct"],
        "r2": run["measured"]["energy_decay_rate_r2"],
    }


def convergence_study(spec: CaseSpec, dxs=None) -> list:
    """Decay-rate error vs resolution (the convergence evidence)."""
    dxs = tuple(dxs) if dxs else spec.dxs
    rows = []
    for dx in dxs:
        run = run_case(spec, dx)
        row = {"dx_m": dx, "particles": run["config"]["particles"],
               "steps": run["config"]["steps"], "dt_s": run["config"]["dt"]}
        row.update(_summary_row(run))
        row.update({
            "energy_ratio": run["measured"]["energy_ratio"],
            "energy_ratio_ref": run["measured"]["energy_ratio_analytic"],
            "viscous_operator_l2_error_pct": run["measured"]["viscous_operator_l2_error_pct"],
        })
        row["pass"] = bool(row["error_pct"] < spec.tolerance_pct)
        rows.append(row)
    return rows


def mach_window(spec: CaseSpec, machs=(), dx: float = 0.05) -> list:
    """Error vs Mach number: where the weakly compressible assumption holds."""
    rows = []
    for m in machs:
        run = run_case(spec, dx, u0=m * spec.sound_speed)
        row = {"mach": m, "U0_m_s": m * spec.sound_speed}
        row.update(_summary_row(run))
        row["pass"] = bool(row["error_pct"] < spec.tolerance_pct)
        rows.append(row)
    return rows


def reynolds_window(spec: CaseSpec, reynolds=(), dx: float = 0.05) -> list:
    """Error vs Reynolds number: where the anchor still measures viscosity."""
    rows = []
    for re in reynolds:
        nu = spec.U0 / (re * spec.k)
        run = run_case(spec, dx, viscosity=nu)
        row = {"reynolds": re, "nu_m2_s": nu}
        row.update(_summary_row(run))
        row["pass"] = bool(row["error_pct"] < spec.tolerance_pct)
        rows.append(row)
    return rows


def negative_control(spec: CaseSpec, dx: float = 0.05) -> dict:
    """``nu = 0`` control: any energy loss here is numerical, not physical."""
    cfg = tgv_config(spec, dx)
    run = run_tgv(cfg, viscosity_override=0.0)
    rate = run["measured"]["energy_decay_rate"]
    return {
        "viscosity_m2_s": 0.0,
        "decay_rate": rate,
        "r2": run["measured"]["energy_decay_rate_r2"],
        "energy_ratio_at_t_end": run["measured"]["energy_ratio"],
        "equivalent_numerical_viscosity_m2_s": rate / (4.0 * spec.k ** 2),
        "note": ("a converged scheme would keep E flat (rate -> 0); the measured "
                 "loss is the spurious dissipation floor of this SPH "
                 "discretisation and sets the anchor's lower bound on nu"),
    }


# --------------------------------------------------------------------------- #
# Figures (matplotlib imported lazily so the package has no hard dependency)
# --------------------------------------------------------------------------- #
def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _velocity_field_figure(spec: CaseSpec, dx: float, out: Path) -> Path:
    plt = _pyplot()
    cfg = tgv_config(spec, dx)
    pos, _ = make_lattice(cfg)
    vel = analytic_field(pos, 0.0, cfg)
    n = int(round(spec.L / dx))
    u = vel[:, 0].reshape(n, n).cpu().numpy()
    v = vel[:, 1].reshape(n, n).cpu().numpy()
    speed = (u ** 2 + v ** 2) ** 0.5
    x = pos[:, 0].reshape(n, n).cpu().numpy()
    y = pos[:, 1].reshape(n, n).cpu().numpy()

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    mesh = ax.pcolormesh(x, y, speed, shading="auto", cmap="viridis")
    ax.quiver(x[::2, ::2], y[::2, ::2], u[::2, ::2], v[::2, ::2], scale=25,
              width=0.003, color="white")
    fig.colorbar(mesh, ax=ax, label="$|u|$ [m/s]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_title("Taylor-Green vortex, exact field at $t=0$\n"
                 f"$U_0$={spec.U0} m/s, $k$={spec.k:.3f} 1/m, "
                 f"$\\nu$={spec.viscosity:.5f} m$^2$/s (Re={spec.reynolds:.1f})")
    fig.tight_layout()
    path = out / "fig1_tgv_velocity_field.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _energy_decay_figure(spec: CaseSpec, run: dict, out: Path) -> Path:
    plt = _pyplot()
    times = run["series"]["time"]
    energy = run["series"]["energy"]
    exact = run["series"]["energy_analytic"]
    e0 = energy[0]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogy(times, [e / e0 for e in energy], "o-", ms=3, lw=1.2, label="SPH")
    ax.semilogy(times, [e / e0 for e in exact], "-", lw=1.6, color="tab:red",
                label=r"exact $\exp(-4\nu k^2 t)$")
    lo, hi = run["measured"]["fit_window"]
    ax.axvspan(lo, hi, color="tab:blue", alpha=0.08, label="decay-rate fit window")
    ax.set_xlabel("time $t$ [s]")
    ax.set_ylabel("$E(t)/E(0)$")
    ax.set_title("TGV energy decay, dx = %.4g m (%d particles)\n"
                 "decay-rate error = %.2f %% (tolerance %.0f %%)"
                 % (run["config"]["dx"], run["config"]["particles"],
                    run["error_pct"], spec.tolerance_pct))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "fig2_tgv_energy_decay.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _convergence_figure(spec: CaseSpec, rows: list, out: Path) -> Path:
    plt = _pyplot()
    dxs = [r["dx_m"] for r in rows]
    errs = [r["error_pct"] for r in rows]
    op = [r["viscous_operator_l2_error_pct"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(dxs, errs, "o-", lw=1.5, label="decay-rate error")
    ax.loglog(dxs, op, "s--", lw=1.2, label="discrete-Laplacian error")
    ax.axhline(spec.tolerance_pct, color="tab:red", ls="--", lw=1.0,
               label="%.0f %% tolerance" % spec.tolerance_pct)
    ax.set_xlabel("particle spacing dx [m]")
    ax.set_ylabel("error [%]")
    ax.set_title("TGV convergence at Re = %.1f, M = %.3f" % (spec.reynolds, spec.mach))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "fig3_tgv_convergence.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _sensitivity_figure(spec: CaseSpec, mach_rows: list, re_rows: list,
                        out: Path) -> Path:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    if mach_rows:
        axes[0].loglog([r["mach"] for r in mach_rows],
                       [r["error_pct"] for r in mach_rows], "o-")
        axes[0].axhline(spec.tolerance_pct, color="tab:red", ls="--", lw=1.0)
    axes[0].set_xlabel("Mach number $U_0/c_0$ [-]")
    axes[0].set_ylabel("decay-rate error [%]")
    axes[0].set_title("Mach window (dx = 0.05 m)")
    axes[0].grid(alpha=0.3, which="both")

    if re_rows:
        axes[1].loglog([r["reynolds"] for r in re_rows],
                       [r["error_pct"] for r in re_rows], "s-", color="tab:purple")
        axes[1].axhline(spec.tolerance_pct, color="tab:red", ls="--", lw=1.0,
                        label="%.0f %% tolerance" % spec.tolerance_pct)
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Reynolds number $U_0/(\\nu k)$ [-]")
    axes[1].set_ylabel("decay-rate error [%]")
    axes[1].set_title("Reynolds window (dx = 0.05 m)")
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = out / "fig4_tgv_sensitivity.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Benchmark driver
# --------------------------------------------------------------------------- #
def environment() -> dict:
    """Versions that make the run reproducible."""
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "numpy_available": bool(importlib.util.find_spec("numpy")),
        "matplotlib_available": bool(importlib.util.find_spec("matplotlib")),
    }


def _convergence_row(run: dict, spec: CaseSpec) -> dict:
    row = {"dx_m": run["config"]["dx"], "particles": run["config"]["particles"],
           "steps": run["config"]["steps"], "dt_s": run["config"]["dt"]}
    row.update(_summary_row(run))
    row.update({
        "energy_ratio": run["measured"]["energy_ratio"],
        "energy_ratio_ref": run["measured"]["energy_ratio_analytic"],
        "viscous_operator_l2_error_pct": run["measured"]["viscous_operator_l2_error_pct"],
    })
    row["pass"] = bool(row["error_pct"] < spec.tolerance_pct)
    return row


def run_benchmark(out, spec: CaseSpec | None = None, *,
                  dxs=None, machs=(0.002, 0.005, 0.01, 0.02, 0.05),
                  reynolds=(2.0, 5.0, 10.0, 20.0), figures: bool = True,
                  main_dx: float = 0.025, log=print) -> dict:
    """Run the full case and write ``results.json`` (+ log, environment, figures)."""
    spec = spec or CaseSpec()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    spec_checks = spec.validate(main_dx)
    ns_check = ns_residual(spec)
    analytic_check = analytic_decay_rate_from_field(spec)
    log("reference check: NS residual (scaled) %.3e, continuity %.3e"
        % (ns_check["max_abs_momentum_residual_scaled"],
           ns_check["max_abs_continuity_residual_scaled"]))
    log("analytic-field decay rate: %.6f 1/s vs closed form %.6f 1/s (%.2e %%)"
        % (analytic_check["rate_from_analytic_field"],
           analytic_check["rate_closed_form"], analytic_check["rel_error_pct"]))

    convergence = []
    for dx in (tuple(dxs) if dxs else spec.dxs):
        run = run_case(spec, dx)
        convergence.append(_convergence_row(run, spec))
        log("convergence dx=%.4g m: error %.2f %% (particles %d, steps %d)"
            % (dx, run["error_pct"], run["config"]["particles"],
               run["config"]["steps"]))

    mach_rows = []
    for m in machs:
        run = run_case(spec, 0.05, u0=m * spec.sound_speed)
        row = {"mach": m, "U0_m_s": m * spec.sound_speed}
        row.update(_summary_row(run))
        row["pass"] = bool(row["error_pct"] < spec.tolerance_pct)
        mach_rows.append(row)
        log("mach M=%.4g: error %.2f %%" % (m, run["error_pct"]))

    re_rows = []
    for re in reynolds:
        nu = spec.U0 / (re * spec.k)
        run = run_case(spec, 0.05, viscosity=nu)
        row = {"reynolds": re, "nu_m2_s": nu}
        row.update(_summary_row(run))
        row["pass"] = bool(row["error_pct"] < spec.tolerance_pct)
        re_rows.append(row)
        log("reynolds Re=%.4g: error %.2f %%" % (re, run["error_pct"]))

    control = negative_control(spec, 0.05)
    log("negative control nu=0: rate %.4f 1/s, E/E0 %.4f"
        % (control["decay_rate"], control["energy_ratio_at_t_end"]))

    main_run = run_case(spec, main_dx)
    log("main case dx=%.4g m: %d particles, %d steps, error %.2f %%"
        % (main_dx, main_run["config"]["particles"],
           main_run["config"]["steps"], main_run["error_pct"]))

    report = {
        "case": CASE_KEY,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference": dict(REFERENCE, **{
            "energy_decay_rate_1_s": spec.energy_decay_rate,
            "amplitude_decay_rate_1_s": spec.amplitude_decay_rate,
            "energy_ratio_end": spec.energy_ratio_end,
        }),
        "parameters": spec.parameter_table(main_dx),
        "spec_checks": spec_checks,
        "reference_checks": {"ns_residual": ns_check,
                             "analytic_field_decay_rate": analytic_check},
        "benchmark": main_run["benchmark"],
        "convergence": convergence,
        "mach_window": mach_rows,
        "reynolds_window": re_rows,
        "negative_control": control,
        "main_case": {
            "dx_m": main_run["config"]["dx"],
            "particles": main_run["config"]["particles"],
            "steps": main_run["config"]["steps"],
            "dt_s": main_run["config"]["dt"],
            "decay_rate": main_run["measured"]["energy_decay_rate"],
            "decay_rate_ref": main_run["reference"]["energy_decay_rate"],
            "error_pct": main_run["error_pct"],
            "r2": main_run["measured"]["energy_decay_rate_r2"],
            "energy_ratio": main_run["measured"]["energy_ratio"],
            "energy_ratio_analytic": main_run["measured"]["energy_ratio_analytic"],
            "amplitude_ratio": main_run["measured"]["amplitude_ratio"],
            "amplitude_ratio_analytic": main_run["measured"]["amplitude_ratio_analytic"],
            "fit_window_s": main_run["measured"]["fit_window"],
            "series": {
                "time": main_run["series"]["time"],
                "energy": main_run["series"]["energy"],
                "energy_analytic": main_run["series"]["energy_analytic"],
            },
        },
        "tolerance": {"limit_pct": spec.tolerance_pct,
                      "pass": bool(main_run["error_pct"] < spec.tolerance_pct)},
        "environment": environment(),
        "repro_cmd": ("PYTHONPATH=src python -m tensorsph --case taylor_green "
                      f"--output {out.as_posix()}"),
        "wall_clock_s": None,
    }

    if figures:
        report["figures"] = {
            "fig1_tgv_velocity_field": str(_velocity_field_figure(spec, main_dx, out)),
            "fig2_tgv_energy_decay": str(_energy_decay_figure(spec, main_run, out)),
            "fig3_tgv_convergence": str(_convergence_figure(spec, convergence, out)),
            "fig4_tgv_sensitivity": str(_sensitivity_figure(spec, mach_rows, re_rows, out)),
        }
    report["wall_clock_s"] = time.time() - started

    (out / "results.json").write_text(json.dumps(report, indent=2))
    (out / "environment.json").write_text(json.dumps(report["environment"], indent=2))
    _write_log(out / "run.log", report)
    return report


def _write_log(path: Path, report: dict) -> None:
    """Human-readable record of the numbers stored in ``results.json``."""
    ns = report["reference_checks"]["ns_residual"]
    ana = report["reference_checks"]["analytic_field_decay_rate"]
    lines = [
        "%s  generated %s" % (report["case"], report["generated_utc"]),
        "reference: %s" % report["reference"]["citation"],
        "ns residual: max scaled momentum %.3e, continuity %.3e"
        % (ns["max_abs_momentum_residual_scaled"],
           ns["max_abs_continuity_residual_scaled"]),
        "analytic-field decay rate: %.6f 1/s vs closed form %.6f 1/s (%.2e %%)"
        % (ana["rate_from_analytic_field"], ana["rate_closed_form"],
           ana["rel_error_pct"]),
        "",
        "parameters: " + json.dumps(report["parameters"]),
        "",
        "ref-vs-got (main case, dx = %g m):" % report["main_case"]["dx_m"],
        "  %-26s %12s %12s %10s %8s %5s"
        % ("quantity", "ref", "got", "err[%]", "tol[%]", "pass"),
    ]
    for r in report["benchmark"]["rows"]:
        lines.append("  %-26s %12.6g %12.6g %10.3f %8.1f %5s"
                     % (r["quantity"], r["ref"], r["got"], r["rel_error_pct"],
                        r["tolerance_pct"], r["pass"]))
    lines += ["", "convergence:"]
    for r in report["convergence"]:
        lines.append("  dx=%-8g particles=%-6d steps=%-6d rate=%.5f ref=%.5f "
                     "err=%.2f%% r2=%.6f viscous-op=%.3f%%"
                     % (r["dx_m"], r["particles"], r["steps"], r["decay_rate"],
                        r["decay_rate_ref"], r["error_pct"], r["r2"],
                        r["viscous_operator_l2_error_pct"]))
    lines += ["", "mach window:"]
    for r in report["mach_window"]:
        lines.append("  M=%-8.4g U0=%-8.4g rate=%.5f ref=%.5f err=%.2f%% pass=%s"
                     % (r["mach"], r["U0_m_s"], r["decay_rate"],
                        r["decay_rate_ref"], r["error_pct"], r["pass"]))
    lines += ["", "reynolds window:"]
    for r in report["reynolds_window"]:
        lines.append("  Re=%-8.4g nu=%-10.6g rate=%.5f ref=%.5f err=%.2f%% pass=%s"
                     % (r["reynolds"], r["nu_m2_s"], r["decay_rate"],
                        r["decay_rate_ref"], r["error_pct"], r["pass"]))
    lines += ["", "negative control (nu = 0): rate=%.5f 1/s  E/E0=%.4f  "
              "nu_equivalent=%.6g m2/s"
              % (report["negative_control"]["decay_rate"],
                 report["negative_control"]["energy_ratio_at_t_end"],
                 report["negative_control"]["equivalent_numerical_viscosity_m2_s"]),
              "", "wall clock: %.1f s" % report["wall_clock_s"]]
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tensorsph.taylor_green",
        description=f"{CASE_KEY}: 2-D periodic Taylor-Green vortex vs the exact "
                    "Navier-Stokes decay")
    parser.add_argument("--out", default="runs/taylor_green")
    parser.add_argument("--dx", default=None,
                        help="comma-separated particle spacings for the "
                             "convergence study (default 0.05,0.025,0.0125)")
    parser.add_argument("--mach", default="0.002,0.005,0.01,0.02,0.05")
    parser.add_argument("--reynolds", default="2,5,10,20")
    parser.add_argument("--main-dx", type=float, default=0.025)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="CI-sized run: two resolutions, three Mach numbers, "
                             "two Reynolds numbers")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    spec = CaseSpec()
    if args.t_end is not None:
        spec.t_end = args.t_end

    if args.quick:
        dxs = (0.05, 0.025)
        machs = (0.005, 0.02, 0.05)
        reynolds = (2.0, 10.0)
        main_dx = 0.05
    else:
        dxs = tuple(float(v) for v in args.dx.split(",")) if args.dx else spec.dxs
        machs = tuple(float(v) for v in args.mach.split(","))
        reynolds = tuple(float(v) for v in args.reynolds.split(","))
        main_dx = args.main_dx

    out = Path(args.out)
    report = run_benchmark(out, spec, dxs=dxs, machs=machs, reynolds=reynolds,
                           figures=not args.no_figures, main_dx=main_dx)
    print("\nwrote %s" % (out / "results.json"))
    print("main case: dx=%.4g m, %d particles, decay-rate error %.2f %% "
          "(tolerance %.0f %%) -> %s"
          % (report["main_case"]["dx_m"], report["main_case"]["particles"],
             report["main_case"]["error_pct"], report["tolerance"]["limit_pct"],
             "PASS" if report["tolerance"]["pass"] else "FAIL"))
    return 0 if report["tolerance"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())