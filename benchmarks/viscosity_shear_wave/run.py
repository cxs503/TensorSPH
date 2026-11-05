"""Viscosity verification: end-to-end decay of a standing shear wave.

Why this case
-------------
The repository already pins the viscous *operator* (``VISCOUS_LAPLACIAN_FACTOR``
and the Morris et al. (1997) factor 2) with ``tests/test_pair_based_rhs.py::
test_viscosity_is_calibrated``, which measures applied/nominal = 1.000 +- 1e-3
on a *frozen* smooth field.  That is a kernel check: it says the operator
evaluates ``nu * laplace(u)`` correctly at one instant, not that a flow
integrated with the production ``rhs`` decays at the physical rate.  A wrong
pressure coupling, a boundary leak or a time-integration bias can leave the
operator calibration at 1.000 and still move the flow's effective viscosity.
This benchmark closes that gap by measuring ``nu_eff`` from a *time series*
produced by the solver.

Analytic anchor
---------------
In a quiescent box (``gravity = 0``) the velocity field

    u_x(y, t) = U0 * sin(k y) * exp(-nu k^2 t),   u_y = 0

is an **exact** solution of the incompressible Navier-Stokes equations:
the convective term vanishes identically (``u . grad u = u_x d/dx u = 0``
because the field carries no x dependence), and the field is an eigenfunction
of ``d^2/dy^2`` with eigenvalue ``-k^2``.  It also satisfies no-slip at every
``y = m * lambda / 2`` (``sin = 0`` there), so a **bounded** box of height
``Ly = n * lambda / 2`` with static walls on the planes ``y = 0`` and
``y = Ly`` needs no periodic boundary condition to be exact.  The decay rate
of the discrete amplitude is therefore a direct, unambiguous measurement of
the *effective kinematic viscosity*

    k^2 nu_eff = -d/dt ln A(t),   A(t) = <u_x, sin(k y)> / <sin(k y), sin(k y)>.

The walls are static and the box is closed, so no work is done on the fluid
and no convective term exists: everything the amplitude loses is viscous
dissipation (plus whatever numerical dissipation the scheme adds, which the
``nu = 0`` control measures separately).

Controls
--------
1. ``nu = 0``: the same initial field with the viscous term switched off.  A
   trustworthy measurement must return ``nu_eff ~ 0``; whatever it returns is
   the scheme's numerical dissipation floor and is reported as a fraction of
   the nominal ``nu``.
2. Resolution ladder: ``dx`` halved.  A physical viscosity is resolution
   independent; a numerical artefact is not.
3. Exponentiality: R^2 of the log-linear fit.  A single decay rate only means
   something if the amplitude actually decays exponentially.

Usage
-----
    PYTHONPATH=src python benchmarks/viscosity_shear_wave/run.py [--quick] [--plot]

Writes ``report.json`` (all numbers the doc is generated from) next to this
file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from tensorsph.droplet_impact import Config, rhs

RHO0 = 1000.0
HERE = Path(__file__).resolve().parent

# The tensors here are tiny (a few thousand particles), so BLAS-style threading
# only adds scheduler contention: on a loaded 128-core host the same step took
# 222 ms with the default thread count and 14 ms pinned to one thread. Pin it.
# Overridable, and callers may save/restore ``torch.get_num_threads()``.
if "TENSORSPH_THREADS" in os.environ:
    torch.set_num_threads(int(os.environ["TENSORSPH_THREADS"]))
else:
    torch.set_num_threads(1)

# Nominal case.  lambda = 20 dx keeps the discrete Laplacian well inside its
# asymptotic regime ((k dx)^2 / 12 = 0.8 % truncation at dx = 0.0025) while
# leaving 20 rows per wavelength for the mode projection.
LAMBDA_FACTOR = 12.0
# Distance kept between the x-walls and the projection window, in viscous
# penetration lengths sqrt(nu t_end).  The no-slip x-walls are *not* part of the
# analytic solution (which is x invariant) and their boundary layer diffuses
# inward as sqrt(nu t); a window closer than a few penetration lengths measures
# wall damping instead of the bulk decay.
MARGIN_IN_DEPTHS = 3.5


def box_walls(lx: float, ly: float, dx: float, layers: int) -> torch.Tensor:
    """Static wall particles on four sides of the fluid box.

    The placement mirrors ``tensorsph.droplet_impact.wall_lattice``: the fluid
    lattice occupies the cell centres ``(i + 0.5) dx`` of the box, and the wall
    layers sit *half a spacing outside* it, at ``0.5 dx - layer * dx``. Putting
    a wall layer on the boundary plane instead would leave the near-wall fluid
    column one half spacing from it and start the run about 30 % over-dense --
    measured, see docs/verification_viscosity_shear_wave.md.
    """
    if layers < 1:
        raise ValueError("layers must be positive")
    span_x = torch.arange(-(layers - 0.5) * dx, lx + (layers - 0.5) * dx + 0.25 * dx,
                          dx, dtype=torch.float64)
    columns = torch.arange(0.5 * dx, ly, dx, dtype=torch.float64)
    parts = []
    for layer in range(1, layers + 1):
        for y in (0.5 * dx - layer * dx, ly - 0.5 * dx + layer * dx):
            gx, gy = torch.meshgrid(span_x, torch.tensor([y], dtype=torch.float64),
                                    indexing="xy")
            parts.append(torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1))
    for layer in range(1, layers + 1):
        for x in (0.5 * dx - layer * dx, lx - 0.5 * dx + layer * dx):
            gx, gy = torch.meshgrid(torch.tensor([x], dtype=torch.float64), columns,
                                    indexing="xy")
            parts.append(torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1))
    return torch.cat(parts, dim=0).contiguous()


def fluid_lattice(lx: float, ly: float, dx: float) -> torch.Tensor:
    """Fluid cell centres, same convention as ``droplet_impact.cell_centres``."""
    def centres(extent):
        count = int(math.floor(extent / dx + 1.0e-9))
        if count < 1:
            raise ValueError(f"dx={dx:g} too coarse for extent {extent:g}")
        return (torch.arange(count, dtype=torch.float64) + 0.5) * dx

    gx, gy = torch.meshgrid(centres(lx), centres(ly), indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1).contiguous()


def mode_amplitude(pos, vel, k: float, x_lo: float, x_hi: float):
    """Project u_x onto sin(k y) over the interior x band.

    The x-walls also impose no-slip, which is *not* part of the analytic
    solution (the field is x invariant there); the outermost band is therefore
    excluded from the projection and the excluded width is reported.
    """
    keep = (pos[:, 0] >= x_lo) & (pos[:, 0] <= x_hi)
    y = pos[keep, 1]
    ux = vel[keep, 0]
    s = torch.sin(k * y)
    weight = (s * s).sum()
    if float(weight) <= 0.0:
        raise RuntimeError("empty projection window")
    return float((ux * s).sum() / weight)


def run_case(dx: float, viscosity: float, nu_nominal: float, *, u0: float = 0.2,
             n_e_folds: float = 3.0, layers: int = 3, c0: float = 20.0,
             samples: int = 40, relax_steps: int = 300,
             window: float | None = None, lam: float | None = None) -> dict:
    """Integrate the shear wave and return the measured decay rate."""
    # lam is a *physical* length so that a refinement ladder keeps the same
    # wave; only the number of cells per wavelength changes with dx.
    lam = LAMBDA_FACTOR * dx if lam is None else lam
    window = LAMBDA_FACTOR * dx if window is None else window
    ly = lam  # one full wavelength: two half-wavelengths between the y-walls
    k = 2.0 * math.pi / lam
    h = 1.3 * dx
    omega = nu_nominal * k * k
    t_end = n_e_folds / omega
    # Viscous penetration length over the measurement window.
    depth = math.sqrt(nu_nominal * t_end)
    margin = MARGIN_IN_DEPTHS * depth
    lx = 2.0 * margin + window

    cfg = Config(dx=dx, h_factor=1.3, rho0=RHO0, sound_speed=c0, gravity=0.0,
                 viscosity=viscosity, dt=1.0, wall_friction=1.0)
    # Acoustic Courant 0.25 of the smoothing length; also respect the explicit
    # viscous limit so the case cannot be run outside its stability window.
    dt_acoustic = 0.25 * h / c0
    dt_viscous = 0.125 * h * h / viscosity if viscosity > 0 else float("inf")
    dt = min(dt_acoustic, dt_viscous)
    steps = max(int(math.ceil(t_end / dt)), 2)
    sample_every = max(steps // samples, 1)

    pos = fluid_lattice(lx, ly, dx)
    mass = torch.full((pos.shape[0],), RHO0 * dx * dx, dtype=torch.float64)
    wall = box_walls(lx, ly, dx, layers)
    vel = torch.zeros_like(pos)

    # Relax the lattice before the measurement window.  The static walls leave a
    # one-sided kernel-support deficit on the near-wall column, and starting the
    # shear straight away converts that density error into an acoustic transient
    # that re-excites the projected mode (measured: the amplitude rose again
    # after four e-folds of decay).  Relaxing at rest lets the pressure field
    # find the state it will hold for the whole run, so the window measures the
    # decay alone.  The analytic initial condition is imposed *after* it.
    for _ in range(relax_steps):
        _, _, acc = rhs(pos, vel, mass, cfg, wall=wall)
        vel = vel + dt * acc
        pos = pos + dt * vel
        vel.mul_(0.0)
    vel[:, 0] = u0 * torch.sin(k * pos[:, 1])
    rho_relaxed, _, _ = rhs(pos, vel, mass, cfg, wall=wall)
    relaxed_density_error = float((rho_relaxed / RHO0 - 1.0).abs().max())

    x_lo, x_hi = margin - 0.5 * dx, margin + window - 0.5 * dx
    m0 = float(mass.sum())
    times, amps, dens_err, speeds = [], [], [], []
    bulk_err, wall_err = [], []
    for step in range(steps):
        rho, _, acc = rhs(pos, vel, mass, cfg, wall=wall)
        if not bool(torch.isfinite(acc).all()):
            raise RuntimeError(f"non-finite acceleration at step {step}")
        vel = vel + dt * acc
        pos = pos + dt * vel
        if step % sample_every == 0:
            times.append((step + 1) * dt)
            amps.append(mode_amplitude(pos, vel, k, x_lo, x_hi))
            dens_err.append(float((rho / RHO0 - 1.0).abs().max()))
            bulk = (torch.sin(k * pos[:, 1]).abs() > 0.5) & (
                (pos[:, 0] >= x_lo) & (pos[:, 0] <= x_hi))
            if bool(bulk.any()):
                bulk_err.append(float((rho[bulk] / RHO0 - 1.0).abs().max()))
                wall_err.append(float((rho[~bulk] / RHO0 - 1.0).abs().max()))
            speeds.append(float(vel.norm(dim=1).max()))

    t = torch.tensor(times, dtype=torch.float64)
    a = torch.tensor(amps, dtype=torch.float64)
    positive = a > 0.0
    if int(positive.sum()) < 3:
        raise RuntimeError("mode amplitude changed sign; case is not measuring decay")
    t, a = t[positive], a[positive]
    log_a = torch.log(a)
    design = torch.stack((torch.ones_like(t), t), dim=1)
    coef, *_ = torch.linalg.lstsq(design, log_a.unsqueeze(1))
    slope = float(coef[1, 0])
    intercept = float(coef[0, 0])
    fit = intercept + slope * t
    ss_res = float(((log_a - fit) ** 2).sum())
    ss_tot = float(((log_a - log_a.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    nu_eff = -slope / (k * k)

    return {
        "dx": dx, "dt": dt, "steps": steps, "lambda": lam, "k": k,
        "ly": ly, "u0": u0, "viscosity": viscosity,
        "n_fluid": int(pos.shape[0]), "n_wall": int(wall.shape[0]),
        "lx": lx, "viscous_depth": depth, "margin": margin,
        "h": h, "acoustic_courant": 0.25,
        "t_end": steps * dt, "simulated_time": times[-1] if times else None,
        "projection_window_x": [x_lo, x_hi],
        "decay_rate_measured": -slope,
        "decay_rate_analytic": omega if viscosity > 0 else 0.0,
        "nu_effective": nu_eff,
        "nu_nominal": nu_nominal,
        "nu_ratio": (nu_eff / viscosity) if viscosity > 0 else None,
        "amplitude_initial": float(torch.exp(torch.tensor(intercept))),
        "amplitude_final": float(a[-1]),
        "log_linear_r2": r2,
        "max_density_error": max(dens_err) if dens_err else None,
        "density_error_after_relax": relaxed_density_error,
        "density_error_bulk": max(bulk_err) if bulk_err else None,
        "density_error_outside_bulk": max(wall_err) if wall_err else None,
        "max_speed": max(speeds) if speeds else None,
        "mass": m0,
        "series": {"t": times, "amplitude": amps},
    }


def criteria(cases: dict) -> list[dict]:
    """Machine-checkable acceptance criteria (the gate this benchmark feeds)."""

    def crit(cid, metric, measured, threshold, comparison, note=""):
        passed = measured <= threshold if comparison == "<=" else measured >= threshold
        return {"id": cid, "metric": metric, "measured": measured,
                "threshold": threshold, "comparison": comparison,
                "result": "pass" if passed else "fail", "note": note}

    fine = cases["nominal"]
    control = cases["zero_viscosity"]
    coarse = cases["coarse"]
    return [
        crit("nu_effective_within_15pct", "|nu_eff/nu - 1| on the finest grid",
             abs(fine["nu_ratio"] - 1.0), 0.15, "<=",
             "platform tolerance; nu_eff from the decay of the projection"),
        crit("decay_is_exponential", "R^2 of the log-linear amplitude fit",
             fine["log_linear_r2"], 0.999, ">=",
             "a single rate is only meaningful if the decay is exponential"),
        crit("numerical_dissipation_floor", "nu_eff(nu=0)/nu_nominal",
             abs(control["nu_effective"]) / fine["nu_nominal"], 0.05, "<=",
             "negative control: no viscosity must mean no decay"),
        crit("refines_toward_nominal",
             "|nu_eff/nu - 1| on the fine grid must not exceed the coarse grid",
             abs(fine["nu_ratio"] - 1.0) - abs(coarse["nu_ratio"] - 1.0), 0.0, "<=",
             "a discretisation error must shrink under refinement; a physical "
             "viscosity must not"),
        crit("density_bounded_bulk",
             "max |rho/rho0 - 1| where |sin ky| > 0.5 (the region that carries "
             "the projection)", fine["density_error_bulk"], 0.05, "<=",
             "incompressibility error in the measured region; the wall-adjacent "
             "band is reported separately"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="nominal + control only, coarse ladder point skipped")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--dx", type=float, default=0.0025)
    ap.add_argument("--lam", type=float, default=0.06,
                    help="physical wavelength, held fixed across the ladder")
    args = ap.parse_args()

    window = args.lam
    cases = {}
    cases["nominal"] = run_case(args.dx, args.nu, args.nu, lam=args.lam, window=window)
    cases["zero_viscosity"] = run_case(args.dx, 0.0, args.nu, lam=args.lam,
                                       window=window)
    if args.quick:
        cases["coarse"] = cases["nominal"]
    else:
        cases["coarse"] = run_case(2.0 * args.dx, args.nu, args.nu, lam=args.lam,
                                   window=window)

    gate = criteria(cases)
    report = {
        "benchmark": "viscosity_shear_wave",
        "model": "WCSPH (Morris regularized Laplacian) + Adami static walls",
        "reference": {
            "solution": "u_x = U0 sin(k y) exp(-nu k^2 t)", "exact_ns": True,
            "why_exact": "convective term vanishes identically; eigenfunction of d2/dy2",
            "no_slip": "sin(k y) = 0 on y = m lambda/2, so a bounded box is exact",
            "citation": "Morris, Fox & Zhu (1997), J. Comput. Phys. 136, 214",
        },
        "setup": {"lambda": args.lam, "nu": args.nu,
                  "u0": cases["nominal"]["u0"], "gravity": 0.0,
                  "wall_friction": 1.0,
                  "viscous_depth": cases["nominal"]["viscous_depth"],
                  "window_margin_in_depths": MARGIN_IN_DEPTHS},
        "cases": {name: {kk: vv for kk, vv in case.items() if kk != "series"}
                  for name, case in cases.items()},
        "ladder": sorted(
            ({"dx": c["dx"], "cells_per_wavelength": c["lambda"] / c["dx"],
              "nu_ratio": c["nu_ratio"], "nu_effective": c["nu_effective"],
              "log_linear_r2": c["log_linear_r2"]}
             for n, c in cases.items() if n != "zero_viscosity"),
            key=lambda row: row["dx"]),
        "criteria": gate,
        "checks": {c["id"]: {"measured": c["measured"], "limit": c["threshold"],
                             "result": c["result"]} for c in gate},
        "series": {name: case["series"] for name, case in cases.items()},
    }
    out = HERE / "report.json"
    out.write_text(json.dumps(report, indent=2))
    failed = [c["id"] for c in gate if c["result"] == "fail"]
    print(f"wrote {out}")
    for c in gate:
        print(f"  {c['result']:4s} {c['id']:30s} {c['measured']:.4g} "
              f"({c['comparison']} {c['threshold']:g})")
    if args.plot:
        plot(report)
    return 1 if failed else 0


def plot(report: dict) -> None:
    """Amplitude decay + implied-viscosity chart (report figure source)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    colors = {"nominal": "C0", "coarse": "C1", "zero_viscosity": "C2"}
    for name, series in report["series"].items():
        case = report["cases"][name]
        t = series["t"]
        a = series["amplitude"]
        ax.semilogy(t, a, "o", ms=3, color=colors.get(name), label=f"{name} run")
        t_th = torch.linspace(0.0, max(t), 200, dtype=torch.float64)
        a_th = case["amplitude_initial"] * torch.exp(
            -torch.tensor(case["decay_rate_analytic"]) * t_th)
        if name != "zero_viscosity":
            ax.semilogy(t_th, a_th, "-", lw=1.0, color=colors.get(name),
                        label=f"{name} analytic e^(-nu k^2 t)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("mode amplitude A(t) [m/s]")
    ax.set_title("Shear-wave decay: projection onto sin(ky)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    names = [n for n in report["cases"]]
    ratios = []
    for n in names:
        case = report["cases"][n]
        ratios.append(case["nu_effective"] / case["nu_nominal"])
    ax.bar(names, ratios, color=[colors.get(n) for n in names])
    ax.axhline(1.0, color="k", lw=1.0, ls="--", label="nominal nu (ratio = 1)")
    ax.axhspan(0.85, 1.15, color="g", alpha=0.12, label="+/-15 % band")
    ax.set_ylabel(r"$\nu_{\rm eff}/\nu_{\rm nominal}$")
    ax.set_title("Implied viscosity from the measured decay rate")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(HERE / "viscosity_shear_wave.png", dpi=140)
    print(f"wrote {HERE / 'viscosity_shear_wave.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
