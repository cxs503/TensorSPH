#!/usr/bin/env python3
"""Taylor-Green vortex benchmark (TensorSPH).

Runs the 2-D periodic Taylor-Green vortex against the exact analytic solution
``E(t) = E(0) exp(-4 nu k^2 t)`` at multiple resolutions and parameters, and
produces a JSON report plus figures.

Reference
---------
The two-dimensional Taylor-Green vortex

    u(x,y,t) = U0 exp(-2 nu k^2 t) sin(kx) cos(ky)
    v(x,y,t) = -U0 exp(-2 nu k^2 t) cos(kx) sin(ky)
    p(x,y,t) = (rho U0^2 / 4) exp(-4 nu k^2 t) (cos 2kx + cos 2ky)

is an **exact, unsteady** solution of the incompressible Navier-Stokes
equations at any Reynolds number (Taylor & Green 1937; the 2-D form is the
standard periodic-vortex benchmark). Two identities make it exact:

* the convective term is a pure gradient, ``(u.grad)u = -grad[ ... ]``, so the
  pressure field absorbs it at every instant;
* ``lap u = -2 k^2 u`` for this field, so viscosity only rescales the amplitude.

Because the *shape* is an exact invariant, the kinetic energy must decay as

    E(t) = E(0) exp(-4 nu k^2 t)

so the measured decay rate ``s = -d ln E / dt`` can be compared with the
analytic exponent ``4 nu k^2`` directly.  That is the quantity gated here.

Usage::

    PYTHONPATH=src python benchmarks/taylor_green/run.py --out artifacts/taylor_green

Writes ``report.json`` plus four PNGs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.vv.tgv import (TGVConfig, analytic_field, kinetic_energy,  # noqa: E402
                              make_lattice, run_tgv, viscous_operator_error)

# --------------------------------------------------------------------------- #
# Reference parameters (the exact solution's inputs)
# --------------------------------------------------------------------------- #
L = 1.0                    # periodic box side (m)
K = 2.0 * math.pi          # wavenumber (one wavelength per side)
NU = 1.0 / (10.0 * 2.0 * math.pi)   # nu = U0/(Re k), Re = 10 at U0 = 1
U0 = 0.2                   # peak initial velocity (m/s)
SOUND_SPEED = 10.0         # acoustic speed (m/s)
RHO0 = 1000.0
GAMMA = 7.0
T_END = 0.6                # simulated time (s)

# The exact decay rate of the reference solution.
ANALYTIC_RATE = 4.0 * NU * K * K


def _run_case(dx: float, u0: float = U0, sound_speed: float = SOUND_SPEED,
              viscosity: float = NU, t_end: float = T_END) -> dict:
    """Run one TGV case and return the full result dict."""
    cfg = TGVConfig(dx=dx, t_end=t_end, U0=u0, sound_speed=sound_speed,
                    viscosity=viscosity)
    run = run_tgv(cfg)
    run["measured"]["viscous_operator_l2_error_pct"] = viscous_operator_error(cfg)
    return run


def _velocity_field_plot(cfg: TGVConfig, out: Path) -> Path:
    """Quiver plot of the analytic velocity field at t = 0."""
    pos, _ = make_lattice(cfg)
    vel = analytic_field(pos, 0.0, cfg)
    n = int(round(cfg.L / cfg.dx))
    u = vel[:, 0].reshape(n, n).cpu().numpy()
    v = vel[:, 1].reshape(n, n).cpu().numpy()
    x = pos[:, 0].reshape(n, n).cpu().numpy()
    y = pos[:, 1].reshape(n, n).cpu().numpy()

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    q = ax.quiver(x[::2, ::2], y[::2, ::2], u[::2, ::2], v[::2, ::2],
                  scale=30, width=0.004)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Taylor-Green vortex, t = 0, dx = {cfg.dx} m\n"
                 f"$U_0$ = {cfg.U0} m/s, $k$ = {cfg.k:.2f} 1/m")
    ax.set_aspect("equal")
    fig.tight_layout()
    path = out / "fig1_tgv_velocity_field.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _energy_decay_plot(run: dict, out: Path) -> Path:
    """SPH kinetic energy vs the exact exp(-4 nu k^2 t) decay."""
    times = run["series"]["time"]
    energy = run["series"]["energy"]
    energy_analytic = run["series"]["energy_analytic"]
    e0 = energy[0]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(times, [e / e0 for e in energy], "o-", ms=3, lw=1.2,
                label="SPH")
    ax.semilogy(times, [e / e0 for e in energy_analytic], "-", lw=1.5,
                color="tab:red", label="exact $\\exp(-4\\nu k^2 t)$")
    ax.set_xlabel("time $t$ [s]")
    ax.set_ylabel("$E(t)/E(0)$")
    ax.set_title(f"TGV energy decay, dx = {run['config']['dx']} m, "
                 f"$\\nu$ = {run['config']['viscosity']:.5f} m$^2$/s\n"
                 f"decay-rate error = {run['error_pct']:.2f} %")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "fig2_tgv_energy_decay.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _convergence_plot(rows: list, out: Path) -> Path:
    """Decay-rate error vs resolution (dx), with the 8 % tolerance line."""
    dxs = [r["dx"] for r in rows]
    errs = [r["error_pct"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(dxs, errs, "o-", lw=1.5)
    ax.axhline(8.0, color="tab:red", ls="--", lw=1.0, label="8 % tolerance")
    ax.axhline(3.0, color="tab:orange", ls=":", lw=1.0, label="3 % gate")
    ax.set_xlabel("particle spacing $dx$ [m]")
    ax.set_ylabel("decay-rate error [%]")
    ax.set_title("TGV convergence: decay-rate error vs resolution")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "fig3_tgv_convergence.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _mach_window_plot(rows: list, out: Path) -> Path:
    """Decay-rate error vs Mach number (parameter sensitivity)."""
    machs = [r["mach"] for r in rows]
    errs = [r["error_pct"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogx(machs, errs, "o-", lw=1.5)
    ax.axhline(8.0, color="tab:red", ls="--", lw=1.0, label="8 % tolerance")
    ax.axhline(3.0, color="tab:orange", ls=":", lw=1.0, label="3 % gate")
    ax.set_xlabel("Mach number $U_0/c_0$")
    ax.set_ylabel("decay-rate error [%]")
    ax.set_title("TGV parameter sensitivity: error vs Mach number")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "fig4_tgv_mach_window.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "artifacts" / "taylor_green"))
    parser.add_argument("--dx", default="0.05,0.025,0.0125",
                        help="resolutions for the convergence study (comma-separated)")
    parser.add_argument("--mach", default="0.002,0.005,0.01,0.02,0.05",
                        help="Mach numbers for the sensitivity study (comma-separated)")
    parser.add_argument("--reynolds", default="2,5,10",
                        help="Reynolds numbers for the sensitivity study (comma-separated)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(4)  # keep the small tensors from spawning a huge pool

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # ---- convergence study -------------------------------------------------
    convergence = []
    for dx in [float(v) for v in args.dx.split(",")]:
        run = _run_case(dx)
        convergence.append({
            "dx": dx,
            "particles": run["config"]["particles"],
            "steps": run["config"]["steps"],
            "decay_rate": run["measured"]["energy_decay_rate"],
            "decay_rate_analytic": run["reference"]["energy_decay_rate"],
            "error_pct": run["error_pct"],
            "r2": run["measured"]["energy_decay_rate_r2"],
            "viscous_operator_l2_error_pct": run["measured"]["viscous_operator_l2_error_pct"],
            "energy_ratio": run["measured"]["energy_ratio"],
            "energy_ratio_analytic": run["measured"]["energy_ratio_analytic"],
        })
        print(f"dx={dx:g}: {run['config']['particles']} particles, "
              f"err={run['error_pct']:.2f} %")

    # ---- Mach-number sensitivity -------------------------------------------
    mach_rows = []
    for m in [float(v) for v in args.mach.split(",")]:
        u0 = m * SOUND_SPEED
        run = _run_case(dx=0.05, u0=u0)
        mach_rows.append({
            "mach": m,
            "u0": u0,
            "decay_rate": run["measured"]["energy_decay_rate"],
            "decay_rate_analytic": run["reference"]["energy_decay_rate"],
            "error_pct": run["error_pct"],
            "r2": run["measured"]["energy_decay_rate_r2"],
        })
        print(f"M={m:g}: err={run['error_pct']:.2f} %")

    # ---- Reynolds-number sensitivity ----------------------------------------
    re_rows = []
    for re in [float(v) for v in args.reynolds.split(",")]:
        # nu = U0 / (Re k); keep U0 and k fixed, vary nu.
        nu = U0 / (re * K)
        run = _run_case(dx=0.05, viscosity=nu)
        re_rows.append({
            "reynolds": re,
            "viscosity": nu,
            "decay_rate": run["measured"]["energy_decay_rate"],
            "decay_rate_analytic": run["reference"]["energy_decay_rate"],
            "error_pct": run["error_pct"],
            "r2": run["measured"]["energy_decay_rate_r2"],
        })
        print(f"Re={re:g}: err={run['error_pct']:.2f} %")

    # ---- main case (for the energy-decay figure) ----------------------------
    main_run = _run_case(dx=0.025)
    main_cfg = TGVConfig(dx=0.025, t_end=T_END, U0=U0, sound_speed=SOUND_SPEED,
                         viscosity=NU)

    # ---- assemble the report ------------------------------------------------
    report = {
        "case": "taylor_green_vortex_2d_periodic",
        "reference": {
            "kind": "analytic",
            "solution": (
                "u = U0 exp(-2 nu k^2 t) sin(kx) cos(ky), "
                "v = -U0 exp(-2 nu k^2 t) cos(kx) sin(ky), "
                "p = (rho U0^2/4) exp(-4 nu k^2 t) (cos 2kx + cos 2ky)"),
            "source": "Taylor & Green (1937), Proc. R. Soc. Lond. A 158, 499-521; "
                      "the 2-D periodic form is the standard exact benchmark",
            "energy_decay_rate": ANALYTIC_RATE,
            "energy_ratio_analytic": math.exp(-ANALYTIC_RATE * T_END),
        },
        "parameters": {
            "L": L, "k": K, "nu": NU, "U0": U0, "c0": SOUND_SPEED,
            "rho0": RHO0, "gamma": GAMMA, "t_end": T_END,
            "reynolds": U0 / (NU * K),
            "mach": U0 / SOUND_SPEED,
            "h_over_dx": 1.3,
            "max_courant": 0.3,
        },
        "convergence": convergence,
        "mach_sensitivity": mach_rows,
        "reynolds_sensitivity": re_rows,
        "main_case": {
            "dx": main_run["config"]["dx"],
            "particles": main_run["config"]["particles"],
            "steps": main_run["config"]["steps"],
            "decay_rate": main_run["measured"]["energy_decay_rate"],
            "decay_rate_analytic": main_run["reference"]["energy_decay_rate"],
            "error_pct": main_run["error_pct"],
            "r2": main_run["measured"]["energy_decay_rate_r2"],
            "energy_ratio": main_run["measured"]["energy_ratio"],
            "energy_ratio_analytic": main_run["measured"]["energy_ratio_analytic"],
            "series": {
                "time": main_run["series"]["time"],
                "energy": main_run["series"]["energy"],
                "energy_analytic": main_run["series"]["energy_analytic"],
            },
        },
        "tolerance": {"limit_pct": 8.0, "pass": bool(main_run["error_pct"] < 8.0)},
        "wall_clock_s": time.time() - started,
    }

    (out / "report.json").write_text(json.dumps(report, indent=2))

    # ---- figures -------------------------------------------------------------
    if not args.no_figures:
        figs = {}
        figs["fig1_tgv_velocity_field"] = str(_velocity_field_plot(main_cfg, out))
        figs["fig2_tgv_energy_decay"] = str(_energy_decay_plot(main_run, out))
        figs["fig3_tgv_convergence"] = str(_convergence_plot(convergence, out))
        figs["fig4_tgv_mach_window"] = str(_mach_window_plot(mach_rows, out))
        report["figures"] = figs
        (out / "report.json").write_text(json.dumps(report, indent=2))

    print(f"\nwrote {out / 'report.json'}")
    print(f"main case: dx={main_run['config']['dx']} m, "
          f"{main_run['config']['particles']} particles, "
          f"err={main_run['error_pct']:.2f} % "
          f"(tolerance {report['tolerance']['limit_pct']} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
