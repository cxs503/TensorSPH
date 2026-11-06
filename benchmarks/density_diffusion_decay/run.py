#!/usr/bin/env python3
"""Analytic gate for the delta-SPH diffusive term: gamma = D k^2, D = delta c0 h.

The quantity
------------
``density_diffusion_rate`` implements the diffusive term of Antuono, Colagrossi &
Marrone (2012, J. Comput. Phys. 231:946) as

    drho_i/dt = 2 delta c0 h * sum_j V_j (rho_i - rho_j) L_ij ,
    L_ij = -(r_ij . grad_i W_ij) / |r_ij|^2 .

``2 sum_j V_j (rho_i - rho_j) L_ij`` is the kernel estimate of ``lap rho``
(Brookshaw 1985, J. Comput. Phys. 61:43), so the operator is a plain diffusion

    drho/dt = D lap rho ,        D = delta * c0 * h ,     h = h_factor * dx .

For the plane wave ``rho = rho0 (1 + A cos kx)`` the exact solution is
``A(t) = A exp(-gamma t)`` with

    gamma = D k^2 = delta * c0 * h * k^2 ,

a closed form with **no free constant**: delta, c0 and h are model inputs.  This
harness measures gamma by projecting the operator's output onto the mode (a
three-column least-squares fit of ``[1, cos kx, sin kx]`` over a rim-free band,
so the constant part of rho cannot leak into the amplitude), and compares it with
the closed form across a resolution ladder and a wavenumber ladder.

Tolerance
---------
The discrete symbol of the kernel Laplacian is not ``-k^2``; the error is governed
by ``k dx`` and vanishes as ``(k dx)^2``.  The gate is 3 % on gamma, about 3x the
largest measured truncation error and 5x below the platform's 15 % engineering
bar, and 3.5 % on the finite-time amplitude (same truncation error plus the
first-order error of the explicit substep).

Usage
-----
    PYTHONPATH=src python3 benchmarks/density_diffusion_decay/run.py --figure

Writes ``decay_rate.json`` (machine-readable, used as run evidence) and, with
``--figure``, ``decay_rate.png`` next to this file.  Exit status is non-zero when
a gate fails.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    Config,
    cubic_spline_gradient,
    density_diffusion_rate,
)
from tensorsph.neighbor_search import cell_linked_pairs

OUT_DIR = Path(__file__).resolve().parent
RHO0 = 1000.0
DELTA = 1.0
SOUND_SPEED = 20.0
H_FACTOR = 1.3
AMPLITUDE = 1.0e-3
GATE_RATE = 0.03
GATE_AMPLITUDE = 0.035
DTYPE = torch.float64


def plane_wave_state(dx: float, wavelength: float, ny: int,
                     h_factor: float = H_FACTOR):
    """Lattice carrying ``rho = rho0 (1 + A cos kx)`` plus its pair geometry."""
    cfg = Config(dx=dx, h_factor=h_factor, rho0=RHO0, sound_speed=SOUND_SPEED,
                 gamma=7.0, gravity=0.0, viscosity=0.0, density_diffusion=DELTA,
                 device="cpu")
    h = cfg.h_factor * cfg.dx
    nx = int(round(wavelength / dx))
    xs = (torch.arange(nx, dtype=DTYPE) + 0.5) * dx
    ys = (torch.arange(ny, dtype=DTYPE) + 0.5) * dx
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    k = 2.0 * math.pi / wavelength

    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    separation = pos[i] - pos[j]
    grad_w = cubic_spline_gradient(separation, h)
    distance = torch.linalg.vector_norm(separation, dim=1)
    radial = -(separation * grad_w).sum(dim=1)

    rho = cfg.rho0 * (1.0 + AMPLITUDE * torch.cos(k * pos[:, 0]))
    mass = torch.full((pos.shape[0],), cfg.rho0 * dx * dx, dtype=DTYPE)
    interior = ((pos[:, 0] > 2.0 * h) & (pos[:, 0] < nx * dx - 2.0 * h)
                & (pos[:, 1] > 2.0 * h) & (pos[:, 1] < ny * dx - 2.0 * h))
    return cfg, pos, rho, mass, (distance, radial, (i, j)), k, interior


def cosine_amplitude(field, pos, k, interior) -> float:
    """``cos(kx)`` coefficient of a least-squares fit of ``[1, cos kx, sin kx]``."""
    x = pos[interior, 0]
    basis = torch.stack((torch.ones_like(x), torch.cos(k * x), torch.sin(k * x)), dim=1)
    return float(torch.linalg.lstsq(basis, field[interior]).solution[1])


def theory_rate(dx: float, wavelength: float, h_factor: float = H_FACTOR) -> float:
    """``gamma = D k^2`` with ``D = delta c0 h`` and ``h = h_factor dx``."""
    k = 2.0 * math.pi / wavelength
    return DELTA * SOUND_SPEED * h_factor * dx * k * k


def rate_row(dx: float, wavelength: float, ny: int,
             h_factor: float = H_FACTOR) -> dict:
    cfg, pos, rho, mass, geometry, k, interior = plane_wave_state(
        dx, wavelength, ny, h_factor)
    rate = density_diffusion_rate(rho, mass, geometry[0], geometry[1],
                                  geometry[2], cfg)
    measured = -cosine_amplitude(rate, pos, k, interior) / (cfg.rho0 * AMPLITUDE)
    theory = theory_rate(dx, wavelength, h_factor)
    return {
        "dx": dx,
        "wavelength": wavelength,
        "h_over_dx": h_factor,
        "k": k,
        "k_dx": k * dx,
        "ny": ny,
        "particles": int(pos.shape[0]),
        "pairs": int(geometry[0].shape[0]),
        "interior_particles": int(interior.sum()),
        "gamma_measured_1_per_s": measured,
        "gamma_theory_1_per_s": theory,
        "relative_error": abs(measured - theory) / theory,
        "passed": abs(measured - theory) / theory < GATE_RATE,
    }


def finite_time_trace(dx: float = 0.02, wavelength: float = 1.0, ny: int = 12,
                      dt: float = 1.2e-4, steps: int = 100) -> dict:
    """Explicit integration of the operator: amplitude vs ``exp(-gamma t)``."""
    cfg, pos, rho, mass, geometry, k, interior = plane_wave_state(dx, wavelength, ny)
    theory = theory_rate(dx, wavelength)
    initial = cosine_amplitude(rho, pos, k, interior) / (cfg.rho0 * AMPLITUDE)
    field = rho.clone()
    trace = [{"t": 0.0, "amplitude": initial, "theory": initial}]
    for step in range(1, steps + 1):
        rate = density_diffusion_rate(field, mass, geometry[0], geometry[1],
                                      geometry[2], cfg)
        field = field + dt * rate
        t = step * dt
        trace.append({
            "t": t,
            "amplitude": cosine_amplitude(field, pos, k, interior) / (cfg.rho0 * AMPLITUDE),
            "theory": initial * math.exp(-theory * t),
        })
    final = trace[-1]
    relative = abs(final["amplitude"] - final["theory"]) / abs(final["theory"])
    return {
        "dx": dx,
        "wavelength": wavelength,
        "dt": dt,
        "steps": steps,
        "gamma_theory_1_per_s": theory,
        # dt < 2 / (D k_max^2) with k_max = pi/dx (stiffest lattice mode).
        "stability_limit_dt": 2.0 * dx**2 / (DELTA * SOUND_SPEED * H_FACTOR * dx
                                             * math.pi**2),
        "final_amplitude": final["amplitude"],
        "final_theory": final["theory"],
        "relative_error": relative,
        "passed": relative < GATE_AMPLITUDE,
        "trace": trace,
    }


def figure(payload: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ladder = payload["resolution_ladder"]
    axes[0].plot([r["k_dx"] for r in ladder],
                 [100.0 * r["relative_error"] for r in ladder], "o-")
    axes[0].axhline(100.0 * GATE_RATE, color="crimson", ls="--", label="gate 3 %")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("k dx  (mode resolution, smaller is better)")
    axes[0].set_ylabel("|gamma_num - gamma_theory| / gamma_theory  [%]")
    axes[0].set_title("(a) rate error vs mode resolution")
    axes[0].legend()

    waves = payload["wavenumber_ladder"]
    axes[1].plot([r["k"] for r in waves], [r["gamma_measured_1_per_s"] for r in waves],
                 "o", label="measured operator")
    axes[1].plot([r["k"] for r in waves], [r["gamma_theory_1_per_s"] for r in waves],
                 "-", label=r"theory $Dk^2$, $D=\delta c_0 h$")
    axes[1].set_xlabel("wavenumber k  [1/m]")
    axes[1].set_ylabel("decay rate gamma  [1/s]")
    axes[1].set_title("(b) gamma = D k^2 over one decade in k")
    axes[1].legend()

    trace = payload["finite_time"]["trace"]
    axes[2].plot([p["t"] for p in trace], [p["amplitude"] for p in trace],
                 "o", ms=3, label="numerical (explicit Euler)")
    axes[2].plot([p["t"] for p in trace], [p["theory"] for p in trace],
                 "-", label=r"theory $e^{-\gamma t}$")
    axes[2].set_xlabel("t  [s]")
    axes[2].set_ylabel("mode amplitude a(t)/a(0)")
    axes[2].set_title("(c) amplitude decay over a finite time")
    axes[2].legend()

    figure_.tight_layout()
    figure_.savefig(path, dpi=140)
    plt.close(figure_)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", action="store_true", help="also write decay_rate.png")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch intra-op threads (the calls are dispatch-bound)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    resolution = [rate_row(dx, 0.5, 20) for dx in (0.02, 0.01, 0.005)]
    wavenumber = [rate_row(0.01, lam, 20) for lam in (1.0, 0.5, 0.25)]
    smoothing = [rate_row(0.01, 0.5, 20, h_factor=hf) for hf in (1.3, 1.6, 2.0)]
    finite = finite_time_trace()

    payload = {
        "description": ("plane density wave decay rate vs the analytic "
                        "gamma = D k^2 with D = delta c0 h"),
        "model": {
            "delta": DELTA,
            "sound_speed": SOUND_SPEED,
            "h_factor": H_FACTOR,
            "rho0": RHO0,
            "amplitude": AMPLITUDE,
        },
        "gates": {"rate_relative_error": GATE_RATE,
                  "finite_time_relative_error": GATE_AMPLITUDE},
        "resolution_ladder": resolution,
        "wavenumber_ladder": wavenumber,
        "smoothing_length_ladder": smoothing,
        "finite_time": finite,
        "max_rate_relative_error": max(r["relative_error"]
                                       for r in resolution + wavenumber + smoothing),
        "verdict": ("pass" if all(r["passed"] for r in resolution + wavenumber + smoothing)
                    and finite["passed"] else "FAIL"),
    }
    (OUT_DIR / "decay_rate.json").write_text(json.dumps(payload, indent=2) + "\n")
    if args.figure:
        figure(payload, OUT_DIR / "decay_rate.png")

    print(f"{'dx':>6} {'lambda':>7} {'h/dx':>5} {'k dx':>6} {'gamma_num':>12} "
          f"{'gamma_th':>12} {'error %':>8}")
    for row in resolution + wavenumber + smoothing:
        print(f"{row['dx']:>6.3g} {row['wavelength']:>7.3g} {row['h_over_dx']:>5.1f} "
              f"{row['k_dx']:>6.4f} "
              f"{row['gamma_measured_1_per_s']:>12.5e} "
              f"{row['gamma_theory_1_per_s']:>12.5e} "
              f"{100.0 * row['relative_error']:>8.3f}")
    print(f"finite time: a({finite['steps']} x {finite['dt']:g} s) = "
          f"{finite['final_amplitude']:.6f} vs {finite['final_theory']:.6f} "
          f"({100.0 * finite['relative_error']:.3f} %)")
    print(f"max rate error {100.0 * payload['max_rate_relative_error']:.3f} % "
          f"(gate {100.0 * GATE_RATE:g} %)  verdict={payload['verdict']}")
    return 0 if payload["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
