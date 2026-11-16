#!/usr/bin/env python3
"""Plane Poiseuille flow verification case for TensorSPH.

Runs the canonical Poiseuille channel flow at multiple resolutions,
compares against the closed-form analytic solution, and produces
verification artifacts (JSON report + figures).

Usage:
    PYTHONPATH=src:. python examples/poiseuille_channel.py [--resolutions 8 12 16] [--t-end 1.0] [--plot]

Expected output:
    Resolution ladder with centreline error, profile RMS, wall shear error.
    All metrics must be < 5% for PASS verdict.
    With --plot: generates profile comparison and convergence figures.
    With --json: writes artifacts/poiseuille_verification/result.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Ensure the repo root is on the path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from tensorsph.droplet_impact import Config, resolve_device, rhs
from tensorsph.wall_mirror import plane_walls


# --------------------------------------------------------------------------- #
# Physics constants
# --------------------------------------------------------------------------- #

HEIGHT = 0.1          # Channel height (m)
BODY_FORCE = 9.81     # Body force (m/s^2)
VISCOSITY = 0.01      # Kinematic viscosity (m^2/s)
RHO0 = 1000.0         # Density (kg/m^3)
T_END = 1.0           # Total simulation time (s)
T_AVERAGE = 0.2       # Averaging window (s)


def analytic_u(y: float) -> float:
    """Analytic Poiseuille profile: u(y) = g(H-y)y/(2ν)."""
    return BODY_FORCE * (HEIGHT - y) * y / (2.0 * VISCOSITY)


def analytic_u_max() -> float:
    """Analytic maximum velocity at centreline."""
    return BODY_FORCE * HEIGHT**2 / (8.0 * VISCOSITY)


def analytic_wall_shear() -> float:
    """Analytic wall shear stress τ_w = ρ·g·H/2."""
    return RHO0 * BODY_FORCE * HEIGHT / 2.0


# --------------------------------------------------------------------------- #
# SPH simulation
# --------------------------------------------------------------------------- #

def build_fluid(cfg: Config, height: float, length: float, device):
    """Create fluid particles on a square lattice."""
    dx = cfg.dx
    xs = torch.arange(dx * 0.5, length, dx, device=device)
    ys = torch.arange(dx * 0.5, height, dx, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    mass = torch.full((pos.shape[0],), cfg.rho0 * dx * dx, device=device)
    return pos, mass


def periodic_images(pos, length: float):
    """Create periodic copies in x for a channel flow."""
    shift = torch.tensor([length, 0.0], dtype=pos.dtype, device=pos.device)
    return torch.cat((pos, pos + shift, pos - shift), dim=0)


def bin_rows(pos, u_mean, height: float, dx: float):
    """Average velocity onto lattice rows for profile comparison."""
    row = torch.round((pos[:, 1] - dx * 0.5) / dx).to(torch.long)
    n_rows = int(round(height / dx))
    profile = []
    for k in range(n_rows):
        sel = row == k
        if int(sel.sum()) == 0:
            continue
        u = u_mean[sel, 0]
        y_val = float(dx * 0.5 + k * dx)
        profile.append({
            "y": y_val,
            "y_over_H": y_val / height,
            "u_mean": float(u.mean()),
            "u_analytic": analytic_u(y_val),
            "n_particles": int(sel.sum()),
        })
    return profile


def run_single_resolution(resolution: int, device, t_end: float = T_END,
                          t_average: float = T_AVERAGE):
    """Run Poiseuille case at one resolution and return metrics."""
    dx = HEIGHT / resolution
    length = 12 * dx  # 12 columns of particles
    cfg = Config(
        dx=dx,
        viscosity=VISCOSITY,
        sound_speed=20.0,
        rho0=RHO0,
        h_factor=1.3,
        wall_layers=3,
        wall_friction=1.0,
    )

    pos, mass = build_fluid(cfg, HEIGHT, length, device)
    n = pos.shape[0]
    vel = torch.zeros_like(pos)

    h = cfg.h_factor * dx
    support = 2.0 * h
    planes = [(0.0, +1), (HEIGHT, -1)]

    dt_acoustic = 0.3 * h / cfg.sound_speed
    dt_viscous = 0.125 * h * h / cfg.viscosity
    dt = min(dt_acoustic, dt_viscous)
    steps = int(round(t_end / dt))
    avg_from = t_end - t_average

    acc_sum = torch.zeros_like(pos)
    samples = 0

    t0 = time.perf_counter()
    for step in range(steps):
        pos_ext = periodic_images(pos, length)
        vel_ext = torch.cat((vel, vel, vel), dim=0)
        mass_ext = torch.cat((mass, mass, mass), dim=0)

        ghost_pos, ghost_vel, _ = plane_walls(
            pos_ext, vel_ext, mass_ext, planes, support, slip="no_slip"
        )

        _, _, acc = rhs(pos_ext, vel_ext, mass_ext, cfg,
                        wall=ghost_pos, wall_vel=ghost_vel)
        acc = acc[:n].clone()
        acc[:, 0] += BODY_FORCE

        t = step * dt
        if t >= avg_from:
            acc_sum += vel
            samples += 1

        vel = vel + dt * acc
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], length)
    wall_clock = time.perf_counter() - t0

    # Average velocity over the measurement window
    u_mean = acc_sum / max(samples, 1)
    profile = bin_rows(pos, u_mean, HEIGHT, dx)

    # Compute metrics
    u_max_ref = analytic_u_max()
    u_max_meas = max(p["u_mean"] for p in profile) if profile else 0.0
    centreline_err = abs(u_max_meas - u_max_ref) / u_max_ref * 100

    # Profile RMS (interior 10%-90%)
    interior = [p for p in profile if 0.1 <= p["y_over_H"] <= 0.9]
    if interior:
        rms = math.sqrt(
            sum(((p["u_mean"] - p["u_analytic"]) / p["u_analytic"])**2 for p in interior)
            / len(interior)
        ) * 100
    else:
        rms = 0.0

    # Wall shear from parabola fit
    ys = [p["y"] for p in profile]
    us = [p["u_mean"] for p in profile]
    if len(ys) >= 3:
        a = torch.tensor([[1.0, y, y*y] for y in ys], dtype=torch.float64)
        b = torch.tensor(us, dtype=torch.float64)
        coef = torch.linalg.lstsq(a, b).solution
        c1 = float(coef[1])
        tau_fit = RHO0 * VISCOSITY * abs(c1)
        tau_ref = analytic_wall_shear()
        ws_err = (tau_fit / tau_ref - 1.0) * 100
    else:
        ws_err = 0.0

    # Max Mach number
    max_ux = float(vel[:, 0].abs().max())
    max_mach = max_ux / cfg.sound_speed

    return {
        "resolution": resolution,
        "dx": dx,
        "n_fluid": n,
        "dt": dt,
        "steps": steps,
        "wall_clock_s": round(wall_clock, 3),
        "u_max_ref": u_max_ref,
        "u_max_meas": round(u_max_meas, 6),
        "centreline_error_pct": round(centreline_err, 4),
        "profile_rms_interior_pct": round(rms, 4),
        "wall_shear_error_pct": round(ws_err, 4),
        "max_mach": round(max_mach, 6),
        "profile": profile,
    }


# --------------------------------------------------------------------------- #
# Figure generation
# --------------------------------------------------------------------------- #

def make_profile_figure(results, save_path: str):
    """Generate profile comparison figure (measured vs analytic)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5))
    y_fine = np.linspace(0, HEIGHT, 200)
    u_fine = np.array([analytic_u(y) for y in y_fine])
    ax.plot(u_fine, y_fine / HEIGHT, "k-", linewidth=2, label="Analytic")

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    markers = ["o", "s", "^"]
    for idx, res in enumerate(results):
        prof = res["profile"]
        ys = [p["y_over_H"] for p in prof]
        us = [p["u_mean"] for p in prof]
        lbl = f"H/dx={res['resolution']} (err={res['centreline_error_pct']:+.2f}%)"
        ax.plot(us, ys, markers[idx], color=colors[idx], markersize=5,
                label=lbl, alpha=0.8)

    ax.set_xlabel("Velocity u (m/s)", fontsize=12)
    ax.set_ylabel("y / H", fontsize=12)
    ax.set_title("Plane Poiseuille Flow: SPH vs Analytic Profile", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-0.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def make_convergence_figure(results, save_path: str):
    """Generate convergence figure (error vs resolution)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    resolutions = [r["resolution"] for r in results]
    cl_errors = [r["centreline_error_pct"] for r in results]
    rms_errors = [r["profile_rms_interior_pct"] for r in results]
    ws_errors = [abs(r["wall_shear_error_pct"]) for r in results]

    # Left: error vs resolution
    ax = axes[0]
    ax.plot(resolutions, cl_errors, "ro-", label="Centreline error", markersize=8)
    ax.plot(resolutions, rms_errors, "bs-", label="Profile RMS", markersize=8)
    ax.plot(resolutions, ws_errors, "g^-", label="Wall shear error", markersize=8)
    ax.axhline(y=5.0, color="gray", linestyle="--", alpha=0.5, label="5% threshold")
    ax.set_xlabel("Resolution H/dx", fontsize=11)
    ax.set_ylabel("Error (%)", fontsize=11)
    ax.set_title("Convergence: Error vs Resolution", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(resolutions)

    # Right: log-log convergence rate
    ax = axes[1]
    res_arr = np.array(resolutions, dtype=float)
    # Reference second-order line
    ref_slope = cl_errors[0] * (resolutions[0] / res_arr)**2
    ax.plot(res_arr, ref_slope, "k--", alpha=0.4, label="O(Δx²) reference")
    ax.plot(res_arr, cl_errors, "ro-", label="Centreline", markersize=8)
    ax.plot(res_arr, rms_errors, "bs-", label="Profile RMS", markersize=8)
    ax.set_xlabel("Resolution H/dx", fontsize=11)
    ax.set_ylabel("Error (%)", fontsize=11)
    ax.set_title("Convergence Rate (log-log)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Plane Poiseuille flow verification")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[8, 12, 16],
                        help="Resolution ladder (H/dx values)")
    parser.add_argument("--t-end", type=float, default=T_END, help="Total simulation time")
    parser.add_argument("--t-average", type=float, default=T_AVERAGE, help="Averaging window")
    parser.add_argument("--plot", action="store_true", help="Generate figures")
    parser.add_argument("--json", action="store_true", help="Write JSON artifact")
    args = parser.parse_args()

    device = resolve_device("cpu")
    torch.set_num_threads(4)

    print("=" * 60)
    print("  Plane Poiseuille Flow Verification — TensorSPH")
    print("  Mirror/Ghost no-slip wall boundary condition")
    print("=" * 60)
    print()
    print(f"  Physics: H={HEIGHT} m, ν={VISCOSITY} m²/s, g={BODY_FORCE} m/s²")
    print(f"  Analytic u_max = {analytic_u_max():.6f} m/s")
    print(f"  Analytic τ_w   = {analytic_wall_shear():.2f} Pa")
    print(f"  Resolutions: {args.resolutions}")
    print()

    results = []
    all_pass = True

    for res in args.resolutions:
        print(f"  Running H/dx = {res} ...")
        r = run_single_resolution(res, device, args.t_end, args.t_average)
        results.append(r)

        pass_cl = r["centreline_error_pct"] <= 5.0
        pass_rms = r["profile_rms_interior_pct"] <= 5.0
        pass_ws = abs(r["wall_shear_error_pct"]) <= 5.0
        verdict = "PASS" if (pass_cl and pass_rms and pass_ws) else "FAIL"
        if verdict == "FAIL":
            all_pass = False

        print(f"    N_fluid={r['n_fluid']}, dt={r['dt']:.6f}, steps={r['steps']}")
        print(f"    u_max = {r['u_max_meas']:.6f} m/s  (ref: {r['u_max_ref']:.6f})")
        print(f"    Centreline error = {r['centreline_error_pct']:+.4f}%  {'✓' if pass_cl else '✗'}")
        print(f"    Profile RMS      = {r['profile_rms_interior_pct']:.4f}%  {'✓' if pass_rms else '✗'}")
        print(f"    Wall shear error = {r['wall_shear_error_pct']:+.4f}%  {'✓' if pass_ws else '✗'}")
        print(f"    Max Mach         = {r['max_mach']:.4f}")
        print(f"    Wall clock       = {r['wall_clock_s']:.1f} s")
        print(f"    Verdict: {verdict}")
        print()

    # Summary
    print("-" * 60)
    print("  Summary")
    print("-" * 60)
    print(f"  {'Resolution':>12}  {'Centreline%':>12}  {'RMS%':>8}  {'WallShear%':>11}  {'Verdict':>7}")
    for r in results:
        v = "PASS" if (r["centreline_error_pct"] <= 5.0
                        and r["profile_rms_interior_pct"] <= 5.0
                        and abs(r["wall_shear_error_pct"]) <= 5.0) else "FAIL"
        print(f"  H/dx={r['resolution']:>6}  {r['centreline_error_pct']:>+11.4f}  "
              f"{r['profile_rms_interior_pct']:>7.4f}  {r['wall_shear_error_pct']:>+10.4f}  {v:>7}")
    print()
    print(f"  Overall verdict: {'PASS' if all_pass else 'FAIL'} (all metrics within 5%)")
    print()

    # Generate figures
    if args.plot:
        out_dir = REPO / "artifacts" / "poiseuille_verification"
        out_dir.mkdir(parents=True, exist_ok=True)
        make_profile_figure(results, str(out_dir / "poiseuille_profile.png"))
        make_convergence_figure(results, str(out_dir / "poiseuille_convergence.png"))
        print(f"  Figures saved to {out_dir}/")

    # Write JSON artifact
    if args.json:
        out_dir = REPO / "artifacts" / "poiseuille_verification"
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "case": "CASE-POISEUILLE-SPH",
            "title": "算例池·泊肃叶流",
            "solver": "TensorSPH WCSPH (Morris Laplacian + Mirror no-slip wall)",
            "physics": {
                "height_m": HEIGHT,
                "viscosity_m2s": VISCOSITY,
                "density_kgm3": RHO0,
                "body_force_m2s": BODY_FORCE,
                "analytic_u_max": analytic_u_max(),
                "analytic_wall_shear_pa": analytic_wall_shear(),
            },
            "reference": [
                "White, Viscous Fluid Flow 3rd ed. §3-2.2",
                "Batchelor 1967 §4.2",
            ],
            "acceptance_threshold_pct": 5.0,
            "results": [
                {
                    "resolution": r["resolution"],
                    "dx": r["dx"],
                    "n_fluid": r["n_fluid"],
                    "dt": r["dt"],
                    "steps": r["steps"],
                    "wall_clock_s": r["wall_clock_s"],
                    "u_max_meas": r["u_max_meas"],
                    "u_max_ref": r["u_max_ref"],
                    "centreline_error_pct": r["centreline_error_pct"],
                    "profile_rms_interior_pct": r["profile_rms_interior_pct"],
                    "wall_shear_error_pct": r["wall_shear_error_pct"],
                    "max_mach": r["max_mach"],
                    "verdict": "PASS" if (
                        r["centreline_error_pct"] <= 5.0
                        and r["profile_rms_interior_pct"] <= 5.0
                        and abs(r["wall_shear_error_pct"]) <= 5.0
                    ) else "FAIL",
                }
                for r in results
            ],
            "overall_verdict": "PASS" if all_pass else "FAIL",
        }
        json_path = out_dir / "result.json"
        with open(json_path, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"  JSON artifact saved to {json_path}")

    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
