#!/usr/bin/env python
"""Benchmark runner for the 3-D dam-break case (IN16-K6).

Uses a moderate resolution suitable for demonstration (~200 particles).
Produces figures and results JSON in artifacts/.
"""
import json
import math
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorsph.dam_break_3d import (
    DamBreak3DConfig,
    run_case,
    ritter_front_over_a,
)

# --------------------------------------------------------------------------- #
# Benchmark configuration (moderate resolution for demonstration)
# --------------------------------------------------------------------------- #

cfg = DamBreak3DConfig(
    dx=0.025,
    pool_width=0.8,
    tank_depth=0.2,
    column_width=0.1,
    column_depth=0.1,
    tank_height=0.4,
    column_height=0.2,
    sound_speed=25.0,
    steps=500,
    output_every=50,
    device="cpu",
    hydrostatic_init=True,
    viscosity=0.001,
    wall_layers=3,
)
cfg.dt = cfg.dt_from_courant()

print(f"Running 3-D dam break: dx={cfg.dx}, dt={cfg.dt:.2e}, steps={cfg.steps}")
n_est = int(cfg.column_width/cfg.dx)*int(cfg.column_height/cfg.dx)*int(cfg.column_depth/cfg.dx)
print(f"  Estimated particles: ~{n_est}")
print(f"  Simulated time: {cfg.steps * cfg.dt:.4f} s")

t0 = time.time()
run = run_case(cfg, seed=12345)
t1 = time.time()
print(f"  Actual particles: {run.final_positions.shape[0]}")
print(f"  Steps taken: {run.steps_taken}")
print(f"  Simulated time: {run.simulated_time:.4f} s")
print(f"  Wall clock: {t1-t0:.1f} s")
print(f"  Front: {run.front[0]:.4f} -> {run.front[-1]:.4f}")
print(f"  Energy drift: {run.energy_drift:.4e}")
print(f"  Max compression: {run.max_density_compression:.4e}")
print(f"  Escaped: {run.escaped}")

# --------------------------------------------------------------------------- #
# Figure 1: Front position vs time
# --------------------------------------------------------------------------- #

tau, za = run.dimensionless(cfg)
tau_ritter = [t for t in tau]
za_ritter = [ritter_front_over_a(t) for t in tau_ritter]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(tau, za, "b-o", markersize=3, label="SPH 3-D", zorder=3)
ax.plot(tau_ritter, za_ritter, "r--", linewidth=2, label="Ritter 2-D reference", zorder=2)
ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$", fontsize=12)
ax.set_ylabel(r"$Z/a$", fontsize=12)
ax.set_title("3-D Dam Break: Leading Edge vs Ritter Reference")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("artifacts/dam_break_3d_front.png", dpi=150)
plt.close()
print("  Saved artifacts/dam_break_3d_front.png")

# --------------------------------------------------------------------------- #
# Figure 2: Energy evolution
# --------------------------------------------------------------------------- #

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(run.times, run.kinetic_energy, "b-", label="Kinetic", zorder=3)
ax.plot(run.times, run.potential_energy, "r-", label="Potential", zorder=3)
total_e = [k + p for k, p in zip(run.kinetic_energy, run.potential_energy)]
ax.plot(run.times, total_e, "k--", label="Total (KE+PE)", zorder=2)
ax.axhline(run.initial_energy, color="gray", linestyle=":", alpha=0.5, label="Initial PE")
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Energy (J)", fontsize=12)
ax.set_title("3-D Dam Break: Energy Evolution")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("artifacts/dam_break_3d_energy.png", dpi=150)
plt.close()
print("  Saved artifacts/dam_break_3d_energy.png")

# --------------------------------------------------------------------------- #
# Figure 3: Particle cloud (mid-plane slice)
# --------------------------------------------------------------------------- #

pos = run.final_positions.detach().cpu()
mid_z = cfg.tank_depth / 2.0
dz = cfg.dx * 1.5
mask = torch.abs(pos[:, 2] - mid_z) < dz
slice_pos = pos[mask]

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(slice_pos[:, 0].numpy(), slice_pos[:, 1].numpy(),
                c=slice_pos[:, 2].numpy(), cmap="viridis", s=8, alpha=0.7)
ax.set_xlabel("x (m)", fontsize=12)
ax.set_ylabel("y (m)", fontsize=12)
ax.set_title(f"3-D Dam Break: Mid-Plane Particle Cloud (t={run.simulated_time:.3f} s)")
ax.set_xlim(-0.05, cfg.pool_width + 0.05)
ax.set_ylim(-0.05, cfg.tank_height + 0.05)
ax.set_aspect("equal")
plt.colorbar(sc, label="z (m)")
plt.tight_layout()
plt.savefig("artifacts/dam_break_3d_particles.png", dpi=150)
plt.close()
print("  Saved artifacts/dam_break_3d_particles.png")

# --------------------------------------------------------------------------- #
# Results JSON
# --------------------------------------------------------------------------- #

results = {
    "case": "3-D dam break (IN16-K6)",
    "config": {
        "dx": cfg.dx,
        "dt": cfg.dt,
        "steps": cfg.steps,
        "column_width": cfg.column_width,
        "column_depth": cfg.column_depth,
        "column_height": cfg.column_height,
        "tank_width": cfg.pool_width,
        "tank_depth": cfg.tank_depth,
        "tank_height": cfg.tank_height,
        "sound_speed": cfg.sound_speed,
        "viscosity": cfg.viscosity,
        "h_factor": cfg.h_factor,
        "seed": 12345,
    },
    "results": {
        "particle_count": int(run.final_positions.shape[0]),
        "steps_taken": run.steps_taken,
        "simulated_time": run.simulated_time,
        "wall_clock_s": t1 - t0,
        "initial_mass": run.initial_mass,
        "final_mass": run.final_mass,
        "mass_drift": run.mass_drift,
        "initial_energy": run.initial_energy,
        "final_energy": run.final_energy,
        "energy_drift": run.energy_drift,
        "max_speed": run.max_speed,
        "max_density_compression": run.max_density_compression,
        "escaped": run.escaped,
        "front_initial": run.front[0],
        "front_final": run.front[-1],
    },
    "front_data": {
        "tau": tau,
        "za": za,
        "tau_ritter": tau_ritter,
        "za_ritter": za_ritter,
    },
    "energy_data": {
        "times": run.times,
        "kinetic": run.kinetic_energy,
        "potential": run.potential_energy,
        "total": total_e,
    },
    "verdict": {
        "mass_conserved": abs(run.mass_drift) < 1e-10,
        "energy_drift_pct": abs(run.energy_drift) * 100,
        "no_escape": run.escaped == 0,
        "bounded_compression": run.max_density_compression < 0.20,
    },
}

with open("artifacts/dam_break_3d_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved artifacts/dam_break_3d_results.json")

print("\nBenchmark complete.")
print(f"  Energy drift: {abs(run.energy_drift)*100:.2f}%")
print(f"  Mass drift: {run.mass_drift:.2e}")
print(f"  Front advanced: {run.front[-1] - run.front[0]:.4f} m")
