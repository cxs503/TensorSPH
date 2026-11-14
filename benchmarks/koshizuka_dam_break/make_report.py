#!/usr/bin/env python3
"""Generate figures for Koshizuka dam-break pressure benchmark."""
import sys
import json
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.koshizuka_dam_break.koshizuka import (
    KoshizukaConfig, run_koshizuka, KOSHIZUKA_REF_PRESSURE_PA
)

# Run a single case for the report (dx=0.014, moderate resolution)
print("Running dx=0.014 case for report...")
cfg = KoshizukaConfig(dx=0.014, steps=4000, output_every=20, adaptive_dt=True)
cfg.dt = cfg.dt_from_courant()
run = run_koshizuka(cfg)

n_particles = int(run.initial_mass / (cfg.rho0 * cfg.dx * cfg.dx))
error_pct = abs(run.peak_pressure - KOSHIZUKA_REF_PRESSURE_PA) / KOSHIZUKA_REF_PRESSURE_PA * 100

print(f"dx={cfg.dx}: P_max={run.peak_pressure:.1f} Pa, t_peak={run.peak_pressure_time:.4f} s")
print(f"  particles={n_particles}, error={error_pct:.1f}%")

# Save results
result = {
    "dx": cfg.dx,
    "particles": n_particles,
    "peak_pressure_pa": run.peak_pressure,
    "peak_pressure_time_s": run.peak_pressure_time,
    "wall_impact_time": run.wall_impact_time,
    "simulated_time": run.simulated_time,
    "error_pct": error_pct,
    "reference_pressure_pa": KOSHIZUKA_REF_PRESSURE_PA,
    "tolerance_pct": 12,
    "passes_tolerance": error_pct < 12,
    "max_speed": run.max_speed,
    "max_density_compression": run.max_density_compression,
    "escaped": run.escaped,
    "pressure_history": list(zip(run.pressure_probe_times, run.pressure_probe))
}

with open("benchmarks/koshizuka_dam_break/result_dx0014.json", "w") as f:
    json.dump(result, f, indent=2)

# Generate pressure time history plot
fig, ax = plt.subplots(figsize=(10, 6))
times = np.array(run.pressure_probe_times)
pressures = np.array(run.pressure_probe)

ax.plot(times, pressures, "b-", linewidth=1.5, label="TensorSPH (WCSPH)")
ax.axhline(y=KOSHIZUKA_REF_PRESSURE_PA, color="r", linestyle="--", linewidth=2,
           label=f"Reference (Koshizuka 1996): {KOSHIZUKA_REF_PRESSURE_PA} Pa")
ax.axhline(y=KOSHIZUKA_REF_PRESSURE_PA * 1.12, color="orange", linestyle=":", linewidth=1,
           label="+12% tolerance")
ax.axhline(y=KOSHIZUKA_REF_PRESSURE_PA * 0.88, color="orange", linestyle=":", linewidth=1,
           label="-12% tolerance")
if run.wall_impact_time:
    ax.axvline(x=run.wall_impact_time, color="gray", linestyle="-.", linewidth=1,
               label=f"Wall impact (t={run.wall_impact_time:.3f} s)")
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Pressure at probe (Pa)", fontsize=12)
ax.set_title(f"Koshizuka & Oka (1996) Dam-Break Pressure Peak\n"
             f"dx={cfg.dx} m, {n_particles} particles, P_max={run.peak_pressure:.0f} Pa "
             f"(error: {error_pct:.1f}%)", fontsize=14)
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, run.simulated_time)
ax.set_ylim(0, max(pressures.max(), KOSHIZUKA_REF_PRESSURE_PA * 1.3))

plt.tight_layout()
plt.savefig("benchmarks/koshizuka_dam_break/pressure_history.png", dpi=150)
plt.savefig("benchmarks/koshizuka_dam_break/pressure_history.pdf")
print("Saved pressure_history.png")

print("\nDone. Results saved to benchmarks/koshizuka_dam_break/")
