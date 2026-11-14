#!/usr/bin/env python3
"""Resolution study for Koshizuka dam-break pressure benchmark."""
import sys
import time
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from benchmarks.koshizuka_dam_break.koshizuka import KoshizukaConfig, run_koshizuka

for dx in [0.02, 0.014, 0.01]:
    cfg = KoshizukaConfig(dx=dx, steps=4000, output_every=50, adaptive_dt=True)
    cfg.dt = cfg.dt_from_courant()
    t0 = time.time()
    run = run_koshizuka(cfg)
    elapsed = time.time() - t0
    n_particles = int(run.initial_mass / (cfg.rho0 * cfg.dx * cfg.dx))
    print(f"dx={dx:.4f}: P_max={run.peak_pressure:.1f} Pa, t_peak={run.peak_pressure_time:.4f} s, "
          f"particles={n_particles}, wall_clock={elapsed:.1f} s, escaped={run.escaped}")
