"""Probe: does the shell-pressure closure move the Young-Laplace reading?

Usage: PYTHONPATH=src:. python .scratch/probe_fs.py <dx> <c0> <t_end> [--on]
"""
import sys
import time

from benchmarks.laplace_pressure_floor.run import relax

dx = float(sys.argv[1]) if len(sys.argv) > 1 else 0.002
c0 = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
t_end = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
on = "--on" in sys.argv

t0 = time.time()
run = relax(dx, c0, t_end_over_t_sigma=t_end, free_surface_pressure=on)
wall = time.time() - t0
print(f"dx={dx:g} c0={c0:g} closure={on} wall={wall:.1f}s steps={run['steps']}")
print("  abs   %+7.1f %%" % run["definition_absolute"]["rel_error_pct"])
print("  tail  %+7.1f %%" % run["absolute_time_average_tail"]["rel_error_pct"])
print("  bandA %+7.1f %%" % run["definition_bandA"]["rel_error_pct"])
print("  comp %.2e / need %.2e"
      % (run["density"]["rho_core_over_rho0"] - 1.0,
         run["density"]["compression_required"]))
print("  p_shell %.1f Pa  fs_particles %d"
      % (run["free_surface_pressure"]["p_shell_mean_Pa"],
         run["free_surface_pressure"]["n_free_surface_particles"]))
print("  spurious vmax/vref %.3f  finite %s"
      % (run["spurious_current"]["vmax_over_vref_final"], run["finite"]))