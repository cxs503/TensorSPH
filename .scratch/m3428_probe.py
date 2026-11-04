"""m3428 probe: A/B the free-surface pressure closure at one (dx, c0) point.

Usage: PYTHONPATH=src python .scratch/m3428_probe.py [dx] [c0]
"""
import importlib.util
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure_floor" / "run.py"
spec = importlib.util.spec_from_file_location("lpf_run", BENCH)
lpf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lpf)

dx = float(sys.argv[1]) if len(sys.argv) > 1 else 0.002
c0 = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

for mode in ("off", "shell", "shell_density"):
    t0 = time.time()
    run = lpf.relax(dx, c0, free_surface_pressure=mode)
    print(f"mode={mode:14s} c0={c0:g} dx={dx:g} "
          f"abs={run['definition_absolute']['rel_error_pct']:+7.1f}% "
          f"tail={run['absolute_time_average_tail']['rel_error_pct']:+7.1f}% "
          f"bandA={run['definition_bandA']['rel_error_pct']:+7.1f}% "
          f"comp={run['density']['rho_core_over_rho0']-1.0:+.3e}"
          f"/{run['density']['compression_required']:.3e} "
          f"finite={run['finite']} steps={run['steps']} "
          f"vmax/vref={run['spurious_current']['vmax_over_vref_final']:.3f} "
          f"[{time.time()-t0:.1f}s]", flush=True)