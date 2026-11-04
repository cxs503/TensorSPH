import importlib.util, json, sys, time
from pathlib import Path
spec = importlib.util.spec_from_file_location("lpf", "benchmarks/laplace_pressure_floor/run.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for fs in (False, True):
    t0 = time.time()
    r = m.relax(0.002, 20.0, free_surface_pressure=fs, t_end_over_t_sigma=1.5)
    print(f"fs={fs}: abs {r['definition_absolute']['rel_error_pct']:+7.1f}% "
          f"tail {r['absolute_time_average_tail']['rel_error_pct']:+7.1f}% "
          f"bandA {r['definition_bandA']['rel_error_pct']:+7.1f}% "
          f"comp {r['density']['rho_core_over_rho0']-1:+.2e} "
          f"vmax/vref {r['spurious_current']['vmax_over_vref_final']:.3f} "
          f"steps {r['steps']} {time.time()-t0:.0f}s", flush=True)
