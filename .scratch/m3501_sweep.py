import importlib.util, sys
spec = importlib.util.spec_from_file_location("lpf", "benchmarks/laplace_pressure_floor/run.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import torch
from tensorsph.droplet_impact import Config
dx, c0, passes, ratio = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
orig = m.Config
class P(orig):
    def __init__(self, **kw):
        kw.setdefault("free_surface_pressure", True)
        kw.setdefault("free_surface_pressure_passes", passes)
        kw.setdefault("free_surface_rho0_ratio", ratio)
        super().__init__(**kw)
m.Config = P
r = m.relax(dx, c0, free_surface_pressure=True)
print(f"dx={dx} c0={c0} passes={passes} ratio={ratio} | "
      f"abs {r['definition_absolute']['rel_error_pct']:+6.1f}% "
      f"tail {r['absolute_time_average_tail']['rel_error_pct']:+6.1f}% "
      f"bandA {r['definition_bandA']['rel_error_pct']:+6.1f}% "
      f"comp {(r['density']['rho_core_over_rho0']-1)/r['density']['compression_required']:.2f} "
      f"vmax {r['spurious_current']['vmax_over_vref_final']:.3f}", flush=True)
