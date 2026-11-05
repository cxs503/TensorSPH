"""m3246 probe: does the free-surface pressure closure close the Laplace gap?"""
import importlib.util
import sys
import time
from pathlib import Path
import torch
torch.set_num_threads(4)
ROOT = Path('/root/tp-agent/work/m3246')
spec = importlib.util.spec_from_file_location(
    "lpf", ROOT / "benchmarks/laplace_pressure_floor/run.py")
lpf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lpf)
dx = float(sys.argv[1])
c0 = float(sys.argv[2])
configs = [("off", False, 1), ("on-p1", True, 1), ("on-p3", True, 3), ("on-p6", True, 6)]
for label, flag, passes in configs:
    t0 = time.time()
    r = lpf.relax(dx=dx, c0=c0, free_surface_pressure=flag, free_surface_pressure_passes=passes)
    a = r["absolute_time_average_tail"]
    f = r["definition_absolute"]
    print(f"{label:7s} N={r['n_particles']:5d} steps={r['steps']:4d} "
          f"tail={a['delta_p_Pa']:7.1f}Pa ({a['rel_error_pct']:+6.1f}%) "
          f"final={f['delta_p_Pa']:7.1f}Pa ({f['rel_error_pct']:+6.1f}%) "
          f"vmax/vref={r['spurious_current']['vmax_over_vref_final']:.3f} "
          f"rho_core={r['density']['rho_core_over_rho0']:.6f} "
          f"finite={r['finite']} {time.time()-t0:.0f}s", flush=True)
