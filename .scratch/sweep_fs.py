"""Threshold / window sensitivity of the free-surface pressure closure.

Sweep structure of the P0 closure at the hardest point of the Laplace
verification (dx = 2 mm, c0 = 20 m/s, where the EOS compression floor is the
tightest).  Each entry reports the tail-averaged absolute reading, which is the
acceptance metric of docs/verification_free_surface_pressure.md.

Usage: PYTHONPATH=src:. python .scratch/sweep_fs.py
"""
import json
import time
from pathlib import Path

from benchmarks.laplace_pressure_floor.run import relax

CASES = [
    # (dx, c0, threshold, t_end)   threshold > 1 means "all particles"
    (0.002, 20.0, 0.98, 1.5),
    (0.002, 20.0, 1.00, 1.5),
    (0.002, 20.0, 1.02, 1.5),
    (0.002, 20.0, 0.99, 3.0),
]

rows = []
for dx, c0, thr, t_end in CASES:
    t0 = time.time()
    run = relax(dx, c0, t_end_over_t_sigma=t_end, free_surface_pressure=True,
                free_surface_threshold=thr)
    rows.append({
        "dx_m": dx, "sound_speed": c0, "threshold": thr,
        "t_over_t_sigma": t_end, "steps": run["steps"],
        "tail_rel_error_pct": run["absolute_time_average_tail"]["rel_error_pct"],
        "abs_rel_error_pct": run["definition_absolute"]["rel_error_pct"],
        "rho_core_over_rho0": run["density"]["rho_core_over_rho0"],
        "compression_required": run["density"]["compression_required"],
        "n_free_surface": run["free_surface_pressure"]["n_free_surface_particles"],
        "p_shell_Pa": run["free_surface_pressure"]["p_shell_mean_Pa"],
        "vmax_over_vref": run["spurious_current"]["vmax_over_vref_final"],
        "wall_s": round(time.time() - t0, 1),
    })
    r = rows[-1]
    print(f"dx={dx:g} c0={c0:g} thr={thr:.2f} t_end={t_end:g}: "
          f"tail {r['tail_rel_error_pct']:+7.1f} %  abs {r['abs_rel_error_pct']:+7.1f} %"
          f"  comp {r['rho_core_over_rho0'] - 1.0:+.2e}/{r['compression_required']:.2e}"
          f"  fs {r['n_free_surface']}  pshell {r['p_shell_Pa']:.1f}"
          f"  v {r['vmax_over_vref']:.3f}  {r['wall_s']}s", flush=True)

out = Path("runs/free_surface_pressure_sweep.json")
out.write_text(json.dumps({"sweep": rows}, indent=2))
print("wrote", out)