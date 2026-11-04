"""Probe: does the shell closure move the Laplace jump? (mission m3289)

Runs a single (dx, c0) point twice, closure off and on, and prints the three
measurement definitions plus the spurious-current gate.
"""
import importlib.util
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure_floor"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load():
    spec = importlib.util.spec_from_file_location("lpf_run", BENCH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import torch
    torch.set_num_threads(2)
    lpf = load()
    dx = float(sys.argv[1]) if len(sys.argv) > 1 else 0.002
    c0 = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    sweeps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    for closure in (False, True):
        t0 = time.time()
        out = lpf.relax(dx, c0, closure=closure, sweeps=sweeps)
        dt = time.time() - t0
        print(f"dx={dx} c0={c0} closure={closure} sweeps={sweeps} "
              f"steps={out['steps']} wall={dt:.1f}s")
        for key in ("absolute_time_average_tail", "definition_absolute",
                    "definition_bandA", "definition_bandB"):
            s = out[key]
            print(f"   {key:28s} dp={s['delta_p_Pa']:9.2f} Pa "
                  f"err={s['rel_error_pct']:7.2f} %")
        print(f"   rho_core={out['density']['rho_core_over_rho0']:.6f} "
              f"comp_req={out['density']['compression_required']:.6f} "
              f"vmax/vref={out['spurious_current']['vmax_over_vref_final']:.4f}")


if __name__ == "__main__":
    main()