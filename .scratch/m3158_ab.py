"""m3158: A/B the free-surface pressure closure on the Laplace drop."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

BENCH = Path("benchmarks/laplace_pressure_floor")
spec = importlib.util.spec_from_file_location("lpf", BENCH / "run.py")
lpf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lpf)

# patch relax()'s Config through a module flag
import tensorsph.droplet_impact as di

_orig_init = di.Config.__init__


def _patched_init(self, *a, **kw):
    kw.setdefault("free_surface_pressure", FS)
    _orig_init(self, *a, **kw)


di.Config.__init__ = _patched_init

for FS in (False, True):
    for dx, c0 in ((0.002, 20.0), (0.002, 5.0)):
        t0 = time.time()
        run = lpf.relax(dx, c0)
        print(f"fs={FS} dx={dx} c0={c0:g}: abs {run['definition_absolute']['rel_error_pct']:+7.1f}%"
              f"  tail {run['absolute_time_average_tail']['rel_error_pct']:+7.1f}%"
              f"  vmax/vref {run['spurious_current']['vmax_over_vref_final']:.3f}"
              f"  finite {run['finite']}  N={run['n_particles']}  steps={run['steps']}"
              f"  [{time.time()-t0:.0f}s]", flush=True)