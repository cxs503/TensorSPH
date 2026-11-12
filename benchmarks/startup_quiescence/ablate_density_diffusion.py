"""Reproduce density_diffusion_form_ablation.json (resting-flume A/B).

    PYTHONPATH=src python benchmarks/startup_quiescence/ablate_density_diffusion.py

Runs the shipped resting-column probe twice, changing only
``Config.density_diffusion_form``.  Each case takes ~1-3 min on CPU because the
flume probe integrates 0.06 s of acoustic time at dx = 10 mm.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import run_flume_case  # noqa: E402

out = []
for form in ("molteni", "antuono"):
    started = time.perf_counter()
    r = run_flume_case(0.20, 0.01, 0.60, 0.06, 4,
                       {"density_diffusion_form": form})
    out.append({"label": form, "density_diffusion_form": form,
                "v_peak": r["v_peak"],
                "interior_v_late_rms": r["interior_v_late_rms"],
                "initial_pressure_deviation": r["initial_pressure_deviation"],
                "wall_seconds": time.perf_counter() - started})
    print(out[-1], flush=True)
Path(__file__).with_name("density_diffusion_form_ablation.json").write_text(
    json.dumps(out, indent=1) + "\n")
