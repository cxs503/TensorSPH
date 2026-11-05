"""Measure the mode-2 frequency error of the preserved PR #87 implementation
on the current (main) kernel, for the merge report numbers.

Runs exactly what tests/test_oscillating_droplet.py::TestSolverRegression uses
(radius 0.02 m, dx 2.5 mm, n = 2, eps = 0.05, one analytic period).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent.parent
mod_path = HERE / "benchmarks" / "oscillating_droplet" / "run_m2919.py"
spec = importlib.util.spec_from_file_location("run_m2919", mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

rep = mod.run_case(radius=0.02, dx=0.0025, mode=2, eps=0.05, periods=1.0)
out = {
    "case": "radius=0.02, dx=0.0025, mode=2, eps=0.05, periods=1.0",
    "omega_measured_rad_s": rep["measured"]["omega_rad_per_s"],
    "omega_analytic_rad_s": rep["analytic"]["omega_rad_per_s"],
    "omega_rel_error_pct": rep["measured"]["omega_rel_error_pct"],
    "fit_residual_rel": rep["measured"]["fit_residual_rel"],
    "gamma_measured_1_s": rep["measured"].get("gamma_1_s"),
    "n_particles": rep["case"].get("n_particles"),
    "steps": rep.get("steps"),
    "max_speed": rep.get("max_speed"),
}
print(json.dumps(out, indent=2))
(HERE / ".scratch" / "merge38" / "omega_probe.json").write_text(json.dumps(out, indent=2))
