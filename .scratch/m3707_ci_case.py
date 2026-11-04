"""Reproduce the CI release case (R=0.01, dx=0.001, 1.8 periods) with the
PRE-PATCH estimator, and store the raw trace for offline instrument analysis."""
import importlib.util, json, time
from pathlib import Path
import torch
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("osc_legacy", ROOT / ".scratch" / "run.py.bak")
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
t0 = time.time()
case = B.run_case(radius=0.01, dx=0.001, sigma=7.2, rho0=1000.0,
                  sound_speed=20.0, viscosity=5e-5, eps=0.10, periods=1.8,
                  sample_every=2)
case["wall_clock_s"] = time.time() - t0
json.dump(case, open(ROOT / ".scratch" / "m3707_ci_trace.json", "w"))
f = case["frequency"]
print("n", case["setup"]["n_particles"], "steps", case["setup"]["steps"], "wall", round(case["wall_clock_s"],1))
print("mom_pct", f["fit_second_moment_rel_error_pct"], "ke_half_pct", f["fit_kinetic_energy_rel_error_pct"])
print("zc_pct", f["zero_crossing_rel_error_pct"], "agreement", f["estimator_agreement_pct"])
print("r2 mom", f["fit"]["r2"], "lambda", f["fit"]["lambda_per_s"], "amp", f["fit"]["amplitude0"])
print("r2 ke", f["fit_kinetic_energy"]["r2"])
