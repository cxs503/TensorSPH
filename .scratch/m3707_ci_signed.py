"""CI release case (R=0.01, dx=0.001, 1.8 periods) with the signed observable."""
import importlib.util, json, time
from pathlib import Path
import torch
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("osc", ROOT / "benchmarks" / "oscillating_droplet" / "run.py")
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
t0 = time.time()
case = B.run_case(radius=0.01, dx=0.001, sigma=7.2, rho0=1000.0,
                  sound_speed=20.0, viscosity=5e-5, eps=0.10, periods=1.8,
                  sample_every=2)
case["wall_clock_s"] = time.time() - t0
json.dump(case, open(ROOT / ".scratch" / "m3707_ci_signed.json", "w"))
f = case["frequency"]
print("n", case["setup"]["n_particles"], "steps", case["setup"]["steps"], "wall", round(case["wall_clock_s"],1))
print("LEGACY |s|: mom", round(f["fit_second_moment_rel_error_pct"],2), "r2", round(f["fit"]["r2"],4), "lam", round(f["fit"]["lambda_per_s"],2))
print("SIGNED   : fit", round(f["fit_signed_rel_error_pct"],2), "zc", round(f["zero_crossing_signed_rel_error_pct"],2), "r2", round(f["fit_signed"]["r2"],4))
print("ke_half", round(f["fit_kinetic_energy_rel_error_pct"],2), "zc_legacy", round(f["zero_crossing_rel_error_pct"],2))
