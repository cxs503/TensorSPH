import importlib.util, json, time
from pathlib import Path
import torch
torch.set_num_threads(1)
RUN = Path(__file__).resolve().parents[1] / "benchmarks" / "oscillating_droplet" / "run.py"
spec = importlib.util.spec_from_file_location("osc_run", RUN)
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
t0 = time.time()
case = B.run_case(radius=0.01, dx=0.0015, sigma=7.2, rho0=1000.0,
                  sound_speed=20.0, viscosity=5e-5, eps=0.10, periods=1.8,
                  sample_every=4)
case["wall_clock_s"] = time.time() - t0
json.dump(case, open(Path(__file__).with_suffix(".json"), "w"))
f = case["frequency"]
print("n", case["setup"]["n_particles"], "steps", case["setup"]["steps"], "wall", round(case["wall_clock_s"]))
print("mom_pct", f["fit_second_moment_rel_error_pct"], "ke_half_pct", f["fit_kinetic_energy_rel_error_pct"])
print("zc_pct", f["zero_crossing_rel_error_pct"], "agreement", f["estimator_agreement_pct"])
print("ambiguity_gap", f["fit"].get("ambiguity_rms_gap"), "basins", f["fit"].get("starts"))
print("alts", f["fit"].get("omega_alternatives"))
