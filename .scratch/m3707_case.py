import importlib.util, json, sys
from pathlib import Path
RUN = Path(__file__).resolve().parents[1] / "benchmarks" / "oscillating_droplet" / "run.py"
spec = importlib.util.spec_from_file_location("osc_run", RUN)
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
case = B.run_case(radius=0.01, dx=0.001, sigma=7.2, rho0=1000.0,
                  sound_speed=20.0, viscosity=5e-5, eps=0.10, periods=1.8)
json.dump(case, open(Path(__file__).with_suffix(".json"), "w"))
f = case["frequency"]
print("n", case["setup"]["n_particles"], "steps", case["setup"]["steps"])
print("mom", f["fit_second_moment_rel_error_pct"], "ke/2", f["fit_kinetic_energy_rel_error_pct"])
print("zc", f["zero_crossing_rel_error_pct"], "agreement", f["estimator_agreement_pct"])
print("om_ref", case["reference"]["omega_rad_per_s"])
