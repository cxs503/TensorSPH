"""m3707 probe: reproduce the CI release case and compare the three frequency
estimators; then probe the fit's multi-modality on a synthetic damped cosine."""
import importlib.util
import json
import math
from pathlib import Path

import torch

RUN = Path(__file__).resolve().parents[1] / "benchmarks" / "oscillating_droplet" / "run.py"
spec = importlib.util.spec_from_file_location("osc_run", RUN)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

case = B.run_case(radius=0.01, dx=0.001, sigma=7.2, rho0=1000.0,
                  sound_speed=20.0, viscosity=5e-5, eps=0.10, periods=1.8)
f = case["frequency"]
print("n_particles", case["setup"]["n_particles"], "steps", case["setup"]["steps"])
print("omega_ref", case["reference"]["omega_rad_per_s"])
fit_mom = f["fit_second_moment_rad_per_s"]
fit_mom_err = f["fit_second_moment_rel_error_pct"]
print("fit_mom      ", fit_mom, fit_mom_err)
fit_ke = 0.5 * f["fit_kinetic_energy_rad_per_s"]
fit_ke_err = f["fit_kinetic_energy_rel_error_pct"]
print("fit_ke/2     ", fit_ke, fit_ke_err)
print("zero_crossing", f["zero_crossing_rad_per_s"], f["zero_crossing_rel_error_pct"])
print("fit r2", f["fit"]["r2"], "lambda", f["fit"]["lambda_per_s"])
tr = case["diagnostics"]["trace"]
trace_n = len(tr["t"])
duration = tr["t"][-1]
periods = case["setup"]["periods_simulated"]
print("trace n", trace_n, "duration", duration, "periods", periods)

# synthetic: exact damped cosine at the same sampling, known omega
om_true = case["reference"]["omega_rad_per_s"]
dt = case["setup"]["dt_s"]
n = len(tr["t"])
t = [i * dt for i in range(n)]
for phase in (0.0, math.pi / 4, math.pi / 2):
    s = [0.3 * math.exp(-20.0 * tt) * math.cos(om_true * tt + phase) + 0.5 for tt in t]
    fit = B.damped_cosine_fit(t, s, om_true)
    err = (fit['omega'] - om_true) / om_true * 100
    print(f"synthetic phase={phase:.2f}: omega={fit['omega']:.3f} "
          f"err={err:+.2f}% r2={fit['r2']:.6f} "
          f"lambda={fit['lambda_per_s']:.2f} (true 20)")

# multi-modality of the coarse grid: scan omega with lambda fixed to the fitted value
print("--- objective scan (lambda fixed at fit value) ---")
ts = [x for x in t]
ss = None
scan = []
for om in torch.linspace(0.4 * om_true, 1.6 * om_true, 120):
    env = torch.exp(-torch.tensor(f["fit"]["lambda_per_s"]) * torch.tensor(ts))
    A = torch.stack([torch.ones(len(ts)), env * torch.cos(float(om) * torch.tensor(ts)),
                     env * torch.sin(float(om) * torch.tensor(ts))], dim=1)
    y = torch.tensor(tr["s_second_moment"], dtype=torch.float64)
    coef, *_ = torch.linalg.lstsq(A.reshape(len(ts), 3), y[:, None])
    res = A.reshape(len(ts), 3) @ coef[:, 0] - y
    scan.append((float(om), float(torch.sqrt((res * res).mean()))))
best = min(scan, key=lambda z: z[1])
# local minima
lm = [scan[i] for i in range(1, len(scan) - 1)
      if scan[i][1] <= scan[i - 1][1] and scan[i][1] <= scan[i + 1][1]]
print("best", best, "err%", (best[0] - om_true) / om_true * 100)
print("local minima:", [(round(o, 2), round(r, 6)) for o, r in lm])
json.dump({"scan": scan, "local_minima": lm}, open(Path(__file__).with_suffix(".scan.json"), "w"))
