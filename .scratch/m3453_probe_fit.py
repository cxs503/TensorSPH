"""Probe: reproduce the recorded mode-2 fit from the stored trace (read-only)."""
import json
import math
import sys

sys.path.insert(0, "benchmarks/oscillating_droplet")
import torch

import run as od

tr = json.load(open("benchmarks/oscillating_droplet/trace_data.json"))["frequency"]
t, s = tr["t"], tr["s_second_moment"]
rep = json.load(open("benchmarks/oscillating_droplet/validation_report.json"))
case = rep["frequency"]
om_ref = None
for k in ("reference",):
    if isinstance(case, dict) and k in case:
        om_ref = case[k]["omega_rad_per_s"]
print("n samples", len(t), "t_end", t[-1], "omega_ref", om_ref)
print("recorded fit", case["frequency"]["fit_second_moment_rad_per_s"],
      case["frequency"]["fit_second_moment_rel_error_pct"])

fit = od.damped_cosine_fit(t, s, om_ref)
print("reproduced fit", fit["omega"], (fit["omega"] - om_ref) / om_ref * 100.0,
      "rms", fit["rms"], "lam", fit["lambda_per_s"], "r2", fit["r2"])

# signal roughness: how noisy is s(t)?
import statistics
d = [s[i + 1] - s[i] for i in range(len(s) - 1)]
print("s range", min(s), max(s), "mean|ds|", sum(abs(x) for x in d) / len(d))
# dominant period from a plain FFT to see where the energy is
ts = torch.tensor(t, dtype=torch.float64)
ss = torch.tensor(s, dtype=torch.float64)
y = (ss - ss.mean()) * torch.hann_window(len(t), dtype=torch.float64)
sp = torch.fft.rfft(y).abs()
fr = torch.fft.rfftfreq(len(t), d=float(ts[1] - ts[0]))
k = int(torch.argmax(sp[1:]) + 1)
print("fft peak omega", float(2 * math.pi * fr[k]), "rel err %",
      float(2 * math.pi * fr[k] - om_ref) / om_ref * 100.0)
top = torch.topk(sp[1:], 6)
print("top fft peaks (omega, rel%):",
      [(round(float(2 * math.pi * fr[int(i) + 1]), 2),
        round(float((2 * math.pi * fr[int(i) + 1] - om_ref) / om_ref * 100), 1))
       for i in top.indices])