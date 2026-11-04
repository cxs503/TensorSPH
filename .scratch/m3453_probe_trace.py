"""Probe 2: trace shape + cost of one fit + basin landscape."""
import json
import math
import sys
import time

sys.path.insert(0, "benchmarks/oscillating_droplet")
import torch

import run as od

tr = json.load(open("benchmarks/oscillating_droplet/trace_data.json"))["frequency"]
t, s = tr["t"], tr["s_second_moment"]
om_ref = 73.48469228349533
rep = json.load(open("benchmarks/oscillating_droplet/validation_report.json"))["frequency"]
print("initial_state", rep["initial_state"])
print("final_state", rep["final_state"])
print("diagnostics", {k: v for k, v in rep["diagnostics"].items() if k != "trace"})
print("period_ref", 2 * math.pi / om_ref, "t_end", t[-1])
# s(t) coarse history
print("s(t) every 100th sample:")
print([round(s[i], 4) for i in range(0, len(s), 100)])

t0 = time.time()
fit = od.damped_cosine_fit(t, s, om_ref)
print("one full fit took %.2f s" % (time.time() - t0), fit["omega"], fit["rms"], fit["r2"])