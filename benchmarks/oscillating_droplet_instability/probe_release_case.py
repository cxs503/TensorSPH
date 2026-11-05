"""Re-run the CI release case and keep its raw trace, to test whether it diverges.

Background: ``tests/test_oscillating_droplet.py`` pins the mode-2 frequency of a
cheap release case (R = 1 cm, dx = 1 mm, 1.8 periods).  The evaluation of that
case is recorded there as -31.78 % on loong64 and +8.64 % on x86_64 CI for the
same code, which was booked as "architecture-fragile fit".  The recorded
reference-case trace (``benchmarks/oscillating_droplet/trace_data.json``) shows
kinetic energy growing from 7.6e-5 J to 5.8e7 J over 1.8 periods, i.e. the
reference case diverges; if the release case does too, the pinned number is a
fit over a blow-up and not a physical frequency at all.

This probe stores the raw (t, s, KE) trace of the release case so the claim can
be checked offline, without another 5-minute run.

Usage:  python benchmarks/oscillating_droplet_instability/probe_release_case.py
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCH = ROOT / "benchmarks" / "oscillating_droplet"
OUT = HERE / "release_case_trace.json"


def _bench():
    spec = importlib.util.spec_from_file_location("oscillating_run", BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    bench = _bench()
    t0 = time.time()
    case = bench.run_case(radius=0.01, dx=0.001, sigma=7.2, rho0=1000.0,
                          sound_speed=20.0, viscosity=5e-5, eps=0.10,
                          periods=1.8, sample_every=5)
    trace = case["diagnostics"]["trace"]
    ke = trace["kinetic_energy_J"]
    payload = {
        "probe": "oscillating_droplet_instability/release_case",
        "wall_clock_s": time.time() - t0,
        "setup": case["setup"],
        "reference": case["reference"],
        "frequency": {k: v for k, v in case["frequency"].items()
                      if not isinstance(v, dict)},
        "final_state": case["final_state"],
        "kinetic_energy": {"first_J": ke[0], "min_J": min(ke),
                           "max_J": max(ke), "last_J": ke[-1],
                           "growth_ratio": max(ke) / ke[0]},
        "trace": trace,
    }
    OUT.write_text(json.dumps(payload))
    print(json.dumps({k: payload[k] for k in
                      ("wall_clock_s", "kinetic_energy", "final_state")}, indent=1))
    print("steps", case["setup"]["steps"], "N", case["setup"]["n_particles"],
          "fit_err_pct", case["frequency"]["fit_second_moment_rel_error_pct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
