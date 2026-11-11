"""m14697 diagnostic round 2: disentangle transient bias from statistics.

Round 1 (``fluctuating_hydro_diag.json``) showed
 * the T=0 residual temperature is deterministic (seed-independent, bitwise),
 * the N=64 / 500-step equipartition ratio of the committed test (1.31) is not
   reproduced by the ensemble (mean 1.12 +- 0.03) and vanishes by t=20,
 * the committed dt-sensitivity test compares t=2.5 with t=1.25.

Round 2 repeats the three probes with a controlled protocol:
 A2  T=0 at fixed physical time t=2 (steps scale as 1/dt) -> expect T_res ~ dt^2
 C2  equipartition at n_steps=2000 (t=10) for seeds 1..3 -> transient decayed?
 D2  dt sensitivity at fixed physical time t=5, ensemble mean over 2 seeds
 E2  Green-Kubo at T=0 with a production window long enough to reach the
     k_BT-normalised integral (reproduces the latent ZeroDivisionError).

Usage:  PYTHONPATH=src python scripts/m14697_fluctuating_hydro_diag2.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

from tensorsph.fluctuating_hydro import LJFluidConfig, run_lj_fluid

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "m14697")


def _cfg(**kw):
    d = dict(n_particles=64, dimension=3, number_density=0.8, temperature=1.0,
             friction=1.0, timestep=0.005, n_steps=500, seed=42)
    d.update(kw)
    return LJFluidConfig(**d)


def exp_a2_t0_fixed_time():
    rows = []
    for dt in (0.02, 0.01, 0.005, 0.0025, 0.00125):
        n_steps = int(round(2.0 / dt))
        r = run_lj_fluid(_cfg(timestep=dt, n_steps=n_steps, temperature=0.0,
                              zero_temperature=True))
        rows.append(dict(timestep=dt, n_steps=n_steps, mean_T=r.mean_temperature,
                         std_T=r.std_temperature,
                         max_ke_per_particle=r.max_ke_per_particle))
    xs = [math.log(r["timestep"]) for r in rows]
    ys = [math.log(r["mean_T"]) for r in rows]
    n = len(xs)
    slope = ((n * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys)) /
             (n * sum(x * x for x in xs) - sum(xs) ** 2))
    # normalised by dt^2 to show the plateau
    for r in rows:
        r["T_over_dt2"] = r["mean_T"] / r["timestep"] ** 2
    return dict(rows=rows, log_log_slope_T_vs_dt=slope)


def exp_c2_equipartition_long():
    rows = []
    for seed in (1, 2, 3):
        r = run_lj_fluid(_cfg(seed=seed, n_steps=2000))
        rows.append(dict(seed=seed, ratio=r.equipartition_ratio,
                         mean_T=r.mean_temperature))
    ratios = [r["ratio"] for r in rows]
    return dict(rows=rows, mean=statistics.fmean(ratios),
                stdev=statistics.stdev(ratios))


def exp_d2_dt_fixed_time_ensemble():
    t_end = 5.0
    out = {}
    for dt in (0.005, 0.0025):
        temps = []
        for seed in (1, 2):
            r = run_lj_fluid(_cfg(seed=seed, timestep=dt,
                                  n_steps=int(round(t_end / dt))))
            temps.append(r.mean_temperature)
        out[f"dt{dt}"] = temps
    a = statistics.fmean(out["dt0.005"])
    b = statistics.fmean(out["dt0.0025"])
    out["ensemble_mean_dt005"] = a
    out["ensemble_mean_dt0025"] = b
    out["rel_diff_equal_time_ensemble"] = abs(a - b) / a
    return out


def exp_e2_green_kubo_t0():
    try:
        r = run_lj_fluid(_cfg(temperature=0.0, zero_temperature=True,
                              n_steps=2000))
        return {"result": "no error", "eta_gk": r.viscosity_green_kubo}
    except Exception as exc:  # noqa: BLE001
        return {"result": f"{type(exc).__name__}: {exc}"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = {"experiments": {
        "A2_t0_fixed_time": exp_a2_t0_fixed_time(),
        "C2_equipartition_n2000": exp_c2_equipartition_long(),
        "D2_dt_fixed_time_ensemble": exp_d2_dt_fixed_time_ensemble(),
        "E2_green_kubo_t0_long": exp_e2_green_kubo_t0(),
    }}
    path = os.path.join(OUT_DIR, "fluctuating_hydro_diag2.json")
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(json.dumps(data, indent=2))
    print(f"\nwrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
