"""m14697 diagnostic round 3: evidence for the re-stated sweep / dt criteria.

S1  equipartition vs target temperature at the committed 500-step setting
    (T in {0.5, 1.0, 2.0}, two seeds) -> how large is the 2.5-tau transient
    bias the re-stated smoke test has to tolerate?
S2  timestep sensitivity at fixed physical time t=2.5 comparing dt=0.01 with
    dt=0.005 (ensemble of 3 seeds) -> does the re-designed criterion resolve
    discretisation below the 5% claim, or is ensemble noise still dominant?

Usage:  PYTHONPATH=src python scripts/m14697_fluctuating_hydro_diag3.py
"""
from __future__ import annotations

import json
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


def sweep():
    rows = []
    for T in (0.5, 1.0, 2.0):
        for seed in (42, 7):
            r = run_lj_fluid(_cfg(temperature=T, seed=seed, n_steps=500))
            rows.append(dict(target_T=T, seed=seed, ratio=r.equipartition_ratio,
                             mean_T=r.mean_temperature))
    return rows


def dt_pairs():
    out = {}
    temps = {0.01: [], 0.005: []}
    for dt in (0.01, 0.005):
        for seed in (1, 2, 3):
            n = int(round(2.5 / dt))
            r = run_lj_fluid(_cfg(seed=seed, timestep=dt, n_steps=n))
            temps[dt].append(r.mean_temperature)
    out["T_by_dt"] = {str(k): v for k, v in temps.items()}
    out["ensemble_mean_001"] = statistics.fmean(temps[0.01])
    out["ensemble_mean_0005"] = statistics.fmean(temps[0.005])
    out["rel_diff_ensemble"] = abs(out["ensemble_mean_001"] -
                                   out["ensemble_mean_0005"]) / out["ensemble_mean_0005"]
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = {"experiments": {"S1_temperature_sweep_n500": sweep(),
                            "S2_dt_001_vs_0005_fixed_time": dt_pairs()}}
    path = os.path.join(OUT_DIR, "fluctuating_hydro_diag3.json")
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(json.dumps(data, indent=2))
    print(f"\nwrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
