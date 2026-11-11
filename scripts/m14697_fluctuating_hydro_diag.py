"""m14697 diagnostic: root-cause of the HEAD-red fluctuating_hydro criteria.

Runs the LJ Langevin fluid (N=64, 3-D, reduced units) through a set of
controlled experiments and dumps machine-readable evidence to
``docs/m14697/fluctuating_hydro_diag.json``.

Experiments
-----------
A  T=0 residual temperature vs timestep  -> is the residual the BAOAB
   half-kick artifact (T_res ~ (dt*|F|/2)^2) or genuine stochastic noise?
B  T=0 seed independence (perfect FCC lattice, N=32) -> the "no fluctuation"
   statement that can actually be falsified: with the noise channel off the
   trajectory must be bitwise independent of the RNG seed.
C  Equipartition ensemble (T=1) -> is the 1.31 ratio seen at N=64/500 steps a
   bias or an undersampling artifact of a 2.5 tau run?
D  Timestep sensitivity at equal physical time (dt halved, steps doubled)
   -> the committed test compares T at t=2.5 vs t=1.25, which is not a
   discretisation test at all.
E  Green-Kubo at T=0 -> reproduce the ZeroDivisionError.

Usage:  PYTHONPATH=src python scripts/m14697_fluctuating_hydro_diag.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

import torch

from tensorsph.fluctuating_hydro import LJFluidConfig, run_lj_fluid

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "m14697")


def _cfg(**kw):
    d = dict(n_particles=64, dimension=3, number_density=0.8, temperature=1.0,
             friction=1.0, timestep=0.005, n_steps=500, seed=42)
    d.update(kw)
    return LJFluidConfig(**d)


def exp_a_t0_dt_ladder():
    rows = []
    for dt in (0.02, 0.01, 0.005, 0.0025, 0.00125):
        r = run_lj_fluid(_cfg(timestep=dt, n_steps=400, temperature=0.0,
                              zero_temperature=True))
        rows.append(dict(timestep=dt, mean_T=r.mean_temperature,
                         std_T=r.std_temperature,
                         mean_ke=r.equipartition_ratio * 0.5 * 3 * 0.0,
                         equipartition_ratio=r.equipartition_ratio))
    # log-log slope of T_res vs dt
    xs = [math.log(r["timestep"]) for r in rows if r["mean_T"] > 0]
    ys = [math.log(r["mean_T"]) for r in rows if r["mean_T"] > 0]
    n = len(xs)
    slope = ((n * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys)) /
             (n * sum(x * x for x in xs) - sum(xs) ** 2)) if n > 1 else float("nan")
    return dict(rows=rows, log_log_slope_T_vs_dt=slope)


def exp_b_t0_seed_independence():
    # N=32 -> 2x2x2 FCC cells = exactly 32 sites: identical initial lattice
    # for every seed, so any seed dependence is pure stochastic noise.
    out = {}
    finals = {}
    for seed in (1, 2, 3):
        # capture positions by monkey-free re-run of the public driver is not
        # possible (positions are not returned), so reconstruct via a short
        # direct integration using the public API pieces.
        from tensorsph.fluctuating_hydro import (initialize_lattice,
                                                 build_neighbor_list,
                                                 compute_lj_forces,
                                                 velocity_verlet_step,
                                                 LJFluidState)
        pos, vel, box = initialize_lattice(32, 3, 0.8, 0.0, seed,
                                           torch.float64, "cpu", True)
        st = LJFluidState(positions=pos, velocities=vel,
                          forces=torch.zeros_like(pos), box=box)
        cfg = _cfg(n_particles=32, seed=seed, temperature=0.0,
                   zero_temperature=True, n_steps=200)
        pairs = build_neighbor_list(st.positions, box, cfg.lj_cutoff)
        st.forces = compute_lj_forces(st.positions, box, cfg.lj_cutoff, pairs)
        for step in range(200):
            velocity_verlet_step(st, cfg, pairs, noise_seed=cfg.seed + step + 1)
        finals[seed] = st.positions.clone()
        out[f"max_abs_v_seed{seed}"] = float(st.velocities.abs().max())
    pairs_27 = [(1, 2), (1, 3), (2, 3)]
    for a, b in pairs_27:
        out[f"positions_seed{a}_identical_to_seed{b}"] = bool(
            torch.equal(finals[a], finals[b]))
        out[f"max_dx_seed{a}_{b}"] = float((finals[a] - finals[b]).abs().max())
    return out


def exp_c_equipartition_ensemble():
    rows = []
    for seed in (1, 2, 3, 4, 5, 6):
        r = run_lj_fluid(_cfg(seed=seed, n_steps=500))
        rows.append(dict(seed=seed, ratio=r.equipartition_ratio,
                         mean_T=r.mean_temperature,
                         n_samples_msd=r.n_samples_msd))
    ratios = [r["ratio"] for r in rows]
    long_rows = []
    for seed in (1, 2):
        r = run_lj_fluid(_cfg(seed=seed, n_steps=4000))
        long_rows.append(dict(seed=seed, ratio=r.equipartition_ratio,
                              mean_T=r.mean_temperature))
    return dict(
        short_n500=rows,
        short_mean=statistics.fmean(ratios),
        short_std=statistics.stdev(ratios),
        short_se=statistics.stdev(ratios) / math.sqrt(len(ratios)),
        long_n4000=long_rows,
        # ideal-gas expectation of the single-run scatter at N=64, d=3
        expected_single_run_rel_std=math.sqrt(2.0 / (3 * 64)),
    )


def exp_d_dt_at_equal_physical_time():
    rows = []
    for seed in (1, 2, 3):
        r1 = run_lj_fluid(_cfg(seed=seed, timestep=0.005, n_steps=500))
        r2 = run_lj_fluid(_cfg(seed=seed, timestep=0.0025, n_steps=1000))
        rows.append(dict(
            seed=seed,
            T_dt005=r1.mean_temperature, T_dt0025=r2.mean_temperature,
            rel_diff_equal_time=abs(r1.mean_temperature - r2.mean_temperature)
            / r1.mean_temperature))
    # the committed criterion (equal step count => different physical time)
    r1 = run_lj_fluid(_cfg(seed=42, timestep=0.005, n_steps=500))
    r2 = run_lj_fluid(_cfg(seed=42, timestep=0.0025, n_steps=500))
    return dict(
        equal_time_rows=rows,
        equal_time_mean_rel_diff=statistics.fmean(
            r["rel_diff_equal_time"] for r in rows),
        committed_criterion_rel_diff=abs(r1.mean_temperature -
                                         r2.mean_temperature) / r1.mean_temperature,
        committed_T_dt005=r1.mean_temperature,
        committed_T_dt0025=r2.mean_temperature)


def exp_e_green_kubo_t0():
    out = {}
    try:
        run_lj_fluid(_cfg(temperature=0.0, zero_temperature=True, n_steps=300))
        out["result"] = "no error"
    except Exception as exc:  # noqa: BLE001 - we want the exact type/message
        out["result"] = f"{type(exc).__name__}: {exc}"
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = {
        "module": "tensorsph.fluctuating_hydro",
        "experiments": {
            "A_t0_residual_vs_dt": exp_a_t0_dt_ladder(),
            "B_t0_seed_independence": exp_b_t0_seed_independence(),
            "C_equipartition_ensemble": exp_c_equipartition_ensemble(),
            "D_dt_sensitivity": exp_d_dt_at_equal_physical_time(),
            "E_green_kubo_t0": exp_e_green_kubo_t0(),
        },
    }
    path = os.path.join(OUT_DIR, "fluctuating_hydro_diag.json")
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(json.dumps(data, indent=2))
    print(f"\nwrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
