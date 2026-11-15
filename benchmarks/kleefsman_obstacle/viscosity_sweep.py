"""Kleefsman (2005) obstacle case: does the laminar-viscosity setting explain
the +24 % arrival-time bias?

Background.  The case is scored on the water height just upstream of the box:
peak height and arrival time, both within 10 % of the digitised experiment
(``reference.json`` -> ``experimental_values``).  The published ladder
(docs/benchmarks/kleefsman-validation-m21019.md) reaches 2.0 % on the peak at
dx = 0.01 but converges to an arrival time ~24 % late, and
docs/benchmarks/kleefsman-arrival-attribution-m21173.md showed the bias
survives mesh refinement (GCI 1.3 %) and is not acoustic (c0 ladder: 2 %).

One candidate was never tested: the case inherits ``Config.viscosity = 0.01``
m^2/s, i.e. a kinematic viscosity four orders of magnitude above water.  At
U = 3.3 m/s, L = 1.2 m that is Re = UL/nu = ~400, which puts a
delta = sqrt(nu L / U) = 60 mm boundary layer over the whole bed -- three
particle spacings at dx = 0.02 -- and a Ritter-type inviscid front is not
supposed to apply.  This sweep measures arrival time and peak against nu so the
bias can be attributed to a *modelling* choice instead of left as an
unexplained solver error (or shown to be none of the two).

Usage::

    python benchmarks/kleefsman_obstacle/viscosity_sweep.py \
        --dx 0.04 --viscosities 0.01 0.003 0.001 1e-4 --t-end 1.2

Writes ``results/viscosity_sweep/<tag>/summary.json`` plus one metrics JSON and
one probe time series per viscosity.  Zero hand transcription: every number in
the generated document comes from these files.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.kleefsman_obstacle import KleefsmanConfig, run_case  # noqa: E402

HERE = Path(__file__).resolve().parent


def build_cfg(ref: dict, dx: float, t_end: float, viscosity: float,
              sound_speed: float | None, gauge_aperture: float) -> KleefsmanConfig:
    g = ref["geometry_m"]
    cfg = KleefsmanConfig(
        dx=dx, t_end=t_end,
        tank_length=g["tank_length"], tank_height=g["tank_height"],
        column_length=g["column_length"], column_height=g["column_height"],
        obstacle_x=g["obstacle_upstream_face_x"],
        obstacle_length=g["obstacle_length"],
        obstacle_height=g["obstacle_height"],
        pool_width=g["tank_length"], pool_depth=g["tank_height"],
        device="cpu",
    )
    cfg.viscosity = viscosity
    if sound_speed is not None:
        cfg.sound_speed = sound_speed
    cfg.gauge_aperture = gauge_aperture
    cfg.dt = min(2.0e-4, cfg.dt_acoustic())
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, nargs="+", default=[0.04])
    ap.add_argument("--viscosities", type=float, nargs="+",
                    default=[0.01, 0.003, 0.001, 1e-4])
    ap.add_argument("--t-end", type=float, default=1.2)
    ap.add_argument("--sound-speed", type=float, default=None)
    ap.add_argument("--gauge-aperture", type=float, default=0.08)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--time-budget", type=float, default=None)
    args = ap.parse_args()

    ref = json.loads((HERE / "reference.json").read_text())
    exp = ref["experimental_values"]
    tol = ref["tolerance"]["relative"]
    tag = args.tag or ("dx" + "_".join(f"{d:g}" for d in args.dx))
    outdir = HERE / "results" / "viscosity_sweep" / tag
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dx in args.dx:
        for nu in args.viscosities:
            cfg = build_cfg(ref, dx, args.t_end, nu, args.sound_speed,
                            args.gauge_aperture)
            name = f"dx{dx:g}_nu{nu:g}"
            t0 = time.monotonic()
            run = run_case(cfg, time_budget=args.time_budget)
            wall = time.monotonic() - t0
            # length actually travelled by the arrival, and the bed boundary
            # layer that length implies at the arrival speed
            travel = cfg.probe_x() - cfg.column_length
            arrival = run.arrival_time
            u_mean = travel / arrival if arrival and arrival > 0 else float("nan")
            delta = (math.sqrt(nu * travel / u_mean)
                     if u_mean == u_mean and nu > 0 else 0.0)
            row = {
                "dx": dx, "viscosity": nu, "tag": name,
                "finished": bool(run.finished),
                "t_end": args.t_end, "simulated_time_s": run.simulated_time,
                "steps": run.steps_taken, "dt": cfg.dt,
                "sound_speed": cfg.sound_speed,
                "peak_height_m": run.peak_height, "peak_time_s": run.peak_time,
                "arrival_time_s": arrival,
                "max_speed": run.max_speed,
                "max_density_compression": run.max_density_compression,
                "mass_error_rel": run.mass_error,
                "boundary_escapes": run.escaped,
                "mean_front_speed": u_mean,
                "reynolds": (u_mean * travel / nu) if nu > 0 else float("inf"),
                "boundary_layer_m": delta,
                "boundary_layer_over_dx": delta / dx,
                "wall_clock_s": wall,
                "rel_error": {}, "pass": {},
            }
            for key, got, want in (("peak_height", run.peak_height, exp["peak_height_m"]),
                                   ("arrival_time", arrival, exp["arrival_time_s"])):
                if got is None or want is None or got != got:
                    continue
                err = abs(got - want) / abs(want)
                row["rel_error"][key] = err
                row["pass"][key] = bool(err < tol)
            (outdir / f"{name}.json").write_text(json.dumps(row, indent=1))
            with (outdir / f"{name}_time_series.csv").open("w") as fh:
                fh.write("t_s,height_m,front_x_m\n")
                for t, hh, fr in zip(run.times, run.height, run.front):
                    fh.write(f"{t:.6f},{hh:.6f},{fr:.6f}\n")
            rows.append(row)
            print(f"{name:>18s}  peak={run.peak_height:.4f} m  "
                  f"arr={arrival if arrival is None else round(arrival, 4)} s  "
                  f"vmax={run.max_speed:.2f}  comp={run.max_density_compression:.3f}  "
                  f"esc={run.escaped}  {wall:.1f}s", flush=True)

    summary = {
        "case": "CASE-SPH-KLEEFSMAN-DAMBREAK-OBST",
        "repo": "TensorSPH",
        "purpose": "attribute the arrival-time bias to the laminar-viscosity setting",
        "experiment": {"peak_height_m": exp.get("peak_height_m"),
                       "arrival_time_s": exp.get("arrival_time_s"),
                       "probe_x_m": exp.get("probe_x_m")},
        "tolerance_relative": tol,
        "reference": {"citation": ref["reference"]["citation"],
                      "doi": ref["reference"]["doi"]},
        "runs": rows,
        "monotonic_checks": {},
    }
    # monotonicity of the two scored metrics in nu (inviscid limit is nu -> 0):
    # if the front accelerates as nu falls, viscosity is a cause, not noise.
    for dx in args.dx:
        sub = sorted([r for r in rows if r["dx"] == dx],
                     key=lambda r: -r["viscosity"])
        arr = [r["arrival_time_s"] for r in sub]
        peak = [r["peak_height_m"] for r in sub]
        summary["monotonic_checks"][f"dx{dx:g}"] = {
            "viscosity_desc": [r["viscosity"] for r in sub],
            "arrival_s": arr,
            "peak_m": peak,
            "arrival_monotonically_decreasing_in_1_over_nu": all(
                a is not None and b is not None and b <= a + 1e-12
                for a, b in zip(arr, arr[1:])),
            "arrival_span_s": (max(arr) - min(arr)) if all(a is not None for a in arr) else None,
        }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {outdir/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())