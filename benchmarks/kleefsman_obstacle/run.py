"""Run the Kleefsman (2005) dam-break-with-obstacle case and score it.

Usage (from the repo root)::

    python benchmarks/kleefsman_obstacle/run.py --dx 0.01 --t-end 3.0
    python benchmarks/kleefsman_obstacle/run.py --dx 0.005 --t-end 3.0 --tag fine

Writes ``artifacts/kleefsman_obstacle/<tag>/`` with the probe time series
(CSV), the metrics (JSON) and, with ``--figure``, a water-height plot.  The
verdict against Kleefsman et al. (2005) is computed **only** when
``reference.json`` carries numeric experimental values; until then it is
reported as WITHHELD (see the reason recorded there) and the process exits 0.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.kleefsman_obstacle import KleefsmanConfig, run_case  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.01)
    ap.add_argument("--t-end", type=float, default=3.0)
    ap.add_argument("--probe-offset", type=float, default=0.05)
    ap.add_argument("--arrival-threshold", type=float, default=0.02)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ref = json.loads((HERE / "reference.json").read_text())
    g = ref["geometry_m"]
    cfg = KleefsmanConfig(
        dx=args.dx, t_end=args.t_end, probe_offset=args.probe_offset,
        arrival_threshold=args.arrival_threshold, device=args.device,
        tank_length=g["tank_length"], tank_height=g["tank_height"],
        column_length=g["column_length"], column_height=g["column_height"],
        obstacle_x=g["obstacle_upstream_face_x"],
        obstacle_length=g["obstacle_length"],
        obstacle_height=g["obstacle_height"],
        pool_width=g["tank_length"], pool_depth=g["tank_height"],
        dt=min(2.0e-4, 0.3 * 1.3 * args.dx / 20.0),
    )
    tag = args.tag or f"dx{args.dx:g}"
    outdir = Path(args.outdir) if args.outdir else HERE / "artifacts" / tag
    outdir.mkdir(parents=True, exist_ok=True)

    run = run_case(cfg)

    metrics = {
        "tag": tag, "dx": cfg.dx, "t_end": cfg.t_end,
        "probe_x": cfg.probe_x(), "arrival_threshold": cfg.arrival_threshold,
        "peak_height_m": run.peak_height, "peak_time_s": run.peak_time,
        "arrival_time_s": run.arrival_time,
        "steps": run.steps_taken, "simulated_time_s": run.simulated_time,
        "dt_min": run.dt_min, "dt_max": run.dt_max,
        "max_density_compression": run.max_density_compression,
        "max_speed": run.max_speed, "mass_error_rel": run.mass_error,
        "boundary_escapes": run.escaped,
        "particles_fluid": int(math.floor(cfg.column_length / cfg.dx)
                               * math.floor(cfg.column_height / cfg.dx)),
    }
    with (outdir / "time_series.csv").open("w") as fh:
        fh.write("t_s,height_m,front_x_m\n")
        for t, hh, fr in zip(run.times, run.height, run.front):
            fh.write(f"{t:.6f},{hh:.6f},{fr:.6f}\n")

    exp = ref["experimental_values"]
    tol = ref["tolerance"]["relative"]
    verdict, rows = "WITHHELD", []
    if exp.get("peak_height_m") is not None:
        for name, got, want, unit in (
                ("peak_height", run.peak_height, exp["peak_height_m"], "m"),
                ("arrival_time", run.arrival_time, exp["arrival_time_s"], "s")):
            err = abs(got - want) / abs(want)
            rows.append({"metric": name, "ref": want, "got": got, "unit": unit,
                         "rel_error": err, "pass": bool(err < tol)})
        verdict = "PASS" if all(r["pass"] for r in rows) else "FAIL"
    metrics["comparison"] = {"verdict": verdict, "tolerance": tol, "rows": rows,
                             "experimental_status": exp["status"]}

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = torch.tensor(run.times)
        hh = torch.tensor(run.height)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, hh, lw=1.4, label=f"TensorSPH WCSPH (dx={cfg.dx:g} m)")
        ax.axhline(cfg.arrival_threshold, color="0.6", ls=":", lw=0.8,
                   label=f"arrival threshold {cfg.arrival_threshold:g} m")
        ax.axvline(run.arrival_time, color="C1", ls="--", lw=0.9,
                   label=f"arrival t={run.arrival_time:.3f} s")
        ax.plot([run.peak_time], [run.peak_height], "o", ms=4, color="C3",
                label=f"peak {run.peak_height:.3f} m @ {run.peak_time:.3f} s")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("water height at probe [m]")
        ax.set_title(f"Kleefsman dam-break-with-obstacle, probe at x={cfg.probe_x():.3f} m")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "water_height.png", dpi=140)
        print("figure ->", outdir / "water_height.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
