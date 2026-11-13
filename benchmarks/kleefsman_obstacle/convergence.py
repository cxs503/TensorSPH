"""Resolution study of the Kleefsman (2005) obstacle case.

Runs the same physical case at several ``dx`` and reports the probe diagnostics
side by side, so the case is judged on **resolution consistency** (the only
criterion that is decidable without the experimental traces, which stay
WITHHELD -- see ``reference.json``).  The driver is shard-friendly::

    python benchmarks/kleefsman_obstacle/convergence.py --shard 1 --shards 3
    python benchmarks/kleefsman_obstacle/convergence.py --shard 1 --shards 3 --promote

``--shard i --shards n`` runs ``RESOLUTIONS[i-1::n]``.  Results are written to
``benchmarks/kleefsman_obstacle/results/`` -- a **committed** path, unlike
``artifacts/`` which ``.gitignore`` strips, so the numbers survive the merge
gate (the m16070 round lost exactly this way).  ``--promote`` additionally
copies the per-run files into ``artifacts/<tag>/`` for figure tooling.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.kleefsman_obstacle import (  # noqa: E402
    KleefsmanConfig,
    run_case,
    validate_case,
)
from convergence_analysis import check_protocol  # noqa: E402

HERE = Path(__file__).resolve().parent
RESOLUTIONS = (0.04, 0.02, 0.01, 0.005)
T_END = 1.2   # long enough for the surge to reach and pass the box


def driver_config(dx: float, sound_speed: float, t_end: float, geom: dict) -> KleefsmanConfig:
    """Case config with a Courant-consistent dt for *this* sound speed."""
    cfg = KleefsmanConfig(
        dx=dx, t_end=t_end, device="cpu",
        tank_length=geom["tank_length"], tank_height=geom["tank_height"],
        column_length=geom["column_length"], column_height=geom["column_height"],
        obstacle_x=geom["obstacle_upstream_face_x"],
        obstacle_length=geom["obstacle_length"], obstacle_height=geom["obstacle_height"],
        pool_width=geom["tank_length"], pool_depth=geom["tank_height"],
        sound_speed=sound_speed,
    )
    # The old literal 20.0 made c0 > 78 m/s silently exceed the acoustic
    # Courant limit at dx = 0.04 m (defect class fixed in run.py).
    cfg.dt = min(2.0e-4, cfg.dt_acoustic())
    validate_case(cfg)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--t-end", type=float, default=T_END)
    ap.add_argument("--outdir", default=str(HERE / "results"))
    ap.add_argument("--resolutions", default=None,
                    help="comma separated dx ladder, coarse -> fine "
                         "(default: %s)" % ",".join(f"{d:g}" for d in RESOLUTIONS))
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--sound-speed", type=float, default=20.0,
                    help="artificial sound speed c0 [m/s]; dt follows the CFL of *this* c0")
    args = ap.parse_args()

    ref = json.loads((HERE / "reference.json").read_text())
    g = ref["geometry_m"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ladder = (tuple(float(v) for v in args.resolutions.split(","))
              if args.resolutions else RESOLUTIONS)
    todo = list(ladder)[args.shard - 1::args.shards]

    rows = []
    for dx in todo:
        cfg = KleefsmanConfig(
            dx=dx, t_end=args.t_end, device="cpu",
            tank_length=g["tank_length"], tank_height=g["tank_height"],
            column_length=g["column_length"], column_height=g["column_height"],
            obstacle_x=g["obstacle_upstream_face_x"],
            obstacle_length=g["obstacle_length"],
            obstacle_height=g["obstacle_height"],
            pool_width=g["tank_length"], pool_depth=g["tank_height"],
            sound_speed=args.sound_speed,
        )
        # Courant-consistent dt for the selected c0 (the old literal 20.0 made
        # c0 > 78 m/s silently exceed the acoustic limit at dx = 0.04 m).
        cfg.dt = min(2.0e-4, cfg.dt_acoustic())
        validate_case(cfg)
        run = run_case(cfg)
        tag = f"dx{dx:g}"
        n_fluid = math.floor(cfg.column_length / dx) * math.floor(cfg.column_height / dx)
        row = {
            "tag": tag, "dx": dx, "t_end": args.t_end, "particles_fluid": n_fluid,
            "steps": run.steps_taken,
            "peak_height_m": run.peak_height, "peak_time_s": run.peak_time,
            "column_peak_m": (max(run.column) if run.column else float("nan")),
            "arrival_time_s": run.arrival_time,
            "mass_error_rel": run.mass_error, "boundary_escapes": run.escaped,
            "max_density_compression": run.max_density_compression,
            "max_speed": run.max_speed, "sound_speed": cfg.sound_speed, "dt": cfg.dt,
            # protocol fingerprint -- see convergence_analysis.check_protocol
            "probe_offset": cfg.probe_offset, "probe_halfwidth": cfg.probe_halfwidth,
            "surface_particles": cfg.surface_particles,
            "surface_gap": cfg.surface_gap,
            "arrival_threshold": cfg.arrival_threshold,
            "h_factor": cfg.h_factor,
            "height_capped_by_tank": bool(run.peak_height >= 0.995 * cfg.tank_height),
        }
        rows.append(row)
        (outdir / f"{tag}.json").write_text(json.dumps(row, indent=2))
        with (outdir / f"{tag}_time_series.csv").open("w") as fh:
            fh.write("t_s,height_m,front_x_m,column_level_m\n")
            cols = run.column if len(run.column) == len(run.times) else [float("nan")] * len(run.times)
            for t, hh, fr, cl in zip(run.times, run.height, run.front, cols):
                fh.write(f"{t:.6f},{hh:.6f},{fr:.6f},{cl:.6f}\n")
        (outdir / f"shard{args.shard}.json").write_text(json.dumps(rows, indent=2))
        study = outdir / "resolution_study.json"
        merged = json.loads(study.read_text()) if study.exists() else {"t_end": args.t_end,
                                                                      "runs": []}
        merged.setdefault("runs_stale_protocol", [])
        others = [r for r in merged["runs"] if r["dx"] != dx]
        # The freshly computed row defines the protocol.  A rung produced with a
        # different t_end / sound speed / probe definition is NOT part of this
        # ladder: mixing them (dx=0.01 at t_end=1.0 next to dx=0.02 at 1.2) makes
        # the residual differences meaningless, so it is quarantined instead.
        keep = [row]
        for cand in sorted(others, key=lambda r: -r["dx"]):
            diff = check_protocol([row, cand])
            if diff:
                stale = dict(cand)
                stale["quarantine_reason"] = "protocol fingerprint differs: " + json.dumps(diff)
                merged["runs_stale_protocol"] = [
                    s for s in merged["runs_stale_protocol"] if s.get("dx") != cand["dx"]] + [stale]
                print("QUARANTINED", cand.get("tag"), stale["quarantine_reason"], flush=True)
            else:
                keep.append(cand)
        merged["runs"] = sorted(keep, key=lambda r: -r["dx"])
        merged["protocol_mismatch"] = check_protocol(merged["runs"])
        merged["t_end"] = merged["runs"][0].get("t_end") if merged["runs"] else args.t_end
        study.write_text(json.dumps(merged, indent=2))
        print(json.dumps(row), flush=True)
        if args.promote:
            dest = HERE / "artifacts" / tag
            dest.mkdir(parents=True, exist_ok=True)
            for name in (f"{tag}.json", f"{tag}_time_series.csv"):
                shutil.copy2(outdir / name, dest / name.replace(f"{tag}_", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())