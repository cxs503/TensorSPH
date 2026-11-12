"""Sound-speed (c0) sensitivity ladder for the Kleefsman (2005) obstacle case.

Why this exists
---------------
``docs/verification_kleefsman_resolution_m16452.md`` recorded two facts:

* the probe water-height peak is **not** a monotone function of dx
  (0.425 m @ dx=0.04, 0.518 m @ dx=0.02, >=0.377 m @ dx=0.01 and still rising),
  so it is not a convergent sequence and no order can be quoted;
* the impact region carries **12-14 %** density compression, well above the 5 %
  controllability line, with c0 = 20 m/s.

That document named the next step explicitly: "restart from c0 / delta-SPH, not
from post-processing".  This driver executes the c0 half of that recommendation
as a parameter scan: for a fixed grid it runs the same case at several
artificial sound speeds and records how the compression and the peak respond.

What is measured (all read straight out of ``run_case``, nothing hand-typed):

* ``max_density_compression`` -- max over the run of ``rho/rho0 - 1``;
* ``peak_height_m`` / ``peak_time_s`` -- probe surface height (contiguous-stack
  estimator, ``robust_surface=True``);
* ``arrival_time_s`` -- first time the probe surface crosses ``arrival_threshold``;
* ``max_speed``, ``steps``, ``dt_min``/``dt_max`` to document the CFL actually used.

Pre-declared diagnostics (written before the scan was run)
----------------------------------------------------------
``compressibility_ratio`` = measured compression divided by the weakly
compressible estimate ``(max_speed / c0)**2``.  If the compression is truly the
acoustic response of the WCSPH scheme, this ratio stays O(1) and the measured
compression falls like ``c0**-2``.  The scan is a *falsification* test of that
picture, not a tuning exercise.

Usage
-----
    # one shard of the scan (idempotent: finished cases are skipped)
    python benchmarks/kleefsman_obstacle/c0_sensitivity.py --shard 0/6
    # aggregate the completed shards into one summary
    python benchmarks/kleefsman_obstacle/c0_sensitivity.py --aggregate

The script never modifies the solver; it only selects ``sound_speed`` and the
Courant-consistent ``dt`` on top of :class:`KleefsmanConfig`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.kleefsman_obstacle import (  # noqa: E402
    KleefsmanConfig,
    run_case,
    validate_case,
)

HERE = Path(__file__).resolve().parent
REF = json.loads((HERE / "reference.json").read_text())
GEOM = REF["geometry_m"]

#: Grids and sound speeds of the scan.  dx is fixed inside each pair so that a
#: c0 comparison never mixes two discretisations.
DXS = (0.04, 0.02)
SOUND_SPEEDS = (20.0, 40.0, 80.0)

#: Wall-clock budget passed to ``run_case`` in chunked mode (seconds).
CHUNK_BUDGET = 90.0


def case_tag(dx: float, c0: float) -> str:
    return f"dx{dx:g}_c0{c0:g}"


def build_config(dx: float, c0: float, t_end: float) -> KleefsmanConfig:
    """Kleefsman config with an explicit sound speed and a Courant-safe dt.

    ``dt`` is the smaller of the historical 2e-4 s cap and the acoustic limit
    ``max_courant * h / c0`` of *this* c0.  Using a fixed 2e-4 s for every c0
    would either violate the CFL condition (c0 >= 80 m/s at dx = 0.04) or
    cripple the high-c0 runs for no reason.
    """
    cfg = KleefsmanConfig(
        dx=dx,
        sound_speed=c0,
        t_end=t_end,
        tank_length=GEOM["tank_length"],
        tank_height=GEOM["tank_height"],
        column_length=GEOM["column_length"],
        column_height=GEOM["column_height"],
        obstacle_x=GEOM["obstacle_upstream_face_x"],
        obstacle_length=GEOM["obstacle_length"],
        obstacle_height=GEOM["obstacle_height"],
        pool_width=GEOM["tank_length"],
        pool_depth=GEOM["tank_height"],
    )
    cfg.dt = min(2.0e-4, cfg.dt_acoustic())
    return cfg


def run_one(dx: float, c0: float, t_end: float, outdir: Path,
            chunk_budget: float | None = CHUNK_BUDGET) -> dict:
    """Run one (dx, c0) pair, resuming across process boundaries if needed."""
    outdir.mkdir(parents=True, exist_ok=True)
    tag = case_tag(dx, c0)
    metrics_path = outdir / f"{tag}.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())

    state_path = outdir / f"{tag}.state.pt"
    cfg = build_config(dx, c0, t_end)
    validate_case(cfg)

    import torch

    state = None
    chunks = 0
    t_wall = 0.0
    steps_total = 0
    dt_lo, dt_hi = float("inf"), 0.0
    if state_path.exists():
        state = torch.load(state_path, weights_only=False)
    while True:
        t0 = time.monotonic()
        run = run_case(cfg, resume=state, time_budget=chunk_budget)
        t_wall += time.monotonic() - t0
        chunks += 1
        # ``run`` only describes the chunk just integrated: its step count and
        # dt extremes restart from the resumed state, so aggregate explicitly.
        steps_total += run.steps_taken
        if run.dt_min > 0:
            dt_lo = min(dt_lo, run.dt_min)
        dt_hi = max(dt_hi, run.dt_max)
        if run.finished or chunk_budget is None:
            break
        state = run.state
        torch.save(state, state_path)
        if chunks > 200:                     # pragma: no cover - safety valve
            raise RuntimeError("chunk loop did not converge")
    if state_path.exists():
        state_path.unlink()

    speed = float(run.max_speed)
    compression = float(run.max_density_compression)
    metrics = {
        "tag": tag,
        "dx": dx,
        "sound_speed": c0,
        "mach_max": speed / c0,
        "t_end": t_end,
        "dt_nominal": cfg.dt,
        "dt_acoustic_limit": cfg.dt_acoustic(),
        "dt_min": dt_lo if dt_lo < float("inf") else cfg.dt,
        "dt_max": dt_hi,
        "steps": steps_total,
        "simulated_time_s": run.simulated_time,
        "particles_fluid": int(math.floor(cfg.column_length / cfg.dx)
                               * math.floor(cfg.column_height / cfg.dx)),
        "peak_height_m": float(run.peak_height),
        "peak_time_s": float(run.peak_time),
        "arrival_time_s": float(run.arrival_time),
        "max_density_compression": compression,
        "max_speed": speed,
        "compressibility_ratio": (compression / (speed / c0) ** 2) if speed > 0 else None,
        "mass_error_rel": float(run.mass_error),
        "boundary_escapes": int(run.escaped),
        "wall_clock_s": round(t_wall, 1),
        "chunks": chunks,
        "host": os.uname().nodename,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def aggregate(outdir: Path, t_end: float) -> dict:
    """Collect the finished cases into one machine-readable table."""
    rows = []
    for dx in DXS:
        for c0 in SOUND_SPEEDS:
            p = outdir / f"{case_tag(dx, c0)}.json"
            if p.exists():
                rows.append(json.loads(p.read_text()))
    expected = len(DXS) * len(SOUND_SPEEDS)

    # Per-grid checks: compression must fall as c0 grows (falsifiable claim),
    # and the compressibility_ratio must stay O(1) if the acoustic picture holds.
    per_grid = {}
    for dx in DXS:
        sub = sorted((r for r in rows if r["dx"] == dx), key=lambda r: r["sound_speed"])
        if len(sub) < 2:
            continue
        mono = all(b["max_density_compression"] < a["max_density_compression"]
                   for a, b in zip(sub, sub[1:]))
        # empirical decay exponent of the compression vs c0: d log(comp) / d log(c0)
        first, last = sub[0], sub[-1]
        exponent = None
        if first["max_density_compression"] > 0 and first["sound_speed"] != last["sound_speed"]:
            exponent = (math.log(last["max_density_compression"])
                        - math.log(first["max_density_compression"])) \
                / (math.log(last["sound_speed"]) - math.log(first["sound_speed"]))
        per_grid[f"dx{dx:g}"] = {
            "monotone_decreasing": bool(mono),
            "compression_decay_exponent": exponent,
            "compression_below_5pct_at_max_c0":
                bool(sub[-1]["max_density_compression"] < 0.05),
            "compressibility_ratio_range": [
                min(r["compressibility_ratio"] for r in sub),
                max(r["compressibility_ratio"] for r in sub),
            ],
            "peak_height_spread_m": [min(r["peak_height_m"] for r in sub),
                                     max(r["peak_height_m"] for r in sub)],
        }

    summary = {
        "case": "Kleefsman (2005) dam break with obstacle, probe upstream of the box",
        "reference": "J. Comput. Phys. 206:363-393, DOI 10.1016/j.jcp.2004.12.007",
        "experimental_status": REF["experimental_values"]["status"],
        "t_end": t_end,
        "matrix": {"dx": list(DXS), "sound_speed": list(SOUND_SPEEDS)},
        "completed": len(rows),
        "expected": expected,
        "rows": sorted(rows, key=lambda r: (r["dx"], r["sound_speed"])),
        "per_grid": per_grid,
        "thresholds": {
            "density_compression_controllable": 0.05,
            "peak_monotone_in_dx_required_for_order": True,
        },
    }
    (outdir / "c0_scan_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None, help="i/n shard selector for the scan")
    ap.add_argument("--dx", type=float, action="append", default=None,
                    help="restrict to this dx (repeatable)")
    ap.add_argument("--sound-speed", type=float, action="append", default=None,
                    help="restrict to this c0 (repeatable)")
    ap.add_argument("--t-end", type=float, default=1.2)
    ap.add_argument("--outdir", default=str(HERE / "results" / "c0_scan"))
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--no-chunk", action="store_true",
                    help="run without a wall-clock budget (single process)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if args.aggregate:
        summary = aggregate(outdir, args.t_end)
        print(json.dumps({k: summary[k] for k in
                          ("completed", "expected", "per_grid")}, indent=2))
        return 0

    dxs = tuple(args.dx) if args.dx else DXS
    c0s = tuple(args.sound_speed) if args.sound_speed else SOUND_SPEEDS
    cases = [(dx, c0) for dx in dxs for c0 in c0s]
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        cases = cases[i::n]

    budget = None if args.no_chunk else CHUNK_BUDGET
    for dx, c0 in cases:
        m = run_one(dx, c0, args.t_end, outdir, chunk_budget=budget)
        print(json.dumps({k: m[k] for k in
                          ("tag", "peak_height_m", "peak_time_s", "arrival_time_s",
                           "max_density_compression", "compressibility_ratio",
                           "steps", "wall_clock_s")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())