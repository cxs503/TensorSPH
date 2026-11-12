#!/usr/bin/env python3
"""CASE-SPH-GRESHO-VORTEX: driver (ladder + controls + aggregation).

The physics and every number live in :mod:`tensorsph.gresho`; this script only
sequences the resolution ladder, writes one JSON per resolution and aggregates
the machine verdict. Nothing is retyped by hand: ``accept`` reads the per-run
JSON files it just wrote.

    PYTHONPATH=src python benchmarks/gresho_vortex/run.py --stage matrix \
        --resolutions 0.025,0.02,0.0125 --t-end 1.0
    PYTHONPATH=src python benchmarks/gresho_vortex/run.py --stage accept
    python benchmarks/gresho_vortex/make_figures.py
    python benchmarks/gresho_vortex/make_doc.py

Why the results go to ``benchmarks/gresho_vortex/results/`` and not ``runs/`` or
``artifacts/``: the repository ``.gitignore`` excludes both, so evidence written
there is silently dropped by the merge gate (lesson from mission m16452).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.gresho import (GATES, GreshoConfig, accept,  # noqa: E402
                              analytic_reference, convergence_study,
                              spurious_acceleration)

RESULTS = Path(__file__).resolve().parent / "results"


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(type(obj))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default,
                               sort_keys=False) + "\n")


def environment() -> dict:
    import platform
    import torch
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "nproc": os.cpu_count(),
        "cwd": str(Path.cwd()),
    }


def stage_matrix(args) -> int:
    resolutions = [float(x) for x in args.resolutions.split(",")]
    t_end = float(args.t_end)
    started = time.time()
    rows = []
    for dx in resolutions:
        cfg = GreshoConfig(dx=dx, t_end=t_end)
        t0 = time.time()
        run = convergence_study((dx,), t_end=t_end)[0]
        row = {k: v for k, v in run.items() if k != "run"}
        row["wall_seconds"] = time.time() - t0
        write_json(RESULTS / f"dx{dx:g}.json", run["run"])
        rows.append(row)
        print(f"[matrix] dx={dx:g} particles={row['particles']} "
              f"a={row['spurious_acceleration_pct']:.3f}% "
              f"profile_rms={row['profile_error_pct_rms']:.3f}% "
              f"wall={row['wall_seconds']:.1f}s", flush=True)

    # --- controls on the coarsest resolution ------------------------------ #
    dx_ctl = resolutions[0]
    cfg = GreshoConfig(dx=dx_ctl)
    controls = {
        "forward": spurious_acceleration(cfg, init_mode="analytic"),
        "reverse": spurious_acceleration(cfg, init_mode="uniform"),
        "half_pressure": spurious_acceleration(cfg, init_mode="analytic",
                                               pressure_scale=0.5),
        "dx": dx_ctl,
    }
    controls["half_pressure"]["note"] = (
        "pressure field scaled by 0.5: the balance error must grow, which shows "
        "the metric responds to the pressure field and not to lattice noise")
    write_json(RESULTS / "controls.json", controls)

    # --- analytic reference (self-verified) ------------------------------- #
    ref = analytic_reference(GreshoConfig(dx=dx_ctl))
    table = {"r": ref.pop("r")[::100].tolist(),
             "u_theta": ref.pop("u_theta")[::100].tolist(),
             "p": ref.pop("p")[::100].tolist()}
    ref.update(table)                      # subsampled radial tables for figures
    write_json(RESULTS / "reference.json", ref)

    write_json(RESULTS / "ladder_raw.json", {
        "rows": rows, "t_end": t_end, "resolutions": resolutions,
        "environment": environment(), "wall_seconds": time.time() - started,
        "gates": GATES,
    })
    print(f"[matrix] done in {time.time() - started:.1f}s")
    return 0


def stage_accept(args) -> int:
    raw = json.loads((RESULTS / "ladder_raw.json").read_text())
    controls = json.loads((RESULTS / "controls.json").read_text())
    for row in raw["rows"]:
        row.setdefault("observed_order_acceleration", float("nan"))
        row.setdefault("observed_order_velocity", float("nan"))
        row.setdefault("observed_order_profile", float("nan"))
    rows = raw["rows"]
    # orders across adjacent resolutions, computed here so the JSON is complete
    for coarse, fine in zip(rows[:-1], rows[1:]):
        for key, out in (("spurious_acceleration_pct", "observed_order_acceleration"),
                         ("velocity_error_pct_end", "observed_order_velocity"),
                         ("profile_error_pct_rms", "observed_order_profile")):
            num, den = coarse[key], fine[key]
            if den > 0.0 and num > den:
                fine[out] = math.log(num / den) / math.log(coarse["dx"] / fine["dx"])
    verdict = accept(rows, controls=controls)
    write_json(RESULTS / "verdict.json", verdict)
    write_json(RESULTS / "summary.json", {
        "case": "cup_case_gresho_vortex",
        "anchor": "CASE-SPH-GRESHO-VORTEX",
        "rows": rows,
        "controls": controls,
        "verdict": verdict,
        "gates": GATES,
        "environment": raw["environment"],
        "t_end": raw["t_end"],
        "wall_seconds": raw["wall_seconds"],
    })
    csv = ["dx,particles,steps,spurious_acceleration_pct,velocity_error_pct_end,"
           "profile_error_pct_rms,energy_ratio,equivalent_viscosity,"
           "equivalent_reynolds,angular_momentum_drift_rel,"
           "observed_order_acceleration,observed_order_profile"]
    for r in rows:
        csv.append(",".join(str(r[k]) for k in [
            "dx", "particles", "steps", "spurious_acceleration_pct",
            "velocity_error_pct_end", "profile_error_pct_rms", "energy_ratio",
            "equivalent_viscosity", "equivalent_reynolds",
            "angular_momentum_drift_rel", "observed_order_acceleration",
            "observed_order_profile"]))
    (RESULTS / "table.csv").write_text("\n".join(csv) + "\n")
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["verdict"] == "PASS" else 0  # honest FAIL is still output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("matrix", "accept"), required=True)
    parser.add_argument("--resolutions", default="0.025,0.02,0.0125")
    parser.add_argument("--t-end", default="1.0")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args(argv)
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.stage == "matrix":
        return stage_matrix(args)
    return stage_accept(args)


if __name__ == "__main__":
    raise SystemExit(main())