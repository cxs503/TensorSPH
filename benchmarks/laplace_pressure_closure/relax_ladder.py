"""Relaxation ladder for the free-surface pressure closure (Young-Laplace read-out).

Runs the drop of ``benchmarks/laplace_pressure_floor`` with the closure off and
on (two donor sets) and records the absolute, time-averaged Laplace read-out.
Each case is flushed to the evidence file as soon as it finishes, so a partial
run still leaves usable numbers.

    PYTHONPATH=src python benchmarks/laplace_pressure_closure/relax_ladder.py --dx 0.002 --c0 20

This is the expensive half of the verification (a few capillary times of SPH per
case, ~1-2 min each on two cores); the static force-balance diagnostic in
``analysis.py`` is the cheap half.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = HERE / "raw"


def ladder_path(dx: float, c0: float) -> Path:
    """One evidence file per (grid, sound speed).

    The first ladder (`dx = 2.0 mm`, `c0 = 20 m/s`) keeps the historical name so
    that the existing figure/doc pipeline reads it unchanged; every other rung
    gets its own file, so a ladder can never mix resolutions in one record.
    """
    if abs(dx - 0.002) < 1e-12 and abs(c0 - 20.0) < 1e-12:
        return RAW / "relax_ladder.json"
    return RAW / f"relax_ladder_dx{dx * 1e3:.4f}mm_c0{c0:g}.json"

CASES = (
    ("closure_off", dict(closure=False)),
    ("closure_on_all_donors", dict(closure=True, donors="all")),
    ("closure_on_interior_layers2", dict(closure=True, donors="interior", layers=2)),
    ("closure_on_interior_layers4", dict(closure=True, donors="interior", layers=4)),
)


def load_floor():
    spec = importlib.util.spec_from_file_location(
        "lpf", HERE.parent / "laplace_pressure_floor" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.002)
    ap.add_argument("--c0", type=float, default=20.0)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--cases", nargs="+", default=[c[0] for c in CASES])
    args = ap.parse_args()

    lpf = load_floor()
    out = ladder_path(args.dx, args.c0)
    record = {"dx_m": args.dx, "sound_speed": args.c0,
              "max_steps": args.max_steps, "analytic_dp_Pa": lpf.DP_ANALYTIC,
              "cases": []}
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        prev = json.loads(out.read_text())
        if (abs(prev.get("dx_m", args.dx) - args.dx) < 1e-12
                and abs(prev.get("sound_speed", args.c0) - args.c0) < 1e-12):
            record["cases"] = prev.get("cases", [])

    for name, kw in CASES:
        if name not in args.cases:
            continue
        t0 = time.time()
        run = lpf.relax(args.dx, args.c0, max_steps=args.max_steps, **kw)
        entry = {
            "case": name, "config": kw, "wall_clock_s": round(time.time() - t0, 1),
            "tail_mean_Pa": run["absolute_time_average_tail"]["delta_p_Pa"],
            "tail_rel_error_pct": run["absolute_time_average_tail"]["rel_error_pct"],
            "final_abs_rel_error_pct": run["definition_absolute"]["rel_error_pct"],
            "tail_within_15pct": run["absolute_time_average_tail"]["within_15pct"],
            "rho_core_over_rho0": run["density"]["rho_core_over_rho0"],
            "compression_required": run["density"]["compression_required"],
            "vmax_over_vref": run["spurious_current"]["vmax_over_vref_final"],
            "steps": run["steps"], "n_particles": run["n_particles"],
            "pressure_profile": run["pressure_profile"],
            "density_profile": run["density_profile"],
        }
        record["cases"] = [c for c in record["cases"] if c["case"] != name] + [entry]
        out.write_text(json.dumps(record, indent=2))
        print(f"{name:30s} tail {entry['tail_rel_error_pct']:+7.1f} % "
              f"final {entry['final_abs_rel_error_pct']:+7.1f} % "
              f"vmax/vref {entry['vmax_over_vref']:.3f} "
              f"{entry['wall_clock_s']}s", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
