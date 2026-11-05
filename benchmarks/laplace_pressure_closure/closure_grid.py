"""Resolution / sound-speed grid for the free-surface pressure closure.

Why this exists
---------------
``docs/verification_laplace_pressure_closure.md`` (m3xxx) implemented the
Adami / Ihmsen / Adams (2012, sec. 3.3) pressure extrapolation and measured the
absolute Young-Laplace read-out for four closure variants -- but at **one**
resolution only (``dx = 2.0 mm``, ``c0 = 20 m/s``), and with two loose ends of
its own:

* the best variant (``donors = interior``, 2 layers) reached ``+8.8 %`` on the
  tail average while its *final snapshot* read ``+142.8 %``, i.e. the drop was
  still breathing hard at the end of the window;
* its spurious current, ``v_max / v_ref = 0.388``, is 3.9x Adami's own
  criterion (< 0.1), so the closure is not obviously healthy.

A single-resolution reading cannot answer the question the verification record
needs answered: **is the closure residual a discretisation error that shrinks
with ``dx``, or a fixed offset?**  This script runs the same relaxation on a
``dx x c0`` grid, closure off vs closure on, and records the reading under
three definitions (final snapshot, tail average, late-window average) plus the
spurious-current ratio, so the answer is measured rather than assumed.

Nothing in ``src/`` is touched: this is verification/instrumentation.

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_pressure_closure/closure_grid.py
    PYTHONPATH=src python benchmarks/laplace_pressure_closure/closure_grid.py \\
        --cases grid_dx1.5mm_off

Cases are flushed to ``raw/closure_grid.json`` one at a time, so an interrupted
run still leaves usable numbers (and a re-run skips what is already there unless
``--force`` is given).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "raw" / "closure_grid.json"

# (name, dx [m], c0 [m/s], closure kwargs).  ``grid_*_off`` is the reference the
# ``grid_*_on`` case is scored against; the pairs are what make the closure gain
# measurable at each resolution instead of against a number from elsewhere.
CASES = (
    ("grid_dx2.0mm_off", 0.002, 20.0, dict(closure=False)),
    ("grid_dx2.0mm_on", 0.002, 20.0, dict(closure=True, donors="interior", layers=2)),
    ("grid_dx1.5mm_off", 0.0015, 20.0, dict(closure=False)),
    ("grid_dx1.5mm_on", 0.0015, 20.0, dict(closure=True, donors="interior", layers=2)),
    ("grid_dx2.0mm_c0_10_on", 0.002, 10.0, dict(closure=True, donors="interior", layers=2)),
    ("grid_dx1.5mm_c0_10_on", 0.0015, 10.0, dict(closure=True, donors="interior", layers=2)),
)

PAIRS = (
    # (closure-off case, closure-on case) -- same dx, same c0
    ("grid_dx2.0mm_off", "grid_dx2.0mm_on"),
    ("grid_dx1.5mm_off", "grid_dx1.5mm_on"),
)


def load_floor():
    spec = importlib.util.spec_from_file_location(
        "lpf", HERE.parent / "laplace_pressure_floor" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def curve(run: dict, field: str = "absolute") -> list[tuple[float, float]]:
    """(t/t_sigma, reading) pairs from the recorded trace."""
    t = run["trace"]["t"]
    return list(zip(t, run["trace"][field]))


def summarise(run: dict, lpf) -> dict:
    """The reading under every definition this record uses, plus health metrics."""
    dp = lpf.DP_ANALYTIC
    tail = run["absolute_time_average_tail"]
    final = run["definition_absolute"]
    late = run["spurious_current"]["vmax_over_vref_final"]
    comp_req = (1.0 + dp / (lpf.RHO0 * run["sound_speed"] ** 2 / lpf.GAMMA)) ** (
        1.0 / lpf.GAMMA) - 1.0
    return {
        "case": run.get("case"),
        "dx_m": run["dx_m"],
        "n_particles": run["n_particles"],
        "sound_speed": run["sound_speed"],
        "steps": run["steps"],
        "t_over_t_sigma": run["t_over_t_sigma"],
        "closure": run["closure"],
        "delta_p_final_Pa": final["delta_p_Pa"],
        "err_final_pct": final["rel_error_pct"],
        "delta_p_tail_Pa": tail["delta_p_Pa"],
        "err_tail_pct": tail["rel_error_pct"],
        "within_3pct_tail": bool(tail["within_3pct"]),
        "within_15pct_tail": bool(tail["within_15pct"]),
        "vmax_over_vref_final": late,
        "rho_core_over_rho0": run["density"]["rho_core_over_rho0"],
        "compression_required": comp_req,
        "compression_achieved_over_required": (
            run["density"]["rho_core_over_rho0"] - 1.0) / comp_req if comp_req else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=[c[0] for c in CASES])
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--t-end", type=float, default=1.5,
                    help="window length in capillary times")
    ap.add_argument("--force", action="store_true",
                    help="re-run cases already present in the evidence file")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    lpf = load_floor()
    out = Path(args.out)
    record = {
        "benchmark": "laplace_pressure_closure",
        "question": (
            "is the residual Young-Laplace deficit after the free-surface pressure closure a "
            "discretisation error (shrinks with dx) or a fixed offset, and does the closure "
            "meet the 15 % / 3 % criteria at more than one resolution?"
        ),
        "reference": {
            "sigma_N_per_m": lpf.SIGMA,
            "radius_m": lpf.RADIUS,
            "rho0": lpf.RHO0,
            "analytic_dp_Pa": lpf.DP_ANALYTIC,
            "t_sigma_s": lpf.T_SIGMA,
        },
        "method": (
            "benchmarks/laplace_pressure_floor/run.py:relax() -- calibrated-mass hex drop, "
            "consistent CSF with fit_quadratic curvature, zero gravity, 1.5 t_sigma of "
            "WCSPH; readings: absolute p_core - p_vacuum (Tait normalised at rho0), tail = "
            "mean over the last 40 % of the window, final = last snapshot"
        ),
        "max_steps": args.max_steps,
        "cases": [],
    }
    if out.exists():
        record["cases"] = json.loads(out.read_text()).get("cases", [])
    done = {c["case"] for c in record["cases"]} if not args.force else set()

    out.parent.mkdir(parents=True, exist_ok=True)
    for name, dx, c0, kw in CASES:
        if name not in args.cases or name in done:
            if name in done:
                print(f"{name}: already in {out.name}, skipping (--force to redo)", flush=True)
            continue
        t0 = time.time()
        run = lpf.relax(dx, c0, t_end_over_t_sigma=args.t_end,
                        max_steps=args.max_steps, **kw)
        row = summarise(run, lpf)
        row["case"] = name
        row["wall_clock_s"] = round(time.time() - t0, 1)
        record["cases"] = [c for c in record["cases"] if c["case"] != name] + [row]
        record["cases"].sort(key=lambda c: [x[0] for x in CASES].index(c["case"]))
        out.write_text(json.dumps(record, indent=1))
        print(f"{name}: dx={dx * 1e3:.1f} mm c0={c0:g} N={row['n_particles']} "
              f"tail {row['err_tail_pct']:+6.1f} %  final {row['err_final_pct']:+7.1f} %  "
              f"v/v_ref {row['vmax_over_vref_final']:.3f}  ({row['wall_clock_s']:.0f} s)",
              flush=True)

    # closure gain, paired at fixed dx and c0
    gains = []
    by_name = {c["case"]: c for c in record["cases"]}
    for off, on in PAIRS:
        if off in by_name and on in by_name:
            gains.append({
                "dx_m": by_name[off]["dx_m"],
                "sound_speed": by_name[off]["sound_speed"],
                "off_err_tail_pct": by_name[off]["err_tail_pct"],
                "on_err_tail_pct": by_name[on]["err_tail_pct"],
                # gain in points of |error|: positive means the closure moved the
                # reading closer to sigma/R, whatever the sign of the residual.
                "gain_pt": (abs(by_name[off]["err_tail_pct"])
                            - abs(by_name[on]["err_tail_pct"])),
                "off_vmax_over_vref": by_name[off]["vmax_over_vref_final"],
                "on_vmax_over_vref": by_name[on]["vmax_over_vref_final"],
            })
    record["closure_gain"] = gains
    out.write_text(json.dumps(record, indent=1))
    print(f"\nwrote {out}", flush=True)
    for g in gains:
        print(f"dx={g['dx_m'] * 1e3:.1f} mm c0={g['sound_speed']:g}: "
              f"{g['off_err_tail_pct']:+.1f} % -> {g['on_err_tail_pct']:+.1f} % "
              f"(gain {g['gain_pt']:+.1f} pt of |error|)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
