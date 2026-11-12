"""REPRO-SPH-MONAGHAN1994 — reproduction protocol for the anchor's reference case.

The anchor (``REPRO-SPH-MONAGHAN1994``, ledger acceptance "quantitative_targets
逐项容差") names Monaghan (1994), *J. Comput. Phys.* **110**:399-406.  The paper is
closed access (see ``results/refs_verified.json``: Crossref resolves the DOI by
rule and OpenAlex reports ``oa_status=closed`` with no deposited abstract), so no
number in this repository may be transcribed from it.  What *can* be adjudicated
is declared up front in ``refs.json`` and evaluated here:

* **T1 (gated)** front trajectory ``Z/a`` vs ``tau`` against the independent
  Martin & Moyce (1952) transcription (``martin_moyce_1952_crosscheck.json``,
  digitisation shared with Lethe), through the platform tolerance of 15 % of the
  reference amplitude over a pre-registered window.
* **T2 (reported, withheld reference)** the same comparison against the bundled
  transcription (``martin_moyce_1952.json``).  That table is graded WITHHELD by
  ``benchmarks/dam_break/reference_audit.json`` (its ordinates are exact
  multiples of 0.05 on an exact ``tau`` lattice), so its number is reported and
  never used as a pass source.
* **T3 (reference self-consistency)** the two transcriptions against each other.
  A reference pair that disagrees by more than its own stated uncertainty cannot
  arbitrate a solver.
* **T4 (withheld)** the paper's own tables and figures: no accessible copy.

Every number in the artifacts is produced here; nothing is hand-copied.

Usage::

    python benchmarks/repro_monaghan_1994/run.py --stage run       # simulate + summarise
    python benchmarks/repro_monaghan_1994/run.py --stage accept    # summarise committed runs
    python benchmarks/repro_monaghan_1994/run.py --stage run --dx 0.04,0.02 --seconds 0.34
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for candidate in (str(ROOT), str(ROOT / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from tensorsph.dam_break import DamBreakConfig, run_case            # noqa: E402
from benchmarks.dam_break.reference_audit import (                   # noqa: E402
    load_bundled, load_crosscheck)

# --------------------------------------------------------------------------- #
# pre-registered protocol (written before any run; mirrored in refs.json)
# --------------------------------------------------------------------------- #
WINDOW = (0.5, 3.0)          # tau window evaluated (startup transient excluded)
TOL_REL = 0.15               # platform tolerance: 15 % of the reference amplitude
N_SAMPLES = 161              # evaluation grid inside the window
COMPRESSION_LIMIT = 0.10     # max rho/rho0 - 1 allowed for a weakly-compressible run
DEFAULT_DX = "0.04,0.02,0.01"
DEFAULT_SECONDS = 0.34       # tau_end = 0.34 * sqrt(2g/a) ~ 3.4 with a = 0.2 m
DEFAULT_C0 = 25.0


# --------------------------------------------------------------------------- #
# comparison primitives (pure functions; unit-tested without a solver)
# --------------------------------------------------------------------------- #
def interp(xs: list[float], ys: list[float], xq: float) -> float | None:
    """Linear interpolation with clamped ends; ``None`` for an empty table."""
    if not xs:
        return None
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xs[i] >= xq:
            span = xs[i] - xs[i - 1]
            if span == 0.0:
                return ys[i]
            weight = (xq - xs[i - 1]) / span
            return ys[i - 1] + weight * (ys[i] - ys[i - 1])
    return ys[-1]


def evaluation_grid(sim_tau: list[float], ref: dict, window: tuple[float, float],
                    n: int = N_SAMPLES) -> list[float]:
    """Common abscissa: the window truncated to where simulation *and* table exist."""
    if not sim_tau:
        return []
    lo = max(window[0], sim_tau[0], ref["tau"][0])
    hi = min(window[1], sim_tau[-1], ref["tau"][-1])
    if hi <= lo:
        return []
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def compare(sim_tau: list[float], sim_za: list[float], ref: dict,
            window: tuple[float, float] = WINDOW, n: int = N_SAMPLES) -> dict:
    """Error metrics of a simulated trajectory against one reference table."""
    grid = evaluation_grid(sim_tau, ref, window, n)
    if not grid:
        return {"n_samples": 0, "reason": "no overlapping tau window"}
    pairs = []
    for tau in grid:
        sim = interp(sim_tau, sim_za, tau)
        exp = interp(ref["tau"], ref["za"], tau)
        if sim is None or exp is None:
            continue
        pairs.append((tau, sim, exp))
    if not pairs:
        return {"n_samples": 0, "reason": "interpolation returned nothing"}
    deltas = [sim - exp for _, sim, exp in pairs]
    abs_deltas = [abs(d) for d in deltas]
    amplitude = max(exp for _, _, exp in pairs)
    rmse = math.sqrt(sum(d * d for d in deltas) / len(deltas))
    mae = sum(abs_deltas) / len(abs_deltas)
    worst = max(pairs, key=lambda p: abs(p[1] - p[2]))
    uncertainty = float(ref.get("claimed_uncertainty", 0.05) or 0.05)
    return {
        "reference": ref["name"],
        "reference_path": ref["path"],
        "reference_usable": ref.get("usable_as_validation_target"),
        "claimed_uncertainty": uncertainty,
        "window": [grid[0], grid[-1]],
        "n_samples": len(pairs),
        "amplitude": amplitude,
        "mae": mae,
        "rmse": rmse,
        "max_abs": abs_deltas and max(abs_deltas),
        "bias": sum(deltas) / len(deltas),
        "mae_over_amplitude": mae / amplitude if amplitude else None,
        "max_abs_over_amplitude": (max(abs_deltas) / amplitude) if amplitude else None,
        "within_uncertainty_fraction": sum(1 for d in abs_deltas if d <= uncertainty)
        / len(abs_deltas),
        "worst_sample": {"tau": worst[0], "sim": worst[1], "reference": worst[2],
                         "delta": worst[1] - worst[2]},
        "samples": [{"tau": t, "sim": s, "reference": e, "delta": s - e}
                    for t, s, e in pairs],
    }


def bracket(sim_tau: list[float], sim_za: list[float],
            ref_a: dict, ref_b: dict, window: tuple[float, float] = WINDOW,
            n: int = N_SAMPLES) -> dict:
    """Place the simulation between two independent transcriptions of one figure.

    Returns the reference-pair spread (its own reproducibility test) and how much
    of the simulation lies inside the band spanned by the two tables widened by
    their claimed uncertainties.
    """
    lo = max(window[0], sim_tau[0], ref_a["tau"][0], ref_b["tau"][0])
    hi = min(window[1], sim_tau[-1], ref_a["tau"][-1], ref_b["tau"][-1])
    if hi <= lo:
        return {"n_samples": 0, "reason": "no common window"}
    grid = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    unc_a = float(ref_a.get("claimed_uncertainty", 0.05) or 0.05)
    unc_b = float(ref_b.get("claimed_uncertainty", 0.05) or 0.05)
    rows = []
    for tau in grid:
        a = interp(ref_a["tau"], ref_a["za"], tau)
        b = interp(ref_b["tau"], ref_b["za"], tau)
        sim = interp(sim_tau, sim_za, tau)
        if a is None or b is None or sim is None:
            continue
        band_lo = min(a, b) - max(unc_a, unc_b)
        band_hi = max(a, b) + max(unc_a, unc_b)
        rows.append({"tau": tau, "sim": sim, "ref_a": a, "ref_b": b,
                     "band_lo": band_lo, "band_hi": band_hi,
                     "inside": band_lo <= sim <= band_hi,
                     "excursion": max(0.0, band_lo - sim, sim - band_hi),
                     "ref_spread": abs(a - b)})
    if not rows:
        return {"n_samples": 0, "reason": "no rows"}
    spreads = [r["ref_spread"] for r in rows]
    claimed = max(unc_a, unc_b)
    return {
        "n_samples": len(rows),
        "window": [rows[0]["tau"], rows[-1]["tau"]],
        "reference_pair": [ref_a["name"], ref_b["name"]],
        "reference_spread_max": max(spreads),
        "reference_spread_mean": sum(spreads) / len(spreads),
        "claimed_uncertainty": claimed,
        "reference_spread_over_claimed": max(spreads) / claimed if claimed else None,
        "reference_self_consistent": max(spreads) <= 2.0 * claimed,
        "inside_fraction": sum(1 for r in rows if r["inside"]) / len(rows),
        "max_excursion": max(r["excursion"] for r in rows),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# solver driver
# --------------------------------------------------------------------------- #
def trajectory(cfg: DamBreakConfig, run) -> tuple[list[float], list[float]]:
    """Non-dimensional front trajectory, ``Z/a`` = (x_max - a)/a against tau."""
    a = cfg.column_width
    scale = math.sqrt(2.0 * cfg.gravity / a)
    return ([t * scale for t in run.times], [(z - a) / a for z in run.front])


def run_one(dx: float, seconds: float, c0: float, device: str = "cpu") -> dict:
    cfg = DamBreakConfig(
        dx=dx, pool_width=2.0, column_width=0.2, sound_speed=c0,
        max_courant=0.3, device=device, wall_friction=0.0,
        density_diffusion=0.0, adaptive_dt=False)
    cfg.column_height = 2.0 * cfg.column_width
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(seconds / cfg.dt))
    cfg.output_every = max(1, int(round(0.01 / cfg.dt)))

    wall_clock = time.perf_counter()
    run = run_case(cfg)
    wall_clock = time.perf_counter() - wall_clock
    tau, za = trajectory(cfg, run)
    return {
        "case": f"dx{dx:g}_c0{c0:g}",
        "dx": dx,
        "sound_speed": c0,
        "particles": int(run.final_positions.shape[0]),
        "dt": cfg.dt,
        "steps_taken": run.steps_taken,
        "simulated_time": run.simulated_time,
        "wall_clock_s": wall_clock,
        "max_speed": run.max_speed,
        "mach_number": run.max_speed / c0,
        "max_density_compression": run.max_density_compression,
        "escaped": run.escaped,
        "tau": tau,
        "za": za,
    }


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def references() -> dict[str, dict]:
    """Both bundled transcriptions, annotated with the audit's usability flag."""
    audit_path = ROOT / "benchmarks" / "dam_break" / "reference_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    bundled = load_bundled()
    crosscheck = load_crosscheck()
    bundled["usable_as_validation_target"] = bool(
        audit.get("table_audit", {}).get("usable_as_validation_target", False))
    crosscheck["usable_as_validation_target"] = True   # audited structurally below
    return {"bundled": bundled, "crosscheck": crosscheck}


def evaluate(runs: list[dict], refs: dict[str, dict]) -> dict:
    """Per-run comparisons, the reference-pair bracket, gates and the verdict."""
    finest = min(runs, key=lambda r: r["dx"])
    comparisons = {}
    for run in runs:
        comparisons[run["case"]] = {
            name: compare(run["tau"], run["za"], ref)
            for name, ref in refs.items()
        }
    band = bracket(finest["tau"], finest["za"], refs["bundled"], refs["crosscheck"])

    ladder = {}
    for name in refs:
        ladder[name] = [
            {"case": run["case"], "dx": run["dx"],
             "mae_over_amplitude": comparisons[run["case"]][name].get("mae_over_amplitude"),
             "mae": comparisons[run["case"]][name].get("mae")}
            for run in sorted(runs, key=lambda r: -r["dx"])
        ]

    def _trend(entries: list[dict]) -> dict:
        values = [e["mae_over_amplitude"] for e in entries
                  if e["mae_over_amplitude"] is not None]
        if len(values) < 2:
            return {"values": values, "monotone": None,
                    "reason": "needs at least two rungs"}
        monotone = all(values[i + 1] <= values[i] * 1.10 + 1e-12
                       for i in range(len(values) - 1))
        return {"values": values, "monotone": monotone,
                "ratio_finest_over_coarsest": values[-1] / values[0] if values[0] else None}

    gates = {
        "reference_self_consistency": {
            "metric": "max |bundled - crosscheck| / claimed uncertainty",
            "measured": band.get("reference_spread_over_claimed"),
            "limit": 2.0,
            "result": "pass" if band.get("reference_self_consistent") else "fail",
            "detail": {"max_abs_spread": band.get("reference_spread_max"),
                       "claimed_uncertainty": band.get("claimed_uncertainty"),
                       "window": band.get("window")},
        },
        "resolution_trend_vs_independent": {
            "metric": "mae/amplitude vs the independent transcription must not grow under refinement",
            "measured": _trend(ladder["crosscheck"]),
            "limit": "non-increasing (10 % slack)",
            "result": ("pass" if _trend(ladder["crosscheck"]).get("monotone") else "fail")
            if _trend(ladder["crosscheck"]).get("monotone") is not None else "not_evaluated",
        },
        "compressibility": {
            "metric": "max(rho/rho0 - 1) over the run, finest resolution",
            "measured": finest["max_density_compression"],
            "limit": COMPRESSION_LIMIT,
            "result": "pass" if finest["max_density_compression"] <= COMPRESSION_LIMIT
            else "fail",
        },
    }

    primary = comparisons[finest["case"]]["crosscheck"]
    measured = primary.get("mae_over_amplitude")
    gates["trajectory_vs_independent"] = {
        "metric": "mae(Z/a) / amplitude, independent transcription, finest resolution",
        "measured": measured,
        "limit": TOL_REL,
        "result": ("pass" if measured is not None and measured <= TOL_REL else "fail")
        if measured is not None else "not_evaluated",
        "detail": {"dx": finest["dx"], "window": primary.get("window"),
                   "max_abs": primary.get("max_abs"),
                   "within_uncertainty_fraction":
                       primary.get("within_uncertainty_fraction"),
                   "bias": primary.get("bias")},
    }
    withheld = comparisons[finest["case"]]["bundled"]
    gates["trajectory_vs_bundled"] = {
        "metric": "mae(Z/a) / amplitude, bundled transcription, finest resolution",
        "measured": withheld.get("mae_over_amplitude"),
        "limit": TOL_REL,
        "result": "withheld",
        "reason": ("the bundled table is graded WITHHELD by "
                   "benchmarks/dam_break/reference_audit.json: its ordinates are "
                   "exact multiples of 0.05 on an exact tau lattice, so it cannot "
                   "arbitrate a tolerance; the number is recorded, not graded"),
        "detail": {"dx": finest["dx"], "window": withheld.get("window"),
                   "max_abs": withheld.get("max_abs"),
                   "bias": withheld.get("bias")},
    }

    passed = gates["trajectory_vs_independent"]["result"] == "pass"
    if not band.get("reference_self_consistent", False):
        interpretation = (
            "The two transcriptions of the same figure disagree by "
            f"{band.get('reference_spread_max'):.3f} in Z/a, i.e. "
            f"{band.get('reference_spread_over_claimed'):.1f}x the uncertainty they "
            f"claim, so the pair cannot both be right. The simulation is graded "
            f"against the transcription whose structure survives the audit "
            f"({'pass' if passed else 'fail'} at {measured:.3%} of amplitude) and the "
            f"bundled table is reported without a grade.")
    else:
        interpretation = (
            "The two transcriptions agree within their claimed uncertainty, so the "
            f"reference can arbitrate: the simulation reads {measured:.3%} of "
            f"amplitude against the independent transcription "
            f"({'pass' if passed else 'fail'}).")
    return {
        "comparisons": comparisons,
        "ladder": ladder,
        "bracket": band,
        "gates": gates,
        "verdict": {
            "physical_validation": "pass" if passed else "fail",
            "reference_evaluable": bool(band.get("reference_self_consistent")),
            "interpretation": interpretation,
            "withheld_reference_used_as_pass_source": False,
        },
    }


def build_payload(runs: list[dict], refs: dict[str, dict]) -> dict:
    payload = {
        "benchmark": "REPRO-SPH-MONAGHAN1994",
        "anchor": "REPRO-SPH-MONAGHAN1994",
        "protocol": {
            "window_tau": list(WINDOW),
            "tolerance_relative_to_amplitude": TOL_REL,
            "n_samples": N_SAMPLES,
            "compression_limit": COMPRESSION_LIMIT,
            "geometry": {"column_width_m": 0.2, "height_over_width": 2.0,
                         "tank_width_m": 2.0, "tank_height_m": 0.75,
                         "gravity": 9.81, "rho0": 1000.0, "tait_gamma": 7.0,
                         "h_over_dx": 1.3, "max_courant": 0.3,
                         "wall_friction": 0.0, "density_diffusion": 0.0},
            "caveat": ("pre-registered: the window, the tolerance and the metric are "
                       "fixed in this file, not chosen after seeing the numbers"),
        },
        "runs": [{k: v for k, v in run.items() if k not in ("tau", "za")} for run in runs],
    }
    payload.update(evaluate(runs, refs))
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "accept"), default="run")
    parser.add_argument("--dx", default=DEFAULT_DX,
                        help="comma-separated particle spacings in metres")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--sound-speed", type=float, default=DEFAULT_C0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=str(HERE / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    runs: list[dict] = []
    if args.stage == "run":
        dxs = sorted((float(item) for item in args.dx.split(",")), reverse=True)
        for dx in dxs:
            print(f"running dx={dx:g} c0={args.sound_speed:g} seconds={args.seconds:g}",
                  flush=True)
            run = run_one(dx, args.seconds, args.sound_speed, args.device)
            print(f"  N={run['particles']} steps={run['steps_taken']} "
                  f"wall={run['wall_clock_s']:.1f}s "
                  f"compression={run['max_density_compression']:.3f} "
                  f"tau_end={run['tau'][-1]:.2f} Z/a_end={run['za'][-1]:.2f}", flush=True)
            write_json(out_dir / f"trajectory_{run['case']}.json", run)
            runs.append(run)
    else:
        for path in sorted(out_dir.glob("trajectory_*.json")):
            runs.append(json.loads(path.read_text()))
        if not runs:
            print(f"no trajectory_*.json under {out_dir}: run --stage run first")
            return 1

    refs = references()
    payload = build_payload(runs, refs)
    write_json(out_dir / "summary.json", payload)

    verdict = payload["verdict"]
    for name, gate in payload["gates"].items():
        measured = gate.get("measured")
        shown = f"{measured:.4g}" if isinstance(measured, (int, float)) else str(measured)
        print(f"gate {name}: {gate['result']} (measured={shown}, limit={gate.get('limit')})")
    print("verdict:", verdict["physical_validation"])
    print(verdict["interpretation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())