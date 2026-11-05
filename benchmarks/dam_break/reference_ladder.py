"""B2 dam-break: front-trajectory ladder scored against *both* reference tables.

``benchmarks/dam_break/resolution_ladder.json`` measured the Martin & Moyce
error of the B2 case on a three-rung refinement ladder and found it resolution
independent (mean absolute error 0.5988 / 0.5774 / 0.5728 in ``Z/a`` for
``dx`` = 0.04 / 0.02 / 0.01 m). ``reference_crosscheck.py`` then showed that the
bundled reference is a piecewise-linear sketch whose disagreement with an
independent transcription reaches 0.85 in ``Z/a`` -- 17x its own +-0.05 claim.

This module closes the loop: it re-measures *the same simulations* against both
references with every other setting held fixed, so the choice of reference is
the only difference between the columns.

* ``bundled``    - ``martin_moyce_1952.json`` (the table the gate has used);
* ``crosscheck`` - ``martin_moyce_1952_crosscheck.json`` (independent transcription);
* ``ritter``     - analytic shallow-water front, reported, never gated (the case
  is a finite reservoir, so agreement is not expected).

Two readings come out of it and they must not be confused:

* the *gate* reading answers "is MAE <= 0.15 in ``Z/a``?" and it fails against
  every reference -- but the two references differ by 17x that limit, so the
  gate cannot tell a solver error from a reference error;
* the *attribution* reading compares the simulated curve with the reference
  band: when ``MAE(sim, crosscheck)`` is comparable to
  ``max|bundled - crosscheck|``, the residual is reference-sized.

The output is machine readable and feeds
``docs/benchmarks/dam-break-reference-crosscheck.md``; no number in that
document is typed by hand.

Usage::

    python benchmarks/dam_break/reference_ladder.py --sizes 0.04,0.02 --seconds 0.5 \
        --json-out benchmarks/dam_break/reference_ladder.json --plot
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))

from tensorsph.dam_break import ritter_front_over_a  # noqa: E402

try:  # importable both as a script and as benchmarks.dam_break.reference_ladder
    from .reference_audit import run_rung  # noqa: E402
    from .reference_crosscheck import _interp, load_bundled, load_crosscheck  # noqa: E402
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(HERE))
    from reference_audit import run_rung  # noqa: E402
    from reference_crosscheck import _interp, load_bundled, load_crosscheck  # noqa: E402

LIMIT_MAE = 0.15          # the gate limit already used by the B2 case
RITTER_TAU_MAX = 3.0


# --------------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------------- #
def ritter_reference(step: float = 0.01, tau_max: float = RITTER_TAU_MAX) -> dict:
    """Ritter front sampled on a fine tau grid, for reference only."""
    n = int(round(tau_max / step))
    tau = [i * step for i in range(n + 1)]
    return {
        "name": "ritter",
        "path": "tensorsph.dam_break.ritter_front_over_a (analytic)",
        "tau": tau,
        "za": [ritter_front_over_a(t) for t in tau],
        "claimed_uncertainty": None,
    }


def error_vs(tau: list[float], za: list[float], ref: dict) -> dict:
    """MAE / RMSE / max error of a simulated curve against a reference table."""
    errors, rows = [], []
    for tau_ref, za_ref in zip(ref["tau"], ref["za"]):
        if not tau or tau_ref < tau[0] - 1e-12 or tau_ref > tau[-1] + 1e-12:
            continue
        za_sim = _interp(tau, za, tau_ref)
        if za_sim is None:
            continue
        errors.append(za_sim - za_ref)
        rows.append({"tau": tau_ref, "ref": za_ref, "sim": za_sim,
                     "error": za_sim - za_ref})
    if not errors:
        return {"mae": None, "rmse": None, "max_abs_error": None, "bias": None,
                "n_points": 0, "rows": []}
    absd = [abs(e) for e in errors]
    return {
        "mae": sum(absd) / len(absd),
        "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "max_abs_error": max(absd),
        "bias": sum(errors) / len(errors),
        "n_points": len(errors),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# one rung
# --------------------------------------------------------------------------- #
def run_one(dx: float, args, refs: dict) -> dict:
    rung = run_rung(dx, args.wall_friction, sound_speed=args.sound_speed,
                    seconds=args.seconds, tick=0.01)
    tau = [p["tau"] for p in rung["trajectory"]]
    za = [p["za"] for p in rung["trajectory"]]
    rec = {
        "dx": dx,
        "particles": rung["particles"],
        "dt": (rung["simulated_time"] / rung["steps_taken"]
               if rung["steps_taken"] else None),
        "steps_taken": rung["steps_taken"],
        "simulated_time": rung["simulated_time"],
        "wall_clock_s": rung["wall_clock_s"],
        "max_density_compression": rung["max_density_compression"],
        "mach_number": rung["mach_number"],
        "front_coefficient_c": rung["front_coefficient_c"],
        "fit_window_tau": rung["fit_window_tau"],
        "sim_tau": tau,
        "sim_za": za,
    }
    for name, ref in refs.items():
        rec[f"error_vs_{name}"] = error_vs(tau, za, ref)
    return rec


def gate_table(results: list[dict], ref_name: str, limit: float = LIMIT_MAE) -> dict:
    rows = []
    for r in results:
        err = r[f"error_vs_{ref_name}"]
        rows.append({
            "dx": r["dx"],
            "particles": r["particles"],
            "mae": err["mae"],
            "rmse": err["rmse"],
            "max_abs_error": err["max_abs_error"],
            "bias": err["bias"],
            "limit": limit,
            "result": ("pass" if err["mae"] is not None and err["mae"] <= limit
                       else "fail"),
        })
    return {
        "reference": ref_name,
        "metric": f"mean absolute error in Z/a vs {ref_name}",
        "limit": limit,
        "rows": rows,
        "finest": rows[-1] if rows else None,
        "all_pass": all(r["result"] == "pass" for r in rows) if rows else False,
    }


def observed_order(results: list[dict], ref_name: str) -> dict:
    """Slope of log(MAE) against log(dx): is that residual converging at all?"""
    pairs = [(r["dx"], r[f"error_vs_{ref_name}"]["mae"])
             for r in results if r[f"error_vs_{ref_name}"]["mae"] is not None]
    orders = []
    for (dx0, mae0), (dx1, mae1) in zip(pairs, pairs[1:]):
        orders.append(math.log(mae1 / mae0) / math.log(dx1 / dx0))
    return {"pairs": [{"dx": dx, "mae": mae} for dx, mae in pairs],
            "order_mae": orders}


def attribution(results: list[dict], spread: float, limit: float = LIMIT_MAE) -> dict:
    """Can the gate separate a solver error from a reference error?"""
    finest = results[-1]
    mae_b = finest["error_vs_bundled"]["mae"]
    mae_c = finest["error_vs_crosscheck"]["mae"]
    return {
        "reference_spread_max_abs_delta": spread,
        "gate_limit": limit,
        "reference_spread_over_gate_limit": (spread / limit) if spread else None,
        "finest_dx": finest["dx"],
        "mae_bundled": mae_b,
        "mae_crosscheck": mae_c,
        "mae_shift_when_reference_changes": (None if (mae_b is None or mae_c is None)
                                             else mae_b - mae_c),
        "reference_spread_dominates_gate": bool(spread and spread > limit),
        "conclusion": (
            "The two references differ by more than the gate limit itself, so a gate "
            "reading 'MAE > 0.15' cannot be attributed to the solver: changing only the "
            "reference moves the MAE by "
            f"{abs(0.0 if (mae_b is None or mae_c is None) else mae_b - mae_c):.3f}, a "
            "shift of the same order as the failure margin. The B2 physical-validation "
            "verdict therefore stays withheld until a scatter-bearing, citable "
            "reference is installed."
        ),
    }


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_plot(results: list[dict], refs: dict, payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0), dpi=140,
                                  gridspec_kw={"width_ratios": [1.35, 1.0]})

    colors = {0.04: "#1f77b4", 0.03: "#9467bd", 0.02: "#ff7f0e", 0.01: "#2ca02c"}
    b = refs["bundled"]
    c = refs["crosscheck"]
    grid = [0.02 * i for i in range(int(min(b["tau"][-1], c["tau"][-1]) / 0.02) + 1)]
    lo = [_interp(c["tau"], c["za"], t) for t in grid]
    hi = [_interp(b["tau"], b["za"], t) for t in grid]
    ax.fill_between(grid, lo, hi, color="0.75", alpha=0.4, zorder=0,
                    label="cross-source spread (reference uncertainty)")
    for r in results:
        ax.plot(r["sim_tau"], r["sim_za"], "-", lw=2.1,
                color=colors.get(round(r["dx"], 4), "#333333"), zorder=3,
                label=f"TensorSPH dx={r['dx']:.3f} m (N={r['particles']})")
    ax.errorbar(b["tau"], b["za"], yerr=b["claimed_uncertainty"], fmt="ks", ms=4.5,
                capsize=3, zorder=5, label="bundled table (claimed +-0.05)")
    ax.plot(c["tau"], c["za"], "o--", color="tab:red", ms=4, lw=1.1, zorder=4,
            label="independent transcription")
    tau_r = [0.02 * i for i in range(int(RITTER_TAU_MAX / 0.02) + 1)]
    ax.plot(tau_r, [ritter_front_over_a(t) for t in tau_r], ":", color="tab:green",
            lw=1.3, zorder=2, label=r"Ritter $Z/a=2\tau$ (reported, not gated)")
    ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$", fontsize=12)
    ax.set_ylabel(r"$Z/a$", fontsize=12)
    ax.set_title("Same simulations, three references", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.6, loc="upper left")

    names = ["bundled", "crosscheck", "ritter"]
    dxs = [r["dx"] for r in results]
    width = 0.26
    for i, name in enumerate(names):
        vals = [r[f"error_vs_{name}"]["mae"] for r in results]
        ax2.bar([j + (i - 1) * width for j in range(len(dxs))], vals, width=width,
                label=name)
    ax2.axhline(payload["gates"]["bundled"]["limit"], color="k", ls="--", lw=1.2,
                label="gate limit 0.15")
    spread = payload["attribution"]["reference_spread_max_abs_delta"]
    ax2.axhspan(0.0, spread, color="tab:red", alpha=0.12,
                label=f"reference spread ({spread:.2f})")
    ax2.set_xticks(range(len(dxs)))
    ax2.set_xticklabels([f"dx={d:.3f}" for d in dxs], fontsize=9)
    ax2.set_ylabel("MAE in $Z/a$", fontsize=11)
    ax2.set_title("Error depends on the reference chosen, not only on dx", fontsize=10)
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(fontsize=7.6)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="0.04,0.02")
    ap.add_argument("--sound-speed", type=float, default=25.0)
    ap.add_argument("--wall-friction", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--limit", type=float, default=LIMIT_MAE,
                    help="gate limit on the mean absolute error in Z/a")
    ap.add_argument("--json-out", default=str(HERE / "reference_ladder.json"))
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-out", default=str(
        HERE.parent.parent / "artifacts" / "dam_break_reference_crosscheck"
        / "dam_break_reference_ladder.png"))
    args = ap.parse_args()

    sizes = sorted((float(s) for s in args.sizes.split(",") if s.strip()), reverse=True)
    refs = {"bundled": load_bundled(), "crosscheck": load_crosscheck(),
            "ritter": ritter_reference()}

    results = []
    for dx in sizes:
        rec = run_one(dx, args, refs)
        results.append(rec)
        print(f"dx={dx:.3f} N={rec['particles']:5d} wall={rec['wall_clock_s']:6.1f}s "
              f"MAE_bundled={rec['error_vs_bundled']['mae']:.4f} "
              f"MAE_crosscheck={rec['error_vs_crosscheck']['mae']:.4f} "
              f"MAE_ritter={rec['error_vs_ritter']['mae']:.4f}", flush=True)

    cross = json.loads((HERE / "reference_crosscheck.json").read_text()) \
        if (HERE / "reference_crosscheck.json").exists() else None
    spread = (cross["cross_source_spread"]["max_abs_delta"] if cross else
              max(abs(a - b) for a, b in zip(
                  [_interp(refs["bundled"]["tau"], refs["bundled"]["za"], t)
                   for t in refs["crosscheck"]["tau"]],
                  refs["crosscheck"]["za"])))

    gates = {name: gate_table(results, name, args.limit) for name in refs}
    payload = {
        "benchmark": "B2_dam_break_reference_ladder",
        "date": time.strftime("%Y-%m-%d"),
        "generator": "benchmarks/dam_break/reference_ladder.py",
        "simulation_params": {
            "tank_width": 2.0,
            "column_width": 0.2,
            "column_height": 0.4,
            "sound_speed": args.sound_speed,
            "max_courant": 0.3,
            "wall_friction": args.wall_friction,
            "density_diffusion": 0.0,
            "adaptive_dt": False,
            "seconds": args.seconds,
            "device": "cpu",
        },
        "references": {
            "bundled": {"path": refs["bundled"]["path"],
                        "claimed_uncertainty": refs["bundled"]["claimed_uncertainty"]},
            "crosscheck": {"path": refs["crosscheck"]["path"],
                           "claimed_uncertainty": refs["crosscheck"]["claimed_uncertainty"]},
            "ritter": {"path": refs["ritter"]["path"], "claimed_uncertainty": None},
        },
        "results": results,
        "gates": gates,
        "convergence": {name: observed_order(results, name) for name in refs},
        "attribution": attribution(results, spread, args.limit),
        "figures": [],
    }

    if args.plot:
        p = Path(args.plot_out)
        make_plot(results, refs, payload, p)
        payload["figures"].append({
            "name": p.name,
            "caption": "Left: simulated front trajectories on two rungs against the two "
                       "reference transcriptions, the shaded cross-source spread and the "
                       "analytic Ritter front. Right: MAE per rung and per reference "
                       "against the 0.15 gate limit, with the reference spread as the "
                       "band at the bottom.",
            "path": str(p),
        })

    Path(args.json_out).write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.json_out}")
    for name, g in gates.items():
        finest = g["finest"]
        print(f"  gate[{name:10s}] finest dx={finest['dx']:.3f} "
              f"MAE={finest['mae']:.4f} limit={finest['limit']:.2f} "
              f"-> {finest['result'].upper()}")
    print(f"  reference spread {spread:.3f} = "
          f"{payload['attribution']['reference_spread_over_gate_limit']:.1f}x gate limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
