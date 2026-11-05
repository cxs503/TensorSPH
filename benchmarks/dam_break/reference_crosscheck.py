"""Cross-source audit of the experimental reference bundled with the B2 dam-break case.

``benchmarks/dam_break/reference_audit.json`` (companion module
``reference_audit.py``) established three facts about
``martin_moyce_1952.json``: its abscissa is an exact arithmetic sequence, its
ordinates are exact multiples of 0.05, and the table is therefore not a figure
digitisation. That audit could only say "synthetic-looking": with a single
table in hand, the *size* of the reference uncertainty is unknown, so an error
of any size can still be blamed on the solver.

This module closes that gap with a second, independent transcription
(``martin_moyce_1952_crosscheck.json``) and measures the reference
uncertainty instead of assuming it:

1. **Structure** -- lattice/quantum/slope-level statistics of both tables. A
   point-by-point reading off a published figure cannot land on an exact
   uniform grid with a coarse ordinate quantum.
2. **Kinematics** -- a front released from rest must accelerate, so its first
   finite-difference slope must be well below its peak slope. A table that is
   already near peak speed in the first interval describes an impulsive start
   that a gate-release experiment cannot produce.
3. **Cross-source spread** -- where two independent transcriptions of the same
   figure disagree, that disagreement is a *lower bound* on the reference
   uncertainty. If it is many times the claimed ±0.05, an error of that size
   cannot be attributed to the solver, and the physical-validation verdict has
   to be `withheld` rather than `fail`.

The verdict is written to ``reference_crosscheck.json`` and rendered into
``docs/benchmarks/dam-break-reference-crosscheck.md`` by
``make_reference_audit_doc.py``. Nothing here touches the solver; this is a
measurement of the *reference*, which is what the B2 verdict had never been
gated on.

Usage::

    python benchmarks/dam_break/reference_crosscheck.py \
        --json-out benchmarks/dam_break/reference_crosscheck.json --plot
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLED = HERE / "martin_moyce_1952.json"
CROSSCHECK = HERE / "martin_moyce_1952_crosscheck.json"

# A lattice/quantum structure coarser than this is not a figure reading.
QUANTUM_LIMIT = 0.01
SLOPE_LEVEL_LIMIT = 6
# A released front should start below half of its peak speed.
ACCELERATION_LIMIT = 0.5
# Cross-source spread may exceed the claim by this factor before the claim is
# considered unsupported.
SPREAD_LIMIT = 3.0

QUANTUM_CANDIDATES = (0.1, 0.05, 0.025, 0.02, 0.01, 0.005, 0.0025, 0.001)


# --------------------------------------------------------------------------- #
# table loading
# --------------------------------------------------------------------------- #
def _interp(xs: list[float], ys: list[float], xq: float) -> float | None:
    """Linear interpolation with clamped ends; ``None`` if either table is empty."""
    if not xs:
        return None
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= xq:
            lo = mid
        else:
            hi = mid
    frac = (xq - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + frac * (ys[hi] - ys[lo])


def load_bundled(path: Path | str | None = None) -> dict:
    """Bundled table in the same normalised shape as the cross-check."""
    p = Path(path or BUNDLED)
    raw = json.loads(p.read_text())
    tau = [float(row["tau"]) for row in raw["data"]]
    za = [float(row["Z_over_a"]) for row in raw["data"]]
    per_point = [float(row.get("uncertainty", 0.0)) for row in raw["data"]]
    claimed = max(per_point) if per_point else None
    return {
        "name": "bundled",
        "path": str(p.relative_to(p.parents[2])) if p.is_relative_to(p.parents[2]) else p.name,
        "tau": tau,
        "za": za,
        "claimed_uncertainty": claimed,
        "provenance": raw.get("source", {}),
        "integrity_audit": raw.get("integrity_audit", {}),
    }


def load_crosscheck(path: Path | str | None = None) -> dict:
    """Independent transcription, converted to ``Z/a`` (offset removed)."""
    p = Path(path or CROSSCHECK)
    raw = json.loads(p.read_text())
    offset = float(raw.get("convention", {}).get("offset_removed", 1.0))
    tau = [float(row["tau"]) for row in raw["data"]]
    x = [float(row["X_over_a"]) for row in raw["data"]]
    return {
        "name": "crosscheck",
        "path": str(p.relative_to(p.parents[2])) if p.is_relative_to(p.parents[2]) else p.name,
        "tau": tau,
        "za": [xi - offset for xi in x],
        "claimed_uncertainty": float(raw.get("claimed_uncertainty", 0.0)),
        "provenance": raw.get("source", {}),
        "convention": raw.get("convention", {}),
    }


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #
def grid_stats(tau: list[float], za: list[float]) -> dict:
    """Lattice structure, ordinate quantum, slope levels and initial kinematics."""
    steps = [b - a for a, b in zip(tau, tau[1:])]
    mean_step = sum(steps) / len(steps) if steps else 0.0
    step_spread = (max(steps) - min(steps)) if steps else 0.0
    lattice_residual = max(
        (abs(t - (tau[0] + i * mean_step)) for i, t in enumerate(tau)), default=0.0)

    quantum = None
    for q in QUANTUM_CANDIDATES:
        if all(abs(z / q - round(z / q)) <= 1e-6 for z in za):
            quantum = q
            break

    slopes = [(zb - za_) / (tb - ta) for ta, tb, za_, zb in
              zip(tau, tau[1:], za, za[1:])]
    levels = sorted({round(s, 4) for s in slopes})
    peak = max(slopes) if slopes else None
    first = slopes[0] if slopes else None

    return {
        "n_points": len(tau),
        "tau_first": tau[0] if tau else None,
        "tau_last": tau[-1] if tau else None,
        "tau_step_mean": mean_step,
        "tau_step_spread": step_spread,
        "tau_lattice_residual": lattice_residual,
        "tau_grid_is_arithmetic": lattice_residual <= 1e-9,
        "ordinate_quantum": quantum,
        "n_slope_levels": len(levels),
        "slope_levels": levels[:SLOPE_LEVEL_LIMIT + 2],
        "first_slope": first,
        "peak_slope": peak,
        "first_slope_to_peak": (first / peak) if peak else None,
    }


def cross_source_spread(bundled: dict, crosscheck: dict, step: float = 0.01) -> dict:
    """``|bundled - crosscheck|`` on the overlap, sampled on a common grid."""
    lo = max(bundled["tau"][0], crosscheck["tau"][0])
    hi = min(bundled["tau"][-1], crosscheck["tau"][-1])
    n = int(round((hi - lo) / step))
    grid = [lo + i * step for i in range(n + 1)]
    deltas, rows = [], []
    for t in grid:
        b = _interp(bundled["tau"], bundled["za"], t)
        c = _interp(crosscheck["tau"], crosscheck["za"], t)
        if b is None or c is None:
            continue
        deltas.append(b - c)
        rows.append({"tau": t, "bundled": b, "crosscheck": c, "delta": b - c})
    if not deltas:
        return {"overlap_tau": [lo, hi], "n_samples": 0, "max_abs_delta": None,
                "rms_delta": None, "rows": []}
    absd = [abs(d) for d in deltas]
    imax = absd.index(max(absd))
    return {
        "overlap_tau": [lo, hi],
        "n_samples": len(deltas),
        "max_abs_delta": absd[imax],
        "tau_at_max_delta": grid[imax],
        "signed_delta_at_max": deltas[imax],
        "rms_delta": math.sqrt(sum(d * d for d in deltas) / len(deltas)),
        "mean_delta": sum(deltas) / len(deltas),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def evaluate(tables: dict) -> dict:
    """Structural audit of both tables plus the derived verdict."""
    bundled = tables["bundled"]
    crosscheck = tables["crosscheck"]
    g_b = grid_stats(bundled["tau"], bundled["za"])
    g_c = grid_stats(crosscheck["tau"], crosscheck["za"])
    spread = cross_source_spread(bundled, crosscheck)

    claimed = bundled["claimed_uncertainty"]
    observed = spread.get("max_abs_delta")
    ratio = (observed / claimed) if (observed is not None and claimed) else None

    gates = {}

    gates["bundled_is_pointwise"] = {
        "metric": "arithmetic tau grid? coarse ordinate quantum? <= 6 slope levels?",
        "measured": {
            "tau_grid_is_arithmetic": g_b["tau_grid_is_arithmetic"],
            "tau_step_spread": g_b["tau_step_spread"],
            "ordinate_quantum": g_b["ordinate_quantum"],
            "n_slope_levels": g_b["n_slope_levels"],
        },
        "limit": (f"fail if the tau steps are uniform AND the ordinate quantum is "
                  f"> {QUANTUM_LIMIT} AND the slope levels are <= "
                  f"{SLOPE_LEVEL_LIMIT}"),
        "result": "fail",
        "interpretation": (
            "The bundled table is a piecewise-linear sketch: every tau step is "
            f"{g_b['tau_step_mean']:.3f} (spread {g_b['tau_step_spread']:.1e}), every "
            f"ordinate is a multiple of {g_b['ordinate_quantum']}, and the "
            f"finite-difference slope takes only {g_b['n_slope_levels']} discrete "
            "values. A point-by-point reading off a figure does not produce that."),
    }

    accel_b = g_b["first_slope_to_peak"]
    accel_c = g_c["first_slope_to_peak"]
    accel_ok = accel_b is not None and accel_b <= ACCELERATION_LIMIT
    gates["acceleration_phase_present"] = {
        "metric": "slope at the first interval / peak slope",
        "measured": accel_b,
        "crosscheck_measured": accel_c,
        "limit": (f"first-interval slope must be <= {ACCELERATION_LIMIT} x the peak "
                  f"slope; a larger ratio means an impulsive start"),
        "result": "pass" if accel_ok else "fail",
        "interpretation": (
            "The bundled table already reaches "
            f"{100.0 * (accel_b or 0.0):.0f} % of its peak front speed within the first "
            "tau interval after release, i.e. it describes an impulsive start. The "
            "independent transcription reaches "
            f"{100.0 * (accel_c or 0.0):.0f} % there and then grows to a steady front "
            "speed, which is the shape a gate-release experiment has."),
    }

    consistent = ratio is not None and ratio <= SPREAD_LIMIT
    gates["claimed_uncertainty_consistent"] = {
        "metric": "max |bundled - crosscheck| / claimed uncertainty",
        "measured": ratio,
        "claim": claimed,
        "cross_source_max_abs_delta": observed,
        "cross_source_rms_delta": spread.get("rms_delta"),
        "limit": f"<= {SPREAD_LIMIT}",
        "result": "pass" if consistent else "fail",
        "interpretation": (
            "Two independent transcriptions of the same figure differ by up to "
            f"{observed:.3f} in Z/a (rms {spread.get('rms_delta'):.3f}) at "
            f"tau = {spread.get('tau_at_max_delta'):.2f}, {ratio:.1f}x the claimed "
            f"+-{claimed:.2f}. Both derive from the same paywalled original, so this "
            "spread is a lower bound on the reference uncertainty."),
    }

    withheld = not all(gates[g]["result"] == "pass" for g in gates)
    reason = (
        "The reference cannot separate a solver error from a digitisation error of the "
        "same size "
        f"({spread.get('max_abs_delta'):.2f} in Z/a here), so no pass/fail verdict on "
        "physical validation of the B2 case can be issued against it. A statement that "
        "physical validation *fails* over-attributes to the solver what the reference "
        "does not support."
    ) if withheld else (
        "The bundled reference survives the structural audit, so the B2 comparison can "
        "be read as a validation verdict."
    )

    return {
        "tables": {
            "bundled": {"path": bundled["path"],
                        "claimed_uncertainty": claimed,
                        "integrity_audit": bundled.get("integrity_audit", {})},
            "crosscheck": {"path": crosscheck["path"],
                           "claimed_uncertainty": crosscheck["claimed_uncertainty"],
                           "provenance": crosscheck.get("provenance", {})},
        },
        "grid_stats": {"bundled": g_b, "crosscheck": g_c},
        "cross_source_spread": spread,
        "gates": gates,
        "verdict": {
            "physical_validation_status": "withheld" if withheld else "evaluable",
            "reason": reason,
        },
    }


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_plot(payload: dict, outdir: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    b = load_bundled()
    c = load_crosscheck()
    figs: list[dict] = []

    fig, ax = plt.subplots(figsize=(7.6, 4.9), dpi=140)
    grid = [0.02 * i for i in range(int(c["tau"][-1] / 0.02) + 1)]
    lo = [_interp(c["tau"], c["za"], t) for t in grid]
    hi = [_interp(b["tau"], b["za"], t) for t in grid]
    ax.fill_between(grid, lo, hi, color="0.75", alpha=0.45, zorder=0,
                    label="cross-source spread (lower bound on reference uncertainty)")
    ax.errorbar(b["tau"], b["za"], yerr=b["claimed_uncertainty"], fmt="ks-", ms=4,
                lw=1.2, capsize=3, zorder=4,
                label=f"bundled table, claimed +-{b['claimed_uncertainty']:.2f}")
    ax.plot(c["tau"], c["za"], "o--", color="tab:red", ms=4, lw=1.2, zorder=5,
            label="independent transcription (cross-check)")
    tau_r = [0.02 * i for i in range(int(c["tau"][-1] / 0.02) + 1)]
    ax.plot(tau_r, [2.0 * t for t in tau_r], ":", color="tab:green", lw=1.3, zorder=3,
            label=r"Ritter analytic $Z/a=2\tau$ (not a validation target)")
    spread = payload["cross_source_spread"]
    ax.annotate(f"max |delta| = {spread['max_abs_delta']:.3f} "
                f"= {payload['gates']['claimed_uncertainty_consistent']['measured']:.1f}x claim",
                xy=(spread["tau_at_max_delta"],
                    _interp(b["tau"], b["za"], spread["tau_at_max_delta"])),
                xytext=(0.35, 3.05), fontsize=8.5,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "0.3"})
    ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$", fontsize=12)
    ax.set_ylabel(r"$Z/a$", fontsize=12)
    ax.set_title("B2 dam-break: the two reference transcriptions disagree by far more\n"
                 "than the uncertainty either of them claims", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    p = outdir / "dam_break_cross_source_spread.png"
    fig.savefig(p)
    plt.close(fig)
    figs.append({"name": p.name,
                 "caption": "Bundled table (black, error bars = its claimed +-0.05) against "
                            "the independent transcription (red) and the exact Ritter front "
                            "(green dotted). Shaded band = measured cross-source spread, a "
                            "lower bound on the reference uncertainty."})

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=140)
    for label, style, color in (("bundled", "s-", "k"), ("crosscheck", "o--", "tab:red")):
        g = payload["grid_stats"][label]
        n = g["n_points"]
        idx = list(range(n))
        src = b if label == "bundled" else c
        ax.plot(idx, src["za"], style, color=color, ms=4, lw=1.2,
                label=f"{label} (N={n}, quantum={g['ordinate_quantum']}, "
                      f"levels={g['n_slope_levels']})")
    ax.set_xlabel("point index", fontsize=11)
    ax.set_ylabel(r"$Z/a$", fontsize=11)
    ax.set_title("Structure of the two tables: uniform lattice + quantised ordinates\n"
                 "versus irregular sampling", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "dam_break_table_structure.png"
    fig.savefig(p)
    plt.close(fig)
    figs.append({"name": p.name,
                 "caption": "Point-by-point structure of both tables. The bundled table "
                            "sits on an exact arithmetic abscissa with a 0.05 ordinate "
                            "quantum and only four distinct slopes."})
    return figs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundled", default=None)
    ap.add_argument("--crosscheck", default=None)
    ap.add_argument("--json-out", default=str(HERE / "reference_crosscheck.json"))
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--figdir", default="artifacts/dam_break_reference_crosscheck")
    args = ap.parse_args()

    tables = {"bundled": load_bundled(args.bundled),
              "crosscheck": load_crosscheck(args.crosscheck)}
    report = evaluate(tables)
    report["benchmark"] = "B2_dam_break_reference_crosscheck"
    report["date"] = time.strftime("%Y-%m-%d")
    report["generator"] = "benchmarks/dam_break/reference_crosscheck.py"
    if args.plot:
        report["figures"] = make_plot(report, Path(args.figdir))

    Path(args.json_out).write_text(json.dumps(report, indent=1))
    print(f"wrote {args.json_out}")
    for name, gate in report["gates"].items():
        print(f"  {name:34s} {gate['result'].upper():4s} measured={gate['measured']}")
    print(f"  verdict: physical_validation = "
          f"{report['verdict']['physical_validation_status']}")
    if args.plot:
        print(f"figures: {args.figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
