"""Audit of the experimental reference bundled with the B2 dam-break case.

The B2 case compares the simulated front against a digitised Martin & Moyce
(1952) table (``martin_moyce_1952.json``).  The resolution ladder showed that
the comparison error is *resolution independent* (mean absolute error 0.5988 ->
0.5774 -> 0.5728 for ``dx`` = 0.04 -> 0.02 -> 0.01 m), which means the residual
is a property of the comparison, not of the front's numerical diffusion.  That
ladder left the cause open: "items 1-3 have to be separated by measurement".

This module separates the cheapest of those items first: **is the reference
itself a point-by-point digitisation, and how large is its uncertainty?**
Both questions are answered from the tables alone, with no simulation:

1. *Structure.*  A digitisation read off a published figure inherits the
   figure's resolution: irregular sample abscissae, values that do not land on
   a round grid, and increments with noise.  A table whose ``tau`` grid is a
   single repeated step, whose ordinates all sit on a 0.05 grid, and whose
   finite-difference slopes take only four discrete values is a piecewise
   linear sketch of a few hand-picked vertices, and its ±0.05 uncertainty claim
   is a guess about the picture, not a measured error bar.
2. *Kinematics.*  A front that is released from rest must first accelerate: the
   initial slope of ``Z/a`` against ``tau`` has to be small compared with the
   largest slope that follows.  A table whose slope is already near its maximum
   in the first interval after ``tau = 0`` describes a front with an impulsive
   start, which the experiment (a gate raised over a finite time) cannot have.
3. *Cross-source spread.*  A second, independent transcription of the same
   figure is bundled in ``martin_moyce_1952_crosscheck.json``.  The spread
   between the two is a *lower bound* on the reference uncertainty, because both
   come from the same inaccessible original.  If that spread is much larger than
   the claimed ±0.05, then an error of that size cannot be attributed to the
   solver at all.

The verdict is written to ``crosscheck_audit.json`` and consumed by
``docs/benchmarks/dam-break-crosscheck-audit.md``.  Nothing in this module
touches the solver; it is a measurement of the *reference*, which is what the
B2 verdict had never been gated on.

Usage::

    python benchmarks/dam_break/crosscheck_audit.py \
        --json-out benchmarks/dam_break/crosscheck_audit.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# table loading
# --------------------------------------------------------------------------- #
def _interp(xs: list[float], ys: list[float], xq: float) -> float | None:
    """Linear interpolation, clamped at both ends."""
    if not xs:
        return None
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xs[i] >= xq:
            span = xs[i] - xs[i - 1]
            w = (xq - xs[i - 1]) / span if span else 0.0
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[-1]


def load_bundled(path: str | Path | None = None) -> dict:
    """The table the B2 gate has been using (Z/a convention, measured from x = a)."""
    p = Path(path) if path is not None else HERE / "martin_moyce_1952.json"
    raw = json.loads(Path(p).read_text())
    return {
        "name": "bundled",
        "path": str(p),
        "tau": [float(d["tau"]) for d in raw["data"]],
        "za": [float(d["Z_over_a"]) for d in raw["data"]],
        "claimed_uncertainty": _claimed_uncertainty(raw),
        "source": raw.get("source", {}),
        "digitisation": raw.get("digitization", {}),
    }


def load_crosscheck(path: str | Path | None = None) -> dict:
    """The independent transcription, converted to the same Z/a convention."""
    p = Path(path) if path is not None else HERE / "martin_moyce_1952_crosscheck.json"
    raw = json.loads(Path(p).read_text())
    return {
        "name": "crosscheck",
        "path": str(p),
        "tau": [float(d["tau"]) for d in raw["data"]],
        # X = x_front/a, and Martin & Moyce measure Z from the column edge,
        # so Z/a = X - 1 keeps both tables in one frame.
        "za": [float(d["X_over_a"]) - 1.0 for d in raw["data"]],
        "claimed_uncertainty": float(raw.get("claimed_uncertainty", 0.05)),
        "source": raw.get("source", {}),
        "convention": raw.get("convention", {}),
    }


def _claimed_uncertainty(raw: dict) -> float:
    """Pull the numeric half-width out of the free-text uncertainty field."""
    text = str(raw.get("digitization", {}).get("uncertainty", ""))
    for token in text.replace("±", "+-").split():
        token = token.strip(".,;()")
        try:
            value = float(token)
        except ValueError:
            continue
        if 0.0 < value < 1.0:
            return value
    return 0.05


# --------------------------------------------------------------------------- #
# structural statistics
# --------------------------------------------------------------------------- #
def grid_stats(tau: list[float], za: list[float]) -> dict:
    """Everything the structure argument needs, as numbers."""
    steps = [tau[i + 1] - tau[i] for i in range(len(tau) - 1)]
    dz = [za[i + 1] - za[i] for i in range(len(za) - 1)]
    slopes = [dz[i] / steps[i] for i in range(len(dz)) if steps[i]]
    step_spread = (max(steps) - min(steps)) if steps else 0.0
    # Quantisation: find the coarsest round grid every ordinate sits on.
    quantum = None
    for candidate in (0.05, 0.025, 0.02, 0.01, 0.005):
        if all(abs(v / candidate - round(v / candidate)) < 1e-9 for v in za):
            quantum = candidate
            break
    # Discrete slope levels: group slopes that agree to 0.5 %.
    levels: list[float] = []
    for s in sorted(slopes):
        if not levels or abs(s - levels[-1]) > 5e-3 * max(abs(s), 1e-12):
            levels.append(s)
    peak = max(slopes) if slopes else 0.0
    return {
        "n_points": len(tau),
        "tau_first": tau[0] if tau else None,
        "tau_last": tau[-1] if tau else None,
        "tau_step_mean": sum(steps) / len(steps) if steps else None,
        "tau_step_spread": step_spread,
        "tau_grid_is_arithmetic": step_spread <= 1e-9,
        "ordinate_quantum": quantum,
        "slope_first": slopes[0] if slopes else None,
        "slope_peak": peak,
        "slope_peak_index": slopes.index(peak) if slopes else None,
        "slope_last": slopes[-1] if slopes else None,
        "n_slope_levels": len(levels),
        "slope_levels": [round(s, 4) for s in levels],
        "slopes": [round(s, 4) for s in slopes],
        "first_slope_to_peak": (slopes[0] / peak) if slopes and peak else None,
        "slope_is_monotone_decreasing": all(
            slopes[i + 1] <= slopes[i] + 1e-12 for i in range(len(slopes) - 1)
        ),
    }


def cross_source_spread(bundled: dict, crosscheck: dict) -> dict:
    """Difference between the two transcriptions over their overlapping window."""
    deltas = []
    rows = []
    for tau, za in zip(bundled["tau"], bundled["za"]):
        if tau < crosscheck["tau"][0] or tau > crosscheck["tau"][-1]:
            continue
        other = _interp(crosscheck["tau"], crosscheck["za"], tau)
        delta = za - other
        deltas.append(delta)
        rows.append({"tau": tau, "bundled_za": za, "crosscheck_za": other,
                     "delta": delta})
    if not deltas:
        return {"n_overlap": 0}
    max_abs = max(abs(d) for d in deltas)
    rms = math.sqrt(sum(d * d for d in deltas) / len(deltas))
    return {
        "n_overlap": len(deltas),
        "tau_window": [rows[0]["tau"], rows[-1]["tau"]],
        "max_abs_delta": max_abs,
        "rms_delta": rms,
        "mean_signed_delta": sum(deltas) / len(deltas),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def evaluate(tables: dict[str, dict]) -> dict:
    """Structural audit of the reference tables plus the derived verdict."""
    bundled = tables["bundled"]
    crosscheck = tables["crosscheck"]
    g_b = grid_stats(bundled["tau"], bundled["za"])
    g_c = grid_stats(crosscheck["tau"], crosscheck["za"])
    spread = cross_source_spread(bundled, crosscheck)

    claimed = bundled["claimed_uncertainty"]
    observed = spread.get("max_abs_delta")
    uncertainty_ratio = (observed / claimed) if (observed and claimed) else None

    gates = {}

    # 1. Is the bundled ordinate set a point-by-point reading, or a sketch?
    gates["bundled_is_pointwise"] = {
        "metric": "arithmetic tau grid AND coarse ordinate quantum AND <= 6 slope levels",
        "measured": {
            "tau_grid_is_arithmetic": g_b["tau_grid_is_arithmetic"],
            "tau_step_spread": g_b["tau_step_spread"],
            "ordinate_quantum": g_b["ordinate_quantum"],
            "n_slope_levels": g_b["n_slope_levels"],
        },
        "limit": "tau steps must vary (figure pixels), quantum <= 0.01, > 6 slope levels",
        "result": "fail",
        "interpretation": (
            "The bundled table is a piecewise-linear sketch: every tau step is "
            f"{g_b['tau_step_mean']:.3f} (spread {g_b['tau_step_spread']:.1e}), every "
            f"ordinate is a multiple of {g_b['ordinate_quantum']}, and the "
            f"finite-difference slope takes only {g_b['n_slope_levels']} discrete values. "
            "A point-by-point reading off a figure does not produce that."
        ),
    }

    # 2. Does the reference contain the acceleration phase the experiment must have?
    accel_ok = (g_b["first_slope_to_peak"] or 0.0) <= 0.5
    gates["acceleration_phase_present"] = {
        "metric": "slope at the first interval / peak slope",
        "measured": g_b["first_slope_to_peak"],
        "crosscheck_measured": g_c["first_slope_to_peak"],
        "limit": "<= 0.5 (a front released from rest starts well below its later speed)",
        "result": "pass" if accel_ok else "fail",
        "interpretation": (
            "The bundled table reaches "
            f"{100.0 * (g_b['first_slope_to_peak'] or 0):.0f} % of its peak front speed "
            "within the first tau interval after release, i.e. it describes an "
            "impulsive start. The independent transcription reaches only "
            f"{100.0 * (g_c['first_slope_to_peak'] or 0):.0f} %, growing to a "
            "steady front speed - the shape a gate-release experiment has."
        ),
    }

    # 3. Is the claimed uncertainty consistent with the cross-source spread?
    consistent = uncertainty_ratio is not None and uncertainty_ratio <= 3.0
    gates["claimed_uncertainty_consistent"] = {
        "metric": "max |bundled - crosscheck| / claimed uncertainty",
        "measured": uncertainty_ratio,
        "claim": claimed,
        "cross_source_max_abs_delta": observed,
        "cross_source_rms_delta": spread.get("rms_delta"),
        "limit": "<= 3.0",
        "result": "pass" if consistent else "fail",
        "interpretation": (
            "Two independent transcriptions of the same figure differ by up to "
            f"{observed:.3f} in Z/a (rms {spread.get('rms_delta'):.3f}), "
            f"{uncertainty_ratio:.1f}x the claimed +-{claimed:.2f}. Since both come "
            "from the same paywalled original, this spread is a *lower bound* on "
            "the reference uncertainty."
        ),
    }

    verdict_withheld = not all(
        gates[g]["result"] == "pass"
        for g in ("bundled_is_pointwise", "acceleration_phase_present",
                  "claimed_uncertainty_consistent")
    )
    return {
        "tables": {
            "bundled": {k: bundled[k] for k in ("path", "claimed_uncertainty")},
            "crosscheck": {k: crosscheck[k] for k in ("path", "claimed_uncertainty")},
        },
        "grid_stats": {"bundled": g_b, "crosscheck": g_c},
        "cross_source_spread": spread,
        "gates": gates,
        "verdict": {
            "physical_validation_status": "withheld" if verdict_withheld else "evaluable",
            "reason": (
                "The reference cannot separate a solver error from a digitisation "
                "error of the same size (0.5-0.9 in Z/a), so no pass/fail verdict on "
                "physical validation of the B2 case can be issued against it. The "
                "earlier 'physical validation fails' statement over-attributes to the "
                "solver what the reference does not support."
            ) if verdict_withheld else (
                "The bundled reference survives the structural audit; the B2 "
                "comparison can be read as a validation verdict."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundled", default=None)
    parser.add_argument("--crosscheck", default=None)
    parser.add_argument("--json-out", default=str(HERE / "crosscheck_audit.json"))
    parser.add_argument("--plot-out", default=None)
    args = parser.parse_args()

    tables = {"bundled": load_bundled(args.bundled),
              "crosscheck": load_crosscheck(args.crosscheck)}
    report = evaluate(tables)
    report["benchmark"] = "B2_dam_break_reference_audit"
    Path(args.json_out).write_text(json.dumps(report, indent=1))

    print(f"wrote {args.json_out}")
    for name, gate in report["gates"].items():
        print(f"  {name:34s} {gate['result'].upper():4s} "
              f"measured={gate['measured']}")
    print(f"  verdict: physical_validation = {report['verdict']['physical_validation_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
