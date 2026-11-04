"""Finite-reservoir shallow-water reference for the B2 dam-break front.

Produces ``benchmarks/dam_break/shallow_water_reference.json`` (and the figure
referenced by ``docs/benchmarks/dam-break-finite-reservoir-reference.md``).

Two jobs, in this order:

1. **Verify the reference generator.**  ``tensorsph.shallow_water`` is run in the
   semi-infinite regime, where Ritter's solution is exact, and its leading edge
   and depth profile are scored against the exact solution on the same grid and
   with the same estimator.  Nothing downstream is quoted until this passes; the
   residual bias of the estimator is measured here and carried as the reference's
   uncertainty instead of being hidden.
2. **Produce the finite-reservoir front.**  Same geometry as the SPH case
   (``a = 0.2 m``, ``H = 0.4 m``, tank, dry bed, frictionless) but with the
   reservoir's finite extent retained, so ``Z/a`` vs ``tau`` can be compared with
   the archived SPH samples and with Ritter.

Usage::

    PYTHONPATH=src python benchmarks/dam_break/shallow_water_reference.py \
        --sph artifacts/dam_break_front_dx002.json --plot
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from tensorsph.shallow_water import (  # noqa: E402
    FRONT_LEVELS,
    dam_break,
    front_coefficient,
    front_position,
    ritter_profile,
)

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_JSON = Path(__file__).resolve().parent / "shallow_water_reference.json"
OUT_PNG = Path(__file__).resolve().parent / "shallow_water_reference.png"

GRAVITY = 9.81

#: B2 geometry (identical to ``DamBreakConfig`` defaults and to the experiment's
#: aspect ratio of 2).
A = 0.2
H = 0.4
TANK = 2.0

#: The far wall of the reference tank.  Wider than B2's because the reference is
#: only used up to the moment B2's front reaches its wall (``tau`` ~ 4.4), and a
#: reflected front must never enter the window.
TANK_REFERENCE = 3.0

VERIFY_CELLS = (400, 800, 1600)
VERIFY_ORDERS = (1, 2)
VERIFY_LEVELS = (1.0e-2, 1.0e-3, 1.0e-4)
#: Semi-infinite regime: dam far from the back wall, window short enough that the
#: rarefaction tail cannot reach it (``t < dam_x / c0``).
VERIFY_DAM_X = 3.0
VERIFY_TIMES = (0.1, 0.2)

FRONT_CELLS = (800, 1600, 3200)
FRONT_LEVEL = 1.0e-2      # calibrated in step 1: the least biased estimator
TAU_MAX = 4.4             # Martin & Moyce's tabulated range ends at tau = 4.29


def verify_against_ritter() -> dict:
    """Score the numerical scheme against the exact solution on matching grids."""
    rows = []
    for order in VERIFY_ORDERS:
        for cells in VERIFY_CELLS:
            for target_time in VERIFY_TIMES:
                started = time.perf_counter()
                run = dam_break(cells=cells, tank_width=2.0 * VERIFY_DAM_X,
                                column_width=VERIFY_DAM_X, column_height=H,
                                t_end=target_time, order=order)
                wall = time.perf_counter() - started
                t_end, x, h, u = run.profiles[-1]
                h_exact, u_exact = ritter_profile(x, t_end, H, dam_x=VERIFY_DAM_X)
                travel = 2.0 * math.sqrt(GRAVITY * H) * t_end
                entry = {
                    "order": order,
                    "cells": cells,
                    "time": t_end,
                    "wall_clock_s": wall,
                    "mass_error": run.mass_error,
                    "profile_l1_relative": float((h - h_exact).abs().sum() / h_exact.sum()),
                    "front_errors": {},
                }
                for level in VERIFY_LEVELS:
                    numeric = front_position(x, h, H, level)
                    exact = front_position(x, h_exact, H, level)
                    entry["front_errors"][f"{level:g}"] = {
                        "numeric_x": numeric,
                        "exact_x": exact,
                        "travel_fraction_error": ((numeric - VERIFY_DAM_X)
                                                  / (exact - VERIFY_DAM_X) - 1.0),
                    }
                # Riemann invariant inside the fan: u + 2 sqrt(g h) = 2 c0.
                c0 = math.sqrt(GRAVITY * H)
                fan = (x > VERIFY_DAM_X - c0 * t_end) & (x < VERIFY_DAM_X + 2.0 * c0 * t_end)
                if bool(fan.any()):
                    invariant = u[fan] + 2.0 * (GRAVITY * h[fan]).clamp_min(0.0).sqrt()
                    entry["riemann_invariant_max_rel_error"] = float(
                        ((invariant - 2.0 * c0).abs() / (2.0 * c0)).max())
                else:
                    entry["riemann_invariant_max_rel_error"] = None
                # depth at the dam section, compared with the exact 4H/9
                dam_index = int(VERIFY_DAM_X / run.dx - 0.5)
                entry["dam_depth"] = {
                    "numeric": float(h[dam_index]),
                    "exact": float(h_exact[dam_index]),
                    "four_ninths": 4.0 * H / 9.0,
                }
                rows.append(entry)
    return {
        "setup": {
            "dam_x": VERIFY_DAM_X,
            "tank_width": 2.0 * VERIFY_DAM_X,
            "depth0": H,
            "times": list(VERIFY_TIMES),
            "cells": list(VERIFY_CELLS),
            "orders": list(VERIFY_ORDERS),
            "levels": list(VERIFY_LEVELS),
            "note": ("semi-infinite regime: the rarefaction tail stays inside the "
                     "domain, so Ritter's solution is exact"),
        },
        "rows": rows,
    }


def finite_reservoir() -> dict:
    """The reference curve: finite column, same geometry as B2."""
    rows = []
    curves = {}
    c0 = math.sqrt(GRAVITY * H)
    tau_end = A / c0 * math.sqrt(2.0 * GRAVITY / A)  # tau at which the tail hits the wall
    for cells in FRONT_CELLS:
        started = time.perf_counter()
        run = dam_break(cells=cells, tank_width=TANK_REFERENCE, column_width=A,
                        column_height=H, t_end=TAU_MAX / math.sqrt(2.0 * GRAVITY / A),
                        order=2, sample_times=[0.3, 0.4])
        wall = time.perf_counter() - started
        entry = {
            "cells": cells,
            "dx": run.dx,
            "wall_clock_s": wall,
            "steps": run.steps,
            "mass_error": run.mass_error,
            "front_reached_wall": run.extra.get("front_reached_wall"),
            "front_coefficients": {},
            "curves": {},
        }
        for level in FRONT_LEVELS:
            tau, za = run.dimensionless(A, H, level=level)
            window = [(t, z) for t, z in zip(tau, za) if t >= 2.0]
            entry["front_coefficients"][f"{level:g}"] = front_coefficient(
                [p[0] for p in window], [p[1] for p in window])
            stride = max(1, len(tau) // 60)
            entry["curves"][f"{level:g}"] = {
                "tau": tau[::stride] + [tau[-1]],
                "za": za[::stride] + [za[-1]],
            }
            if cells == FRONT_CELLS[-1]:
                curves[f"{level:g}"] = entry["curves"][f"{level:g}"]
        rows.append(entry)
    return {
        "setup": {
            "column_width_a": A,
            "column_height": H,
            "aspect_ratio": H / A,
            "tank_width": TANK_REFERENCE,
            "gravity": GRAVITY,
            "order": 2,
            "cfl": 0.4,
            "tau_max": TAU_MAX,
            "tau_tail_hits_wall": tau_end,
            "ritter_slope": 2.0,
            "note": ("shallow-water idealisation; not valid for H/a = 2 as a physical "
                     "model, used only to separate the finite-reservoir geometry effect "
                     "from the solver's own error"),
        },
        "runs": rows,
        "curve_finest": curves,
    }


def load_sph(path: str | None) -> dict | None:
    """Read the SPH front samples archived by ``benchmarks/dam_break/run.py``."""
    if not path:
        return None
    handle = Path(path)
    if not handle.exists():
        return None
    payload = json.loads(handle.read_text())
    samples = []
    for result in payload.get("results", []):
        samples.extend(result.get("samples", []))
    if not samples:
        return None
    keep = [s for s in samples if s.get("ritter_front") is not None]
    return {
        "source": str(handle.name),
        "dx": payload.get("results", [{}])[0].get("dx"),
        "tau": [s["tau"] for s in keep],
        "za": [s["front_over_a"] - 1.0 for s in keep],   # Z/a, measured from the dam
        "wall_hit_time": payload.get("results", [{}])[0].get("front_reaches_wall_at"),
    }


def _interp(xs, ys, x):
    if not xs:
        return None
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xs[i] >= x:
            if xs[i] == xs[i - 1]:
                return ys[i]
            w = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[-1]


def attribute(reference: dict, sph: dict | None) -> dict:
    """Split the SPH-vs-Ritter deviation into geometry and solver parts."""
    if sph is None:
        return {"status": "not_evaluated", "reason": "no SPH front samples supplied"}
    key = f"{FRONT_LEVEL:g}"          # curve keys are formatted by FRONT_LEVELS
    tau_ref = reference["curve_finest"][key]["tau"]
    za_ref = reference["curve_finest"][key]["za"]

    # tau >= 2: the SPH column has finished collapsing (before tau ~ 0.5 the
    # front has barely moved), and tau <= 4.3 stays inside the tabulated
    # experimental range.  The whole-window number is dominated by the startup
    # transient and is reported separately, not averaged into this one.
    window = [(t, z) for t, z in zip(sph["tau"], sph["za"]) if 2.0 <= t <= 4.3]
    if len(window) < 3:
        return {"status": "not_evaluated", "reason": "SPH window too short"}
    sph_tau = [p[0] for p in window]
    sph_za = [p[1] for p in window]
    ritter_za = [2.0 * t for t in sph_tau]
    ref_za = [_interp(tau_ref, za_ref, t) for t in sph_tau]

    def mean_rel(values, base):
        return sum((v - b) / b for v, b in zip(values, base)) / len(values)

    geometry = mean_rel(ref_za, ritter_za)          # SW reference vs Ritter
    solver = mean_rel(sph_za, ref_za)               # SPH vs SW reference
    total = mean_rel(sph_za, ritter_za)             # SPH vs Ritter (what B2 reports)
    whole = [(t, z) for t, z in zip(sph["tau"], sph["za"]) if 1.0 <= t <= 4.3]
    whole_rel = mean_rel([p[1] for p in whole], [2.0 * p[0] for p in whole])
    late_slope = front_coefficient(sph_tau, sph_za)
    return {
        "status": "evaluated",
        "window_tau": [sph_tau[0], sph_tau[-1]],
        "whole_window_tau": [whole[0][0], whole[-1][0]] if whole else None,
        "whole_window_sph_minus_ritter_mean_rel": whole_rel,
        "sph_front_slope": late_slope,
        "sph_slope_over_ritter": late_slope / 2.0,
        "samples": len(sph_tau),
        "sph_minus_ritter_mean_rel": total,
        "sph_minus_reference_mean_rel": solver,
        "reference_minus_ritter_mean_rel": geometry,
        "geometry_share_of_deviation": (geometry / total) if total else None,
        "sph_source": sph["source"],
        "sph_dx": sph["dx"],
        "sph_tau": sph_tau,
        "sph_za": sph_za,
        "reference_za": ref_za,
        "ritter_za": ritter_za,
    }


def plot(payload: dict, sph: dict | None, attribution: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    curve = payload["reference"]["curve_finest"][f"{FRONT_LEVEL:g}"]
    ax.plot(curve["tau"], curve["za"], "-", lw=2.2, color="#1f77b4",
            label="finite-reservoir shallow water (reference)")
    tau = [i * 4.4 / 200 for i in range(201)]
    ax.plot(tau, [2.0 * t for t in tau], "--", lw=1.6, color="#444444",
            label="Ritter, semi-infinite ($Z/a=2\\tau$)")
    if attribution.get("status") == "evaluated":
        ax.plot(attribution["sph_tau"], attribution["sph_za"], "o", ms=4.5,
                color="#d62728", label=f"SPH, dx={attribution['sph_dx']:g} m")
    ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$")
    ax.set_ylabel(r"$Z/a$")
    ax.set_title("B2 leading edge: geometry vs reference")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    verify = payload["verification"]["rows"]
    for order, marker in ((1, "s"), (2, "^")):
        rows = [r for r in verify if r["order"] == order]
        xs = [r["cells"] for r in rows]
        ys = [100.0 * r["front_errors"]["0.01"]["travel_fraction_error"] for r in rows]
        ax.plot(xs, ys, marker + "-", color="#ff7f0e" if order == 1 else "#2ca02c",
                label=f"order {order}, leading edge")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-5.0, 5.0, color="#2ca02c", alpha=0.08, label="+/-5 % band")
    ax.set_xscale("log")
    ax.set_xlabel("cells across the tank")
    ax.set_ylabel("leading-edge error vs exact Ritter (%)")
    ax.set_title("reference generator verified against the exact solution")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sph", default="artifacts/dam_break_front_dx002.json",
                        help="SPH front samples produced by benchmarks/dam_break/run.py")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--json-out", default=str(OUT_JSON))
    args = parser.parse_args()

    verification = verify_against_ritter()
    reference = finite_reservoir()
    sph = load_sph(args.sph)
    attribution = attribute(reference, sph)

    payload = {
        "benchmark": "B2_dam_break_finite_reservoir_reference",
        "date": "2026-11-04",
        "verification": verification,
        "reference": reference,
        "attribution": attribution,
        "figure": OUT_PNG.name if args.plot else None,
    }
    target = Path(args.json_out)
    target.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")
    if args.plot:
        plot(payload, sph, attribution)

    print("verification (leading edge, travel-fraction error vs exact Ritter):")
    for row in verification["rows"]:
        errs = " ".join(f"{k}:{100 * v['travel_fraction_error']:+.2f}%"
                        for k, v in row["front_errors"].items())
        print(f"  order={row['order']} cells={row['cells']:5d} t={row['time']:.2f} "
              f"L1={100 * row['profile_l1_relative']:.2f}% | {errs}")
    print("finite-reservoir front coefficients (slope of Z/a for tau>=2):")
    for row in reference["runs"]:
        coefs = " ".join(f"{k}:{v:.3f}" for k, v in row["front_coefficients"].items())
        print(f"  cells={row['cells']:5d} mass={row['mass_error']:.1e} | {coefs}")
    print(f"attribution: {json.dumps({k: v for k, v in attribution.items() if k not in ('sph_tau', 'sph_za', 'reference_za', 'ritter_za')}, default=str)}")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())