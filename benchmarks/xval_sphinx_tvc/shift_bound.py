"""Is the accuracy-optimal TVC coefficient ``alpha`` a *fit* or a *prediction*?

Anchor: ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.

Why this leg exists
-------------------
Every earlier leg grades the ported transport-velocity correction with
``alpha = 0.10``, a value obtained by sweeping ``alpha`` **against the same
Martin & Moyce curve it is then scored on**.  ``UNCALIBRATED_DEFAULTS.md``
states the consequence plainly: with the coefficients SPHinXsys actually ships
(``0.05`` per acoustic step) the owner gate is missed, so the headline
"10.6 % at dx = 0.03 m" is a *fit*, not a prediction.  ``CADENCE_INVARIANCE.md``
then showed the gate-passing point is a genuine ``2x`` **correction-rate**
overshoot relative to the shipped coefficient, and listed "``alpha`` is still
not resolution independent" as an open limitation.

This leg asks whether ``alpha`` can be **predicted without touching the
experimental reference**, from a criterion that is a property of position
correction itself:

    a position correction may move a particle by at most a fraction ``C`` of
    the particle spacing per application, ``|delta r| <= C * dx``

which is the standard shifting/moving-particle bound (Lind et al. 2012 use
``0.25 h``; ``src/tensorsph/particle_shifting.py`` ships exactly that cap), and
in the same family as the ``0.1--0.25 dx`` bounds used by the delta-SPH
particle-shifting literature.  The ported operator's displacement is
``delta r = alpha h^2 g``, so in units of the spacing

    s(alpha) = max_t |delta r| / dx = (h/dx) * alpha * max_t (h |g|)          (1)

and the bound is the equation ``s(alpha) = C``.  ``s`` is **measured**, not
assumed: it is state dependent (the corrected lattice feeds back on ``|g|``),
which is why it is read off the runs instead of evaluated from (1) analytically.

Predeclared hypotheses (written before any run; see ``GRID``/``C_GRID``)
-----------------------------------------------------------------------
H1 (bound sets the optimum).  The ``alpha`` that minimises the front error is
    the ``alpha`` at which the *measured* displacement bound is reached, i.e.
    ``C_implied(dx) = s(alpha_opt(dx))`` is a level-independent constant.
H2 (transferability).  Calibrating ``C`` on **one** resolution and predicting
    ``alpha`` at the others reproduces the gates there -- i.e. the "tuned"
    ``alpha`` is derivable from a criterion that never saw the experiment.

Both are falsifiable: if ``C_implied`` drifts with ``dx`` (as the empirical
``alpha_opt ~ dx^-0.37`` fit of the earlier rounds suggests it must), H1 is
rejected and the honest reading is "``alpha`` is a fit and does not transfer".

Gates
-----
* owner accuracy gate: mean relative front error ``<= 15 %`` (``OWNER_GATE_REL``)
* ledger gate: ``MAE(Z/a) <= 0.15`` (``LEDGER_GATE_ABS``)
* holdout gate: ``|alpha_pred - alpha_opt| / alpha_opt <= 25 %`` on the
  *resolution* axis and the predicted ``alpha`` reaches the owner gate.
* digitisation band: the reference ``Z/a`` carries ``+-0.05``; a verdict that
  flips inside the band is ``PASS_marginal``, never ``PASS``.

Run (single core, ~6 min)::

    python benchmarks/xval_sphinx_tvc/shift_bound.py --resolutions 0.04 0.03 0.02
    python benchmarks/xval_sphinx_tvc/shift_bound.py --resolutions 0.01 \
        --alphas 0.075 0.10 0.125 0.15          # extends the same payload

The solver is *imported* from ``benchmarks/xval_sphinx_tvc/run.py`` (which
imports the in-repo WCSPH case), so no physics is re-implemented here.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import run as R  # noqa: E402  (the anchor's own benchmark driver)

OWNER_GATE_REL = R.OWNER_GATE_REL
LEDGER_GATE_ABS = R.LEDGER_GATE_ABS
BAND = 0.05          # digitisation uncertainty of the reference, in Z/a

# ---------------------------------------------------------------------------
# Predeclared grid.  Nothing below is chosen after looking at a result: the
# alpha grid brackets the upstream-shipped rate (0.05) and the fitted rate
# (0.10) from both sides, and the C grid spans the published shifting-bound
# range (0.1 .. 0.3 dx) with a 0.025 resolution.
# ---------------------------------------------------------------------------
ALPHA_GRID = (0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20)
C_GRID = (0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30)
SECONDS = 0.45
OUTPUT_SAMPLES = 240


def variant_for(alpha: float) -> dict:
    """Solver config of one grid point (``alpha = 0`` is the baseline)."""
    if alpha == 0.0:
        return {}
    return dict(tvc_every=1, tvc_alpha=float(alpha), tvc_scope="all")


def band_bracket(rows: list[dict], sign: float) -> float | None:
    """Mean relative error when the whole reference curve is shifted by
    ``sign * BAND`` -- a systematic digitisation bias, which unlike a random
    one does *not* average out."""
    rels = []
    for r in rows:
        ref = r["Z_over_a_ref"] + sign * BAND
        if ref <= 0:
            continue
        rels.append(abs(r["Z_over_a_got"] - ref) / ref)
    return sum(rels) / len(rels) if rels else None


def run_point(dx: float, alpha: float) -> dict:
    """One (dx, alpha) measurement, including the displacement statistic."""
    name = f"a{alpha:.3f}"
    rec = R.run_variant(name, variant_for(alpha), dx, SECONDS,
                        output_samples=OUTPUT_SAMPLES)
    sc = rec["score"]
    h_over_dx = 1.3  # DamBreakConfig.h_factor: h = 1.3 dx
    shift_over_dx = rec["max_tvc_shift_over_h"] * h_over_dx
    rel = sc.get("mean_rel_error")
    mae = sc.get("mae")
    lo = band_bracket(sc.get("rows", []), -1.0)
    hi = band_bracket(sc.get("rows", []), +1.0)
    if rel is None or mae is None:
        verdict = "n/a"
    elif mae <= LEDGER_GATE_ABS and rel <= OWNER_GATE_REL:
        verdict = "PASS" if (hi is not None and hi <= OWNER_GATE_REL) else "PASS_marginal"
    else:
        verdict = "FAIL"
    return {
        "dx": dx,
        "alpha": alpha,
        "steps": rec["steps"],
        "wall_clock_s": rec["wall_clock_s"],
        "rel_error": rel,
        "mae": mae,
        "rel_error_band_low": lo,
        "rel_error_band_high": hi,
        "max_shift_over_dx": shift_over_dx,
        "max_shift_over_h": rec["max_tvc_shift_over_h"],
        "max_density_compression": rec["max_density_compression"],
        "mass_drift": rec["mass_drift"],
        "escaped": rec["escaped"],
        "verdict": verdict,
        "tau": rec["tau"],
        "Z_over_a": rec["Z_over_a"],
        "score_rows": sc.get("rows", []),
    }


def _interp(x: float, xs: list[float], ys: list[float]) -> float | None:
    if not xs:
        return None
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            if xs[i + 1] == xs[i]:
                return ys[i]
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return None


def analyse_level(points: list[dict]) -> dict:
    """Per-resolution: measured optimum, implied bound, monotonicity of s(alpha)."""
    pts = sorted(points, key=lambda p: p["alpha"])
    grid = [p for p in pts if p["alpha"] > 0 and p["rel_error"] is not None]
    if not grid:
        return {}
    best = min(grid, key=lambda p: p["rel_error"])
    # H1 test statistic: the bound the optimum sits on.
    alphas = [p["alpha"] for p in pts]
    shifts = [p["max_shift_over_dx"] for p in pts]
    c_implied = _interp(best["alpha"], alphas, shifts)
    # is s(alpha) monotone increasing over the grid?  (inversion validity)
    mono = all(shifts[i + 1] >= shifts[i] for i in range(len(shifts) - 1))
    # upstream-rate row, for the "shipped coefficient" comparison
    shipped = next((p for p in grid if abs(p["alpha"] - 0.05) < 1e-9), None)
    return {
        "dx": pts[0]["dx"],
        "alpha_opt": best["alpha"],
        "rel_error_at_opt": best["rel_error"],
        "mae_at_opt": best["mae"],
        "verdict_at_opt": best["verdict"],
        "C_implied": c_implied,
        "shift_curve_monotone": mono,
        "shift_at_opt_over_dx": best["max_shift_over_dx"],
        "shipped_alpha": 0.05,
        "shipped_rel_error": shipped["rel_error"] if shipped else None,
        "shipped_verdict": shipped["verdict"] if shipped else None,
        "baseline_rel_error": next((p["rel_error"] for p in pts
                                    if p["alpha"] == 0.0), None),
        "n_points": len(grid),
    }


def holdout(levels: dict[float, dict], points: dict[float, list[dict]],
            cal_dx: float, target_dx: float) -> dict:
    """Predict ``alpha`` at ``target_dx`` from ``C`` calibrated at ``cal_dx``."""
    cal = levels[cal_dx]
    C = cal["C_implied"]
    pts = sorted(points[target_dx], key=lambda p: p["alpha"])
    alphas = [p["alpha"] for p in pts]
    shifts = [p["max_shift_over_dx"] for p in pts]
    a_pred = _interp(C, shifts, alphas)  # invert s(alpha) = C
    errs = [p["rel_error"] for p in pts]
    e_pred = _interp(a_pred, alphas, errs) if a_pred is not None else None
    a_opt = levels[target_dx]["alpha_opt"]
    return {
        "calibration_dx": cal_dx,
        "target_dx": target_dx,
        "C_calibrated": C,
        "alpha_pred": a_pred,
        "alpha_opt_target": a_opt,
        "alpha_rel_dev": (abs(a_pred - a_opt) / a_opt
                          if a_pred is not None and a_opt else None),
        "rel_error_pred_interp": e_pred,
        "rel_error_at_alpha_opt": levels[target_dx]["rel_error_at_opt"],
        "reaches_owner_gate": (e_pred is not None and e_pred <= OWNER_GATE_REL),
    }


def summarise(payload: dict) -> dict:
    points = {}
    for rec in payload["records"]:
        points.setdefault(rec["dx"], []).append(rec)
    levels = {dx: analyse_level(pts) for dx, pts in points.items()}
    levels = {dx: lv for dx, lv in levels.items() if lv}
    out = {"levels": {f"{dx:g}": lv for dx, lv in sorted(levels.items())}}
    # H1: is C_implied level independent?
    cs = [lv["C_implied"] for lv in levels.values() if lv["C_implied"] is not None]
    if len(cs) >= 2:
        out["C_implied"] = {
            "values": cs,
            "mean": sum(cs) / len(cs),
            "spread_over_mean": (max(cs) - min(cs)) / (sum(cs) / len(cs)),
            "H1_holds_25pct": (max(cs) - min(cs)) / (sum(cs) / len(cs)) <= 0.25,
        }
    # H2: leave-one-resolution-out prediction of alpha
    hold = []
    dxs = sorted(levels)
    for cal in dxs:
        for tgt in dxs:
            if tgt == cal:
                continue
            hold.append(holdout(levels, points, cal, tgt))
    if hold:
        devs = [h["alpha_rel_dev"] for h in hold if h["alpha_rel_dev"] is not None]
        out["holdout"] = hold
        out["H2"] = {
            "max_alpha_rel_dev": max(devs) if devs else None,
            "all_reach_owner_gate": all(h["reaches_owner_gate"] for h in hold),
            "n": len(hold),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolutions", type=float, nargs="+", default=[0.04, 0.03, 0.02])
    ap.add_argument("--alphas", type=float, nargs="+", default=list(ALPHA_GRID))
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "xval_sphinx_tvc" / "data" / "shift_bound.json")
    args = ap.parse_args(argv)

    payload = {
        "anchor": "XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION",
        "leg": "shift-bound prediction of the optimal alpha",
        "predeclared": {
            "alpha_grid": list(ALPHA_GRID),
            "C_grid": list(C_GRID),
            "seconds": SECONDS,
            "output_samples": OUTPUT_SAMPLES,
            "hypotheses": ("H1: C_implied = s(alpha_opt) is level independent; "
                           "H2: C calibrated on one dx predicts alpha at the others"),
        },
        "gates": {"owner_rel": OWNER_GATE_REL, "ledger_abs": LEDGER_GATE_ABS,
                  "band": BAND, "holdout_alpha_dev": 0.25},
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       capture_output=True, text=True).stdout.strip(),
        },
        "records": [],
    }
    if args.out.exists():
        try:
            old = json.loads(args.out.read_text())
            payload["records"] = old.get("records", [])
            payload["environment"] = old.get("environment", payload["environment"])
        except json.JSONDecodeError:
            pass

    def key(rec):
        return (round(rec["dx"], 6), round(rec["alpha"], 6))

    done = {key(r) for r in payload["records"]}
    t0 = time.time()
    for dx in args.resolutions:
        for alpha in args.alphas:
            if (round(dx, 6), round(alpha, 6)) in done:
                print(f"skip dx={dx:g} alpha={alpha:g} (already measured)", flush=True)
                continue
            rec = run_point(dx, alpha)
            payload["records"].append(rec)
            print(f"dx={dx:<5g} alpha={alpha:<6.3f} s/dx={rec['max_shift_over_dx']:.3f} "
                  f"rel={rec['rel_error']:.3f} mae={rec['mae']:.3f} "
                  f"[{rec['verdict']}] {rec['wall_clock_s']:.1f}s", flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=1))

    payload["summary"] = summarise(payload)
    payload["elapsed_s"] = time.time() - t0
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {args.out}  ({payload['elapsed_s']:.0f}s this invocation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())