"""Resolution-ladder / self-convergence analysis for the TVC cross-validation.

Why this script exists
----------------------
``docs/xval_sphinx_tvc/README.md`` grades the claim *"the ported transport
velocity correction improves convergence"* as **WITHHELD**: the two grids
available at the time (``dx = 0.04``, ``0.03``) show the error against the
Martin & Moyce experiment falling (17.0 % -> 10.8 % mean relative error), but
two levels are a trend, not a demonstrated order, and the experimental
reference itself carries a digitisation uncertainty of +-0.05 in ``Z/a``.

This script closes that gap with two independent legs:

1. **Self-convergence (no experimental input).** For a variant, the front
   history ``Z/a(tau)`` at two neighbouring resolutions is compared on a common
   ``tau`` grid; the norm of the difference is a *Cauchy* difference.  It must
   fall as ``dx -> 0``; if three or more levels are present, the observed order
   follows from the difference ratio,
   ``p = log(||f_coarse - f_mid|| / ||f_mid - f_fine||) / log(r)``.
   This leg does not depend on the experiment at all, so it cannot be biased by
   the digitisation uncertainty.
2. **Extrapolated limit vs experiment.** A Richardson extrapolation of the two
   finest levels (``f* = (r^p f_fine - f_coarse) / (r^p - 1)``) estimates the
   ``dx -> 0`` front history; comparing *that* with the experimental data says
   whether the remaining gap is discretisation or *model* error (missing
   physics), which no amount of refinement can remove.

Both legs are pure post-processing of the raw per-``tau`` histories written by
``benchmarks/xval_sphinx_tvc/run.py`` -- no numbers are typed in by hand.

Usage
-----
::

    python benchmarks/xval_sphinx_tvc/convergence.py \
        --inputs docs/xval_sphinx_tvc/data/ladder_dense.json \
        --variants baseline tvc_a010_all \
        --out docs/xval_sphinx_tvc/data/convergence_ladder.json

``--inputs`` accepts several payload files (they are merged on ``dx``); the
earlier two-level payloads under ``docs/xval_sphinx_tvc/data/`` can be mixed
with the new dense ladder.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EXP = ROOT / "benchmarks" / "dam_break" / "martin_moyce_1952.json"
DEFAULT_FIG = ROOT / "docs" / "xval_sphinx_tvc" / "figure_convergence_ladder.png"


# --------------------------------------------------------------------------- #
# curve helpers
# --------------------------------------------------------------------------- #
def interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    """Piecewise-linear interpolation, clamped at both ends.

    Same convention as ``run.py`` so the metrics are comparable.
    """
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xs[i] >= x:
            w = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[-1]


def common_grid(series: Iterable[Sequence[float]], n: int = 401) -> list[float]:
    """Uniform grid on the ``tau`` interval shared by *every* series."""
    lo = max(s[0] for s in series)
    hi = min(s[-1] for s in series)
    if not hi > lo:
        raise ValueError(f"no common tau interval: [{lo}, {hi}]")
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def on_grid(taus: Sequence[float], za: Sequence[float],
            grid: Sequence[float]) -> list[float]:
    return [interp(taus, za, t) for t in grid]


def cauchy_norm(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """``(L2_rms, Linf)`` of the pointwise difference of two curves."""
    d = [x - y for x, y in zip(a, b)]
    l2 = math.sqrt(sum(e * e for e in d) / len(d))
    return l2, max(abs(e) for e in d)


def observed_order(e_coarse: float, e_fine: float, r: float) -> float | None:
    """``p`` from two successive Cauchy norms and the refinement ratio ``r``."""
    if e_coarse <= 0.0 or e_fine <= 0.0 or r <= 1.0:
        return None
    return math.log(e_coarse / e_fine) / math.log(r)


def richardson(coarse: Sequence[float], fine: Sequence[float],
               r: float, p: float) -> list[float]:
    """``dx -> 0`` estimate from two levels, refinement ratio ``r``, order ``p``."""
    rp = r ** p
    if abs(rp - 1.0) < 1e-12:
        raise ValueError("degenerate Richardson factor r^p == 1")
    return [(rp * f - c) / (rp - 1.0) for c, f in zip(coarse, fine)]


# --------------------------------------------------------------------------- #
# payload handling
# --------------------------------------------------------------------------- #
def load_payloads(paths: Sequence[Path]) -> dict[str, dict[float, dict]]:
    """Merge payload files into ``{variant: {dx: record}}`` (later files win)."""
    out: dict[str, dict[float, dict]] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        for rec in payload.get("records", []):
            out.setdefault(rec["variant"], {})[float(rec["dx"])] = rec
    return out


def level_series(records: dict[float, dict]) -> list[tuple[float, list, list]]:
    """``[(dx, tau, Z/a)]`` sorted coarse -> fine (refinement ratio ``r > 1``)."""
    return [(dx, records[dx]["tau"], records[dx]["Z_over_a"])
            for dx in sorted(records, reverse=True)]


def experiment_curve(exp: dict) -> tuple[list[float], list[float]]:
    data = exp["data"]
    return [d["tau"] for d in data], [d["Z_over_a"] for d in data]


def error_vs_experiment(taus: Sequence[float], za: Sequence[float],
                        exp: dict) -> dict:
    """Mean absolute / mean relative error of a curve against the experiment."""
    ts, zs = experiment_curve(exp)
    rows = []
    for t_ref, z_ref in zip(ts, zs):
        if t_ref < taus[0] or t_ref > taus[-1]:
            continue
        got = interp(taus, za, t_ref)
        rows.append((abs(got - z_ref), abs(got - z_ref) / z_ref if z_ref else None))
    if not rows:
        return {"points": 0}
    rel = [r for _, r in rows if r is not None]
    return {
        "points": len(rows),
        "mae": sum(a for a, _ in rows) / len(rows),
        "mean_rel_error": sum(rel) / len(rel) if rel else None,
        "max_abs_error": max(a for a, _ in rows),
        "within_digitisation_uncertainty": sum(
            1 for a, _ in rows if a <= 0.05) / len(rows),
    }


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def analyse_variant(records: dict[float, dict], exp: dict,
                    assumed_order: float | None = None,
                    n_grid: int = 401) -> dict:
    """Cauchy differences, observed orders and the Richardson limit."""
    levels = level_series(records)
    if len(levels) < 2:
        return {"levels": [dx for dx, _, _ in levels],
                "note": "fewer than two levels: no convergence metric"}
    grid = common_grid([tau for _, tau, _ in levels], n=n_grid)
    curves = [(dx, on_grid(tau, za, grid)) for dx, tau, za in levels]

    pairs = []
    for (dx_c, c), (dx_f, f) in zip(curves[:-1], curves[1:]):
        r = dx_c / dx_f
        l2, linf = cauchy_norm(c, f)
        pairs.append({"dx_coarse": dx_c, "dx_fine": dx_f, "r": r,
                      "cauchy_l2": l2, "cauchy_linf": linf})

    orders = []
    for a, b in zip(pairs[:-1], pairs[1:]):
        r_mean = math.sqrt(a["r"] * b["r"])
        p = observed_order(a["cauchy_l2"], b["cauchy_l2"], r_mean)
        orders.append({"dx_fine_interval": [a["dx_fine"], b["dx_fine"]],
                       "r": r_mean, "observed_order_l2": p,
                       "cauchy_ratio": a["cauchy_l2"] / b["cauchy_l2"]})

    # Richardson limit from the two finest levels
    dx_c, coarse = curves[-2]
    dx_f, fine = curves[-1]
    r = dx_c / dx_f
    p_used = assumed_order
    source = "assumed"
    if p_used is None and orders:
        p_used = orders[-1]["observed_order_l2"]
        source = "measured (finest interval)"
    limit = None
    if p_used is not None and math.isfinite(p_used) and p_used > 0:
        limit = richardson(coarse, fine, r, p_used)

    per_level = {}
    for dx, tau, za in levels:
        per_level[f"{dx}"] = error_vs_experiment(tau, za, exp)

    out = {
        "levels": [dx for dx, _, _ in levels],
        "grid_points": n_grid,
        "pairs": pairs,
        "orders": orders,
        "richardson": None,
        "per_level_vs_experiment": per_level,
        "cauchy_slope_fit": _fit_order([(p_["dx_fine"], p_["cauchy_l2"])
                                        for p_ in pairs]),
    }
    if limit is not None:
        out["richardson"] = {
            "p_used": p_used, "p_source": source,
            "from_levels": [dx_c, dx_f], "r": r,
            "vs_experiment": error_vs_experiment(grid, limit, exp),
        }
    return out


def _fit_order(points: Sequence[tuple[float, float]]) -> dict | None:
    """Least-squares slope of ``log(cauchy)`` against ``log(dx)`` (all pairs)."""
    pts = [(math.log(dx), math.log(e)) for dx, e in points if e > 0 and dx > 0]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    slope = sxy / sxx if sxx > 0 else float("nan")
    return {"slope": slope, "points": n,
            "note": "slope 1 = first order, 2 = second order in dx"}


def analyse(payloads: dict[str, dict[float, dict]], exp: dict,
            variants: Sequence[str] | None = None) -> dict:
    names = list(variants) if variants else list(payloads)
    results = {}
    for name in names:
        if name not in payloads:
            continue
        results[name] = analyse_variant(payloads[name], exp)
    return results


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_figure(payloads: dict[str, dict[float, dict]], exp: dict,
                variants: Sequence[str], out_png: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ts, zs = experiment_curve(exp)
    ax0.plot(ts, zs, "k^", ms=4, label="Martin & Moyce 1952 (digitised)")

    colours = {"baseline": "tab:red", "tvc_a010_all": "tab:blue"}
    for name in variants:
        if name not in payloads:
            continue
        for dx, tau, za in level_series(payloads[name]):
            ax0.plot(tau, za, lw=1.0, alpha=0.75,
                     color=colours.get(name, "tab:gray"),
                     label=f"{name} dx={dx:g}")
    ax0.set_xlabel(r"$\tau = t\sqrt{2g/a}$")
    ax0.set_ylabel("Z/a")
    ax0.set_title("front history vs resolution")
    ax0.legend(fontsize=6, loc="lower right")
    ax0.grid(alpha=0.3)

    for name in variants:
        if name not in payloads:
            continue
        res = analyse_variant(payloads[name], exp)
        if "pairs" not in res:
            continue
        dxs = [p["dx_fine"] for p in res["pairs"]]
        l2 = [p["cauchy_l2"] for p in res["pairs"]]
        ex = [res["per_level_vs_experiment"][f"{dx:g}"]["mae"]
              for dx in res["levels"]]
        col = colours.get(name, "tab:gray")
        ax1.loglog(dxs, l2, "o-", color=col, label=f"{name}: Cauchy L2")
        ax1.loglog(res["levels"], ex, "s--", color=col, alpha=0.6,
                   label=f"{name}: MAE vs exp")
    ax1.set_xlabel("dx [m]")
    ax1.set_ylabel("error in Z/a")
    ax1.set_title("convergence (slope 1 / 2 reference)")
    ref_dx = sorted({dx for name in variants if name in payloads
                     for dx in payloads[name]})
    if ref_dx:
        base = ref_dx[-1]
        y0 = 0.06
        ax1.loglog([base, ref_dx[0]],
                   [y0, y0 * (ref_dx[0] / base)], "k:", lw=0.8, alpha=0.5)
        ax1.loglog([base, ref_dx[0]],
                   [y0, y0 * (ref_dx[0] / base) ** 2], "k:", lw=0.8, alpha=0.5)
    ax1.legend(fontsize=6)
    ax1.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return [str(out_png)]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=["baseline", "tvc_a010_all"])
    ap.add_argument("--experiment", type=Path, default=DEFAULT_EXP)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "xval_sphinx_tvc" / "data"
                    / "convergence_ladder.json")
    ap.add_argument("--figure", type=Path, default=DEFAULT_FIG)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    exp = json.loads(args.experiment.read_text())
    payloads = load_payloads(args.inputs)
    results = analyse(payloads, exp, args.variants)

    figures = []
    if not args.no_figure:
        figures = make_figure(payloads, exp, args.variants, args.figure)

    payload = {
        "anchor": "XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION",
        "analysis": "resolution ladder: Cauchy self-convergence + Richardson "
                    "limit vs Martin & Moyce",
        "environment": {"python": platform.python_version(),
                        "machine": platform.machine()},
        "inputs": [str(p) for p in args.inputs],
        "variants": results,
        "figures": figures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    for name, res in results.items():
        print(f"== {name}: levels {res['levels']}")
        for p in res.get("pairs", []):
            print(f"   dx {p['dx_coarse']:g}->{p['dx_fine']:g} r={p['r']:.2f} "
                  f"Cauchy L2={p['cauchy_l2']:.4f} Linf={p['cauchy_linf']:.4f}")
        for o in res.get("orders", []):
            po = o["observed_order_l2"]
            print(f"   order over {o['dx_fine_interval']}: "
                  f"p={po if po is None else round(po, 2)}")
        ex = res.get("per_level_vs_experiment", {})
        for dx, e in sorted(ex.items(), key=lambda kv: float(kv[0])):
            print(f"   dx={dx}: MAE={e.get('mae', float('nan')):.4f} "
                  f"rel={e.get('mean_rel_error', float('nan')):.3f}")
        if res.get("richardson"):
            r = res["richardson"]
            e = r["vs_experiment"]
            print(f"   Richardson (p={r['p_used']:.2f}, {r['p_source']}) vs exp: "
                  f"MAE={e['mae']:.4f} rel={e['mean_rel_error']:.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())