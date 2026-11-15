"""Droplet-oscillation leg: is the ported TVC non-intrusive on analytic free-surface dynamics?

What debt this closes
---------------------
Every physics leg of this anchor so far grades the *dam break*: a violent,
gravity-driven, wall-bounded free-surface flow with no exact solution, judged on
the front position against digitised experiment.  Two questions were therefore
open, and they are the ones a user of the ported operator actually asks:

1. **Is the correction confined to the quantity it minimises?**  TVC drives the
   SPH quadrature defect ``h|grad-1|`` down by *moving particles*.  A correction
   that achieves that while changing the resolved dynamics is not a
   regularisation, it is a modelling change.  The dam break cannot separate the
   two: both the defect and the answer move together.
2. **What does the correction do to a free surface in a smooth flow?**  The
   measured dam-break failure mode at ``alpha = 0.20`` is *over-dispersion*
   (median neighbour distance 1.26 dx, no close pairs, front error back up to
   0.563) and the known direction of the surface term is *outward*.  A closed
   free surface is where an outward bias must show up as a size change.

The 2-D capillary drop (Rayleigh 1879; Lamb, *Hydrodynamics* 6th ed., arts.
313-315) answers both, because it has an **exact frequency** and an **exact size
law**::

    omega_m^2 = sigma m (m^2 - 1) / (rho R^3)     =>   T ∝ R^{3/2}

and the case is smooth, free of gravity, free of walls, and (mode ``m = 2``)
area-preserving to first order.  Nothing in it needs calibration, so the
correction's effect cannot hide behind a tuned reference.

The leg runs the production drop-oscillation integrator
(:func:`tensorsph.capillary_oscillation.run_point`) with the ported operator
switched on at several coefficients, and decomposes any period change into

* a **geometric channel** -- the mean radius ``rbar`` moves, and ``T ∝ rbar^{3/2}``
  converts that into a period shift that is *predicted*, not fitted;
* a **dynamic residual** -- the part of the measured period shift the geometry
  does not explain.

If the residual is small the correction acts purely as a re-spacing operator on
this flow (the property the dam-break claims rely on).  If it is not, the
scheme's domain of validity is narrower than the dam-break legs alone suggest,
and that is what this leg is for.

Gate
----
The case's own ledger acceptance is ``|period error| < 15 %``
(``benchmarks/surface_tension/droplet_oscillation.py`` -> ``GATE_PCT``, the
owner standard).  The leg reports that gate for every coefficient, so a
coefficient that would be unsafe to use as a default is visible as a FAIL rather
than as a number in a table.

Run::

    PYTHONPATH=src python benchmarks/xval_sphinx_tvc/droplet_nonintrusive.py \
        --out docs/xval_sphinx_tvc/data/droplet_nonintrusive.json

Every number printed below is read back from that payload.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time

import torch

from tensorsph.capillary_oscillation import (MODE_DEFAULT, RADIUS_DEFAULT,
                                            RHO_WATER, SIGMA_WATER_AIR,
                                            rayleigh_2d, run_point)

GATE_PCT = 15.0          # case ledger acceptance (owner standard)
AMP_ALPHAS = (0.0, 0.05, 0.10, 0.20)
AMP_RESOLUTIONS = (10, 14)   # radius / dx
PERIODS = 1.5


def _damping(series: dict, mode: int | None = None) -> dict:
    """Envelope ratio: second-half vs first-half oscillation range of ``q``."""
    q = series["q"]
    n = len(q)
    if n < 8:
        return {"amp_first_half": float("nan"), "amp_second_half": float("nan"),
                "decay_ratio": float("nan")}
    a = 0.5 * (max(q[: n // 2]) - min(q[: n // 2]))
    b = 0.5 * (max(q[n // 2:]) - min(q[n // 2:]))
    return {"amp_first_half": a, "amp_second_half": b,
            "decay_ratio": (b / a if a > 0.0 else float("nan"))}


def run_matrix(alphas=AMP_ALPHAS, resolutions=AMP_RESOLUTIONS,
               periods: float = PERIODS, sigma: float = SIGMA_WATER_AIR,
               radius: float = RADIUS_DEFAULT, mode: int = MODE_DEFAULT,
               sample_every: int = 2, verbose: bool = True) -> dict:
    """Run the coefficient x resolution matrix and grade every point."""
    torch.set_num_threads(1)
    rows, t_matrix = [], time.time()
    for n in resolutions:
        base = None
        for alpha in alphas:
            t0 = time.time()
            r = run_point(n, sigma, periods, radius=radius, mode=mode,
                          sample_every=sample_every, tvc_alpha=float(alpha))
            r["wall_s"] = round(time.time() - t0, 2)
            damp = _damping(r["series"])
            r["amp_first_half"] = damp["amp_first_half"]
            r["amp_second_half"] = damp["amp_second_half"]
            r["decay_ratio"] = damp["decay_ratio"]
            r["gate_pct"] = GATE_PCT
            r["gate_pass"] = bool(abs(r["period_err_pct"]) < GATE_PCT)
            if alpha == 0.0:
                base = r
            # --- geometric channel: T ∝ R^{3/2}; rbar is the measured size ---
            if base is not None:
                ratio_geom = (r["r_mean_over_R"] / base["r_mean_over_R"]) ** 1.5
                r["period_ratio_measured"] = r["period_measured_s"] / base["period_measured_s"]
                r["period_ratio_geometric_pred"] = ratio_geom
                r["period_ratio_residual_pct"] = 100.0 * (
                    r["period_ratio_measured"] / ratio_geom - 1.0)
                r["period_shift_vs_baseline_pct"] = 100.0 * (
                    r["period_measured_s"] / base["period_measured_s"] - 1.0)
                r["radius_shift_vs_baseline_pct"] = 100.0 * (
                    r["r_mean_over_R"] / base["r_mean_over_R"] - 1.0)
            r.pop("series_keys", None)
            rows.append(r)
            if verbose:
                print(f"  n={n:>2} alpha={alpha:<5} steps={r['steps']:>5} "
                      f"T_err={r['period_err_pct']:+7.3f}% "
                      f"T vs base={r.get('period_shift_vs_baseline_pct', 0.0):+7.3f}% "
                      f"rbar/R={r['r_mean_over_R']:.5f} "
                      f"shift/h={r['tvc_shift_peak_over_h']:.4f} "
                      f"decay={r['decay_ratio']:.3f} "
                      f"gate={'PASS' if r['gate_pass'] else 'FAIL'} ({r['wall_s']}s)",
                      flush=True)
    # compact payload: drop the per-step series (kept for the figure script
    # only where it is needed -> store a decimated copy)
    for r in rows:
        s = r["series"]
        keep = max(1, len(s["t"]) // 400)
        r["series"] = {k: v[::keep] for k, v in s.items()}
    payload = {
        "leg": "droplet-nonintrusive",
        "anchor": "XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION",
        "generated_at_unix": round(time.time(), 1),
        "case": {
            "module": "tensorsph.capillary_oscillation",
            "exact_reference": "omega_m^2 = sigma m (m^2-1)/(rho R^3) "
                               "(Rayleigh 1879; Lamb, Hydrodynamics 6th ed., arts. 313-315)",
            "size_law": "T ∝ R^{3/2}",
            "sigma_n_per_m": sigma, "rho_kg_m3": RHO_WATER,
            "radius_m": radius, "mode": mode, "periods": periods,
            "gate_pct": GATE_PCT,
            "gate_source": "benchmarks/surface_tension/droplet_oscillation.py :: GATE_PCT",
        },
        "matrix": {"alphas": list(alphas), "resolutions": list(resolutions)},
        "wall_s_total": round(time.time() - t_matrix, 2),
        "torch": torch.__version__,
        "rows": rows,
    }
    return payload


def summarise(payload: dict) -> dict:
    """Gate summary per coefficient plus the geometric-channel decomposition."""
    out = {"by_alpha": {}, "worst": {}}
    for r in payload["rows"]:
        a = f"{r['tvc_alpha']:.2f}"
        d = out["by_alpha"].setdefault(a, {"period_err_pct": [], "shift_over_h": [],
                                          "decay_ratio": [], "residual_pct": [],
                                          "radius_shift_pct": [], "levels": []})
        d["period_err_pct"].append(r["period_err_pct"])
        d["shift_over_h"].append(r["tvc_shift_peak_over_h"])
        d["decay_ratio"].append(r["decay_ratio"])
        d["levels"].append(r["radius_over_dx"])
        if "period_ratio_residual_pct" in r:
            d["residual_pct"].append(r["period_ratio_residual_pct"])
            d["radius_shift_pct"].append(r["radius_shift_vs_baseline_pct"])
    for a, d in out["by_alpha"].items():
        d["worst_abs_period_err_pct"] = max(abs(x) for x in d["period_err_pct"])
        d["all_levels_pass_gate"] = bool(d["worst_abs_period_err_pct"] < payload["case"]["gate_pct"])
        d["worst_abs_residual_pct"] = (max(abs(x) for x in d["residual_pct"])
                                       if d["residual_pct"] else 0.0)
        d["mean_radius_shift_pct"] = (sum(d["radius_shift_pct"]) / len(d["radius_shift_pct"])
                                      if d["radius_shift_pct"] else 0.0)
    out["worst"] = max(out["by_alpha"].items(),
                       key=lambda kv: kv[1]["worst_abs_period_err_pct"])
    return out


def write_doc(payload: dict, summary: dict, path: pathlib.Path) -> None:
    """Emit the markdown leg report from the payload (no hand-copied numbers)."""
    c = payload["case"]
    lines = [
        "# XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION -- droplet non-intrusiveness leg",
        "",
        f"_Generated by `benchmarks/xval_sphinx_tvc/droplet_nonintrusive.py` from "
        f"`{payload['_out']}`; every number below is read from that payload._",
        "",
        "## 1. What this leg adds",
        "",
        "All previous physics legs grade the dam break, which has no exact "
        "solution, so they cannot separate *regularisation* from *a change of "
        "the resolved dynamics*. This leg uses the 2-D capillary drop, whose "
        "frequency is exact and whose size law is exact:",
        "",
        f"    {c['exact_reference']}",
        "",
        f"    {c['size_law']}",
        "",
        f"Case: `{c['module']}`, mode m = {c['mode']}, R = {c['radius_m']} m, "
        f"sigma = {c['sigma_n_per_m']} N/m, rho = {c['rho_kg_m3']} kg/m^3, "
        f"{c['periods']} periods. Gate: `|period error| < {c['gate_pct']} %` "
        f"(`{c['gate_source']}`).",
        "",
        "The period change is decomposed into a **geometric channel** "
        f"(`{c['size_law']}`, predicted from the measured mean radius) and the "
        "**dynamic residual** the geometry does not explain.",
        "",
        "## 2. Results",
        "",
        "| n = R/dx | alpha | steps | period err vs exact | period vs baseline "
        "| rbar/R | radius vs baseline | residual | shift/h | envelope decay | gate |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in payload["rows"]:
        lines.append(
            f"| {r['radius_over_dx']} | {r['tvc_alpha']:.2f} | {r['steps']} | "
            f"{r['period_err_pct']:+.3f} % | "
            f"{r.get('period_shift_vs_baseline_pct', 0.0):+.3f} % | "
            f"{r['r_mean_over_R']:.5f} | "
            f"{r.get('radius_shift_vs_baseline_pct', 0.0):+.3f} % | "
            f"{r.get('period_ratio_residual_pct', 0.0):+.3f} % | "
            f"{r['tvc_shift_peak_over_h']:.4f} | {r['decay_ratio']:.3f} | "
            f"{'PASS' if r['gate_pass'] else '**FAIL**'} |")
    lines += ["", "## 3. Gate summary per coefficient", "",
              "| alpha | worst |period err| | all levels pass | worst |dynamic "
              "residual| | mean radius shift | max shift/h |",
              "|---|---|---|---|---|"]
    for a, d in sorted(summary["by_alpha"].items(), key=lambda kv: float(kv[0])):
        lines.append(
            f"| {a} | {d['worst_abs_period_err_pct']:.3f} % | "
            f"{'yes' if d['all_levels_pass_gate'] else '**no**'} | "
            f"{d['worst_abs_residual_pct']:.3f} % | "
            f"{d['mean_radius_shift_pct']:+.3f} % | "
            f"{max(d['shift_over_h']):.4f} |")
    lines += [
        "",
        "## 4. Reading",
        "",
        f"- Worst coefficient overall: alpha = {summary['worst'][0]} at "
        f"{summary['worst'][1]['worst_abs_period_err_pct']:.3f} % period error.",
        f"- Largest unexplained (dynamic) period residual: "
        f"{max(d['worst_abs_residual_pct'] for d in summary['by_alpha'].values()):.3f} %.",
        "- A residual of order the measurement resolution means the correction "
        "behaves on this flow as a pure re-spacing operator: every period change "
        "it causes is accounted for by the change of the mean radius under the "
        "exact size law. A large residual would mean the scheme alters the "
        "surface-tension dynamics itself, i.e. a modelling change, and would "
        "narrow the anchor's stated domain of validity.",
        "",
        "See `docs/xval_sphinx_tvc/VALIDATION_REPORT.md` (limited to the dam "
        "break) and `docs/xval_sphinx_tvc/README.md` for the other legs.",
        "",
    ]
    path.write_text("\n".join(lines))


def make_figure(payload: dict, path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["rows"]
    levels = sorted({r["radius_over_dx"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))

    for n in levels:
        sel = [r for r in rows if r["radius_over_dx"] == n]
        a = [r["tvc_alpha"] for r in sel]
        ax[0].plot(a, [r["period_err_pct"] for r in sel], "o-",
                   label=f"TVC on, R/dx = {n}")
    ax[0].axhline(payload["case"]["gate_pct"], color="r", ls="--", lw=1,
                  label=f"gate {payload['case']['gate_pct']:.0f} %")
    ax[0].axhline(-payload["case"]["gate_pct"], color="r", ls="--", lw=1)
    ax[0].axhline(0.0, color="k", lw=0.8)
    ax[0].set_xlabel(r"correction coefficient $\alpha$")
    ax[0].set_ylabel("period error vs exact Lamb frequency [%]")
    ax[0].set_title("(a) exact-reference gate on analytic free-surface dynamics")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    for n in levels:
        sel = [r for r in rows if r["radius_over_dx"] == n and r["tvc_alpha"] > 0.0]
        if not sel:
            continue
        ax[1].plot([r["tvc_alpha"] for r in sel],
                   [r["period_ratio_residual_pct"] for r in sel], "s-",
                   label=f"dynamic residual, R/dx = {n}")
        ax[1].plot([r["tvc_alpha"] for r in sel],
                   [r["radius_shift_vs_baseline_pct"] for r in sel], "^--",
                   label=f"geometric channel, R/dx = {n}")
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].set_xlabel(r"correction coefficient $\alpha$")
    ax[1].set_ylabel("period change [%]")
    ax[1].set_title(r"(b) geometric channel vs dynamic residual ($T \propto \bar r^{3/2}$)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/xval_sphinx_tvc/data/droplet_nonintrusive.json")
    ap.add_argument("--doc", default="docs/xval_sphinx_tvc/DROPLET_NONINTRUSIVE.md")
    ap.add_argument("--figure", default="docs/xval_sphinx_tvc/figure_droplet_nonintrusive.png")
    ap.add_argument("--alphas", type=float, nargs="+", default=list(AMP_ALPHAS))
    ap.add_argument("--resolutions", type=int, nargs="+", default=list(AMP_RESOLUTIONS))
    ap.add_argument("--periods", type=float, default=PERIODS)
    args = ap.parse_args(argv)

    payload = run_matrix(tuple(args.alphas), tuple(args.resolutions),
                         periods=args.periods)
    summary = summarise(payload)
    payload["summary"] = summary
    payload["_out"] = args.out
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    doc = pathlib.Path(args.doc)
    write_doc(payload, summary, doc)
    fig = pathlib.Path(args.figure)
    fig.parent.mkdir(parents=True, exist_ok=True)
    make_figure(payload, fig)
    print(f"\nwrote {out}, {doc}, {fig}")
    print(f"wall: {payload['wall_s_total']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())