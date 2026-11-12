"""Audit of the published Couette wall-shear verdict (m16438).

The merged record (`docs/dev_report_couette_verification.md`) reports
``wall shear ratio = 0.6735`` (-32.6 %) at ``H/dx = 8`` and calls it a wall-model
deficit.  That number comes from ``--quick``: ``t_end = 0.15 s = 0.3 H^2/nu``,
i.e. **before the run reached its own predeclared terminal time**
(``VISCOUOUS_TIMES = 1.2 H^2/nu = 0.60 s``, see ``DESIGN.md``), with a steady
drift of 5.45 % against a 1 % gate.  A shear read off a flow that is still
accelerating is not the transmitted wall stress.

This script re-measures the same estimator on the same geometry at the
predeclared terminal time -- only ``t_end`` (and, at the second rung, ``dx``)
changes -- and separates the three candidate explanations:

* **transient**: if the ratio moves to ~1 when t_end is the design value, the
  published deficit was the startup, not the wall;
* **estimator**: the published tau is a one-spacing secant
  ``rho nu (u_1 - u_0)/dx``, which reads the wall layer shape, not the interior
  gradient (the Poiseuille audit measured a -49.7 % bias for the same stencil:
  ``docs/verification_poiseuille_wall_shear_estimator.md``);
* **effective plane**: the wall bands sit ``dx/2`` outside the nominal plane and
  the fitted no-slip planes sit further out, so the solved channel is wider than
  ``H``; the fitted interior slope ``b`` gives ``H_eff = U/b`` and the offset
  ``(H_eff - H)/dx`` is reported in spacing units.

Outputs (all machine-written, nothing transcribed by hand):

* ``benchmarks/couette/out_full/shear_reading.json``
* ``benchmarks/couette/out_full/shear_reading.png``
* ``benchmarks/couette/out_full/summary.md``

Run (CPU, single thread, ~1.5 min for the two rungs):

    PYTHONPATH=src python benchmarks/couette/shear_reading_audit.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_PY = Path(__file__).resolve().parent / "run.py"
OUT = REPO / "benchmarks" / "couette" / "out"
OUT_FULL = REPO / "benchmarks" / "couette" / "out_full"


def load_run_module():
    spec = importlib.util.spec_from_file_location("couette_run", RUN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def row_metrics(entry: dict, height: float, wall_speed: float,
                viscosity: float) -> dict:
    """Reduce one ladder entry to the quantities this audit compares."""
    m = entry["metrics"]
    tau_ana = entry.get("wall_shear_analytic", None)
    if tau_ana is None:
        tau_ana = 1000.0 * viscosity * wall_speed / height
    return {
        "label": entry.get("label"),
        "H_over_dx": entry["resolution_H_over_dx"],
        "t_end": entry["t_end"],
        "steps": entry["steps"],
        "n_fluid": entry["n_fluid"],
        "dt": entry["dt"],
        "steady_drift": entry["steady_drift"],
        "interior_l2": m["interior_l2"],
        "flux_ratio": m["flux_ratio"],
        "wall_shear_ratio_secant": m["wall_shear_ratio"],
        "wall_shear_ratio_secant_top": m["wall_shear_ratio_top"],
        "wall_shear_ratio_fit": m["line_fit_slope_ratio"],
        "tau_secant_Pa": m["wall_shear_measured"],
        "tau_ana_Pa": m["wall_shear_analytic"],
        "slip_bottom_over_dx": m["slip_length_bottom_over_dx"],
        "slip_top_over_dx": m["slip_length_top_over_dx"],
        "H_eff_over_H_from_fit": (1.0 / m["line_fit_slope_ratio"]
                                  if m["line_fit_slope_ratio"] else float("nan")),
        "wall_clock_s": entry["wall_clock_s"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolutions", type=int, nargs="+", default=[8, 16],
                    help="rungs re-run at the predeclared terminal time")
    ap.add_argument("--out", type=Path, default=OUT_FULL)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    run = load_run_module()
    import torch
    torch.set_num_threads(args.threads)
    device = run.resolve_device("cpu")

    height, speed = run.HEIGHT, run.WALL_SPEED
    nu = run.VISCOSITY
    t_end = run.VISCOUS_TIMES * height ** 2 / nu
    t_avg = run.AVERAGE_FRACTION * t_end

    long_runs = []
    for r in args.resolutions:
        t0 = time.time()
        entry = run.run_one(r, nu, speed, run.FRICTION, height, t_end, t_avg,
                            device, label=f"ladder-r{r}-steady")
        long_runs.append(entry)
        red = row_metrics(entry, height, speed, nu)
        print(f"[audit] H/dx={r} t_end={t_end:.3f}s steps={red['steps']} "
              f"drift={red['steady_drift']:.2e} L2={red['interior_l2']:.4f} "
              f"flux={red['flux_ratio']:.4f} "
              f"tau_secant={red['wall_shear_ratio_secant']:.4f} "
              f"tau_fit={red['wall_shear_ratio_fit']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # The published artefact: --quick, t_end = 0.15 s, H/dx = 8.
    published_path = OUT / "results.json"
    published = None
    if published_path.exists():
        pub = json.loads(published_path.read_text())
        if pub.get("ladder"):
            published = row_metrics(pub["ladder"][0], height, speed, nu)

    rows = ([published] if published else []) + \
        [row_metrics(e, height, speed, nu) for e in long_runs]

    payload = {
        "case": "couette_shear_reading_audit",
        "question": "is the published -32.6% wall-shear deficit a wall-model "
                    "defect, a startup transient, or a reading artefact?",
        "reference": run.REFERENCE_NOTE if hasattr(run, "REFERENCE_NOTE") else "",
        "gates": run.GATES,
        "design_t_end_s": t_end,
        "published_artifact": str(published_path),
        "rows": rows,
    }
    (args.out / "shear_reading.json").write_text(json.dumps(payload, indent=2))

    # ---- figure -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = [f"H/dx={r['H_over_dx']}\nt_end={r['t_end']:.2f}s" for r in rows]
    xs = range(len(rows))
    ax1.plot(xs, [r["wall_shear_ratio_secant"] for r in rows], "o-",
             label="1-dx secant (published estimator)")
    ax1.plot(xs, [r["wall_shear_ratio_fit"] for r in rows], "s--",
             label="interior least-squares slope")
    ax1.axhspan(0.95, 1.05, color="0.8", alpha=0.6, label="5 % gate")
    ax1.axhline(1.0, color="k", lw=0.8)
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel(r"$\tau_{meas}/\tau_{ana}$,  $\tau_{ana}=\rho\nu U/H$")
    ax1.set_title("wall shear ratio: same estimator, design $t_{end}$")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    for r in rows:
        ax2.plot([r["slip_bottom_over_dx"], r["slip_top_over_dx"]],
                 [r["interior_l2"], r["interior_l2"]], "o-")
    ax2.set_xlabel("fitted slip length / dx  (bottom, top)")
    ax2.set_ylabel("interior L2 error [-]", color="C0")
    ax2.set_title("slip at the two walls vs interior error")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "shear_reading.png", dpi=140)

    # ---- machine-written summary table ------------------------------------
    def fm(v, n=4):
        return "nan" if isinstance(v, float) and math.isnan(v) else f"{v:.{n}f}"

    lines = [
        "# Couette wall-shear reading audit (machine-generated)",
        "",
        "| run | H/dx | t_end [s] | steps | drift | interior L2 | flux "
        "| tau secant | tau fit | slip bottom/dx | slip top/dx |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['H_over_dx']} | {fm(r['t_end'],3)} "
            f"| {r['steps']} | {fm(r['steady_drift'])} | {fm(r['interior_l2'])} "
            f"| {fm(r['flux_ratio'])} | {fm(r['wall_shear_ratio_secant'])} "
            f"| {fm(r['wall_shear_ratio_fit'])} "
            f"| {fm(r['slip_bottom_over_dx'],3)} "
            f"| {fm(r['slip_top_over_dx'],3)} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
