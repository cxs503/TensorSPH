"""Grade one or more wave-flume records against the literature references.

Usage (from the repository root)::

    PYTHONPATH=src python benchmarks/wave_flume/analyse.py \\
        --record runs/wf_D20.json --record runs/wf_D30.json \\
        --out docs/reports/wave_flume_report.json --figures docs/figures/wave_flume

Each record is the JSON written by :mod:`benchmarks.wave_flume.run`; the grading
itself lives in :mod:`tensorsph.wave_flume_grade` and is unit-tested on synthetic
signals (``tests/test_wave_flume_grade.py``), so this script only assembles the
table, the ladder comparison and the figures.  Nothing here invents a number: the
report is written from the records.

Acceptance criterion of the case-pool entry ``CASE-WAVE-FLUME``: the wave height
at the gauges must be within 10 % of the literature target (piston-transfer
height), and the measured attenuation along the gauge array is compared with the
laminar-boundary-layer reference ``mu`` of :func:`laminar_boundary_layer_damping`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorsph.wave_flume_grade import (  # noqa: E402
    grade_regular_record,
    laminar_boundary_layer_damping,
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def label(record: dict) -> str:
    cfg = record.get("config", {})
    if record.get("failed"):
        return "failed"
    return f"dx={cfg.get('dx', float('nan')):.5f} (H/dx={cfg.get('height_cells', float('nan')):.1f})"


def grade_all(paths, start=None, tolerance=0.10) -> dict:
    rows = []
    for path in paths:
        record = load(Path(path))
        graded = grade_regular_record(record, start=start, tolerance=tolerance)
        graded["record"] = str(path)
        graded["label"] = label(record)
        graded["config"] = record.get("config", {})
        graded["diagnostics"] = record.get("diagnostics", {})
        graded["spec"] = record.get("spec", {})
        rows.append(graded)
    return {"cases": rows, "tolerance": tolerance,
            "criterion": ("CASE-WAVE-FLUME: |H_meas - H_target| / H_target <= 10 % "
                          "at every gauge; attenuation compared with the "
                          "laminar-boundary-layer reference")}


def figure_series(record: dict, graded: dict, path: Path) -> str:
    times = record["times"]
    target = graded["target_height"]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    for x, eta in zip(record["probe_x"], record["probe_eta"]):
        axes[0].plot(times, eta, lw=0.8, label=f"x = {x:.3f} m")
    axes[0].axhline(0.5 * target, color="k", ls="--", lw=0.8)
    axes[0].axhline(-0.5 * target, color="k", ls="--", lw=0.8,
                    label=r"$\pm H_{\rm target}/2$")
    axes[0].set_ylabel(r"$\eta$ [m]")
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")
    axes[0].set_title("Wave-flume gauges, " + graded["label"])

    xs = [g["x"] for g in graded["gauges"] if math.isfinite(g["height"])]
    hs = [g["height"] for g in graded["gauges"] if math.isfinite(g["height"])]
    axes[1].plot(xs, hs, "o-", label="measured $H$ (harmonic fit)")
    hc = [(g["x"], g["height_crossings"]) for g in graded["gauges"]
          if math.isfinite(g.get("height_crossings", float("nan")))]
    if hc:
        axes[1].plot([p[0] for p in hc], [p[1] for p in hc], "s--", ms=3,
                     label="measured $H$ (crossings)")
    axes[1].axhline(target, color="k", lw=1.0, label=f"target $H$ = {target:.4f} m")
    theory = graded["theory"]
    if xs:
        xx = torch.linspace(min(xs) - 0.05, max(xs) + 0.05, 50, dtype=torch.float64)
        axes[1].plot(xx, target * torch.exp(-theory["mu_per_m"] * (xx - min(xs))),
                     ":", label=r"physics: $e^{-\mu x}$, $\mu$ = "
                                f"{theory['mu_per_m']:.2e} 1/m")
    axes[1].set_xlabel("$x$ [m]")
    axes[1].set_ylabel("$H$ [m]")
    axes[1].set_ylim(0.0, 1.4 * max([target] + hs) if hs else 1.0)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def figure_ladder(rows, path: Path) -> str | None:
    usable = [r for r in rows if not r["failed"] and "height_cells" in r]
    if len(usable) < 2:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for r in usable:
        xs = [g["x"] for g in r["gauges"]]
        errs = [100.0 * g.get("error_vs_target", float("nan")) for g in r["gauges"]]
        axes[0].plot(xs, errs, "o-", label=f"H/dx = {r['height_cells']:.1f}")
        axes[1].plot([r["height_cells"]], [r["worst_error_vs_target"] * 100.0],
                     "o", label=f"H/dx = {r['height_cells']:.1f}")
    axes[0].axhline(10.0, color="r", ls="--", lw=1.0, label="10 % gate")
    axes[0].axhline(-10.0, color="r", ls="--", lw=1.0)
    axes[0].set_xlabel("gauge $x$ [m]")
    axes[0].set_ylabel("height error vs target [%]")
    axes[0].legend(fontsize=7)
    axes[1].axhline(10.0, color="r", ls="--", lw=1.0)
    axes[1].set_xlabel("particles per wave height $H/dx$")
    axes[1].set_ylabel("worst |error| [%]")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def figure_profile(record: dict, path: Path) -> str | None:
    profiles = record.get("profiles") or []
    if not profiles:
        return None
    spec = record["spec"]
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    for prof in profiles:
        ax.plot(prof["x"], prof["eta"], lw=0.9, label=f"t = {prof['time']:.2f} s")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel(r"$\eta$ [m]")
    ax.set_title("Free-surface profiles, " + label(record))
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="append", required=True,
                        help="JSON record from benchmarks/wave_flume/run.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--figures", default=None)
    parser.add_argument("--start", type=float, default=None,
                        help="analysis window start [s] (default: skip ramp + transit)")
    parser.add_argument("--tolerance", type=float, default=0.10)
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.record]
    report = grade_all(paths, start=args.start, tolerance=args.tolerance)
    figures = []
    if args.figures:
        fig_dir = Path(args.figures)
        for path, graded, record in zip(paths, report["cases"],
                                        (load(p) for p in paths)):
            if not graded["failed"]:
                figures.append(figure_series(record, graded,
                                             fig_dir / f"series_{path.stem}.png"))
        ladder = figure_ladder(report["cases"], fig_dir / "ladder.png")
        if ladder:
            figures.append(ladder)
        if paths:
            first = load(paths[0])
            profile = figure_profile(first, fig_dir / "profile.png")
            if profile:
                figures.append(profile)
    report["figures"] = figures
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for row in report["cases"]:
        if row["failed"]:
            print(f"{row['label']:<24} FAILED: {row['error']}")
            continue
        errors = [100.0 * g.get("error_vs_target", float("nan")) for g in row["gauges"]]
        print(f"{row['label']:<24} target {row['target_height']:.4f} m  "
              f"height errors [%] {['%.1f' % e for e in errors]}  "
              f"worst {100 * row['worst_error_vs_target']:.1f} %  "
              f"mu_meas {row['attenuation']['mu_per_m']:.2e} 1/m  "
              f"mu_theory {row['theory']['mu_per_m']:.2e} 1/m  "
              f"passed {row['passed']}")
    print(f"wrote {out} (+{len(figures)} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())