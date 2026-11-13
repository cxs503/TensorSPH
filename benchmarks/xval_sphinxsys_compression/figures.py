"""Figures + summary tables for the SPHinXsys compression-regularization XVAL.

Reads the record file(s) written by ``run.py`` and emits, into ``--figdir``:

    fig_pressure_profile.png    band-mean pressure vs the analytic hydrostatic line
    fig_band_error.png          relative band error per arm (the comparison table)
    fig_time_series.png         bottom-band pressure, spurious speed and Kref(t)
    fig_compression_profile.png compression K(y) vs the analytic Tait profile

and prints a markdown table of the ref-vs-got numbers (no hand copying: every
number below comes out of the JSON the solver wrote).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "summation": "TensorSPH summation density (baseline)",
    "compression_none": "compression, no reference mean (SPHinXsys <= 2026-07-18)",
    "compression_refmean": "compression, reference mean (SPHinXsys master)",
    "compression_refmean_internal": "compression, reference mean, rule=internal",
}
COLORS = {
    "summation": "#1f77b4",
    "compression_none": "#2ca02c",
    "compression_refmean": "#d62728",
    "compression_refmean_internal": "#ff7f0e",
}


def load(paths):
    records = []
    for path in paths:
        records.extend(json.loads(Path(path).read_text())["records"])
    return records


def figure_pressure(records, dx, out):
    rows = [r for r in records if abs(r["dx"] - dx) < 1e-12]
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ref = rows[0]["profile"]
    ys = [r["y"] for r in ref]
    ax.plot([r["pressure_analytic"] for r in ref], ys, "k--", lw=2,
            label="analytic $\\rho_0 g (H-y)$")
    for r in rows:
        ax.plot([row["pressure"] for row in r["profile"]],
                [row["y"] for row in r["profile"]], "o-", ms=4,
                color=COLORS.get(r["arm"]), label=LABELS.get(r["arm"], r["arm"]))
    ax.set_xlabel("pressure [Pa]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"resting column, dx = {dx*1000:.0f} mm")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return True


def figure_band_error(records, dx, out):
    rows = [r for r in records if abs(r["dx"] - dx) < 1e-12]
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    width = 0.8 / len(rows)
    bands = [r["band"] for r in rows[0]["profile"]]
    for k, r in enumerate(rows):
        xs = [b + (k - (len(rows) - 1) / 2) * width for b in bands]
        ax.bar(xs, [row["rel_error_pct"] for row in r["profile"]], width=width,
               color=COLORS.get(r["arm"]), label=LABELS.get(r["arm"], r["arm"]))
    for level in (3.0, 15.0):
        ax.axhline(level, color="grey", ls=":", lw=1)
        ax.axhline(-level, color="grey", ls=":", lw=1)
    ax.set_xticks(bands)
    ax.set_xticklabels([f"b{b}\n{r['y']*100:.0f} cm"
                        for b, r in zip(bands, rows[0]["profile"])], fontsize=7)
    ax.set_ylabel("band error vs analytic [%]")
    ax.set_title(f"pressure error, dx = {dx*1000:.0f} mm (dotted: 3 % and 15 %)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return True


def figure_series(records, dx, out):
    rows = [r for r in records if abs(r["dx"] - dx) < 1e-12]
    if not rows:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    for r in rows:
        s = r["series"]
        kw = dict(color=COLORS.get(r["arm"]), lw=1.2,
                  label=LABELS.get(r["arm"], r["arm"]))
        axes[0].plot(s["t"], s["p_bottom"], **kw)
        axes[1].plot(s["t"], [1e3 * v for v in s["v_rms"]], **kw)
        axes[2].plot(s["t"], s["kref"], **kw)
    axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("bottom-band p [Pa]")
    axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("RMS speed [mm/s]")
    axes[2].set_xlabel("t [s]"); axes[2].set_ylabel("$K_{ref}$ (inverse mean compression)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    fig.suptitle(f"time histories (averaging window starts at half the run), dx = {dx*1000:.0f} mm",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return True


def figure_compression(records, dx, out):
    rows = [r for r in records if abs(r["dx"] - dx) < 1e-12]
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ref = rows[0]["profile"]
    ax.plot([(row["rho_analytic"] / 1000.0 - 1.0) for row in ref],
            [row["y"] for row in ref], "k--", lw=2, label="analytic Tait inverse")
    for r in rows:
        ax.plot([(row["rho"] / 1000.0 - 1.0) for row in r["profile"]],
                [row["y"] for row in r["profile"]], "o-", ms=4,
                color=COLORS.get(r["arm"]), label=LABELS.get(r["arm"], r["arm"]))
    ax.set_xlabel("$\\rho/\\rho_0 - 1$ [ - ]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"measured compression, dx = {dx*1000:.0f} mm")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--figdir", default="artifacts/xval_sphinxsys_compression/figures")
    ap.add_argument("--dx", type=float, default=None)
    args = ap.parse_args(argv)
    records = load(args.records)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    dxs = sorted({r["dx"] for r in records})
    dx = args.dx if args.dx is not None else dxs[-1]
    made = []
    for name, fn in (("fig_pressure_profile", figure_pressure),
                     ("fig_band_error", figure_band_error),
                     ("fig_time_series", figure_series),
                     ("fig_compression_profile", figure_compression)):
        out = figdir / f"{name}.png"
        if fn(records, dx, out):
            made.append(str(out))
    print("figures:", *made, sep="\n  ")

    print("\n| arm | dx [m] | max gated band p error [%] | max gated band rho error [%] "
          "| RMS speed [m/s] | K_ref | K_mean | wall [s] |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sorted(records, key=lambda r: (r["dx"], r["arm"])):
        print(f"| {r['arm']} | {r['dx']:.3f} | {r['gate']['max_band_error_pct']:.2f} "
              f"| {r['max_rho_band_error_pct']:.3f} | {r['v_rms_mean_ms']:.4f} "
              f"| {r['k_ref_final']:.6f} | {r['k_mean_final']:.6f} "
              f"| {r['wall_seconds']:.0f} |")
    print("\n| arm | band | y [m] | ref p [Pa] | got p [Pa] | err [%] | gated |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted([r for r in records if abs(r["dx"] - dx) < 1e-12],
                    key=lambda r: r["arm"]):
        for row in r["profile"]:
            print(f"| {r['arm']} | {row['band']} | {row['y']:.3f} "
                  f"| {row['pressure_analytic']:.1f} | {row['pressure']:.1f} "
                  f"| {row['rel_error_pct']:+.2f} | {row['gated']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())