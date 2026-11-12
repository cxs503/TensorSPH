#!/usr/bin/env python3
"""Bounded-gap Stokes second problem (oscillatory Couette): verification.

The case, its analytic solution and the measurement (complex amplitude and
phase per lattice row, effective-viscosity fit) live in
``tensorsph.stokes2``.  This script runs the resolution ladder, writes the
artifact JSON, renders the figures and prints the report tables -- all from the
JSON, so no number in the report is typed by hand.

Usage
-----
    PYTHONPATH=src python benchmarks/stokes2/run.py --out benchmarks/stokes2/out
    PYTHONPATH=src python benchmarks/stokes2/run.py --from-json artifacts/stokes2/ladder.json

The second form only re-renders tables and figures from an existing run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from tensorsph.stokes2 import run_case

DEFAULT_CASES = dict(gap=0.16, viscosity=0.002, omega=2.5, amplitude=0.05,
                     periods=2.0)
DEFAULT_DX = (0.01, 0.006666666666666667, 0.005)


def run_ladder(out_dir: Path, dxs, cases, history_dx=None) -> list:
    results = []
    for dx in dxs:
        t0 = time.time()
        print(f"[ladder] dx = {dx:g}", flush=True)
        res = run_case(dx=dx, progress=True, device="cpu",
                       store_history=(history_dx is not None
                                      and abs(dx - history_dx) < 1e-12),
                       history_every=4, **cases)
        res["input"] = {"dx": dx, **cases}
        res["wall_clock_total_s"] = time.time() - t0
        results.append(res)
        (out_dir / "ladder.json").write_text(json.dumps(results, indent=1))
    return results


def table_rows(results) -> str:
    """Markdown table of the ladder, generated from the artifacts."""
    head = ("| delta/dx | rows | steps/period | relative L2 | "
            "max |amp err|/U0 | max phase err (deg) | nu_eff/nu | L2 at nu_eff |\n"
            "|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in results:
        m = r["metrics"]
        g = r["geometry"]
        i = r["integration"]
        lines.append(
            f"| {g['delta_over_dx']:.2f} | {g['rows']} | "
            f"{i['steps_per_period']:.0f} | {m['relative_l2'] * 100:.2f}% | "
            f"{m['max_amplitude_error_over_U0'] * 100:.2f}% | "
            f"{m['max_phase_error_deg']:.2f} | "
            f"{m['nu_effective_over_nu']:.4f} | "
            f"{m['relative_l2_at_nu_effective'] * 100:.2f}% |")
    return head + "\n".join(lines) + "\n"


def probe_table(result, rows=(0, 1, 2, 3)) -> str:
    """Per-row amplitude and phase for a single resolution."""
    r = result["rows"]
    head = ("| y/delta | amplitude meas (m/s) | amplitude analytic (m/s) | "
            "err/U0 | phase meas (deg) | phase analytic (deg) | phase err (deg) |\n"
            "|---|---|---|---|---|---|---|\n")
    lines = []
    for k in rows:
        if k >= len(r["y"]):
            continue
        lines.append(
            f"| {r['y_over_delta'][k]:.3f} | {r['amplitude_measured'][k]:.5f} | "
            f"{r['amplitude_analytic'][k]:.5f} | "
            f"{r['amplitude_error_over_U0'][k] * 100:+.2f}% | "
            f"{r['phase_measured_deg'][k]:+.2f} | "
            f"{r['phase_analytic_deg'][k]:+.2f} | "
            f"{r['phase_error_deg'][k]:+.2f} |")
    return head + "\n".join(lines) + "\n"


def figures(results, out_dir: Path) -> list:
    """Amplitude/phase profiles, the u(y,t) space-time comparison, the ladder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    made = []
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. amplitude profile, measured vs analytic, every resolution
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = cm.viridis([0.15 + 0.7 * k / max(1, len(results) - 1)
                         for k in range(len(results))])
    for r, c in zip(results, colors):
        g = r["geometry"]
        yod = r["rows"]["y_over_delta"]
        ax.plot(yod, [a / r["parameters"]["amplitude"]
                      for a in r["rows"]["amplitude_measured"]],
                "o", ms=4, color=c,
                label=f"SPH delta/dx={g['delta_over_dx']:.1f}")
        ax.plot(yod, [a / r["parameters"]["amplitude"] for a in r["rows"]["amplitude_analytic"]],
                "-", lw=1.2, color=c, alpha=0.55)
    ax.set_xlabel(r"$y/\delta$")
    ax.set_ylabel(r"$|\hat{u}|/U_0$")
    ax.set_title("Stokes 2nd problem: amplitude envelope (o SPH, line analytic)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    p = fig_dir / "amplitude_profile.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    made.append(p)

    # 2. phase error per row
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r, c in zip(results, colors):
        ax.plot(r["rows"]["y_over_delta"], r["rows"]["phase_error_deg"],
                "o-", ms=3.5, lw=1.1, color=c,
                label=f"delta/dx={r['geometry']['delta_over_dx']:.1f}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$y/\delta$")
    ax.set_ylabel("phase error (deg)")
    ax.set_title("Phase lag error against the analytic solution")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    p = fig_dir / "phase_error.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    made.append(p)

    # 3. space-time field, measured vs analytic (needs the stored history)
    hist = next((r for r in results if r.get("history")), None)
    if hist is not None:
        t = torch.tensor(hist["history"]["t"])
        yod = torch.tensor(hist["rows"]["y_over_delta"])
        meas = torch.tensor(hist["history"]["u_measured_rows"]).T   # rows x time
        ana = torch.tensor(hist["history"]["u_analytic_rows"]).T
        U0 = hist["parameters"]["amplitude"]
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
        for ax, field, title in ((axes[0], ana, "analytic"),
                                 (axes[1], meas, "WCSPH")):
            im = ax.pcolormesh(t, yod, field / U0, shading="auto",
                               cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_title(f"{title}  $u_x/U_0$", fontsize=10)
            ax.set_xlabel("t (s)")
        axes[0].set_ylabel(r"$y/\delta$")
        fig.colorbar(im, ax=axes, label="$u_x/U_0$")
        p = fig_dir / "spacetime.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        made.append(p)

    # 4. convergence of the error with the wall-layer resolution
    if len(results) > 1:
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        x = [r["geometry"]["delta_over_dx"] for r in results]
        ax.loglog(x, [r["metrics"]["relative_l2"] for r in results], "o-",
                  label="relative $L_2$")
        ax.loglog(x, [r["metrics"]["max_amplitude_error_over_U0"]
                      for r in results], "s--", label=r"max $|\Delta\hat u|/U_0$")
        ax.set_xlabel(r"$\delta/\Delta x$")
        ax.set_ylabel("error")
        ax.set_title("Error vs wall-layer resolution")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
        p = fig_dir / "convergence.png"
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        made.append(p)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="benchmarks/stokes2/out")
    ap.add_argument("--from-json", default=None,
                    help="re-render tables/figures from an existing ladder.json")
    ap.add_argument("--dx", default=None,
                    help="comma-separated dx list (default: the three-point ladder)")
    ap.add_argument("--periods", type=float, default=DEFAULT_CASES["periods"])
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_json:
        results = json.loads(Path(args.from_json).read_text())
    else:
        dxs = (tuple(float(v) for v in args.dx.split(",")) if args.dx
               else DEFAULT_DX)
        cases = dict(DEFAULT_CASES)
        cases["periods"] = args.periods
        results = run_ladder(out_dir, dxs, cases, history_dx=dxs[0])

    (out_dir / "ladder.json").write_text(json.dumps(results, indent=1))
    figures_made = figures(results, out_dir)
    table = table_rows(results)
    (out_dir / "tables.md").write_text(
        "<!-- generated by benchmarks/stokes2/run.py; do not hand-edit -->\n\n"
        "## Resolution ladder\n\n" + table + "\n## Probe rows\n\n"
        + probe_table(results[0]) + "\n")
    print(table)
    print(probe_table(results[0]))
    print("figures:", *[str(p) for p in figures_made], sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())