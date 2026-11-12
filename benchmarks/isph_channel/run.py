#!/usr/bin/env python3
"""GAP-SPH-ISPH-PROJECTION: resolution ladder, gates and figures.

Runs the incompressible (projection) SPH channel of
:mod:`tensorsph.isph_channel` over ``H/dx in {8, 12, 16}`` and writes

* ``results.json`` -- every measured quantity plus the three gate verdicts,
* four figures: the measured profile against the analytic parabola, the gate
  errors against resolution, the density fluctuation and the divergence
  reduction history.

Usage::

    PYTHONPATH=src python benchmarks/isph_channel/run.py --out runs/isph_channel/m16024
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorsph.isph_channel import (
    ChannelConfig,
    fit_profile,
    gates,
    operator_signatures,
    run_case,
)


def analytic(y, height, body_force, viscosity):
    return body_force * y * (height - y) / (2.0 * viscosity)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("runs/isph_channel/latest"))
    ap.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--columns", type=int, default=12)
    ap.add_argument("--t-end", type=float, default=1.0)
    ap.add_argument("--t-average", type=float, default=0.1)
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    results = []
    for res in args.resolutions:
        cfg = ChannelConfig(columns=args.columns, resolution=res,
                            t_end=args.t_end, t_average=args.t_average)
        out = run_case(cfg, device)
        fit = fit_profile(out["profile"], cfg.height, cfg.body_force,
                          cfg.viscosity, lo=0.05, hi=0.95)
        out["fit"] = fit
        out["gates"] = gates(out, fit, cfg)
        print(f"[isph] H/dx={res}: rho_err={out['gates']['density_fluctuation_max']['measured']:.3e} "
              f"u_max={out['gates']['centre_velocity']['rel_error_pct']:+.2f}% "
              f"tau={out['gates']['wall_shear_budget']['rel_error_pct']:+.2f}% "
              f"({out['wall_clock_s']:.1f} s)", flush=True)
        results.append(out)

    cfg0 = ChannelConfig(columns=args.columns, resolution=args.resolutions[0])
    payload = {
        "case": "isph_plane_poiseuille_channel",
        "anchor": "GAP-SPH-ISPH-PROJECTION",
        "method": "incompressible SPH projection (Cummins & Rudman 1999; Shao & Lo 2003)",
        "analytic": "u(y)=g y (H-y)/(2 nu); u_max=g H^2/(8 nu); tau_wall=rho g H/2",
        "config": {
            "height": cfg0.height, "columns": args.columns,
            "h_factor": cfg0.h_factor, "wall_layers": cfg0.wall_layers,
            "viscosity": cfg0.viscosity, "body_force": cfg0.body_force,
            "rho0": cfg0.rho0, "t_end": args.t_end, "t_average": args.t_average,
            "xsph": cfg0.xsph,
        },
        "operator_signatures": operator_signatures(cfg0, device),
        "results": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))

    # ---- figures ---------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for r in results:
        y = [p["y_over_H"] for p in r["profile"]]
        u = [p["u_mean"] for p in r["profile"]]
        ax.plot(u, y, "o-", ms=3, lw=1, label=f"H/dx={r['resolution_H_over_dx']}")
    yy = np.linspace(0, 1, 200)
    ref = analytic(yy * cfg0.height, cfg0.height, cfg0.body_force, cfg0.viscosity)
    ax.plot(ref, yy, "k--", lw=1.2, label="analytic")
    ax.set_xlabel("u [m/s]"); ax.set_ylabel("y / H")
    ax.set_title("ISPH channel: mean profile vs analytic parabola")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out / "fig1_profile.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    res = [r["resolution_H_over_dx"] for r in results]
    ax.plot(res, [r["gates"]["centre_velocity"]["rel_error_pct"] for r in results],
            "o-", label="centre velocity")
    ax.plot(res, [r["gates"]["wall_shear_budget"]["rel_error_pct"] for r in results],
            "s-", label="wall shear (fitted wall slope)")
    ax.axhline(3, color="r", ls=":", label="gate 3%")
    ax.axhline(-3, color="r", ls=":")
    ax.set_xlabel("H/dx"); ax.set_ylabel("error [%]")
    ax.set_title("Velocity and wall-shear error vs resolution")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out / "fig2_gates.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for r in results:
        t = [s["t"] for s in r["trace"]]
        d = [s["density_error_max"] for s in r["trace"]]
        ax.semilogy(t, np.maximum(d, 1e-16), lw=1,
                    label=f"H/dx={r['resolution_H_over_dx']}")
    ax.axhline(1e-3, color="r", ls=":", label="gate 1e-3")
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"max $|\rho/\rho_0-1|$")
    ax.set_title("Summation-density fluctuation in time")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(args.out / "fig3_density.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for r in results:
        t = [s["t"] for s in r["trace"]]
        q = [s["div_rms_after"] / max(s["div_rms_before"], 1e-30) for s in r["trace"]]
        ax.plot(t, q, lw=0.8, label=f"H/dx={r['resolution_H_over_dx']}")
    ax.set_xlabel("t [s]"); ax.set_ylabel("div rms after / before")
    ax.set_title("Projection removes the discrete divergence")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out / "fig4_divergence.png", dpi=150)
    plt.close(fig)

    print(f"[isph] wrote {args.out}/results.json and 4 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
