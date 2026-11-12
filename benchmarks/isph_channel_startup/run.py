#!/usr/bin/env python3
"""Transient start-up verification of the ISPH (projection) channel.

Why this benchmark exists
-------------------------
Every gate that ships with :mod:`tensorsph.isph_channel` is a *steady-state*
gate: the end-of-run profile is averaged over the last ``t_average`` seconds and
compared to the Hagen-Poiseuille parabola, the wall shear to a momentum budget,
the divergence to its pre-projection value.  A solver can reproduce the steady
parabola while integrating the transient wrongly -- the steady state is an
attractor of the semi-discrete system, so any consistent discretisation ends
there.  The start-up phase is where the splitting error, the explicit viscous
update and the pressure-projection correction sequence show up.

This benchmark pins the transient against an exact solution.  From rest, with a
constant body force ``g`` and no-slip at ``y = 0, H``, the Navier-Stokes system
reduces to the linear diffusion equation whose solution is the classical series
(see :func:`tensorsph.isph_channel.analytic_startup_profile`)

    u(y,t) = u_s(y) - (4 g H^2 / (nu pi^3)) sum_{n odd} sin(n pi y / H)
                       exp(-nu n^2 pi^2 t / H^2) / n^3 ,

    u_s(y) = g y (H - y) / (2 nu) ,      tau = H^2 / (nu pi^2) .

Both the profile at a set of times and the centreline history are measured.  The
reference is analytic, so there is no experimental scatter to hide behind.

Usage::

    PYTHONPATH=src python benchmarks/isph_channel_startup/run.py \\
        --out runs/isph_channel_startup/latest
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorsph.isph_channel import (
    ChannelConfig,
    analytic_startup_centerline,
    analytic_startup_profile,
    diffusive_time,
    run_case,
)

#: times at which a full profile is compared, as fractions of the diffusive
#: time ``tau``.  ``8 tau`` is ``exp(-64) = 2e-28`` from steady, i.e. steady.
TIME_FRACTIONS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def interior_rows(profile, cfg: ChannelConfig):
    """Rows of the measured profile with ``0.1 H <= y <= 0.9 H``.

    The two rows that touch a no-slip wall carry the wall boundary treatment,
    not the interior stencil, so they are excluded from the *interior* metric and
    reported separately (``--all-rows`` totals are also written out).
    """
    keep = [p for p in profile if 0.1 * cfg.height <= p["y"] <= 0.9 * cfg.height]
    return keep


def compare_profile(profile, t_ref: float, cfg: ChannelConfig) -> dict:
    rows = interior_rows(profile, cfg)
    y = torch.tensor([p["y"] for p in rows], dtype=torch.float64)
    u = torch.tensor([p["u_mean"] for p in rows], dtype=torch.float64)
    exact = analytic_startup_profile(y, t_ref, cfg.height, cfg.body_force, cfg.viscosity)
    diff = u - exact
    rel_l2 = float(torch.linalg.norm(diff) / torch.linalg.norm(exact))
    all_rows = profile
    ya = torch.tensor([p["y"] for p in all_rows], dtype=torch.float64)
    ua = torch.tensor([p["u_mean"] for p in all_rows], dtype=torch.float64)
    ea = analytic_startup_profile(ya, t_ref, cfg.height, cfg.body_force, cfg.viscosity)
    rel_l2_all = float(torch.linalg.norm(ua - ea) / torch.linalg.norm(ea))
    centre_idx = int(torch.argmin(torch.abs(ya - 0.5 * cfg.height)))
    return {
        "t": t_ref,
        "t_over_tau": t_ref / diffusive_time(cfg),
        "n_rows_interior": len(rows),
        "profile_rel_l2_interior": rel_l2,
        "profile_rel_l2_all_rows": rel_l2_all,
        "centre_measured": float(ua[centre_idx]),
        "centre_exact": float(ea[centre_idx]),
        "centre_rel_error_pct": float(
            100.0 * (ua[centre_idx] - ea[centre_idx]) / ea[centre_idx]
        ),
        "profile": [
            {"y_over_H": p["y"] / cfg.height, "u_measured": p["u_mean"],
             "u_exact": float(analytic_startup_profile(
                 torch.tensor([p["y"]], dtype=torch.float64), t_ref,
                 cfg.height, cfg.body_force, cfg.viscosity)[0])}
            for p in profile
        ],
    }


def transient_history(result: dict, cfg: ChannelConfig) -> dict:
    trace = [r for r in result["trace"] if r["t"] > 0.0]
    t = np.array([r["t"] for r in trace], dtype=float)
    u = np.array([r["u_centre_rows"] for r in trace], dtype=float)
    exact = np.array([analytic_startup_centerline(x, cfg) for x in t], dtype=float)
    # exclude the first sample: at t -> 0 both are ~0 and the relative error is
    # dominated by round-off; the wall-clock cost is reported instead.
    mask = exact > 0.05 * cfg.u_max_analytic
    rel = (u[mask] - exact[mask]) / exact[mask]
    return {
        "t": t.tolist(),
        "u_centre": u.tolist(),
        "u_centre_exact": exact.tolist(),
        "n_samples_compared": int(mask.sum()),
        "centre_rel_error_max_abs_pct": float(100.0 * np.max(np.abs(rel))) if mask.any() else None,
        "centre_rel_error_rms_pct": (
            float(100.0 * np.sqrt(np.mean(rel ** 2))) if mask.any() else None
        ),
    }


def run_one(cfg: ChannelConfig, device) -> dict:
    t0 = time.time()
    result = run_case(cfg, device)
    result["wall_clock_s"] = time.time() - t0
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("runs/isph_channel_startup/latest"))
    ap.add_argument("--resolutions", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--columns", type=int, default=12)
    ap.add_argument("--criterion-pct", type=float, default=15.0)
    args = ap.parse_args(argv)
    torch.set_num_threads(2)
    device = torch.device("cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    # the a-priori [V&V] admissibility rule of the repository ledger: a run whose
    # numerical error is asked to be < 3 % cannot be graded against a 15 % gate
    # without saying so, so the 3 % paper criterion is reported next to it.
    paper_pct = 3.0

    tau = None
    table = []
    histories = {}
    for res in args.resolutions:
        cfg0 = ChannelConfig(resolution=res, columns=args.columns)
        tau = diffusive_time(cfg0)
        for frac in TIME_FRACTIONS:
            t_end = frac * tau
            t_avg = 0.05 * t_end
            cfg = ChannelConfig(resolution=res, columns=args.columns,
                                t_end=t_end, t_average=t_avg)
            res_dict = run_one(cfg, device)
            t_ref = t_end - 0.5 * t_avg          # centre of the averaging window
            cmp = compare_profile(res_dict["profile"], t_ref, cfg)
            cmp.update({
                "resolution_H_over_dx": res,
                "n_fluid": res_dict["n_fluid"],
                "n_wall": res_dict["n_wall"],
                "steps": res_dict["steps"],
                "dt": t_end / max(res_dict["steps"], 1),
                "t_end": t_end,
                "t_average": t_avg,
                "wall_clock_s": res_dict["wall_clock_s"],
                "density_error_max": res_dict["density_error_max"],
                "divergence_reduction": res_dict["divergence_reduction"],
                "n_boundary_clamped": res_dict["n_boundary_clamped"],
            })
            table.append(cmp)
            print(f"res={res:2d} t/tau={frac:5.2f} steps={res_dict['steps']:5d} "
                  f"L2_int={100*cmp['profile_rel_l2_interior']:6.2f}% "
                  f"L2_all={100*cmp['profile_rel_l2_all_rows']:6.2f}% "
                  f"centre={cmp['centre_rel_error_pct']:+6.2f}%")
        # one long run per resolution for the centreline history
        cfg = ChannelConfig(resolution=res, columns=args.columns,
                            t_end=8.0 * tau, t_average=0.05 * tau)
        res_dict = run_one(cfg, device)
        histories[res] = transient_history(res_dict, cfg)
        histories[res]["steps"] = res_dict["steps"]
        histories[res]["wall_clock_s"] = res_dict["wall_clock_s"]
        print(f"res={res:2d} history: n={len(res_dict['trace'])} "
              f"max|err|={histories[res]['centre_rel_error_max_abs_pct']:.2f}% "
              f"rms={histories[res]['centre_rel_error_rms_pct']:.2f}%")

    # ---- temporal refinement at fixed resolution: the residual error after the
    # transient has died out can only be spatial, but the error *during* the
    # start-up could still be a time-splitting artefact.  Refining dt alone
    # separates the two.
    temporal = []
    res_fine = max(args.resolutions)
    t_ref_case = 1.0 * tau
    for cfl, vsafe in ((0.25, 0.125), (0.25, 0.0625), (0.25, 0.03125)):
        cfg = ChannelConfig(resolution=res_fine, columns=args.columns,
                            t_end=t_ref_case, t_average=0.05 * t_ref_case,
                            cfl=cfl, viscous_safety=vsafe)
        res_dict = run_one(cfg, device)
        cmp = compare_profile(res_dict["profile"],
                              t_ref_case - 0.5 * cfg.t_average, cfg)
        temporal.append({"cfl": cfl, "viscous_safety": vsafe,
                         "steps": res_dict["steps"],
                         "dt": t_ref_case / max(res_dict["steps"], 1),
                         "profile_rel_l2_interior": cmp["profile_rel_l2_interior"],
                         "centre_rel_error_pct": cmp["centre_rel_error_pct"],
                         "wall_clock_s": res_dict["wall_clock_s"]})
        print(f"temporal res={res_fine} cfl={cfl:5.3f} visc={vsafe:7.5f} "
              f"steps={res_dict['steps']:5d} "
              f"L2_int={100*cmp['profile_rel_l2_interior']:5.2f}% "
              f"centre={cmp['centre_rel_error_pct']:+5.2f}%")

    # ---- measured spatial order on the steady-state error (t = 8 tau), where
    # the exact transient contribution is exp(-64) and the reading is purely the
    # discretisation error of the steady parabola.
    steady_rows = [r for r in table if r["t_over_tau"] > 7.0]
    steady_rows.sort(key=lambda r: r["resolution_H_over_dx"])
    order = []
    for a, b in zip(steady_rows, steady_rows[1:]):
        ea, eb = a["profile_rel_l2_interior"], b["profile_rel_l2_interior"]
        if ea > 0 and eb > 0:
            order.append({
                "from_H_over_dx": a["resolution_H_over_dx"],
                "to_H_over_dx": b["resolution_H_over_dx"],
                "ratio": ea / eb,
                "order_p": math.log(ea / eb) / math.log(
                    b["resolution_H_over_dx"] / a["resolution_H_over_dx"]),
            })

    worst_l2 = max(table, key=lambda r: r["profile_rel_l2_interior"])
    worst_centre = max(table, key=lambda r: abs(r["centre_rel_error_pct"]))
    verdict = {
        "criterion_pct": args.criterion_pct,
        "paper_criterion_pct": paper_pct,
        "profile_rel_l2_interior_max_pct": 100.0 * worst_l2["profile_rel_l2_interior"],
        "profile_rel_l2_interior_max_at": {
            "resolution_H_over_dx": worst_l2["resolution_H_over_dx"],
            "t_over_tau": worst_l2["t_over_tau"],
        },
        "centre_rel_error_max_abs_pct": abs(worst_centre["centre_rel_error_pct"]),
        "centre_rel_error_max_at": {
            "resolution_H_over_dx": worst_centre["resolution_H_over_dx"],
            "t_over_tau": worst_centre["t_over_tau"],
        },
        "pass_15pct": bool(100.0 * worst_l2["profile_rel_l2_interior"] <= args.criterion_pct
                           and abs(worst_centre["centre_rel_error_pct"]) <= args.criterion_pct),
        "pass_3pct": bool(100.0 * worst_l2["profile_rel_l2_interior"] <= paper_pct
                          and abs(worst_centre["centre_rel_error_pct"]) <= paper_pct),
        "history_centre_max_abs_pct": max(
            h["centre_rel_error_max_abs_pct"] for h in histories.values()),
        "history_centre_rms_pct": max(h["centre_rel_error_rms_pct"] for h in histories.values()),
        "temporal_refinement_sensitivity_pct": abs(
            temporal[0]["profile_rel_l2_interior"] - temporal[-1]["profile_rel_l2_interior"]
        ) * 100.0,
        "spatial_order_steady": order,
    }
    payload = {
        "case": "isph_channel_startup",
        "title": "ISPH plane channel: start-up from rest vs the exact Rayleigh series",
        "config": {
            "height_m": ChannelConfig().height,
            "viscosity_m2_s": ChannelConfig().viscosity,
            "body_force_m_s2": ChannelConfig().body_force,
            "columns": args.columns,
            "resolutions": args.resolutions,
            "time_fractions_of_tau": list(TIME_FRACTIONS),
        },
        "diffusive_time_s": tau,
        "steady_u_max_analytic_m_s": ChannelConfig().u_max_analytic,
        "verdict": verdict,
        "table": table,
        "temporal_refinement": temporal,
        "centreline_history": {str(k): v for k, v in histories.items()},
        "generated_by": "benchmarks/isph_channel_startup/run.py",
    }
    out = args.out / "results.json"
    out.write_text(json.dumps(payload, indent=1))
    print("verdict:", json.dumps(verdict, indent=1))
    print("wrote", out)
    make_figures(payload, args.out)
    return 0 if verdict["pass_15pct"] else 1


def make_figures(payload: dict, out: Path) -> None:
    table = payload["table"]
    hist = payload["centreline_history"]
    res_all = sorted({r["resolution_H_over_dx"] for r in table})
    u_max = payload["steady_u_max_analytic_m_s"]

    # ---- figure 1: centreline history vs the exact series
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    taus = np.array(hist[str(res_all[0])]["t"]) / payload["diffusive_time_s"]
    top = np.array(hist[str(res_all[0])]["u_centre_exact"]) / u_max
    ax.plot(taus, top, "k-", lw=1.8, label="exact Rayleigh start-up series")
    for res in res_all:
        h = hist[str(res)]
        ax.plot(np.array(h["t"]) / payload["diffusive_time_s"],
                np.array(h["u_centre"]) / u_max, lw=1.0, alpha=0.85,
                label=f"ISPH H/dx = {res}")
    ax.set_xlabel(r"$t/\tau$,  $\tau = H^2/(\nu\pi^2)$")
    ax.set_ylabel(r"$u_\mathrm{centre}/u_{s,\max}$")
    ax.set_title("ISPH channel start-up from rest: centreline vs exact series")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_centreline_history.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: interior profile error vs non-dimensional time
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for res in res_all:
        rows = [r for r in table if r["resolution_H_over_dx"] == res]
        ax.plot([r["t_over_tau"] for r in rows],
                [100.0 * r["profile_rel_l2_interior"] for r in rows],
                "o-", label=f"H/dx = {res}, all rows (dashed)")
        ax.plot([r["t_over_tau"] for r in rows],
                [100.0 * r["profile_rel_l2_all_rows"] for r in rows],
                "s--", ms=3.5, alpha=0.6)
    ax.axhline(payload["verdict"]["criterion_pct"], color="r", ls=":",
               label=f"acceptance {payload['verdict']['criterion_pct']:.0f} %")
    ax.axhline(payload["verdict"]["paper_criterion_pct"], color="orange", ls=":",
               label=f"paper criterion {payload['verdict']['paper_criterion_pct']:.0f} %")
    ax.set_xlabel(r"$t/\tau$")
    ax.set_ylabel("profile $L_2$ error [%]")
    ax.set_xscale("log")
    ax.set_title("Profile error against the exact start-up series")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_profile_error.png", dpi=150)
    plt.close(fig)

    # ---- figure 3: profiles at the time ladder, finest resolution
    finest = max(res_all)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    sel = [r for r in table if r["resolution_H_over_dx"] == finest]
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(sel):
        prof = r["profile"]
        ax.plot([p["u_measured"] / u_max for p in prof],
                [p["y_over_H"] for p in prof], "o", ms=3.2,
                color=cmap(i / max(len(sel) - 1, 1)),
                label=fr"$t/\tau={r['t_over_tau']:.2f}$")
        ax.plot([p["u_exact"] / u_max for p in prof],
                [p["y_over_H"] for p in prof], "-", lw=1.0,
                color=cmap(i / max(len(sel) - 1, 1)))
    ax.set_xlabel(r"$u/u_{s,\max}$")
    ax.set_ylabel(r"$y/H$")
    ax.set_title(f"ISPH profiles vs exact start-up (H/dx = {finest}): marks = SPH, lines = exact")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_profile_ladder.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
