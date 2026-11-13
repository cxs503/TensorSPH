"""XVAL-SPH-MACH-LIMITED-RIEMANN-DISSIPATION — reproducible study driver.

Produces (all numbers in the report come from results.json, nothing hand-copied):

  1. Sod (1978) shock tube at N = 400 / 800 for four dissipation models
       - "av-full"      : Monaghan artificial viscosity, alpha = 1, beta = 2
       - "av-mach-abs"  : round-1 limiter, alpha_eff = alpha*min(|v_i|/c_i, 1)
       - "av-mach-rel"  : Mach-limited, alpha_eff = alpha*min(M_ij, 1), M_ij the
                          compressive *relative* normal pair Mach number
       - "riemann-rel"  : acoustic (Godunov-SPH) interface pressure with the same
                          relative-Mach limiter
  2. Resolution ladder N = 200 ... 1600 for "av-mach-rel" and "av-full".
  3. Linear standing acoustic wave in a closed box (low Mach): effective
     viscosity nu_eff of each model vs the intrinsic (alpha = 0) floor.

Run:  PYTHONPATH=src python3 benchmarks/xval_mach_limited_riemann/run.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from tensorsph.compressible.riemann_dissipation import (
    RiemannDissipationConfig,
    evaluate_metrics,
    run_acoustic_pulse,
    run_sod_riemann,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LIT = {"p_star": 0.30313, "u_star": 0.92745,
       "rho_star_L": 0.42632, "rho_star_R": 0.26557}

MODELS = {
    "av-full": dict(mach_limit=False),
    "av-mach-abs": dict(mach_limit=True, mach_limit_mode="absolute"),
    "av-mach-rel": dict(mach_limit=True, mach_limit_mode="relative"),
    "riemann-rel": dict(scheme="riemann", mach_limit=True, mach_limit_mode="relative"),
}


def _exact_profile(snap, cfg):
    from tensorsph.wcsph.sod_shock import exact_riemann_profile
    return exact_riemann_profile(
        snap["x"], snap["t"], gamma=cfg.gamma,
        rho_L=cfg.rho_L, p_L=cfg.p_L, u_L=cfg.u_L,
        rho_R=cfg.rho_R, p_R=cfg.p_R, u_R=cfg.u_R, x0=cfg.x_diaphragm)


def _rarefaction_l1(snap, cfg):
    """L1 density error restricted to the (smooth, low-Mach) rarefaction fan."""
    from tensorsph.compressible.sod import exact_wave_positions
    w = exact_wave_positions(cfg.t_end, x0=cfg.x_diaphragm, gamma=cfg.gamma,
                             rho_L=cfg.rho_L, p_L=cfg.p_L, u_L=cfg.u_L,
                             rho_R=cfg.rho_R, p_R=cfg.p_R, u_R=cfg.u_R)
    rho_ex, _, _ = _exact_profile(snap, cfg)
    mask = (snap["x"] > w["x_rarefaction_head"]) & (snap["x"] < w["x_rarefaction_tail"])
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(snap["rho"][mask] - rho_ex[mask])) / np.mean(rho_ex[mask]))


def _shock_width(snap, cfg, window=(0.60, 1.0), frac_lo=0.10, frac_hi=0.90):
    """10-90 % pressure-transition thickness across the shock, in units of h.

    The shock is detected inside ``window`` (right of the contact discontinuity),
    where p rises monotonically from the star value to p_R.
    """
    x, p = snap["x"], snap["p"]
    m = (x >= window[0]) & (x <= window[1])
    x, p = x[m], p[m]
    order = np.argsort(x)
    x, p = x[order], p[order]
    if len(p) < 5:
        return float("nan")
    p_min, p_max = float(p.min()), float(p.max())
    lvl_lo = p_min + frac_lo * (p_max - p_min)
    lvl_hi = p_min + frac_hi * (p_max - p_min)
    i_lo = np.where(p >= lvl_lo)[0]
    i_hi = np.where(p >= lvl_hi)[0]
    if len(i_lo) == 0 or len(i_hi) == 0:
        return float("nan")
    return float(abs(x[i_hi[0]] - x[i_lo[0]]) / cfg.h)


def sod_case(model: str, n: int):
    cfg = RiemannDissipationConfig(n_particles=n, **MODELS[model])
    t0 = time.time()
    res = run_sod_riemann(cfg, output_times=[cfg.t_end])
    wall = time.time() - t0
    row = {
        "model": model, "n_particles": n, "h": cfg.h, "steps": res["steps"],
        "t_final": res["t_final"], "wall_s": round(wall, 2),
        "dt_min": res["dt_min"], "mach_max": res["mach_max"],
        "energy_rel_change": res["energy_rel_change"],
        "completed": bool(res["snapshots"]) and abs(res["t_final"] - cfg.t_end) < 1e-9,
        "neighbours": res["neighbours"],
        "scheme": cfg.scheme, "mach_limit": cfg.mach_limit,
        "mach_limit_mode": cfg.mach_limit_mode,
    }
    if not row["completed"]:
        return row, None, cfg
    snap = res["snapshots"][-1]
    met = evaluate_metrics(snap, cfg)
    row.update({
        "max_abs_rel_err": met["max_abs_rel_err"],
        "rel_err": met["rel_err"],
        "l1": met["l1"],
        "l1_rarefaction": _rarefaction_l1(snap, cfg),
        "shock_width_over_h": _shock_width(snap, cfg),
        "contact_pressure_jump": met["contact_pressure_jump"],
        "plateau": {
            "rho_star_L": float(np.mean(snap["rho"][np.abs(snap["x"] - 0.55) < 0.02])),
            "rho_star_R": float(np.mean(snap["rho"][np.abs(snap["x"] - 0.75) < 0.02])),
            "u_star": float(np.mean(snap["u"][np.abs(snap["x"] - 0.65) < 0.02])),
            "p_star": float(np.mean(snap["p"][np.abs(snap["x"] - 0.65) < 0.02])),
        },
    })
    row["plateau_rel_err"] = {
        k: float((row["plateau"][k] - LIT[k]) / LIT[k]) for k in LIT}
    return row, snap, cfg


def acoustic_case(model: str, n: int = 200, amplitude: float = 0.02,
                  n_periods: float = 6.0):
    kw = MODELS[model]
    cfg = RiemannDissipationConfig(n_particles=n, **kw)
    t0 = time.time()
    res = run_acoustic_pulse(cfg, amplitude=amplitude, n_periods=n_periods)
    return {
        "model": model, "n_particles": n, "amplitude": amplitude,
        "n_periods": n_periods, "steps": res["steps"],
        "wall_s": round(time.time() - t0, 2),
        "nu_measured": res["nu_measured"],
        "nu_av_theory": res["nu_av_theory"],
        "e_acoustic_ratio": res["e_acoustic_ratio"],
        "damping_rate": res["damping_rate"],
        "series": {k: np.asarray(v).tolist() for k, v in res["series"].items()},
    }


def acoustic_inviscid_floor(n: int = 200):
    cfg = RiemannDissipationConfig(n_particles=n, mach_limit=False,
                                   alpha_av=0.0, beta_av=0.0)
    t0 = time.time()
    res = run_acoustic_pulse(cfg, amplitude=0.02, n_periods=6.0)
    return {
        "model": "inviscid-floor", "n_particles": n, "alpha_av": 0.0,
        "n_periods": 6.0, "steps": res["steps"], "wall_s": round(time.time() - t0, 2),
        "nu_measured": res["nu_measured"], "nu_av_theory": 0.0,
        "e_acoustic_ratio": res["e_acoustic_ratio"],
        "series": {k: np.asarray(v).tolist() for k, v in res["series"].items()},
    }


def _dump(out):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(out, f, indent=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {"literature": LIT, "models": MODELS, "sod": [], "ladder": [],
           "acoustic": [], "figures": {}}
    snaps = {}

    print("== Sod matrix ==", flush=True)
    for model in MODELS:
        for n in (400, 800):
            row, snap, cfg = sod_case(model, n)
            out["sod"].append(row)
            print("  %-11s N=%4d completed=%s steps=%5d max_rel=%s l1_rho=%s"
                  % (model, n, row["completed"], row["steps"],
                     row.get("max_abs_rel_err"), row.get("l1", {}).get("rho")), flush=True)
            if snap is not None and n == 800:
                snaps[model] = (snap, cfg)
            _dump(out)

    print("== Resolution ladder ==", flush=True)
    for model in ("av-mach-rel", "av-full"):
        for n in (200, 400, 800, 1600):
            row, snap, cfg = sod_case(model, n)
            out["ladder"].append(row)
            if snap is not None and n == 1600:
                snaps["ladder-" + model] = (snap, cfg)
            print("  %-11s N=%4d l1_rho=%s" % (model, n, row.get("l1", {}).get("rho")),
                  flush=True)
            _dump(out)

    print("== Acoustic (low-Mach) damping ==", flush=True)
    out["acoustic"].append(acoustic_inviscid_floor())
    for model in MODELS:
        out["acoustic"].append(acoustic_case(model))
        _dump(out)
        print("  %-11s nu_meas=%.3e  E_end/E0=%.4f"
              % (model, out["acoustic"][-1]["nu_measured"],
                 out["acoustic"][-1]["e_acoustic_ratio"]), flush=True)

    # ---- convergence order (least squares on log L1 vs log dx) ----------
    orders = {}
    for model in ("av-mach-rel", "av-full"):
        rows = [r for r in out["ladder"] if r["model"] == model and r["completed"]]
        dx = np.array([r["h"] / 2.5 for r in rows])
        e = np.array([r["l1"]["rho"] for r in rows])
        if len(rows) >= 2 and np.all(e > 0):
            orders[model] = float(np.polyfit(np.log(dx), np.log(e), 1)[0])
    out["convergence_order"] = orders

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(out, f, indent=1)

    # ---- figures --------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    if snaps:
        fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True)
        for model, color in (("av-full", "tab:orange"), ("av-mach-rel", "tab:blue"),
                             ("riemann-rel", "tab:green")):
            if model not in snaps:
                continue
            snap, cfg = snaps[model]
            rho_ex, u_ex, p_ex = _exact_profile(snap, cfg)
            axes[0].plot(snap["x"], snap["rho"], ".", ms=2, color=color, label=model)
            axes[1].plot(snap["x"], snap["u"], ".", ms=2, color=color)
            axes[2].plot(snap["x"], snap["p"], ".", ms=2, color=color)
        axes[0].plot(snap["x"], rho_ex, "-", lw=1.2, color="k", label="exact Riemann")
        axes[1].plot(snap["x"], u_ex, "-", lw=1.2, color="k")
        axes[2].plot(snap["x"], p_ex, "-", lw=1.2, color="k")
        for ax, lab in zip(axes, ("density  rho", "velocity  u", "pressure  p")):
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8, loc="lower left")
        axes[2].set_xlabel("x")
        axes[0].set_title("Sod shock tube t=0.2, N=800: dissipation models vs exact")
        fig.tight_layout()
        path = os.path.join(OUT, "fig1_sod_profiles.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        figs.append(("fig1_sod_profiles.png", "Sod t=0.2 profiles, N=800"))

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for model, color in (("av-full", "tab:orange"), ("av-mach-rel", "tab:blue"),
                             ("riemann-rel", "tab:green")):
            if model not in snaps:
                continue
            snap, cfg = snaps[model]
            rho_ex, _, _ = _exact_profile(snap, cfg)
            ax.plot(snap["x"], snap["rho"] - rho_ex, color=color, lw=1.0, label=model)
        ax.axvline(0.5, color="k", ls=":", lw=0.8)
        ax.set_xlim(0.35, 0.9)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\rho - \rho_{exact}$")
        ax.set_title("Density error vs exact solution (N=800)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig2_error_profiles.png"), dpi=140)
        plt.close(fig)
        figs.append(("fig2_error_profiles.png", "density error vs exact solution"))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for model, marker, color in (("av-mach-rel", "o", "tab:blue"),
                                 ("av-full", "s", "tab:orange")):
        rows = [r for r in out["ladder"] if r["model"] == model and r["completed"]]
        if not rows:
            continue
        dx = [r["h"] / 2.5 for r in rows]
        e = [r["l1"]["rho"] for r in rows]
        ax.loglog(dx, e, marker + "-", color=color,
                  label="%s (order %.2f)" % (model, out["convergence_order"].get(model, float("nan"))))
    ax.set_xlabel("dx")
    ax.set_ylabel(r"$L_1$ relative density error")
    ax.set_title("Sod resolution ladder")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_convergence.png"), dpi=140)
    plt.close(fig)
    figs.append(("fig3_convergence.png", "L1 density error vs dx"))

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for row in out["acoustic"]:
        s = row["series"]
        ax.semilogy(s["t"], np.maximum(s["e_acoustic"], 1e-16), lw=1.2,
                    label="%s (nu=%.2e)" % (row["model"], row["nu_measured"]))
    ax.set_xlabel("t")
    ax.set_ylabel(r"$E_{ac}(t)/E_{ac}(0)$ (mode acoustic energy)")
    ax.set_title("Standing acoustic wave, A=0.02 (M=0.02): spurious damping")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_acoustic_damping.png"), dpi=140)
    plt.close(fig)
    figs.append(("fig4_acoustic_damping.png", "acoustic modal energy decay"))

    out["figures"] = {name: cap for name, cap in figs}
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.join(OUT, "results.json"))


if __name__ == "__main__":
    main()