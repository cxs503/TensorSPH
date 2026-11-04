"""A/B the free-surface pressure closure (Adami, Ihmsen & Adams 2012 §3.3).

Why this benchmark exists
-------------------------
``benchmarks/laplace_pressure_floor/`` decomposed the WITHHELD Young-Laplace
residual into four error sources and left one of them as the blocking item:

  C3  free-surface pressure closure.  ``tait_pressure`` clamps ``p >= 0``, and
      the drop's outer shell carries a kernel sum-density deficit (-11.6 % at
      0.80-0.95 R, -34 % at 0.95-1.05 R), so every particle there sits at
      ``p = 0`` and the interface cannot carry ``sigma/R``. The interior settles
      at 0.57 +/- 0.06 of the analytic jump, insensitive to c0 and dx.

This benchmark measures exactly the same four configurations as the floor
benchmark, once with the closure off (the "before" record, which must reproduce
the published numbers bit for bit) and once with it on, plus a two-point
sensitivity pass over the two knobs the operator exposes. It is the acceptance
evidence for ``tests/test_free_surface_pressure_closure.py`` and for
``docs/verification_free_surface_pressure_closure.md``.

The measured quantity is unchanged from the floor benchmark: the *absolute*
pressure of the drop core, averaged over the last 40 % of a 1.5 t_sigma window
(``absolute_time_average_tail``). Tait is normalised so that ``p(rho0) = 0``, so
for a drop in vacuum the physical reference is ``p_core = sigma/R`` directly,
not ``p_core - p_shell`` (the band definitions are kept as historical columns).

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_pressure_closure/run.py
    PYTHONPATH=src python benchmarks/laplace_pressure_closure/run.py --quick

Nothing is tuned per case: the operator defaults (rho/rho0 < 0.99, 3 layers)
are the ones ``Config`` ships. The sensitivity pass reports what the two knobs
cost, it does not select a winner after the fact.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# The physics and the measurement are owned by the floor benchmark; this file
# only adds the closure on/off axis. Re-implementing ``relax`` would fork the
# instrument, which is exactly what the previous three verification rounds did.
_spec = importlib.util.spec_from_file_location(
    "laplace_pressure_floor_run", REPO / "benchmarks" / "laplace_pressure_floor" / "run.py")
FLOOR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FLOOR)

DP_ANALYTIC = FLOOR.DP_ANALYTIC        # sigma/R = 360 Pa for sigma=.072, R=.1
SIGMA = FLOOR.SIGMA
RADIUS = FLOOR.RADIUS
RHO0 = FLOOR.RHO0
T_SIGMA = FLOOR.T_SIGMA

ALLOWED_REL_ERROR = 0.15


def _summary(run, dx, c0, free_surface, wall_s):
    return {
        "dx_m": dx, "sound_speed": c0, "free_surface_pressure": bool(free_surface),
        "rho_ratio": run["free_surface_rho_ratio"],
        "layers": run["free_surface_layers"],
        "n_particles": run["n_particles"], "steps": run["steps"],
        "wall_s": round(wall_s, 1),
        "delta_p_tail_Pa": run["absolute_time_average_tail"]["delta_p_Pa"],
        "tail_rel_error_pct": run["absolute_time_average_tail"]["rel_error_pct"],
        "final_snapshot_rel_error_pct": run["definition_absolute"]["rel_error_pct"],
        "within_15pct": run["absolute_time_average_tail"]["within_15pct"],
        "within_3pct": run["absolute_time_average_tail"]["within_3pct"],
        "shell_fraction": run["density"]["shell_fraction"],
        "rho_core_rel": run["density"]["rho_core_over_rho0"] - 1.0,
        "compression_required": run["density"]["compression_required"],
        "vmax_over_v_ref": run["spurious_current"]["vmax_over_vref_final"],
        "passes_adami_0p1": run["spurious_current"]["passes_adami_0p1"],
        "finite": run["finite"],
        "pressure_profile": run["pressure_profile"],
        "density_profile": run["density_profile"],
        "trace": run["trace"],
    }


def measure(dx, c0, free_surface, **kw):
    t0 = time.time()
    run = FLOOR.relax(dx, c0, free_surface=free_surface, **kw)
    return _summary(run, dx, c0, free_surface, time.time() - t0)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(report, out_dir=HERE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = report["runs"]
    by = {(r["sound_speed"], r["free_surface_pressure"]): r for r in runs}
    c0s = sorted({r["sound_speed"] for r in runs})
    figs = []

    # 1. radial pressure profile: the missing interface jump, before/after
    fig, axes = plt.subplots(1, len(c0s), figsize=(5.2 * len(c0s), 4.2),
                             sharey=True, squeeze=False)
    for ax, c0 in zip(axes[0], c0s):
        off, on = by[(c0, False)], by[(c0, True)]
        for run, label, colour in ((off, "closure off", "0.45"), (on, "closure on", "C0")):
            prof = run["pressure_profile"]
            ax.plot([row["r_over_R_mid"] for row in prof],
                    [row["mean"] for row in prof], "o-", color=colour, ms=3.5,
                    lw=1.4, label=f"{label} ({run['tail_rel_error_pct']:+.1f} %)")
        ax.axhline(DP_ANALYTIC, color="r", ls="--", lw=1.2,
                   label=fr"$\sigma/R$ = {DP_ANALYTIC:.0f} Pa")
        ax.axvline(1.0, color="k", ls=":", lw=1.0)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_title(fr"$c_0$ = {c0:g} m/s, dx = {off['dx_m']*1000:g} mm")
        ax.set_xlabel(r"$r/R$")
        ax.set_ylabel("mean pressure [Pa]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Young-Laplace jump across the drop interface (absolute, time-averaged)")
    fig.tight_layout()
    p1 = out_dir / "pressure_profiles.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    figs.append({"name": p1.name,
                 "caption": "Radial pressure profile, closure off vs on, against sigma/R. "
                            "Off: the deficit shell is pinned at p = 0 and only the interior "
                            "settles. On: the interface carries the jump."})

    # 2. residual bars with the tolerance bands
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    width = 0.36
    for k, (fs, label, colour) in enumerate(((False, "closure off", "0.6"),
                                             (True, "closure on", "C0"))):
        vals = [by[(c0, fs)]["tail_rel_error_pct"] for c0 in c0s]
        bars = ax.bar([i + (k - 0.5) * width for i in range(len(c0s))], vals,
                      width, color=colour, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (4 if v > 0 else -6),
                    f"{v:+.1f} %", ha="center", fontsize=8.5)
    ax.axhspan(-15, 15, color="tab:green", alpha=0.10, label="+/-15 % band")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(c0s)))
    ax.set_xticklabels([fr"$c_0$ = {c0:g} m/s" for c0 in c0s])
    ax.set_ylabel(r"$\Delta p$ residual vs $\sigma/R$  [%]")
    ax.set_title("Free-surface pressure closure: Laplace residual (absolute, tail-averaged)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "ab_residuals.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    figs.append({"name": p2.name,
                 "caption": "Tail-averaged Laplace residual, closure off vs on, with the "
                            "15 % acceptance band."})

    # 3. time trace: does the closure stabilise, or just shift the mean?
    fig, axes = plt.subplots(1, len(c0s), figsize=(5.2 * len(c0s), 3.6),
                             sharey=True, squeeze=False)
    for ax, c0 in zip(axes[0], c0s):
        for fs, label, colour in ((False, "closure off", "0.45"), (True, "closure on", "C0")):
            tr = by[(c0, fs)]["trace"]
            ax.plot(tr["t"], [v / DP_ANALYTIC for v in tr["absolute"]],
                    color=colour, lw=1.2, label=label)
        ax.axhline(1.0, color="r", ls="--", lw=1.0, label=r"$\sigma/R$")
        ax.set_xlabel(r"$t / t_\sigma$")
        ax.set_ylabel(r"$p_{core} / (\sigma/R)$")
        ax.set_title(fr"$c_0$ = {c0:g} m/s")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    fig.suptitle("Core pressure history (breathing mode averages out over the window)")
    fig.tight_layout()
    p3 = out_dir / "time_traces.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    figs.append({"name": p3.name,
                 "caption": "Core pressure history normalised by sigma/R. The final snapshot "
                            "is not an equilibrium value; the tail average is."})
    return figs


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="single c0, no sensitivity pass")
    ap.add_argument("--dx", type=float, default=0.002)
    ap.add_argument("--t-end", type=float, default=1.5)
    args = ap.parse_args()

    c0s = [5.0] if args.quick else [5.0, 20.0]
    runs, pairs = [], []
    for c0 in c0s:
        off = measure(args.dx, c0, False, t_end_over_t_sigma=args.t_end)
        on = measure(args.dx, c0, True, t_end_over_t_sigma=args.t_end)
        runs += [off, on]
        pairs.append({
            "sound_speed": c0,
            "off_rel_error_pct": off["tail_rel_error_pct"],
            "on_rel_error_pct": on["tail_rel_error_pct"],
            "improved": abs(on["tail_rel_error_pct"]) < abs(off["tail_rel_error_pct"]),
            "sign_flip": (off["tail_rel_error_pct"] < 0) != (on["tail_rel_error_pct"] < 0),
        })
        print(f"c0={c0:5g}: off {off['tail_rel_error_pct']:+7.1f} %  "
              f"on {on['tail_rel_error_pct']:+7.1f} %  "
              f"({off['wall_s']:.0f} s + {on['wall_s']:.0f} s)", flush=True)

    sensitivity = []
    if not args.quick:
        ref_c0 = c0s[-1]
        for kw, tag in (({"free_surface_rho_ratio": 0.97}, "rho_ratio=0.97"),
                        ({"free_surface_layers": 1}, "layers=1")):
            rec = measure(args.dx, ref_c0, True, t_end_over_t_sigma=args.t_end, **kw)
            rec["variant"] = tag
            sensitivity.append(rec)
            print(f"c0={ref_c0:g} {tag}: {rec['tail_rel_error_pct']:+7.1f} %  "
                  f"vmax/vref {rec['vmax_over_v_ref']:.3f}", flush=True)

    report = {
        "benchmark": "laplace_pressure_closure",
        "operator": {
            "reference": "Adami, Ihmsen & Adams (2012), JCP 231:7057, section 3.3",
            "default_rho_ratio": 0.99,
            "default_layers": 3,
            "default_enabled": False,
        },
        "reference": {"delta_p_Pa": DP_ANALYTIC, "sigma_N_per_m": SIGMA,
                      "radius_m": RADIUS, "rho0": RHO0, "t_sigma_s": T_SIGMA,
                      "acceptance": "|residual| < 15 % on every closure-on case"},
        "runs": runs,
        "ab": pairs,
        "sensitivity": sensitivity,
        "all_on_within_15pct": all(r["within_15pct"] for r in runs
                                   if r["free_surface_pressure"]),
        "all_on_improve": all(p["improved"] for p in pairs),
        "sensitivity_rank": sorted(
            [{"variant": "defaults", "rel_error_pct": p["on_rel_error_pct"],
              "vmax_over_v_ref": next(r["vmax_over_v_ref"] for r in runs
                                      if r["free_surface_pressure"]
                                      and r["sound_speed"] == p["sound_speed"])}
             for p in pairs] + [{"variant": s["variant"],
                                 "rel_error_pct": s["tail_rel_error_pct"],
                                 "vmax_over_v_ref": s["vmax_over_v_ref"]}
                                for s in sensitivity],
            key=lambda d: d["rel_error_pct"]),
    }
    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    report["figures"] = make_figures(report)
    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    print("wrote", HERE / "report.json")
    print(json.dumps(report["sensitivity_rank"], indent=1))


if __name__ == "__main__":
    main()