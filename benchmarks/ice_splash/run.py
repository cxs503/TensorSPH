"""Bow ice-splash benchmark: run the case, build the criteria table and the figures.

Usage
-----
    python benchmarks/ice_splash/run.py [--dx 0.0125] [--out artifacts/ice_splash]

Produces, under ``artifacts/ice_splash/``:

* ``results.json``      criteria table + every number quoted in the report
* ``dem_init.json``     particle-initialisation payload for the DEM channel scenario
* ``fig1_fields.png``   free-surface field maps (|u| and p) at four instants
* ``fig2_front.png``    first-wave front in the Martin & Moyce form + Ritter
* ``fig3_energy.png``   mechanical and total energy drift + resolution ladder
* ``fig4_gauges.png``   deck inundation / apex / throw time series
* ``fig5_floes.png``    reduced-model floe launch estimate vs its energy bound

Everything is computed here; nothing is transcribed by hand.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorsph.ice_splash import (
    IceSplashConfig,
    build_dem_exchange,
    deck_start_x,
    floe_launch_estimate,
    floe_layout,
    front_coefficient,
    front_timing_error,
    ice_bed_lattice,
    run_case,
    write_dem_exchange,
)

REPO = Path(__file__).resolve().parents[2]
MM_TABLE = REPO / "benchmarks" / "dam_break" / "martin_moyce_1952.json"
# Platform baseline for the reference-free front coefficient, flat bed, free
# slip, dx = 0.04 m (docs/benchmarks/dam-break-reference-audit.md, section 7).
AUDIT_C_DX004 = 1.2016
AUDIT_RITTER_C = 2.0
TOL_FRONT = 0.15          # platform precision standard for a benchmark anchor
TOL_ENERGY = 0.01         # total-energy drift over the event window
TOL_BALLISTIC = 0.05      # median relative error of the splash apex
TOL_SEED = 0.0            # bit-for-bit


def splash_config(dx: float, **overrides) -> IceSplashConfig:
    base = dict(
        dx=dx, column_width=0.20, column_height=0.28, pool_width=1.0,
        tank_height=0.75, sheet_start=0.20, sheet_angle_deg=25.0, deck_height=0.28,
        sound_speed=20.0, viscosity=0.0, device="cpu", output_every=5,
        max_courant=0.3, n_floes=5, seed=7,
    )
    base.update(overrides)
    t_target = float(base.pop("t_target", 0.46))
    cfg = IceSplashConfig(**base)
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(t_target / cfg.dt))
    return cfg


def flat_bed_config(dx: float) -> IceSplashConfig:
    cfg = IceSplashConfig(
        dx=dx, column_width=0.20, column_height=0.40, pool_width=2.0, tank_height=0.70,
        sheet_start=2.0, sheet_angle_deg=25.0, deck_height=0.10, sound_speed=25.0,
        viscosity=0.01, device="cpu", output_every=5, max_courant=0.3, n_floes=0)
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(0.50 / cfg.dt))
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=0.0125,
                        help="production resolution of the splash case")
    parser.add_argument("--ladder", default="0.02,0.0125,0.01",
                        help="comma separated resolutions for the energy ladder")
    parser.add_argument("--out", default=str(REPO / "artifacts" / "ice_splash"))
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--c0-study", default="25,30",
                        help="comma separated sound speeds for the compressibility study")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- production run ----------------------------------------------------- #
    cfg = splash_config(args.dx)
    snapshot_times = [0.06, 0.18, 0.30, 0.40]
    started = time.time()
    run = run_case(cfg, snapshot_times=snapshot_times)
    wall_prod = time.time() - started
    summary = run.summary()
    window = run.energy_window_end()
    drift_mech = run.energy_drift_peak(window)
    drift_total = run.energy_drift_peak(window, include_internal=True)
    ballistic = summary["ballistic"]

    # ---- seed reproducibility ---------------------------------------------- #
    # A short case is enough: the claim is bit-for-bit determinism of the same
    # configuration, and the production run above is already repeated below in
    # the ladder at other resolutions.
    seed_cfg = splash_config(0.025, t_target=0.12)
    first = run_case(seed_cfg)
    second = run_case(splash_config(0.025, t_target=0.12))
    seed_identical = bool(torch.equal(first.final_positions, second.final_positions))
    seed_gauges_identical = (first.times == second.times
                             and first.deck_volume == second.deck_volume)
    seed_floe_differs = bool((floe_layout(splash_config(0.025, seed=8), torch.device("cpu"))
                              - floe_layout(splash_config(0.025, seed=7), torch.device("cpu"))
                              ).abs().max() > 0.0)

    # ---- energy ladder ------------------------------------------------------ #
    ladder = []
    for dx in [float(v) for v in args.ladder.split(",") if v.strip()]:
        if abs(dx - args.dx) < 1e-12:
            ladder.append({"dx": dx, "energy_drift_mech": drift_mech,
                           "energy_drift_total": drift_total, "wall_s": wall_prod,
                           "window_s": window, "reused_production": True})
            continue
        ladder_cfg = splash_config(dx)
        tick = time.time()
        ladder_run = run_case(ladder_cfg)
        ladder.append({"dx": dx,
                       "energy_drift_mech": ladder_run.energy_drift_peak(
                           ladder_run.energy_window_end()),
                       "energy_drift_total": ladder_run.energy_drift_peak(
                           ladder_run.energy_window_end(), include_internal=True),
                       "wall_s": time.time() - tick,
                       "window_s": ladder_run.energy_window_end(),
                       "reused_production": False})
    ladder.sort(key=lambda row: -row["dx"])

    # ---- compressibility study --------------------------------------------- #
    # The zero-pressure clamp at the free surface is the scheme's main energy
    # defect and it scales with the compression, so the drift is reported over
    # sound speed. It is *not* monotone, so no value is picked to pass a gate.
    c0_study = [{"sound_speed_m_s": cfg.sound_speed,
                 "energy_drift_total": drift_total, "energy_drift_mech": drift_mech,
                 "peak_compression": run.peak_compression, "reused_production": True}]
    for c0 in [float(v) for v in args.c0_study.split(",") if v.strip()]:
        if abs(c0 - cfg.sound_speed) < 1e-12:
            continue
        c0_cfg = splash_config(args.dx, sound_speed=c0)
        c0_run = run_case(c0_cfg)
        c0_window = c0_run.energy_window_end()
        c0_study.append({
            "sound_speed_m_s": c0,
            "energy_drift_total": c0_run.energy_drift_peak(c0_window, include_internal=True),
            "energy_drift_mech": c0_run.energy_drift_peak(c0_window),
            "peak_compression": c0_run.peak_compression,
            "deck_arrival_time_s": c0_run.deck_arrival_time,
            "reused_production": False})
    c0_study.sort(key=lambda row: row["sound_speed_m_s"])

    # ---- flat-bed benchmark anchor ----------------------------------------- #
    fb_cfg = flat_bed_config(0.04)
    fb_run = run_case(fb_cfg)
    fb = front_coefficient(fb_run, fb_cfg)
    fb_error = abs(fb["c"] - AUDIT_C_DX004) / AUDIT_C_DX004 if fb["c"] else None
    fb_ritter = front_timing_error(fb_run, fb_cfg)
    with open(MM_TABLE) as handle:
        table = json.load(handle)["data"]

    # ---- reduced-model floe estimate -------------------------------------- #
    floes = floe_layout(cfg, torch.device("cpu"))
    floe = floe_launch_estimate(run, cfg, floes)

    # ---- criteria ----------------------------------------------------------- #
    def passed(value, tol, lower=False):
        if value is None:
            return False
        return value < tol if not lower else value > tol

    criteria = {
        "front_coefficient": fb["c"],
        "front_coefficient_baseline": AUDIT_C_DX004,
        "front_coefficient_error": fb_error,
        "front_coefficient_pass": passed(fb_error, TOL_FRONT),
        "ritter_rel_error": fb_ritter["ritter_rel_error"],
        "ritter_speed_ratio": fb_ritter["speed_ratio"],
        "energy_drift_total": drift_total,
        "energy_drift_mech": drift_mech,
        "fragmentation_time_s": run.fragmentation_time,
        "fragmentation_stage_drift": (run.energy_drift_peak(
            run.fragmentation_time, include_internal=True)
            if run.fragmentation_time is not None else drift_total),
        "energy_window_s": window,
        "energy_pass": passed(drift_total, TOL_ENERGY),
        "energy_tolerance": TOL_ENERGY,
        "ballistic_median_error": ballistic["median_rel_error"],
        "ballistic_p90_error": ballistic["p90_rel_error"],
        "ballistic_count": ballistic["count"],
        "ballistic_pass": passed(ballistic["median_rel_error"], TOL_BALLISTIC),
        "seed_identical": seed_identical,
        "seed_gauges_identical": seed_gauges_identical,
        "seed_pass": bool(seed_identical and seed_gauges_identical and seed_floe_differs),
        "seed_layout_changes_with_seed": seed_floe_differs,
        "escaped": run.escaped,
        "mass_conserved": abs(run.initial_mass - run.final_mass) <= 1e-12 * run.initial_mass,
        "peak_compression": run.peak_compression,
        "tolerances": {"front": TOL_FRONT, "energy": TOL_ENERGY,
                       "ballistic": TOL_BALLISTIC, "seed": TOL_SEED},
    }
    payload = {
        "case": {
            "name": "bow ice-breaking transient free surface (2-D WCSPH)",
            "dx_m": cfg.dx, "dt_s": cfg.dt, "steps": cfg.steps,
            "water_column_m": [cfg.column_width, cfg.column_height],
            "sheet_start_m": cfg.sheet_start, "sheet_angle_deg": cfg.sheet_angle_deg,
            "deck_height_m": cfg.deck_height, "deck_start_m": deck_start_x(cfg),
            "pool_width_m": cfg.pool_width, "tank_height_m": cfg.tank_height,
            "sound_speed_m_s": cfg.sound_speed, "viscosity_m2_s": cfg.viscosity,
            "rho0_kg_m3": cfg.rho0, "gamma": cfg.gamma, "gravity_m_s2": cfg.gravity,
            "seed": cfg.seed, "n_floes": cfg.n_floes,
            "floe_radius_m": cfg.floe_radius, "floe_rho_kg_m3": cfg.floe_rho,
            "particles_water": int(run.final_positions.shape[0]),
            "particles_bed": int(ice_bed_lattice(cfg, torch.device("cpu")).shape[0]),
            "wall_clock_s": wall_prod,
        },
        "gauges": {
            "deck_arrival_time_s": summary["deck_arrival_time_s"],
            "crest_arrival_time_s": summary["crest_arrival_time_s"],
            "peak_deck_volume_m3_per_m": summary["peak_deck_volume_m3_per_m"],
            "peak_apex_m": summary["peak_apex_m"],
            "peak_throw_m": summary["peak_throw_m"],
            "peak_splash_ke_J_per_m": summary["peak_splash_ke_J_per_m"],
            "energy_full_run_drift": summary["energy_drift_full"],
            "energy_full_run_drift_total": summary["energy_drift_full_with_internal"],
            "initial_mass_kg_per_m": summary["initial_mass_kg_per_m"],
            "simulated_time_s": summary["simulated_time_s"],
        },
        "floe_estimate": floe,
        "energy_ladder": ladder,
        "compressibility_study": c0_study,
        "flat_bed_anchor": {"front_coefficient": fb, "ritter": fb_ritter,
                            "audit_baseline_c": AUDIT_C_DX004,
                            "audit_ritter_c": AUDIT_RITTER_C,
                            "dx_m": fb_cfg.dx, "viscosity_m2_s": fb_cfg.viscosity},
        "criteria": criteria,
        "series": {
            "times": run.times,
            "front": run.front,
            "kinetic_energy": run.kinetic_energy,
            "potential_energy": run.potential_energy,
            "internal_energy": run.internal_energy,
            "deck_volume": run.deck_volume,
            "apex": run.apex,
            "throw": run.throw,
            "flat_bed_tau": [t * math.sqrt(2.0 * fb_cfg.gravity / fb_cfg.column_width)
                             for t in fb_run.times],
            "flat_bed_z_over_a": [x / fb_cfg.column_width - 1.0 for x in fb_run.front],
        },
        "snapshots": [
            {"t": t, "position": pos.tolist(),
             "speed": torch.linalg.vector_norm(vel, dim=-1).tolist(),
             "pressure": p.tolist()}
            for t, pos, vel, _rho, p in run.snapshots
        ],
    }
    with open(out / "results.json", "w") as handle:
        json.dump(payload, handle, indent=1)

    dem_payload = build_dem_exchange(run, cfg, floes)
    dem_path = write_dem_exchange(dem_payload, out / "dem_init.json")

    if not args.no_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"figures skipped: {exc}")
        else:
            _figures(payload, table, out, plt, cfg)
    print(json.dumps({"criteria": criteria, "gauges": payload["gauges"],
                      "floe_estimate": floe, "dem_payload": str(dem_path),
                      "results": str(out / "results.json")}, indent=1))
    return 0


def _figures(payload, table, out, plt, cfg) -> None:
    crit = payload["criteria"]
    series = payload["series"]

    # ---- fig 1: field maps ------------------------------------------------- #
    snaps = payload["snapshots"]
    if snaps:
        fig, axes = plt.subplots(1, len(snaps), figsize=(3.4 * len(snaps), 3.2))
        if len(snaps) == 1:
            axes = [axes]
        vmax = max(max(s["speed"]) for s in snaps) or 1.0
        for ax, snap in zip(axes, snaps):
            pos = torch.tensor(snap["position"])
            speed = torch.tensor(snap["speed"])
            ax.scatter(pos[:, 0], pos[:, 1], c=speed, s=6, cmap="viridis",
                       vmin=0.0, vmax=vmax, linewidths=0)
            x = [0.0, cfg.sheet_start, deck_start_x(cfg), cfg.pool_width]
            y = [0.0, 0.0, cfg.deck_height, cfg.deck_height]
            ax.plot(x, y, color="0.35", lw=1.2)
            ax.set_xlim(0, cfg.pool_width)
            ax.set_ylim(0, cfg.tank_height)
            ax.set_title(f"t = {snap['t']:.2f} s  (|u| m/s)")
            ax.set_xlabel("x [m]")
        axes[0].set_ylabel("y [m]")
        fig.tight_layout()
        fig.savefig(out / "fig1_fields.png", dpi=130)
        plt.close(fig)

    # ---- fig 2: front in the Martin & Moyce form -------------------------- #
    tau = series["flat_bed_tau"]
    z = series["flat_bed_z_over_a"]
    tau_t = [d["tau"] for d in table]
    z_t = [d["Z_over_a"] for d in table]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(tau_t, z_t, "o", ms=3, color="0.5", label="Martin & Moyce (1952) table")
    fb_dx = payload["flat_bed_anchor"]["dx_m"]
    ax.plot(tau, z, "-", lw=1.6, label=f"SPH flat bed (dx = {fb_dx} m)")
    tc = crit["front_coefficient"]
    ax.plot([0, max(tau_t)], [0, AUDIT_RITTER_C * max(tau_t)], "--", lw=1.0,
            label="Ritter (c = 2)")
    ax.plot([0, max(tau_t)], [0, tc * max(tau_t)], ":", lw=1.4,
            label=f"fitted c = {tc:.3f}")
    ax.set_xlabel(r"$\tau = t\sqrt{2g/a}$")
    ax.set_ylabel("Z / a")
    c_err = crit["front_coefficient_error"] * 100
    ax.set_title(f"first-wave front: c err = {c_err:.1f}% vs verified baseline")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_front.png", dpi=130)
    plt.close(fig)

    # ---- fig 3: energy budget and ladder ---------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.6))
    t = series["times"]
    ke = torch.tensor(series["kinetic_energy"]).double()
    pe = torch.tensor(series["potential_energy"]).double()
    ie = torch.tensor(series["internal_energy"]).double()
    mech = (ke + pe)
    total = (ke + pe + ie)
    ax1.plot(t, ((mech - mech[0]) / mech[0] * 100).tolist(), lw=1.4,
             label="KE + PE")
    ax1.plot(t, ((total - total[0]) / total[0] * 100).tolist(), lw=1.4,
             label="KE + PE + elastic")
    ax1.axvspan(0, crit["energy_window_s"], color="0.85", zorder=0,
                label="criterion window")
    ax1.axhline(1.0, color="crimson", ls="--", lw=0.9)
    ax1.axhline(-1.0, color="crimson", ls="--", lw=0.9)
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("relative drift [%]")
    ax1.set_title("energy budget (window closed at deck arrival)")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    ladder = payload["energy_ladder"]
    ax2.plot([r["dx"] for r in ladder], [r["energy_drift_total"] * 100 for r in ladder],
             "o-", label="total")
    ax2.plot([r["dx"] for r in ladder], [r["energy_drift_mech"] * 100 for r in ladder],
             "s--", label="KE + PE")
    ax2.axhline(1.0, color="crimson", ls="--", lw=0.9, label="1% criterion")
    ax2.set_xscale("log")
    ax2.set_xlabel("dx [m]")
    ax2.set_ylabel("peak drift over window [%]")
    ax2.set_title("resolution ladder")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig3_energy.png", dpi=130)
    plt.close(fig)

    # ---- fig 4: gauges ----------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ax.plot(t, series["apex"], lw=1.4, label="splash apex [m]")
    ax.plot(t, [cfg.deck_height] * len(t), color="0.4", ls="--", lw=0.9,
            label="deck level")
    ax2 = ax.twinx()
    ax2.plot(t, [v * 1e3 for v in series["deck_volume"]], color="tab:orange", lw=1.4,
             label="deck water [L/m]")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("height [m]")
    ax2.set_ylabel("deck inundation [L/m]")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=7, loc="upper left")
    ax.set_title(f"deck arrival {crit['energy_window_s']:.2f} s, "
                 f"apex {payload['gauges']['peak_apex_m']:.3f} m")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig4_gauges.png", dpi=130)
    plt.close(fig)

    # ---- fig 5: floe estimate --------------------------------------------- #
    floe = payload["floe_estimate"]
    rows = floe["floes"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    idx = [r["floe"] for r in rows]
    ax.bar(idx, [r["launch_speed_m_s"] for r in rows], color="tab:blue",
           label="impulse model")
    ax.axhline(floe["upper_bound_speed_m_s"], color="crimson", ls="--", lw=1.2,
               label="energy upper bound")
    ax.set_xlabel("floe index (deck position)")
    ax.set_ylabel("launch speed [m/s]")
    ax.set_title("碎冰抛掷 reduced-model estimate")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig5_floes.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
