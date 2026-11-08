"""Verification bundle: does the reduced floe launch model account for added mass?

Usage
-----
    python benchmarks/ice_floe_added_mass/run.py [--out artifacts/ice_floe_added_mass]

Produces, under the output directory:

* ``report.json``   every number quoted in ``docs/verification_ice_floe_added_mass.md``
* ``fig1_added_mass.png``   potential-flow anchor: C_m vs resolution, isotropy, plane limits
* ``fig2_launch.png``       dry-mass vs added-mass launch speed, apex and range for the
                            reduced bow ice-splash floe model

The anchor is the closed-form value for a 2-D circular section in an unbounded
fluid, ``C_m = 1`` (Lamb, *Hydrodynamics*, §71; Landau & Lifshitz, *Fluid
Mechanics*, §11). The potential-flow solve is validated against it, its
truncation and resolution bias is measured, and the resulting inertia is then
applied to the launch estimate of :mod:`tensorsph.ice_splash`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorsph.added_mass import (
    DISC_ADDED_MASS_COEFFICIENT,
    cylinder_added_mass,
    speed_ratio_with_added_mass,
)
from tensorsph.ice_splash import (
    IceSplashConfig,
    floe_layout,
    floe_launch_estimate,
    run_case,
)

REPO = Path(__file__).resolve().parents[2]

LADDER = (8, 12, 16, 24)
BOX_REPORT = 6.0
GAPS = (2.0, 4.0)          # thin-gap values are unresolved on a Cartesian grid


def splash_case(dx: float = 0.0125, t_target: float = 0.46) -> IceSplashConfig:
    cfg = IceSplashConfig(
        dx=dx, column_width=0.20, column_height=0.28, pool_width=1.0,
        tank_height=0.75, sheet_start=0.20, sheet_angle_deg=25.0,
        deck_height=0.28, sound_speed=20.0, viscosity=0.0, device="cpu",
        output_every=5, max_courant=0.3, n_floes=5, seed=7,
    )
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(round(t_target / cfg.dt))
    return cfg


def anchor_ladder() -> list[dict]:
    rows = []
    for cpr in LADDER:
        started = time.time()
        result = cylinder_added_mass(cells_per_radius=cpr, box_radii=BOX_REPORT,
                                     initial="zero", tolerance=1e-5,
                                     max_iterations=20000)
        rows.append({
            "cells_per_radius": cpr,
            "grid": list(result.grid),
            "coefficient": result.coefficient,
            "error_vs_lamb_pct": 100.0 * (result.coefficient
                                          / DISC_ADDED_MASS_COEFFICIENT - 1.0),
            "iterations": result.iterations,
            "residual": result.residual,
            "tail_fraction": result.tail_fraction,
            "wall_clock_s": time.time() - started,
        })
    return rows


def box_sensitivity(cpr: int = 12) -> list[dict]:
    rows = []
    for box in (6.0, 10.0):
        result = cylinder_added_mass(cells_per_radius=cpr, box_radii=box,
                                     initial="zero", tolerance=1e-5,
                                     max_iterations=20000)
        rows.append({"box_radii": box, "grid": list(result.grid),
                     "coefficient": result.coefficient})
    return rows


def isotropy(cpr: int = 12) -> dict:
    out = {"cells_per_radius": cpr}
    for axis in ("x", "y"):
        out[axis] = cylinder_added_mass(cells_per_radius=cpr, box_radii=BOX_REPORT,
                                        direction=axis, initial="zero",
                                        tolerance=1e-5).coefficient
    out["relative_difference_pct"] = 100.0 * abs(out["x"] / out["y"] - 1.0)
    return out


def plane_limits(cpr: int = 12) -> dict:
    free_x = cylinder_added_mass(cells_per_radius=cpr, box_radii=BOX_REPORT,
                                 direction="x", tolerance=1e-5).coefficient
    free_y = cylinder_added_mass(cells_per_radius=cpr, box_radii=BOX_REPORT,
                                 direction="y", tolerance=1e-5).coefficient
    rows = []
    for boundary in ("rigid_wall", "free_surface"):
        for axis in ("x", "y"):
            free = free_x if axis == "x" else free_y
            for gap in GAPS:
                result = cylinder_added_mass(cells_per_radius=cpr, box_radii=BOX_REPORT,
                                             boundary=boundary, direction=axis,
                                             gap=gap, tolerance=1e-5)
                rows.append({
                    "boundary": boundary,
                    "direction": axis,
                    "gap_radii": gap,
                    "coefficient": result.coefficient,
                    "ratio_to_free_field": result.coefficient / free,
                })
    worst = max(abs(row["ratio_to_free_field"] - 1.0) for row in rows)
    return {"cells_per_radius": cpr, "rows": rows,
            "max_ratio_deviation_pct": 100.0 * worst}


def launch_bias() -> dict:
    cfg = splash_case()
    started = time.time()
    run = run_case(cfg)
    splash_wall = time.time() - started
    floes = floe_layout(cfg, torch.device("cpu"))
    dry = floe_launch_estimate(run, cfg, floes)
    wet_cfg = IceSplashConfig(**{**cfg.__dict__,
                                 "floe_added_mass_coefficient": DISC_ADDED_MASS_COEFFICIENT})
    wet = floe_launch_estimate(run, wet_cfg, floes)
    rows = []
    for dry_row, wet_row in zip(dry["floes"], wet["floes"]):
        rows.append({
            "floe": dry_row["floe"],
            "impulse_Ns_per_m": dry_row["impulse_Ns_per_m"],
            "peak_force_N_per_m": dry_row["peak_force_N_per_m"],
            "dry_mass_kg_per_m": dry_row["mass_kg_per_m"],
            "added_mass_kg_per_m": wet_row["added_mass_kg_per_m"],
            "speed_dry_m_s": dry_row["launch_speed_m_s"],
            "speed_with_added_mass_m_s": wet_row["launch_speed_with_added_mass_m_s"],
            "speed_ratio": wet_row["speed_ratio_with_added_mass"],
            "apex_dry_m": dry_row["ballistic_apex_m"],
            "apex_with_added_mass_m": wet_row["ballistic_apex_with_added_mass_m"],
            "range_dry_m": dry_row["ballistic_range_m"],
            "range_with_added_mass_m": wet_row["ballistic_range_with_added_mass_m"],
        })
    return {
        "case": {
            "dx_m": cfg.dx, "t_end_s": cfg.steps * cfg.dt,
            "particles": int(run.final_positions.shape[0]),
            "floe_radius_m": cfg.floe_radius, "floe_rho_kg_m3": cfg.floe_rho,
            "rho_water_kg_m3": cfg.rho0, "sound_speed_m_s": cfg.sound_speed,
            "wall_clock_s": splash_wall,
        },
        "ratio_closed_form": speed_ratio_with_added_mass(cfg.floe_rho, cfg.rho0,
                                                        DISC_ADDED_MASS_COEFFICIENT),
        "rows": rows,
        "peak_splash_ke_J_per_m": run.peak_splash_kinetic_energy,
        "upper_bound_speed_m_s": dry["upper_bound_speed_m_s"],
    }


def figures(report: dict, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ladder = report["anchor_ladder"]
    cpr = [row["cells_per_radius"] for row in ladder]
    err = [row["error_vs_lamb_pct"] for row in ladder]
    axes[0].plot(cpr, err, "o-", label="potential-flow solve")
    axes[0].axhline(0.0, color="k", lw=1, label="Lamb (1932): $C_m = 1$")
    axes[0].set_xlabel("grid cells per radius")
    axes[0].set_ylabel("error vs closed form [%]")
    axes[0].set_title("anchor: unbounded circular section")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].bar(["x", "y"], [report["isotropy"]["x"], report["isotropy"]["y"]],
                color=["#3b6fb6", "#b66b3b"])
    axes[1].axhline(1.0, color="k", lw=1)
    axes[1].set_ylim(0.0, 1.3)
    axes[1].set_ylabel("$C_m$ / ($\\rho\\pi a^2$)")
    axes[1].set_title(f"isotropy at {report['isotropy']['cells_per_radius']} cells/radius")
    axes[1].grid(alpha=0.3, axis="y")

    for boundary, style, colour in (("rigid_wall", "o-", "#b6453b"),
                                    ("free_surface", "s--", "#2f8f6f")):
        for axis, filled in (("x", True), ("y", False)):
            pts = [row for row in report["plane_limits"]["rows"]
                   if row["boundary"] == boundary and row["direction"] == axis]
            axes[2].plot([row["gap_radii"] for row in pts],
                         [row["ratio_to_free_field"] for row in pts],
                         style, color=colour,
                         markerfacecolor=colour if filled else "none",
                         label=f"{boundary.replace('_', ' ')}, {axis}-translation")
    axes[2].axhline(1.0, color="k", lw=1)
    axes[2].set_xlabel("gap to the plane [radii]")
    axes[2].set_ylabel("$C_m(\\mathrm{gap}) / C_m(\\infty)$")
    axes[2].set_title("plane limits: both surfaces recover the free field")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    path1 = out / "fig1_added_mass.png"
    fig.savefig(path1, dpi=140)
    plt.close(fig)
    made.append(str(path1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = report["launch_bias"]["rows"]
    idx = [row["floe"] for row in rows]
    width = 0.38
    axes[0].bar([i - width / 2 for i in idx], [row["speed_dry_m_s"] for row in rows],
                width, label="dry mass (upper bound)")
    axes[0].bar([i + width / 2 for i in idx],
                [row["speed_with_added_mass_m_s"] for row in rows],
                width, label="with added mass ($C_m=1$)")
    axes[0].set_xlabel("floe index")
    axes[0].set_ylabel("launch speed [m/s]")
    axes[0].set_title("reduced-model floe launch speed")
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    axes[1].bar([i - width / 2 for i in idx], [row["apex_dry_m"] for row in rows],
                width, label="dry mass")
    axes[1].bar([i + width / 2 for i in idx],
                [row["apex_with_added_mass_m"] for row in rows],
                width, label="with added mass ($C_m=1$)")
    axes[1].set_xlabel("floe index")
    axes[1].set_ylabel("ballistic apex [m]")
    axes[1].set_title("implied floe toss height")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path2 = out / "fig2_launch.png"
    fig.savefig(path2, dpi=140)
    plt.close(fig)
    made.append(str(path2))
    return made


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "artifacts" / "ice_floe_added_mass"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    report = {
        "reference": {
            "quantity": "added-mass coefficient of a 2-D circular section",
            "closed_form": DISC_ADDED_MASS_COEFFICIENT,
            "source": "Lamb, Hydrodynamics, 6th ed., §71; Landau & Lifshitz, "
                      "Fluid Mechanics, §11",
        },
        "anchor_ladder": anchor_ladder(),
        "box_sensitivity": box_sensitivity(),
        "isotropy": isotropy(),
        "plane_limits": plane_limits(),
        "launch_bias": launch_bias(),
        "config": {
            "ladder_cells_per_radius": list(LADDER),
            "box_radii": BOX_REPORT,
            "gaps_radii": list(GAPS),
            "tolerance": 1e-5,
            "initial": "zero (anchor) / dipole (plane cases)",
        },
    }
    report["wall_clock_s"] = time.time() - started
    if not args.no_figures:
        report["figures"] = figures(report, out)
    with open(out / "report.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("anchor_ladder", "isotropy", "launch_bias")}, indent=1)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
