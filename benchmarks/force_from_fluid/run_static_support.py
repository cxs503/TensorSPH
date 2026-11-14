"""Static hydrostatic support of the ``fsi2`` gate case: root cause and repair.

What this benchmark decides
---------------------------
``docs/force_from_fluid/README.md`` recorded an honest FAIL: at ``t = 0`` the
fluid column supported only 73.4 % of its own weight on the floor and pushed on
the gate with 74.0 % of the analytic thrust ``0.5 rho g H^2``, both load routes
(pair and panel) sharing the shortfall, and the root cause was left *unlocated*.

This benchmark localises it and measures the repair. The deficit is concentrated
in the two rows adjacent to the wall, where the pressure gradient is **inverted**
(``p`` grows with height), because the wall particles enter the fluid's
summation density with the nominal ``rho0 dx^2`` while the fluid beside them is
compressed to ``rho0 (1 + rho0 g d / B)^(1/gamma)``. With ``c0 = 20 m/s`` and
``gamma = 7`` the bulk modulus is 57 kPa, so a 0.3 % density step across the
interface is a 30 % pressure step: the stiff Tait law is the amplifier.

Measured here, per resolution ``dx`` in {0.02, 0.01, 0.005} m:

* baseline (switch off),
* ``boundary_consistent_density`` (the wall's EOS density from its own
  extrapolated face pressure is fed into the summation density),
* a diagnostic ceiling in which the wall mass is set from the *closed-form*
  hydrostatic density (an oracle, not a usable scheme - it shows how much of the
  deficit is attributable to the wall mass by construction),

each with ``floor_pair / rho g A``, ``gate_pair / 0.5 rho g H^2`` and
``gate_panel / 0.5 rho g H^2``, plus a band decomposition of the vertical
support (bottom two rows / interior / top four rows) that shows where the
missing support sits, and a 900-step conservative transient that must keep mass
conservation and the momentum budget untouched.

Nothing is calibrated: the references are geometry facts
(:func:`tensorsph.force_from_fluid.hydrostatic_reference`).

Repro::

    PYTHONPATH=src python benchmarks/force_from_fluid/run_static_support.py
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch

from tensorsph.dam_break import make_column
from tensorsph.droplet_impact import (hydrostatic_density, resolve_device, rhs,
                                      tait_pressure, wall_lattice)
from tensorsph.force_from_fluid import (GateCaseConfig, gate_lattice,
                                        group_force, hydrostatic_reference,
                                        run_gate_case, wall_groups)

LADDER = (0.02, 0.01, 0.005)
MODES = ("baseline", "boundary-consistent", "oracle")


def _dt(cfg) -> float:
    """Acoustic Courant 0.25, so the ladder refines in time as well as space."""
    return 0.25 * cfg.dx / cfg.sound_speed


def _wall(cfg, dev):
    base = wall_lattice(cfg, dev, top=cfg.tank_top())
    gate = gate_lattice(cfg, dev)
    return torch.cat((base, gate), dim=0), base.shape[0]


def _oracle_wall_mass(wall, cfg, dt):
    """Diagnostic ceiling: wall mass from the closed-form hydrostatic density."""
    rho_w = hydrostatic_density(wall[:, 1], cfg, surface=cfg.column_height)
    return (rho_w * cfg.dx * cfg.dx).to(wall.dtype)


def _state(cfg, mode):
    """One ``t = 0`` state; returns the loads, per-row table and band sums."""
    dev = resolve_device(cfg.device)
    pos, mass = make_column(cfg, dev)
    wall, gate_start = _wall(cfg, dev)
    groups = wall_groups(wall, cfg, gate_start)
    ref = hydrostatic_reference(cfg)

    override = None
    if mode == "oracle":
        import tensorsph.droplet_impact as di
        override = _oracle_wall_mass(wall, cfg, cfg.dt)
        original = di.wall_mass
        di.wall_mass = lambda c, nw, device: override
        try:
            rho, _, acc, loads = rhs(pos, torch.zeros_like(pos), mass, cfg,
                                     wall=wall, dt=cfg.dt, split=True)
        finally:
            di.wall_mass = original
    else:
        rho, _, acc, loads = rhs(pos, torch.zeros_like(pos), mass, cfg,
                                 wall=wall, dt=cfg.dt, split=True)

    reaction = group_force(loads["reaction"], groups)
    panel = float(loads["face_pressure"][groups["gate_face"]].sum()) * cfg.dx
    p = tait_pressure(rho, cfg)

    # Per-row vertical balance: the force a row transmits downwards is
    # m (a_z + g); normalised by the row's own weight it is the fraction of the
    # weight the row hands to the particles below it (1 = fully supported).
    total_weight = cfg.gravity * float(mass.sum())
    ny = int(round(cfg.column_height / cfg.dx))
    rows = []
    for k in range(ny):
        sel = (pos[:, 1] >= k * cfg.dx) & (pos[:, 1] < (k + 1) * cfg.dx)
        row_weight = float(mass[sel].sum()) * cfg.gravity
        yc = float(pos[sel, 1].mean())
        rows.append({
            "row": k,
            "n": int(sel.sum()),
            "acc_z_over_g": float(acc[sel, 1].mean()) / -cfg.gravity,
            "support_fraction": float((mass[sel] * (acc[sel, 1] + cfg.gravity)).sum())
                                / row_weight,
            # share of the total column weight this row hands downwards; the
            # rows sum exactly to floor_pair / W (Newton's third law on the
            # fluid body: sum_i m_i (a_z + g) = F_floor).
            "force_fraction": float((mass[sel] * (acc[sel, 1] + cfg.gravity)).sum())
                              / total_weight,
            "p_mean": float(p[sel].mean()),
            "p_analytic": cfg.rho0 * cfg.gravity * (cfg.column_height - yc),
            "rho_over_rho0": float(rho[sel].mean()) / cfg.rho0,
            "rho_analytic_over_rho0": float(hydrostatic_density(
                torch.tensor(yc, dtype=pos.dtype), cfg,
                surface=cfg.column_height)) / cfg.rho0,
        })

    bands = {
        "bottom_two_rows": sum(r["force_fraction"] for r in rows[:2]),
        "interior": sum(r["force_fraction"] for r in rows[2:-4]),
        "top_four_rows": sum(r["force_fraction"] for r in rows[-4:]),
        "band_mass_share": None,
    }
    bands["band_mass_share"] = None
    mass_share = {
        "bottom_two_rows": sum(float(mass[(pos[:, 1] >= k * cfg.dx)
                                         & (pos[:, 1] < (k + 1) * cfg.dx)].sum())
                               for k in (0, 1)) * cfg.gravity / total_weight,
        "interior": sum(float(mass[(pos[:, 1] >= k * cfg.dx)
                                  & (pos[:, 1] < (k + 1) * cfg.dx)].sum())
                        for k in range(2, ny - 4)) * cfg.gravity / total_weight,
        "top_four_rows": sum(float(mass[(pos[:, 1] >= k * cfg.dx)
                                       & (pos[:, 1] < (k + 1) * cfg.dx)].sum())
                             for k in range(ny - 4, ny)) * cfg.gravity / total_weight,
    }
    return {
        "mode": mode,
        "dx": cfg.dx,
        "dt": cfg.dt,
        "fluid_particles": int(pos.shape[0]),
        "wall_particles": int(wall.shape[0]),
        "floor_pair": -float(reaction["floor"][1]),
        "floor_pair_ratio": -float(reaction["floor"][1]) / ref["fluid_weight"],
        "gate_pair_ratio": (float(reaction["gate"][0])
                            / ref["vertical_wall_thrust"]),
        "gate_panel_ratio": panel / ref["vertical_wall_thrust"],
        "pair_vs_panel": (float(reaction["gate"][0]) / panel) if panel else float("nan"),
        "bottom_row_inverted": bool(rows[0]["p_mean"] < rows[1]["p_mean"]),
        "bottom_row_p_ratio": rows[0]["p_mean"] / rows[1]["p_mean"],
        "bottom_row_p_ratio_to_analytic": rows[0]["p_mean"] / rows[0]["p_analytic"],
        "peak_rho_compression": float((rho.max() / cfg.rho0) - 1.0),
        "support_bands": bands,
        "band_mass_share": mass_share,
        "rows": rows,
    }


def _transient(fixed: bool, steps: int = 900) -> Dict[str, float]:
    base = dict(dx=0.02, pool_width=1.0, tank_height=0.5, viscosity=0.0,
                steps=steps)
    cfg = GateCaseConfig(boundary_consistent_density=fixed, **base)
    out = run_gate_case(cfg, record_every=steps)
    s = out["summary"]
    thrust = [f[0] for f in out["reaction_forces"]["gate"]]
    tail = thrust[len(thrust) // 2:]
    ref = hydrostatic_reference(cfg)
    return {
        "boundary_consistent_density": fixed,
        "steps": steps,
        "mass_drift_relative": s["mass_drift_relative"],
        "boundary_clamp_events": s["boundary_clamp_events"],
        "residual_relative": s["residual_relative"],
        "gate_tail_mean_ratio": (sum(tail) / len(tail)) / ref["vertical_wall_thrust"],
        "peak_rho_compression": s["peak_rho_compression"],
        "final_time": out["times"][-1],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/force_from_fluid/static_support.json")
    ap.add_argument("--table", action="store_true",
                    help="print the full per-row table of the coarsest case")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--no-transient", action="store_true")
    args = ap.parse_args(argv)

    records: List[Dict] = []
    for dx in LADDER:
        for mode in MODES:
            cfg = GateCaseConfig(
                dx=dx, steps=0, dt=_dt(GateCaseConfig(dx=dx)),
                boundary_consistent_density=(mode == "boundary-consistent"))
            rec = _state(cfg, mode)
            records.append(rec)
            print("%-21s dx=%.3f  floor/W=%.4f  gate_pair/F=%.4f  gate_panel/F=%.4f"
                  "  inverted=%s" % (mode, dx, rec["floor_pair_ratio"],
                                     rec["gate_pair_ratio"],
                                     rec["gate_panel_ratio"],
                                     rec["bottom_row_inverted"]))
            if args.table and mode != "oracle" and dx == 0.02:
                print("   row  a_z/g  support  p_mean  p_analytic  rho/rho0  rho_ana")
                for r in rec["rows"]:
                    print("   %3d %7.3f %8.3f %8.1f %10.1f %9.5f %9.5f"
                          % (r["row"], r["acc_z_over_g"], r["support_fraction"],
                             r["p_mean"], r["p_analytic"], r["rho_over_rho0"],
                             r["rho_analytic_over_rho0"]))

    print("\nband decomposition (fraction of the total column weight supported,"
          " dx=0.02):")
    for rec in records:
        if rec["dx"] != 0.02:
            continue
        b = rec["support_bands"]
        print("  %-21s bottom2=%+.4f  interior=%+.4f  top4=%+.4f"
              % (rec["mode"], b["bottom_two_rows"], b["interior"],
                 b["top_four_rows"]))

    transient = {}
    if not args.no_transient:
        transient["baseline"] = _transient(False)
        transient["boundary_consistent"] = _transient(True)
        for k, v in transient.items():
            print("transient %-20s gate tail %.4f  mass drift %.2e  budget %.2e"
                  % (k, v["gate_tail_mean_ratio"], v["mass_drift_relative"],
                     v["residual_relative"]))

    record = {
        "case": "fsi2 rigid-gate static hydrostatic support (force from fluid)",
        "ladder": list(LADDER),
        "modes": list(MODES),
        "states": records,
        "transient_conservative": transient,
        "repro": ("PYTHONPATH=src python benchmarks/force_from_fluid/"
                  "run_static_support.py"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(record, fh, indent=2)
    print("\nwrote", args.out)
    if not args.no_figure:
        _figure(record, os.path.join(os.path.dirname(args.out),
                                     "fig_static_support.png"))
    return 0


def _figure(record, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    style = {"baseline": "o--", "boundary-consistent": "s-", "oracle": "^:"}
    label = {"baseline": "baseline (switch off)",
             "boundary-consistent": "boundary-consistent density",
             "oracle": "ceiling: exact EOS wall density (diagnostic)"}
    for mode in MODES:
        sel = sorted([r for r in record["states"] if r["mode"] == mode],
                     key=lambda r: r["dx"])
        axes[0].plot([r["dx"] for r in sel], [r["floor_pair_ratio"] for r in sel],
                     style[mode], label=label[mode])
        axes[1].plot([r["dx"] for r in sel], [r["gate_pair_ratio"] for r in sel],
                     style[mode])
        axes[1].plot([r["dx"] for r in sel], [r["gate_panel_ratio"] for r in sel],
                     style[mode][0] + ":", alpha=0.6)
    for ax, ylab, title in (
            (axes[0], r"floor reaction / $\rho g A$", "vertical support"),
            (axes[1], r"gate thrust / $\frac{1}{2}\rho g H^2$", "horizontal thrust")):
        ax.axhline(1.0, color="k", lw=0.8)
        ax.axhspan(0.97, 1.03, color="g", alpha=0.12)
        ax.axhspan(0.85, 1.15, color="orange", alpha=0.07)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("dx [m]")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[1].set_title("horizontal thrust (solid: pair, dotted: panel)")
    axes[0].legend(fontsize=7, loc="lower right")

    for mode, colour in (("baseline", "tab:red"),
                         ("boundary-consistent", "tab:blue")):
        rec = next(r for r in record["states"]
                   if r["mode"] == mode and r["dx"] == 0.02)
        rows = rec["rows"][:6]
        axes[2].plot([r["row"] for r in rows], [r["p_mean"] for r in rows],
                     "o-", color=colour, label=mode)
    rows = next(r for r in record["states"] if r["dx"] == 0.02)["rows"][:6]
    axes[2].plot([r["row"] for r in rows], [r["p_analytic"] for r in rows],
                 "k--", label=r"analytic $\rho_0 g(H-y)$")
    axes[2].set_xlabel("row above the floor (dx = 0.02 m)")
    axes[2].set_ylabel("mean pressure [Pa]")
    axes[2].set_title("wall-adjacent pressure gradient")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print("wrote", path)


if __name__ == "__main__":
    raise SystemExit(main())