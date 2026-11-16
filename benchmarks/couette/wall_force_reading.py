#!/usr/bin/env python3
"""Force-based (Newton) reading of the Couette wall shear.

Anchor: ``XVAL-SPH-METHOD-SELECTION-2026-11-12``.

Why this script exists
----------------------
The method-selection registry banks, for ``wcsph_wall_no_slip``, wall-shear errors
of 14.20 % / 17.08 % at the *sliding* wall (``x_couette_viscous_time_top_h8/h16``)
against ``tau = rho nu U / H = 40 Pa``.  ``docs/dev_report_couette_sliding_wall_reading_m20907.md``
showed that the *banked* number is a one-spacing secant ``(u[N-1]-u[N-2])/dy`` and
that the same profile re-read with a 4-row fit gives 2.83 % / 8.60 %.  That audit
left one question open by construction (§5): a slope is not a stress.  The
textbook definition of wall shear is the **tangential force the fluid exerts on
the wall**, available from ``droplet_impact.rhs(..., split=True)`` as
``loads["reaction"]``.  The attempt in ``diag_wall_force.py`` was discarded
because on the *periodic* Couette lattice (three copies of the fluid cloud, one
wall band) the reported reaction carried a 12 Pa residual on a 40 Pa stress.

This script fixes that bookkeeping and turns the force reading into a verdict:

* ``periodic3`` -- the wall lattice is **tripled in x** exactly like the fluid
  cloud (three non-overlapping ``L``-wide copies), so each fluid copy sees a
  complete periodic wall.  The force on the ``x in [0,L)`` copy *is* the
  per-period load; no division by the number of copies and no ``x``-window
  masking is needed.  Two internal identities come out as gates:
  Newton's third law (``sum_all F_x + sum m a_x = 0``) and periodic invariance
  (``F_copy0 = F_all / 3``).
* ``legacy`` -- the single wall band of ``build_walls`` (extended 3 dx past both
  ends) with the ``x in [0,L)`` filter used by ``diag_wall_force.py``.  Kept only
  to reproduce, and quantify, the 12 Pa residual that made the force reading be
  dropped; its numbers are not used for any accuracy claim.

The accuracy claims read ``|1 - tau_force/tau_exact|`` at both walls and are
compared with the banked secant estimator on the very same run.

Run::

    PYTHONPATH=src python benchmarks/couette/wall_force_reading.py --res 8 16
    PYTHONPATH=src python benchmarks/couette/wall_force_reading.py --check
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
_RUN = Path(__file__).resolve().parent / "run.py"
_spec = importlib.util.spec_from_file_location("couette_run", _RUN)
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)

OUT = REPO / "benchmarks" / "couette" / "out_force"
ARTIFACT = OUT / "wall_force_reading.json"

#: Predeclared acceptance thresholds (owner standard 15 %, solver gate 5 %) plus
#: the two bookkeeping identities the force reading needs before it means
#: anything.  Written before any number was read.
GATES = {
    "third_law_residual_pct": 1.0,   # |sum F + sum m a| / (tau_ex L)
    "copy_invariance_pct": 1.0,      # |F_copy0 - F_all/3| / (tau_ex L)
    "wall_force_pct": 5.0,           # |1 - tau_force/tau_exact| per wall
    "owner_tolerance_pct": 15.0,
}


# --------------------------------------------------------------------------- #
# Wall lattices
# --------------------------------------------------------------------------- #

def periodic_walls(cfg, height: float, length: float, layers: int,
                   wall_speed: float, device, copies: int = 3):
    """``copies`` non-overlapping ``L``-wide wall bands tiling ``x``.

    Same vertical layout as :func:`benchmarks.couette.run.build_walls`
    (``layers`` bands of spacing ``dx`` outside each wall, centres at
    ``-dx/2 - (k-1)dx`` and ``H + dx/2 + (k-1)dx``), but the horizontal lattice
    is exactly one period wide (``length/dx`` columns at mid-cell) and is then
    tiled without overlap.  Returns ``(pos, wall_vel, copy_id)``.
    """
    dx = cfg.dx
    ncol = int(round(length / dx))
    assert abs(ncol * dx - length) < 1e-12, "length must be an integer multiple of dx"
    xs = (torch.arange(ncol, device=device) + 0.5) * dx
    parts, vels, ids = [], [], []
    for k in range(copies):
        shift = (k - (copies // 2)) * length
        for layer in range(1, layers + 1):
            yb = torch.full_like(xs, -dx * 0.5 - (layer - 1) * dx)
            bot = torch.stack((xs + shift, yb), dim=-1)
            yt = torch.full_like(xs, height + dx * 0.5 + (layer - 1) * dx)
            top = torch.stack((xs + shift, yt), dim=-1)
            for band in (bot, top):
                parts.append(band)
                vel = torch.zeros_like(band)
                if band is top:
                    vel[:, 0] = wall_speed
                vels.append(vel)
                ids.append(torch.full((band.shape[0],), k, dtype=torch.long,
                                      device=device))
    return torch.cat(parts, 0), torch.cat(vels, 0), torch.cat(ids, 0)


# --------------------------------------------------------------------------- #
# One arm = one full run
# --------------------------------------------------------------------------- #

def run_arm(res: int, viscous_times: float = cr.VISCOUS_TIMES,
            layout: str = "periodic3", threads: int = 1) -> dict:
    torch.set_num_threads(threads)
    dev = torch.device("cpu")
    dx = cr.HEIGHT / res
    length = cr.COLUMNS * dx
    cfg = cr.make_config(dx, cr.VISCOSITY, cr.FRICTION, dev)
    pos, mass = cr.build_fluid(cfg, cr.HEIGHT, length, dev)
    n = pos.shape[0]

    if layout == "periodic3":
        wall, wall_vel, copy_id = periodic_walls(cfg, cr.HEIGHT, length,
                                                 cfg.wall_layers, cr.WALL_SPEED, dev)
        measured = copy_id == 1          # the x in [0, L) copy
    elif layout == "legacy":
        wall, wall_vel = cr.build_walls(cfg, cr.HEIGHT, length, cfg.wall_layers,
                                        cr.WALL_SPEED, dev)
        copy_id = torch.zeros((wall.shape[0],), dtype=torch.long, device=dev)
        # the x-window filter of diag_wall_force.py, applied at sampling time
        measured = torch.ones((wall.shape[0],), dtype=torch.bool, device=dev)
    else:
        raise ValueError(f"unknown layout {layout!r}")

    top = wall[:, 1] > cr.HEIGHT / 2
    bot = ~top

    h = cfg.h_factor * dx
    dt = min(0.3 * h / cfg.sound_speed, 0.125 * h * h / cr.VISCOSITY)
    t_end = viscous_times * cr.HEIGHT ** 2 / cr.VISCOSITY
    avg_from = t_end - cr.AVERAGE_FRACTION * t_end
    steps = int(round(t_end / dt))

    vel = torch.zeros_like(pos)
    acc_sum = torch.zeros_like(pos)
    fx_meas_top = 0.0
    fx_meas_bot = 0.0
    fx_all = 0.0
    fx_all_top = 0.0
    fx_all_bot = 0.0
    fx_copy_all = 0.0          # sum over the measured copy only, all walls
    mom_gain = 0.0
    samples = 0
    wall_clock = time.time()

    for step in range(steps):
        pos_ext = cr.periodic_images(pos, length)
        vel_ext = torch.cat((vel, vel, vel), dim=0)
        mass_ext = torch.cat((mass, mass, mass), dim=0)
        _, _, acc_ext, loads = cr.rhs(pos_ext, vel_ext, mass_ext, cfg, wall=wall,
                                      wall_vel=wall_vel, split=True)
        acc = acc_ext[:n].clone()
        if step * dt >= avg_from:
            rx = loads["reaction"][:, 0].detach()
            if layout == "legacy":
                inx = (wall[:, 0] > 0.0) & (wall[:, 0] < length)
                sel = inx
            else:
                sel = measured
            fx_meas_top += float(rx[top & sel].sum())
            fx_meas_bot += float(rx[bot & sel].sum())
            fx_all += float(rx.sum())
            fx_all_top += float(rx[top].sum())
            fx_all_bot += float(rx[bot].sum())
            fx_copy_all += float(rx[sel].sum())
            # every fluid copy is present in the extended cloud; the wall reaction
            # is reported for all wall copies, so the identity below is per full system
            mom_gain += float((mass_ext * acc_ext[:3 * n, 0]).sum())
            acc_sum += vel
            samples += 1
        vel = vel + dt * acc
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], length)

    span = max(samples, 1) * dt
    tau_ex = cr.analytic_scalars(cr.HEIGHT, cr.WALL_SPEED, cr.VISCOSITY)["wall_shear"]
    scale = tau_ex * length                       # N per unit depth

    # force readings, both walls (fluid on wall, tangential x component)
    # sign convention: the fluid drags the sliding wall backwards (F_x < 0) and
    # pushes the fixed wall forwards (F_x > 0); the stress magnitude is what the
    # analytic tau measures, so the top reading is negated.
    tau_top = -fx_meas_top / samples / length
    tau_bot = fx_meas_bot / samples / length

    u_mean = acc_sum / max(samples, 1)
    rows, profile = cr.bin_rows(pos, u_mean, cr.HEIGHT, dx)
    m = cr.profile_metrics(profile, cr.HEIGHT, cr.WALL_SPEED, cr.VISCOSITY)
    return {
        "layout": layout,
        "resolution_H_over_dx": res,
        "viscous_times": viscous_times,
        "steps": steps,
        "dt": dt,
        "n_fluid": int(n),
        "n_wall": int(wall.shape[0]),
        "samples": samples,
        "averaged_span_s": span,
        "length_m": length,
        "tau_exact_pa": tau_ex,
        # force-based readings (the quantity a stress is defined by)
        "tau_top_force_pa": tau_top,
        "tau_bottom_force_pa": tau_bot,
        "err_top_force_pct": 100.0 * (tau_top / tau_ex - 1.0),
        "err_bottom_force_pct": 100.0 * (tau_bot / tau_ex - 1.0),
        # bookkeeping identities
        "third_law_residual_pct": 100.0 * (mom_gain + fx_all) / samples / length
                                   / tau_ex,
        "copy_invariance_pct": 100.0 * max(
            abs(fx_meas_top / samples / length - fx_all_top / 3.0 / samples / length),
            abs(fx_meas_bot / samples / length - fx_all_bot / 3.0 / samples / length),
        ) / tau_ex,
        "tau_top_all_pa": -fx_all_top / 3.0 / samples / length,
        "tau_bottom_all_pa": fx_all_bot / 3.0 / samples / length,
        "net_force_on_fluid_measured_pa": -(tau_top + tau_bot),
        # the banked (secant) estimator on the very same run, for contrast
        "err_top_secant_pct": 100.0 * (m["wall_shear_ratio_top"] - 1.0),
        "err_bottom_secant_pct": 100.0 * (m["wall_shear_ratio"] - 1.0),
        "interior_l2_pct": 100.0 * m["interior_l2"],
        "flux_ratio": m["flux_ratio"],
        "profile": [{"y_over_H": r["y_over_H"],
                     "u_over_U": r["u_mean"] / cr.WALL_SPEED} for r in profile],
        "wall_clock_s": time.time() - wall_clock,
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def verdict(arms: list[dict]) -> dict:
    """Apply the predeclared gates to the periodic3 arms only."""
    main = [a for a in arms if a["layout"] == "periodic3"]
    legacy = [a for a in arms if a["layout"] == "legacy"]
    per_arm = []
    for a in main:
        per_arm.append({
            "resolution_H_over_dx": a["resolution_H_over_dx"],
            "top_pct": a["err_top_force_pct"],
            "bottom_pct": a["err_bottom_force_pct"],
            "third_law_pct": a["third_law_residual_pct"],
            "copy_invariance_pct": a["copy_invariance_pct"],
            "force_gate": (abs(a["err_top_force_pct"]) <= GATES["wall_force_pct"]
                           and abs(a["err_bottom_force_pct"]) <= GATES["wall_force_pct"]),
            "owner_standard": (abs(a["err_top_force_pct"]) <= GATES["owner_tolerance_pct"]
                               and abs(a["err_bottom_force_pct"])
                               <= GATES["owner_tolerance_pct"]),
            "third_law_ok": abs(a["third_law_residual_pct"]) <= GATES["third_law_residual_pct"],
            "copy_invariance_ok": abs(a["copy_invariance_pct"])
                                  <= GATES["copy_invariance_pct"],
            "bookkeeping_ok": (abs(a["third_law_residual_pct"])
                               <= GATES["third_law_residual_pct"]
                               and abs(a["copy_invariance_pct"])
                               <= GATES["copy_invariance_pct"]),
        })
    return {
        "gates": GATES,
        "periodic3": per_arm,
        "legacy_third_law_pct": [a["third_law_residual_pct"] for a in legacy],
        "all_bookkeeping_ok": bool(per_arm) and all(p["bookkeeping_ok"] for p in per_arm),
        "all_third_law_ok": bool(per_arm) and all(p["third_law_ok"] for p in per_arm),
        "all_under_owner_standard": bool(per_arm) and all(p["owner_standard"] for p in per_arm),
        "all_under_force_gate": bool(per_arm) and all(p["force_gate"] for p in per_arm),
    }


def payload(arms: list[dict]) -> dict:
    ratio = (cr.analytic_scalars(cr.HEIGHT, cr.WALL_SPEED, cr.VISCOSITY)["wall_shear"])
    return {
        "case": "plane_couette_channel",
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "generated_by": "benchmarks/couette/wall_force_reading.py",
        "quantity": (f"tangential force the fluid exerts on each wall, per unit depth, "
                     f"divided by the period length: |1 - tau_force/tau_exact| [%], "
                     f"tau_exact = rho nu U/H = {ratio:.3f} Pa"),
        "setup": {
            "height_m": cr.HEIGHT,
            "viscosity_m2_s": cr.VISCOSITY,
            "wall_speed_m_s": cr.WALL_SPEED,
            "sound_speed_m_s": cr.SOUND_SPEED,
            "wall_friction": cr.FRICTION,
            "columns": cr.COLUMNS,
            "wall_layers": cr.WALL_LAYERS,
            "viscous_times_by_arm": "per arm (1.2 primary grid, 3.6 ladder point)",
            "averaging_fraction": cr.AVERAGE_FRACTION,
            "banked_records": {"H_over_dx_8": 14.203286170959451,
                               "H_over_dx_16": 17.076468467712292},
        },
        "recipe": {
            "periodic3": ("wall lattice tiled in x with the same three non-overlapping "
                          "copies as the fluid cloud; loads summed over the "
                          f"x in [0,L) copy, time-averaged over the last "
                          f"{100 * cr.AVERAGE_FRACTION:.0f} % of the run"),
            "legacy": ("single wall band extended 3 dx past both x ends with the "
                       "x in [0,L) filter of diag_wall_force.py (reproduction only)"),
        },
        "verdict": verdict(arms),
        "arms": arms,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--vt", type=float, default=cr.VISCOUS_TIMES)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--layouts", nargs="+", default=["periodic3", "legacy"],
                    choices=["periodic3", "legacy"])
    ap.add_argument("--out", type=Path, default=ARTIFACT)
    ap.add_argument("--merge", action="store_true",
                    help="merge the new arms into an existing artifact instead of "
                         "overwriting it (one invocation per local time budget)")
    ap.add_argument("--check", action="store_true",
                    help="verify the committed artifact is up to date, run nothing heavy")
    args = ap.parse_args(argv)

    if args.check:
        if not args.out.exists():
            print(f"MISSING: {args.out}")
            return 1
        data = json.loads(args.out.read_text())
        ok = data["verdict"]["all_bookkeeping_ok"] and \
            data["verdict"]["all_under_owner_standard"]
        print("OK: artifact present, bookkeeping and 15 % standard",
              "hold" if ok else "DO NOT hold")
        return 0 if ok else 1

    OUT.mkdir(parents=True, exist_ok=True)
    arms = []
    for res in args.res:
        for layout in args.layouts:
            a = run_arm(res, args.vt, layout, args.threads)
            print(f"{layout:>10} H/dx={res:<3d} tau_top={a['tau_top_force_pa']:8.3f} "
                  f"({a['err_top_force_pct']:+7.2f} %) "
                  f"tau_bot={a['tau_bottom_force_pa']:8.3f} "
                  f"({a['err_bottom_force_pct']:+7.2f} %) "
                  f"3rd_law={a['third_law_residual_pct']:+7.2f} % "
                  f"copy_inv={a['copy_invariance_pct']:+6.2f} % "
                  f"secant_top={a['err_top_secant_pct']:+7.2f} % "
                  f"secant_bot={a['err_bottom_secant_pct']:+7.2f} % "
                  f"[{a['wall_clock_s']:.0f} s]")
            arms.append(a)
    if args.merge and args.out.exists():
        old = json.loads(args.out.read_text())["arms"]
        keys = {(a["layout"], a["resolution_H_over_dx"], a["viscous_times"]) for a in arms}
        arms = [a for a in old
                if (a["layout"], a["resolution_H_over_dx"], a["viscous_times"])
                not in keys] + arms
    data = payload(arms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {args.out}")
    print("verdict:", json.dumps(data["verdict"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())