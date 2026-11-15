#!/usr/bin/env python3
"""Diagnostics for the Navier-slip wall layer: where does the wall coupling live?

Run one channel to steady state and dump, per row, the mean tangential velocity,
the *implied* wall velocity of each mirrored pair, and the extremes of the state
(particles outside the channel, largest particle speed and its location).  This
is the tool used to separate "profile wrong" from "one particle ran away".

Usage
-----
    PYTHONPATH=src python benchmarks/poiseuille_slip/diagnose_wall_layer.py \
        --resolution 8 --slip-ratio 0.25 --t-end 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from benchmarks.poiseuille.run import (
    bin_rows,
    build_fluid,
    make_config,
    periodic_images,
)
from benchmarks.poiseuille_slip.run import analytic_slip
from tensorsph.droplet_impact import resolve_device, rhs
from tensorsph.wall_mirror import plane_walls

REPO = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolution", type=int, default=8)
    p.add_argument("--slip-ratio", type=float, default=0.25)
    p.add_argument("--t-end", type=float, default=2.0)
    p.add_argument("--t-average", type=float, default=0.2)
    p.add_argument("--height", type=float, default=0.1)
    p.add_argument("--viscosity", type=float, default=0.01)
    p.add_argument("--body-force", type=float, default=9.81)
    p.add_argument("--columns", type=int, default=12)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path,
                   default=REPO / "benchmarks/poiseuille_slip/out/wall_layer.json")
    args = p.parse_args(argv)

    torch.set_num_threads(args.threads)
    device = resolve_device("cpu")
    H = args.height
    dx = H / args.resolution
    cfg = make_config(dx, args.viscosity, 1.0, device)
    length = args.columns * dx
    lam = args.slip_ratio * H

    pos, mass = build_fluid(cfg, H, length, device)
    n = pos.shape[0]
    vel = torch.zeros_like(pos)
    h = cfg.h_factor * dx
    support = 2.0 * h
    planes = [(0.0, +1), (H, -1)]
    dt = min(0.3 * h / cfg.sound_speed, 0.125 * h * h / cfg.viscosity)
    steps = int(round(args.t_end / dt))
    avg_from = args.t_end - args.t_average
    acc = torch.zeros_like(pos)
    samples = 0
    top = []
    for step in range(steps):
        pos_ext = periodic_images(pos, length)
        vel_ext = torch.cat((vel, vel, vel), dim=0)
        mass_ext = torch.cat((mass, mass, mass), dim=0)
        gp, gv, _ = plane_walls(pos_ext, vel_ext, mass_ext, planes, support,
                                slip="navier" if lam > 0 else "no_slip",
                                slip_length=lam if lam > 0 else None)
        _, _, a = rhs(pos_ext, vel_ext, mass_ext, cfg, wall=gp, wall_vel=gv)
        a = a[:n].clone()
        a[:, 0] += args.body_force
        t = step * dt
        if t >= avg_from:
            acc += vel
            samples += 1
        vel = vel + dt * a
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], length)
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            sp = vel.norm(dim=1)
            k = int(sp.argmax())
            top.append({"t": t, "max_speed": float(sp[k]),
                        "y_at_max": float(pos[k, 1]),
                        "u_max": float(vel[:, 0].max()),
                        "u_min": float(vel[:, 0].min()),
                        "y_min": float(pos[:, 1].min()),
                        "y_max": float(pos[:, 1].max())})

    u_mean = acc / max(samples, 1)
    _, profile = bin_rows(pos, u_mean, H, dx)
    for row in profile:
        row["u_analytic"] = analytic_slip(row["y"], H, args.body_force,
                                          args.viscosity, lam)

    outside = int(((pos[:, 1] < 0.0) | (pos[:, 1] > H)).sum())
    sp = vel.norm(dim=1)
    k = int(sp.argmax())
    payload = {
        "resolution_H_over_dx": args.resolution,
        "slip_ratio_lam_over_H": args.slip_ratio,
        "slip_over_dx": lam / dx,
        "t_end": args.t_end,
        "dt": dt,
        "steps": steps,
        "n_fluid": int(n),
        "n_outside_channel": outside,
        "final_max_speed": float(sp[k]),
        "final_y_at_max_speed": float(pos[k, 1]),
        "final_rows_outside": [float(v) for v in
                               pos[(pos[:, 1] < 0) | (pos[:, 1] > H), 1][:10]],
        "profile": profile,
        "trace": top,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))

    print(f"H/dx={args.resolution} lam/H={args.slip_ratio:g} "
          f"(lam/dx={lam / dx:.2f})  outside={outside}  "
          f"max_speed={payload['final_max_speed']:.3f} at y="
          f"{payload['final_y_at_max_speed']:.4f}")
    print(" t      max|v|  y@max   u_max   ymin    ymax")
    for r in top:
        print(f" {r['t']:.2f}  {r['max_speed']:7.3f} {r['y_at_max']:.4f} "
              f"{r['u_max']:7.3f} {r['y_min']:.4f} {r['y_max']:.4f}")
    print(" y/H    u_mean   u_an    rel     u_std")
    for row in profile:
        rel = ((row["u_mean"] - row["u_analytic"]) / row["u_analytic"] * 100
               if row["u_analytic"] else float("nan"))
        print(f" {row['y_over_H']:.3f} {row['u_mean']:8.4f} "
              f"{row['u_analytic']:8.4f} {rel:+7.2f}% {row['u_std_x']:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())