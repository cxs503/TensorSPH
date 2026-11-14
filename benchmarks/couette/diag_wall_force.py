#!/usr/bin/env python3
"""Diagnostic 3: measure the wall shear the way it is *defined* -- the tangential
force the fluid exerts on each wall (Newton) -- instead of the one-spacing secant.

If the moving-wall no-slip condition were really 17 % deficient, the force on the
sliding wall would be 17 % below rho*nu*U/H, and the fixed wall would have to
carry the difference (momentum balance in the steady state).  If instead both
walls carry ~rho*nu*U/H, the profile secant is the biased estimator.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
RUN = Path(__file__).resolve().parent / "run.py"
spec = importlib.util.spec_from_file_location("couette_run", RUN)
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def run(res, vt, threads=1, images=True):
    torch.set_num_threads(threads)
    dev = torch.device("cpu")
    dx = cr.HEIGHT / res
    length = cr.COLUMNS * dx
    cfg = cr.make_config(dx, cr.VISCOSITY, cr.FRICTION, dev)
    pos, mass = cr.build_fluid(cfg, cr.HEIGHT, length, dev)
    wall, wall_vel = cr.build_walls(cfg, cr.HEIGHT, length, cfg.wall_layers,
                                    cr.WALL_SPEED, dev)
    n = pos.shape[0]
    vel = torch.zeros_like(pos)
    h = cfg.h_factor * dx
    dt = min(0.3 * h / cfg.sound_speed, 0.125 * h * h / cr.VISCOSITY)
    t_end = vt * cr.HEIGHT ** 2 / cr.VISCOSITY
    avg_from = t_end - cr.AVERAGE_FRACTION * t_end
    steps = int(round(t_end / dt))

    top = wall[:, 1] > cr.HEIGHT / 2
    bot = ~top
    inx = (wall[:, 0] > 0.0) & (wall[:, 0] < length)   # only the fluid footprint
    acc_sum = torch.zeros_like(pos)
    net_fluid_force = 0.0
    mom0 = None
    mom1 = None
    samples = 0
    fx_top = 0.0
    fx_bot = 0.0
    fx_top_all = 0.0
    fx_bot_all = 0.0
    fx_gain = 0.0
    for step in range(steps):
        if images:
            pos_ext = cr.periodic_images(pos, length)
            vel_ext = torch.cat((vel, vel, vel), dim=0)
            mass_ext = torch.cat((mass, mass, mass), dim=0)
        else:
            pos_ext, vel_ext, mass_ext = pos, vel, mass
        _, _, acc, loads = cr.rhs(pos_ext, vel_ext, mass_ext, cfg, wall=wall,
                                  wall_vel=wall_vel, split=True)
        acc = acc[:n].clone()
        t = step * dt
        if t >= avg_from:
            acc_sum += vel
            samples += 1
            rx = loads["reaction"][:, 0]
            fx_top += float(rx[top & inx].sum()) * dt
            fx_bot += float(rx[bot & inx].sum()) * dt
            net_fluid_force += float((mass * acc[:, 0]).sum()) * dt
            fx_gain += float((mass * acc[:, 0]).sum()) * dt
            mom0 = float((mass * vel[:, 0]).sum()) if mom0 is None else mom0
            mom1 = float((mass * vel[:, 0]).sum())
            fx_top_all += float(rx[top].sum()) * dt
            fx_bot_all += float(rx[bot].sum()) * dt
        vel = vel + dt * acc
        pos = pos + dt * vel
        pos[:, 0] = torch.remainder(pos[:, 0], length)

    # time average of the instantaneous force  (sum F dt / T)
    T = samples * dt
    tau_top = fx_top / (T * length)
    tau_bot = fx_bot / (T * length)
    u_mean = acc_sum / max(samples, 1)
    rows, prof = cr.bin_rows(pos, u_mean, cr.HEIGHT, dx)
    m = cr.profile_metrics(prof, cr.HEIGHT, cr.WALL_SPEED, cr.VISCOSITY)
    tau_ex = cr.analytic_scalars(cr.HEIGHT, cr.WALL_SPEED, cr.VISCOSITY)["wall_shear"]
    return {
        "res": res, "vt": vt, "steps": steps, "n": n, "images": images,
        "tau_exact": tau_ex,
        "tau_bottom_force": tau_bot, "tau_top_force": tau_top,
        "tau_bottom_all": fx_bot_all / (T * length),
        "tau_top_all": fx_top_all / (T * length),
        "third_law_residual": (fx_gain + fx_top_all + fx_bot_all) / (T * length),
        "net_force_on_fluid": -(tau_top + tau_bot),
        "net_force_on_fluid_all": -(fx_top_all + fx_bot_all) / (T * length),
        "err_bottom_force_pct": 100 * (tau_bot / tau_ex - 1),
        "err_top_force_pct": 100 * (tau_top / tau_ex - 1),
        "err_bottom_secant_pct": 100 * (m["wall_shear_ratio"] - 1),
        "err_top_secant_pct": 100 * (m["wall_shear_ratio_top"] - 1),
        "interior_l2_pct": 100 * m["interior_l2"],
        "flux_ratio": m["flux_ratio"],
        "mean_fluid_net_force": net_fluid_force / (T * length),
        "fluid_momentum_rate": (mom1 - mom0) / T,
        "increments": [round(float(prof[k + 1]["u_mean"] - prof[k]["u_mean"])
                             / cr.WALL_SPEED * res, 4)
                       for k in range(len(prof) - 1)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--vt", type=float, default=1.2)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    print(f"{'res':>4} {'steps':>6} {'tau_ex':>8} {'tau_bot':>9} {'tau_top':>9} "
          f"{'err_bot%':>9} {'err_top%':>9} {'sec_bot%':>9} {'sec_top%':>9} "
          f"{'l2%':>6} {'Q':>7}")
    for r in args.res:
        o = run(r, args.vt, args.threads, images=not args.no_images)
        print(f"{o['res']:>4} {o['steps']:>6} {o['tau_exact']:>8.2f} "
              f"{o['tau_bottom_force']:>9.2f} {o['tau_top_force']:>9.2f} "
              f"{o['err_bottom_force_pct']:>9.2f} {o['err_top_force_pct']:>9.2f} "
              f"{o['err_bottom_secant_pct']:>9.2f} {o['err_top_secant_pct']:>9.2f} "
              f"{o['interior_l2_pct']:>6.2f} {o['flux_ratio']:>7.4f}")
        print(f"       gradient/analytic per row spacing: {o['increments']}")
        print(f"       all-wall incl. x-margin: bot={o['tau_bottom_all']:.2f} "
              f"top={o['tau_top_all']:.2f}  net on fluid(in)= {o['net_force_on_fluid']:.2f} "
              f"vs all={o['net_force_on_fluid_all']:.2f}")
        print(f"       fluid side: mean sum(m*a_x)={o['mean_fluid_net_force']:.2f} Pa  "
              f"d(momentum)/dt={o['fluid_momentum_rate']:.4f} kg/s")


if __name__ == "__main__":
    main()