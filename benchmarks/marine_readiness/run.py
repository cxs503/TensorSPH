"""Compute 3-D still-water wall loads and compare ice-wave speeds with HSVA."""
import argparse
import csv
from bisect import bisect_left
import json
import math
from pathlib import Path

import torch

from tensorsph.dam_break_3d import DamBreak3DConfig, make_column_3d, rhs_3d, wall_lattice_3d
from tensorsph.kleefsman_obstacle_3d import (REFERENCE_ARRIVAL_TIME_S, REFERENCE_PEAK_HEIGHT_M,
                                          REFERENCE_PEAK_TIME_S, arrival_time)
from tensorsph.neighbor_search import PairListCache
from tensorsph.wave_ice_interaction import wave_number_under_ice


def water_load(n, courant=.20, end_s=.06):
    """Fixed 0.32 x 0.16 x 0.16 m tank of water, gravity in -y, atmospheric top."""
    depth, width = .16, .32
    dx = depth/n
    cfg = DamBreak3DConfig(dx=dx, pool_width=width, tank_depth=depth, tank_height=.24,
                           column_width=width, column_depth=depth, column_height=depth,
                           sound_speed=24., wall_layers=3, viscosity=1e-6,
                           density_diffusion=0., wall_friction=0., surface_pressure_closure=True)
    xyz, mass = make_column_3d(cfg, torch.device("cpu"))
    wall = wall_lattice_3d(cfg, torch.device("cpu"))
    vel = torch.zeros_like(xyz)
    cache = PairListCache(2*cfg.h_factor*dx, .3*dx, fluid_count=len(xyz))
    count = math.ceil(end_s/(courant*cfg.h_factor*dx/cfg.sound_speed))
    dt = end_s/count
    weight = float(mass.sum())*cfg.gravity
    ref_end_load = .5*cfg.rho0*cfg.gravity*depth**3
    right = (wall[:, 0] >= width) & (wall[:, 1] >= 0)
    sum_support = sum_end_load = 0.
    impulse = torch.zeros(3, dtype=xyz.dtype)
    max_balance = max_speed = 0.
    for step in range(count):
        pairs, _ = cache.get(torch.cat((xyz, wall)))
        _, _, acc, load = rhs_3d(xyz, vel, mass, cfg, pairs=pairs, wall=wall, dt=dt,
                                 return_wall_force=True)
        force = load.sum(0)
        sum_support -= float(force[1])
        sum_end_load += float(load[right, 0].sum())
        impulse -= dt*force
        impulse[1] -= dt*weight
        vel += dt*acc
        xyz += dt*vel
        max_speed = max(max_speed, float(vel.norm(dim=1).max()))
        balance = (mass[:, None]*vel).sum(0)-impulse
        max_balance = max(max_balance, float(balance.norm())/(weight*end_s))
    return dict(depth_cells=n, dx_m=dx, end_s=end_s, dt_s=dt, steps=count, particles=len(xyz),
                weight_N=weight, mean_support_N=sum_support/count,
                vertical_error_pct=100*(sum_support/count/weight-1),
                right_wall_reference_N=ref_end_load, mean_right_wall_N=sum_end_load/count,
                right_wall_error_pct=100*(sum_end_load/count/ref_end_load-1),
                max_momentum_residual_over_weight_impulse=max_balance, max_speed_m_s=max_speed)


def hsva():
    # Marchenko et al., Water 13 (2021) 1080, Tables 2 and 4, stationary sheets.
    # Young's moduli are from independent quasi-static loading, not dispersion fits.
    cases = (("TG1", .7, 46., 3.), ("TG2a", 1., 107., 3.5),
             ("TG3", .8, 371., 4.), ("TG3", 1., 371., 4.3))
    rows = []
    for label, f, modulus, measured in cases:
        w = 2*math.pi*f
        k = wave_number_under_ice(w, .05, 2.5, E=modulus*1e6)
        rows.append(dict(group=label, frequency_Hz=f, modulus_MPa=modulus,
                         measured_m_s=measured, predicted_m_s=w/k,
                         error_pct=100*(w/k/measured-1)))
    return rows


REFERENCE = (Path(__file__).resolve().parents[2] /
             "docs/m20884/outputs/kleefsman_h2_ref/h2_fig19.csv")


def read_trace(path, column):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    t = [float(row["t_s"]) for row in rows]
    y = [float(row[column]) for row in rows]
    if len(t) < 2 or not all(math.isfinite(v) for v in t+y) or any(
            b <= a for a, b in zip(t, t[1:])):
        raise ValueError(f"invalid time series in {path}")
    return t, y


def impact_score(path):
    t, h = read_trace(path, "slab_height_m")
    ref_t, ref_h = read_trace(REFERENCE, "h_m")
    residual = []
    for time, observed in zip(ref_t, ref_h):
        i = bisect_left(t, time)
        if i == 0 or i == len(t):
            continue
        w = (time-t[i-1])/(t[i]-t[i-1])
        residual.append(h[i-1]*(1-w) + h[i]*w - observed)
    arrival = arrival_time(t, h, .02)
    peak = max(h) if t[-1] >= REFERENCE_PEAK_TIME_S else None
    rmse = math.sqrt(sum(v*v for v in residual)/len(residual)) if residual else None
    record = dict(trace=path.name, start_s=t[0], end_s=t[-1], samples=len(t),
                event_window_peak_m=peak, peak_t_s=t[h.index(peak)] if peak is not None else None,
                peak_error_pct=100*(peak/REFERENCE_PEAK_HEIGHT_M-1) if peak is not None else None,
                arrival_s=arrival, experimental_first_wet_sample_s=REFERENCE_ARRIVAL_TIME_S,
                arrival_error_pct=100*(arrival/REFERENCE_ARRIVAL_TIME_S-1) if arrival else None,
                experimental_interpolated_arrival_s=arrival_time(ref_t, ref_h, .02),
                n_experimental_samples=len(residual),
                trace_rmse_m=rmse)
    summary = path.with_name(path.stem.removesuffix("_time_series") + ".json")
    if summary.exists():
        run = json.loads(summary.read_text())["run"]
        record.update(max_mach=run["max_speed"]/run["sound_speed"],
                      peak_density_compression=run["max_density_compression"],
                      obstacle_clamps=run["obstacle_clamps"], steps=run["steps"],
                      mass_error_rel=run["mass_error_rel"])
    return record


def plot_impact(paths, outfile):
    from matplotlib import pyplot as plt
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True, layout="constrained")
    t, h = read_trace(REFERENCE, "h_m")
    top.plot(t, h, color="black", lw=1.2, label="MARIN H2 (digitised)")
    duration = 0.
    for path in paths:
        t, h = read_trace(path, "slab_height_m")
        _, force = read_trace(path, "Fx_N")
        tag = path.stem.removesuffix("_time_series")
        metadata = path.with_name(tag + ".json")
        label = tag
        if metadata.exists():
            summary = json.loads(metadata.read_text())
            run = summary["run"]
            label = f"{1000*run['dx']:.0f} mm, c₀={run['sound_speed']:g} m/s"
            nu = run["fingerprint"].get("viscosity", summary.get("kinematic_viscosity_m2_s"))
            if nu is not None:
                label += f", ν={nu:g} m²/s"
        line, = top.plot(t, h, lw=1.1, label=label)
        bottom.plot(t, force, lw=.9, color=line.get_color())
        duration = max(duration, t[-1])
    top.set(ylabel="H2 water height (m)")
    top.legend()
    bottom.axhline(0., color="grey", lw=.5)
    bottom.set(xlabel="Time since release (s)", ylabel="Box SPH force, x (N)", xlim=(0, duration))
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--impact", nargs="+", type=Path, default=[])
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    data = dict(steady_water=[water_load(n) for n in (4, 6, 8)],
                half_step=water_load(6, .10), hsva=hsva(),
                hsva_reference="10.3390/w13081080, Tables 2 and 4")
    if args.impact:
        data["green_water"] = [impact_score(path) for path in args.impact]
    if args.plot:
        if not args.impact:
            parser.error("--plot requires --impact")
        plot_impact(args.impact, args.plot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
