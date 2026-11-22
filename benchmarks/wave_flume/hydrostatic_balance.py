"""Measure the initial pressure/gravity residual on the production flume lattice."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from tensorsph.droplet_impact import rhs
from tensorsph.wave_flume import (
    WaveFlumeConfig,
    flume_density_offset,
    flume_particle_mass,
    flume_water_lattice,
    paddle_lattice,
    static_wall_lattice,
)


def audit(cells, **options):
    dx = 0.25 / cells
    cfg = WaveFlumeConfig(
        dx=dx,
        depth=0.25,
        wave_height=0.04,
        flume_length=round(3.2 / dx) * dx,
        wall_layers=3,
        **options,
    )
    cfg = replace(cfg, dt=0.25 * cfg.h_factor * dx / cfg.sound_speed)
    pos = flume_water_lattice(cfg, "cpu")
    mass = flume_particle_mass(cfg, pos, "cpu")
    fixed = static_wall_lattice(cfg, "cpu")
    wall = torch.cat((fixed, paddle_lattice(cfg, "cpu")))
    offset = flume_density_offset(cfg, pos, mass, wall)
    rho, p, a, load = rhs(
        pos,
        torch.zeros_like(pos),
        mass,
        cfg,
        wall=wall,
        dt=cfg.dt,
        split=True,
        density_offset=offset,
    )
    depth = cfg.depth - pos[:, 1]
    ph = cfg.rho0 * cfg.gravity * depth
    head = cfg.rho0 * cfg.gravity * cfg.depth
    reaction = load["reaction"]
    weight = mass.sum() * cfg.gravity
    vertical_pressure = (mass * (a[:, 1] + cfg.gravity)).sum()
    bands = {}
    side = (pos[:, 0] < 2 * cfg.h_factor * dx) | (
        cfg.flume_length - pos[:, 0] < 2 * cfg.h_factor * dx
    )
    bottom = pos[:, 1] < 2 * cfg.h_factor * dx
    surface = depth < 2 * cfg.h_factor * dx
    for name, mask in dict(
        all=torch.ones_like(side),
        interior=~(side | bottom | surface),
        surface=surface & ~side,
        bottom=bottom & ~side,
        sides=side,
    ).items():
        errors = (p[mask] - ph[mask]) / head
        acceleration = a[mask] / cfg.gravity
        bands[name] = dict(
            particles=int(mask.sum()),
            pressure_rmse_over_head=float(errors.square().mean().sqrt()),
            residual_rms_g=float(acceleration.square().sum(1).mean().sqrt()),
            residual_max_g=float(acceleration.norm(dim=1).max()),
            worst_residual_xy_m=pos[mask][acceleration.norm(dim=1).argmax()].tolist(),
            horizontal_max_g=float(acceleration[:, 0].abs().max()),
            mean_vertical_residual_g=float(acceleration[:, 1].mean()),
        )
    return dict(
        dx_m=dx,
        dt_s=cfg.dt,
        water_weight_N_per_m=float(weight),
        fluid_vertical_pressure_force_over_weight=float(vertical_pressure / weight),
        vertical_wall_load_over_weight=float(-reaction[:, 1].sum() / weight),
        pressure_momentum_closure_over_weight=float(
            (vertical_pressure + reaction[:, 1].sum()).abs() / weight
        ),
        min_density_over_reference=float(rho.min() / cfg.rho0),
        bands=bands,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    torch.set_num_threads(1)
    modes = {
        "production": {},
        "wall_consistent": {"boundary_consistent_density": True},
        "reference_increment": {"reference_density_increment": True},
        "increment_no_diffusion": {"reference_density_increment": True, "density_diffusion": 0.0},
    }
    results = {str(n): {name: audit(n, **opts) for name, opts in modes.items()} for n in (16, 20)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
