"""Field snapshot for the 3-D Kleefsman obstacle case (report figure only).

Runs the *same* solver stack as ``run_case`` (cubic-spline 3-D kernel,
``rhs_3d``, ``apply_boundary_3d``, ``clamp_out_of_obstacle``, Verlet pair
cache) for a short window and dumps the fluid particle positions + speeds so a
particle "field" plot can be made without touching the committed case module.

This is a diagnostic driver, not part of the acceptance protocol: it does not
score anything and writes only to ``results/snapshot/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.dam_break_3d import (  # noqa: E402
    apply_boundary_3d,
    make_column_3d,
    rhs_3d,
    wall_lattice_3d,
)
from tensorsph.kleefsman_obstacle_3d import (  # noqa: E402
    Kleefsman3DConfig,
    clamp_out_of_obstacle,
    obstacle_lattice_3d,
)
from tensorsph.neighbor_search import PairListCache  # noqa: E402

HERE = Path(__file__).resolve().parent


def run_snapshot(cfg: Kleefsman3DConfig, out_csv: Path,
                 t_snap: float, threads: int = 4) -> dict:
    torch.set_num_threads(threads)
    dev = torch.device("cpu")
    positions, mass = make_column_3d(cfg, dev)
    fluid_count = positions.shape[0]
    wall = wall_lattice_3d(cfg, dev)
    obstacle = obstacle_lattice_3d(cfg, dev).to(wall.dtype)
    wall = torch.cat((wall, obstacle), dim=0)

    clamp_out_of_obstacle(positions, torch.zeros_like(positions), cfg)
    velocity = torch.zeros_like(positions)
    all_positions = torch.cat((positions, wall), dim=0)
    cache = PairListCache(cutoff=2.0 * cfg.h_factor * cfg.dx,
                           skin=cfg.pair_skin_factor * cfg.h_factor * cfg.dx,
                           fluid_count=fluid_count)

    dt = cfg.dt
    elapsed = 0.0
    max_steps = int(round(t_snap / dt))
    for _step in range(max_steps):
        all_positions[:fluid_count] = positions
        pairs, _ = cache.get(all_positions)
        density, _, acc = rhs_3d(positions, velocity, mass, cfg, pairs=pairs,
                                 wall=wall, dt=dt)
        if not torch.isfinite(acc).all():
            raise FloatingPointError("non-finite acceleration")
        velocity = velocity + dt * acc
        positions = positions + dt * velocity
        apply_boundary_3d(positions, velocity, cfg,
                          top=cfg.tank_top() if hasattr(cfg, "tank_top") else None)
        clamp_out_of_obstacle(positions, velocity, cfg)
        elapsed += dt

    speed = torch.linalg.vector_norm(velocity, dim=-1)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as fh:
        fh.write("x_m,y_m,z_m,speed_m_s\n")
        for i in range(fluid_count):
            fh.write(f"{float(positions[i, 0]):.5f},{float(positions[i, 1]):.5f},"
                    f"{float(positions[i, 2]):.5f},{float(speed[i]):.5f}\n")
    return {"t_s": elapsed, "n_fluid": int(fluid_count),
            "max_speed": float(speed.max())}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.05)
    ap.add_argument("--t-snap", type=float, default=0.62)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "results" / "snapshot" / "particles.csv"))
    args = ap.parse_args()
    cfg = Kleefsman3DConfig(dx=args.dx, t_end=args.t_snap, wall_layers=3)
    cfg.dt = 0.9 * cfg.dt_from_courant()
    info = run_snapshot(cfg, Path(args.out), args.t_snap, args.threads)
    print(info)
