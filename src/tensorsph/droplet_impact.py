"""Small, transparent 2-D WCSPH droplet-impact prototype (PyTorch)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math

import torch

from .neighbor_search import cell_linked_pairs


@dataclass
class Config:
    dx: float = 0.025
    h_factor: float = 1.3
    rho0: float = 1000.0
    sound_speed: float = 20.0
    gamma: float = 7.0
    gravity: float = 9.81
    viscosity: float = 0.01
    dt: float = 2.0e-5
    steps: int = 500
    output_every: int = 25
    pool_width: float = 1.0
    pool_depth: float = 0.35
    drop_radius: float = 0.10
    drop_center_y: float = 0.49
    device: str = "auto"
    seed: int = 7


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def cubic_spline_kernel(r: torch.Tensor, h: float) -> torch.Tensor:
    """2-D cubic spline kernel W(r,h), normalized by 10/(7*pi*h**2)."""
    q = r / h
    sigma = 10.0 / (7.0 * math.pi * h * h)
    first = 1.0 - 1.5 * q.square() + 0.75 * q.pow(3)
    second = 0.25 * (2.0 - q).clamp_min(0).pow(3)
    shape = torch.where(q < 1.0, first, torch.where(q < 2.0, second, 0.0))
    return sigma * shape.clamp_min(0.0)


def cubic_spline_gradient(
    delta: torch.Tensor, h: float, eps: float = 1.0e-12
) -> torch.Tensor:
    """Gradient with respect to x_i for delta_ij = x_i - x_j."""
    r = torch.linalg.vector_norm(delta, dim=-1)
    q = r / h
    sigma = 10.0 / (7.0 * math.pi * h * h)
    dshape = torch.where(
        q < 1.0,
        -3.0 * q + 2.25 * q.square(),
        torch.where(q < 2.0, -0.75 * (2.0 - q).square(), 0.0),
    )
    scale = sigma * dshape / (h * r.clamp_min(eps))
    return delta * scale.unsqueeze(-1)


def make_particles(cfg: Config, device: torch.device):
    """Generate a rectangular pool and a circular drop on a regular lattice."""
    dx = cfg.dx
    xs = torch.arange(dx * 0.5, cfg.pool_width, dx, device=device)
    ys = torch.arange(dx * 0.5, cfg.pool_depth, dx, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pool = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)

    drop_x = torch.arange(dx * 0.5, cfg.pool_width, dx, device=device)
    drop_y = torch.arange(cfg.drop_center_y - cfg.drop_radius,
                          cfg.drop_center_y + cfg.drop_radius, dx, device=device)
    dxg, dyg = torch.meshgrid(drop_x, drop_y, indexing="xy")
    candidate = torch.stack((dxg.reshape(-1), dyg.reshape(-1)), dim=-1)
    center = torch.tensor([cfg.pool_width * 0.5, cfg.drop_center_y], device=device)
    drop = candidate[torch.linalg.vector_norm(candidate - center, dim=-1) <= cfg.drop_radius]
    positions = torch.cat((pool, drop), dim=0)
    fixed = torch.zeros(positions.shape[0], dtype=torch.bool, device=device)
    mass = torch.full((positions.shape[0],), cfg.rho0 * dx * dx, device=device)
    return positions, fixed, mass


def density_from_neighbor_pairs(pos, mass, h: float, pairs=None):
    """Compute the SPH density sum from exact compact-support particle pairs.

    The self contribution m_i W(0) is included explicitly. Each unordered
    pair contributes to both endpoint particles, matching the all-pairs sum.
    Pair construction currently bins coordinates on the CPU; this reduces
    density storage but is not yet a GPU-resident end-to-end acceleration.
    """
    if pos.ndim != 2 or mass.ndim != 1 or mass.shape[0] != pos.shape[0]:
        raise ValueError("pos must be (N, D) and mass must be (N,)")
    if not pos.is_floating_point() or not mass.is_floating_point():
        raise TypeError("pos and mass must be floating-point tensors")
    if pos.device != mass.device:
        raise ValueError("pos and mass must be on the same device")
    if not math.isfinite(float(h)) or h <= 0:
        raise ValueError("h must be finite and positive")
    if pairs is None:
        pairs = cell_linked_pairs(pos, 2.0 * h)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
        raise ValueError("pairs must be a long tensor with shape (pair_count, 2)")
    if pairs.device != pos.device:
        raise ValueError("pairs and pos must be on the same device")

    n = pos.shape[0]
    rho = mass * cubic_spline_kernel(
        torch.zeros(n, dtype=pos.dtype, device=pos.device), h
    )
    if pairs.numel() == 0:
        return rho
    i, j = pairs.unbind(dim=1)
    rij = torch.linalg.vector_norm(pos[i] - pos[j], dim=1)
    wij = cubic_spline_kernel(rij, h)
    rho = rho.index_add(0, i, mass[j] * wij)
    rho = rho.index_add(0, j, mass[i] * wij)
    return rho


def rhs(pos, vel, mass, cfg: Config):
    """Compute density, Tait EOS pressure, pressure force and viscous acceleration."""
    h = cfg.h_factor * cfg.dx
    delta = pos[:, None, :] - pos[None, :, :]
    r = torch.linalg.vector_norm(delta, dim=-1)
    rho = density_from_neighbor_pairs(pos, mass, h).clamp_min(0.1 * cfg.rho0)
    pressure = (cfg.rho0 * cfg.sound_speed**2 / cfg.gamma) * (
        (rho / cfg.rho0).clamp_min(0.01).pow(cfg.gamma) - 1.0
    )
    grad_w = cubic_spline_gradient(delta, h)
    pressure_term = (
        pressure[:, None] / rho[:, None].square()
        + pressure[None, :] / rho[None, :].square()
    )
    acc_pressure = -(mass[None, :, None] * pressure_term[:, :, None] * grad_w).sum(dim=1)

    # Symmetric, regularized laminar-viscosity approximation.
    dv = vel[:, None, :] - vel[None, :, :]
    denom = r.square() + 0.01 * h * h
    laplace_weight = torch.relu(-(delta * grad_w).sum(dim=-1)) / denom
    mu_weight = (mass[None, :] / rho[None, :]) * laplace_weight
    acc_visc = -cfg.viscosity * (dv * mu_weight[:, :, None]).sum(dim=1)
    gravity = torch.zeros_like(pos)
    gravity[:, 1] = -cfg.gravity
    return rho, pressure, acc_pressure + acc_visc + gravity


def run(cfg: Config, output: str | Path):
    if cfg.dx <= 0 or cfg.dt <= 0 or cfg.h_factor <= 0:
        raise ValueError("dx, dt, and h_factor must be positive")
    if cfg.rho0 <= 0 or cfg.sound_speed <= 0 or cfg.gamma <= 0:
        raise ValueError("rho0, sound_speed, and gamma must be positive")
    if cfg.steps < 0 or cfg.output_every <= 0:
        raise ValueError("steps must be non-negative and output_every must be positive")
    if cfg.pool_width <= 0 or cfg.pool_depth <= 0 or cfg.drop_radius <= 0:
        raise ValueError("pool dimensions and drop_radius must be positive")

    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    pos, fixed, mass = make_particles(cfg, device)
    vel = torch.zeros_like(pos)
    initial_n = pos.shape[0]
    rows = []
    snapshots = []

    for step in range(cfg.steps + 1):
        rho, pressure, acc = rhs(pos, vel, mass, cfg)
        if not (torch.isfinite(pos).all() and torch.isfinite(vel).all()
                and torch.isfinite(rho).all() and torch.isfinite(acc).all()):
            raise FloatingPointError(f"non-finite state at step {step}")
        if step % cfg.output_every == 0 or step == cfg.steps:
            kinetic = 0.5 * (mass[:, None] * vel.square()).sum().item()
            speed = torch.linalg.vector_norm(vel, dim=-1)
            accel = torch.linalg.vector_norm(acc, dim=-1)
            total_mass = mass.sum().item()
            momentum = (mass[:, None] * vel).sum(dim=0)
            max_speed = speed.max().item()
            max_acceleration = accel.max().item()
            courant_number = cfg.dt * (cfg.sound_speed + max_speed) / (cfg.h_factor * cfg.dx)
            rows.append({
                "step": step, "time": step * cfg.dt,
                "particle_count": int(pos.shape[0]),
                "mass_total": total_mass,
                "momentum_x": momentum[0].item(), "momentum_y": momentum[1].item(),
                "rho_min": rho.min().item(), "rho_max": rho.max().item(),
                "rho_mean": rho.mean().item(),
                "rho_max_relative_error": (rho / cfg.rho0 - 1.0).abs().max().item(),
                "pressure_min": pressure.min().item(), "pressure_max": pressure.max().item(),
                "max_speed": max_speed, "max_acceleration": max_acceleration,
                "courant_number": courant_number,
                "kinetic_energy": kinetic,
                "y_min": pos[:, 1].min().item(), "y_max": pos[:, 1].max().item(),
            })
            snapshots.append((step, pos.detach().cpu().numpy().copy(),
                              rho.detach().cpu().numpy().copy()))
        if step == cfg.steps:
            break
        # Semi-implicit Euler. Fixed wall particles are added in later milestone;
        # this prototype uses a simple domain clamp as a transparent baseline.
        vel = vel + cfg.dt * acc
        pos = pos + cfg.dt * vel
        pos[:, 0].clamp_(min=0.5 * cfg.dx, max=cfg.pool_width - 0.5 * cfg.dx)
        pos[:, 1].clamp_(min=0.5 * cfg.dx, max=cfg.pool_depth + 2.0 * cfg.drop_radius)

    with (out / "diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for step, xyz, density in snapshots:
        torch.save({"step": step, "time": step * cfg.dt,
                    "position": torch.from_numpy(xyz),
                    "density": torch.from_numpy(density)},
                   out / f"particles_{step:06d}.pt")
    metadata = {
        "case": "droplet_impact_2d_wcsph",
        "status": "prototype_not_physically_validated",
        "device": str(device), "initial_particle_count": int(initial_n),
        "config": cfg.__dict__,
        "notes": [
            "All-pairs interactions are intended only for small prototype cases.",
            "No surface tension model is included yet.",
            "Wall handling and density consistency require benchmark validation.",
            "Execution success is not evidence of physical accuracy."
        ],
    }
    (out / "run_manifest.json").write_text(json.dumps(metadata, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4),
                                 squeeze=False)
        for ax, (step, xyz, density) in zip(axes[0], snapshots):
            sc = ax.scatter(xyz[:, 0], xyz[:, 1], c=density, s=5, cmap="viridis")
            ax.set_title(f"t={step * cfg.dt:.4f} s")
            ax.set_xlim(0, cfg.pool_width)
            ax.set_ylim(0, cfg.pool_depth + 2 * cfg.drop_radius)
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
        fig.colorbar(sc, ax=axes.ravel().tolist(), label="density (kg/m³)")
        fig.savefig(out / "particle_snapshots.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/droplet_impact_2d")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = Config(device=args.device)
    if args.steps is not None:
        config.steps = args.steps
    run(config, args.output)
