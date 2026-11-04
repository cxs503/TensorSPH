"""Reproduction + evidence for the three non-droplet reds on TensorSPH main.

Run with::

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src python3 \
        benchmarks/main_red_clearance/repro.py

It regenerates every number quoted in the mission report
(``artifacts`` -> ``summary.json``, ``*.png``):

1. **lattice/wall clearance.**  The pre-fix lattice is rebuilt here with the old
   ``arange(dx/2, extent, dx)`` rule (``ceil`` cells) and compared with
   :func:`tensorsph.droplet_impact.cell_centres` (``floor`` cells) at the same
   spacings: column count, distance from the last fluid column to the wall
   layer, and the initial density error of the assembled state.
2. **Ritter front speed.**  The shallow-limit dam-break case used by
   ``tests/test_ritter.py`` is integrated at two resolutions; the report gets
   the front trajectory, the position error against ``Z = a + 2 c0 t``, the
   whole-record mean speed (the quantity the old assertion measured) and the
   developed-stage speed (:func:`tensorsph.ritter.front_speed`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorsph.dam_break import DamBreakConfig, run_case  # noqa: E402
from tensorsph.droplet_impact import (  # noqa: E402
    Config,
    cell_centres,
    make_particles,
    rhs,
    wall_lattice,
)
from tensorsph.ritter import front_speed, front_x, wave_speed  # noqa: E402

HERE = Path(__file__).resolve().parent
SPACINGS = (0.025, 0.02, 0.009, 0.006, 0.0037)


def old_lattice(extent: float, dx: float, device) -> torch.Tensor:
    """The pre-fix rule: ``arange(dx/2, extent, dx)`` (``ceil`` cells)."""
    return torch.arange(dx * 0.5, extent, dx, device=device)


def old_pool(cfg: Config, device):
    gx, gy = torch.meshgrid(old_lattice(cfg.pool_width, cfg.dx, device),
                            old_lattice(cfg.pool_depth, cfg.dx, device),
                            indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def initial_compression(cfg: Config, pos: torch.Tensor, device) -> float:
    """Max density of the assembled initial state over ``rho0``, one ``rhs``."""
    mass = torch.full((pos.shape[0],), cfg.rho0 * cfg.dx * cfg.dx, device=device)
    wall = wall_lattice(cfg, device)
    rho, _, _ = rhs(pos, torch.zeros_like(pos), mass, cfg, wall=wall)
    return float(rho.max()) / cfg.rho0 - 1.0


def lattice_study(device):
    rows = []
    for dx in SPACINGS:
        cfg = Config(dx=dx, device=str(device))
        pool_x = cell_centres(cfg.pool_width, dx, device)
        new = make_particles(cfg, device)[0]
        n_pool = pool_x.numel() * cell_centres(cfg.pool_depth, dx, device).numel()
        old = old_pool(cfg, device)
        wall = wall_lattice(cfg, device)
        # only the near-wall columns matter for the clearance, and a full
        # cdist at dx = 0.0037 would be 25 k x 1.8 k x 8 B of distances
        near = torch.tensor(0.9 * cfg.pool_width)
        gap_new = float(torch.cdist(new[:n_pool][new[:n_pool, 0] > near].double(),
                                    wall.double()).min())
        gap_old = float(torch.cdist(old[old[:, 0] > near].double(),
                                    wall.double()).min())
        rows.append({
            "dx": dx,
            "cells_old": int(old_lattice(cfg.pool_width, dx, device).numel()),
            "cells_new": int(pool_x.numel()),
            "ratio_W_over_dx": cfg.pool_width / dx,
            "xmax_old": float(old[:, 0].max()),
            "xmax_new": float(pool_x.max()),
            "gap_to_wall_old_over_dx": gap_old / dx,
            "gap_to_wall_new_over_dx": gap_new / dx,
            "compression_old": initial_compression(cfg, old, device),
            "compression_new": initial_compression(cfg, new, device),
            "gate": cfg.max_density_error,
        })
    return rows


def ritter_run(dx: float, window: float = 0.4):
    a, h0 = 1.0, 0.1
    c0 = wave_speed(h0)
    span = window * a / c0
    cfg = DamBreakConfig(dx=dx, column_width=a, column_height=h0,
                         pool_width=a + 2.0 * c0 * span + 6 * dx,
                         tank_height=1.5 * h0, sound_speed=20.0 * c0,
                         viscosity=0.0, hydrostatic_init=False, device="cpu",
                         output_every=20)
    cfg.dt = cfg.dt_from_courant()
    cfg.steps = int(math.ceil(span / cfg.dt)) + 1
    return run_case(cfg), c0


def ritter_study():
    out = {}
    for dx in (0.04, 0.02):
        run, c0 = ritter_run(dx)
        t, z = run.times, run.front
        naive = (z[-1] - z[0]) / (t[-1] - t[0])
        dev = front_speed(t, z)
        ref = front_x(t[-1], 0.1, dam_x=1.0)
        out[f"dx_{dx:g}"] = {
            "H_over_dx": 0.1 / dx,
            "steps": run.steps_taken,
            "particles": int(run.final_positions.shape[0]),
            "window": t[-1],
            "front_final": z[-1],
            "Z_rit_final": ref,
            "position_error": (z[-1] - ref) / ref,
            "mean_speed": naive,
            "mean_speed_ratio": naive / (2.0 * c0),
            "developed_speed": dev,
            "developed_speed_ratio": dev / (2.0 * c0),
            "mass_drift": run.mass_drift,
            "escaped": run.escaped,
            "times": list(map(float, t)),
            "front": list(map(float, z)),
        }
    return out


def figure_lattice(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    axes[0].plot([r["ratio_W_over_dx"] for r in rows],
                 [r["xmax_old"] + 0.5 * r["dx"] for r in rows], "o-",
                 label="pre-fix lattice (ceil cells)")
    axes[0].plot([r["ratio_W_over_dx"] for r in rows],
                 [r["xmax_new"] + 0.5 * r["dx"] for r in rows], "s-",
                 label="fixed lattice (floor cells)")
    axes[0].axhline(1.0, color="k", lw=1, ls="--", label="tank wall x = 1")
    axes[0].set_xlabel("pool_width / dx")
    axes[0].set_ylabel("outer edge of the lattice  [m]")
    axes[0].set_title("lattice overhang vs spacing")
    axes[0].legend(fontsize=7)
    axes[1].semilogy([r["dx"] for r in rows], [r["compression_old"] for r in rows],
                     "o-", label="pre-fix")
    axes[1].semilogy([r["dx"] for r in rows], [r["compression_new"] for r in rows],
                     "s-", label="fixed")
    axes[1].axhline(rows[0]["gate"], color="r", ls="--", label="gate 5 %")
    axes[1].set_xlabel("dx  [m]")
    axes[1].set_ylabel("initial max compression")
    axes[1].set_title("initial density error (one rhs)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure_ritter(data, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for key, style in (("dx_0.04", "o"), ("dx_0.02", "s")):
        d = data[key]
        t = d["times"]
        axes[0].plot(t, d["front"], style + "-", ms=3,
                     label=f"SPH dx={key.split('_')[1]} (H/dx={d['H_over_dx']:g})")
    t = data["dx_0.02"]["times"]
    axes[0].plot(t, [front_x(x, 0.1, dam_x=1.0) for x in t], "k--",
                 label="Ritter $Z = a + 2c_0 t$")
    axes[0].set_xlabel("t  [s]")
    axes[0].set_ylabel("leading-edge x  [m]")
    axes[0].set_title("front trajectory (release transient then linear stage)")
    axes[0].legend(fontsize=7)

    ratios = [data["dx_0.04"]["developed_speed_ratio"],
              data["dx_0.02"]["developed_speed_ratio"]]
    means = [data["dx_0.04"]["mean_speed_ratio"],
             data["dx_0.02"]["mean_speed_ratio"]]
    labels = [f"H/dx={data['dx_0.04']['H_over_dx']:g}",
              f"H/dx={data['dx_0.02']['H_over_dx']:g}"]
    idx = range(len(ratios))
    axes[1].bar([i - 0.18 for i in idx], means, width=0.34,
                label="mean over whole record")
    axes[1].bar([i + 0.18 for i in idx], ratios, width=0.34,
                label="developed stage (last 40 %)")
    axes[1].axhline(1.0, color="k", ls="--", label="$2c_0$ (analytic)")
    axes[1].axhline(1.25, color="r", ls=":", label="old assertion band $\\pm$25 %")
    axes[1].set_xticks(list(idx), labels)
    axes[1].set_ylabel("leading-edge speed / $2c_0$")
    axes[1].set_title("what the two speed measures see")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    device = torch.device("cpu")
    lattice = lattice_study(device)
    ritter = ritter_study()
    summary = {"lattice": lattice, "ritter": ritter}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    figure_lattice(lattice, HERE / "lattice_clearance.png")
    figure_ritter(ritter, HERE / "ritter_front_speed.png")
    for row in lattice:
        print(f"dx={row['dx']:<7} cells {row['cells_old']}->{row['cells_new']} "
              f"gap {row['gap_to_wall_old_over_dx']:.2f}->{row['gap_to_wall_new_over_dx']:.2f} dx "
              f"compression {100*row['compression_old']:6.2f}%->{100*row['compression_new']:6.2f}%")
    for key, d in ritter.items():
        print(f"{key}: position_error={d['position_error']:+.4f} "
              f"mean_ratio={d['mean_speed_ratio']:.4f} "
              f"developed_ratio={d['developed_speed_ratio']:.4f}")


if __name__ == "__main__":
    main()
