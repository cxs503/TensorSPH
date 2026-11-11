#!/usr/bin/env python3
"""Auditor field map: pressure deviation p - rho0 g (H - y) over the pool at t = 0.

Turns the delivered table numbers (bottom row deficit, free surface band) into a
2-D field map so the defect location is visible rather than asserted. Runs the
production RHS of the audited worktree; writes one PNG with the shipped and the
well-balanced mode side by side.

Run (from the audited worktree):  PYTHONPATH=src:benchmarks/lake_at_rest \
    python3 <this file> --out <png>
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorsph import droplet_impact as di  # noqa: E402
from run import build_state  # noqa: E402


def field(mode: str, dx: float = 0.02):
    torch.set_num_threads(1)
    cfg = di.Config(dx=dx, h_factor=1.3, hydrostatic_init=True,
                    density_calibration=True)
    if mode == "wb":
        cfg.equilibrium_level = cfg.pool_depth
    pos, mass, wall = build_state(cfg, torch.device("cpu"))
    _rho, p, acc = di.rhs(pos, torch.zeros_like(pos), mass, cfg, wall=wall,
                          dt=cfg.dt)
    pex = cfg.rho0 * cfg.gravity * (cfg.pool_depth - pos[:, 1])
    dev = 100.0 * (p - pex) / (cfg.rho0 * cfg.gravity * cfg.pool_depth)
    return pos, dev, cfg, float(acc.abs().max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True)
    lim = 0.0
    data = {}
    for ax, mode in zip(axes, ("shipped", "wb")):
        pos, dev, cfg, amax = field(mode)
        data[mode] = (pos, dev, cfg, amax)
        lim = max(lim, float(dev.abs().max()))
    for ax, mode in zip(axes, ("shipped", "wb")):
        pos, dev, cfg, amax = data[mode]
        sc = ax.scatter(1000 * pos[:, 0], 1000 * pos[:, 1], c=dev, s=14,
                        cmap="RdBu_r", vmin=-lim, vmax=lim, marker="s")
        ax.set_ylabel("y [mm]")
        ax.set_title("%s: p - rho0 g (H-y) at t=0  [%% of rho0 g H],  max|a| = %.2f m/s2"
                     % (mode, amax), fontsize=9)
        ax.set_aspect("equal")
    axes[-1].set_xlabel("x [mm]")
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("pressure deviation [%]")
    fig.suptitle("CASE-LAKE-AT-REST-SPH, dx = 20 mm, 850 particles, t = 0 "
                 "(auditor probe on the m15278 production RHS)", fontsize=9)
    fig.savefig(a.out, dpi=140, bbox_inches="tight")
    print("wrote", a.out, "symmetric color limit %.2f %%" % lim)


if __name__ == "__main__":
    main()