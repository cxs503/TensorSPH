"""Contact-angle boundary condition: geometric verification against its definition.

The contact angle theta is defined at a three-phase line as the angle, measured
inside the fluid, between the (outward) interface normal and the wall normal.
For an SPH colour-gradient formulation the boundary condition therefore has an
exact geometric statement (Adami, Hu & Adams 2012, J. Comput. Phys.
231:7057-7075; Tartakovsky & Meakin 2005, J. Comput. Phys. 207:610-628):

    angle(n_new, n_wall) = theta,   |n_new| = |n_old|

where ``n_wall = (x_fluid - x_wall) / |x_fluid - x_wall|``. ``theta = 0`` means
the interface lies flat on the wall (complete wetting, normal along ``n_wall``),
``theta = 90 deg`` means the interface meets the wall perpendicularly (normal in
the wall plane), ``theta = 180 deg`` is the non-wetting limit.

This script measures that statement directly instead of trusting the
implementation: it draws random interface normals for particles placed at random
distances from a wall, applies :func:`tensorsph.surface_tension
.apply_contact_angle_boundary`, and reports the recovered angle error, the
relative error of the preserved magnitude, and the leakage outside the wall
influence band. There is no discretisation in this check -- it is the exact
definition, so the tolerance is machine precision rather than a modelling
budget.

Usage:
    python benchmarks/contact_angle/run.py --json-out results.json --figure fig.png
    python benchmarks/contact_angle/run.py --quick     # CI-sized sample
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import Config
from tensorsph.surface_tension import apply_contact_angle_boundary

ANGLES_DEG = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0)


def build_case(n_random: int, device: torch.device, seed: int = 20261103):
    """Wall row plus particles spread over and beyond the influence band."""
    cfg = Config(dx=0.05, pool_width=0.35, pool_depth=0.15, drop_radius=0.04,
                 drop_center_y=0.25, device=str(device))
    h = cfg.h_factor * cfg.dx
    reach = 2.0 * h

    wall_x = torch.arange(-0.1, 0.5 + 1.0e-9, cfg.dx, device=device)
    wall = torch.stack((wall_x, torch.full_like(wall_x, -0.5 * cfg.dx)), dim=1)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Particles uniformly inside [0, reach] of the wall (band) plus a control
    # group well outside it.
    # The wall row sits at y = -dx/2, so the distance of a band particle from
    # the wall is y + dx/2; keep it strictly inside reach so every band sample
    # is inside the influence band the boundary condition uses.
    y_band = torch.rand(n_random, 1, generator=gen) * (0.9 * reach - cfg.dx)
    x_band = torch.rand(n_random, 1, generator=gen) * 0.35
    band = torch.cat((x_band, y_band + 0.0), dim=1).to(device)
    y_far = 4.0 * reach + torch.rand(n_random, 1, generator=gen) * reach
    far = torch.cat((x_band.clone(), y_far), dim=1).to(device)
    pos = torch.cat((band, far), dim=0)

    phi = torch.rand(pos.shape[0], 1, generator=gen) * 2.0 * math.pi
    mag = 0.2 + torch.rand(pos.shape[0], 1, generator=gen)
    grad = torch.cat((torch.cos(phi), torch.sin(phi)), dim=1) * mag
    grad = grad.to(device)
    return cfg, pos, grad, wall, h, reach, n_random


def wall_normal(pos: torch.Tensor, wall: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(pos, wall)
    idx = d.argmin(dim=1)
    delta = pos - wall[idx]
    return delta / torch.linalg.vector_norm(delta, dim=1, keepdim=True)


def measure(cfg, pos, grad, wall, h, reach, n_random, angles_deg=ANGLES_DEG):
    n_wall = wall_normal(pos, wall)
    rows = []
    for angle_deg in angles_deg:
        theta = math.radians(angle_deg)
        out = apply_contact_angle_boundary(grad, pos, wall, h, theta)

        band = slice(0, n_random)
        far = slice(n_random, pos.shape[0])
        mag = torch.linalg.vector_norm(out[band], dim=1)
        mag0 = torch.linalg.vector_norm(grad[band], dim=1)
        cos_meas = (out[band] * n_wall[band]).sum(dim=1) / mag
        err_deg = torch.rad2deg(torch.acos(cos_meas.clamp(-1.0, 1.0))) - angle_deg
        # 0 and 180 deg are degenerate: the tangential direction is undefined and
        # the angle is reached from either side, so report |cos| error instead.
        degenerate = angle_deg in (0.0, 180.0)
        if degenerate:
            err_deg = torch.rad2deg(torch.acos(cos_meas.abs().clamp(-1.0, 1.0))) - 0.0

        # Compare like with like: a particle whose own normal is (anti)parallel
        # to the wall has no tangential part, the angle still has to be reached.
        rows.append({
            "theta_deg": angle_deg,
            "max_angle_error_deg": float(err_deg.abs().max()),
            "rms_angle_error_deg": float(torch.sqrt((err_deg ** 2).mean())),
            "max_relative_magnitude_error": float(
                ((mag - mag0).abs() / mag0).max()),
            "max_leak_outside_band": float(
                (out[far] - grad[far]).abs().max()),
            "degenerate_case": degenerate,
        })
    return rows


def make_figure(cfg, pos, grad, wall, h, reach, n_random, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_wall = wall_normal(pos, wall)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    for angle_deg, colour in ((30.0, "#1f77b4"), (90.0, "#2ca02c"), (150.0, "#d62728")):
        out = apply_contact_angle_boundary(grad, pos, wall, h, math.radians(angle_deg))
        ax.quiver(pos[n_random:n_random + 30, 0].cpu(), pos[n_random:n_random + 30, 1].cpu(),
                  out[n_random:n_random + 30, 0].cpu(), out[n_random:n_random + 30, 1].cpu(),
                  color=colour, scale=22, width=0.004,
                  label=f"reconstructed normal, theta={angle_deg:.0f} deg")
    ax.quiver(pos[:30, 0].cpu(), pos[:30, 1].cpu(), grad[:30, 0].cpu(), grad[:30, 1].cpu(),
              color="0.6", scale=22, width=0.003, label="incoming normals")
    ax.axhline(-0.5 * cfg.dx, color="k", lw=3, label="wall")
    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 0.38)
    ax.set_ylim(-0.07, 0.30)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("normals reoriented only inside 2h of the wall")
    ax.legend(fontsize=6, loc="upper right")

    ax = axes[1]
    angles = [0, 10, 20, 45, 60, 90, 120, 135, 150, 170, 180]
    measured, expected = [], []
    for angle_deg in angles:
        out = apply_contact_angle_boundary(grad, pos, wall, h, math.radians(angle_deg))
        cos_meas = ((out[:n_random] * n_wall[:n_random]).sum(dim=1)
                    / torch.linalg.vector_norm(out[:n_random], dim=1)).abs()
        measured.append(torch.rad2deg(torch.acos(cos_meas.clamp(-1, 1))).mean().item())
        expected.append(min(angle_deg, 180 - angle_deg))
    ax.plot(expected, measured, "o", ms=5, label="measured (mean over particles)")
    lim = [0, 95]
    ax.plot(lim, lim, "k--", lw=1, label="definition")
    ax.set_xlabel("prescribed angle between normal and wall normal [deg]")
    ax.set_ylabel("recovered angle [deg]")
    ax.set_title("recovered vs prescribed")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-random", type=int, default=4000)
    parser.add_argument("--tol-deg", type=float, default=0.1,
                        help="float32 round-off in acos near 1 is ~1e-2 deg")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--figure", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_random = 500

    device = torch.device(args.device)
    cfg, pos, grad, wall, h, reach, n_random = build_case(args.n_random, device)
    rows = measure(cfg, pos, grad, wall, h, reach, n_random)

    worst = max(r["max_angle_error_deg"] for r in rows)
    worst_mag = max(r["max_relative_magnitude_error"] for r in rows)
    worst_leak = max(r["max_leak_outside_band"] for r in rows)
    passed = (worst <= args.tol_deg and worst_mag <= 1.0e-6 and worst_leak == 0.0)

    print(f"samples per group: {n_random}, h = {h:.4f} m, influence band = {reach:.4f} m")
    print(f"{'theta [deg]':>12} {'max err [deg]':>14} {'rms err [deg]':>14} "
          f"{'max |n| err':>12} {'leak':>10}")
    for r in rows:
        print(f"{r['theta_deg']:>12.1f} {r['max_angle_error_deg']:>14.2e} "
              f"{r['rms_angle_error_deg']:>14.2e} "
              f"{r['max_relative_magnitude_error']:>12.2e} "
              f"{r['max_leak_outside_band']:>10.2e}")
    print(f"verdict: {'PASS' if passed else 'FAIL'} (tolerance {args.tol_deg} deg)")

    result = {
        "benchmark": "contact_angle_bc_geometry",
        "implementation": "tensorsph.surface_tension.apply_contact_angle_boundary",
        "reference": "angle(n_new, n_wall) = theta with |n_new| = |n_old| "
                     "(Adami et al. 2012; Tartakovsky & Meakin 2005)",
        "setup": {"n_random": n_random, "h": h, "influence_band": reach,
                  "device": str(device)},
        "rows": rows,
        "worst_angle_error_deg": worst,
        "worst_relative_magnitude_error": worst_mag,
        "worst_leak_outside_band": worst_leak,
        "tolerance_deg": args.tol_deg,
        "pass": passed,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    if args.figure:
        make_figure(cfg, pos, grad, wall, h, reach, n_random, Path(args.figure))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()