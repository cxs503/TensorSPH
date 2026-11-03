"""Verification harness for the contact-angle boundary condition of the CSF model.

Why this harness exists
-----------------------
``apply_contact_angle_boundary`` (restored in src/tensorsph/surface_tension.py)
rotates the colour gradient of near-wall particles so that the CSF normal
satisfies ``n . n_wall = cos(theta)``.  The unit tests that shipped with the
feature only checked shapes, "far particles unchanged" and magnitude
preservation -- no test ever measured the *angle* that comes out, and the two
integration tests used a six-particle cluster whose curvature is identically
zero, so they could not tell a working BC from a silent no-op.

This harness measures, on a hex-packed droplet resting on a wall lattice:

1. the realised normal angle ``theta_meas = arccos(n . n_wall)`` versus the
   requested ``theta`` for theta = 30..150 deg (identity residual);
2. |grad(c)| preservation, i.e. the BC must rotate, not rescale;
3. that the BC leaves the CSF force finite and that theta = 90 deg is the
   neutral case (no wall-normal component added).

Scope note (honest): these are *implementation* checks of the enforced normal,
not a physical wetting-angle validation.  Turning the BC into a measured
sessile-drop angle needs a free-surface simulation and is *not* claimed here.

Usage::

    PYTHONPATH=src python benchmarks/contact_angle/run.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import Config
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension import (
    apply_contact_angle_boundary,
    compute_surface_tension_force,
    csf_colour_field,
)

torch.set_num_threads(2)

THETAS_DEG = (30.0, 60.0, 90.0, 120.0, 150.0)


def hexagonal_drop(radius: float, dx: float, rho0: float,
                   center: tuple[float, float] = (0.0, 0.0)):
    """Hexagonally packed circular drop (lattice constant set by rho0)."""
    dy = dx * math.sqrt(3.0) / 2.0
    mass = torch.tensor(rho0 * dx * dy, dtype=torch.float64)
    pts = []
    j = 0
    y = -radius
    while y <= radius:
        x = -radius + (0.5 * dx if (j % 2) else 0.0)
        while x <= radius:
            if x * x + y * y <= radius * radius:
                pts.append((center[0] + x, center[1] + y))
            x += dx
        y += dy
        j += 1
    pos = torch.tensor(pts, dtype=torch.float64)
    return pos, torch.full((pos.shape[0],), float(mass), dtype=torch.float64)


def wall_strip(dx: float, half_width: float, height: float = 0.0) -> torch.Tensor:
    """Two rows of wall particles below ``y = 0`` (a flat solid bottom)."""
    xs = torch.arange(-half_width, half_width + 0.5 * dx, dx, dtype=torch.float64)
    rows = []
    for k in range(2):
        y = height - (k + 0.5) * dx
        rows.append(torch.stack([xs, torch.full_like(xs, y)], dim=1))
    return torch.cat(rows, dim=0)


def normalise(v: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v, dim=1, keepdim=True).clamp_min(eps)


def measure(pos, mass, density, pairs, wall, h, theta_deg):
    """Enforce ``theta`` and report the realised interface angle + residuals."""
    color, grad_c, is_surface = csf_colour_field(pos, mass, density, pairs, h)
    mag0 = torch.linalg.vector_norm(grad_c, dim=1)

    # near-wall = within 2 h of a wall particle; wall normal points into fluid
    diff = pos[:, None, :] - wall[None, :, :]
    dist = torch.linalg.vector_norm(diff, dim=2)
    nearest = dist.argmin(dim=1)
    d_min = dist.min(dim=1).values
    near_wall = d_min < 2.0 * h
    n_wall = normalise(pos - wall[nearest])

    theta = math.radians(theta_deg)
    grad_bc = apply_contact_angle_boundary(grad_c.clone(), pos, wall, h, theta)
    mag1 = torch.linalg.vector_norm(grad_bc, dim=1)

    sel = near_wall & is_surface & (mag0 > 1.0e-8)
    n_before = normalise(-grad_c)
    n_after = normalise(-grad_bc)
    cos_before = (n_before * n_wall).sum(dim=1)
    cos_after = (n_after * n_wall).sum(dim=1)
    angle_after = torch.rad2deg(torch.arccos(cos_after.clamp(-1.0, 1.0)))

    rel_mag = ((mag1[sel] - mag0[sel]).abs() / mag0[sel]).max()

    return {
        "theta_deg": theta_deg,
        "target_cos": float(math.cos(theta)),
        "n_particles_scored": int(sel.sum()),
        "cos_before_median": float(cos_before[sel].median()),
        "cos_after_median": float(cos_after[sel].median()),
        "cos_after_max_abs_residual": float((cos_after[sel] - math.cos(theta)).abs().max()),
        "angle_after_error_deg_max": float((angle_after[sel] - theta_deg).abs().max()),
        "grad_magnitude_rel_change_max": float(rel_mag),
    }


def main():
    radius, dx, rho0, sigma = 0.02, 0.002, 1000.0, 0.072
    out = Path(__file__).resolve().parent
    cfg = Config(dx=dx, rho0=rho0, gravity=0.0, sound_speed=20.0,
                 surface_tension_coefficient=sigma)
    h = cfg.h_factor * cfg.dx

    drop, mass = hexagonal_drop(radius, dx, rho0, center=(0.0, radius + 0.5 * dx))
    wall = wall_strip(dx, half_width=1.5 * radius)
    density = torch.full((drop.shape[0],), rho0, dtype=torch.float64)
    pairs = cell_linked_pairs(drop, 2.0 * h)

    rows = [measure(drop, mass, density, pairs, wall, h, t) for t in THETAS_DEG]

    force_rows = []
    for t in THETAS_DEG:
        cfg_t = Config(dx=dx, rho0=rho0, gravity=0.0, sound_speed=20.0,
                       surface_tension_coefficient=sigma,
                       contact_angle=math.radians(t))
        acc = compute_surface_tension_force(drop, density, mass, pairs, cfg_t, wall=wall)
        force_rows.append({
            "theta_deg": t,
            "max_acc_m_per_s2": float(torch.linalg.vector_norm(acc, dim=1).max()),
            "total_acc_y": float(acc[:, 1].sum()),
            "finite": bool(torch.isfinite(acc).all()),
        })

    report = {
        "benchmark": "csf_contact_angle_boundary",
        "setup": {
            "radius_m": radius, "dx_m": dx, "h_m": h, "n_particles": int(drop.shape[0]),
            "n_wall_particles": int(wall.shape[0]), "rho0": rho0,
            "sigma_N_per_m": sigma, "lattice": "hexagonal", "wall": "flat bottom, 2 rows",
            "theory": "n . n_wall = cos(theta); |grad(c)| preserved by the rotation",
        },
        "angle_identity": rows,
        "force_probe": force_rows,
        "verdict": {
            "max_cos_residual": max(r["cos_after_max_abs_residual"] for r in rows),
            "max_angle_error_deg": max(r["angle_after_error_deg_max"] for r in rows),
            "max_grad_magnitude_rel_change": max(r["grad_magnitude_rel_change_max"] for r in rows),
            "within_1e-6_cos": all(r["cos_after_max_abs_residual"] < 1e-6 for r in rows),
            "magnitude_preserved_to_1e-6": all(
                r["grad_magnitude_rel_change_max"] < 1e-6 for r in rows),
        },
    }
    (out / "results.json").write_text(json.dumps(report, indent=2))

    print(f"# contact-angle BC on hex drop R={radius} dx={dx} N={drop.shape[0]}")
    print("theta  cos_before  cos_after  max|cos-cos_t|  max|dtheta| deg  d|grad|")
    for r in rows:
        print(f"{r['theta_deg']:5.0f}  {r['cos_before_median']:+10.4f}  "
              f"{r['cos_after_median']:+9.4f}  {r['cos_after_max_abs_residual']:14.2e}  "
              f"{r['angle_after_error_deg_max']:15.2e}  "
              f"{r['grad_magnitude_rel_change_max']:8.2e}")
    print("verdict:", report["verdict"])
    return report


if __name__ == "__main__":
    main()