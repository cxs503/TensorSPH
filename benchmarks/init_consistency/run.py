"""Initial-condition audit: kernel-sum (zeroth-moment) consistency at ``t = 0``.

Answers three questions with numbers instead of folklore:

1. **What is the discrete kernel-sum bias ``C - 1`` as a function of lattice and
   ``h / dx``?**  Swept over square and hexagonal lattices, ``h / dx`` in
   ``[1.0, 2.2]``.
2. **Does it depend on ``dx``?**  On a uniform lattice ``C`` is a lattice sum in
   units of ``h``, so it must be resolution independent -- verified on a
   ``dx`` ladder.  A bias that changed with ``dx`` would be a different bug.
3. **Does rescaling the mass by ``1 / C`` clear the spurious pressure?**
   Acceptance criterion (from the m2971 backlog, P1):
   ``p_false / (rho0 c0^2) < 1e-4`` at ``t = 0``.

The hexagonal disc of ``benchmarks/laplace_plateau`` is reproduced verbatim so
the ``+5.889e-3`` reported there is traceable to its lattice, not to the solver.

Run::

    PYTHONPATH=src python benchmarks/init_consistency/run.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensorsph.droplet_impact import cubic_spline_kernel  # noqa: F401  (API traceability)
from tensorsph.lattice_consistency import (
    calibrate_mass,
    interior_mask,
    kernel_sum_factor,
    radial_interior_mask,
    spurious_pressure,
)

RHO0 = 1000.0
GAMMA = 7.0
C0 = 20.0
H_FACTORS = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2)
DX_LADDER = (0.0020, 0.0015, 0.0010)
DTYPE = torch.float64


def square_lattice(extent: float, dx: float) -> torch.Tensor:
    """Square lattice with cell size ``dx`` (area per particle ``dx^2``)."""
    n = int(round(extent / dx)) + 1
    axis = torch.arange(n, dtype=DTYPE) * dx
    x, y = torch.meshgrid(axis, axis, indexing="xy")
    return torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)


def hex_lattice(extent: float, dx: float) -> torch.Tensor:
    """Hexagonal lattice with nearest-neighbour distance ``dx``.

    Row spacing ``dx sqrt(3)/2`` and half-cell stagger, so the area per particle
    is ``dx^2 sqrt(3)/2`` -- identical to the ``hexagonal_droplet`` generator of
    ``benchmarks/laplace_plateau``.
    """
    dy = dx * math.sqrt(3.0) / 2.0
    rows = []
    y, row = 0.0, 0
    while y <= extent:
        offset = 0.5 * dx if row % 2 else 0.0
        xs = torch.arange(int(round(extent / dx)) + 1, dtype=DTYPE) * dx + offset
        xs = xs[xs <= extent]
        rows.append(torch.stack((xs, torch.full_like(xs, y)), dim=-1))
        y += dy
        row += 1
    return torch.cat(rows, dim=0)


def hex_disc_laplace_plateau(radius: float, dx: float) -> torch.Tensor:
    """Verbatim copy of ``benchmarks/laplace_plateau/run.py::hexagonal_droplet`` positions."""
    dy = dx * math.sqrt(3.0) / 2.0
    pts = []
    j, y = 0, -radius
    while y <= radius:
        off = 0.5 * dx if (j % 2) else 0.0
        x = -radius + off
        while x <= radius:
            if x * x + y * y <= radius * radius:
                pts.append((x, y))
            x += dx
        y += dy
        j += 1
    return torch.tensor(pts, dtype=DTYPE)


def _spacing_volume(kind: str, dx: float) -> float:
    return dx * dx if kind == "square" else dx * dx * math.sqrt(3.0) / 2.0


def audit_case(kind: str, dx: float, h_factor: float, positions: torch.Tensor) -> dict:
    h = h_factor * dx
    volume = torch.full((positions.shape[0],), _spacing_volume(kind, dx), dtype=DTYPE)
    mass = volume * RHO0
    c = kernel_sum_factor(positions, volume, h)
    mask = interior_mask(positions, h)
    mass_cal, diag = calibrate_mass(positions, mass, h, RHO0)
    c_cal = kernel_sum_factor(positions, mass_cal / RHO0, h)
    bias_raw = float(c[mask].mean()) - 1.0
    bias_cal = float(c_cal[mask].mean()) - 1.0
    return {
        "lattice": kind,
        "dx_m": dx,
        "h_over_dx": h_factor,
        "n_particles": int(positions.shape[0]),
        "n_interior": int(mask.sum()),
        "kernel_sum_raw": float(c[mask].mean()),
        "density_bias_raw": bias_raw,
        "interior_spread_raw": float(c[mask].std()),
        "p_false_raw_Pa_c0_20": spurious_pressure(bias_raw, RHO0, C0, GAMMA),
        "p_false_raw_Pa_c0_5": spurious_pressure(bias_raw, RHO0, 5.0, GAMMA),
        "kernel_sum_calibrated": float(c_cal[mask].mean()),
        "density_bias_calibrated": bias_cal,
        "p_false_calibrated_Pa_c0_20": spurious_pressure(bias_cal, RHO0, C0, GAMMA),
        "mass_correction": float(diag["kernel_sum"]),
        "criterion_rel_pressure": abs(spurious_pressure(bias_cal, RHO0, C0, GAMMA))
        / (RHO0 * C0 * C0),
        "criterion_met": (bool(diag["applied"])
                          and abs(spurious_pressure(bias_cal, RHO0, C0, GAMMA))
                          / (RHO0 * C0 * C0) < 1e-4),
    }


def audit() -> dict:
    dx = 0.01
    extent = 0.20
    sweep = []
    for kind in ("square", "hex"):
        pos = square_lattice(extent, dx) if kind == "square" else hex_lattice(extent, dx)
        for hf in H_FACTORS:
            sweep.append(audit_case(kind, dx, hf, pos))

    resolution = []
    for kind in ("square", "hex"):
        for d in DX_LADDER:
            pos = square_lattice(0.05, d) if kind == "square" else hex_lattice(0.05, d)
            resolution.append(audit_case(kind, d, 1.3, pos))

    # Reproduce the laplace_plateau hexagonal disc, which is *not* uniform: the
    # circle cut leaves surface particles, so C is read on the r < 0.5 R core.
    legacy = []
    for d in (0.0020, 0.0015, 0.0010):
        radius = 0.025
        pos = hex_disc_laplace_plateau(radius, d)
        h = 1.3 * d
        volume = torch.full((pos.shape[0],), d * d * math.sqrt(3.0) / 2.0, dtype=DTYPE)
        c = kernel_sum_factor(pos, volume, h)
        core = torch.linalg.vector_norm(pos, dim=1) < 0.5 * radius
        bias = float(c[core].mean()) - 1.0
        # A disc needs a radial interior: the axis-aligned box mask reaches the
        # free surface on the diagonal (see radial_interior_mask docstring).
        mass_cal, diag = calibrate_mass(
            pos, volume * RHO0, h, RHO0,
            interior=radial_interior_mask(pos, (0.0, 0.0), radius, h))
        c_cal = kernel_sum_factor(pos, mass_cal / RHO0, h)
        bias_cal = float(c_cal[core].mean()) - 1.0
        # Cross-check against the convention used by laplace_plateau itself (r < 0.6 R).
        near_core = torch.linalg.vector_norm(pos, dim=1) < 0.6 * radius
        bias_06r = float(c[near_core].mean()) - 1.0
        legacy.append({
            "case": "laplace_plateau.hexagonal_droplet",
            "dx_m": d,
            "radius_m": radius,
            "n_particles": int(pos.shape[0]),
            "kernel_sum_core_raw": float(c[core].mean()),
            "density_bias_core_raw": bias,
            "p_false_core_raw_Pa_c0_20": spurious_pressure(bias, RHO0, C0, GAMMA),
            "density_bias_core_calibrated": bias_cal,
            "p_false_core_calibrated_Pa_c0_20": spurious_pressure(bias_cal, RHO0, C0, GAMMA),
            "calibration_factor": diag["kernel_sum"],
            "calibration_interior_count": diag["interior_count"],
            "density_bias_core_raw_convention_r06R": bias_06r,
        })

    return {
        "mission": "m3114",
        "what": "discrete kernel-sum (zeroth moment) audit of WCSPH initial conditions",
        "constants": {"rho0": RHO0, "gamma": GAMMA, "c0": C0},
        "criterion": {"name": "initial spurious pressure",
                      "expr": "|p_false| / (rho0 c0^2) < 1e-4",
                      "note": "m2971 backlog P1 acceptance criterion"},
        "sweep_lattice_h_over_dx": sweep,
        "resolution_ladder": resolution,
        "laplace_plateau_reproduction": legacy,
        "summary": {
            "square_h1.3_bias": next(r["density_bias_raw"] for r in sweep
                                     if r["lattice"] == "square" and r["h_over_dx"] == 1.3),
            "hex_h1.3_bias": next(r["density_bias_raw"] for r in sweep
                                  if r["lattice"] == "hex" and r["h_over_dx"] == 1.3),
            "all_calibrated_meet_criterion": all(r["criterion_met"] for r in sweep),
            "worst_calibrated_rel_pressure": max(r["criterion_rel_pressure"] for r in sweep),
            "worst_raw_rel_pressure": max(
                abs(r["p_false_raw_Pa_c0_20"]) / (RHO0 * C0 * C0) for r in sweep),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "report.json"))
    args = ap.parse_args()
    result = audit()
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=1, sort_keys=True))
    s = result["summary"]
    print(f"square h=1.3dx bias = {s['square_h1.3_bias']:+.3e}")
    print(f"hex    h=1.3dx bias = {s['hex_h1.3_bias']:+.3e}")
    print(f"all calibrated meet |p|/(rho0 c0^2) < 1e-4 : {s['all_calibrated_meet_criterion']}"
          f" (worst {s['worst_calibrated_rel_pressure']:.2e},"
          f" raw worst {s['worst_raw_rel_pressure']:.2e})")
    for r in result["laplace_plateau_reproduction"]:
        print(f"  plateau hex disc dx={r['dx_m']}: bias {r['density_bias_core_raw']:+.4e}"
              f" -> {r['p_false_core_raw_Pa_c0_20']:.1f} Pa"
              f"; calibrated {r['density_bias_core_calibrated']:+.2e}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
