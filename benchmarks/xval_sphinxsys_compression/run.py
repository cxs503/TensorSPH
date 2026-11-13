"""Cross-check (XVAL) SPHinXsys' compression-based density regularization.

The reference is the SPHinXsys implementation
(``Xiangyu-Hu/SPHinXsys``, ``src/shared/shared_ck/particle_dynamics/fluid_dynamics/
density_regularization``), ported in :mod:`tensorsph.compression_regularization`
through the ``Config.density_regularization`` switch of
:mod:`tensorsph.droplet_impact`.  This driver runs the same resting column with
each density scheme and reduces every arm to the same reference: the analytic
hydrostatic field, ``p(y) = rho0 g (H - y)`` and its Tait inverse
``rho(y) = rho0 (1 + rho0 g (H - y) / B) ** (1/gamma)``.

Arms
----
``summation``
    The shipped summation density (``density_regularization="none"``) -- the
    TensorSPH baseline.
``compression_none``
    The compression formulation as it stood in SPHinXsys before commit
    ``375ec50b42`` (2026-07-19): a plain kernel sum of reference volumes.
``compression_refmean``
    SPHinXsys master: the neighbour sum scaled by ``1/Kref``, the global inverse
    mean compression over the particles whose sum exceeds one.
``compression_refmean_internal``
    Same, with the ``internal`` flow-type rule (no ``max(K, 1)`` free-surface
    clip) -- the reference's own sensitivity knob.

Everything is fixed except the density scheme (same lattice, mass, kernel, time
step, wall, boundary handling), so a difference between arms is attributable.

Usage
-----
    python benchmarks/xval_sphinxsys_compression/run.py --quick
    python benchmarks/xval_sphinxsys_compression/run.py --dx 0.02 --dx 0.015
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorsph.case_hydrostatic import (  # noqa: E402
    analytic_pressure, evaluate, pressure_profile, support_width)
from tensorsph.droplet_impact import (  # noqa: E402
    Config, PairListCache, apply_boundary, cell_centres, hydrostatic_density,
    lattice_mass_unit, resolve_device, rhs, wall_lattice)

RHO0 = 1000.0
G = 9.81
DEPTH = 0.4
WIDTH = 0.4
H_FACTOR = 1.3
BANDS = 8

#: (arm name, Config overrides) -- the only thing that changes between arms.
ARMS = (
    ("summation", dict(density_regularization="none")),
    ("compression_none", dict(density_regularization="compression",
                              compression_normalization="none",
                              compression_regularization_rule="free_surface")),
    ("compression_refmean", dict(density_regularization="compression",
                                 compression_normalization="reference_mean",
                                 compression_regularization_rule="free_surface")),
    ("compression_refmean_internal", dict(density_regularization="compression",
                                          compression_normalization="reference_mean",
                                          compression_regularization_rule="internal")),
)


def build_pool(dx: float, width: float, depth: float, device, **overrides):
    """Uniform lattice with the calibrated lattice mass.

    ``hydrostatic_init`` is deliberately *off*: the reference code initialises
    particles with a uniform volumetric measure and ``rho = rho0``, so the
    reference volumes here are the lattice volumes ``m / rho0`` and nothing else.
    """
    cfg = Config(dx=dx, pool_width=width, pool_depth=depth, gravity=G,
                 sound_speed=20.0, boundary="wall", device=str(device),
                 hydrostatic_init=False, h_factor=H_FACTOR, **overrides)
    xs = cell_centres(width, dx, device)
    ys = cell_centres(depth, dx, device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    unit = lattice_mass_unit(cfg)
    mass = torch.full((pos.shape[0],), unit, dtype=pos.dtype, device=device)
    return cfg, pos, mass


def build_cfg(dx: float, width: float, depth: float, device, **overrides) -> Config:
    return Config(dx=dx, pool_width=width, pool_depth=depth, gravity=G,
                  sound_speed=20.0, boundary="wall", device=str(device),
                  hydrostatic_init=False, h_factor=H_FACTOR, **overrides)


def run_arm(name: str, overrides: dict, dx: float, width: float, depth: float,
            seconds: float, bands: int, average_from: float,
            sample_every: int, device) -> dict:
    """One arm of the controlled comparison; the loop is the shipped one."""
    cfg, pos, mass = build_pool(dx, width, depth, device, **overrides)
    fluid = pos.shape[0]
    wall = wall_lattice(cfg, device, top=depth)
    vel = torch.zeros_like(pos)
    h = cfg.h_factor * dx
    cache = PairListCache(2.0 * h, 0.25 * h, fluid_count=fluid)
    dt = 0.3 * H_FACTOR * dx / cfg.sound_speed
    steps = int(math.ceil(seconds / dt))
    dt = seconds / steps
    state: dict = {}

    accum = {k: torch.zeros(v, dtype=pos.dtype, device=device)
             for k, v in (("p", fluid), ("p2", fluid), ("rho", fluid),
                          ("rho2", fluid), ("x", pos.shape), ("m", fluid))}
    accum_v2 = torch.zeros(fluid, dtype=pos.dtype, device=device)
    samples = 0
    series = {"t": [], "p_bottom": [], "v_rms": [], "kref": [], "k_mean": []}
    support = support_width(dx)
    started = time.time()
    for step in range(steps):
        t = step * dt
        pairs, _ = cache.get(torch.cat((pos, wall), dim=0))
        rho, pressure, acc = rhs(pos, vel, mass, cfg, pairs=pairs, wall=wall,
                                 dt=dt, state=state)
        if not bool(torch.isfinite(acc).all()):
            raise FloatingPointError(f"{name}: non-finite acceleration at step {step}")
        if t >= average_from * seconds and step % sample_every == 0:
            for key, value in (("p", pressure), ("p2", pressure.square()),
                               ("rho", rho), ("rho2", rho.square()),
                               ("x", pos), ("m", mass)):
                accum[key] += value
            accum_v2 += vel.square().sum(dim=1)
            samples += 1
            bottom = pos[:, 1] < dx * 1.5
            series["t"].append(t)
            series["p_bottom"].append(float(pressure[bottom].mean()))
            series["v_rms"].append(float(vel.square().sum(dim=1).mean().sqrt()))
            series["kref"].append(
                float(state["compression"].inverse_reference_compression)
                if "compression" in state else 1.0)
            series["k_mean"].append(float((rho / RHO0).mean()))
        vel = vel + dt * acc
        pos = pos + dt * vel
        apply_boundary(pos, vel, cfg, top=depth)
    wall_seconds = time.time() - started
    if samples == 0:
        raise RuntimeError(f"{name}: no samples in the averaging window")

    mean_pos = accum["x"] / samples
    mean_p = accum["p"] / samples
    std_p = (accum["p2"] / samples - mean_p.square()).clamp_min(0.0).sqrt()
    mean_rho = accum["rho"] / samples
    mean_mass = accum["m"] / samples
    v_rms = float((accum_v2 / samples).mean().sqrt())

    rows = pressure_profile(mean_pos, mean_p, mean_mass, depth, dx, bands)
    gate = evaluate(rows)
    rho_ref = hydrostatic_density(mean_pos[:, 1], cfg, surface=depth)
    for row in rows:
        mask = (mean_pos[:, 1] >= row["y_low"]) & (mean_pos[:, 1] < row["y_high"])
        row["rho"] = float(mean_rho[mask].mean()) if bool(mask.any()) else None
        row["rho_analytic"] = float(rho_ref[mask].mean()) if bool(mask.any()) else None
        row["rho_rel_error_pct"] = (
            (row["rho"] - row["rho_analytic"]) / row["rho_analytic"] * 100.0
            if row["rho_analytic"] else None)
        row["fluctuation_pct"] = (
            float((std_p[mask] / mean_p[mask].clamp_min(1.0)).max() * 100.0)
            if bool(mask.any()) else None)
    gated_rho = [abs(r["rho_rel_error_pct"]) for r in rows
                 if r["gated"] and r["rho_rel_error_pct"] is not None]
    return {
        "arm": name,
        "overrides": overrides,
        "fluid_particles": int(fluid),
        "wall_particles": int(wall.shape[0]),
        "dt": dt,
        "steps": steps,
        "samples": samples,
        "wall_seconds": round(wall_seconds, 2),
        "gate": gate,
        "max_rho_band_error_pct": max(gated_rho) if gated_rho else None,
        "v_rms_mean_ms": v_rms,
        "k_ref_final": series["kref"][-1] if series["kref"] else None,
        "k_mean_final": series["k_mean"][-1] if series["k_mean"] else None,
        "profile": rows,
        "series": series,
        "nondimensional": {
            "H_over_dx": depth / dx,
            "W_over_H": width / depth,
            "h_over_dx": H_FACTOR,
            "c0": cfg.sound_speed,
            "mach_hydrostatic": math.sqrt(2.0 * G * depth) / cfg.sound_speed,
            "acoustic_wave_courant": cfg.sound_speed * dt / dx,
            "gravity_wave_courant": math.sqrt(G * depth) * dt / dx,
            "viscosity_m2_s": cfg.viscosity,
            "acoustic_periods": (cfg.sound_speed / (2.0 * depth)) * seconds,
            "hydrostatic_compression": float(
                (hydrostatic_density(torch.tensor([0.0], dtype=torch.float64),
                                     cfg, surface=depth) / RHO0 - 1.0)),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, nargs="+", default=[0.02])
    ap.add_argument("--width", type=float, default=WIDTH)
    ap.add_argument("--depth", type=float, default=DEPTH)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--bands", type=int, default=BANDS)
    ap.add_argument("--average-from", type=float, default=0.5)
    ap.add_argument("--sample-every", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--arms", default=",".join(a[0] for a in ARMS))
    ap.add_argument("--out", default=str(Path("xval_output") /
                                         "xval_sphinxsys_compression" /
                                         "records.json"))
    ap.add_argument("--quick", action="store_true",
                    help="CI-sized smoke configuration (coarse, short)")
    args = ap.parse_args(argv)
    if args.quick:
        args.dx = [0.04]
        args.seconds = 0.4
        args.bands = 4
    selected = [a for a in ARMS if a[0] in set(args.arms.split(","))]
    if not selected:
        raise SystemExit("no arm selected")
    device = resolve_device(args.device)

    records = []
    for dx in args.dx:
        for name, overrides in selected:
            print(f"[xval] dx={dx} arm={name}", flush=True)
            record = run_arm(name, overrides, dx, args.width, args.depth,
                             args.seconds, args.bands, args.average_from,
                             args.sample_every, device)
            record["dx"] = dx
            record["width"] = args.width
            record["depth"] = args.depth
            record["seconds"] = args.seconds
            record["bands"] = args.bands
            print(f"[xval]   max gate error {record['gate']['max_band_error_pct']:.2f} %"
                  f"  rho {record['max_rho_band_error_pct']}"
                  f"  v_rms {record['v_rms_mean_ms']:.4f} m/s"
                  f"  wall {record['wall_seconds']} s", flush=True)
            records.append(record)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": "XVAL-SPHINXSYS-COMPRESSION-DENSITY-REGULARIZATION",
        "reference": {
            "kind": "analytic",
            "pressure": "p(y) = rho0 g (H - y)",
            "density": "rho(y) = rho0 (1 + rho0 g (H - y) / B) ** (1/gamma)",
            "rho0": RHO0, "g": G,
        },
        "oss_reference": {
            "repo": "Xiangyu-Hu/SPHinXsys",
            "tree_sha": "1612330e44c8947406f24b6756449e80dcd7158e",
            "file": "src/shared/shared_ck/particle_dynamics/fluid_dynamics/"
                    "density_regularization.{h,hpp,cpp}",
            "formulation_without_reference_mean": "commit d5c793e08c (2026-06-19)",
            "reference_mean_introduced": "commit 375ec50b42 (2026-07-19)",
        },
        "records": records,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"[xval] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())