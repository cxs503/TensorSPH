"""Benchmark: transport-velocity correction on a hydrostatic column."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph import Config, run


def run_arm(arm_name: str, cfg: Config, out_dir: Path) -> dict:
    arm_dir = out_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    run(cfg, str(arm_dir), plot=False)

    import csv
    last_row = None
    with open(arm_dir / "diagnostics.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row

    snapshots = sorted(arm_dir.glob("particles_*.pt"))
    last_snap = torch.load(snapshots[-1], weights_only=True)

    return {
        "arm": arm_name,
        "max_speed": float(last_row["max_speed"]),
        "rho_max_relative_error": float(last_row["rho_max_relative_error"]),
        "steps": int(last_row["step"]),
        "time": float(last_row["time"]),
    }


def main():
    out_dir = Path(__file__).resolve().parents[2] / "xval_output" / "transport_velocity"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = dict(
        dx=0.02, h_factor=1.3, rho0=1000.0, sound_speed=20.0, gamma=7.0,
        gravity=9.81, viscosity=0.0, dt=3.0e-4, steps=1333, output_every=100,
        pool_width=0.4, pool_depth=0.4, drop_radius=0.0, drop_center_y=0.49,
        device="cpu", boundary="wall", wall_layers=3, max_courant=0.3,
        max_density_error=0.05, max_mach=0.10, fail_fast=False,
        pair_skin_factor=0.0, hydrostatic_init=False, calibrate_initial_mass=True,
        wall_friction=0.0, density_calibration=True, adaptive_dt=False,
        surface_tension_coefficient=0.0, contact_angle=None,
        csf_scheme="consistent", csf_curvature="fit_quadratic",
        csf_delta_normalization="legacy", surface_pressure_closure=False,
        surface_pressure_donors="interior", surface_pressure_layers=2,
        density_renormalization=False, density_regularization="none",
        compression_regularization_rule="free_surface",
        compression_normalization="reference_mean",
    )

    arms = {
        "baseline": Config(**base),
        "baseline+diffusion_molteni": Config(**base, density_diffusion=0.1),
        "transport_velocity": Config(**base, transport_velocity="on", transport_velocity_v_ref=1.0),
        "transport_velocity+diffusion_antuono": Config(
            **base, transport_velocity="on", transport_velocity_v_ref=1.0,
            density_diffusion=0.1, density_diffusion_form="antuono"),
    }

    results = []
    for name, cfg in arms.items():
        print(f"Running arm: {name} ...")
        r = run_arm(name, cfg, out_dir)
        results.append(r)
        print(f"  max_speed={r['max_speed']:.6f} m/s, rho_err={r['rho_max_relative_error']:.4f}")

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Comparison Table ===")
    print(f"{'Arm':<40} {'max_speed (m/s)':>15} {'rho_err (%)':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['arm']:<40} {r['max_speed']:>15.6f} {r['rho_max_relative_error']*100:>11.4f}")


if __name__ == "__main__":
    main()
