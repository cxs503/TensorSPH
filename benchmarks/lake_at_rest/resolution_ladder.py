"""Resolution ladder for CASE-SPH-LAKE-AT-REST (periodic tank).

Measures the convergence of the interior-band profile error (density and
pressure vs the exact Tait equilibrium) at multiple resolutions.  This fills
the gap documented in ``docs/verification_lake_at_rest.md`` §5 ("Not done"):
the interior-band profile error had no measured convergence order because a
second resolution needs the Newton solve again.

Usage:
    PYTHONPATH=src python3 benchmarks/lake_at_rest/resolution_ladder.py --json out.json

The ladder runs at dx = 0.04, 0.02, 0.01 (periodic tank) and reports:
- equilibrium residual (both components)
- interior-band density error vs exact Tait profile
- interior-band pressure error vs exact Tait profile
- convergence order (computed from the two finest resolutions)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

from tensorsph.lake_at_rest import (LakeConfig, analytic_profile, build_state,
                                    reconstruct, residual_report, run_case)
from tensorsph import eos_hydrostatics as eos


def interior_band_mask(state, cfg, margin_factor=2.0):
    """Boolean mask for particles at least ``margin_factor * h`` from bottom and surface."""
    h = cfg.h()
    y = state["pos"][:, 1]
    bottom = y > margin_factor * h
    surface = y < cfg.depth - margin_factor * h
    return bottom & surface


def profile_error(state, cfg, mass):
    """Interior-band profile error: density and pressure vs exact Tait equilibrium."""
    mask = interior_band_mask(state, cfg)
    if mask.sum() == 0:
        return {"density_max_rel": None, "pressure_max_rel": None,
                "n_interior": 0}

    pos = state["pos"][mask]
    y = pos[:, 1]
    H = cfg.depth

    # Exact Tait equilibrium density and pressure
    rho_exact = eos.eos_hydrostatic_density(
        y, H, rho0=cfg.rho0, sound_speed=cfg.sound_speed,
        gamma=cfg.gamma, gravity=cfg.gravity)
    p_exact = eos.eos_hydrostatic_pressure(
        y, H, rho0=cfg.rho0, sound_speed=cfg.sound_speed,
        gamma=cfg.gamma, gravity=cfg.gravity)

    # Reconstructed density and pressure from the mass field
    report = residual_report(state, cfg, mass)
    rho_recon = report["density"][mask]
    p_recon = report["pressure"][mask]

    # Relative errors
    density_rel = ((rho_recon - rho_exact) / rho_exact).abs().max().item()
    # Pressure error normalised by the bottom pressure scale
    p_scale = cfg.rho0 * cfg.gravity * H
    pressure_rel = ((p_recon - p_exact) / p_scale).abs().max().item()

    return {
        "density_max_rel": density_rel,
        "pressure_max_rel": pressure_rel,
        "n_interior": int(mask.sum()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
    }


def convergence_order(err_coarse, err_fine, dx_coarse, dx_fine):
    """Compute observed order of accuracy: p = log(err_coarse/err_fine) / log(dx_coarse/dx_fine)."""
    if err_coarse is None or err_fine is None or err_fine == 0:
        return None
    ratio = err_coarse / err_fine
    if ratio <= 1.0:
        return None  # not converging
    return math.log(ratio) / math.log(dx_coarse / dx_fine)


def run_ladder(dx_values=None, depth=0.30, width=0.40, t_end=0.50):
    """Run the resolution ladder and return the results."""
    if dx_values is None:
        dx_values = [0.04, 0.02, 0.01]

    results = []
    for dx in dx_values:
        # periodic_pairs requires width >= 4h = 4 * 1.3 * dx = 5.2 * dx
        # For dx=0.08, h=0.104, need width >= 0.416 -> use width=0.52
        h_factor = 1.3
        min_width = 4.0 * h_factor * dx
        actual_width = max(width, min_width + 1e-9)

        print(f"\n--- dx = {dx} (width={actual_width:.3f}) ---")
        torch.set_default_dtype(torch.float64)
        cfg = LakeConfig(dx=dx, depth=depth, width=actual_width, t_end=t_end,
                         periodic_x=True, solver_max_iter=40)
        state = build_state(cfg)
        prior = analytic_profile(state, cfg)
        sol = reconstruct(state, cfg, prior=prior["mass"])
        prof_err = profile_error(state, cfg, sol["mass"])

        row = {
            "dx": dx,
            "width": actual_width,
            "h": cfg.h(),
            "n_fluid": state["n_fluid"],
            "n_wall": state["n_wall"],
            "n_pairs": int(state["pairs"].shape[0]),
            "equilibrium": {
                "converged": sol["converged"],
                "iterations": sol["iterations"],
                "a_max_over_g": sol["a_max_over_g"],
                "a_max_over_g_horizontal": sol["a_max_over_g_horizontal"],
            },
            "profile_error": prof_err,
        }
        results.append(row)
        print(f"  particles={state['n_fluid']}, pairs={row['n_pairs']}")
        print(f"  equilibrium: a_y/g={sol['a_max_over_g']:.3e}, "
              f"a_x/g={sol['a_max_over_g_horizontal']:.3e}, "
              f"converged={sol['converged']}, iter={sol['iterations']}")
        print(f"  interior band: {prof_err['n_interior']} particles, "
              f"y in [{prof_err['y_min']:.3f}, {prof_err['y_max']:.3f}]")
        print(f"  density error: {prof_err['density_max_rel']:.3e}")
        print(f"  pressure error: {prof_err['pressure_max_rel']:.3e}")

    # Compute convergence orders from the two finest resolutions
    if len(results) >= 2:
        fine = results[-1]
        coarse = results[-2]
        dx_ratio = coarse["dx"] / fine["dx"]
        fine["convergence"] = {
            "dx_ratio": dx_ratio,
            "density_order": convergence_order(
                coarse["profile_error"]["density_max_rel"],
                fine["profile_error"]["density_max_rel"],
                coarse["dx"], fine["dx"]),
            "pressure_order": convergence_order(
                coarse["profile_error"]["pressure_max_rel"],
                fine["profile_error"]["pressure_max_rel"],
                coarse["dx"], fine["dx"]),
        }
        if len(results) >= 3:
            fine["convergence"]["density_order_3pt"] = convergence_order(
                results[0]["profile_error"]["density_max_rel"],
                fine["profile_error"]["density_max_rel"],
                results[0]["dx"], fine["dx"])
            fine["convergence"]["pressure_order_3pt"] = convergence_order(
                results[0]["profile_error"]["pressure_max_rel"],
                fine["profile_error"]["pressure_max_rel"],
                results[0]["dx"], fine["dx"])

    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", type=float, nargs="+", default=None,
                        help="Resolution values (default: 0.04 0.02 0.01)")
    parser.add_argument("--depth", type=float, default=0.30)
    parser.add_argument("--width", type=float, default=0.40)
    parser.add_argument("--t-end", type=float, default=0.50)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    results = run_ladder(dx_values=args.dx, depth=args.depth,
                         width=args.width, t_end=args.t_end)

    output = {
        "case": "CASE-SPH-LAKE-AT-REST",
        "variant": "periodic_x",
        "ladder": results,
    }

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(output, indent=2, sort_keys=True))
        print(f"\nwrote {args.json}")

    # Print summary table
    print("\n" + "=" * 90)
    print("Resolution ladder summary (periodic tank)")
    print("=" * 90)
    print(f"{'dx':>8s} {'width':>8s} {'N':>6s} {'a_y/g':>12s} {'a_x/g':>12s} "
          f"{'rho_err':>12s} {'p_err':>12s}")
    print("-" * 90)
    for row in results:
        pe = row["profile_error"]
        print(f"{row['dx']:8.3f} {row['width']:8.3f} {row['n_fluid']:6d} "
              f"{row['equilibrium']['a_max_over_g']:12.3e} "
              f"{row['equilibrium']['a_max_over_g_horizontal']:12.3e} "
              f"{pe['density_max_rel']:12.3e} "
              f"{pe['pressure_max_rel']:12.3e}")

    if len(results) >= 2:
        conv = results[-1].get("convergence", {})
        print("-" * 90)
        print(f"Convergence order (2-pt, dx={results[-2]['dx']}->{results[-1]['dx']}):")
        print(f"  density:  {conv.get('density_order', 'n/a')}")
        print(f"  pressure: {conv.get('pressure_order', 'n/a')}")
        if "density_order_3pt" in conv:
            print(f"Convergence order (3-pt, dx={results[0]['dx']}->{results[-1]['dx']}):")
            print(f"  density:  {conv.get('density_order_3pt', 'n/a')}")
            print(f"  pressure: {conv.get('pressure_order_3pt', 'n/a')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
