"""Rayleigh-Plateau ladder: analytic anchors, curvature anchors, column attempt.

Produces ``benchmarks/rayleigh_plateau/report.json`` with every number the
verification report quotes, so nothing is transcribed by hand.

Usage (a full run takes tens of minutes on a loaded control plane -- the
parameter sweep is meant for the cluster queue):

    python benchmarks/rayleigh_plateau/run.py --quick          # CI-scale
    python benchmarks/rayleigh_plateau/run.py                  # full
    python benchmarks/rayleigh_plateau/run.py --sweep          # kR sweep
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorsph import rayleigh_plateau as rp                      # noqa: E402
from tensorsph.threed import liquid_column as lc                 # noqa: E402
from tensorsph.threed.neighbors3 import periodic_pairs           # noqa: E402

OUT = Path(__file__).resolve().parent


def analytic_block() -> dict:
    """Rayleigh/Lamb/Tomotika anchors, all closed form."""
    sigma, rho, radius, mu = 0.072, 1000.0, 0.006, 1.0e-3
    x_star = rp.most_unstable_wavenumber()
    lambda_star = rp.most_unstable_wavelength(1.0)
    k = x_star / radius
    omega = rp.dispersion_rate(sigma, rho, radius, k)
    # small-k limit of the same dispersion (slender jet)
    k_small = 0.01 / radius
    omega_small = rp.dispersion_rate(sigma, rho, radius, k_small)
    slender = math.sqrt(rp.slender_jet_dispersion(sigma, rho, radius, k_small))
    return {
        "most_unstable_kR": x_star,
        "most_unstable_kR_literature": 0.697,
        "most_unstable_kR_error": x_star / 0.697 - 1.0,
        "naive_kR_without_bessel": rp.naive_wavenumber(),
        "most_unstable_lambda_over_R": lambda_star,
        "most_unstable_lambda_over_R_literature": 9.02,
        "most_unstable_lambda_over_R_error": lambda_star / 9.02 - 1.0,
        "omega_at_x_star": omega,
        "capillary_time": rp.capillary_time(sigma, rho, radius),
        "omega_times_capillary_time": omega * rp.capillary_time(sigma, rho, radius),
        "ohnesorge": rp.ohnesorge(sigma, rho, radius, mu),
        "inviscid_anchor_valid": rp.ohnesorge(sigma, rho, radius, mu) < 0.05,
        "slender_jet_relative_difference": omega_small / slender - 1.0,
        "planar_strip_omega2": rp.planar_strip_dispersion(sigma, rho, radius, k),
        "planar_strip_stable": rp.planar_strip_dispersion(sigma, rho, radius, k) > 0.0,
        "dispersion_curve": [
            {"kR": x / 100.0,
             "omega": rp.dispersion_rate(sigma, rho, radius, x / 100.0 / radius)
             if x < 100.0 else 0.0,
             "omega_naive": math.sqrt(max(
                 sigma / (rho * radius ** 3) * (x / 100.0) * (1 - (x / 100.0) ** 2), 0.0))}
            for x in range(5, 101, 5)
        ],
    }


def curvature_block(resolutions) -> list:
    """``kappa R`` of a static cylinder for every estimator."""
    rows = []
    for rdx in resolutions:
        for estimator in ("profile", "mls", "axisym"):
            info = lc.static_cylinder_curvature(radius=rdx * 1e-3, dx=1e-3,
                                                curvature=estimator)
            rows.append({"radius_over_dx": rdx, "estimator": estimator,
                         "particles": info["particles"],
                         "relative_error": info["relative_error_mean"]})
    return rows


def initial_condition_block(resolutions) -> list:
    """Recovered mode amplitude of the initial condition (should be ``eps``)."""
    rows = []
    for rdx in resolutions:
        cfg = lc.ColumnConfig(radius=rdx * 1e-3, dx=1e-3, kR=0.6971, epsilon=0.02)
        pos, mass, box = lc.build_column(cfg)
        pairs = periodic_pairs(pos, box, cfg.cutoff)
        probe = lc.surface_radius_profile(pos, mass, cfg.h, pairs, rho0=cfg.rho0,
                                          box=box, nbin=cfg.nz)
        calib = cfg.radius / float(probe["radius"].mean())
        prof = lc.surface_radius_profile(pos, mass, cfg.h, pairs, rho0=cfg.rho0,
                                         box=box, nbin=cfg.nz, calibration=calib)
        k = 2.0 * math.pi / cfg.wavelength
        radius = prof["radius"]
        ref = float(radius.mean())
        rows.append({
            "radius_over_dx": rdx,
            "particles": int(pos.shape[0]),
            "epsilon_target": cfg.epsilon,
            "epsilon_measured": float(abs(lc.mode_amplitude(radius, prof["dz"], k))) / ref,
            "null_mode": float(abs(lc.mode_amplitude(radius, prof["dz"], 3.5 * k))) / ref,
            "shell_radius_over_R": float(probe["radius"].mean()) / cfg.radius,
            "radius_calibration": calib,
            "kR_actual": cfg.kR_actual,
            "nz": cfg.nz,
            "mass_total_over_target": float(mass.sum())
            / (cfg.rho0 * math.pi * cfg.radius ** 2 * cfg.wavelength),
        })
    return rows


def run_block(radius_over_dx: float, steps: int, sample_every: int,
              sound_speed: float, closure: bool) -> dict:
    """One column run, with the stability diagnostics the FAIL is based on."""
    cfg = lc.ColumnConfig(radius=radius_over_dx * 1e-3, dx=1e-3, kR=0.6971,
                          epsilon=0.02, sound_speed=sound_speed,
                          renormalise_density=closure, surface_pressure=closure)
    start = time.time()
    result = lc.run_column(cfg, steps=steps, sample_every=sample_every)
    blob = result.to_dict()
    blob.update({
        "steps": steps,
        "sample_every": sample_every,
        "wall_seconds": time.time() - start,
        "closure": closure,
        "times": result.times,
        "amplitudes": result.amplitudes,
        "null_amplitudes": result.null_amplitudes,
        "harmonic_amplitudes": result.harmonic_amplitudes,
        "density_error_series": result.density_error,
        "volume_error_series": result.volume_error,
        "max_velocity_series": result.max_velocity,
        "stability": stability_verdict(result),
    })
    return blob


def stability_verdict(result) -> dict:
    """Predeclared stability checks for a liquid-column run.

    ``pass`` needs all three: (i) no velocity blow-up beyond 10x the capillary
    velocity ``sqrt(sigma/(rho R))``, (ii) max density error below 10 %
    (WCSPH keeps 1 %-scale density errors; ten times that means the pressure
    field has lost meaning), (iii) the reconstructed body volume within 1 % of
    the initial cylinder.
    """
    cfg = result.cfg
    u_cap = math.sqrt(cfg.sigma / (cfg.rho0 * cfg.radius))
    max_u = max(result.max_velocity) if result.max_velocity else float("nan")
    max_rho = max(result.density_error) if result.density_error else float("nan")
    vol = result.volume_error[-1] if result.volume_error else float("nan")
    checks = {
        "max_velocity_over_capillary": max_u / u_cap,
        "max_density_error": max_rho,
        "final_volume_error": vol,
        "velocity_ok": bool(max_u <= 10.0 * u_cap),
        "density_ok": bool(max_rho < 0.10),
        "volume_ok": bool(abs(vol) < 0.01),
    }
    checks["pass"] = bool(checks["velocity_ok"] and checks["density_ok"]
                          and checks["volume_ok"])
    return checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="CI scale: coarse resolution, few steps")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sound-speed", type=float, default=3.0)
    parser.add_argument("--closure", action="store_true",
                        help="enable Shepard density + free-surface pressure closure")
    parser.add_argument("--sweep", action="store_true",
                        help="run the kR sweep (slow; cluster job)")
    parser.add_argument("--out", default=str(OUT / "report.json"))
    args = parser.parse_args(argv)

    if args.quick:
        resolutions = (3, 4)
        steps, sample_every = 80, 20
        run_resolution = 3
    else:
        resolutions = (4, 6, 8)
        steps, sample_every = 600, 25
        run_resolution = 4

    report = {
        "case": "Rayleigh-Plateau liquid column (TensorSPH, 3-D periodic)",
        "environment": {
            "torch": torch.__version__,
            "dtype": "float64",
            "device": "cpu",
            "threads": torch.get_num_threads(),
        },
        "analytic": analytic_block(),
        "curvature": curvature_block(resolutions),
        "initial_condition": initial_condition_block(resolutions),
    }

    if args.steps is not None:
        steps = args.steps
    report["column_run"] = run_block(run_resolution, steps, sample_every,
                                     args.sound_speed, args.closure)

    if args.sweep and not args.quick:
        sweep = []
        for kr in (0.35, 0.50, 0.6971, 0.85):
            cfg = lc.ColumnConfig(radius=4e-3, dx=1e-3, kR=kr, epsilon=0.02,
                                  sound_speed=args.sound_speed,
                                  renormalise_density=args.closure,
                                  surface_pressure=args.closure)
            result = lc.run_column(cfg, steps=steps, sample_every=sample_every)
            blob = result.to_dict()
            blob["stability"] = stability_verdict(result)
            sweep.append(blob)
        report["kR_sweep"] = sweep

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    print(json.dumps({k: report["column_run"][k]
                      for k in ("omega_measured", "omega_analytic", "omega_error",
                                "growth_fit_r2", "max_density_error",
                                "final_volume_error")}, indent=2))
    print("stability:", json.dumps(report["column_run"]["stability"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
