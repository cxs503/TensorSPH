#!/usr/bin/env python3
"""Minimal slamming demonstration for quick verification.

This runs a very coarse, short simulation to demonstrate the pipeline works.
For production results, use benchmarks/slamming/run.py on the cluster.
"""

import json
import math
import time
from pathlib import Path

import torch

from tensorsph.slamming import SlammingConfig, SlammingImpact
from tensorsph.wedge_entry import wagner_c_coeff

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "slamming_demo"


def run_demo():
    """Run a minimal slamming case and save artifacts."""
    print("=" * 70)
    print("TensorSPH Slamming Demo (minimal verification)")
    print("=" * 70)
    
    # Minimal configuration for quick execution
    # Need at least 4x4 fluid particles: half_width/dx >= 4, depth/dx >= 4
    cfg = SlammingConfig(
        beta_deg=20.0,
        velocity=2.0,
        dx=5.0e-3,  # Coarse but valid
        half_width=0.04,  # nx = 8
        depth=0.025,  # ny = 5
        headroom=0.02,
        c_end=0.012,
        frames=6,
        delta_sph=0.1,
        mls_density=True,
        initial_gap_dx=0.0,
    )
    cfg.validate()
    
    print(f"\nConfiguration:")
    print(f"  β = {cfg.beta_deg}°")
    print(f"  V = {cfg.velocity} m/s")
    print(f"  dx = {cfg.dx*1e3} mm")
    print(f"  half_width = {cfg.half_width} m")
    print(f"  depth = {cfg.depth} m")
    print(f"  frames = {cfg.frames}")
    print(f"  δ-SPH = {cfg.delta_sph}")
    print(f"  MLS = {cfg.mls_density}")
    
    print("\nInitializing simulation...")
    t0 = time.time()
    sim = SlammingImpact(cfg)
    print(f"  n_fluid = {sim.n_fluid}")
    print(f"  n_wall = {sim.wall_local.shape[0]}")
    
    print("\nRunning simulation...")
    series = sim.run()
    runtime = time.time() - t0
    print(f"  Completed in {runtime:.2f} s")
    print(f"  Steps = {sim.steps}")
    
    # Extract key metrics
    beta = cfg.beta
    c_s_wagner = math.pi ** 2 / math.tan(beta)
    
    post = [s for s in series if s["t"] > 0.0 and s["c"] > 0.0]
    if not post:
        print("\n⚠ No post-contact samples (wedge didn't reach water)")
        return None
    
    c_s_max = max(s["c_s"] for s in post)
    c_s_err = (c_s_max - c_s_wagner) / c_s_wagner
    
    noise_vals = [s["noise"] for s in post if s["noise"] == s["noise"]]
    shape_vals = [s["shape_l2"] for s in post if s["shape_l2"] == s["shape_l2"]]
    
    noise_rms = (sum(n*n for n in noise_vals) / len(noise_vals))**0.5 if noise_vals else float("nan")
    shape_l2_mean = sum(shape_vals) / len(shape_vals) if shape_vals else float("nan")
    
    mass_drift_max = max(s["mass_drift"] for s in post)
    
    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"  C_s,max = {c_s_max:.4f}")
    print(f"  C_s,Wagner = {c_s_wagner:.4f}")
    print(f"  Relative error = {c_s_err*100:.2f}%")
    print(f"  Noise RMS = {noise_rms:.4f}")
    print(f"  Shape L2 = {shape_l2_mean:.4f}")
    print(f"  Mass drift = {mass_drift_max:.2e}")
    
    # Save artifacts
    ART.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "demo": True,
        "config": {
            "beta_deg": cfg.beta_deg,
            "velocity": cfg.velocity,
            "dx": cfg.dx,
            "frames": cfg.frames,
            "delta_sph": cfg.delta_sph,
            "mls_density": cfg.mls_density,
        },
        "derived": {
            "c_s_wagner": c_s_wagner,
            "runtime_s": runtime,
            "steps": sim.steps,
            "n_fluid": sim.n_fluid,
        },
        "metrics": {
            "c_s_max": c_s_max,
            "c_s_err_rel": c_s_err,
            "noise_rms": noise_rms,
            "shape_l2_mean": shape_l2_mean,
            "mass_drift_max": mass_drift_max,
        },
        "series": series,
    }
    
    out_path = ART / "demo_result.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n✓ Artifacts saved to {out_path}")
    
    # Quick criterion check
    print("\n" + "=" * 70)
    print("Criterion Check (demo thresholds):")
    print("=" * 70)
    checks = [
        ("C_s,max vs Wagner ≤ 30%", abs(c_s_err) <= 0.30),
        ("Noise RMS ≤ 20%", noise_rms <= 0.20 or math.isnan(noise_rms)),
        ("Shape L2 ≤ 30%", shape_l2_mean <= 0.30 or math.isnan(shape_l2_mean)),
        ("Mass drift ≤ 1e-3", mass_drift_max <= 1e-3),
    ]
    
    all_pass = True
    for name, ok in checks:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}: {name}")
        all_pass = all_pass and ok
    
    if all_pass:
        print("\n✓ All demo criteria passed")
    else:
        print("\n⚠ Some criteria failed (expected for coarse demo)")
    
    return payload


if __name__ == "__main__":
    run_demo()
