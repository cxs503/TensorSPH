#!/usr/bin/env python3
"""Ultra-minimal slamming test for pipeline verification.

This is NOT a production run - it's a smoke test to verify the pipeline works.
Uses extremely coarse resolution and short duration.
"""

import json
import math
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from tensorsph.slamming import SlammingConfig, SlammingImpact

def main():
    print("Ultra-minimal slamming smoke test")
    print("=" * 60)
    
    # Absolute minimum: just enough particles to not crash
    # Need nx >= 4, ny >= 4
    cfg = SlammingConfig(
        beta_deg=30.0,  # Larger angle = faster wetting
        velocity=1.0,
        dx=10.0e-3,  # Very coarse: 10mm
        half_width=0.05,  # nx = 5
        depth=0.05,  # ny = 5
        headroom=0.03,
        c_end=0.01,  # Short duration
        frames=3,  # Minimal frames
        delta_sph=0.1,
        mls_density=True,
        initial_gap_dx=0.0,
    )
    
    print(f"Config: β={cfg.beta_deg}°, V={cfg.velocity} m/s, dx={cfg.dx*1e3} mm")
    print(f"Domain: {cfg.half_width*2}m x {cfg.depth}m")
    print(f"Expected particles: ~{int(2*cfg.half_width/cfg.dx) * int(cfg.depth/cfg.dx)}")
    
    try:
        cfg.validate()
        print("\nInitializing...")
        t0 = time.time()
        sim = SlammingImpact(cfg)
        print(f"  n_fluid={sim.n_fluid}, n_wall={sim.wall_local.shape[0]}")
        
        print("\nRunning (timeout 45s)...")
        series = sim.run()
        runtime = time.time() - t0
        
        print(f"\n✓ Completed in {runtime:.2f}s")
        print(f"  Steps: {sim.steps}")
        print(f"  Frames: {len(series)}")
        
        # Check if we got any post-contact data
        post = [s for s in series if s.get("c", 0) > 0]
        if post:
            print(f"  Post-contact frames: {len(post)}")
            print(f"  Max c: {max(s['c'] for s in post):.4f} m")
            print(f"  Max C_s: {max(s['c_s'] for s in post):.2f}")
        else:
            print("  ⚠ No post-contact data (wedge didn't reach water)")
        
        # Save minimal artifact
        art_dir = Path(__file__).resolve().parents[1] / "artifacts" / "smoke_test"
        art_dir.mkdir(parents=True, exist_ok=True)
        
        artifact = {
            "smoke_test": True,
            "config": {
                "beta_deg": cfg.beta_deg,
                "velocity": cfg.velocity,
                "dx": cfg.dx,
            },
            "runtime_s": runtime,
            "steps": sim.steps,
            "n_frames": len(series),
            "n_post_contact": len(post),
            "status": "success" if post else "no_contact",
        }
        
        out_path = art_dir / "smoke_test.json"
        out_path.write_text(json.dumps(artifact, indent=2))
        print(f"\n✓ Artifact saved: {out_path}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
