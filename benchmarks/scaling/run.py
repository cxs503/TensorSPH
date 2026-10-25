"""GPU scaling benchmark for TensorSPH.

Measures wall-clock time per integration step as a function of particle count,
on both CPU and CUDA. Locates the GPU speedup crossover point N* where CUDA
becomes faster than CPU.

The benchmark reuses the droplet-impact case (``tensorsph.droplet_impact.run``)
with varying resolution ``dx``. Physical parameters are held constant, so
dimensionless numbers (We, Fr, Ma) are preserved; only the particle count changes.

Usage::

    # Quick test: 3 sizes on CPU only
    python benchmarks/scaling/run.py --sizes 0.015,0.008,0.006 --devices cpu

    # Full benchmark: 5 sizes on CPU and CUDA
    python benchmarks/scaling/run.py --sizes 0.015,0.008,0.006,0.003,0.002

    # With JSON output and plot
    python benchmarks/scaling/run.py --json-out artifacts/scaling/report.json --plot

Output is a JSON report with per-size timing on each device, plus an optional
log-log plot of wall-clock time vs particle count.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from tensorsph.droplet_impact import (
    Config,
    SimulationUnstableError,
    make_particles,
    run,
    resolve_device,
    suggested_dt,
)


def benchmark_one(dx: float, device: str, steps: int, output: Path) -> dict:
    """Run a single (dx, device) configuration and return timing metrics."""
    cfg = Config(
        dx=dx,
        device=device,
        steps=steps,
        output_every=steps,  # minimize I/O overhead
        fail_fast=True,
    )
    # Scale dt with dx to maintain the same Courant number
    cfg.dt = suggested_dt(cfg) * 0.8
    
    dev = resolve_device(device)
    pos, _ = make_particles(cfg, dev)
    n_particles = pos.shape[0]
    
    # Reset peak memory stats (GPU only)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    
    start = time.perf_counter()
    try:
        rows = run(cfg, output, plot=False)
    except SimulationUnstableError as exc:
        return {
            "dx": dx,
            "n_particles": n_particles,
            "device": str(dev),
            "status": "unstable",
            "error": str(exc),
        }
    elapsed = time.perf_counter() - start
    
    result = {
        "dx": dx,
        "n_particles": n_particles,
        "device": str(dev),
        "status": "success",
        "wall_clock_s": elapsed,
        "steps": steps,
        "steps_per_second": steps / elapsed,
        "particles_per_second": n_particles * steps / elapsed,
    }
    
    # GPU memory
    if dev.type == "cuda":
        result["peak_vram_bytes"] = torch.cuda.max_memory_allocated(dev)
    
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", default="0.020,0.009,0.006,0.003,0.002",
        help="comma-separated dx values (default: 0.020,0.009,0.006,0.003,0.002 → ~1k,5k,10k,50k,100k)",
    )
    parser.add_argument(
        "--devices", default="cpu,cuda",
        help="comma-separated devices (default: cpu,cuda)",
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="integration steps per run (default: 50)",
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="warmup runs per (dx, device) to exclude (default: 1)",
    )
    parser.add_argument(
        "--json-out", default=None,
        help="write JSON report to this path",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/scaling",
        help="base directory for run artifacts",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="generate log-log plot (requires matplotlib)",
    )
    args = parser.parse_args()
    
    dx_values = [float(x) for x in args.sizes.split(",")]
    devices = [d.strip() for d in args.devices.split(",")]
    
    # Check CUDA availability
    if "cuda" in devices and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, skipping CUDA runs",
              file=sys.stderr)
        devices.remove("cuda")
    
    all_results = []
    out_base = Path(args.output_dir)
    crossover_n = None
    
    for device in devices:
        print(f"\n{'='*70}")
        print(f"Device: {device}")
        print(f"{'='*70}")
        
        for dx in dx_values:
            print(f"\n--- dx={dx:.4f} ---")
            
            # Warmup runs (not counted)
            for w in range(args.warmup):
                output = out_base / f"dx{dx:.4f}_{device}_warmup{w}"
                benchmark_one(dx, device, args.steps, output)
                print(f"  warmup {w+1}/{args.warmup} done")
            
            # Measured run
            output = out_base / f"dx{dx:.4f}_{device}"
            result = benchmark_one(dx, device, args.steps, output)
            all_results.append(result)
            
            if result["status"] == "success":
                print(f"  N={result['n_particles']}: "
                      f"{result['wall_clock_s']:.3f}s "
                      f"({result['steps_per_second']:.1f} steps/s, "
                      f"{result['particles_per_second']:.0f} particles/s)")
                if "peak_vram_bytes" in result:
                    vram_mb = result["peak_vram_bytes"] / 1e6
                    print(f"  peak VRAM: {vram_mb:.1f} MB")
            else:
                print(f"  FAILED: {result['error']}")
    
    # Summary table
    print("\n" + "="*90)
    print("Summary")
    print("="*90)
    header = f"{'N':<8} {'device':<8} {'time(s)':<10} {'steps/s':<12} {'particles/s':<15}"
    print(header)
    print("-"*90)
    
    for r in sorted(all_results, key=lambda x: (x['n_particles'], x['device'])):
        if r["status"] == "success":
            print(f"{r['n_particles']:<8d} {r['device']:<8} "
                  f"{r['wall_clock_s']:<10.3f} {r['steps_per_second']:<12.1f} "
                  f"{r['particles_per_second']:<15.0f}")
        else:
            print(f"{r['n_particles']:<8d} {r['device']:<8} FAILED")
    
    # GPU speedup analysis
    cpu_results = {r['dx']: r for r in all_results 
                   if r['device'] == 'cpu' and r['status'] == 'success'}
    cuda_results = {r['dx']: r for r in all_results 
                    if r['device'].startswith('cuda') and r['status'] == 'success'}
    
    if cpu_results and cuda_results:
        print("\n--- GPU Speedup ---")
        print(f"{'N':<8} {'CPU(s)':<10} {'GPU(s)':<10} {'Speedup':<10}")
        print("-"*50)
        
        for dx in dx_values:
            if dx in cpu_results and dx in cuda_results:
                cpu_r = cpu_results[dx]
                cuda_r = cuda_results[dx]
                speedup = cpu_r['wall_clock_s'] / cuda_r['wall_clock_s']
                print(f"{cpu_r['n_particles']:<8d} "
                      f"{cpu_r['wall_clock_s']:<10.3f} "
                      f"{cuda_r['wall_clock_s']:<10.3f} "
                      f"{speedup:<10.2f}x")
                
                if crossover_n is None and speedup > 1.0:
                    crossover_n = cpu_r['n_particles']
        
        if crossover_n:
            print(f"\n✓ GPU crossover point N* ≈ {crossover_n}")
        else:
            print("\n✗ No crossover found (CPU faster at all tested sizes)")
    
    # JSON output
    if args.json_out:
        report = {
            "benchmark": "gpu_scaling",
            "case": "droplet_impact_2d",
            "steps_per_run": args.steps,
            "warmup_runs": args.warmup,
            "crossover_n": crossover_n,
            "results": all_results,
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report: {args.json_out}")
    
    # Plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for device in devices:
                device_results = [r for r in all_results 
                                 if r['device'].startswith(device) 
                                 and r['status'] == 'success']
                if not device_results:
                    continue
                ns = [r['n_particles'] for r in device_results]
                times = [r['wall_clock_s'] for r in device_results]
                ax.loglog(ns, times, 'o-', label=device, linewidth=2, markersize=8)
            
            # Annotate crossover
            if crossover_n:
                ax.axvline(crossover_n, color='gray', linestyle='--', alpha=0.5,
                          label=f'N* ≈ {crossover_n}')
            
            ax.set_xlabel("Particle count N", fontsize=12)
            ax.set_ylabel(f"Wall-clock time per {args.steps} steps (s)", fontsize=12)
            ax.set_title("TensorSPH GPU Scaling: Droplet Impact Case", fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, which="both", ls=":", alpha=0.5)
            ax.tick_params(labelsize=10)
            
            plot_path = Path(args.output_dir) / "scaling.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {plot_path}")
            plt.close(fig)
        except ImportError:
            print("WARNING: matplotlib not available, skipping plot",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
