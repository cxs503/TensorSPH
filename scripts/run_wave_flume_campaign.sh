#!/usr/bin/env bash
# CASE-WAVE-FLUME campaign: run simulations, analyse, generate report & figures.
# Produces: runs/wf_*.json, docs/reports/wave_flume_report.json,
#           docs/reports/wave_flume_numbers.json, docs/reports/wave_flume_case.md,
#           docs/figures/wave_flume/*.png
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="$PWD/src:$PYTHONPATH"

mkdir -p runs docs/reports docs/figures/wave_flume

echo "=== CASE-WAVE-FLUME: Running simulations ==="

# Primary case: T=1.0s, H/h=0.16, depth=0.25m, resolution H/dx≈8
# depth-cells=50 → dx=0.005, H/dx=8.0
python benchmarks/wave_flume/run.py \
    --case regular \
    --depth 0.25 \
    --depth-cells 50 \
    --height-ratio 0.16 \
    --wave-period 1.0 \
    --simulate-time 6.0 \
    --settle 1.0 \
    --viscosity-settle 0.02 \
    --xsph-settle 0.1 \
    --xsph 0.0 \
    --active-absorption 0.5 \
    --ramp-periods 2.0 \
    --sponge-strength 4.0 \
    --density-diffusion 0.1 \
    --sound-speed 30.0 \
    --threads 8 \
    --device cpu \
    --out runs/wf_D50.json

echo "=== Primary case done ==="

# Secondary case: T=0.8s, H/h=0.10 (different regime)
python benchmarks/wave_flume/run.py \
    --case regular \
    --depth 0.25 \
    --depth-cells 50 \
    --height-ratio 0.10 \
    --wave-period 0.8 \
    --simulate-time 6.0 \
    --settle 1.0 \
    --viscosity-settle 0.02 \
    --xsph-settle 0.1 \
    --xsph 0.0 \
    --active-absorption 0.5 \
    --ramp-periods 2.0 \
    --sponge-strength 4.0 \
    --density-diffusion 0.1 \
    --sound-speed 30.0 \
    --threads 8 \
    --device cpu \
    --out runs/wf_D50_T08.json

echo "=== Secondary case done ==="

# Coarser resolution for convergence ladder: depth-cells=30 → H/dx≈4.8
python benchmarks/wave_flume/run.py \
    --case regular \
    --depth 0.25 \
    --depth-cells 30 \
    --height-ratio 0.16 \
    --wave-period 1.0 \
    --simulate-time 6.0 \
    --settle 1.0 \
    --viscosity-settle 0.02 \
    --xsph-settle 0.1 \
    --xsph 0.0 \
    --active-absorption 0.5 \
    --ramp-periods 2.0 \
    --sponge-strength 4.0 \
    --density-diffusion 0.1 \
    --sound-speed 30.0 \
    --threads 8 \
    --device cpu \
    --out runs/wf_D30.json

echo "=== Coarser case done ==="

echo "=== Running analysis ==="
python benchmarks/wave_flume/analyse.py \
    --record runs/wf_D50.json \
    --record runs/wf_D30.json \
    --record runs/wf_D50_T08.json \
    --out docs/reports/wave_flume_report.json \
    --figures docs/figures/wave_flume \
    --start 2.0

echo "=== Generating report ==="
python benchmarks/wave_flume/make_report.py \
    --report docs/reports/wave_flume_report.json \
    --numbers docs/reports/wave_flume_numbers.json \
    --out docs/reports/wave_flume_case.md

echo "=== Campaign complete ==="
ls -la runs/wf_*.json
ls -la docs/reports/wave_flume_*
ls -la docs/figures/wave_flume/
