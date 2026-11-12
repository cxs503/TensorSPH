#!/usr/bin/env bash
# Shard entry point: shard 1 -> dx = 0.014 m, shard 2 -> dx = 0.010 m (t_end = 2.0 s).
set -euo pipefail
i=${1:-1}
case "$i" in
  1) dx=0.014 ;;
  2) dx=0.010 ;;
  *) echo "unknown shard $i"; exit 0 ;;
esac
echo "shard $i : dx=$dx  host=$(hostname)  nproc=$(nproc)"
python3 benchmarks/kleefsman_obstacle/run.py --dx "$dx" --t-end 2.0 --tag "dx${dx}" --figure
