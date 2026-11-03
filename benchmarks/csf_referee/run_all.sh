#!/bin/sh
# Score every unmerged surface-tension candidate on the referee instrument.
set -e
OUT=benchmarks/csf_referee/results
mkdir -p "$OUT"
for cand in "$@"; do
  sh benchmarks/csf_referee/materialize.sh "$cand" >/dev/null
  PYTHONPATH=/tmp/referee/$cand timeout 1200 python benchmarks/csf_referee/score.py \
      --candidate "$cand" --out "$OUT/$cand.json" --mode both \
      > "$OUT/$cand.log" 2>&1 || echo "[$cand] SCORE FAILED (see $OUT/$cand.log)"
done
