#!/usr/bin/env bash
# Reproduce the IN13W2-N07 benchmark run: environment snapshot, the anchor set,
# a log of the run, and the evidence bundle that audits the artefacts.
#
#   bash benchmarks/ice_thermal_multiphysics/run_campaign.sh [OUT_DIR]
#
# OUT_DIR defaults to runs/IN13W2-N07 (a *tracked* path: the campaign gate reads
# the numbers out of the repository, not out of a developer's scratch dir).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
OUT="${1:-runs/IN13W2-N07}"
mkdir -p "$OUT"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

python - "$OUT/environment.json" <<'PY'
import json, platform, sys
try:
    import torch
    torch_version, threads = torch.__version__, torch.get_num_threads()
except Exception as exc:                      # pragma: no cover - reporting only
    torch_version, threads = f"unavailable: {exc}", None
json.dump({"python": platform.python_version(), "torch": torch_version,
           "platform": platform.platform(), "machine": platform.machine(),
           "processor": platform.processor(), "torch_threads": threads,
           "git": None}, open(sys.argv[1], "w"), indent=2)
PY
{
  echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) uname: $(uname -a)"
  echo "# PYTHONPATH=$PYTHONPATH OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo "# git: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
} > "$OUT/run.log"
python benchmarks/ice_thermal_multiphysics/run.py --out "$OUT" 2>&1 | tee -a "$OUT/run.log"
# the canonical numbers live next to the harness (repository convention: every
# benchmarks/<name>/ ships its report), the run dir keeps the log + figures
cp "$OUT/results.json" benchmarks/ice_thermal_multiphysics/validation_report.json
python benchmarks/ice_thermal_multiphysics/evidence.py --run-dir "$OUT" 2>&1 | tee -a "$OUT/run.log"