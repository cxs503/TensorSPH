#!/bin/sh
# Full WCSPH verification-chain measurement, run inside one checkout.
# Every step writes into artifacts/wcsph_chain/; the last two steps turn those
# artefacts into report.json and the generated document.
set -e
cd "$(dirname "$0")/../.."
R="python3 benchmarks/wcsph_chain/run.py"
$R pair --steps 2000 --cfl 0.1
for dx in 0.04 0.02 0.01; do $R dam2d --dx $dx --seconds 0.5; done
$R dam3d --dx 0.02 --tank-length 1.2 --width 0.4 --seconds 0.5
for f in 0.5 1.0 1.5; do $R slosh --freq $f --dx 0.02 --seconds 5.0 --settle-fraction 0.5; done
$R tests
$R assemble
python3 benchmarks/wcsph_chain/make_doc.py
echo PIPELINE_DONE
