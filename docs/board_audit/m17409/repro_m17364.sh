#!/usr/bin/env bash
# m17409 board audit — verified reproduction of m17364 (XVAL-SPH-COMPRESSION-BASED-DENSITY-REGULARIZATION)
#
# Why this wrapper exists: the *delivered* repro_cmd
#     python3 benchmarks/xval_sphinxsys_compression/run.py --quick
# crashes at argument-parse time in commit fef547d (agent/m17364):
#     run.py:222  TypeError: unsupported operand type(s) for /: 'str' and 'str'
# because the argparse default is built as ``str("xval_output" / "xval_sphinxsys_compression" / "records.json")``
# i.e. str/str before str().  The argparse default is evaluated unconditionally, so *every*
# invocation of the delivered driver fails — the shipped artefacts could not have been produced
# by the shipped code.  This wrapper applies the one-line fix inside a scratch copy and then runs
# the documented command, so the audit can state whether the numbers are reproducible at all.
#
# Run:  bash docs/board_audit/m17409/repro_m17364.sh
set -euo pipefail

SRC=${SRC:-/root/tp-agent/work/m17364}
W=$(cd "$(dirname "$0")/../.." && pwd)/artifacts/repro_m17364
TREE=$W/tree

mkdir -p "$W"
if [ ! -d "$TREE/src" ]; then
  mkdir -p "$TREE/benchmarks"
  cp -a "$SRC/src" "$TREE/"
  cp -a "$SRC/benchmarks/xval_sphinxsys_compression" "$TREE/benchmarks/"
fi

python3 - "$TREE" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]) / "benchmarks/xval_sphinxsys_compression/run.py"
s = p.read_text()
bad = '    ap.add_argument("--out", default=str("xval_output" /\n                                         "xval_sphinxsys_compression" /\n                                         "records.json"))'
good = '    ap.add_argument("--out", default="xval_output/xval_sphinxsys_compression/records.json")'
if bad in s:
    p.write_text(s.replace(bad, good))
    print("[repro] applied 1-line fix to the delivered argparse default")
elif good not in s:
    raise SystemExit("[repro] delivered run.py changed; re-derive the patch before trusting this wrapper")
PY

cd "$TREE"
python3 benchmarks/xval_sphinxsys_compression/run.py --quick --out "$W/records_repro.json"
python3 benchmarks/xval_sphinxsys_compression/figures.py \
        --records "$W/records_repro.json" --figdir "$W/figures" >/dev/null

echo "--- sha256: audit re-run vs delivered artefacts (identity ⇒ deterministic reproduction) ---"
sha256sum "$W"/figures/*.png "$SRC"/artifacts/xval_sphinxsys_compression/figures/*.png \
  | awk '{print substr($1,1,16), $2}'