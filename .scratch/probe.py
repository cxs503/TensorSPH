"""Scratch probe: B2 dam-break front-speed coefficient at ``dx = 0.04``.

Not part of the test suite. The reproducible numbers live in
``benchmarks/dam_break/reference_audit.json``; this file keeps the ad-hoc probe
that produced ``.scratch/sim040.json`` (free-slip floor, single resolution).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "benchmarks" / "dam_break"))

from reference_audit import run_rung  # noqa: E402

record = run_rung(0.04, 0.0)
(HERE / "sim040.json").write_text(json.dumps(record, indent=1))
print(json.dumps({k: record[k] for k in ("dx", "wall_friction",
                                        "front_coefficient_c", "particles")}))
