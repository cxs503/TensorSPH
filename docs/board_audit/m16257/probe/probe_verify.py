"""Audit probe (m16257): reproduce m16214's V1/V2 verification claims.

Runs the shipped ladder as-is, then a falsification test on V2 (heavier
velocity damping => true static equilibrium) to separate a harness defect
(under-relaxation) from a rod-stiffness defect.

Usage: python probe_verify.py <out.json>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from tensorsph import fsi_elastic_gate as fg  # noqa: E402


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "probe_verify.json")
    case = fg.ElasticGateCase()
    res: dict = {}

    t0 = time.time()
    v1 = fg.verify_hydrostatic_load(case)
    v1["wall_seconds"] = time.time() - t0
    v1["verdict"] = "PASS" if abs(v1["rel_error"]) < 0.10 else "FAIL"
    res["V1_as_shipped"] = v1
    print("V1", {k: round(v, 5) if isinstance(v, float) else v for k, v in v1.items()})

    t0 = time.time()
    v2 = fg.verify_cantilever(case)  # shipped: 30000 steps, damping 0.995
    v2["wall_seconds"] = time.time() - t0
    v2["verdict"] = "PASS" if abs(v2["rel_error"]) < 0.10 else "FAIL"
    res["V2_as_shipped"] = v2
    print("V2", {k: round(v, 6) if isinstance(v, float) else v for k, v in v2.items()})

    # falsification: heavy per-step damping (overdamped => static state in a few
    # hundred steps) and 10x the shipped horizon.  If the deflection moves to
    # the analytic value, the shipped FAIL was a relaxation defect; if it stays
    # at ~1/13 of the analytic value, the rod discretisation is too stiff.
    for damp, nsteps, tag in ((0.90, 20000, "V2_overdamped"),
                              (1.00, 200000, "V2_undamped_long")):
        t0 = time.time()
        v = fg.verify_cantilever(case, n_steps=nsteps, damping=damp)
        v["wall_seconds"] = time.time() - t0
        v["verdict"] = "PASS" if abs(v["rel_error"]) < 0.10 else "FAIL"
        res[tag] = v
        print(tag, {k: round(x, 6) if isinstance(x, float) else x
                    for k, x in v.items()})

    out.write_text(json.dumps(res, indent=2))
    print("json:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())