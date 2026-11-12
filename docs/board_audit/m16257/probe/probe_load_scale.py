"""Audit probe (m16257): sensitivity of the reported gate-tip peak to the
``load_scale`` calibration factor that m16214 derived from its FAILED V1.

m16214 divided the measured -20.37 % V1 transfer bias out of the production
run (load_scale = 1.25576) and reported only the calibrated peak.  This probe
re-runs the same coarse lattice with load_scale = 1.0 (raw transfer) so the
reported peak can be compared against the un-calibrated physics.

Usage: python probe_load_scale.py <out.json>
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
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "probe_load_scale.json")
    case0 = fg.ElasticGateCase()
    v1 = fg.verify_hydrostatic_load(case0)
    calibrated = 1.0 / (1.0 + v1["rel_error"])
    res = {"V1_rel_error": v1["rel_error"], "load_scale_from_V1": calibrated,
           "runs": []}
    for scale, tag in ((1.0, "raw_transfer"), (calibrated, "calibrated")):
        case = fg.ElasticGateCase(dx=0.0125, t_end=0.4, load_scale=scale)
        t0 = time.time()
        _state, m = fg.run(case)
        res["runs"].append({
            "tag": tag, "load_scale": scale, "dx": case.dx,
            "n_fluid": m["n_fluid"], "n_steps": m["n_steps"],
            "wall_seconds": time.time() - t0,
            "peak_tip_x": m["peak_tip_x"], "peak_time": m["peak_time"],
            "final_tip_x": m["final_tip_x"], "min_tip_x": m["min_tip_x"]})
        print(tag, res["runs"][-1], flush=True)
    raw = res["runs"][0]
    cal = res["runs"][1]
    res["peak_inflation_from_calibration"] = (
        cal["peak_tip_x"] - raw["peak_tip_x"]) / max(abs(raw["peak_tip_x"]), 1e-12)
    res["peak_rel_diff_vs_m16214"] = (
        cal["peak_tip_x"] - 0.3543333617266935) / 0.3543333617266935
    out.write_text(json.dumps(res, indent=2))
    print("summary", {k: res[k] for k in res if k != "runs"}, flush=True)
    print("json:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())