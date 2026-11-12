#!/usr/bin/env python3
"""m15815 board audit step 1 — inspect + grade the committed m15493 run records.

Reads (read-only) /root/tp-agent/work/m15493/runs/*.json, prints the config,
the diagnostics (blow-up screen), and the grade of each record using the
subject's own committed grading chain (PYTHONPATH=subject/src).
"""
import json
import math
import sys
from pathlib import Path

SUBJ = Path("/root/tp-agent/work/m15493")
OUT = Path("/root/tp-agent/work/m15815/runs/m15815")
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SUBJ / "src"))

from tensorsph.wave_flume_grade import (  # noqa: E402
    grade_regular_record, wave_height_target, laminar_boundary_layer_damping,
)

recs = {}
for name in ("wf_local_D20", "wf_quick"):
    p = SUBJ / "runs" / (name + ".json")
    rec = json.loads(p.read_text())
    recs[name] = rec
    cfg, spec, diag = rec["config"], rec["spec"], rec["diagnostics"]
    print("=" * 78)
    print(name, "  size=%.0f kB" % (p.stat().st_size / 1024))
    print(" config:", json.dumps({k: cfg[k] for k in cfg if k not in ("case",)}, default=str))
    print(" spec  :", json.dumps(spec, default=str))
    print(" t_end :", rec["times"][-1], " samples:", len(rec["times"]), " probes:", len(rec["probe_x"]))
    print(" diag  :", json.dumps(diag, default=str)[:600])
    print(" log   :", (SUBJ / "runs" / (name + ".log")).read_text().strip())
    g = grade_regular_record(rec, start=None)
    print(" grade passed=%s worst_err=%.4f" % (g.get("passed"), g.get("worst_error_vs_target", float("nan"))))
    for gg in g.get("gauges", []):
        print("   x=%.3f  H=%.6g  err=%+.3f%%  dry=%.3f" % (
            gg["x"], gg.get("height", float("nan")),
            100.0 * gg.get("error_vs_target", float("nan")), gg.get("dry_fraction", -1)))
    print(" attenuation:", json.dumps(g.get("attenuation"), default=str)[:300])
    print(" dispersion :", json.dumps(g.get("dispersion"), default=str)[:400])
    recs[name] = {"config": cfg, "spec": spec, "diag": diag, "grade": g}

(OUT / "grade_committed_records.json").write_text(json.dumps(recs, indent=1, default=str))
print("\nwritten", OUT / "grade_committed_records.json")