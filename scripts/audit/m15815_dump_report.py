#!/usr/bin/env python3
"""Dump the m15493 mission report markdown from the platform registry."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m15815_registry_probe import q

rows = q("SELECT id,title,scenario_key,mission_id,created_at,markdown,figures FROM tensor_reports "
         "WHERE mission_id=15493 ORDER BY id")
out = Path("/root/tp-agent/work/m15815/runs/m15815/")
out.mkdir(parents=True, exist_ok=True)
for r in rows:
    print("== report", r["id"], r["title"])
    print(r["markdown"])
    print("---- figures:", str(r.get("figures"))[:400])
(out / "m15493_report.json").write_text(json.dumps([dict(r, markdown=r["markdown"][:200]) for r in rows],
                                                   indent=1, default=str, ensure_ascii=False))
for r in rows:
    (out / ("m15493_report_%d.md" % r["id"])).write_text(r["markdown"])