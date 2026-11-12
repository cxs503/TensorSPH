#!/usr/bin/env python3
"""Read-only probe of the platform registry for the CASE-WAVE-FLUME audit."""
import json
import sys
from pathlib import Path

DSN = None
for line in open("/root/tensor-ecosystem/TensorPlatform/.env"):
    if line.startswith("DATABASE_URL="):
        DSN = line.split("=", 1)[1].strip().replace("postgresql+psycopg://", "postgresql://")
        break


def q(sql, params=None):
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(DSN, row_factory=dict_row) as c, c.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchall()


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "cases"
    if what == "cases":
        rows = q("SELECT id,anchor_slug,case_name,ref_source,metric,tolerance,repo,status,passed_at "
                 "FROM tensor_benchmark_cases WHERE anchor_slug=%s ORDER BY id", ("CASE-WAVE-FLUME",))
        print(json.dumps(rows, indent=1, default=str, ensure_ascii=False))
        print("--- total rows for anchor:", len(rows))
        allrows = q("SELECT anchor_slug,count(*) n FROM tensor_benchmark_cases GROUP BY 1 ORDER BY 1")
        print("all anchors:", [(r["anchor_slug"], r["n"]) for r in allrows])
    elif what == "mission":
        rows = q("SELECT * FROM tensor_dev_missions WHERE id=%s", (15493,))
        print(json.dumps(rows, indent=1, default=str, ensure_ascii=False)[:6000])
    elif what == "reports":
        rows = q("SELECT id,title,created_at,mission_id,scenario_key,length(markdown) ml,"
                 " LEFT(markdown,200) head FROM tensor_reports WHERE mission_id=15493 ORDER BY id")
        print(json.dumps(rows, indent=1, default=str, ensure_ascii=False)[:6000])
    elif what == "verdicts":
        rows = q("SELECT id,repo,scope,mission_id,level,verdict_by,gate_state,gate_level,created_at,"
                 " LEFT(evidence::text,300) ev FROM tensor_maturity_verdicts "
                 "WHERE scope ILIKE %s ORDER BY id", ("%WAVE-FLUME%",))
        print(json.dumps(rows, indent=1, default=str, ensure_ascii=False)[:6000])
    elif what == "verdict_cols":
        rows = q("SELECT column_name,data_type FROM information_schema.columns "
                 "WHERE table_name='tensor_maturity_verdicts' ORDER BY ordinal_position")
        print(json.dumps(rows, indent=1, default=str))