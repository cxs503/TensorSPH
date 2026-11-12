#!/usr/bin/env python3
"""Render + apply the m16318 board-audit verdict INSERT for mission 16244.

psycopg2 is a stub in this sandbox, so the statement is rendered to
audit/insert_verdict_m16318.sql and applied with psql (upsert on
(repo, scope, verdict_by), re-arming gate_state='pending').
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "m16318_verdict_evidence.json"
SQLFILE = HERE / "insert_verdict_m16318.sql"

REPO = "TensorSPH"
SCOPE = "m16244 CASE-SPH-RAYLEIGH-TAYLOR-MP"
MISSION_ID = 16318
LEVEL = 3
VERDICT_BY = "board-audit"

TEMPLATE = """INSERT INTO tensor_maturity_verdicts
  (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('{repo}', '{scope}', {mission_id}, {level}, '{evidence}'::jsonb, '{verdict_by}')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id,
      level = EXCLUDED.level,
      evidence = EXCLUDED.evidence,
      gate_state = 'pending',
      gate_level = NULL,
      gate_note = NULL
RETURNING id, repo, scope, mission_id, level, verdict_by, gate_state, gate_note;
"""


def main():
    ev = json.loads(EVIDENCE.read_text())
    ev_text = json.dumps(ev, ensure_ascii=False).replace("'", "''")
    sql = TEMPLATE.format(repo=REPO, scope=SCOPE, mission_id=MISSION_ID,
                          level=LEVEL, evidence=ev_text,
                          verdict_by=VERDICT_BY)
    SQLFILE.write_text(sql)
    out = subprocess.run(
        ["psql", "-U", "postgres", "-d", "tensorplatform", "-f", str(SQLFILE)],
        capture_output=True, text=True)
    print(out.stdout, out.stderr)
    print("sql rendered ->", SQLFILE)


if __name__ == "__main__":
    main()