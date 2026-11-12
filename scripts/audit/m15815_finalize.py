#!/usr/bin/env python3
"""m15815 finalize — regenerate doc/evidence, insert the maturity verdict, post
the mission report with embedded figures.

Idempotent: re-running updates the doc and (with --insert) adds one verdict row.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs" / "m15815"
FIG = REPO / "docs" / "board_audit" / "m15815"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
HARDENED = RUNS / "audit_run"


def sh(cmd, cwd=REPO):
    return subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)


def post_report(md: str, figures):
    user, pw = os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"]
    body = json.dumps({"username": user, "password": pw}).encode()
    r = urllib.request.Request(BASE + "/auth/login", data=body, method="POST")
    r.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(r, timeout=30) as f:
        tok = json.loads(f.read().decode())["access_token"]
    payload = {"title": "M15815 交差核验：m15493 CASE-WAVE-FLUME 判 L2 返修（已提交记录实测判负 30.5%/33.8%）",
               "summary": "board-audit 取证 m15493 全部工件：唯一裁判 tensor_benchmark_cases 无本锚点行（0/22）；"
                          "已提交两条 record 既非分级工况且数值发散（max|v| 109.3/42.6 m/s vs c0=30，压缩 0.449）；"
                          "用仓库自身判级链判负 worst 30.54%/33.82%（10% 门），μ_meas 0.44 vs 理论 1.79e-3；"
                          "repro 的 --start 4.0 在原样工件上不可执行；独立闭式复算参考尺子 3 工况相对差 ≤5e-16。verdict=L2/reopened。",
               "scenario_key": "CASE-WAVE-FLUME", "mission_id": 15815,
               "tags": ["board-audit", "CASE-WAVE-FLUME", "TensorSPH", "maturity-verdict", "L2"],
               "markdown": md, "figures": figures}
    r = urllib.request.Request(BASE + "/tensor/reports", data=json.dumps(payload).encode(), method="POST")
    r.add_header("Content-Type", "application/json")
    r.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(r, timeout=90) as f:
        return f.status, json.loads(f.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insert", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    HARDENED.mkdir(parents=True, exist_ok=True)
    src = RUNS / "repro_wt" / "runs"
    for f in ("wf_D20_audit.json", "wf_D20_audit.log"):
        if (src / f).exists():
            (HARDENED / f).write_bytes((src / f).read_bytes())
            print("hardened:", f)

    r = sh("python3 scripts/audit/m15815_step4_doc_verdict.py" + (" --insert" if args.insert else ""))
    print(r.stdout[-800:], r.stderr[-800:])

    if args.report:
        md = (REPO / "docs/board_audit/M15815_CASE_WAVE_FLUME_BOARD_AUDIT.md").read_text()
        figs = json.loads((RUNS / "figures.json").read_text())
        figures = [{"name": f["name"], "caption": f["caption"],
                    "png_base64": base64.b64encode((FIG / f["name"]).read_bytes()).decode()}
                   for f in figs]
        # report body: audit doc + verdict evidence digest
        ev = json.loads((RUNS / "verdict_evidence.json").read_text())
        extra = """

---

## 附：verdict 落表内容（程序化生成，零手抄）

```json
%s
```
""" % json.dumps({k: ev[k] for k in ("impl_files", "benchmark", "disposition", "level_rationale")},
                 indent=1, ensure_ascii=False)
        status, res = post_report(md + extra, figures)
        print("report:", status, str(res)[:300])


if __name__ == "__main__":
    main()