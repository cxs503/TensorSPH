"""Publish the m3790 mission report to the platform ledger.

Credentials come from the environment (never from the command line), the figure
travels as base64 inside the JSON payload, and the payload is size-checked
against the 250 KB ceiling documented in the platform trap list.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]
ROOT = Path("/root/tp-agent/work/m3790")
MISSION = 3790

DOC = ROOT / "docs/reports/m3790-merge-m2926-into-main.md"
FIG = ROOT / "docs/reports/m3790-delta-s-merge-evidence.png"
LOG = ROOT / "runs/pytest_merged.log"


def post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main() -> int:
    auth = post("/auth/login", {"username": USER, "password": PASSWORD})
    token = auth.get("access_token") or auth.get("token")
    if not token:
        print("login failed:", json.dumps(auth)[:300])
        return 1

    tail = "".join(LOG.read_text().splitlines(keepends=True)[-6:])
    markdown = DOC.read_text()
    markdown += "\n\n### 附：测试原始日志尾部（runs/pytest_merged.log）\n\n```\n" + tail + "\n```\n"

    figures = [{
        "name": FIG.name,
        "caption": ("m3790: delta_s schemes after merging PR #86 into main. Left: "
                    "int delta_s dn (analytic target 1); right: |F_cut|/(2 sigma) "
                    "(analytic target 1); grey band = +/-15 % engineering tolerance. "
                    "Numbers re-plotted programmatically from "
                    "benchmarks/delta_s_normalization/validation_report.json in the merged tree."),
        "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
    }]

    payload = {
        "title": "m3790 治愈 TensorSPH PR#86：agent/m2926 → main 语义并集合并（0 静默丢失 + 359 测试回归）",
        "summary": (
            "把脏 PR #86（agent/m2926）三方合并进 main：真 base 5800532、目标 a661054。"
            "2 个冲突文件逐块手写语义并集（参数并集 / docstring 双段 / dataclass 默认值取 main 而新增开关并集），"
            "surface_tension.py 自动合并；用行+符号 multiset 审计证明两侧新增零丢失；AST 与重复定义检查通过；"
            "合并树内 359 条测试回归。"
        ),
        "scenario_key": "tensorsph-merge-heal",
        "mission_id": MISSION,
        "tags": ["TensorSPH", "merge", "PR#86", "delta_s", "CSF", "m3790"],
        "markdown": markdown,
        "figures": figures,
    }
    body = json.dumps(payload).encode()
    print(f"payload {len(body)/1024:.1f} KB")
    if len(body) > 250 * 1024:
        print("payload too large, dropping figure")
        payload["figures"] = []
        payload["markdown"] += "\n\n（图因 250 KB 负载上限未内嵌，见 docs/reports/m3790-delta-s-merge-evidence.png）\n"
    res = post("/tensor/reports", payload, token)
    print("report:", json.dumps(res)[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())