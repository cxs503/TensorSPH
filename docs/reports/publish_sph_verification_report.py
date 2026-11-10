"""Publish the P482-N04 verification report to the platform report API.

Reads ``docs/reports/sph-verification-P482-N04.md`` (generated from the evidence
stores) and posts it, with the four figures inlined as base64 PNG, to
``$TENSOR_API_BASE/tensor/reports``.  Credentials come from the environment
(``TENSOR_AGENT_USER`` / ``TENSOR_AGENT_PASS``); nothing is stored.

    python docs/reports/publish_sph_verification_report.py --mission 10344
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/reports/sph-verification-P482-N04.md"
FIGDIR = ROOT / "docs/reports/figures"
FIGURES = [
    ("dam_break_front_profile.png",
     "溃坝前沿 Z/a(tau) 与实验表对比（含 Ritter 参考线）+ 不同 tau 的自由面剖面"),
    ("hydrostatic_profile.png",
     "静水压力剖面 vs rho g (H-z)，右图为分带相对误差（红=自由面带，不门控）"),
    ("convergence.png",
     "SPH 插值 L2 误差随 h 收敛（Shepard 归一化 / 原始求和 / 截断支持域）"),
    ("mass_bookkeeping.png",
     "质量记账：闭槽、开口槽（记账）与负控（忘记出流）"),
]


def _post(path: str, payload: dict, token: str | None = None) -> dict:
    base = os.environ["TENSOR_API_BASE"]
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    try:
        return json.loads(urllib.request.urlopen(req, timeout=120).read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}: {exc.read()[:400]!r}")


def login() -> str:
    body = {"username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"]}
    base = os.environ["TENSOR_API_BASE"]
    req = urllib.request.Request(base + "/auth/login",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["access_token"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mission", type=int, default=10344)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    markdown = REPORT.read_text()
    figures = []
    for name, caption in FIGURES:
        path = FIGDIR / name
        if not path.exists():
            print(f"warning: missing figure {path}")
            continue
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode(path.read_bytes()).decode()})
    payload = {
        "title": "P482-N04 TensorSPH 自由面流动验证面：溃坝/静水/质量守恒/收敛"
                 "（含负控与 SQL 证据链）",
        "summary": "建成 TensorSPH 自由面验证面：M1 溃坝前沿与柱高、M2 静水压力 N=1000、"
                   "M3 质量记账、M4 插值收敛，各配负控；J2/J3/J4 通过，J1 前沿如实 FAIL "
                   "并定位到自由面压力钳制与耗散缺失，全部数字 SQL 可溯。",
        "scenario_key": "tensorsph_free_surface_verification",
        "mission_id": args.mission,
        "tags": ["TensorSPH", "verification", "dam-break", "SPH", "P482-N04", "honest-fail"],
        "markdown": markdown,
        "figures": figures,
    }
    if args.dry_run:
        print(json.dumps({k: (v if k != "markdown" else v[:200] + "...")
                          for k, v in payload.items() if k != "figures"},
                         ensure_ascii=False, indent=1)[:1200])
        print(f"figures: {len(figures)}")
        return 0
    token = login()
    out = _post("/tensor/reports", payload, token)
    print("published:", json.dumps(out)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
