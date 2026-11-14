#!/usr/bin/env python3
"""Publish the Wo-envelope report for mission 20678 to the platform report API.

Body is assembled from the committed artefacts (note + figures), so nothing is
retyped by hand.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "agent")
PASS = os.environ.get("TENSOR_AGENT_PASS", "agent")


def post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main() -> int:
    tok = post("/auth/login", {"username": USER, "password": PASS})
    token = tok.get("access_token") or tok.get("token") or tok.get("data", {}).get("access_token")
    if not token:
        print("login failed:", str(tok)[:300])
        return 1

    doc = (REPO / "docs" / "case_poiseuille_wo_envelope.md").read_text()
    figs = []
    for name, cap in (
        ("case_poiseuille_wo_envelope.png",
         "图1 (a) H/dx=16 频率扫描的判据量 vs Wo；(b) Wo=6.5 分辨率阶梯：内区判据 vs 壁面行/解析行平均地板；"
         "(c) 壁面行误差塌缩到 delta/dx（颜色=Wo，大小=H/dx）"),
        ("case_poiseuille_wo_envelope_profiles.png",
         "图2 (a) 四个频率的振幅剖面（灰带=不参与判据的壁面带）；(b) Wo=12.5 实测 vs Womersley 闭式解与逐行相位误差"),
    ):
        p = REPO / "docs" / "figures" / name
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode(p.read_bytes()).decode()})

    verdict = json.loads((REPO / "benchmarks/case_poiseuille/out/wo_envelope.json").read_text())["verdict"]
    tail = ["", "## 附：关键裁定（由工件程序化判定）", "",
            "| 假设 | 裁定 |", "|---|---|"]
    labels = [("H1_wo_free", "H1 频响不受 Wo 限制（预声明）"),
              ("H2_layer_resolution", "H2 `delta/dx >= 3` 规则（预声明，复述既有范围声明）"),
              ("H3_transient_artefact", "H3 伪迹=未收敛周期态"),
              ("H4_noise_floor", "H4 噪声地板"),
              ("H5_two_ceiling", "H5 天花板=时间采样而非物理（Stage 2）"),
              ("H6_interior_gate_blind_to_wall_layer", "H6 内区判据看不见 Stokes 层（Stage 2）"),
              ("H7_row_mean_floor_share", "H7 行平均地板占比 <=50%（post-hoc）"),
              ("H8_wall_error_collapse", "H8 壁面误差是 delta/dx 的单值函数（post-hoc）")]
    for key, label in labels:
        s = verdict[key]["supported"]
        tail.append(f"| {label} | {'成立' if s else '**被证伪**'} |")
    tail += ["",
             f"采样足够的点：{verdict['n_pass']} 通过 / {verdict['n_fail']} 未通过 / "
             f"{verdict['n_invalid']} 采样无效，共 {verdict['n_points']} 点。"]

    summary = ("CASE-POISEUILLE-SPH：把既有验证笔记里断言的适用上界（Wo~3、delta/dx 3-7）实测出来。"
               "预声明的 delta/dx>=3 规则在第一个探针点即被证伪（delta/dx=1.74 时内区振幅 RMS 仅 0.60%）；"
               "14 点三组操纵表明：采样足够时 Wo 一直到 12.5（delta/dx 0.90）全部判据 <5%（最坏 1.27%），"
               "唯一越界点是每周期 50 步 < 64 采样的时间采样不足（标 invalid_sampling 而非 FAIL，同 Wo 加密即回到带内）；"
               "不被判据覆盖的壁面行误差 5.7%→31%，其中 21-40% 由解析行平均地板解释，其余为亚格子层抹平，"
               "匹配 delta/dx 的不同 (Wo, H/dx) 点误差一致在 1.02 pp 内。")

    payload = {
        "title": "CASE-POISEUILLE-SPH · Wo 包线：调和判据到底在哪里失效（H2 被证伪）",
        "summary": summary,
        "scenario_key": "CASE-POISEUILLE-SPH",
        "mission_id": 20678,
        "tags": ["TensorSPH", "CASE-POISEUILLE-SPH", "Womersley", "verification",
                 "wo-envelope", "falsified-hypothesis"],
        "markdown": doc + "\n".join(tail),
        "figures": figs,
    }
    out = post("/tensor/reports", payload, token)
    print(json.dumps(out)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())