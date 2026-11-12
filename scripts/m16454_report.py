"""Publish the m16454 progress report to the platform (no secrets in the file).

Reads the measured metrics JSON, renders the six-element markdown with the
numbers injected programmatically, embeds the figure as base64 and POSTs it to
``$TENSOR_API_BASE/api/v1/tensor/reports`` after logging in with
``$TENSOR_AGENT_USER`` / ``$TENSOR_AGENT_PASS``.

Usage::  python scripts/m16454_report.py [--dry-run]
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
METRICS = "runs/fsi_elastic_gate_a2/a2_metrics.json"
FIG = "docs/fsi_elastic_gate/fig_a2_frame_fix.png"


def pct(x: float) -> str:
    return "%+.2f%%" % (100.0 * x)


def post(path: str, payload: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer %s" % token
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    d = json.load(open(METRICS))
    r = {x["n_gate_nodes"]: x for x in d["sweep"]}
    eb = d["elastica_vs_euler_bernoulli"][-1]
    png = base64.b64encode(open(FIG, "rb").read()).decode()

    rows = "\n".join(
        "| %d | %.4f / %.4f | %s | %s | %s | %s |"
        % (x["n_gate_nodes"], x["tip_transverse_rod"], x["tip_transverse_elastica"],
           pct(x["rel_error"]), pct(x["axial_rel_error"]), pct(x["slope_rel_error"]),
           pct(x["frame_mismatch_rel_error"]))
        for x in d["sweep"])

    md = f"""## ① 案例介绍
CASE-SPH-FSI-ELASTIC-GATE（Antoci 2007 弹性闸门溃坝）的**结构腿 A2**：单宽竖直悬臂杆 H={d['H']} m、厚 4 mm、
E=1e6 Pa，EI={d['EI']:.4e} N·m²；端部集中横向力 P={d['P']:.5f} N/m 使 PH²/EI={d['P_H2_over_EI']:.1f}
（端部转角约 82°，大变形区）。上一轮（m16352）报告 A2 未达标 **+86.8%**，并据此判定「闸门顶点位移指标不可信」。

## ② 怎么做的
诊断为**坐标系错配**：杆立在 +y、载荷沿 +x（横向量=Δx）；`elastica_tip` 沿自身 +x 积分、横向载荷沿 +y
（横向量=tip_y）。交付版拿 `Δx` 去比 `tip_x`（轴向坐标）。修复：①同一坐标系重算三条指标（横向挠度 /
轴向缩短 / 端部转角，用末段切向）；②错配量降级为诊断量（永不作为门）；③elastica 参照加
「PH²/EI→0 退化为 PH³/(3EI)」极限自校核；④离散收敛阶梯 n=13/25/50；⑤测试由「已知缺口
0.2<err<1.2」改为三条真门 + 收敛单调断言。工具链：能量极小化（L-BFGS-B + autograd 梯度）求杆静平衡，
elastica 为弧长 ODE + 打靶（无抄表）。

## ③ 计算结果（程序化注入，源 {METRICS}）
| n | 横向挠度 杆/elastica [m] | 横向误差 | 轴向缩短误差 | 端部转角误差 | 交付版错配量（诊断） |
|---|---|---|---|---|---|
{rows}

elastica 参照自校核（vs Euler-Bernoulli `P H³/(3EI)`，PH²/EI=0.1）：{pct(eb['rel_error'])}。

## ④ 结果图
(a) PH²/EI=10 变形构型：25 节点离散杆 vs 独立积分 elastica（同一坐标系重绘）；
(b) 三条指标误差随 n 收敛（对数轴）+ 交付版错配量 + 5% 门。

## ⑤ 注意点 / 局限
- 只修比法与门，杆本构（拉伸+弯曲、固支端半格）不变；残差是分段常曲率的一阶离散误差（随 n 减半，
  n=50 横向 {pct(r[50]['rel_error'])}）。
- A2 仅静力、重力关；动载下显式杆仍受 `Ω_ax·dt` 声学步长限制（m16352 的 V4 not_run 原因②仍在）。
- Antoci2007 实验峰值/相位参照值仍 **WITHHELD**，算例级 <10% 判据**仍不可判**（本轮未改变）。

## ⑥ 误差判定（是否 <3%）
| 指标（n=25） | 误差 | <3%? |
|---|---|---|
| 横向挠度（判据量的物理对应） | {pct(r[25]['rel_error'])} | 否（在 <5% 门内） |
| 轴向缩短 | {pct(r[25]['axial_rel_error'])} | 否 |
| 端部转角 | {pct(r[25]['slope_rel_error'])} | 是 |
| elastica 参照自校核 | {pct(eb['rel_error'])} | 是 |

结论：**+86.8% 的「杆大变形精度缺口」被证伪**——错配量 ≈85–91% 且与 n 基本无关，是比法造成的；
杆的大变形精度**不是**闸门顶点位移指标的阻塞项（阻塞项只剩显式步长限制与参照值不可得）。
"""

    report = {
        "title": "CASE-SPH-FSI-ELASTIC-GATE · A2 大变形判据坐标系错配修复（+86.8% → +2.55%）",
        "summary": ("A2 把杆的横向挠度与 elastica 的轴向坐标相比，误报 +86.8% 杆精度缺口；同坐标系复算 "
                    "25 节点横向 +2.55% / 轴向 +3.67% / 转角 +0.59%，收敛台阶 4.91%→2.55%→1.28%，"
                    "守门测试改为三条真门。"),
        "scenario_key": "fsi-elastic-gate-a2-frame-fix",
        "mission_id": 16454,
        "tags": ["TensorSPH", "FSI", "elastic-gate", "elastica", "verification",
                 "frame-fix", "CASE-SPH-FSI-ELASTIC-GATE"],
        "markdown": md,
        "figures": [{"name": "fig_a2_frame_fix.png",
                     "caption": ("(a) PH²/EI=10 变形构型：离散杆 vs elastica（同坐标系）；"
                                 "(b) 三指标误差随 n 收敛 + 交付版错配量"),
                     "png_base64": png}],
    }

    if "--dry-run" in sys.argv:
        json.dump(report, open("/tmp/report.json", "w"), ensure_ascii=False)
        print("dry-run: wrote /tmp/report.json", len(report["markdown"]), "chars md")
        return 0

    login = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                        "password": os.environ["TENSOR_AGENT_PASS"]})
    token = (login.get("data") or login).get("access_token") or (login.get("data") or {}).get("token")
    print("login ok:", bool(token))
    out = post("/tensor/reports", report, token)
    data = out.get("data", out)
    print(json.dumps({"id": data.get("id"), "title": data.get("title")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())