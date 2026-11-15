"""Publish the m21208 progress report to the platform (no shell secret handling).

Run:  PYTHONPATH=.:src python3 scripts/report_m21208.py
Reads TENSOR_API_BASE / TENSOR_AGENT_USER / TENSOR_AGENT_PASS from the
environment, logs in, and POSTs the report body with the regenerated figure
embedded as base64.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.error
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
REPO = pathlib.Path(__file__).resolve().parents[1]

MARKDOWN = """# CASE-POISEUILLE-SPH 相位单位审计与修复（对抗复核）

## ① 案例介绍
平面泊肃叶槽道 `0<=y<=H`，两壁无滑移、x 向周期，单位质量体积力 `g(t)` 驱动：
`u_t = nu u_yy + g(t)`。三种驱动同一方程：R1 常力（启动 + 稳态抛物），R2 简谐
（`g0 cos wt` -> Womersley 周期解）。参数 H=0.1 m、nu=0.01 m^2/s、rho0=1000 kg/m^3、
g0=9.81 m/s^2、omega=13.0 rad/s（Wo=1.803、delta/H=0.39）、omega_fast=31.4 rad/s
（Wo=2.802）、8 列周期、H/dx=8/12/16。解析参考
`Uhat=(g0/(i omega))[1-cosh(alpha(y-H/2))/cosh(alpha H/2)]`，独立参考=Crank-Nicolson
有限差分（与 SPH 路径零共享代码）。

## ② 怎么做的（本单）
对**已合并**算例（#716/#739，9/9 门通过）做独立对抗复核：
1) 重读驱动/参考/文档/图/测试，逐条重算被引用的量；
2) 用"已知坏解"反推每个门的判伪力（quasi-steady 解相位恒 0，G3/G4 应以多少倍超差抓它）；
3) 逐字段核对单位（rad vs deg），追平所有消费者：文档文本、两张图、测试；
4) 修复 + 用"预修产物必然失败"的测试锁定 + 重生成文档与图。

## ③ 结果
| 量（H/dx=16） | 修复前报告 | 真值（修复后） | 门限 | 判定 |
|---|---|---|---|---|
| 中心线相位滞后 | -0.95 "deg"（实为 rad） | -54.70 deg（解析 -54.81） | 4.5 deg（5% of pi/2） | 误差 0.12 deg = 0.13% ✅ |
| 双频相位差 | -0.39 "deg" | -22.33 deg（解析 -22.14） | 报告项 | -0.19 deg |
| 相位剖面 RMS | — | 0.21 % of pi/2 | 5 % | ✅ |
| 中心线幅值误差 | +0.07 % | 不变 | 5 % | ✅ |
| 相位剖面图 y 轴 | -57.52 .. +2.01（rad/deg 混轴） | -56.28 .. -38.86 deg | — | 三条序列同单位 |

门判据本身未受影响（G3/G4 用 `phase_error_deg`，真度数值），9/9 门仍 PASS：
修复只是让证据说出它真正的量级——quasi-steady 解会以 12 倍超差被 G3/G4 判伪，
实测误差在门内 38 倍。

## ④ 图
幅值/相位剖面三联比较图（SPH 三档阶梯 + Womersley 闭式 + CN-FD 参考）。修复前相位面板
被 FD 的 deg 数值拉到 -57.5..2.0，SPH 与闭式曲线（rad，约 -0.95）被压成顶部直线；
修复后三条序列统一以 deg 显示，互相重合。

## ⑤ 注意点（坑与局限）
- 单位陷阱：`harmonic_fit` 与 `womersley.phase` 都用 rad，汇总字段却叫 `*_deg`，
  产生 57.3 倍误差；更危险的是"将来按 `*_deg` 写容差"的门会宽松 57 倍而审查看不出来。
- 测试空转：`... if "phase_centre_deg" in p else True`、`... or True` 三处断言从不生效，
  恰恰是能抓住本 bug 的那一条；已替换为真实不变量。
- 局限：本单**未重跑**全阶梯（递交集群作业 cj3270 排队中；本机执行上限 120 s），
  `out/result.json` 的数值仍是原 run（#739）的结果，仅用存储的 rad 拟合重写派生相位
  字段（`migrate_units.py` 幂等迁移，原始采样/剖面/门判定未动）——"独立复现"仍未闭环。

## ⑥ 误差判定
关键量 vs 解析基准：中心线幅值 0.07 %、幅值剖面 RMS 0.39 %、中心线相位 0.13 %
（of pi/2）、相位剖面 RMS 0.21 %；FD 参考 vs 闭式 0.004 %（G9）。**全部 < 3%**
（门限 5%）。无未声明基准的项。
"""


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    user = os.environ.get("TENSOR_AGENT_USER", "")
    password = os.environ.get("TENSOR_AGENT_PASS", "")
    token = ""
    if user and password:
        try:
            auth = _post(f"{BASE}/auth/login", {"username": user, "password": password})
            token = auth.get("access_token") or auth.get("token") or ""
        except urllib.error.HTTPError as exc:               # try the form endpoint too
            print("login(json) failed:", exc.code)
    fig = REPO / "docs" / "figures" / "case_poiseuille_profiles.png"
    body = {
        "title": "CASE-POISEUILLE-SPH 相位单位审计与修复：-54.8° 被误报为 -0.95°",
        "summary": ("对已合并算例池条目做对抗复核：SPH 驱动把 rad 相位写入 *_deg 字段"
                    "（57.3 倍），文档 headline 与相位图双双失真，门判据未受影响。"
                    "已修复、幂等迁移产物、重生成文档/图，新增单位契约测试（81 passed）。"),
        "scenario_key": "case_poiseuille_sph_phase_units",
        "mission_id": 21208,
        "tags": ["TensorSPH", "CASE-POISEUILLE-SPH", "womersley", "audit",
                 "units", "verification"],
        "markdown": MARKDOWN,
        "figures": [{
            "name": fig.name,
            "caption": ("幅值/相位剖面：SPH 三档 + Womersley 闭式 + CN-FD 参考"
                        "（修复后三序列同以 deg 显示）"),
            "png_base64": base64.b64encode(fig.read_bytes()).decode(),
        }],
    }
    print("body bytes:", len(json.dumps(body)))
    try:
        out = _post(f"{BASE}/tensor/reports", body, token)
    except urllib.error.HTTPError as exc:
        print("POST failed:", exc.code, exc.read().decode()[:400])
        return 1
    kept = {k: out[k] for k in ("id", "status", "title") if isinstance(out, dict) and k in out}
    print("report:", kept or str(out)[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())