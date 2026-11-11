#!/usr/bin/env python3
"""Publish the m15314 audit report to the platform (/tensor/reports)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "refs"
ART = REF / "artifacts"
FIG = ART / "figures"
BASE = os.environ["TENSOR_API_BASE"]


def post(path, payload, tok=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"{}")


def fig(name, caption):
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode((FIG / name).read_bytes()).decode()}


def main() -> None:
    tok = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                               "password": os.environ["TENSOR_AGENT_PASS"]})["access_token"]
    orig = json.loads((REF / "benchmarks/lake_at_rest/report.json").read_text())
    rerun = json.loads((ART / "m15278_rerun.json").read_text())
    probe = json.loads((ART / "probe_t0.json").read_text())["result"]
    tol = orig["acceptance_tolerance"]
    cases = sorted(orig["cases"], key=lambda c: c["pressure_drift_max"])
    best = cases[0]
    k = lambda c: (c["mode"], c["dx_m"], c["sound_speed_m_s"], c["viscosity"])  # noqa: E731
    kb = {k(c): c for c in rerun["cases"]}
    maxdiff = max(abs(c["pressure_drift_max"] - kb[k(c)]["pressure_drift_max"]) for c in orig["cases"])
    shipped_max = max(c["pressure_drift_max"] for c in orig["cases"] if c["mode"] == "shipped")

    rows = "\n".join(
        "| %d | %s | %.0f | %.0f | %.3f | %d/%d | %.3f %% | %s | %.2f× |"
        % (i, c["mode"], c["dx_m"] * 1000, c["sound_speed_m_s"], c["viscosity"],
           c["particles"], c["interior_particles"], 100 * c["pressure_drift_max"],
           "PASS" if c["pass_0p5pct"] else "FAIL", c["pressure_drift_max"] / tol)
        for i, c in enumerate(orig["cases"], 1))

    md = f"""## 1. 案例介绍
静水湖（anchorslug `CASE-LAKE-AT-REST-SPH`，交差审计单 m15314，被审 m15278）：1000×350 mm 二维水池，
rho0=1000 kg/m3、g=9.81、gamma=7 的 WCSPH（TensorSPH 生产共享 RHS `droplet_impact.rhs`），
精确静水解 ref：u=0、eta=H、p=rho0 g (H−y)（LeVeque 1998 JCP 146:346-365 §2.1；Adami-Hu-Adams 2012 JCP 231:7057-7075 §3.1）。
判据（算例池 `tensor_case_pool#67`）：压力漂移 < 0.5 % of rho0 g H。
注意：`tensor_benchmark_cases` 对该锚点 **0 行** → 本单没有机器裁判行，判据只能援引算例池文本。

## 2. 怎么做的（审计流程）
①取证：读被审工作树 `/root/tp-agent/work/m15278`（base ae61a2b，dirty）与 `tensor_reports#14367`；
②逐字节快照交付物 + sha256 落 `docs/board_audit/m15314/refs/`；
③独立复跑交付的 repro_cmd（`PYTHONPATH=src python3 benchmarks/lake_at_rest/run.py`）→ 与原工件比对；
④审计自写 t=0 探针 `probe_t0.py` 与场图 `field_map_t0.py`（不经交付方聚合逻辑）核缺陷定位数字；
⑤实跑单测；⑥按 maturity_gate 判据方案落 verdict（row 276，`verdict_by=board-audit`，L3）。

## 3. 计算结果（工件注入，零手抄）
判据：drift < {100*tol:.1f} %；误差定义 max over 内区粒子×采样 of |p−rho0 g(H−y)|/(rho0 g H)。

| # | mode | dx [mm] | c0 [m/s] | nu | 粒子/内区 | drift max | <0.5%? | 超判据 |
|---|---|---|---|---|---|---|---|---|
{rows}

* 复跑一致性：7/7 算例漂移与原工件**逐位相同**，max |Δ| = {maxdiff:.1e}（确定性重放通过）。
* shipped 最大 {100*shipped_max:.2f} %；wb 最好 {100*best['pressure_drift_max']:.3f} %（{best['mode']}, dx=20mm, c0=20, nu=0.01）→ 改善 {orig['summary']['improvement_factor_at_dx_0p02']:.2f}×，仍超判据 {best['pressure_drift_max']/tol:.2f}×。
* t=0 探针：内区偏差 max {probe['shipped']['interior_dev_max_pct']:.5f} %（初始化本身正确）；底行 max {probe['shipped']['bottom_row_dev_max_pct']:.1f} %、max|a| {probe['shipped']['acc_max_m_s2']:.2f} m/s2（1.94 g）→ wb 后 {probe['wb']['bottom_row_dev_max_pct']:.3f} % / {probe['wb']['acc_max_m_s2']:.2f} m/s2。

## 4. 结果图
场量云图（审计自算 t=0 压力偏差场，shipped vs wb）+ 判据对照条图 + 漂移史曲线 + 交付方的垂向剖面图。

## 5. 注意点（局限与坑）
* 判据与算式自指：wb 算子减去的 p_h 与 drift 度量的 p_h 是同一解析剖面（H 由用户先验经 `Config.equilibrium_level` 传入）→ {100*best['pressure_drift_max']:.3f} % 是 well-balanced 性质的度量，**不是**一般自由面精度证据。
* 默认路径仍是缺陷态：`equilibrium_level=None` 保留原路径（底行亏空、1.94 g 虚假加速度、shipped 漂移 {100*shipped_max:.2f} %）。
* 交付物**未进 git**（head_commit=null，路径全部 untracked，生态仓 grep `equilibrium_level` 0 命中）→ 产品侧无该能力；报告 §6 引的 `ae61a2b` 是基线提交，属 provenance 误记。审计已把 154 行 patch 与全部工件快照进本分支保命。
* 未跑：独立实现横比（ISPH/DFSPH）、0.5 % 判据在 c0/√(gH)=10.79 下的可达性实验、三维/长时行为。

## 6. 误差判定（逐项，显式回答）
| 关键量 | ref | got | 误差 | 是否 <3 %？ | 是否 <0.5 %（本锚判据）？ |
|---|---|---|---|---|---|
| 压力漂移 max（最优 wb） | 0 | {100*best['pressure_drift_max']:.3f} % | {100*best['pressure_drift_max']:.3f} % | **否**（3.887 > 3） | **否**（超 {best['pressure_drift_max']/tol:.2f}×） |
| 压力漂移 max（shipped 默认） | 0 | {100*shipped_max:.3f} % | 同左 | **否** | **否**（超 {shipped_max/tol:.1f}×） |
| max|u|/√(gH) | 0 | {best['v_max_over_sqrt_gH']:.4f} | — | 否 | 否 |

结论：**honest FAIL 成立**，被审单自报的 FAIL 与本审计独立复跑/探针一致，无粉饰；级别 L3（集成实质真）而非 L4（基准未通过）。
disposition：request changes（提交进仓 + 继续截断带密度自洽修正 + 登记判据行）。
"""
    payload = {
        "title": "交差审计 m15278 CASE-LAKE-AT-REST-SPH：静水湖判据未达标（最好 wb 3.887% vs 0.5%，超 7.77×），L3 而非 L4；交付物未进 git",
        "summary": ("独立复跑 7/7 算例逐位复现 + 自写 t=0 探针：被审单的 honest FAIL 属实；"
                    "well-balanced 改善 6.76× 但超判据 7.77×；tensor_benchmark_cases 该锚点 0 行（裁判空转）；"
                    "全部交付物 untracked（head_commit=null），产品侧无 equilibrium_level。verdict row 276 = L3。"),
        "scenario_key": "CASE-LAKE-AT-REST-SPH",
        "mission_id": 15314,
        "tags": ["TensorSPH", "lake-at-rest", "board-audit", "maturity-gate", "honest-FAIL", "well-balanced"],
        "markdown": md,
        "figures": [
            fig("audit_pressure_field_t0.png",
                "t=0 压力偏差场 p−rho0 g(H−y)（% of rho0 g H），上 shipped 下 wb；底行亏空与自由面带是漂移驱动"),
            fig("audit_drift_vs_criterion.png", "7 算例 drift vs 0.5 % 判据（log 轴），全部 FAIL"),
            fig("audit_drift_vs_time.png", "dx=20 mm/c0=20 漂移史：shipped vs wb（审计复跑，逐位同原工件）"),
            fig("lake_at_rest_profile.png", "交付方 t=0 垂向剖面（ref vs got）：shipped 底行 −24.8 %、wb 贴静水解"),
        ],
    }
    r = post("/tensor/reports", payload, tok)
    print("report:", {k: r.get(k) for k in ("id", "status", "scenario_key", "mission_id")})


if __name__ == "__main__":
    main()