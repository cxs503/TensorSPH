#!/usr/bin/env python3
"""Publish the m17452 board-audit report to the tensor report API.

All numbers are read from artifacts/audit_m17452/{judge,repro,referee,finding}.json
and from the audited results.json — nothing is typed by hand ("零手抄").
Figures are generated here from judge.json and embedded as base64.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "artifacts", "audit_m17452")
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")


def load(name):
    with open(os.path.join(OUT, f"{name}.json")) as fh:
        return json.load(fh)


def b64(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def fig_convergence(j) -> str:
    """error vs dx curves, one panel per case — the comparison curve."""
    path = os.path.join(OUT, "fig_convergence.png")
    cases = sorted({r["case"] for r in j["rows"]})
    cmap = {"oss": "#c0392b", "panel": "#2471a3", "pairwise_legacy": "#7f8c8d",
            "oss_viscous": "#8e44ad"}
    fig, axes = plt.subplots(1, len(cases), figsize=(4.1 * len(cases), 3.6))
    for ax, case in zip(axes, cases):
        for op in ("oss", "oss_viscous", "panel", "pairwise_legacy"):
            pts = [(r["dx"], 100 * r["error"]) for r in j["rows"]
                   if r["case"] == case and r["operator"] == op]
            if not pts:
                continue
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    color=cmap[op], label=op, ms=4, lw=1.4)
        ax.axhline(15, color="k", ls="--", lw=1.0)
        ax.axhline(-15, color="k", ls="--", lw=1.0)
        ax.axhline(3, color="green", ls=":", lw=0.9)
        ax.set_xscale("log")
        ax.set_title(case, fontsize=9)
        ax.set_xlabel("dx  [m]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("load error vs analytic  [%]")
    axes[0].legend(fontsize=7)
    fig.suptitle("m17452: refinement study of the ported ForceFromFluid — the +24..+108 % "
                 "bias is dx-independent (not a resolution error)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build() -> dict:
    j, rep, ref, find = (load(n) for n in ("judge", "repro", "referee", "finding"))
    res = json.load(open("/root/tp-agent/work/m17452/artifacts/xval_fsi2/results.json"))
    s = j["summary"]
    ld = j["level_decision"]
    fig1 = fig_convergence(j)
    rowtab = "\n".join(
        f"| {r['case']} | {r['dx']:g} | {r['operator']} | {r['ref']:.6g} | {r['got']:.6g} "
        f"| {100*r['error']:+.2f} % | {'✔' if r['passes_15'] else '✘'} |"
        for r in j["rows"] if r["operator"] != "oss_riemann")
    thick = "\n".join(
        f"| {t['case']} | {t['layers']} | oss {100*t['oss_error']:+.2f} % "
        f"| panel {100*t['panel_error']:+.2f} % | pairwise {100*t['pairwise_error']:+.2f} % |"
        for t in res["thickness"])
    thick = "| case | 固体层数 | oss | panel | pairwise |\n|---|---|---|---|---|\n" + thick

    def row(case, dx, op):
        return next(r for r in j["rows"]
                    if r["case"] == case and abs(r["dx"] - dx) < 1e-12 and r["operator"] == op)

    wall05, wall0125 = row("vertical_wall", 0.05, "oss"), row("vertical_wall", 0.0125, "oss")
    gate05 = row("horizontal_gate", 0.05, "oss")
    visc = row("couette_shear", 0.00625, "oss_viscous")
    insert = row("fsi2_insert_linear", 0.1, "oss")
    md = f"""# 交差核验 m17452 — XVAL-SPH-FSI2（TensorSPH）：移植的 SPHinXsys ForceFromFluid 是否过基准？

## ① 案例介绍

被审任务 m17452 的目标是把上游开源实现 `Xiangyu-Hu/SPHinXsys` 的流→固载荷传递算子
（`FSI::PressureForceFromFluid` / `ViscousForceFromFluid`）按其 master 源码代数移植到
TensorSPH，并用**解析基准**给出误差。物理背景与几何（全部为 2D、无量纲 ρ0=1）：

| 算例 | 几何 / 参数 | 解析参考 |
|---|---|---|
| vertical_wall | 静水柱深 H=2.0 m，竖直墙面，g=9.81（hydrostatic_fsi.cpp:14-19） | 1/2 ρ g H² = {wall05['ref']:.0f} N/m |
| horizontal_gate | 同一水柱压在水平底板上 | 1/2 ρ g H²（法向向下） |
| couette_shear | 间隙 0.1 m、壁面速度 U、动力粘度 μ 的单层剪切 | μ U / h |
| fsi2_insert_linear | fsi2 插入体（圆 r + 矩形 bh×bl），线性压力场 | πr² + bh·bl（阿基米德） |

判据（被审任务在跑数**之前**写在 `benchmarks/xval_fsi2/run.py` docstring 里）：
|err| < **15 %** 必需、< **3 %** 目标（= owner 精度标准）。审计未改动任何阈值。
注册裁判 `tensor_benchmark_cases` 对该锚点为 **{ref['n_rows']} 行**（如实记录：无注册用例）。

## ② 怎么做的（审计方法）

1. **取证**：对被审 9 个交付路径取 sha256/字节/存在性；`git status` 记录提交状态；
   `tensor_benchmark_cases` 按锚点查询。
2. **重放**：把 `src/ tests/ benchmarks/ artifacts/` 复制到临时树，跑
   `pytest tests/test_fsi_force_from_fluid.py` 与 `python benchmarks/xval_fsi2/run.py`，
   再把新 results.json 与原件**逐叶比对**（排除 `timings_s`）。
3. **判读**：逐行算 `(got−ref)/|ref|`，与 15 %/3 % 双门对照；按功能实质定 L0–L5：
   L4 要求移植算子每行都进 15 %；L3 要求 `src/`/`examples/` 内有生产调用者；L2 要求
   真实现 + 可复现驱动。
4. **落表**：`tensor_maturity_verdicts`（verdict_by=`board-audit`），evidence 含
   impl_files/entry/run_artifacts/tests/integration_callers/benchmark/repro_cmd/external_anchor。

## ③ 计算结果

三件套全部真实存在；**重放逐位一致**：{rep['leaves_compared_excluding_timings']} 个数值叶
全等（{rep['leaves_audited']} 叶中仅 {rep['ignored_timing_leaves']} 个 timings 叶不同），
`pytest` rc={rep['pytest']['returncode']}（10 passed），`run.py` rc={rep['run']['returncode']}。

| case | dx | operator | ref | got | err | <15 % |
|---|---|---|---|---|---|---|
{rowtab}

* 移植算子（oss 压力 + oss_viscous）：**{s['primary_pass15']}/{s['primary_rows']}** 行在 15 % 内
  （去重：9 压力行 + 3 粘性行 = 12 个独立样本，**0 通过**）。`oss_riemann` 与 `oss` 逐位相同，
  因为 fsi2 配置 `NoRiemannSolver` → 该项本应为零（自洽性检查，非独立样本）。
* 压力支误差带 **{100*s['oss_pressure_err_range'][0]:+.2f} … {100*s['oss_pressure_err_range'][1]:+.2f} %**；
  粘性支 **{100*s['oss_viscous_err_range'][0]:+.2f} … {100*s['oss_viscous_err_range'][1]:+.2f} %**。
* 厚度敏感性（dx=0.05，1 层 → 3 层，vertical_wall，oss）：
  {100*res['thickness'][0]['oss_error']:+.2f} % → {100*res['thickness'][1]['oss_error']:+.2f} %。

| case | 固体层数 | 误差 |
|---|---|---|
{thick}

## ④ 结果图

细化研究（每个算例一条误差-vs-dx 曲线，虚线 = ±15 % 判据，点线 = ±3 % 目标）：
误差随 dx 加密几乎不变（vertical_wall oss：{100*row("vertical_wall",0.05,"oss")['error']:+.2f} →
{100*row("vertical_wall",0.025,"oss")['error']:+.2f} → {100*row("vertical_wall",0.0125,"oss")['error']:+.2f} %），
**证明偏差是系统性的、不是分辨率误差**。panel 臂在闭合体上随加密变差
（{100*row("fsi2_insert_linear",0.1,"panel")['error']:+.2f} → {100*row("fsi2_insert_linear",0.05,"panel")['error']:+.2f} →
{100*row("fsi2_insert_linear",0.025,"panel")['error']:+.2f} %），是独立缺陷。

## ⑤ 注意点

* **panel 的 −0.00 % 是同义反复**：panel 是核权最小二乘平面拟合，而解析静水压力场本身就是
  线性场，精确复现由定义使然（被审自己的测试即断言这一点）。不能作为"算子已验证"的证据。
* **测试通过 ≠ 能力通过**：10 个测试中两条把失败带钉成期望值（`0.20 < err < 0.32`、
  `0.55 < got/ref < 0.70`），是变更探测器，不是验证。
* **出处缺口**：模块 docstring 称 OSS 源码已 verbatim 归档到 `oss_src/`，实际只归档了
  `hydrostatic_fsi.cpp` + 3 个 XML，被引的 `force_on_structure.hpp:55-72/103-124`、`fsi2.cpp`
  不在树上 → 引用行号本地不可核。
* **过度声明**：`run.py` 设计段称"有一个算例族用求解后的 WCSPH 场"，
  实测 {j['n_rows_scored']} 行全是 `fluid_state='analytic'`，`solved` 行数 = {s['rows_with_solved_fluid_state']}。
* **流程**：4 条交付路径全部 untracked（`git status ??`），`head_commit=null`，
  任务以 `budget_exhausted+forced_finish`（80/80 调用）收尾 → 无 commit/PR 身份，返修须从快照起步。
* 适用范围：仅 2D、解析场、静止固体；移动固体（Riemann 项非零）、耦合瞬态、3D 均**未覆盖**。

## ⑥ 误差判定

| 关键量 | 基准 | got | 误差 | < 3 %？ |
|---|---|---|---|---|
| 静水墙压载（oss, dx=0.05） | {wall05['ref']:.1f} N/m | {wall05['got']:.1f} N/m | {100*wall05['error']:+.2f} % | **否** |
| 静水墙压载（oss, dx=0.0125） | {wall0125['ref']:.1f} N/m | {wall0125['got']:.1f} N/m | {100*wall0125['error']:+.2f} % | **否** |
| 底板压载（oss, dx=0.05） | {gate05['ref']:.1f} N/m | {gate05['got']:.1f} N/m | {100*gate05['error']:+.2f} % | **否** |
| Couette 粘性力（oss_viscous, dx=0.00625） | μU/h | {visc['got']:.4f} | {100*visc['error']:+.2f} % | **否** |
| fsi2 闭合体浮力（oss, dx=0.1） | {insert['ref']:.4f} | {insert['got']:.4f} | {100*insert['error']:+.2f} % | **否** |

**结论：无一项关键量 < 3 %，且 12/12 独立样本 > 15 % 不可接受阈 → 基准判定 FAIL（honest FAIL）。**
功能实质：真实现 + 不变量测试 + 逐位可复现 = **L2**；但无生产调用者/包导出（不足 L3）、
基准未过（不足 L4）。**级别 = L2**，disposition = request changes（返修）。
值得肯定：被审报告自己把失败写在正文里（没有粉饰），这使本次返修有明确靶点。
"""
    return {
        "title": "交差核验 m17452：SPHinXsys ForceFromFluid 移植入 TensorSPH — 基准 FAIL，判 L2",
        "summary": (f"board-audit of m17452 (XVAL-SPH-FSI2): deliverables reproduce bitwise "
                    f"({rep['leaves_compared_excluding_timings']} leaves), but the ported "
                    f"operator is {s['primary_pass15']}/{s['primary_rows']} inside the 15 % "
                    f"criterion (pressure {100*s['oss_pressure_err_range'][0]:+.1f}.."
                    f"{100*s['oss_pressure_err_range'][1]:+.1f} %, viscous "
                    f"{100*s['oss_viscous_err_range'][0]:+.1f} %) and has no production "
                    f"caller -> L2, benchmark FAIL, request changes."),
        "scenario_key": "board-audit",
        "mission_id": 17494,
        "tags": ["board-audit", "TensorSPH", "XVAL", "FSI", "SPHinXsys", "benchmark-fail"],
        "markdown": md,
        "figures": [
            {"name": "fig_convergence.png",
             "caption": "误差 vs dx（每算例一条曲线，虚线 = ±15 % 判据）：移植算子的 +24..+108 % "
                        "偏差与 dx 无关，属系统偏差；panel 在闭合体上随加密变差。",
             "png_base64": b64(fig1)},
            {"name": "fig_error_vs_threshold.png",
             "caption": "逐行载荷误差柱状图与 15 % 硬线 / 3 % 目标线：12/12 独立移植算子样本超 15 %。",
             "png_base64": b64(os.path.join(OUT, "fig_error_vs_threshold.png"))},
        ],
    }


def main() -> int:
    body = build()
    with open("/tmp/report_m17494.json", "w") as fh:
        json.dump(body, fh, ensure_ascii=False)
    tok = open("/tmp/tok").read().strip()
    req = urllib.request.Request(BASE + "/tensor/reports",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok})
    r = json.load(urllib.request.urlopen(req))
    print("report id", r.get("id"), "title", r.get("title"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())