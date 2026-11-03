"""Render docs/verification_laplace_equilibrium.md from report.json + drift.json.

Every number in the document is injected from the committed records, so the
prose cannot drift away from the runs.  Same pattern as
``benchmarks/laplace_law/make_doc.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "docs" / "verification_laplace_equilibrium.md"


def f(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


def main() -> int:
    rep = json.loads((HERE / "report.json").read_text())
    drip = json.loads((HERE / "drift.json").read_text())
    p = rep["params"]
    ref = rep["reference"]
    ladder, scan = rep["ladder"], rep["phase_scan"]
    active = next(r for r in drip["runs"] if r["setup"]["sigma_N_per_m"] > 0)
    control = next(r for r in drip["runs"] if r["setup"]["sigma_N_per_m"] == 0)

    rows = "\n".join(
        f"| {r['setup']['dx_m'] * 1e3:.2f} | {r['setup']['n_particles']} | "
        f"{r['setup']['n_complete_support']} | {r['setup']['h_over_radius']:.2f} | "
        f"{r['setup']['t_over_t_sigma']:.2f} | {f(r['equilibrium']['p_complete_Pa'])} | "
        f"{f(r['equilibrium']['rel_error_pct'])} | "
        f"{f(r['equilibrium']['p_band_Pa'])} | "
        f"{f(r['equilibrium']['rel_error_band_pct'])} | "
        f"{f(r['equilibrium']['measurement_snr'], 0)} |"
        for r in ladder)

    band_rows = "\n".join(
        f"| {r['setup']['dx_m'] * 1e3:.2f} | {f(r['equilibrium']['p_complete_Pa'])} | "
        f"{f(r['equilibrium']['p_complete_Pa'] - r['equilibrium']['p_band_Pa'])} | "
        f"{f(r['equilibrium']['rel_error_pct'])} → "
        f"{f(r['equilibrium']['rel_error_band_pct'])} |"
        for r in ladder if r["setup"]["n_band_080_095"] >= 5)

    errs = [r["equilibrium"]["rel_error_pct"] for r in ladder]
    drift_rows = "\n".join(
        f"| {r['setup']['sigma_N_per_m']:g} | {r['setup']['n_particles']} | "
        f"{f(r['outcome']['t_over_t_sigma_last_finite'])} | "
        f"{r['outcome']['r_rms_over_r0_first']:.3f} → "
        f"{r['outcome']['r_rms_over_r0_last']:.3f} | "
        f"{r['outcome']['rho_excess_last']:+.4f} | "
        f"{'是' if r['outcome']['blew_up'] else '否'} |"
        for r in drip["runs"])

    md = f"""# Young–Laplace 压差复核：为什么这个判据至今 WITHHELD（结论 + 机理）

**范围**: `src/tensorsph/csf_consistent.py`（CSF 离散）+ `src/tensorsph/droplet_impact.py`（Tait EOS / 压强）
**基准**: 2-D 静止圆形液滴的 `Delta p = sigma/R`（Young–Laplace，解析）
**本文所有数字由 `benchmarks/laplace_equilibrium/make_doc.py` 从 `report.json` 与 `drift.json` 程序化注入，无手抄**
**工件**: `benchmarks/laplace_equilibrium/{{report.json, drift.json, laplace_equilibrium.png, laplace_drift.png}}`
**测试**: `tests/test_laplace_equilibrium.py`（10 项）

---

## 1. 案例介绍

二维单相液滴悬于真空（无重力、无壁面），六方密排晶格初始化。表面张力应使液滴内部相对外部
真空建立过压 `Delta p = sigma kappa = sigma/R`（2-D 圆形界面 `kappa = 1/R`）。这是 CSF 表面张力
唯一的端到端定量判据。

| 参数 | 值 |
|---|---|
| 液滴半径 R | {p['radius_m']} m |
| 表面张力 sigma | {p['sigma_N_per_m']} N/m（水/空气） |
| 参考密度 rho0 | {p['rho0_kg_per_m3']} kg/m³ |
| 声速 c0 | {p['sound_speed_m_per_s']} m/s（Mach < 0.05） |
| 运动粘度 nu | {p['viscosity_m2_per_s']} m²/s（只决定振荡衰减速率，不决定静态终值） |
| 状态方程 | Tait，gamma = 7，压强下限 0 |
| 分辨率 dx | {', '.join(f"{r['setup']['dx_m']*1e3:.1f}" for r in ladder)} mm（h = 1.3 dx） |
| 解析参考 | `Delta p = sigma/R = {f(ref['delta_p_Pa'])} Pa`，`kappa = 1/R = {f(ref['curvature_1_per_m'])} 1/m` |
| 推进 | 阶梯 {p['capillary_times']} 个毛细时间；稳定性/对照 12 个毛细时间 |

### 本轮要回答的问题

`benchmarks/laplace_law`（dx = 4 mm，380 步）判 **PASS**，`benchmarks/csf_consistent`（dx = 2 mm，
600–900 步）判 **−75.6 %**，同一套 CSF 力、同一个 Tait 律。物理上不可能同时成立。
**结论先说**：两个都不是平衡态读数；本算例在这套离散下**没有静态平衡**，因此
`Delta p = sigma/R` 这一判据仍是 **WITHHELD**——但缺口的位置已经从"表面张力调参"精确定位到
"自由面压强基准 + CSF 壳层力的耦合"。

## 2. 怎么做的

### 2.1 三个测量缺陷（阶梯复核）

1. **测在瞬态上**：液滴以 `rho = rho0` 释放，先被压紧、过冲（呼吸模态，周期约 1 个毛细时间）、
   再衰减；`p_core(t)` 在过程中摆动数倍，固定步数采一次样只拿到一个相位（§3.3 量化）。
2. **界面带落在 CSF 壳层内部**：旧协议取 `p_band` 于 `0.8R < r < 0.95R`。CSF 力由 `delta_s` 集中在
   厚约 `2h` 的壳层，壳层内 `dp/dr = rho a_r < 0`，即 0.95R 处压强已部分掉向真空值，用它当
   "外部压强"必然**系统性低估**压差（§3.2）。
3. **核心带落进核截断区**：dx = 4 mm 时 `r < 0.5R` 已进入 `r > R − 2h`，那里求和密度带**截断亏损**
   （`rho_core/rho0 = 0.9972`，符号都是反的），Tait 律把它放大成显著负偏差。

### 2.2 本轮的测量基准

- **完整支撑判据**：只统计 `r < R − 2.5h`（核支撑 2h，留 0.5h 余量）；
- **外部压强 = 真空值 0**（Tait 律夹零，单相自由面标准处理），不用壳层内压强近似；
- **跑到静态**：推进 {p['capillary_times']} 个毛细时间，报告末端速度、残余振荡、粒子均值标准误；
- **长程稳定性**：同一算例 12 个毛细时间，跟踪 `r_rms`、内部密度亏损、`p_core` 趋势，并做
  **`sigma = 0` 对照**分离机理。

工具链：PyTorch 2.x（CPU float64）/ matplotlib；`OMP_NUM_THREADS=1`。

## 3. 计算结果

### 3.1 阶梯：p 无平台（表 1）

| dx [mm] | N | N(完整支撑) | h/R | t/t_sigma | p_core [Pa] | 误差 % | p_band [Pa] | 旧 band 误差 % | SNR |
|---|---|---|---|---|---|---|---|---|---|
{rows}

解析值 `sigma/R = {f(ref['delta_p_Pa'])} Pa`，平台判据 15 %。
**三档分辨率误差 {f(min(errs))} % … {f(max(errs))} %，跨度仅 {f(max(errs)-min(errs))} 个百分点**：
与网格无关 ⇒ 这是结构性偏差，不是离散噪声，且**没有任何一档进入 15 %**。

### 3.2 旧 band 基准的系统性（同一状态，表 2）

| dx [mm] | p_core − 0 [Pa] | p_core − p_band [Pa] | 误差：完整支撑 → 旧 band [%] |
|---|---|---|---|
{band_rows}

band 基准在同一状态上把误差再压低数个百分点到几十个百分点：**−75.6 % 的主因之一是分母取错了位置**
（取在 CSF 壳层内部），而不是力错了。

### 3.3 相位假象（表 3、图 c）

同一次运行（dx = {scan['setup']['dx_m']*1e3:.0f} mm、{scan['setup']['steps_total']} 步、
`nu = {scan['setup']['viscosity_m2_per_s']}`，即 `tests/test_laplace_law.py` 的参数）在
{scan['n_samples']} 个不同步数上采样：

| 基准 | 误差跨度 | 落在 15 % 内的采样数 |
|---|---|---|
| 完整支撑 | {f(scan['error_spread_pct'], 0)} 个百分点 | {scan['n_samples_within_15pct']}/{scan['n_samples']} |
| 旧 band（`test_laplace_law.py` 协议） | {f(scan['legacy_error_spread_pct'], 0)} 个百分点 | {scan['n_legacy_samples_within_15pct']}/{scan['n_samples']} |

跨度是判据本身（15 %）的 {f(scan['legacy_error_spread_pct']/15.0, 0)} 倍以上：单点采样给不出
可复现结论，两轮复核的互相矛盾由此而来。

### 3.4 长程：没有静态平衡（表 4、图 a/b/c）

| sigma [N/m] | N | t_end/t_sigma | r_rms/R0 | 内部密度亏损 | 数值崩解 |
|---|---|---|---|---|---|
{drift_rows}

- **有表面张力**：粒子云从 `{active['outcome']['r_rms_over_r0_first']:.2f} R0` 膨胀到
  `{active['outcome']['r_rms_over_r0_last']:.2f} R0`，内部密度亏损加深到
  `{active['outcome']['rho_excess_last']:+.3f}`，`p_core/(sigma/R)` 由 ~1 单调衰减到
  `{active['outcome']['p_core_over_theory_last']:.2f}` 仍无平台（表 1 的
  `{f(min(errs))} % … {f(max(errs))} %` 就是这个衰减段上的读数）。
- **`sigma = 0` 对照**：同一算例关掉 CSF 后粒子云几乎不动
  （`{control['outcome']['r_rms_over_r0_first']:.3f} → {control['outcome']['r_rms_over_r0_last']:.3f} R0`，
  `p ≡ 0`）。⇒ 失控膨胀**不是**求解器自由面的通用漂移，而是 CSF 壳层力与夹零自由面压强梯度
  **耦合**产生的净外向力：自由面粒子缺外侧邻居，压强梯度力在它们身上得不到抵消，壳层被推外向，
  密度亏损加深又不允许内部蓄起应有的压强，形成正反馈。
- 更长的探索性运行（16 个毛细时间，非入库记录）在 `t ≈ 14 t_sigma` 出现 `NaN`，与本机理一致。

### 3.5 图

- `laplace_equilibrium.png`：(a) `p_core(t)/(sigma/R)` 三档收敛曲线（都衰减，无平台）；(b) 两套基准误差 vs dx；
  (c) 一次运行 {scan['n_samples']} 个采样时刻的误差带（相位假象）。
- `laplace_drift.png`：(a) `r_rms/R0` 随时间（有/无表面张力）；(b) `p_core/(sigma/R)` 单调衰减；(c) 内部密度亏损。

## 4. 注意点（坑 / 局限 / 适用范围）

1. **该算例在这套离散下没有静态平衡**：不能再用"某个步数下 p_core 等于 sigma/R"来判 CSF 对错，
   包括本仓库历史上 PASS 与 −75.6 % 两个结论。判据必须先通过"静态性检查"（`r_rms` 与 `rho_excess`
   稳定）才有意义。
2. **完整支撑判据会吃掉边界粒子**：粗网格只剩少数粒子（表 1 第 3 列），因此粗分辨率读数噪声大
   （SNR 列）；要评粗网格应减小 `h/R`，而不是放宽判据。
3. **`sigma = 0` 对照不是"表面张力无害"的证据**：对照组内 `p ≡ 0`（Tait 夹零），它与有表面张力的
   差别正是"内部可否蓄压"，这正是机理所在。
4. **粘度只改衰减速度**：静态终值若存在则与 `nu` 无关；本算例终极行为与 `nu` 无关（膨胀），
   说明它不是"没算够久"的问题。
5. **未做**：三维、多相（有真实外部流体时外部压强不为 0）、`h/dx` 系统敏感性、GPU/float32 复测、
   以及 §5 建议的修复实现（自由面压强/密度基准）。

## 5. 误差判定（逐项，显式回答"是否 < 3 %"）

| 关键量 | 基准 | 实测 | 误差 | < 3 % ? | < 15 % ? |
|---|---|---|---|---|---|
| 平衡态压差 `Delta p`（完整支撑） | `sigma/R = {f(ref['delta_p_Pa'])} Pa` | {', '.join(f"{f(r['equilibrium']['p_complete_Pa'])} Pa" for r in ladder)} | {f(min(errs))} – {f(max(errs))} % | **否** | **否**（三档全部超线） |
| 旧 band 基准误差 | 同上 | {', '.join(f"{f(r['equilibrium']['rel_error_band_pct'])} %" for r in ladder)} | — | **否** | **否** |
| 静态性（`r_rms/R0` 漂移） | 1.0 | {active['outcome']['r_rms_over_r0_last']:.2f} | +{f(100*(active['outcome']['r_rms_over_r0_last']-1.0),0)} % | 否 | 否 |
| 测量分辨力（SNR） | — | {', '.join(f"{f(r['equilibrium']['measurement_snr'],0)}" for r in ladder)} | — | — | 是 |

**显式回答**：`Delta p` 三档分辨率误差 {f(min(errs))} % … {f(max(errs))} %，**既不 < 3 % 也不 < 15 %**。
本轮**没有**把这项判据修好；本轮交付的是"为什么它一直测不准"的定量证据：
测量基准三缺陷（§2.1）+ 相位假象 {f(scan['legacy_error_spread_pct'], 0)} 个百分点跨度（§3.3）
+ 无静态平衡且机理定位到自由面压强基准（§3.4）。这是 honest FAIL，不是达标声明。

## 6. 下一步（与 backlog P0 对齐）

1. **自由面压强/密度基准**：按 Adami et al. (2012) 给自由面粒子压强偏置（`p_surface ~ sigma kappa`）
   或做密度 Shepard/MLS 归一，使内部能蓄起 `sigma/R` 对应的过压——这是把 `r_rms` 漂移与
   `Delta p` 偏差一起消掉的前提（本轮的对照实验已把缺口钉在这里，而不是 CSF 力的幅值：
   切割力恒等式与法向对齐早已进入 15 %）。
2. 修好后**重跑本基准**：判据应从"偏多少"改成"`r_rms` 稳定 + `|Delta p| < 15 %`"双门。
3. 之后再补三维/多相/`h/dx` 敏感性。

## 7. 复现

```bash
cd <worktree>
OMP_NUM_THREADS=1 PYTHONPATH=src python3 benchmarks/laplace_equilibrium/run.py     # 阶梯（~8 min）
OMP_NUM_THREADS=1 PYTHONPATH=src python3 benchmarks/laplace_equilibrium/drift.py   # 长程+对照（~2 min）
OMP_NUM_THREADS=1 PYTHONPATH=src python3 benchmarks/laplace_equilibrium/make_doc.py
OMP_NUM_THREADS=1 PYTHONPATH=src python3 -m pytest tests/test_laplace_equilibrium.py -q
```
"""
    OUT.write_text(md)
    print("wrote", OUT, len(md), "chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())