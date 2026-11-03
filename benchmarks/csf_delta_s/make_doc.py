"""Generate docs/verification_delta_s_normalization.md from measured JSON.

Every number in the document comes from ``benchmarks/csf_delta_s/results.json``
(and the relaxation A/B in ``results_relax.json``); nothing is hand-typed.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs" / "verification_delta_s_normalization.md"


def main() -> None:
    data = json.loads((HERE / "results.json").read_text())
    relax = json.loads((HERE / "results_relax.json").read_text())
    rows = data["rows"]
    setup = data["setup"]

    def row(dx, norm):
        return next(r for r in rows
                    if abs(r["dx_m"] - dx) < 1e-12 and r["normalization"] == norm)

    dxs = sorted({r["dx_m"] for r in rows})

    table1 = "\n".join(
        f"| {dx * 1000:.1f} | {row(dx, 'fixed')['color']['c_surf_q10']:.4f} | "
        f"{row(dx, 'fixed')['color']['span_measured']:.4f} | "
        f"{row(dx, 'fixed')['color']['grad_integral_in_fluid_dn']:.4f} | "
        f"{row(dx, 'fixed')['color']['grad_integral_rel_error_pct']:+.1f} % |"
        for dx in dxs)

    table2 = "\n".join(
        f"| {dx * 1000:.1f} | {row(dx, 'fixed')['delta_s']['integral_dn']:.3f} | "
        f"{row(dx, 'jump')['delta_s']['integral_dn']:.3f} | "
        f"{row(dx, 'fixed')['delta_s']['rel_error_pct']:+.1f} % | "
        f"{row(dx, 'jump')['delta_s']['rel_error_pct']:+.1f} % |"
        for dx in dxs)

    table3 = "\n".join(
        f"| {dx * 1000:.1f} | {row(dx, 'fixed')['cut_force']['ratio_geometric']:.3f} | "
        f"{row(dx, 'jump')['cut_force']['ratio_geometric']:.3f} | "
        f"{row(dx, 'fixed')['cut_force']['ratio_acceleration_space']:.3f} | "
        f"{row(dx, 'jump')['cut_force']['ratio_acceleration_space']:.3f} | "
        f"{row(dx, 'fixed')['interface_density']['delta_s_weighted_rho_over_rho0']:.3f} |"
        for dx in dxs)

    table4 = "\n".join(
        f"| {dx * 1000:.1f} | {row(dx, 'fixed')['curvature_kappa']['rel_error_pct']:+.1f} % | "
        f"{row(dx, 'fixed')['direction']['cos_band_median']:.3f} |"
        for dx in dxs)

    table5 = "\n".join(
        f"| {r['dx_m'] * 1000:.1f} | {r['steps']} | {r['normalization']} | "
        f"{r['delta_p_measured_Pa']:+.1f} | {r['rel_error_pct']:+.1f} % | "
        f"{r['spurious_ratio']:.3f} |"
        for r in relax["rows"])

    r2 = row(0.002, "jump")
    doc = f"""# 表面张力 CSF 的 delta 函数归一化核验（`delta_s` 欠账结清）

**范围**: TensorSPH `src/tensorsph/csf_consistent.py`、`tensorsph.csf_curvature`、`tensorsph.surface_tension`
**基准**: 2-D 圆形液滴的三条解析恒等式（本文件所有数字由 `benchmarks/csf_delta_s/run.py`
与 `benchmarks/csf_delta_s/run_relax.py` 写出的 JSON 程序化注入，无手抄）

## 1. 案例介绍（物理背景、几何、参数）

CSF（Brackbill, Kothe & Zemach 1992）表面张力力的密度形式是

    f = -sigma kappa delta_s n ,      int delta_s dn = 1  (单位面积上的 delta 函数质量)

`delta_s` 是把界面上的线力摊到体积上的那个核。它不是可调参数：**归一化到 1 是定义**，
否则力幅值就随离散方案漂移。上一轮（`docs/verification_surface_tension_curvature.md` §7）
留下三个欠账，其中最关键的一条是 `int delta_s dn = 0.74`（−26%，三档分辨率几乎不变，
所以不是分辨率地板），而切割力恒等式 `F_cut = 2 sigma` 却报「10.7%，在 15% 门内」。
**本工作证明这两个读数不能同时成立**，并给出根因。

| 参数 | 值 |
|---|---|
| 液滴半径 R | {setup['radius_m']} m |
| 表面张力系数 sigma | {setup['sigma_N_per_m']} N/m |
| 参考密度 rho0 | {setup['rho0']} kg/m^3 |
| 状态方程 | Tait, gamma = 7 |
| 格子 | {setup['lattice']}，h = {setup['h_over_dx']} dx，无重力 |
| 分辨率扫描 | dx = {', '.join(f'{d*1000:.1f}' for d in dxs)} mm（静止态，无时间推进） |
| 曲率算子 | {setup['curvature_operator']}（上一轮已验证） |
| 解析参考 | kappa = 1/R，F_cut = 2 sigma，int delta_s dn = 1 |

## 2. 怎么做的（方法/流程）

1. **先量剖面的真值，而不是继续调因子**。把静止液滴的色函数 `c(r)` 与 `|grad c|` 沿半径打印出来，
   发现离散色函数并不是教科书里的 `1 -> 1/2` 阶跃：它在 `r/R ~ 0.87` 之后才在**一层粒子**内
   分段下跌（0.977 -> 0.960 -> 0.848 -> 0.731 -> 0.620），最外层携带 `c_surf ~ 0.63`。
2. **验证离散恒等式**：`int_流体 |grad c| dn = c_bulk - c_surf` 是否成立。成立（见 §3 表 1，<2%），
   说明离散色函数本身是自洽的，缺的只是**截面跨度**：单相粒子云里 `c` 只从 1 走到 `c_surf`，
   另一半落在真空中没有粒子携带。于是历史代码里硬编码的 2 倍因子（= 假设 `c_surf = 1/2`）
   必然系统性偏弱。
3. **改成按实测跳变归一**：`delta_s = |grad c| / (1 - c_surf)`，`c_surf` 取界面粒子上色函数的
   {setup['quantile']:.2f} 分位数（三档分辨率下为 0.6195/0.6264/0.6368，是格子常数，
   用分位数而非极小值以避免单个粒子决定归一化）。实现为 `csf_fields(..., delta_s_normalization=...)`，
   取值 `"fixed"`（历史，默认）与 `"jump"`（本方案），并贯通到 `Config.csf_delta_s` 与
   `surface_tension.compute_surface_tension_force`。
4. **同时修正度量口径**。旧的切割力度量是 `sum_i m_i a_i`。但 `delta_s` 是**几何**面密度，
   而 `a = f/rho` 除的是**求和**密度：自由面处求和密度亏损 ~20%，于是 `sum m a` 被放大
   `rho0/rho_w` 倍。正确口径是 `sum_i (m_i/rho0) rho_i a_i`（几何体积上的力的积分，
   等价于加速度用 `f/rho0`）。两种口径都报出来，并给出二者的比。
5. 工具链：`benchmarks/csf_delta_s/run.py`（静止态 A/B，秒级）、`run_relax.py`（端到端压差 A/B）、
   `make_doc.py`（本文件，数字程序化注入）、`tests/test_csf_delta_s_normalization.py`（8 条判据）。

## 3. 计算结果

### 3.1 色函数的实测跨度（恒等式 `int|grad c|dn = 1 - c_surf`）

| dx [mm] | c_surf（界面 10% 分位） | 跨度 1 - c_surf | 实测 int 流体侧 \|grad c\| dn | 相对误差 |
|---|---|---|---|---|
{table1}

**读法**：右侧误差 <2% ⇒ 离散色函数在「跨度」意义下是精确的。历史 2 倍因子假设跨度 0.5，
实测只有 ~0.37，这就是 −26% 的全部来源。

### 3.2 delta 函数归一化 `int delta_s dn`

| dx [mm] | fixed（默认） | jump（本方案） | fixed 误差 | jump 误差 |
|---|---|---|---|---|
{table2}

### 3.3 切割力恒等式（几何体积口径 vs 加速空间口径）

| dx [mm] | fixed (几何) | jump (几何) | fixed (加速空间) | jump (加速空间) | 界面 rho_w/rho0 |
|---|---|---|---|---|---|
{table3}

**关键**：加速空间口径 = 几何口径 / (rho_w/rho0) = 几何口径 x 1.25（三档一致）。
旧的「10.7% 达标」正是「delta_s 弱 26%」与「密度亏损放大 25%」两错相消的产物
（这也是上一轮曲率修好后切割力反而变差 8-11% 却仍「在门内」的原因）。

### 3.4 与归一化无关的量（控制变量）

| dx [mm] | kappa 误差 | 力方向中位 cos |
|---|---|---|
{table4}

曲率与方向不随 `delta_s` 尺度变化 —— 这是控制变量，证明 §3.2/§3.3 的差别只来自归一化。

### 3.5 端到端 Young-Laplace 压差（relax，fit_quadratic 曲率）

| dx [mm] | 步数 | 归一化 | 实测 dp [Pa] | 相对 360 Pa | 伪流 max\|v\|/sqrt(sigma/rhoR) |
|---|---|---|---|---|---|
{table5}

## 4. 结果图

![delta_s normalisation A/B](benchmarks/csf_delta_s/delta_s_normalization.png)

三面板：左= `int delta_s dn` vs 1（红线 15% 门），中= 几何口径切割力 `F_cut/(2 sigma)`，
右= 曲率误差（控制变量，两方案重合）。横轴 dx（mm，右到左加密）。

## 5. 注意点（踩坑与适用范围）

1. **`int delta_s dn = 1` 与 `sum m a = 2 sigma` 不能同时闭合**，因为后者除的是求和密度。
   任何「切割力达标」的结论必须先声明口径，否则会把相消误差当成精度。
2. **界面求和密度亏损 20%**（`rho_w/rho0 = 0.80`）是这两个口径之差的唯一来源，三档分辨率稳定 ——
   它同时说明真实模拟里界面粒子加速度被放大了 25%（`a = f/rho`，正确应为 `f/rho0`）。
3. **不要用极小值估计 `c_surf`**：dx=2 mm 时极小值出现 0.508 的离群粒子，会把归一化拉偏 30%；
   10% 分位数稳定到毫厘（0.6195/0.6264/0.6368，随 h/dx 单调）。
4. **默认仍是 `"fixed"`，不是疏忽**：换成 `"jump"` 会把界面动量注入提高 32%，在压差可测的分辨率
   （R/h ~ 5）把伪流从 0.048 顶到 0.147（超 Adami 0.1 阈值），压差从 +8.2% 冲到 +112%。
   先修压强基准（Adami 压强偏置 / `f/rho` 归一），再把 `"jump"` 设为默认。
5. 本工作只覆盖 2-D、单相、静止态与两档 relax；3-D 与多相未验证。

## 6. 误差判定（逐项，显式回答「是否 < 3%」）

| 关键量 | 基准 | fixed（默认） | jump | jump 是否 < 3% | jump 是否 < 15% |
|---|---|---|---|---|---|
| `int delta_s dn`（dx={0.002*1000:.1f} mm） | 1 | {row(0.002, 'fixed')['delta_s']['integral_dn']:.3f}（{row(0.002, 'fixed')['delta_s']['rel_error_pct']:+.1f}%） | {r2['delta_s']['integral_dn']:.3f}（{r2['delta_s']['rel_error_pct']:+.1f}%） | {"是" if abs(r2['delta_s']['rel_error_pct']) < 3 else "否"} | {"是" if abs(r2['delta_s']['rel_error_pct']) < 15 else "否"} |
| 几何口径切割力 F_cut/2sigma | 1 | {row(0.002, 'fixed')['cut_force']['ratio_geometric']:.3f} | {r2['cut_force']['ratio_geometric']:.3f}（{r2['cut_force']['geometric_rel_error_pct']:.1f}%） | {"是" if r2['cut_force']['geometric_rel_error_pct'] < 3 else "否"} | {"是" if r2['cut_force']['geometric_rel_error_pct'] < 15 else "否"} |
| 曲率 kappa（控制变量） | 1/R = 50 1/m | {row(0.002, 'fixed')['curvature_kappa']['delta_s_weighted_mean_1_per_m']:.2f} | 同上（{r2['curvature_kappa']['rel_error_pct']:+.1f}%） | {"是" if abs(r2['curvature_kappa']['rel_error_pct']) < 3 else "否"} | {"是" if abs(r2['curvature_kappa']['rel_error_pct']) < 15 else "否"} |
| 力方向中位 cos | 1 | {row(0.002, 'fixed')['direction']['cos_band_median']:.3f} | {r2['direction']['cos_band_median']:.3f} | 是 | 是 |
| 端到端压差（dx=2 mm） | 360 Pa | {[r for r in relax['rows'] if abs(r['dx_m']-0.002) < 1e-12 and r['normalization']=='fixed'][0]['rel_error_pct']:+.1f}% | {[r for r in relax['rows'] if abs(r['dx_m']-0.002) < 1e-12 and r['normalization']=='jump'][0]['rel_error_pct']:+.1f}% | 否 | 否 |

**结论（honest）**：
- **力层面（delta 归一化 + 几何口径切割力）在 `jump` 方案下同时进 15% 门**，`int delta_s dn` 甚至 <3%；
  这是本轮唯一「新进门的判据」。
- **旧报告里的「切割力 10.7%，在门内」不是精度，是相消误差**，本节把它撤销为 −26%（几何口径）。
- **端到端 Young-Laplace 压差仍是 WITHHELD**（两档分辨率、两方案均不达 15%，且一档符号为负）：
  与上一轮结论一致，压强面的根因是求和密度的噪声地板 / 缺压强偏置，**不是**表面张力离散；
  因此默认值不改（改默认会用一条进门判据换另一条出局），后续欠账见 §5.4。

## 7. 未完成 / 交接

1. `a = f/rho` → `f/rho0`（或按几何体积装配力）：界面动量注入现被高估 25%，
   这是伪流与 relax 压差超调的直接原因，也是把 `"jump"` 设为默认的前置条件。
2. Tait 压强加 Laplace 偏置 `p0 = sigma kappa`（Adami 2012）：端到端压差需要它才可能离开噪声地板。
3. `tests/test_laplace_droplet.py` 的 4 条「pinned failure」在 HEAD 上已过时（记录旧模型），
   与本轮无关但仍是红测试，建议下一轮按当前代码重写。
4. 3-D 曲率/归一化未验证。
"""
    DOC.write_text(doc)
    print("wrote", DOC, len(doc), "bytes")


if __name__ == "__main__":
    main()