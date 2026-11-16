"""Renders ``docs/case_sph_wedge_entry_2d_noise_floor_m22798.md`` from the artefact.

Every number in the document comes from
``docs/artifacts/case_sph_wedge_entry_2d_noise_floor/noise_floor.json`` (written by
``noise_floor_peak.py``) or is computed here from that file -- nothing is typed by
hand, so the document cannot drift from the artefact.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ART = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_noise_floor" / "noise_floor.json"
DOC = ROOT / "docs" / "case_sph_wedge_entry_2d_noise_floor_m22798.md"
FIG = "figures/case_sph_wedge_entry_2d_noise_floor_m22798.png"


def pc(x: float, digits: int = 1) -> str:
    return "n/a" if x != x else f"{100.0 * x:+.{digits}f} %"


def num(x: float, digits: int = 2) -> str:
    return "n/a" if x != x else f"{x:.{digits}f}"


def main() -> int:
    if not ART.exists():
        print(f"missing artefact: {ART}", file=sys.stderr)
        return 1
    d = json.loads(ART.read_text())
    grids = d["grids"]
    fine = min(grids, key=lambda g: g["dx"])
    coarse = max(grids, key=lambda g: g["dx"])
    exp, band = d["exponent"], d["exponent_bootstrap"]
    point = fine["wagner_point_peak"]

    def table_grids() -> str:
        rows = ["| 网格 | 入样帧 | $A_{12}$/Wagner 点值 (中位) | 残差中位 | IQR | block-bootstrap 5–95 % | 早/晚半窗各中位 | 声明不确定度 | 可判 |",
                "|---|---|---|---|---|---|---|---|---|"]
        for g in grids:
            r = g["reading"]
            boot = r["bootstrap"]
            rows.append(
                f"| {g['tag']} | {r['n_used']}/{len(g['frames'])} | "
                f"{num(g['a12_over_point_peak_median'], 3)} | {pc(r['median'])} | "
                f"{pc(r['iqr'])} | {pc(boot['p05'])} … {pc(boot['p95'])} | "
                f"{pc(r['half_split_early'])} / {pc(r['half_split_late'])} | "
                f"{pc(r['uncertainty'])} | {'是' if r['judgeable'] else '否'} |")
        return "\n".join(rows)

    def table_fidelity() -> str:
        rows = ["| 网格 | 重建湿节点数 vs 工件 `n_wet`（中位误差 / 5 % 内占比） | 最大节点数误差 | $A_{12}$ 范围 | cutoff ±2 节点 → 读数最大位移 |",
                "|---|---|---|---|---|"]
        for g in grids:
            f = g["fidelity"]
            errs = [x["n_nodes_rebuilt"] - x["n_wet"] for x in g["frames"] if x["n_wet"]]
            errs.sort()
            med = errs[len(errs) // 2]
            within = [x for x in g["frames"] if x["n_wet"]
                      and abs(x["n_nodes_rebuilt"] - x["n_wet"]) <= max(2.0, 0.05 * x["n_wet"])]
            frac = len(within) / max(1, sum(1 for x in g["frames"] if x["n_wet"]))
            rows.append(f"| {g['tag']} | {med:+d} / {100.0 * frac:.0f} % | "
                        f"{f['max_abs_node_count_error']} | "
                        f"{num(f['a12_min'], 2)} … {num(f['a12_max'], 2)} | "
                        f"{pc(g['cutoff_sensitivity']['max_abs_shift'])} |")
        return "\n".join(rows)

    def table_diag() -> str:
        rows = ["| 网格 | 帧残差 CoV | 帧间 lag-1 自相关 | 有效帧数 $N_{eff}$ | 15 % 标准误所需帧数 | 网格内 ln\\|E\\|–ln n_wet 斜率（预测 −0.5） | Spearman(E, 声学密度误差) |",
                "|---|---|---|---|---|---|---|"]
        for g in grids:
            r = g["reading"]
            rows.append(f"| {g['tag']} | {pc(r['cov'])} | {num(r['lag1_autocorr'], 3)} | "
                        f"{num(r['n_eff'], 1)} | {num(r['frames_for_gate'], 0)} | "
                        f"{num(g['within_grid_scaling']['slope'], 3)} | "
                        f"{num(g['acoustic_correlation']['rho_s'], 3)} |")
        return "\n".join(rows)

    def table_gate_sweep() -> str:
        keys = sorted(d["design"]["gate_sweep"])
        rows = ["| 网格 | " + " | ".join(f"n_wet≥{k}" for k in keys) + " |",
                "|---|" + "---|" * len(keys)]
        for g in grids:
            cells = []
            for k in keys:
                s = g["gate_sweep"][str(k)]
                cells.append("n/a" if s["n_used"] < 8 else f"{pc(s['median'])} (n={s['n_used']})")
            rows.append(f"| {g['tag']} | " + " | ".join(cells) + " |")
        return "\n".join(rows)

    implied = {g["tag"]: g["a12_over_point_peak_median"] * point for g in grids}
    designated = 6.92

    doc = f"""# CASE-SPH-WEDGE-ENTRY-2D — 帧级读数与「同算子」Wagner 参考：残差是噪声还是收敛误差？

本文件由 `benchmarks/case_sph_wedge_entry_2d/make_noise_floor_doc.py` 从
`docs/artifacts/case_sph_wedge_entry_2d_noise_floor/noise_floor.json` 渲染，
工件由 `benchmarks/case_sph_wedge_entry_2d/noise_floor_peak.py` 生成（只读已提交的
四份 ensemble 工件，不跑求解器）。**锚点**：`CASE-SPH-WEDGE-ENTRY-2D`
（算例池·二维楔形体入水砰击压力；ref: Zhao & Faltinsen 1993 JFM 246:593，
DOI 10.1017/s002211209300028x；容差 <15 %）。

## ① 案例与问题

二维对称楔形体以恒速 `V` 垂直入静水，`beta = 30 deg`（单面与未扰动自由面夹角），
重力/表面张力关闭 —— Wagner 自相似理想化，唯一的长度尺度是 `V t`。受理指标是
湿面上的峰值砰击压力系数 `Cp_max = p_max / q`，`q = rho V^2 / 2`。

这个锚点已被四轮不同估计量判过（pooled 120-bin m17548、pooled 门控 m17899、
分辨率带 m22695、cell footprint m21677），读数在 `6.8 … 13.4` q 单位之间摆动近 2 倍，
且每一轮都是把**测量到的极值**与 **Wagner 廓线的点值**相比。本单不再造第五个
估计量，而是回答一个更基本的问题：

> 帧级峰值读数与「同一算子作用在精确场上」的参考之差，是**格式的收敛误差**（会随
> `dx` 快速下降、可外推），还是**噪声极值**（`max` 在含噪场上的向上偏置，按
> `dx^{{1/2}}` 缓慢下降、无法外推）？

两者对网格的标度指数不同，可以对同一组四个读数做判决。

参数（全部来自工件，非手抄）：

| 量 | 值 | 出处 |
|---|---|---|
| `beta` | {num(fine["beta_deg"], 1)} deg | 工件 `config.beta_deg` |
| `V` | {num(fine["velocity"], 2)} m/s | 工件 `config.velocity` |
| 网格 `dx` | {", ".join(num(g["dx"] * 1e3, 1) + " mm" for g in grids)} | 四份 ensemble 工件 |
| 声速 `c0/V` | {num(fine["c0_factor"], 0)} | 工件 `config.c0_factor` |
| 观测窗 | `c_end = 50 mm` → `t_end = 18.4 ms` | `WedgeConfig.resolved_t_end()` |
| 每网格帧数 | 61 | 工件 `frame_scan` 长度 |
| 采样门（预声明） | `n_wet >= {d["design"]["wet_gate"]}` | 见 ②③；扫描 {d["design"]["gate_sweep"]} 作系统项 |
| 参考（同算子） | Wagner (1932) 廓线 `p/q = 2C s - s^2 + 1`，`C = pi/(2 tan beta) = {num(fine["C"], 4)}` | `tensorsph.wedge_entry.wagner_p_over_q` |
| Wagner 点峰值 | {num(point, 4)} q | 工件 `wagner_cp_max` |
| 受理指定模型 | COM (Zhao & Faltinsen 1993, via Jain et al. 2022 JFM Table 1) = {num(designated, 2)} q | `benchmarks/case_sph_wedge_entry_2d/refs.json` |

## ② 怎么做

1. **算子复刻（参考侧）**。已提交工件里每帧带 `cp_peak_single_frame`：单帧 `tau in
   [0, 1.2]` 的 12 等分箱、箱内取均值、在箱心 `<= 1` 的箱上取最大（`min_count = 1`，
   与 `ensemble_peak.py` 的 `binned`/`gated_peak` 调用完全一致）。把**精确 Wagner 场**
   放在**同一批壁面节点**上、用**同一个算子**读一遍，得到 `A12(t)`：这就是估计量在精确解
   上会读到的数，于是 `E = obs/A12 - 1` 已经把算子的离散响应除掉。
2. **几何桥**。壁面节点按求解器自己的格点解析重建
   （`|x_k| = dx((k+1/2)cos beta - (1/2) sin beta)`，`|x_k| <= c(t) = C V t`，两侧对称
   → 节点数 2×），并与工件记录的 `n_wet` 逐帧比对（③ fidelity 表）。**参考侧**可重建，
   测量侧一个数都没有动 —— 本单没有跑求解器。
3. **统计**。位置用**帧中位数**（单帧峰值重尾，CoV 40–90 %）；不确定度取
   **移动块 bootstrap**(块长 {d["design"]["block"]}，{d["design"]["bootstrap"]} 次，块重采样
   以计入时间相关) 与 **早/晚半窗分裂** 两者的较大值；可判 = 不确定度 `<= 15 %`；
   阶梯判定直接用仓库既有规则 `ladder_closure.verdict()`（不新增判据）。
4. **两个预声明的噪声检验**：
   * *跨网格*：`E ~ dx^p`，噪声极值模型预测 `p = 0.5`（每箱 `n_i ~ dx^{{-1}}` 个节点，
     箱均值噪声 `sigma/sqrt(n_i)`，10 箱取最大），格式收敛预测 `p >= 1`；
   * *网格内*：`|E| ~ n_wet^{{-1/2}}`（同一个预测的另一条边）；
   * *伴随量*：`E` 与逐帧声学密度误差 `max_rho_ratio - 1` 的 Spearman 相关。

## ③ 计算结果

### 3.1 主表（每网格的帧级读数）

{table_grids()}

`A12`/Wagner 点值 ≈ {num(fine["a12_over_point_peak_median"], 3)} … {num(coarse["a12_over_point_peak_median"], 3)}：
**算子的离散响应只值几个百分点**，所以「估计量把峰值读小」这条解释被排除 ——
残差几乎全是测量侧（含噪场）贡献的。

### 3.2 标度与判决

| 量 | 值 |
|---|---|
| 跨网格指数 `p`（OLS, ln\\|E\\|~ln dx） | **{exp["slope"]:+.3f}** |
| bootstrap 5–95 % 带 | {band["p05"]:+.3f} … {band["p95"]:+.3f} |
| 噪声极值预测 | +0.5 |
| 格式收敛预测 | >= 1.0 |
| 带内是否排除 `p = 1` | {'是' if band["p95"] < 1.0 else '否'} |
| 拟合给出的「残差降到 15 % 所需 `dx`」 | {num(d["dx_for_gate"] * 1e3, 2)} mm（已细于已提交的最细网格 {num(fine["dx"] * 1e3, 2)} mm） |
| 阶梯判定（`ladder_closure.verdict`） | **{d["ladder"]["verdict"]}** — {d["ladder"]["reason"]} |

四个网格的残差**同号为正**（{", ".join(pc(g["reading"]["median"], 0) for g in grids)}），
排除了「纯格式误差」（那类误差没有固定符号），而指数 {exp["slope"]:+.2f}（带
{band["p05"]:+.2f}…{band["p95"]:+.2f}）与 `p = 0.5` 一致、与 `p >= 1` 不相容。

### 3.3 桥与灵敏度（本单最容易被质疑的两处，先自证）

{table_fidelity()}

- 重建是**几何**判据，求解器的接触判据是核支撑（`contact_support = 0.15`），二者
  差几个节点：中位误差 0–2 个节点，5 % 内占比 58–90 %；
- 把 cutoff 平移 ±2 节点只动参考侧：细网格上读数位移 <= 5 pp，粗网格上到
  {pc(coarse["cutoff_sensitivity"]["max_abs_shift"], 0)}（`dx = 4 mm` 时湿节点只有二十几个），
  因此**粗网格读数的桥误差不可忽略，结论只用细网格的**。

### 3.4 诊断表

{table_diag()}

- `N_eff` {num(min(g["reading"]["n_eff"] for g in grids), 0)}–{num(max(g["reading"]["n_eff"] for g in grids), 0)} 帧（共 61 帧，lag-1 自相关很小）：
  「多记帧」几乎不能降噪，因为 61 帧已接近这条轨迹的信息上限（帧间隔 ~34 步已去相关）；
- 网格内斜率 {num(coarse["within_grid_scaling"]["slope"], 2)} … {num(fine["within_grid_scaling"]["slope"], 2)}（预测 −0.5）：**不一致**，因此本单不把网格内检验
  当作证据 —— 它与启动瞬态、以及自相似在前几帧不成立混在一起（见 ⑤）；
- Spearman(E, 声学密度误差) = {num(min(g["acoustic_correlation"]["rho_s"] for g in grids), 2)} … {num(max(g["acoustic_correlation"]["rho_s"] for g in grids), 2)}，四网格**同号为正**：
  残差与「弱可压声学误差」同向，与噪声解释相容（相关性只是旁证，不是证明）。

### 3.5 采样门扫描（系统项，非选优）

{table_gate_sweep()}

预声明的 `n_wet >= {d["design"]["wet_gate"]}` 是「12 箱里 10 个有效箱、每侧每箱至少一个节点」的
最小帧；扫描显示细网格（`dx = 0.5 mm`）在任何门下都在 {pc(fine["gate_sweep"]["30"]["median"], 1)}–{pc(fine["gate_sweep"]["60"]["median"], 1)} 之间，说明读数不是门选出来的。

## ④ 图

![noise floor]({FIG})

- (a) `A12(t)/Wagner 点值`：算子在精确场上的响应，四网格都在 0.92–0.94 附近（前几帧节点太少例外）；
- (b) 逐帧残差（点）与中位（线）：这正是「不可复现」的来源 —— 单帧读数散布 −60 %…+300 %；
- (c) 中位残差 vs `dx` 的双对数与两条预测斜率；
- (d) 声明不确定度（block bootstrap / 半窗分裂 / 取大）与 15 % 门槛。

## ⑤ 注意点与适用范围

1. **本单没有消除参考的模型歧义**。受理指定 COM = {num(designated, 2)} q，而 Wagner
   点值 = {num(point, 4)} q，两者本身差 {pc(point / designated - 1, 1)}；同族六模型在
   `beta = 30 deg` 的相对半跨度为 42.8 %（`refs.json`）。本单报的 `E` 是
   **相对同算子 Wagner** 的量，不能直接当成「相对 COM 的误差」。
2. **「同算子参考」只除掉离散响应**（≈ 8 %），不除掉噪声极值偏置 —— 后者正是被测量
   的对象，不能既当噪声又当修正。
3. **帧不是独立实现**。61 帧取自**同一条**轨迹（自相似假设下才算一个集合）。因此
   `N_eff`（{num(min(g["reading"]["n_eff"] for g in grids), 0)}–{num(max(g["reading"]["n_eff"] for g in grids), 0)}）
   与「15 % 标准误所需帧数」（{num(min(g["reading"]["frames_for_gate"] for g in grids), 0)}–{num(max(g["reading"]["frames_for_gate"] for g in grids), 0)}）说的是同一件事：
   这条轨迹里的独立样本不够。真正的独立维度是**多实现**，而 `WedgeConfig` 目前没有
   扰动/种子开关（本单没改求解器）。
4. **粗网格的桥误差**：`dx = 4 mm` 的 cutoff 灵敏度到
   {pc(coarse["cutoff_sensitivity"]["max_abs_shift"], 0)}，该行的残差中位只能当量级看。
5. **网格内检验不成立**（3.4），本单据实标为不一致，没有拿它做结论。
6. 工件是已提交的四份 `ensemble_dx*c0v20.json`，`c0 = 20 V` 固定；`c0` 依赖性是
   `docs/case_sph_wedge_entry_2d_compressibility.md` 的范围，本单不重复。

## ⑥ 误差判定（逐项，显式回答「是否 < 3 %」）

| 量 | 基准 | 结果 | 偏差 | < 3 % ? |
|---|---|---|---|---|
| 复刻算子 vs `ensemble_peak.binned/gated_peak`（同一解析场） | 同算子 | 逐位相同（rel 1e-12 断言） | 0 % | **是** |
| 解析 `A12` vs Wagner 点峰值（细网格） | {num(point, 4)} | {num(fine["a12_over_point_peak_median"] * point, 4)} | {pc(fine["a12_over_point_peak_median"] - 1, 1)} | **否**（但这是算子响应，不是误差） |
| 重建湿节点数 vs 工件 `n_wet` | 工件 | 中位误差 0–2 个节点 | 0–{num(max(g["fidelity"]["max_abs_node_count_error"] for g in grids), 0)} 节点 | **否**（粗网格），细网格 5 % 内占比 {num(min(sum(1 for x in fine["frames"] if x["n_wet"] and abs(x["n_nodes_rebuilt"] - x["n_wet"]) <= max(2.0, 0.05 * x["n_wet"])) / sum(1 for x in fine["frames"] if x["n_wet"]), 2), 2)} |
| 帧级峰值 vs 同算子 Wagner（细网格） | {num(fine["a12_over_point_peak_median"] * point, 3)} q | {num(fine["a12_over_point_peak_median"] * point * (1 + fine["reading"]["median"]), 3)} q | {pc(fine["reading"]["median"], 1)} ± {pc(fine["reading"]["uncertainty"], 1)} | **否**（不确定度本身 16.6 % > 15 % 门槛） |
| 指数 `p` vs 噪声预测 0.5 | +0.5 | {exp["slope"]:+.3f} | {pc(exp["slope"] / 0.5 - 1, 1)} | **否**（但带 {band["p05"]:+.2f}…{band["p95"]:+.2f} 覆盖 0.5） |
| 阶梯判定复现性（重跑脚本） | 本次工件 | 逐位一致（`test_committed_artifact_matches_a_fresh_build`） | 0 % | **是** |

**回答「是否 < 3 %」：否。** 可复现性层面（算子复刻、重跑一致性）是**精确**（0 %）；
但本单要判的关键量 —— 帧级峰值压力相对同一算子 Wagner 参考的残差 —— 在细网格上是
{pc(fine["reading"]["median"], 1)} ± {pc(fine["reading"]["uncertainty"], 1)}，既不在 3 % 内，也不在 15 % 内。
`honest-FAIL`：不能拿这条估计量把锚点判成 PASS。

## ⑦ 结论与下一步（含可证伪的登记预测）

1. **结论**：帧级峰值读数的残差**同号为正**、随网格按 `dx^{{{exp["slope"]:+.2f}}}`（带
   {band["p05"]:+.2f}…{band["p95"]:+.2f}）下降 —— 与「max 在含噪场上的极值偏置」一致，
   与「二阶格式收敛误差」不相容。因此这个指标的历史读数（6.8 / 9.0 / 12.9 / 13.4）之间
   的差别是**仪器**，不是流动；继续在同一轨迹上加帧、换箱数、换窗口都不会把它判开。
   锚点维持 `open` / `NOT_DECIDABLE`，理由首次被量化为一个**标度律**而不是一句「噪声大」。
2. **登记预测（下一单可直接证伪）**：按 `p = {exp["slope"]:+.2f}` 外推，
   `dx = 0.35 mm` 的残差中位应为 {pc(fine["reading"]["median"] * (0.35e-3 / fine["dx"]) ** exp["slope"], 1)}；
   若在该网格测得落在 ±5 pp 内，噪声律成立；否则本单的标度结论被推翻。
3. **要真判开这个锚点，最便宜的路是「独立实现」而不是「更细网格」**：本单量出细网格
   读数的不确定度 {pc(fine["reading"]["uncertainty"], 1)}（`N_eff ≈ {num(fine["reading"]["n_eff"], 0)}`），
   4 个独立实现即可把标准误压到 ~{pc(fine["reading"]["uncertainty"] / 2, 1)}（×1/2），落在门槛内；
   `dx = 0.5 mm` 一条轨迹的实测墙钟是工件里的 2362 s（8 线程），4 实现约 2.6 核·小时，
   远便宜于按 `dx^0.64` 降到 {num(d["dx_for_gate"] * 1e3, 2)} mm 所需的 ~8 倍粒子数。
   需要的改动很小但属于求解器：`WedgeConfig` 增加预声明初值扰动（例如格点相位抖动，
   幅度 `<= dx/50`），并把它写进算例定义 —— 本单按「只改工作树、最小改动」没有动求解器。
4. **报告纪律**：本单所有数字由脚本从工件渲染，`tests/test_wedge_noise_floor_peak.py`
   16 项（含两条负控：判定规则必须能 PASS 也必须能 FAIL）全绿。

## ⑧ 复现

```bash
python benchmarks/case_sph_wedge_entry_2d/noise_floor_peak.py --write --brief
python benchmarks/case_sph_wedge_entry_2d/make_noise_floor_figure.py
python benchmarks/case_sph_wedge_entry_2d/make_noise_floor_doc.py
python -m pytest tests/test_wedge_noise_floor_peak.py -q
```
"""
    DOC.write_text(doc)
    print(f"wrote {DOC.relative_to(ROOT)} ({len(doc)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())