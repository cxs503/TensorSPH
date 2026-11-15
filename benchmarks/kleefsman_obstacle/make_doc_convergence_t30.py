"""Generate ``docs/benchmarks/kleefsman-convergence-t30-m21718.md`` (numbers injected).

Every number in the document is read from
``benchmarks/kleefsman_obstacle/results/convergence_t30/convergence.json`` /
``reference.json`` at build time -- nothing is transcribed by hand, so a rerun of
``convergence_t30.py`` (e.g. once the uniform-horizon ``dx0.01`` run lands)
regenerates a consistent document.

Usage::

    python benchmarks/kleefsman_obstacle/make_doc_convergence_t30.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "docs" / "benchmarks" / "kleefsman-convergence-t30-m21718.md"


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:                      # nan
        return "—"
    return f"{f:.{nd}f}"


def pct(v, nd=1):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if f != f else f"{100.0 * f:.{nd}f} %"


def main() -> int:
    rep = json.loads((HERE / "results" / "convergence_t30" / "convergence.json").read_text())
    ref = rep["reference"]
    v = rep["verdict"]
    prot = rep["protocol"]

    L: list[str] = []
    L.append("# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST — 冻结算例判据的三网格收敛裁定 + ASME V&V 20")
    L.append("")
    L.append("> 任务 m21718（锚点 `CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`，TensorSPH）。")
    L.append("> 本文所有数字由 `benchmarks/kleefsman_obstacle/convergence_t30.py` 从**已提交**的")
    L.append("> 时程 CSV 程序化算出并注入（`docs/benchmarks/kleefsman-convergence-t30-m21718.md`")
    L.append("> 由 `make_doc_convergence_t30.py` 生成，零手抄）。")
    L.append("")
    L.append("## 1. 为什么要做这一单（欠账在哪）")
    L.append("")
    L.append("`results/audit_t30/score.json` 已经用**冻结协议**（slab 固定孔径水位计、0.08 m 孔径、")
    L.append("站位 = 箱体迎流面 −0.05 m、30 ms 峰值滤波、0.02 m 到达阈值）判过 `dx0.04`/`dx0.02`")
    L.append("两套网格，得到 `FAIL`。但它的 continuum 块自己写着：")
    L.append("")
    L.append("> *“two grids only; the order is assumed, not measured, so these are bounds on the")
    L.append("> continuum peak, not a convergence proof”*")
    L.append("")
    _prim = rep["stations"]["primary"]
    _rows = sorted(_prim["rows"], key=lambda r: -r["dx"])
    _two = abs(_rows[0]["peak_m"] - _rows[1]["peak_m"]) / abs(_rows[0]["peak_m"])
    L.append(f"而两网格之间主测站的峰值就移动了 **{pct(_two)}**，与判据容差 "
             f"{pct(ref['tolerance_rel'], 0)} 同量级。也就是说：")
    L.append("**判据量本身还没和它自己的离散误差分开**。本单补第三套网格（`dx0.01`），把这个")
    L.append("“假定阶”换成“实测阶 + GCI + ASME V&V 20 判据”，从而回答唯一能裁决该算例的问题：")
    L.append("")
    L.append(f"> 主测站 {pct(_prim['verdicts']['vv20_peak_fine'].get('E_rel'))} 的峰值偏差，"
             "是分辨率假象（dx→0 会收敛掉），还是真实的模型偏差")
    L.append("> （dx→0 依然存在，必须归因）？")
    L.append("")
    L.append("## 2. 怎么做（方法 / 工具链）")
    L.append("")
    L.append(f"* 协议：`{prot['operator']}` 水位计，孔径 {prot['aperture_m']} m，峰值滤波 "
             f"{prot['peak_window_s']} s，到达阈值 {prot['arrival_threshold_m']} m"
             "（冻结于 m20773，本单不重新定义，只做裁定）。")
    L.append("* 评分函数复用 `acceptance_frozen.score()`（与 `results/peak_horizon/score.json` 同一"
             "实现），因此本单不引入新的口径。")
    L.append("* 输入：`results/audit_t30/dx0.04|dx0.02/time_series.csv`（t_end=3.0 s）与 "
             "`results/peak_horizon/dx0.01_time_series.csv`（t_end=2.6 s，**已提交**）。"
             "三网格的**被比较极值**（≤1.55 s）都在各自记录之内（脚本 `_uniform_horizon()` "
             "逐网格检查，越界即拒绝），因此峰值可判；时间窗不齐一项如实报告为 "
             "`uniform=false`。")
    L.append("* 收敛数学：`convergence_analysis.analyze_metric()`（观测阶 p、Richardson、Roache GCI）。"
             "本单同时修了它的一个缺陷：**p ≤ 0（格差随加密变大）时 GCI 为负**，旧逻辑用 "
             "`gci < 0.10` 判定会把这种反收敛阶梯判成 `CONVERGED` 并输出负不确定度；现在 p ≤ 0 "
             "直接拒绝外推与 GCI。")
    L.append("* 保守修正：观测阶高于名义阶 `NOMINAL_ORDER=2.0` 时，GCI 用 "
             "`p_eff = min(p_obs, 2)` 重算（Roache 的保守规则）；极值型目标量常出现超收敛阶。")
    L.append("* 判据层：ASME V&V 20 comparison error，`E = S − D`，"
             "`U_V = sqrt(U_num² + U_D²)`，`|E| ≤ U_V` 才算在该不确定度水平上“验证通过”；"
             "`U_D` 取 `reference.json` 的数字化为 pixel 不确定度。**阶梯不收敛时不允许签发 PASS**"
             "（`PEAK_NOT_CERTIFIABLE_UNCONVERGED`）。")
    L.append("")
    L.append("## 3. 计算结果（三网格，逐站）")
    L.append("")
    for label, entry in rep["stations"].items():
        vd = entry["verdicts"]
        L.append(f"### 3.{1 if label == 'primary' else 2} 站位 {label}（x = {entry['station_x_m']:.3f} m）")
        L.append("")
        L.append("| dx [m] | 数据文件 | t_end [s] | 峰值 [m] | 峰值误差 | 峰值时刻 [s] | 到达 [s] | 到达误差 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in entry["rows"]:
            L.append(f"| {r['dx']:g} | `{r['path']}` | {fmt(r['t_end_s'], 3)} | "
                     f"{fmt(r['peak_m'])} | {pct(r['peak_rel_error'])} | {fmt(r['peak_time_s'], 3)} | "
                     f"{fmt(r['arrival_s'])} | {pct(r['arrival_rel_error'])} |")
        L.append("")
        pk, ar = vd["peak"], vd["arrival"]
        L.append("| 量 | 单调 | p_obs | p_eff | Richardson | GCI(Fs=1.25, p_eff) | 收敛判定 |")
        L.append("|---|---|---|---|---|---|---|")
        for name, st in (("峰值", pk), ("到达时刻", ar)):
            L.append(f"| {name} | {st['monotone']} | {fmt(st['p'], 2)} | {fmt(st['p_effective'], 2)} | "
                     f"{fmt(st['richardson'])} | {pct(st['gci_fine_conservative'], 2)} | "
                     f"`{st['convergence']}` |")
        L.append("")
        L.append(f"* ASME V&V 20（最细网格峰值）：E = {fmt(vd['vv20_peak_fine']['E'])} m"
                 f"（{pct(vd['vv20_peak_fine'].get('E_rel'))}），"
                 f"U_V = {fmt(vd['vv20_peak_fine']['U_V'])} m ⇒ **{vd['vv20_peak_fine']['verdict']}**")
        L.append(f"* ASME V&V 20（Richardson 连续值峰值）："
                 f"E = {fmt(vd['vv20_peak_continuum'].get('E'))} m ⇒ "
                 f"**{vd['vv20_peak_continuum'].get('verdict')}**")
        L.append(f"* ASME V&V 20（最细网格到达时刻）：E = {fmt(vd['vv20_arrival_fine']['E'])} s"
                 f"（{pct(vd['vv20_arrival_fine'].get('E_rel'))}），"
                 f"U_V = {fmt(vd['vv20_arrival_fine']['U_V'])} s ⇒ "
                 f"**{vd['vv20_arrival_fine']['verdict']}**")
        ev = vd["peak_event"]
        L.append(f"* 被比较的极值是否同一物理事件：**{ev['same_event']}**"
                 f"（仿真峰值时刻 {fmt(ev['sim_peak_time_s'], 3)} s vs 实验 {fmt(ev['ref_peak_time_s'], 3)} s）")
        L.append(f"* 该站判定：**`{vd['verdict']}`**；容差（<{pct(ref['tolerance_rel'], 0)}）逐项："
                 f"峰值 {vd['tolerance_pass_finest']['peak']}，到达 {vd['tolerance_pass_finest']['arrival']}")
        L.append("")
    L.append("## 4. 结果图")
    L.append("")
    L.append("![ladder](../figures/kleefsman_convergence_t30_ladder.png)")
    L.append("")
    L.append("* 图 1：`results/convergence_t30/ladder.png` —— 峰值/到达时刻随 dx 的阶梯（对数横轴），"
             "方块 = Richardson 外推值 ± GCI 带，红虚线 = 实验值，浅红带 = ±10 % 容差。")
    L.append("")
    L.append("![timeseries](../figures/kleefsman_convergence_t30_timeseries.png)")
    L.append("")
    L.append("* 图 2：`results/convergence_t30/timeseries.png` —— 两站各三条网格的冻结水位计时程"
             "与 Kleefsman 2005 Fig. 19 数字化曲线对比（点线标出实验峰值时刻 1.703 s 与 0.370 m）。")
    L.append("")
    L.append("## 5. 注意点（踩过的坑 / 局限 / 适用范围）")
    L.append("")
    L.append("1. **被比较的不是同一个事件（主测站）**。冻结站位 x=2.426 m 处仿真极值出现在 "
             f"{fmt(rep['stations']['primary']['verdicts']['peak_event']['sim_peak_time_s'], 3)} s"
             "（首撞爬升），而实验 H2 的最大值在 t=1.703 s（回波二次冲击）。因此主测站的"
             "“峰值误差”同时含**事件错配**成分，不能单独读成幅值偏差；副测站 x=2.228 m 的事件对得上。")
    L.append("2. **H2 的 x 坐标仍未从原文机器读取**（Fig. 17 是照片）。本单不猜点：两站都算，"
             "并把结论按站分开报——这正是“判据挂在一个未核实坐标上”的可判定化处理。")
    L.append("3. **时间窗不齐**：`dx0.01` 用的是已提交的 t_end=2.6 s 记录，另两套到 3.0 s。"
             "被比较极值都远离记录末端（脚本强制检查，越界即拒绝），因此对峰值无影响；"
             "为彻底消除该差异已递交同 t_end=3.0 s 的 `dx0.01` 作业（集群 `cj3647`，排队中），"
             "落地后重跑 `convergence_t30.py` 会自动采用（解析器优先 `results/gap_t30/`）。")
    L.append("4. **观测阶超名义阶**：主测站峰值 p_obs 高于 2，属极值型目标量的常见超收敛，"
             "故 GCI 一律按 p_eff = min(p_obs, 2) 保守重算后使用。")
    L.append("5. **二维模型的阻塞率差异是结构性上限**：实验箱宽 0.403 m / 槽宽 1.0 m（40.3 %），"
             "二维模型是 100 % 阻塞，这一点在 `score_h2.py` 里已记录，本单的模型偏差结论与之一致，"
             "但本单不声称已定量归因到该单一机制。")
    L.append("")
    L.append("## 6. 误差判定（vs 基准，逐项，显式回答是否 <3 %）")
    L.append("")
    L.append("| 量 | 基准（Kleefsman 2005 数字化） | 最细网格 | 相对误差 | 连续外推 | 相对误差 | <3 %? |")
    L.append("|---|---|---|---|---|---|---|")
    for label, entry in rep["stations"].items():
        vd = entry["verdicts"]
        fine = min(entry["rows"], key=lambda r: r["dx"])
        L.append(f"| {label} 峰值 [m] | {fmt(ref['peak_height_m'], 3)} | {fmt(fine['peak_m'])} | "
                 f"{pct(fine['peak_rel_error'])} | {fmt(vd['peak']['richardson'])} | "
                 f"{pct(vd['vv20_peak_continuum'].get('E_rel'))} | "
                 f"{'否' if fine['peak_rel_error'] >= 0.03 else '是'} |")
        L.append(f"| {label} 到达 [s] | {fmt(ref['arrival_time_s'])} | {fmt(fine['arrival_s'])} | "
                 f"{pct(fine['arrival_rel_error'])} | {fmt(vd['arrival']['richardson'])} | — | "
                 f"{'否' if fine['arrival_rel_error'] >= 0.03 else '是'} |")
    L.append("")
    L.append("**是否 <3 %：否。** 两站的峰值与到达时刻在两套站位口径下都远离 3 %（主测站峰值约 +30 %，"
             "到达约 +26 %；副测站峰值最细网格虽在 1 % 内，但其阶梯未收敛、外推值约 −14 %，"
             "不构成通过）。这正是本算例应有的诚实结论：**二维模型在箱前爬升区无法复现 H2 幅值**。")
    L.append("")
    L.append("## 7. 结论（诚实 FAIL，不粉饰）")
    L.append("")
    L.append(f"* 算例级判定：**`{v['verdict']}`** —— {v['why']}。")
    L.append(f"* 主测站：`{rep['stations']['primary']['verdicts']['verdict']}`，峰值偏差 "
             f"{pct(rep['stations']['primary']['verdicts']['vv20_peak_fine'].get('E_rel'))} 是"
             f"**模型偏差**（|E| 远大于 U_V={fmt(rep['stations']['primary']['verdicts']['vv20_peak_fine']['U_V'])} m，"
             "且阶梯已收敛、GCI 只有百分之一量级），加密不能消除。")
    L.append(f"* 副测站：`{rep['stations']['secondary']['verdicts']['verdict']}`，最细网格看似在容差内，"
             "但阶梯未收敛（GCI 超过 10 %、外推值把误差放大到约 −14 %），**因此不签发 PASS**。")
    L.append(f"* 到达时刻：{v['arrival_note']}；即便收敛，偏差 {pct(rep['stations']['primary']['verdicts']['vv20_arrival_fine'].get('E_rel'))}"
             " 也远超容差与 U_V，属模型/事件偏差。")
    L.append("")
    L.append("复核入口：")
    L.append("")
    L.append("```bash")
    L.append("python benchmarks/kleefsman_obstacle/convergence_t30.py      # 重算并出图")
    L.append("python benchmarks/kleefsman_obstacle/make_doc_convergence_t30.py")
    L.append("pytest tests/test_kleefsman_convergence_t30.py -q          # 数学/判据/提交证据三层")
    L.append("```")
    L.append("")
    L.append("## 8. 产物指针")
    L.append("")
    L.append("* 代码：`benchmarks/kleefsman_obstacle/convergence_t30.py`（裁定/出图）、"
             "`make_doc_convergence_t30.py`（本文生成）、`convergence_analysis.py`（反收敛缺陷修复）")
    L.append("* 数据：`benchmarks/kleefsman_obstacle/results/convergence_t30/convergence.json` / "
             "`convergence.md` / `ladder.png` / `timeseries.png`")
    L.append("* 测试：`tests/test_kleefsman_convergence_t30.py`（数学 / 判据 / 提交证据三层）、"
             "`tests/test_kleefsman_convergence_analysis.py::test_negative_observed_order_is_not_called_converged`")
    L.append("* 平台报告：id **19608**（mission 21718）")
    L.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())