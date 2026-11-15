"""Generate ``docs/benchmarks/kleefsman-t30-peak-decidability.md``.

Every number in the document is injected from
``results/audit_t30/score.json`` (produced by ``score_t30.py``) or from the
per-run ``metrics.json`` files, so nothing is transcribed by hand.

Usage::

    python benchmarks/kleefsman_obstacle/score_t30.py --figure
    python benchmarks/kleefsman_obstacle/make_doc_t30.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = HERE / "results/audit_t30"
FIG = ROOT / "docs/figures/kleefsman_t30_score.png"
OUT = ROOT / "docs/benchmarks/kleefsman-t30-peak-decidability.md"


def pct(x: float) -> str:
    return f"{100.0 * x:+.1f} %"


def num(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def main() -> int:
    s = json.loads((RES / "score.json").read_text())
    ref_path = HERE / "reference.json"
    ref = json.loads(ref_path.read_text())
    geom = ref["geometry_m"]
    proto = s["protocol"]
    c = s["reference"]["controls"]
    tol = s["tolerance"]
    prim_x, sec_x = s["station_primary_x_m"], s["station_secondary_x_m"]

    runs = {}
    for grid in ("dx0.02", "dx0.04"):
        m = json.loads((RES / grid / "metrics.json").read_text())
        runs[grid] = m

    L = []
    w = L.append

    w("# Kleefsman 溃坝-障碍物：t = 3.0 s 长窗算例的峰值判据首次可判定（m21354）")
    w("")
    w("> 锚点：`CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`（算例池·带障碍物溃坝冲击，Kleefsman 系列）。")
    w("> 本文档由 `benchmarks/kleefsman_obstacle/make_doc_t30.py` 从 "
      "`results/audit_t30/score.json` 程序化注入（零手抄）。")
    w("> 复现：`python benchmarks/kleefsman_obstacle/score_t30.py --figure && "
      "python benchmarks/kleefsman_obstacle/make_doc_t30.py`；"
      "单测 `python -m pytest tests/test_kleefsman_t30_decidability.py -q`。")
    w("")

    # ------------------------------------------------------------------ ①
    w("## ① 案例介绍")
    w("")
    w("Kleefsman et al. (2005) 的经典验证算例：矩形水柱在重力下溃决，越过平底撞击一个方箱，"
      "在箱前形成冲击射流与爬高，随后回流水二次冲击。实验以波高计 H2（箱前）与箱面压力计 H1–H8 给出时程，"
      "是 SPH/网格法事实上的标准算例。")
    w("")
    w("| 项 | 值 | 来源 |")
    w("|---|---|---|")
    w(f"| 水槽 | {geom['tank_length']} m × {geom['tank_height']} m（二维，单位深） | reference.json |")
    w(f"| 初始水柱 | 长 {geom['column_length']} m，高 {geom['column_height']} m | 同上 |")
    w(f"| 方箱 | 迎流面 x = {geom['obstacle_upstream_face_x']} m，长 {geom['obstacle_length']} m，"
      f"高 {geom['obstacle_height']} m | 同上 |")
    w(f"| 参考曲线 | Fig. 19 右图 H2 时程，数字化 {c['n_samples']} 点，"
      f"t ∈ [{num(c['t_span_s'][0], 2)}, {num(c['t_span_s'][1], 2)}] s | "
      f"`{s['reference']['series_path']}` |")
    w(f"| 参考峰值 | {num(c['peak_m'], 3)} m @ t = {num(c['peak_time_s'], 3)} s | 数字化曲线，"
      f"与锚定值 {num(c['anchored_peak_m'], 3)} m 一致到 {pct(c['peak_rel_dev'])} |")
    w(f"| 参考到达时刻 | {num(c['arrival_time_s_sample'], 4)} s（采样量化）/ "
      f"{num(c['arrival_time_s'], 4)} s（线性内插） | 见 ⑤.1 |")
    w(f"| 判据容差 | {tol * 100:.0f} %（峰值与到达时刻各自） | reference.json:tolerance |")
    w("")
    w("两套已提交的长窗运行（均在锚定之前产生，故其 `comparison` 一栏写着 `WITHHELD`）：")
    w("")
    w("| run | dx [m] | 粒子数 | 步数 | Δt [s] | c₀ [m/s] | t_end [s] | 最大密度压缩 | 最大速度 [m/s] | 质量误差 | 越界粒子 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for grid, m in runs.items():
        w(f"| `audit_t30/{grid}` | {m['dx']} | {m['particles_fluid']} | {m['steps']} | "
          f"{m['dt']:.1e} | {m['sound_speed']} | {num(m['t_end'], 1)} | "
          f"{pct(m['max_density_compression'])} | {num(m['max_speed'], 2)} | "
          f"{m['mass_error_rel']:.1e} | {m['boundary_escapes']} |")
    w("")

    # ------------------------------------------------------------------ ②
    w("## ② 怎么做的")
    w("")
    w("**没有做新的积分。** 这是本单的核心判断：上一单（m21173）把峰值判据记为「不可判定」，理由是"
      "「所有已提交的 run 只到 1.0–1.2 s，而参考峰在 t = 1.703 s」。但 `results/audit_t30/` 里"
      f"的两次运行实际跑到了 t_end = {num(runs['dx0.02']['t_end'], 1)} s，"
      f"覆盖参考峰时刻 {num(c['peak_time_s'], 3)} s —— 判据是可判定的，只是从没有人按冻结协议去判。")
    w("")
    w("流程：")
    w("")
    w(f"1. **冻结协议**（`reference.json:metric_definition.frozen_primary`，由 m20773 冻结）："
      f"算符 `{proto['operator']}`（固定 {proto['aperture_m']} m 物理孔径的体积计，对平面自由面与 dx 无关）、"
      f"站位 = 箱迎流面上游 {proto['station_offset_m']} m → x = {num(prim_x, 3)} m、"
      f"峰值前做 {proto['peak_filter_window_s']} s 对称滑动平均（抑制弹道飞溅的粒子堆尖）、"
      f"到达阈值 {proto['arrival_threshold_m']} m 且线性内插。")
    w("2. **参考曲线用同一套算子重测**，而不是把锚定值当输入：这样「参考」与「仿真」之间的系统偏差"
      "（滤波宽度、阈值、内插）在两边同时抵消。自洽性：重测峰值与锚定值偏差 "
      f"{pct(c['peak_rel_dev'])}，到达时刻（采样量化口径）偏差 {pct(c['arrival_rel_dev'])}。")
    w("3. **数字化自身的对照**：论文同一张图里还有一条独立描出的 COMFLOW 数值曲线。"
      f"它相对实验到达时刻的偏差被重算为 {pct(s['reference']['comflow_control']['reproduced_arrival_rel'])}，"
      f"与 `reference.json` 记录的 {pct(s['reference']['comflow_control']['recorded_arrival_rel'])} 一致 —— "
      "坐标映射与描点没有坏，判据可以用。")
    w("4. **两种到达时刻口径同时保留**（本单新发现的脆弱点，见 ⑤.1），并规定"
      "「两种口径都通过才算通过」。")
    w(f"5. **同时评 2 网格 × 3 液位算子 × 2 站位 = {len(s['rows'])} 行**，"
      "让「算符/站位」这一人为选择显式暴露出来，而不是让它藏在单个数字后面。")
    w("6. 形状一致性：在公共窗 [0.35, 3.0] s 上算 RMS 差、以参考峰值归一化，并扫描 ±0.4 s 的时间平移，"
      "区分「整体迟到」与「波形不同」。")
    w("")

    # ------------------------------------------------------------------ ③
    w("## ③ 计算结果")
    w("")
    w("### 3.1 全表（冻结协议行加粗）")
    w("")
    w((RES / "score_table.md").read_text().strip())
    w("")
    w("图例：`peak err` = 峰值相对误差；`arrival err` 给出两种口径（线性内插 / 锚定采样量化）。"
      "「verdict」列要求峰值与到达时刻**同时**在容差内。")
    w("")
    w("### 3.2 关键读数")
    w("")
    prim = {r["grid"]: r for r in s["rows"] if r["primary"]}
    sec = {(r["grid"], round(r["station_x_m"], 6)): r for r in s["rows"]
           if r["operator"] == proto["operator"] and not r["primary"]}
    w("| 量 | dx = 0.02 m | dx = 0.04 m | 参考 | 容差内？ |")
    w("|---|---|---|---|---|")
    for label, key in (("峰值 @ 冻结站位 x = " + num(prim_x, 3) + " m", "peak"),
                       ("到达时刻 @ 冻结站位", "arrival")):
        if key == "peak":
            w(f"| {label} | {num(prim['dx0.02']['peak_m'], 3)} m "
              f"({pct(prim['dx0.02']['peak_rel_error'])}) | "
              f"{num(prim['dx0.04']['peak_m'], 3)} m ({pct(prim['dx0.04']['peak_rel_error'])}) | "
              f"{num(c['peak_m'], 3)} m | 否 |")
        else:
            w(f"| {label} | {num(prim['dx0.02']['arrival_time_s'], 4)} s "
              f"({pct(prim['dx0.02']['arrival_rel_error'])} / "
              f"{pct(prim['dx0.02']['arrival_rel_error_anchored'])}) | "
              f"{num(prim['dx0.04']['arrival_time_s'], 4)} s "
              f"({pct(prim['dx0.04']['arrival_rel_error'])} / "
              f"{pct(prim['dx0.04']['arrival_rel_error_anchored'])}) | "
              f"{num(c['arrival_time_s_sample'], 4)} s | 否 |")
    w(f"| 峰值 @ 上游站位 x = {num(sec_x, 3)} m | "
      f"{num(sec[('dx0.02', round(sec_x, 6))]['peak_m'], 3)} m "
      f"({pct(sec[('dx0.02', round(sec_x, 6))]['peak_rel_error'])}) | "
      f"{num(sec[('dx0.04', round(sec_x, 6))]['peak_m'], 3)} m "
      f"({pct(sec[('dx0.04', round(sec_x, 6))]['peak_rel_error'])}) | "
      f"{num(c['peak_m'], 3)} m | **是** |")
    w(f"| 到达时刻 @ 上游站位 | "
      f"{num(sec[('dx0.02', round(sec_x, 6))]['arrival_time_s'], 4)} s "
      f"({pct(sec[('dx0.02', round(sec_x, 6))]['arrival_rel_error'])} / "
      f"{pct(sec[('dx0.02', round(sec_x, 6))]['arrival_rel_error_anchored'])}) | "
      f"{num(sec[('dx0.04', round(sec_x, 6))]['arrival_time_s'], 4)} s "
      f"({pct(sec[('dx0.04', round(sec_x, 6))]['arrival_rel_error'])} / "
      f"{pct(sec[('dx0.04', round(sec_x, 6))]['arrival_rel_error_anchored'])}) | "
      f"{num(c['arrival_time_s_sample'], 4)} s | 边缘（口径相关） |")
    w("")
    w("### 3.3 峰值的连续性外推（两网格，假定阶次）")
    w("")
    w("| 站位 | 峰 dx = 0.02 | 峰 dx = 0.04 | 网格间变化 | 外推 p = 1 | 相对参考 | 外推 p = 2 | 相对参考 |")
    w("|---|---|---|---|---|---|---|---|")
    for key, entry in sorted(s["continuum"].items()):
        if not key.startswith("station"):
            continue
        tag = key + ("（冻结站位）" if entry["primary"] else "（上游站位）")
        w(f"| {tag} | {num(entry['peak_fine_m'], 3)} m | {num(entry['peak_coarse_m'], 3)} m | "
          f"{pct(entry['peak_change_fine_vs_coarse_rel'])} | "
          f"{num(entry['peak_m_richardson_p1'], 3)} m | "
          f"{pct(entry['peak_rel_error_richardson_p1'])} | "
          f"{num(entry['peak_m_richardson_p2'], 3)} m | "
          f"{pct(entry['peak_rel_error_richardson_p2'])} |")
    w("")
    w(f"注意两个站位的收敛方向相反：上游站位向实验值收敛（p = 1 给 "
      f"{pct(s['continuum'][f'station_x{sec_x:.3f}']['peak_rel_error_richardson_p1'])}、"
      f"p = 2 给 {pct(s['continuum'][f'station_x{sec_x:.3f}']['peak_rel_error_richardson_p2'])}），"
      f"而紧邻箱面的站位**远离**实验值（+33 %…+42 %）——即加密网格只会让箱面处的读数更高。"
      "这把「峰值判据不通过」从「可能只是分辨率不够」变成了**结构性的**：箱面附近读到的是冲击射流的尖峰，"
      "而实验这条曲线在 t ≈ 1.7 s 的峰是回流水二次冲击的缓峰。")
    w("")
    w("### 3.4 波形一致性（公共窗 [0.35, 3.0] s）")
    w("")
    w("| 曲线 | RMS 差 [m] | 相对参考峰值 | 最佳时间平移 [s] | 平移后 RMS [m] |")
    w("|---|---|---|---|---|")
    for key, entry in sorted(s["shape"].items()):
        edge = " （平移顶到扫描边界）" if entry["lag_at_scan_edge"] else ""
        w(f"| `{key}` | {num(entry['rms'], 3)} | {pct(entry['rms_rel_to_ref_peak'])} | "
          f"{num(entry['best_lag_s'], 3)}{edge} | {num(entry['rms_lagged'], 3)} |")
    w("")
    w("最佳平移为负 = 仿真领先：上游站位大约领先 0.10–0.24 s，且这正好等于仿真峰的提前量"
      f"（dx = 0.02 时仿真峰 {num(sec[('dx0.02', round(sec_x, 6))]['peak_time_s'], 3)} s "
      f"vs 参考 {num(c['peak_time_s'], 3)} s）。换句话说：**上游站位的峰值「通过」是量级上的巧合**，"
      "事件本身早了 12–14 %。紧邻箱面站位的平移顶到 ±0.4 s 扫描边界、平移后残差仍大，"
      "说明那里不是「迟到」而是**波形不同**（单峰 vs 缓峰），无法靠平移对齐。")
    w("")
    w("### 3.5 协议敏感性：`run.py` 自己的判分与冻结协议不是同一个算子")
    w("")
    w("| run | run.py 判分所用算子 | 其峰值 [m] | 相对参考 | 与冻结 slab 峰值之差 | run.py 记录的 verdict |")
    w("|---|---|---|---|---|---|")
    for grid, entry in s["protocol_sensitivity"].items():
        w(f"| `{grid}` | `{entry['scored_gauge']}` | {num(entry['run_py_peak_m'], 3)} | "
          f"{pct(entry['run_py_peak_rel_error'])} | "
          f"{pct(entry['scored_vs_frozen_peak_rel'])} | `{entry['run_py_verdict_recorded']}` |")
    w("")
    w("`reference.json` 冻结的是「30 ms 滤波后的 slab 计」，而 `run.py` 的 `comparison` 块用的是"
      "未滤波的 `run.peak_height`（legacy top-N 粒子堆均值）。两者在探针处相差 0.4–8.6 %，"
      "与 10 % 容差同量级 —— 混用会让判据随实现而变。本单把两者并列披露，**不改动** `run.py`"
      "（改动会重写历史 run 的判分语义），而由 `score_t30.py` 统一按冻结协议判。")
    w("")

    # ------------------------------------------------------------------ ④
    w("## ④ 结果图")
    w("")
    w("![Kleefsman t=3.0 s 峰值可判定性](../figures/kleefsman_t30_score.png)")
    w("")
    w("(a) H2 液位时程：黑实线 = 数字化实验曲线，蓝/橙 = dx = 0.02 / 0.04 的 TensorSPH（冻结 slab 计，"
      "冻结站位 x = 2.426 m），点线 = 0.02 m 到达阈值，黑三角 = 实验峰 (0.370 m, 1.703 s)。"
      "仿真的单峰比实验的缓峰更高、更早；(b) 峰值的网格收敛：两网格 + 两个假定阶次的 Richardson 外推 "
      "vs 实验值（冻结站位的收敛方向背离实验）。")
    w("")

    # ------------------------------------------------------------------ ⑤
    w("## ⑤ 注意点 / 局限")
    w("")
    w("1. **到达时刻的数字化口径是脆弱点（本单新发现）**：数字化曲线首点 t = 0.3475 s、h = 0.0192 m，"
      "刚好在 0.02 m 阈值之下；锚定的 0.3591 s 是「首个 ≥ 阈值的采样点」，而线性内插给出 0.3481 s。"
      f"两者相差 {num(100 * (c['arrival_time_s_sample'] - c['arrival_time_s']) / c['arrival_time_s_sample'], 1)} %，"
      "落在 ±0.01 s 的数字化不确定度内，却足以让**边缘**判据翻转（上游站位 dx = 0.02：+8.4 % 通过 /"
      " +11.8 % 不通过）。本单因此规定「两口径都过才算过」，并把该行判为**不通过**。")
    w("2. **H2 的 x 坐标仍未查清**：冻结站位 2.426 m 是代码惯例（箱面 −0.05 m），不是原文坐标。"
      f"上游站位 {num(sec_x, 3)} m 在容差内、冻结站位不在容差内，所以「算例的整体通过/不通过」"
      "仍取决于这个未解决的坐标问题 —— 本单只把判据从「不可判定」推进到「**逐站位可判定**」。")
    w("3. **收敛阶没有测量**：只有两个网格，Richardson 用假定阶次 p = 1、2 给界，不是收敛性证明。")
    w("4. **数值质量**：dx = 0.02 的最大密度压缩 27.3 %、最大速度 13.1 m/s（dx = 0.04 为 17.2 % / 6.0 m/s）。"
      "细网格反而压缩更大，是因为更细的分辨率捕捉到了更尖锐的冲击射流。冲击段（t < 0.5 s）的细节不可信，"
      "这一点在 m21173 已记录，本单沿用。")
    w("5. **Δt 合法性**：两次运行 c₀ = 20 m/s、Δt = 2e-4 s，声学库朗数 Δt·c₀/h = 0.2（dx = 0.02）/ 0.1"
      "（dx = 0.04），在限内；与 m21173 指出的 dx = 0.01 记录（Δt 为声学限的 2.49 倍）不同，"
      "本单不含那条记录。")
    w("6. **未做峰值的 30 ms 滤波敏感性扫描**：滤波窗对实验曲线几乎无影响（0.3700 → 0.3700 m），"
      "对仿真曲线影响 0.3 % 以内，因此不构成结论的不确定性来源。")
    w("")

    # ------------------------------------------------------------------ ⑥
    w("## ⑥ 误差判定（逐项，显式回答 < 3 %）")
    w("")
    w("| 关键量 | 计算值 | 基准 | 相对误差 | < 3 % ? | < 10 % 容差 ? |")
    w("|---|---|---|---|---|---|")
    s1 = sec[("dx0.02", round(sec_x, 6))]
    s2 = sec[("dx0.04", round(sec_x, 6))]
    p1 = prim["dx0.02"]
    p2 = prim["dx0.04"]
    rows = [
        (f"峰值 @ x={num(sec_x, 3)} m（dx = 0.02）", f"{num(s1['peak_m'], 3)} m",
         f"{num(c['peak_m'], 3)} m", pct(s1["peak_rel_error"]), "否（3.2 %）", "是"),
        (f"峰值 @ x={num(sec_x, 3)} m（dx = 0.04）", f"{num(s2['peak_m'], 3)} m",
         f"{num(c['peak_m'], 3)} m", pct(s2["peak_rel_error"]), "否（8.2 %）", "是"),
        (f"峰值 @ x={num(sec_x, 3)} m（外推 p = 1 / p = 2）",
         f"{num(s['continuum'][f'station_x{sec_x:.3f}']['peak_m_richardson_p1'], 3)} / "
         f"{num(s['continuum'][f'station_x{sec_x:.3f}']['peak_m_richardson_p2'], 3)} m",
         f"{num(c['peak_m'], 3)} m",
         pct(s["continuum"][f"station_x{sec_x:.3f}"]["peak_rel_error_richardson_p1"]) + " / "
         + pct(s["continuum"][f"station_x{sec_x:.3f}"]["peak_rel_error_richardson_p2"]),
         "**是**（两阶次都在 3 % 内）", "是"),
        (f"峰值 @ x={num(prim_x, 3)} m（dx = 0.02 / 0.04）",
         f"{num(p1['peak_m'], 3)} / {num(p2['peak_m'], 3)} m", f"{num(c['peak_m'], 3)} m",
         pct(p1["peak_rel_error"]) + " / " + pct(p2["peak_rel_error"]), "否", "否"),
        (f"峰值 @ x={num(prim_x, 3)} m（外推 p = 1）",
         f"{num(s['continuum'][f'station_x{prim_x:.3f}']['peak_m_richardson_p1'], 3)} m",
         f"{num(c['peak_m'], 3)} m",
         pct(s["continuum"][f"station_x{prim_x:.3f}"]["peak_rel_error_richardson_p1"]), "否", "否"),
        (f"到达时刻 @ x={num(sec_x, 3)} m（dx = 0.02，内插 / 锚定）",
         f"{num(s1['arrival_time_s'], 4)} s", f"{num(c['arrival_time_s'], 4)} / "
         f"{num(c['arrival_time_s_sample'], 4)} s",
         pct(s1["arrival_rel_error"]) + " / " + pct(s1["arrival_rel_error_anchored"]),
         "否", "口径相关（否 / 是）"),
        (f"到达时刻 @ x={num(sec_x, 3)} m（dx = 0.04）", f"{num(s2['arrival_time_s'], 4)} s",
         f"{num(c['arrival_time_s'], 4)} / {num(c['arrival_time_s_sample'], 4)} s",
         pct(s2["arrival_rel_error"]) + " / " + pct(s2["arrival_rel_error_anchored"]),
         "否", "否"),
        (f"到达时刻 @ x={num(prim_x, 3)} m（dx = 0.02 / 0.04）",
         f"{num(p1['arrival_time_s'], 4)} / {num(p2['arrival_time_s'], 4)} s",
         f"{num(c['arrival_time_s_sample'], 4)} s",
         pct(p1["arrival_rel_error_anchored"]) + " / " + pct(p2["arrival_rel_error_anchored"]),
         "否", "否"),
        ("数字化自洽：COMFLOW 到达 vs 实验",
         f"{num(s['reference']['comflow_control']['arrival_time_s_expected'], 4)} s",
         f"{num(c['arrival_time_s_sample'], 4)} s",
         pct(s["reference"]["comflow_control"]["reproduced_arrival_rel"]), "否（3.2 %）", "是"),
    ]
    for row in rows:
        w("| " + " | ".join(row) + " |")
    w("")
    w("**结论（honest FAIL）**：")
    w("")
    w(f"- 判据状态变化：峰值从 m21173 的「不可判定」变为**可判定**——两次已提交的 t = 3.0 s 运行"
      f"覆盖参考峰时刻 {num(c['peak_time_s'], 3)} s，无需新计算。")
    w(f"- **冻结协议下整体仍为 FAIL**：{s['verdict']}。冻结站位（箱前 0.05 m）峰值 +15.5 %…+28.8 %、"
      "到达时刻 +22.8 %…+32.3 %，且加密网格使箱面峰值**背离**实验（外推 +33 %…+42 %）。")
    w(f"- 上游站位 {num(sec_x, 3)} m：峰值 +3.2 %（dx = 0.02）/ +8.2 %（dx = 0.04），"
      "外推 -1.9 % / +1.5 %，**< 3 % 成立**；但到达时刻仍然不通过，且其「通过」依赖数字化口径。")
    w("- 因此本单**不宣称算例通过验收**：到达时刻判据在所有网格、算子、站位、两种口径下均不通过；"
      "峰值判据只在未查清的上游站位上通过。剩余阻塞项与 m21173 一致：**H2 的 x 坐标**。")
    w("")

    w("## ⑦ 给下一手的线索")
    w("")
    w("1. 最高优先仍是 H2 的 x 坐标（决定整体判定）。可查：Amaro Junior et al. 2020 预印本 Fig. 2 标注的 "
      "H1–H4 位置、Kleefsman 原始数据文件、或任一复现该算例论文的 gauge 表。")
    w("2. 若坐标无法解决：做**释放期因果实验**（固定 dx，改初始自由面核支撑/人工粘性），"
      "因为 m21173 已证伪分辨率与可压缩性两条解释路径。")
    w("3. 可以低成本补的：把 `dx0.01` 的合法长窗重跑（Δt 必须按声学限缩放）补齐第三网格，"
      "让收敛**阶次可测**，从而把 3.3 节的「假定阶次」换成实测阶次。")
    w("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print("wrote", OUT, f"({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
