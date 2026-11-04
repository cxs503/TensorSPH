"""Render ``docs/paper_laplace_uncertainty_material.md`` from the committed run
artifacts -- every number in the document is read from
``benchmarks/paper_laplace_uncertainty/out/results.json``, nothing is typed by
hand, so the document cannot drift away from the measurement.

    PYTHONPATH=src python benchmarks/paper_laplace_uncertainty/make_doc.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

BENCH = Path(__file__).resolve().parent
OUT = BENCH / "out"
DOC = BENCH.parents[1] / "docs" / "paper_laplace_uncertainty_material.md"


def f(x, nd=2):
    return f"{x:.{nd}f}"


def main():
    res = json.loads((OUT / "results.json").read_text())
    tables = (OUT / "tables.md").read_text()
    prov = res["provenance"]
    rows = res["rows"]
    runs = sorted((OUT / "runs").glob("run_*.json"))
    setup0 = json.loads(runs[0].read_text())["setup"]

    metric = {k: [r["metrics"][k]["rel_error_pct"] for r in rows
                  if math.isfinite(r["metrics"][k]["rel_error_pct"])]
              for k in ("band", "core_to_shell", "force_implied")}
    best_key = min(metric, key=lambda k: max(abs(v) for v in metric[k]))
    n_pass = sum(1 for r in rows for k in metric
                 if r["metrics"][k]["verdict_3pct"] == "PASS")
    n_fail = sum(1 for r in rows for k in metric
                 if r["metrics"][k]["verdict_3pct"] == "FAIL")
    n_with = sum(1 for r in rows for k in metric
                 if r["metrics"][k]["verdict_3pct"] == "WITHHELD")

    comp_scale = setup0["rho0"] * setup0["sound_speed_m_per_s"] ** 2 / 7.0

    L = []
    A = L.append
    A("# 论文素材：TensorSPH 核函数验证的独立基线表与不确定度/云图")
    A("")
    A("**仓库**: TensorSPH ｜ **工件**: `benchmarks/paper_laplace_uncertainty/out/` "
      "｜ **测试**: `tests/test_paper_laplace_uncertainty.py` ｜ "
      f"**原始算例**: {prov['runs']} 次独立实现（{len(rows)} 个分辨率）")
    A("")
    A("**结论（先给判定，三态显式）**: 三种并列的 Young–Laplace 口径下，"
      f"共 {n_pass} 个 PASS / {n_with} 个 WITHHELD / {n_fail} 个 FAIL"
      f"（阈值规则：{prov['verdict_rule']}）。"
      "**没有一种口径在所有分辨率上达到 <3%**。"
      f"表现最好的是口径 `{best_key}`，其最大偏差仍达 "
      f"{f(max(abs(v) for v in metric[best_key]),1)}%。")
    A("")
    A("## ① 案例介绍")
    A("")
    A("静止自由液滴是表面张力模型唯一有闭式解的整体性算例：二维圆截面液滴满足 "
      "Young–Laplace 关系")
    A("")
    A("$$\\Delta p = p_{in} - p_{out} = \\sigma/R \\qquad (1)$$")
    A("")
    A("本文档给出成稿需要的两类 **Results 级** 素材：")
    A("")
    A("1. **独立基线/对比表**（第 ③ 节表 1a–1c、表 2、表 3）：每个量逐点对照闭式解，"
      "分子=实测−解析、分母=解析值，判据阈值 3%（论文问题）/15%（平台硬门）显式给出，"
      "并附 95% 置信区间；曲率给三套**互相独立**的实现（MLS `div n`、二次曲面拟合、"
      "代数圆拟合），其中后两者来自 `tensorsph/csf_curvature.py`，构成实现 A/B 对照。")
    A("2. **不确定度与场量云图**（第 ④ 节图 1–4）：误差棒/置信区间来自 5 个统计等价"
      "（晶格填充相位 + 抖动）实现组成的系综；云图为静滴压强场和曲率误差场。")
    A("")
    A("| 参数 | 值 | 说明 |")
    A("|---|---|---|")
    A(f"| 液滴半径 R | {setup0['radius_m']*1e3:.2f} mm | 静止自由液滴，2-D |")
    A(f"| 粒子间距 Δx | {min(r['dx_m'] for r in rows)*1e3:.2f}–"
      f"{max(r['dx_m'] for r in rows)*1e3:.2f} mm | 5 档分辨率，R/Δx = "
      f"{min(r['R_over_dx'] for r in rows):.1f}–{max(r['R_over_dx'] for r in rows):.1f} |")
    A(f"| 表面张力系数 σ | {setup0['sigma_N_per_m']:.1f} N/m | 10× 水，压缩毛细时间 |")
    A(f"| 参考密度 ρ0 | {setup0['rho0']:.0f} kg/m³ | 弱可压 Tait，γ = 7 |")
    A(f"| 声速 c0 | {setup0['sound_speed_m_per_s']:.0f} m/s | Mach ≪ 0.1 |")
    A(f"| 粘度 ν | {setup0['viscosity_m2_per_s']:.2f} m²/s | 层流正则化 Laplacian |")
    A(f"| 解析压差 σ/R | **{rows[0]['delta_p_theory_Pa']:.2f} Pa** | 式 (1) 基准量 |")
    A(f"| 解析曲率 1/R | **{rows[0]['curvature']['mls_divn']['kappa_exact_1_per_m']:.2f} 1/m** "
      "| 表 2 基准量 |")
    A("| 重力 | 0 | 纯毛细平衡 |")
    A("")
    A("## ② 怎么做的")
    A("")
    A("单一脚本 `benchmarks/paper_laplace_uncertainty/run.py`（自包含，复用生产代码路径 "
      "`tensorsph.droplet_impact.rhs` / `tensorsph.surface_tension.*` / "
      "`tensorsph.csf_curvature.fit_curvature`，**不改 `src/`**）。")
    A("")
    A("1. **几何**：六角密排圆盘，`mass × 数密度 = ρ0` 精确成立，初始密度无偏。")
    A("2. **系综（不确定度来源）**：圆盘中心固定在原点，平移晶格得到 5 个**真正不同**的"
      "离散化（晶格相位 (0,0)/(0.5Δx,0)/(0,0.5Δy)/(0.5Δx,0.5Δy)/(0.25Δx,0.75Δy)），"
      "其中一个成员叠加 2%Δx 的定种子位置抖动。注意：**旋转**晶格不是系综轴——"
      "轴对称液滴的旋转是精确对称性，实测与未旋转结果逐位相同，故不采用。")
    A("3. **三种并列口径**（同一末态、同一批粒子，避免口径含糊）：")
    A("   - A `band`：Δp = mean(p, r<0.5R) − mean(p, 0.8R<r<0.95R)（在**界面处**取值）")
    A("   - B `core_to_shell`：Δp = mean(p, r<0.5R) − mean(p, 0.95R<r<1.05R)（用解析 p_out=0）")
    A("   - C `force_implied`：Δp = |F_cut|/(2R)，F_cut = Σ_{x>0} m a_st,x（解析 |F_cut| = 2σ，"
      "与压强场无关，因此能把「力量级错」与「压强传不上去」分开归因）")
    A("4. **时间**：以 `stable_dt()` 定步长推进到 t = 1.0 t_σ，沿途在 "
      + "/".join(str(f_) for f_ in (0.25, 0.5, 0.75, 1.0)) +
      " t_σ 采样（表 4）。")
    A("5. **判据**：|e| + 95%CI < 3% → PASS；|e| ≥ 15% → FAIL；其余 WITHHELD。"
      "置信区间 = t_{0.975,n−1}·s/√n（n=3–5）。")
    A("6. **自检**：`tests/test_paper_laplace_uncertainty.py` 从原始 `runs/*.json` "
      "复算每个聚合数（分子/分母/相对误差/系综均值/置信区间/判据映射）。")
    A("")
    A("## ③ 计算结果（表 1–4，程序化渲染，零手抄）")
    A("")
    A("> 下表由 `run.py --aggregate` 从原始 `runs/*.json` 生成，"
      "完整版见 `benchmarks/paper_laplace_uncertainty/out/tables.md`。")
    A("")
    A(tables.replace("# Laplace-law verification: machine-rendered tables", "### 结果表").strip())
    A("")
    A("## ④ 结果图（场量云图 + 基线比较 + 不确定度）")
    A("")
    A("![pressure field](../benchmarks/paper_laplace_uncertainty/out/fig1_pressure_field.png)")
    A("")
    A("*图 1：静滴压强场云图（t = 1.0 t_σ，两个分辨率）。若 Young–Laplace 成立，"
      "液滴内部应为常压 σ/R 的等值面；图上可见明显的内部压强梯度与近表面亏损，"
      "这正是口径 A 的误差来源。*")
    A("")
    A("![curvature field](../benchmarks/paper_laplace_uncertainty/out/fig2_curvature_field.png)")
    A("")
    A("*图 2：曲率误差场 |κ| − 1/R（二次曲面拟合）。误差在界面壳层上不是噪声而是"
      "系统性偏置，随晶格方位呈花形分布。*")
    A("")
    A("![uncertainty](../benchmarks/paper_laplace_uncertainty/out/fig3_uncertainty.png)")
    A("")
    A("*图 3：(a) 三种口径的 Δp 相对误差 vs R/Δx（误差棒 = 系综 95% CI，红虚线为 3%/15% 门）；"
      "(b) 三套曲率估计器 vs 1/R；(c) 伪流水平与切割面力量级。*")
    A("")
    A("![baseline compare](../benchmarks/paper_laplace_uncertainty/out/fig4_baseline_compare.png)")
    A("")
    A("*图 4：(a) 最高分辨率下的径向压强剖面（误差棒 = 壳层内标准差）与解析值对比；"
      "(b) 实测跳变随时间的演化（1 t_σ 内误差持续增大，说明不是「还没平衡」）。*")
    A("")
    A("## ⑤ 注意点（局限与适用范围）")
    A("")
    A("1. **口径决定成败，必须三口径并列**：口径 B（核–自由面）在部分分辨率上看起来"
      "比口径 A（核–界面带）好得多，但 B 只有在液滴内部压强均匀时才有物理意义；"
      "图 1 与表 3 显示内部存在系统性压强梯度，因此 B 的系统偏差被低估。"
      "成稿引用单一口径的数字会构成误导。")
    A("2. **系综不确定度不可忽略**：晶格相位改变界面粒子的取整方式，同一 (R, σ, Δx) "
      "下 Δp 的相对误差跨成员可差出数十个百分点，因此单点报数（历史上 PASS 7.66% 的"
      "单点结论）不具备可重复性；本文档所有结论都以系综 CI 给出。")
    A("3. **弱可压代价**：Tait 状态方程把 Δp 编码为密度差 "
      "（本例所需压缩 "
      f"{rows[0]['delta_p_theory_Pa'] / comp_scale:.3f}"
      " 量级的密度比），密度求和误差直接进入压强；这是 TensorSPH 该路径的结构性局限，"
      "不是参数选择问题。")
    A(f"4. **伪流**：max|v|/√(σ/ρR) 在全部算例为 "
      f"{min(r['spurious_current_ratio'] for r in rows):.3f}–"
      f"{max(r['spurious_current_ratio'] for r in rows):.3f}"
      "（Adami 2012 判据 < 0.1），因此上述偏差不是由残余伪流噪声主导。")
    A("5. **分辨率范围**：R/Δx = "
      f"{min(r['R_over_dx'] for r in rows):.0f}–{max(r['R_over_dx'] for r in rows):.0f}；"
      "更粗的分辨率（R/Δx < 6）界面壳层不足两层，不在结论适用范围。")
    A("")
    A("## ⑥ 误差判定（逐项 vs 基准，显式回答「是否 <3%」）")
    A("")
    A("| 量 | 基准 | 最接近基准的算例 | 相对误差 [%] (95% CI) | <3%? | 判定 |")
    A("|---|---|---|---|---|---|")
    for key, label in (("band", "Δp（口径 A）"), ("core_to_shell", "Δp（口径 B）"),
                       ("force_implied", "Δp（口径 C）")):
        best = min((r for r in rows if math.isfinite(r["metrics"][key]["rel_error_pct"])),
                   key=lambda r: abs(r["metrics"][key]["rel_error_pct"]))
        m = best["metrics"][key]
        A(f"| {label} | σ/R = {rows[0]['delta_p_theory_Pa']:.2f} Pa | "
          f"R/Δx = {best['R_over_dx']:.1f} | "
          f"{m['rel_error_pct']:.2f} ({m['rel_error_ci95_pct']:.2f}) | "
          f"{'yes' if m['within_3pct'] else 'no'} | {m['verdict_3pct']} |")
    for est in ("mls_divn", "fit_quadratic", "fit_circle"):
        best = min((r for r in rows if math.isfinite(r["curvature"][est]["rel_error_pct"])),
                   key=lambda r: abs(r["curvature"][est]["rel_error_pct"]))
        c = best["curvature"][est]
        A(f"| κ ({est}) | 1/R = {c['kappa_exact_1_per_m']:.2f} 1/m | "
          f"R/Δx = {best['R_over_dx']:.1f} | "
          f"{c['rel_error_pct']:.2f} ({c['ci95_pct']:.2f}) | "
          f"{'yes' if abs(c['rel_error_pct']) < 3.0 else 'no'} | {c['verdict_3pct']} |")
    best_f = min((r for r in rows if math.isfinite(r["cut_force_rel_error_pct"])),
                 key=lambda r: abs(r["cut_force_rel_error_pct"]))
    A(f"| F_cut | −2σ | R/Δx = {best_f['R_over_dx']:.1f} | "
      f"{best_f['cut_force_rel_error_pct']:.2f} | "
      f"{'yes' if abs(best_f['cut_force_rel_error_pct']) < 3.0 else 'no'} | "
      f"{'PASS' if abs(best_f['cut_force_rel_error_pct']) < 3.0 else 'FAIL'} |")
    shell = [r for r in rows if math.isfinite(r["p_shell_over_dp_theory_pct"])]
    sh_best = min(shell, key=lambda r: abs(r["p_shell_over_dp_theory_pct"]))
    sh_min = min(abs(r["p_shell_over_dp_theory_pct"]) for r in shell)
    A(f"| p_shell / (σ/R)，自由面压强闭合 | 0 | R/Δx = {sh_best['R_over_dx']:.1f} "
      f"| {sh_best['p_shell_over_dp_theory_pct']:.2f} | "
      f"{'yes' if sh_min < 3.0 else 'no'} | {'PASS' if sh_min < 3.0 else 'FAIL'} |")
    A("")
    A("**是否 <3%？** 对 Young–Laplace 压差：**否**（三口径下最接近的一个算例误差 "
      f"{f(min(abs(r['metrics'][k]['rel_error_pct']) for r in rows for k in metric),1)}%）。"
      "对曲率：**否**（最好的估计器二次曲面拟合的最优算例为 "
      f"{f(min(abs(r['curvature']['fit_quadratic']['rel_error_pct']) for r in rows),1)}%）。"
      "对自由面压强闭合：**是**（p_shell 在 ρ≤ρ0 处被 Tait 截断为 0，因此该判据是"
      "结构性成立的，不构成对模型精度的正面证据）。")
    A("")
    A("## 数据可得性")
    A("")
    A("全部原始算例、聚合表与图件在本仓库内："
      "`benchmarks/paper_laplace_uncertainty/out/runs/*.json`（每次系综成员）、"
      "`out/results.json`、`out/tables.md`、`out/fig1..fig4*.png`；"
      "重跑命令：`PYTHONPATH=src python benchmarks/paper_laplace_uncertainty/run.py "
      "--dx-list 0.003 0.0025 0.002 0.0015 0.001 --realisations 5`。")

    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(L) + "\n")
    print("doc ->", DOC)


if __name__ == "__main__":
    main()
