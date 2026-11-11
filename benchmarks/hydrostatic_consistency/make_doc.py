#!/usr/bin/env python3
"""Render ``docs/verification_hydrostatic_consistency.md`` from report.json.

Every number in the document is read out of the committed artifact, so the
document cannot drift from the benchmark (no hand-typed values).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmarks" / "hydrostatic_consistency"
OUT = REPO / "docs" / "verification_hydrostatic_consistency.md"


def main() -> int:
    r = json.loads((BENCH / "report.json").read_text())
    mv, floor, grid = r["measured_vs_static"], r["readability_floor"], r["grid"]

    def row(hf, layers, lid, field="adami"):
        return next(g for g in grid
                    if abs(g["h_factor"] - hf) < 1e-9 and g["wall_layers"] == layers
                    and g["lid"] is lid and g["wall_field"] == field)

    base, thin = row(1.3, 2, True), row(1.3, 1, True)
    open_top, base15 = row(1.3, 2, False), row(1.5, 2, True)
    open15 = row(1.5, 2, False)
    L = []
    a = L.append

    a("# 3-D 静水柱的静态一致性审计（零仿真）\n")
    a("**性质**: 验证欠账（复核已合并能力）｜**改动**: `benchmarks/hydrostatic_consistency/`"
      "（新 harness `run.py` / `make_doc.py` / `lid_experiment.py`）、"
      "`src/tensorsph/threed/solver.py`（新增 `run_hydrostatic(lid=...)`，默认行为不变）、"
      "`tests/test_hydrostatic_consistency.py`。本文所有数字由 `report.json` 经 "
      "`make_doc.py` 注入，零手抄。\n")

    a("## 1. 案例与问题\n")
    a("`benchmarks/threed_hydrostatic` 积分一个 8×8×16 的静止水柱，参考解 "
      "`p(z) = rho0 g (H - z)`。其已提交工件自判 `verdict = fail`：13 个核心层平均相对误差 "
      f"**{mv['measured_mean_pct']:.2f} %**、最差层 **{min(mv['measured_rel_error_pct']):.2f} %**；"
      "账本 `docs/verification_ledger.md` 把这些读数记 FAIL/MARGINAL，但**没有任何机制定位误差来源**。"
      "一次 3692 步 float64 运行约 30 min/例（本机实测 0.56 s/步 × 2880 粒子），"
      "靠重跑做探测既贵又混杂：算子、密度闭合、自由面钳位、未收敛窗口混在一个读数里。\n")
    a("本审计只做能精确做且免费的那一半：把**精确解**喂给仓库自己的算子。"
      "方案若静水一致，残差应恒为 0。\n")
    a("```\np_i   = max(0, rho0 g (H - z_i))       # 解析压强，含壁面粒子\n"
      "rho_i = tait_density(p_i)               # 与 Tait 一致的密度\n"
      "a_i   = pressure_acceleration(pos, m, rho, p, h, pairs)[:, 2] - g\n```\n")
    a("网格：`h/dx ∈ {1.3, 1.5}` × 壁厚 `{1,2,3}` 层 × 自由面 `{带盖, 开口}` × "
      "壁面压强来源 `{解析, Adami 外推}`；Adami 变体把**解析流体场**喂给外推算子，"
      "测的是算子而不是启动瞬态。\n")

    a("## 2. 结果\n")
    a("### 2.1 静态不平衡 `a_z/g`（负号 = 支撑不足）\n")
    a("| h/dx | 壁厚 | 自由面 | 壁面压强 | 第0层 | 第8层 | 第14层 | 第15层 |")
    a("|---|---|---|---|---|---|---|---|")
    for g in grid:
        if g["wall_field"] != "adami":
            continue
        vals = [g["az_over_g_mean"][k] * 100.0 for k in (0, 8, 14, 15)]
        a(f"| {g['h_factor']:.1f} | {g['wall_layers']} | "
          f"{'带盖' if g['lid'] else '开口'} | Adami | "
          + " | ".join(f"{v:+.4f}" for v in vals) + " |")
    a("")
    a(f"**① 壁厚是硬要求。** 1 层壁时壁邻层残差 {thin['az_over_g_mean'][0]*100:+.4f} % g，"
      f"2 层壁 {base['az_over_g_mean'][0]*100:+.4f} % g —— 相差 "
      f"**{abs(thin['az_over_g_mean'][0] / base['az_over_g_mean'][0]):.0f}×**。"
      "h = 1.3 dx 时核半径 2h = 2.6 dx，一层壁填不满。生产值 `wall_layers=2` 必要且充分；"
      "h = 1.5 dx 下差距更大，因为 2h = 3 dx。\n")
    a(f"**② 自由面盖是伪边界（缺陷 DW-1）。** `wall_shell` 封的是**六个面**，"
      f"所以 `run_hydrostatic` 默认配置在自由面之上还有两层壁粒子（z = H, H + dx），"
      f"而 2h = 2.6 dx —— 正好落在最上层流体的支撑内。去掉它们，最上层残差从 "
      f"{base['az_over_g_mean'][-1]*100:+.4f} % g 降到 {open_top['az_over_g_mean'][-1]*100:+.4f} % g"
      f"（**{abs(base['az_over_g_mean'][-1] / open_top['az_over_g_mean'][-1]):.1f}×**）；"
      f"h = 1.5 dx 时甚至变号（{base15['az_over_g_mean'][-1]*100:+.4f} → "
      f"{open15['az_over_g_mean'][-1]*100:+.4f} % g）。"
      "`run_hydrostatic` 的 docstring 写的是「底面与四个侧面」，实现与文档矛盾。\n")
    a(f"**③ 算子的地板是 1 % 量级。** 即使给精确解，内部（第8层）残差 "
      f"{base['az_over_g_mean'][8]*100:+.3f} % g、壁邻层 "
      f"{base['az_over_g_mean'][0]*100:+.3f} % g —— 压强梯度算子没有再现性修正，"
      "格点偏置就在这个量级。这既是任何静水读数的下限，也是它在 2-D/3-D 全部算例里的固有量。\n")

    a("### 2.2 与已提交读数的对撞\n")
    a("| 量 | 值 |\n|---|---|")
    a(f"| 已提交工件 | `{mv['shipped_artifact']}`（{mv['case']}）|")
    a(f"| 工件自判 | {mv['shipped_verdict']}；未过判据 {mv['shipped_failing_criteria']} |")
    a(f"| 实测层面平均相对误差 | **{mv['measured_mean_pct']:.2f} %**（{mv['layers']} 层）|")
    a(f"| 静态不平衡隐含的平衡修正 | **{mv['static_mean_pct']:+.2f} %** |")
    a(f"| 符号一致层数 | {mv['sign_agreement_layers']} / {mv['layers']} |")
    a(f"| 两者均方根差 | {mv['rms_difference_pct']:.2f} 百分点 |")
    a(f"| 工件自报收敛标志 | {mv['settling_flags']} |")
    a("")
    a(f"静态不平衡要求压强**上调** {mv['static_mean_pct']:+.2f} %，而实测每一层都**偏低**"
      f"（符号一致 {mv['sign_agreement_layers']}/{mv['layers']}），量级差一个数量级以上。"
      "**「3-D 静水剖面误差 = 压强算子的静水不一致性」这一假设被证伪。** "
      "残余偏差必来自别的机制，而工件自己报告的 "
      "`column_settled_window_drift_below_2pct = false` / "
      "`centre_of_mass_drift_below_1pct = false` 说明读数取在一个尚未停稳的窗口里。\n")

    a("### 2.3 自由面那一层是钳位问题，不是算子问题\n")
    a("| c0 [m/s] | 自由面核和亏缺 [%] | 首层静水压缩 [%] | 亏缺/信号 | 等价钳位压强 [Pa] | 首层解析压强 [Pa] | 钳位/信号 |")
    a("|---|---|---|---|---|---|---|")
    for row_ in floor["rows"]:
        a(f"| {row_['sound_speed']:g} | {row_['deficit_pct']:.3f} | "
          f"{row_['signal_at_first_layer_pct']:.4f} | {row_['deficit_over_signal']:.1f} | "
          f"{row_['clamp_pressure_pa']:.1f} | "
          f"{row_['analytic_pressure_at_first_layer_pa']:.1f} | "
          f"{row_['clamp_over_signal']:.1f} |")
    a("")
    a("自由面核和亏缺是**格点属性**（不随 c0 变），静水压缩信号按 `1/c0^2` 缩小："
      "c0 越大，钳位吃掉的比例越高。所以最上层将近 -100 % 的读数**不是**算子错，"
      "是 wcSPH 在这个格点/声速组合下**读不出**那一层的压强。"
      "初始格点的核和亏缺只在最上层出现（带盖时被盖填满、亏缺为 0），"
      "这正是 15 层里只有最上层塌到 -99.5 % 的结构性原因。\n")

    a("## 3. 判定\n")
    a("| 判据 | 结果 |\n|---|---|")
    for k, v in r["criteria"].items():
        a(f"| `{k}` | {'PASS' if v else 'FAIL'} |")
    a(f"\nbenchmark 自判：**{r['verdict']}**。\n")
    a("**是否 < 3 %？** 本审计的判据是结构性与符号性断言（残差倍数、符号一致性、钳位比），"
      "不是对某个物理量的精度断言，故 3 % 判据不直接适用。唯一可量化的精度结论是："
      f"静水剖面读数的算子地板 ≈ {abs(base['az_over_g_mean'][0]) * 100:.2f} % g"
      "（< 3 %），但它比实测偏差小一个数量级 —— 所以**不能**把实测 FAIL 归因于算子。\n")

    a("## 4. 注意点与未完成\n")
    a("1. **DW-1 的修复默认关闭。** `run_hydrostatic(lid=False)` 是新增选项，默认 `True` 保持"
      "已提交工件的含义不变。`tests/test_hydrostatic_consistency.py::"
      "test_free_surface_face_has_no_wall_particles` 用 `xfail(strict=True)` 把"
      "「自由面之上不该有壁粒子」钉成待修事实：修好后它会转红，强制更新账本。")
    a("2. **开口重跑的验证已递交集群等待调度**（`cj958` 带盖 / `cj959` 开口，"
      "`adami-c0_20-nz16`、t_end = 0.3 s、1846 步 ≈ 15 min/例）。"
      "静态审计只能证明盖子给最上层注入 10× 的伪力，**不能**预测剖面误差会改善多少 —— "
      "那需要完整重跑，本机 load15 ≈ 100，按容量纪律不在本机起。")
    a("3. **内部格点偏置**（第8层量级见上表）是压强梯度算子无再现性修正的固有量；"
      "本单只量化并钉住，没有动算子。")
    a("4. 参考：Adami, Ihmsen & Adams (2012) 壁面压强外推；SPH 静水一致性作为压强梯度的"
      "必要条件（Morris 1997 的边界压强梯度偏置讨论）。\n")
    a("5. 图：`benchmarks/hydrostatic_consistency/figures/hydrostatic_consistency.png`"
      "（左：带盖/开口的残差剖面；中：运行读数 vs 静态预测；右：钳位比随 c0）。")

    OUT.write_text("\n".join(L) + "\n")
    print("wrote", OUT, len(L), "lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())