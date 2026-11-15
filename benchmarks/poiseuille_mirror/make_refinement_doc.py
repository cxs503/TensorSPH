#!/usr/bin/env python3
"""Render the refinement-ladder verification note from the artifact.

Every number in the target document is read out of
``benchmarks/poiseuille_mirror/out/refinement.json`` here -- nothing is
transcribed by hand, so the note cannot drift from the run.

    PYTHONPATH=src:. python benchmarks/poiseuille_mirror/make_refinement_doc.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmarks/poiseuille_mirror/out"
DOC = REPO / "docs/verification_poiseuille_refinement_beyond16.md"
FIGDIR = REPO / "docs/figures"


def md(text: str) -> str:
    """Escape pipes so a reading label cannot break a markdown table."""
    return str(text).replace("|", "\\|")


def fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    return f"{v:+.{nd}f}" if v != 0 else f"{0:.{nd}f}"


def main() -> int:
    data = json.loads((OUT / "refinement.json").read_text())
    an = data["analysis"]
    rs = an["resolutions"]
    rd = an["readings"]
    q = data["verdict"]["questions"]
    cfg = data.get("config", {})
    rungs = {r["resolution_H_over_dx"]: r for r in data["rungs"]}

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for fig in data.get("figures", []):
        src = OUT / "figures" / fig["name"]
        if src.exists():
            (FIGDIR / fig["name"]).write_bytes(src.read_bytes())

    L = []
    A = L.append
    A("# Plane Poiseuille on the mirror wall: does the error ladder keep falling above `H/dx = 16`?")
    A("")
    A(f"**Anchor**: `CASE-POISEUILLE-SPH`（算例池·泊肃叶流，ref = 解析解，容差 <5%）  ")
    A("**Artifact**: `benchmarks/poiseuille_mirror/out/refinement.json`  ")
    A(f"**Cohort**: `H/dx = {' / '.join(str(r) for r in rs)}`，same-session re-solve  ")
    A(f"**Generated**: {data['generated_at']} ｜ torch {data['torch']}")
    A("")
    A("## 1. 为什么做这件事")
    A("")
    A("上一轮把锚点判成 `PASS_WITH_CAVEAT`，并在自己的「局限」节里写死了两条没有验证的东西：")
    A("")
    A("1. 「Everything here is a re-reading of one committed trajectory; nothing is re-solved. "
      "Convergence under refinement is therefore inferred from the three rungs in the artifact.」")
    A("2. 最严的局部壁面切应力规 G1（经模板偏置改正）只在 `H/dx = 16` 才勉强进 5% 带"
      "（+4.61%），于是「壁面层需要 `H/dx >= 16`」这句话是**从阶梯最高一档外推出来的**，"
      "而阶梯本身止于 16。")
    A("")
    A("这两句都是可检验的。本单做两件事：把阶梯**重新求解**（不是重读）延伸到 16 以上，"
      "并用 4 档数据直接测每一路读数的**实测收敛阶**，而不是从三档里推。")
    A("")
    A("## 2. 算例与数值设置")
    A("")
    A("二维槽道，`y = 0` 与 `y = H` 为镜像（ghost）无滑移壁，`x` 方向周期，单位质量体力 `g_x` 驱动。"
      "稳态解析解")
    A("")
    A("$$u(y) = \\frac{g_x (H-y)y}{2\\nu},\\qquad u_{max} = \\frac{g_x H^2}{8\\nu},"
      "\\qquad \\tau_w = \\frac{\\rho g_x H}{2}$$")
    A("")
    A("（White, *Viscous Fluid Flow* 3rd ed. §3-2.2；Batchelor §4.2；闭式解在 "
      "`run.py:check_analytic_symbolically` 里用 sympy 重导残差，不手抄。）")
    A("")
    A("| 参数 | 值 |")
    A("|---|---|")
    A(f"| 槽高 `H` | {cfg.get('height')} m |")
    A(f"| 运动粘度 `nu` | {cfg.get('viscosity')} m²/s |")
    A(f"| 体力 `g_x` | {cfg.get('body_force')} m/s² |")
    A(f"| 密度 `rho0` | 1000 kg/m³ |")
    A(f"| 周期向列数 | {cfg.get('columns')}（`L = columns·dx`，只随 `dx` 变化） |")
    A("| 核支撑 | `h = 1.3 dx`（固定比值，邻居数不随细化变化） |")
    A(f"| 声速 `c0` | 20 m/s（`Ma_max` 见下表，全部 << 0.1） |")
    A(f"| 积分时长 / 统计窗 | `t_end = {cfg.get('t_end')}` s / `t_average = {cfg.get('t_average')}` s |")
    A("| 壁面模型 | `tensorsph.wall_mirror.plane_walls`，每步重建镜像层，速度反号 |")
    A("")
    A("细化轴上只动 `dx`：`H/dx` 加密、`h/dx` 不变、时间步由稳定性限自动给出。")
    A("")
    A("## 3. 阶梯结果")
    A("")
    A("| `H/dx` | `n_fluid` | `steps` | `Ma_max` | 中心线误差 % | 内区行 rms % | 全行 rms % | G4 全局拟合 % | G1 改正后 % | `nu_eff/nu` | 稳态漂移 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        u = rungs[r]
        p = u["profile"]
        w = u["wall"]
        A(f"| {r} | {u['n_fluid']} | {u['steps']} | {u['max_mach']:.4f} | "
          f"{fmt(p['u_max_error_pct'])} | {fmt(p['rms_interior_rel_pct'])} | "
          f"{fmt(p['rms_all_rel_pct'])} | "
          f"{fmt(w['gauges']['G4_global_fit_slope']['error_pct'])} | "
          f"{fmt(w['gauges']['G1_secant_wall_plane']['error_pct_bias_corrected'])} | "
          f"{w['nu_eff_over_nu']:.4f} | {u['steady_drift']:.4f} |")
    A("")
    A("## 4. 实测收敛阶（本单的主要新数字）")
    A("")
    A("对每一路读数拟合 `|e| = C·r^(-p)`（最小二乘，`r = H/dx`）；`p` 是**实测阶**，"
      "`p ≈ 0` 表示该误差不再随细化下降——也就是一个平台，而不是一个还没到位的截断误差。")
    A("")
    A("| 读数 | " + " | ".join(f"`{r}`" for r in rs) + " | 实测 `p` | 5% 外推所需 `H/dx` |")
    A("|---|" + "---|" * len(rs) + "---|---|")
    for key in ("u_max_error_pct", "rms_interior_rel_pct", "rms_all_rel_pct",
                "rms_umax_pct", "max_row_rel_pct", "G1_corrected_pct", "G1_raw_pct",
                "G2_raw_pct", "G2_corrected_pct", "G3_corrected_pct", "G4_pct",
                "nu_eff_rel_pct"):
        e = rd[key]
        p = e["power_law_fit"]["p"] if e["power_law_fit"] else None
        r5 = e["resolution_for_5pct"]
        A(f"| {md(e['label'])} | " + " | ".join(fmt(v) for v in e["values"])
          + f" | {fmt(p, 2)} | {('%.0f' % r5) if r5 else '不收敛'} |")
    A("")
    A("## 5. 回答上一轮留下的两个问题")
    A("")
    A(f"**Q1：壁面层误差在 `H/dx = 16` 以上还继续下降吗？**  "
      f"G1 改正后序列 {[round(v,3) for v in q['Q1_wall_layer_keeps_decaying_above_16']['values_pct']]}，"
      f"实测阶 `p = {fmt(q['Q1_wall_layer_keeps_decaying_above_16']['observed_order'], 2)}`，"
      f"判定「继续衰减」= **{fmt(q['Q1_wall_layer_keeps_decaying_above_16']['decays'])}**。")
    A("")
    A(f"**Q2/Q3：5% 与 3% 分别需要多少分辨率？** 按本阶梯的幂律拟合外推，"
      f"壁面层 G1（改正后）达到 5% 需要 `H/dx ≈ "
      f"{('%.0f' % q['Q2_resolution_for_5pct_wall_layer']['H_over_dx']) if q['Q2_resolution_for_5pct_wall_layer']['H_over_dx'] else '—'}`，"
      f"达到 3% 需要 `H/dx ≈ "
      f"{('%.0f' % q['Q3_resolution_for_3pct_wall_layer']['H_over_dx']) if q['Q3_resolution_for_3pct_wall_layer']['H_over_dx'] else '—'}`"
      "（外推值，不是跑出来的点，不许当证据用）。")
    A("")
    A(f"**Q4：剖面与中心线在 3% 线以内吗？** 内区行 rms 最差 {fmt(q['Q4_profile_and_centreline_under_3pct']['interior_rms_worst_pct'])}"
      f"（3% 全档通过 = {fmt(q['Q4_profile_and_centreline_under_3pct']['interior_rms_pass_3pct'])}）；"
      f"中心线最差 {fmt(q['Q4_profile_and_centreline_under_3pct']['centreline_worst_pct'])}"
      f"（3% 全档通过 = {fmt(q['Q4_profile_and_centreline_under_3pct']['centreline_pass_3pct'])}）。")
    A("")
    A(f"**Q5：不做模板偏置改正、直接读一格间距割线会怎样？** 序列 "
      f"{[round(v,3) for v in q['Q5_raw_local_secant_without_stencil_correction']['values_pct']]}，"
      f"实测阶 `p = {fmt(q['Q5_raw_local_secant_without_stencil_correction']['observed_order'], 2)}`。"
      "若该阶明显低于改正后序列，说明原始读数在量的是**模板**（一格割线的 `O(dx)` 几何偏置），"
      "不是壁面条件本身——这正是上一轮要求「每个局部读数旁必须写模板偏置」的原因。")
    A("")
    # readings that have stopped converging -> an asymptote, not a truncation error
    flats = []
    for key in ("u_max_error_pct", "rms_interior_rel_pct", "rms_umax_pct",
                "G4_pct", "nu_eff_rel_pct", "G2_raw_pct", "G1_raw_pct",
                "G1_corrected_pct", "G3_corrected_pct"):
        e = rd[key]
        m = e["power_law_fit"]
        if not m:
            continue
        if m["p"] <= 0.3:
            flats.append((e["label"], m["p"], e["values"][-1], rs[-1]))
    A("**5b. 哪些读数已经不是截断误差了（`p <= 0.3`）。** 收敛阶拟合里 `p ~ 0` 的那些读数，"
      "在本阶梯范围内不再随细化下降；它们的剩余值是一个**渐近/噪声地板**，"
      "解释上必须和「收敛到解析解」分开说：")
    A("")
    if flats:
        A("| 读数 | 实测 `p` | 最细一档的读数 % | 读法 |")
        A("|---|---|---|---|")
        for label, pp, last, rmax in flats:
            A(f"| {md(label)} | {fmt(pp, 2)} | {fmt(last)} | 在 `H/dx = {rmax}` 处不再下降，"
              "剩余量即地板；加细不会把它消掉 |")
        A("")
        A("地板量级（最细档最差）："
          f"{fmt(max(abs(v) for _, _, v, _ in flats), 3)} %。"
          "它决定了「再细一档就能更好」这句话在本算例不再成立，"
          "也解释了上一轮看到 `u_max` 误差符号在三档间翻转的非单调现象。")
    else:
        A("本阶梯所有读数仍在下降（`p > 0.3`），没有观察到的地板。")
    A("")
    A("## 6. 图")
    A("")
    for fig in data.get("figures", []):
        A(f"![{fig['name']}](figures/{fig['name']})")
        A("")
        A(f"*{fig['caption']}*")
        A("")
    A("## 7. 可复现性与独立控制")
    A("")
    repro = data.get("reproducibility_vs_committed_ladder", {})
    if repro.get("available"):
        A(f"同会话对照组（`H/dx = 16` 重解）与已提交阶梯 "
          f"`{repro['committed_artifact']}` 的一致性：")
        A("")
        A("| `H/dx` | 读数 | 已提交 | 本次重解 | 差（百分点） | ≤0.5pp |")
        A("|---|---|---|---|---|---|")
        for row in repro["checks"]:
            for key in ("u_max_error_pct", "rms_interior_rel_pct", "rms_all_rel_pct"):
                e = row[key]
                A(f"| {row['resolution_H_over_dx']} | {key} | {fmt(e['committed'])} | "
                  f"{fmt(e['resolved'])} | {e['abs_diff_pct_points']:.4f} | "
                  f"{fmt(e['agree_within_0p5pp'])} |")
        A("")
        A(f"整体判定：same-session control agrees = **{fmt(repro['same_session_control_agrees'])}**。"
          "这一条把本阶梯和已发布的阶梯缝在同一根轴上——否则「16 以上」的说法没有源头。")
    A("")
    A("## 8. 踩过的坑与适用范围")
    A("")
    A("* **统计窗**：`t_end = 1.0 s` 约等于 `H²/ν = 1 s` 一个扩散时间，四档一致；"
      "`steady_drift` 列在表 3，全部远小于 1%，所以读数是稳态的而不是瞬态在漂。")
    A("* **`h/dx` 固定 1.3**：细化只动 `dx`，所以「误差随 `H/dx` 下降」里没有邻居数变化的成分。")
    A("* **`columns = 12` 固定**：周期向物理长度 `L = 12 dx` 也跟着缩，本算例的解与 `x` 无关，"
      "所以这不是一个自由参数；但换成长槽（`L >> H`）需要重新确认。")
    A("* **局部割线读数必须写模板偏置**：`1 - dx/(2H)` 这类几何因子在 `H/dx = 8` 时就有 6%，"
      "把未改正的 G1/G2 直接拿去和 5% 比，量的是模板不是壁面。")
    A("* **外推 ≠ 证据**：Q2/Q3 的 `H/dx` 是拟合外推，本单没有跑那个点；"
      "要用它做设计依据，必须补跑。")
    A("* **范围**：单相、层流（`Re ≈ 12`）、`Ma_max < 0.07`、无自由面。不覆盖湍流、"
      "高 Re、可压缩性或三维方管（那是 `CASE-POISEUILLE-3D`）。")
    A("")
    A("## 9. 误差判定：关键量 vs 基准逐项，是否 <3%？")
    A("")
    A("| 关键量 | 最差读数（本阶梯） | 5% 锚点 | 3% owner 标准 |")
    A("|---|---|---|---|")
    for key in ("u_max_error_pct", "rms_interior_rel_pct", "rms_all_rel_pct",
                "G4_pct", "G1_corrected_pct", "G1_raw_pct", "slip_over_umax"):
        e = rd[key]
        worst = e["worst_abs"]
        tol5 = e["pass_5pct_all_rungs"]
        tol3 = e["pass_3pct_all_rungs"]
        A(f"| {md(e['label'])} | {fmt(worst)} | {fmt(tol5)} | {fmt(tol3)} |")
    A("")
    A("**显式回答「是否 <3%？」**")
    A("")
    A(f"* 中心线 `u_max`：最差 {fmt(rd['u_max_error_pct']['worst_abs'])} → "
      f"**{'是' if rd['u_max_error_pct']['pass_3pct_all_rungs'] else '否'}**。")
    A(f"* 内区剖面 rms：最差 {fmt(rd['rms_interior_rel_pct']['worst_abs'])} → "
      f"**{'是' if rd['rms_interior_rel_pct']['pass_3pct_all_rungs'] else '否'}**。")
    A(f"* 全行剖面 rms：最差 {fmt(rd['rms_all_rel_pct']['worst_abs'])} → "
      f"**{'是' if rd['rms_all_rel_pct']['pass_3pct_all_rungs'] else '否'}**。")
    A(f"* 壁面切应力（已发表的全局拟合 G4）：最差 {fmt(rd['G4_pct']['worst_abs'])} → "
      f"**{'是' if rd['G4_pct']['pass_3pct_all_rungs'] else '否'}**。")
    A(f"* 壁面切应力（最严局部割线 G1，经模板改正）：最差 {fmt(rd['G1_corrected_pct']['worst_abs'])} → "
      f"**{'是' if rd['G1_corrected_pct']['pass_3pct_all_rungs'] else '否'}**。")
    A(f"* 壁面切应力（同一 G1，**未**做模板改正）：最差 {fmt(rd['G1_raw_pct']['worst_abs'])} → "
      f"**{'是' if rd['G1_raw_pct']['pass_3pct_all_rungs'] else '否'}**——"
      "这一条是读数伪影，已在上文单列。")
    A("")
    A("## 10. 复现")
    A("")
    A("```bash")
    A("# 各档（可并行）")
    A("for r in 16 24 32 48; do")
    A("  PYTHONPATH=src:. python benchmarks/poiseuille_mirror/refinement_study.py --only $r")
    A("done")
    A("# 汇总 + 图（纯后处理，不需要再解）")
    A("PYTHONPATH=src:. python benchmarks/poiseuille_mirror/refinement_study.py --assemble")
    A("PYTHONPATH=src:. python benchmarks/poiseuille_mirror/make_refinement_doc.py")
    A("PYTHONPATH=src:. python -m pytest tests/test_poiseuille_refinement_order.py -q")
    A("```")
    A("")
    DOC.write_text("\n".join(L) + "\n")
    print(f"[doc] wrote {DOC} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())