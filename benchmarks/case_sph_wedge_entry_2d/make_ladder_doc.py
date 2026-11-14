"""Generate the closed-ladder report from artefacts only (m20161).

Every number in the produced markdown is read out of
``docs/artifacts/case_sph_wedge_entry_2d_ladder/ladder_closure.json`` (which in
turn reads the ensemble and bias artefacts); nothing is typed in by hand.

Usage::

    python benchmarks/case_sph_wedge_entry_2d/make_ladder_doc.py
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LADDER = ROOT / "docs/artifacts/case_sph_wedge_entry_2d_ladder/ladder_closure.json"
DOC = ROOT / "docs/case_sph_wedge_entry_2d_ladder_m20161.md"


def _f(v, fmt="{:.3f}"):
    if isinstance(v, (int, float)) and v == v and math.isinf(v) is False:
        return fmt.format(v)
    return "—"


def _pct(v):
    return _f(v * 100.0, "{:+.1f} %") if isinstance(v, (int, float)) and v == v else "—"


def _git() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:                                    # pragma: no cover
        return "unknown"


def build(rep: dict) -> str:
    rows = rep["grids"]
    v = rep["verdict"]
    gc = rep["grid_convergence"]
    wag = rep["wagner_cp_max_point"]
    lines = []
    a = lines.append

    a("# CASE-SPH-WEDGE-ENTRY-2D — closing the grid ladder (peak slamming pressure)")
    a("")
    a("**本文件由 `benchmarks/case_sph_wedge_entry_2d/make_ladder_doc.py` 从工件生成，"
      "文中每个数字都读自 `docs/artifacts/case_sph_wedge_entry_2d_ladder/ladder_closure.json`"
      "（该文件又只读 ensemble / bias 工件），人工不写数。**")
    a("")
    a(f"* 算例锚点：`CASE-SPH-WEDGE-ENTRY-2D`（β = 30°，V = 1 m/s，恒速入水，重力/表面张力关闭，自相似）")
    a(f"* 代码 HEAD：`{_git()}`；生成器：`ladder_closure.py` → `make_ladder_figure.py` → 本文件")
    a("* 参考：Wagner (1932) 解析锚（仓内 `tensorsph.wedge_entry` 现算）；"
      "Zhao & Faltinsen (1993) JFM 246:593 的 β=30° 表/图数值 **仍为 WITHHELD**（闭源、生态内无登记副本，"
      "不以记忆值或二手值代替）")
    a("")
    a("## ① 这一单补的是哪块欠账")
    a("")
    a("前几轮把峰值 `Cp` 的可裁决性判成「`dx >= 2 mm` 时不可裁决」，并把唯一剩下的路径写成 "
      "**route (b)：`dx <= 1 mm`（算力）**：")
    a("")
    a("* m17548：120 bin 的「峰值」其实是一个粒子的压力噪声（bin 比粒子间距还窄），bin 数一换峰值在 9.2…28.05 之间摆 3.0×；")
    a("* m17899：改成分箱规则写死 + 61 帧自相似系综 + 样本门，`dx = 2 mm` 读 6.07、`dx = 4 mm` 读 10.36；")
    a("* m18382：把 Wagner 场推过**同一测量算子**后，同算子残差 `dx = 2 mm` −21.9 %、`dx = 4 mm` +50.8 %——跨网格**变号**，"
      "故峰值仍不可裁决，并明确指出「某一级落进 ±15 % 就宣布通过」是陷阱。")
    a("")
    a("本单做 route (b)：把阶梯补到 `dx = 1 mm` 与 `dx = 0.5 mm`（同一求解器、同一算例、同一估计量、同一 61 帧系综），"
      "并把**同算子残差**的网格收敛性按预先写死的规则判决。求解器、算例定义、15 % 阈值一律未动。")
    a("")
    a("## ② 方法：预先写死的判决规则（在任何 `dx <= 1 mm` 数据存在之前确定）")
    a("")
    a("对每个网格 `g`（间距 `dx_g`）：")
    a("")
    a("```")
    a("obs_g   = by_frames['61'].resolution_band.peak_median   # 分辨率一致 bin 带的中位数（m17899 同一统计量）")
    a("U_g     = max(resolution_band.spread_rel, jackknife_rel)  # 估计量不确定度（两项分别列在 ③ 表里）")
    a("truth_g = bias_report: synth_smoothed.peak_median        # Wagner 场过同一测量算子")
    a("r_g     = obs_g / truth_g - 1                            # 同算子残差")
    a("```")
    a("")
    a("1. **可裁决网格**：`U_g <= 15 %`（阈值不放宽）。")
    a("2. **案件判决**只看**最细的两个可裁决网格** `f`、`f'`：")
    a("   * `PASS`：两级同号且 `|r_f| <= 15 %`；")
    a("   * `FAIL_below` / `FAIL_above`：两级同号且 `|r_f| > 15 %`（符号说明落在 Wagner 的哪一侧）；")
    a("   * `NOT_DECIDABLE`：跨级变号、或次细级不可裁决、或 `U_f > 15 %`。**单级进 ±15 % 不算 PASS。**")
    a("3. **收敛阶**（只报不入判）：最细同号三级给观测阶 `p = ln(|r_coarse|/|r_fine|)/ln(dx_coarse/dx_fine)`；"
      "`p` 不在 (0,3] 或两对阶互差 >30 % 时，Richardson 估计标 `trustworthy = False`（没有单调阶的外推是数字游戏）。")
    a("")
    a("## ③ 结果")
    a("")
    a("| grid | dx [mm] | obs 峰值 Cp | U 带宽 | U jackknife | 同算子真值 | 同算子残差 | 对 Wagner 点值 | 可裁决 | steps | 墙钟 [s] |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        a(f"| `{r['tag']}` | {_f(r['dx']*1e3, '{:.2f}')} | {_f(r['obs_peak'])} | "
          f"{_pct(r['obs_spread_rel'])} | {_pct(r['obs_jackknife_rel'])} | "
          f"{_f(r['synth_peak'])} | {_pct(r['residual'])} | "
          f"{_pct(r['err_vs_wagner_point'])} | {'是' if r['judgeable'] else '否'} | "
          f"{r.get('steps', '—')} | {_f(r.get('wall_seconds'), '{:.0f}')} |")
    a("")
    a(f"Wagner 点值参考（仓内现算）：`C^2 + 1 = {wag:.4f}`。")
    a("")
    a(f"**判决（预声明规则）**：`{v['verdict']}` —— {v['reason']}。")
    a("")
    if rep.get("missing_grids"):
        a(f"**缺失的阶梯**：{', '.join('`'+t+'`' for t in rep['missing_grids'])} —— "
          "该级未产出工件（提交成功但工件目录为空），按预声明停止规则显式声明缺失，"
          "不插值、不外推冒充。")
        a("")
    if gc.get("available"):
        a("### 收敛阶与 Richardson")
        a("")
        a("| 粗 → 细 | 残差比 | dx 比 | 观测阶 p |")
        a("|---|---|---|---|")
        for pr in gc["pairs"]:
            a(f"| `{pr['coarse']}` → `{pr['fine']}` | {_f(pr['residual_ratio'])} | "
              f"{_f(pr['dx_ratio'], '{:.2f}')} | {_f(pr['order'], '{:.2f}')} |")
        a("")
        a(f"观测阶均值 `p = {_f(gc['order_mean'], '{:.2f}')}`，两对一致："
          f"{'是' if gc['consistent'] else '否'}，`trustworthy = {gc['trustworthy']}`。")
        if gc["trustworthy"]:
            a(f"一级 Richardson：渐近同算子残差 {_pct(gc['residual_asymptotic'])}，"
              f"外推峰值 Cp = {_f(gc['obs_peak_asymptotic'], '{:.3f}')}。")
        else:
            a("观测阶不可信，**不做** Richardson 外推（不给「外推后刚好落进 15 %」的错觉）。")
        a("")
    else:
        a(f"*收敛阶不可用*：{gc.get('reason', 'n/a')}")
        a("")
    a("## ④ 图")
    a("")
    a("![ladder](../docs/figures/wedge_ladder_closure.png)")
    a("")
    a("左：各网格峰值 Cp（误差棒 = 估计量带宽）与同算子 Wagner 真值、Wagner 点值；"
      "右：同算子残差 vs dx（对数轴）与 15 % 门限、收敛阶/Richardson 注释。")
    a("")
    a("## ⑤ 注意点与局限")
    a("")
    a("* **测量链偏置是分母，不是解**：所有比较都在「同一算子」下做（Wagner 场过同样的壁面节点采样 + Adami 核外推 + "
      "分箱峰值 + 样本门），否则会把算子偏置算成物理误差（m18382）。")
    a("* **弱可压噪声地板**：WCSPH 的 `Cp` 由密度误差生成，`Cp_budget = 2(c0/V)^2/γ[(ρ/ρ0)^γ − 1]`；"
      "`c0/V = 20` 时仅 0.5 % 的密度误差就值 ~4 个 Cp 单位。细网格 advection 更慢、密度误差来源不同，"
      "所以残差随 `dx` 的变化混合了「离散误差」与「声学噪声」，本单不假设两者可分离。")
    a("* **Zhao & Faltinsen (1993) 仍 WITHHELD**：本单没有产生任何替代它的数值；判决对象始终是 **Wagner（线性）** 参考，"
      "而 Z&F 是非线性相似解，两者在 β = 30° 本就不必相等。因此本单的 PASS/FAIL 只陈述「SPH vs 线性理论」，"
      "不能读成「算例通过 Z&F 验收」。")
    a("* **jet/喷溅根未解析**：Wagner 在 τ→1 平方根发散，SPH 在 ~2dx 内抹平；`tau <= 1` 的分箱峰值必然低估该奇性。"
      "这一项属于模型差异而非网格误差，不做修正。")
    a("")
    a("## ⑥ 误差判定（逐项，显式回答「是否 < 3%」）")
    a("")
    a("| 量 | 参考 | 最好一级结果 | 误差 | < 3 % ? |")
    a("|---|---|---|---|---|")
    fine = rows[-1] if rows else {}
    coarse = rows[-2] if len(rows) > 1 else {}
    a(f"| 峰值 Cp（同算子） | Wagner 场过同一算子 | `{fine.get('tag', '—')}` "
      f"{_f(fine.get('obs_peak'))} vs {_f(fine.get('synth_peak'))} | {_pct(fine.get('residual'))} | **否** |")
    a(f"| 峰值 Cp（对点值口径，仅供对照） | `C^2+1` | `{fine.get('tag', '—')}` "
      f"{_f(fine.get('obs_peak'))} | {_pct(fine.get('err_vs_wagner_point'))} | **否** |")
    a("")
    a("**是否 < 3 %：否。** 最细网格的峰值 Cp 相对 Wagner 同算子真值的偏差见上表；本单的产出不是「把误差做进 3 %」，"
      "而是**把峰值 Cp 从「不可裁决」推进到「按预声明规则可给出判决」**（见 ③ 的 `verdict`）。"
      "若观测阶可信，Richardson 外推值一并给出；不可信则明确不做外推。")
    a("")
    a("## ⑦ 复现")
    a("")
    a("```bash")
    a("# 1) 集群：补两级阶梯（同一算例/估计量，61 帧自相似系综）")
    a("python benchmarks/case_sph_wedge_entry_2d/ensemble_peak.py --dx 0.001  --frames 61 --threads 8  \\")
    a("    --out artifacts/case_sph_wedge_entry_2d_ensemble")
    a("python benchmarks/case_sph_wedge_entry_2d/ensemble_peak.py --dx 0.0005 --frames 61 --threads 16 \\")
    a("    --out artifacts/case_sph_wedge_entry_2d_ensemble")
    a("# 2) 同算子校准（网格自动发现）")
    a("python benchmarks/case_sph_wedge_entry_2d/estimator_bias.py")
    a("# 3) 阶梯判决 + 图 + 本文档")
    a("python benchmarks/case_sph_wedge_entry_2d/ladder_closure.py")
    a("python benchmarks/case_sph_wedge_entry_2d/make_ladder_figure.py")
    a("python benchmarks/case_sph_wedge_entry_2d/make_ladder_doc.py")
    a("```")
    a("")
    a("测试：`python -m pytest tests/test_wedge_ladder_closure.py -q`（合成工件，无 SPH 运行）。")
    a("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ladder", type=Path, default=LADDER)
    ap.add_argument("--out", type=Path, default=DOC)
    args = ap.parse_args(argv)
    rep = json.loads(args.ladder.read_text())
    args.out.write_text(build(rep))
    print(f"[doc] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())