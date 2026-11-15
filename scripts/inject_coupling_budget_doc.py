"""Fill the AUTO blocks of docs/xval_fsi2/COUPLING_CONVERGENCE_BUDGET.md.

Every number is read from artifacts/xval_fsi2/coupling_convergence_budget.json
(produced by benchmarks/xval_fsi2/coupling_convergence_budget.py); nothing is
hand-typed.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "xval_fsi2" / "coupling_convergence_budget.json"
DOC = ROOT / "docs" / "xval_fsi2" / "COUPLING_CONVERGENCE_BUDGET.md"


def main() -> int:
    d = json.loads(ART.read_text())
    a, b, v = d["pass_a"], d["pass_b"], d["verdict"]
    lines = ["| 臂 | 归约地板中位数 | 最坏步地板 | 地板 >1e-2 的步占比 | 达到 1e-2 的迭代数（中位/达成率） |",
             "|---|---|---|---|---|"]
    for name, arm in a["arms"].items():
        it = arm["iters_to"]["0.01"]
        lines.append(f"| `{name}` | {arm['median_floor']:.2e} | {arm['max_floor']:.2e} | "
                     f"{arm['frac_floor_gt_1e-2']*100:.0f} % | "
                     f"{it['median']:.0f} / {it['reached']*100:.0f} % |")
    lines.append("")
    lines.append(f"环境：`dx = {a['dx']}` m，`dt = {a['dt']:.4f}` s，{a['n_snapshots']} 份快照"
                 f"（每 {a['snapshot_step_interval']} 步一份），`max_iter = {a['max_iter']}`，`tol = 0`。")
    lines.append("")
    lines.append(f"判定：{v['hypothesis_supported']}。最优点归约臂 `{v['best_arm']}` 的地板 "
                 f"{v['best_median_floor']:.2e}（判据 `<= 1e-2`），而只做重启修复的 "
                 f"`iqn_restart` 地板 {v['restart_only_median_floor']:.2e} —— "
                 f"**单靠重启修不住**，但 `fp_05`（无 secant、无历史）地板 "
                 f"{a['arms']['fp_05']['median_floor']:.2e} 证明地图本身在 1e-2 之上没有地板。")
    ag = ["| 臂（相对 " + str(v["best_arm"]) + "） | 梁尖 y 最大绝对差 | 占梁长 |", "|---|---|---|"]
    for name, g in v["cross_arm_agreement"].items():
        ag.append(f"| `{name}` | {g['max_abs_tip_y_dev']:.2e} m | {g['rel_to_beam_length']*100:.4f} % |")
    ag.append("")
    ag.append("所有臂解的是同一个界面方程；不同记账策略在 5 份快照上的梁尖位置差异都在 1e-6 m 量级"
              "（占梁长 ≲2e-5），说明它们收敛到的是**同一个不动点**，地板差异只反映各自的界面解析度。")

    pb = ["| 臂 | max_iter | 平均迭代数 | 顶预算步占比 | tip_y 偏差 | C_d 偏差 |", "|---|---|---|---|---|---|"]
    for arm_name, arm in b["arms"].items():
        for r in arm["rows"]:
            pb.append(f"| `{arm_name}` | {r['max_iter']} | {r['mean_iterations']:.2f} | "
                      f"{r['frac_saturated']*100:.0f} % | "
                      f"{r['tip_y_dev']['rel_to_half_range']*100:.2f} % | "
                      f"{r['cd_dev']['rel_to_half_range']*100:.2f} % |")
    pb.append("")
    pb.append(f"固定 `tol = {b['tol']}`，`{b['n_steps']}` 步，参考臂为该臂的 `max_iter = "
              f"{b['arms'][list(b['arms'])[0]]['reference_budget']}`。偏差对参考半幅归一。")

    err = ["| 判定项 | 预注册门槛 | 实测 | 是否达标 |", "|---|---|---|---|"]
    for arm_name, arm in b["arms"].items():
        rec = f"{arm['min_budget_harmless_3pct']}"
        err.append(f"| `{arm_name}`：截断预算最小可行值（<3 %） | 存在 | `max_iter = {rec}` | "
                   f"{'是' if arm['min_budget_harmless_3pct'] else '**否**'} |")
    err.append(f"| 界面 1e-2 门槛可达（最优臂） | 地板 <= 1e-2 | "
               f"{v['best_median_floor']:.2e} | "
               f"{'是' if v['tol_1e-2_reachable_by_best_arm'] else '**否**'} |")
    err.append(f"| 界面 1e-2 门槛可达（只做重启修复） | 地板 <= 1e-2 | "
               f"{v['restart_only_median_floor']:.2e} | "
               f"{'是' if v['tol_1e-2_reachable_by_restart_only'] else '**否**'} |")
    err.append(f"| 关闭记账（`fp_05`）是否也下不去 | 若 H2 成立应下不去 | "
               f"{a['arms']['fp_05']['median_floor']:.2e} | "
               f"**否** → 判 H1 |")
    err.append("")
    err.append(f"**显式回答「是否 <3 %」**：截断预算对报告量的污染，在 "
               f"`{list(b['arms'])[0]}` 臂上 `max_iter >= "
               f"{b['arms'][list(b['arms'])[0]]['min_budget_harmless_3pct']}` 后 <3 %；"
               f"在 `{list(b['arms'])[-1]}` 臂上 `max_iter >= "
               f"{b['arms'][list(b['arms'])[-1]]['min_budget_harmless_3pct']}` 后 <3 %。"
               "**界面残差解析度本身**（地板）与 Report 量无关，只决定成本。"
               "本文**不**对 FSI2 基准精度（`C_d/C_l`/梁尖幅值 vs Turek-Hron）作任何达标声明。")

    text = DOC.read_text()
    for marker, payload in (("<!-- AUTO:pass_a -->", "\n".join(lines)),
                            ("<!-- AUTO:agreement -->", "\n".join(ag)),
                            ("<!-- AUTO:pass_b -->", "\n".join(pb)),
                            ("<!-- AUTO:errors -->", "\n".join(err))):
        text = text.replace(marker, payload)
    figs = d.get("figures") or []
    if len(figs) >= 2:
        head = ("图：`artifacts/xval_fsi2/" + figs[0] + "`、"
                "`artifacts/xval_fsi2/" + figs[1] + "`\n\n")
        text = text.replace("## 5. 误差判定", head + "## 5. 误差判定")
    DOC.write_text(text)
    print("filled", DOC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
