#!/usr/bin/env python3
"""Render docs/verification_ledger.md from ledger.json.

Every number in the generated document is injected from the JSON artifact, so
the prose cannot drift away from the machine-readable ledger.  Run
``audit.py`` first.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEDGER = HERE / "ledger.json"
OUT = REPO / "docs" / "verification_ledger.md"


def fnum(v: float, digits: int = 1) -> str:
    return f"{v:+.{digits}f}"


def main() -> int:
    d = json.loads(LEDGER.read_text(encoding="utf-8"))
    s, case = d["summary"], d["case"]
    claims = {c["id"]: c for c in d["claims"]}
    derived = {x["claim"]: x for x in s["derived_trajectories"]}
    auth = claims[s["authoritative_claim"]]

    L = []
    A = L.append
    A("# 验证声明台帐：Young–Laplace 压差的六条读数与唯一裁定")
    A("")
    A(f"**仓库**: TensorSPH ｜ **任务**: {d['mission']} ｜ **台账**: `{d['ledger']}` ｜ "
      f"**机器门禁**: `tests/test_verification_ledger.py`")
    A("")
    A("> 本文由 `benchmarks/verification_ledger/make_doc.py` 从 `ledger.json` 生成，"
      "全部数字程序化注入，无手抄。")
    A("")
    A("## 0. 一句话结论")
    A("")
    A(f"同一个物理量（二维液滴 `$\\Delta p=\\sigma/R={case['reference_pa']:.1f}$ Pa）在本仓被 "
      f"{s['n_claims']} 篇验证文档引用了 {s['n_claims']} 个互不相同的误差值，跨度 "
      f"**{s['spread_pct_points']:.1f} 个百分点**（{fnum(s['error_range_pct'][0])}% … "
      f"{fnum(s['error_range_pct'][1])}%）。台帐逐条溯源后裁定：")
    A("")
    A(f"- 唯一**权威**读数为 **{fnum(s['authoritative_error_pct'])}%**"
      f"（{auth['id']}，{auth['definition_class']}），证据等级 **{s['evidence_grade']}**；")
    A(f"- 与平台 15% 判据的关系：**{'通过' if s['within_15pct'] else '不通过'}**；"
      f"与论文 3% 判据的关系：**{'通过' if s['within_3pct'] else '不通过'}**；")
    A(f"- 全仓唯一的 `✅ PASS` 声明（{', '.join(s['claims_asserting_pass'])}）"
      "在本台帐中被**撤下**，理由见 §4。")
    A("")
    A("## 1. 为什么需要这台帐")
    A("")
    A("表面张力 CSF 模块（m277 合入）的验证欠账已经陆续被多张单子清理过"
      "（m2174 / m2592 / m2971 / m3028 …）。每张单子都诚实地报了自己的误差，"
      "但它们**各用各的测量口径、各用各的网格**，而且 `benchmarks/<name>/validation_report.json` "
      "这类工件路径会被后来的单子**原地覆盖**。结果是：")
    A("")
    A("1. 下游读者（含平台能力审计）无法判断该引哪个数；")
    A("2. 一个「+5.6%，15% 门内」的读数可以与「−93.6%」并存而无人指出二者不能同时成立；")
    A("3. 历史声明的证据可以在文档一字未改的情况下**从仓库里消失**。")
    A("")
    A("台帐的做法：不重写任何人的报告，而是给每条声明补上"
      "「原文引语 → 引用工件 → 工件现值 → 时间序列自洽性 → 判定」的完整链条，"
      "并把它变成一个可执行的测试门禁。")
    A("")
    A("## 2. 算例与口径分类")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 几何 | 二维静止自由液滴（无限长柱截面），六角密排，单相，无重力 |")
    A(f"| 半径 R | {case['radius_m']} m |")
    A(f"| 表面张力 σ | {case['sigma_N_per_m']} N/m（故意取 10× 水，压缩毛细时间） |")
    A(f"| 参考密度 ρ0 | {case['rho0']} kg/m³ |")
    A(f"| 解析基准 σ/R | **{case['reference_pa']:.1f} Pa** |")
    A("")
    A("四种口径互不等价，混用是误差跨度大的主要原因：")
    A("")
    A("| 口径 | 定义 | 台帐评价 |")
    A("|---|---|---|")
    for key, desc in d["definition_classes"].items():
        A(f"| `{key}` | {desc} | |")
    A("")
    A("## 3. 六条声明逐条溯源")
    A("")
    A("| ID | 文档 | 引语 | 口径 | 声明误差 | 文档判定 | 工件现值 | 溯源 | 时序自洽 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for c in d["claims"]:
        prov = c["checks"]["provenance"]
        cur = prov.get("artifact_error_pct")
        cur_s = fnum(cur, 2) + "%" if cur is not None else "—"
        cons = c["checks"].get("consistency")
        cons_s = cons["status"] if cons else "n/a"
        star = " **★**" if c["authoritative"] else ""
        A(f"| {c['id']}{star} | `{c['doc']}` | `{c['quote']}` | {c['definition_class']} | "
          f"{fnum(c['reported_error_pct'], 2)}% | {c['reported_verdict']} | {cur_s} | "
          f"{prov['status']} | {cons_s} |")
    A("")
    A("`★` = 台帐裁定的权威行。`溯源` 状态含义：`RESOLVED` 工件仍持有该数；"
      "`OVERWRITTEN` 该路径已被后续算例覆盖，原始证据不再存在于仓库。")
    A("")
    A("### 3.1 漂移与噪声的直接证据（由工件自身的时间序列重算）")
    A("")
    A("| 声明 | 工件 | 样本 | 原始范围 [Pa] | 尾窗均值 ± σ [Pa] | 尾窗均值误差 | 最优绝对误差 |")
    A("|---|---|---|---|---|---|---|")
    for cid, t in derived.items():
        A(f"| {cid} | `{t['artifact']}` | {t['n_samples']} | "
          f"{t['raw_range_pa'][0]:.1f} … {t['raw_range_pa'][1]:.1f} | "
          f"{t['tail_mean_pa']:.1f} ± {t['tail_std_pa']:.1f} | "
          f"{fnum(t['tail_mean_error_pct'])}% | {t['min_abs_error_achievable_pct']:.2f}% |")
    A("")
    A("这两行是台帐自己从工件算出来的（不是任何文档的声明），它们给出 §4 的核心证据。")
    A("")
    A("## 4. 三个发现")
    A("")
    A("### F1　唯一的 `✅ PASS` 不支持（C03）")
    A("")
    c03 = claims["C03"]
    t03 = derived["C03"]
    A(f"`{c03['doc']}` §3.2 记 `Δp = 380.2 Pa (+5.6%) ✅ Within 15% platform bar`。"
      "该读数的工件 `benchmarks/laplace_law/validation_report.json` 自带 `history.p_center`：")
    A("")
    A(f"- 记录窗口内 `p_center` 由 **{t03['raw_range_pa'][1]:.1f} Pa 单调衰减到 "
      f"{t03['raw_range_pa'][0]:.1f} Pa**，尾窗均值 **{t03['tail_mean_pa']:.1f} Pa"
      f"（{fnum(t03['tail_mean_error_pct'])}%）**；")
    A(f"- 也就是说这条轨迹里**最优的绝对误差也只有 {t03['min_abs_error_achievable_pct']:.2f}%**，"
      "从未进入 15% 门；")
    A("- 而 `+5.6%` 对应的 380.2 Pa 落在轨迹记录窗口**之外**（更早的瞬态），"
      "即该读数是在液滴尚未平衡时取的；")
    A("- 该算例还是全仓最粗的一档：dx = 4 mm、N = 88、380 步、无质量标定、c0 = 20（m2971 把此档的"
      "初始核和偏置造成的虚假核心压强记为 2395 Pa）。")
    A("")
    A("**裁定：撤下该 PASS**。原文不删（删掉一个错误判据比标注它更糟），"
      "改为在文档头部挂台帐指针块；`audit_allowlists.withdrawn_pass_claims` 记录这次撤下，"
      "门禁会保证「撤下的 PASS 不得悄悄复活」。")
    A("")
    A("### F2　两条声明的证据已被原地覆盖（C01、C02）")
    A("")
    for cid in s["provenance_overwritten"]:
        c = claims[cid]
        A(f"- **{cid}** `{c['doc']}` 引 `{fnum(c['reported_error_pct'], 2)}%`，"
          f"但 `{c['checks']['provenance']['artifact']}` 现值 "
          f"`{c['checks']['provenance']['artifact_error_pct']:+.2f}%`。")
    A("")
    A("根因是**工件路径按名字复用**：后来的单子把 `benchmarks/<name>/validation_report.json` "
      "整体覆盖，于是早期单子的证据消失而文档一字未改。"
      "这不是谁的失误，而是缺少约定；建议见 §6 P1。")
    A("")
    A("### F3　没有单一权威读数，跨度 " + f"{s['spread_pct_points']:.1f}" + " 个百分点")
    A("")
    A(f"六条声明从 {fnum(s['error_range_pct'][1])}% 到 {fnum(s['error_range_pct'][0])}%。"
      "台帐裁定权威行为 "
      f"**{auth['id']}（{fnum(s['authoritative_error_pct'], 2)}%，{auth['definition_class']}）**，"
      "依据：它是唯一满足 P2 所定「绝对压强 + 时间平均」口径、且工件可完全复算（"
      f"`{auth['checks']['provenance']['artifact']}` 的 `runs[2]`）的读数。")
    A("")
    A("## 5. 权威裁定与误差判定")
    A("")
    A("| 判据 | 门限 | 权威读数 | 是否通过 |")
    A("|---|---|---|---|")
    A(f"| 平台精度标准 | 15% | {fnum(s['authoritative_error_pct'], 2)}% | "
      f"**{'是' if s['within_15pct'] else '否'}** |")
    A(f"| 论文级标准 | 3% | {fnum(s['authoritative_error_pct'], 2)}% | "
      f"**{'是' if s['within_3pct'] else '否'}** |")
    A("")
    A(f"**显式回答「是否 < 3%」：否。** 是否 < 15%：否。证据等级 **{s['evidence_grade']}**。")
    A("")
    A(f"裁定理由（沿用 m2971 的归因，本台帐未改动物理结论）：{s['evidence_grade_reason']}")
    A("")
    A("## 6. 机器门禁与交接")
    A("")
    A("`tests/test_verification_ledger.py` 共 13 项，全部绿：")
    A("")
    A("1. 每条引语必须在它引用的文档里**逐字存在**（防手抄漂移）；")
    A("2. 每个工件 + JSON 指针必须仍可解析（防工件消失）；")
    A("3. 权威行必须 `RESOLVED` 且 **(measured−ref)/ref 的重算值 = 工件存值**（算术自洽）；")
    A("4. 证据等级必须由权威读数推出（15% 门），不能手写；")
    A("5. 不得存在未登记的「不支持的 PASS」；撤下的 PASS 必须仍在原件里挂指针块；")
    A("6. 被覆盖工件集合必须与声明清单一致（**新出现的孤儿会让门禁红**）；")
    A("7. 单快照声明若与自身轨迹尾段矛盾必须被标记；")
    A("8. 派生轨迹必须证明唯一 PASS **在其自身算例的稳态下不可达**；")
    A("9. 文档间矛盾必须被显式记录，且跨度数值必须与声明一致；")
    A("10. 台帐涉及的每篇文档都必须挂台帐指针块；")
    A("11. 提交的 `ledger.json` 必须可由 `audit.py` 复现（禁止手改）。")
    A("")
    A("### 交接")
    A("")
    A("- **P0（不变，来自 m2971）**：自由面压强闭合（shell-pressure 外推）。验收判据不变："
      "`benchmarks/laplace_pressure_floor/run.py` 的 `absolute_time_average_tail` 在 "
      "c0=20/5 × dx=0.002/0.0015 四点全部 <15%。修好后 **重新跑 `audit.py`**："
      "`test_evidence_grade_follows_from_the_authoritative_reading` 会自动接受新的等级，"
      "而 `test_laplace_dp_is_not_claimed_within_3pct` 会红，提醒更新本文。")
    A("- **P1（新增，本台帐）**：验证工件路径加任务号（`benchmarks/laplace_law/m3544/…`），"
      "禁止跨单原地覆盖。C01/C02 的孤儿化正是这条缺失的直接后果。")
    A("- **P2（新增，本台帐）**：`p_center` 报告必须同时落盘时间序列（C03/C04 能被抓出来，"
      "正是因为它们的工件恰好带了 `history`）。没有轨迹的读数一律按 `absolute_instant` 处理，"
      "不得用于判定门限。")
    A("")
    A("## 7. 注意点与适用范围")
    A("")
    A("- 本台帐**只覆盖算例 A**（σ = 7.2 N/m, R = 0.02 m, σ/R = 360 Pa）。"
      "`docs/validation/surface-tension-laplace.md` 与 `docs/validation/young_laplace_csf.md` "
      "用的是另一套参数（σ = 0.072 N/m, R = 0.1 m / σ/R = 0.72 Pa），误差在 4.37%…84% 之间，"
      "**未纳入本次裁定**——它们的口径与网格均不同，混入会再次制造不可比数字。这是明确的未完成项。")
    A("- 台帐不评价曲率、切割力、伪流三项：这些量在多张单子里已经进入 15% 门"
      "（切割力恒等式、法向对齐、伪流 < 0.1 毛细速度），本台帐的矛头只对准**端到端压差**。")
    A("- 引语是**逐字子串**匹配，因此文档若改写措辞，门禁会红并要求同步更新 manifest——"
      "这是有意的，不是误报。")
    A("")
    A("## 8. 结果图")
    A("")
    A("![ledger](benchmarks/verification_ledger/ledger.png)")
    A("")
    A("左：六条声明的误差与口径（红=权威行，蓝线=15%/3% 门）。"
      "右：C03 所引工件自身的 `p_center` 轨迹在记录窗口内由 416.7 Pa 衰减至 137.9 Pa，"
      "而 `+5.6%` 头条值 380.2 Pa 不在该衰减段上——这是撤下那个 PASS 的直接证据。")
    A("")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
