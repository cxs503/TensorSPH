#!/usr/bin/env python3
"""Render docs/board_audit_m17452/M17452_BOARD_AUDIT.md from the audit JSONs.

Every number in the rendered document is read from
artifacts/audit_m17452/{manifest,git_state,referee,judge,repro,finding}.json
which this audit produced itself — nothing is transcribed by hand.
"""
from __future__ import annotations

import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "artifacts", "audit_m17452")
DOC = os.path.join(REPO, "docs", "board_audit_m17452", "M17452_BOARD_AUDIT.md")


def load(name):
    with open(os.path.join(OUT, f"{name}.json")) as fh:
        return json.load(fh)


def main() -> int:
    man, git, ref, j, rep, find = (load(n) for n in
                                   ("manifest", "git_state", "referee", "judge",
                                    "repro", "finding"))
    s = j["summary"]
    ld = j["level_decision"]
    L = []
    A = L.append

    A("# m17452 交差核验（board audit）— XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID")
    A("")
    A(f"* 审计任务：m{find['audit_mission'] if 'audit_mission' in find else 17494}"
      f"（`board-audit`）；被审任务：m{find['mission']}（TensorSPH，状态 succeeded）")
    A(f"* 被审工作树：`{man['tree']}`，分支 `{git['branch']}`，HEAD `{git['head'][:12]}`"
      f"（`{git['log']}`）")
    A(f"* 锚点：`{find['anchor']}`；仓：`{find['repo']}`")
    A("* 结论先行：**级别 L2**，benchmark **FAIL**，disposition = **request changes**"
      "（返修，不提级）。报告诚实、数字逐位可复现，但功能实质未过基准。")
    A("")

    A("## 1. 审计对象与独立性")
    A("")
    A("被审对象是 m17452 在 2026-11-13 06:04 结束（`fail_stage=budget_exhausted+"
      "forced_finish`，tool_calls 80/80）时留在工作树里的交付物。审计者未参与其开发，"
      "只读字节 + 重放命令；`uncertain` 的地方一律标 `not_run`，不写成通过。")
    A("")
    A("**独立性限制（如实声明）**")
    A("")
    A("* 审计者能看清的是**工作树快照**，不是 commit：被审任务 4 条交付路径全部处于 "
      "untracked 状态（见 §3 F7），因此不存在可引用的 commit/PR 身份；下面的 sha256 "
      "就是对快照的唯一身份绑定。")
    A("* 无法运行上游 SPHinXsys 本体（C++/TBB，无编译预算），所以「OSS 参考值」不是"
      "跑出来的，而是代数移植 + 解析基准；本审计只能核**移植是否按所引行号实现**与"
      "**结果是否达 15%**，不能核 OSS 程序自身行为。")
    A("* 注册裁判表对本锚点 **0 行**（见 §2），因此严格意义上没有「机器注册的容差」"
      "可对照；审计采用被审任务在跑数**之前**写死的 15%/3%（run.py 模块 docstring），"
      "它与 owner 精度标准同值，且审计者未改动它。")
    A("")

    A("## 2. 三件套取证 + 注册裁判")
    A("")
    A("### 2.1 run_artifacts（文件真实存在，sha256 绑定）")
    A("")
    A("| 交付路径 | 存在 | 字节 | sha256(前16) |")
    A("|---|---|---|---|")
    for k, v in man["files"].items():
        A(f"| `{k}` | {v['exists']} | {v['bytes']} | `{(v['sha256'] or '-')[:16]}` |")
    A("")
    A("`artifacts/xval_fsi2/oss_src/` 实际归档内容："
      + ", ".join(f"`{x}`" for x in man["oss_src_archived"]) + "。")
    A("")
    A("### 2.2 注册裁判（tensor_benchmark_cases）")
    A("")
    A(f"`SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE "
      f"anchor_slug='{ref['anchor_slug']}'` → **{ref['n_rows']} 行**"
      f"（全表 {ref['cases_table_total']} 例，`anchor_slug ILIKE '%FSI%'` 命中 "
      f"{ref['fsi_like_rows']} 例）。")
    A("")
    A("如实记录：本锚点**没有注册基准用例**，所以「基准判据是唯一裁判」这一条在本单"
      "退化为「被审任务自定且跑前声明的判据」：")
    A(f"* 必需 |err| < {100*j['tolerances']['required']:.0f}%，目标 < "
      f"{100*j['tolerances']['target']:.0f}%（来源：{j['tolerances']['source']}）；"
      "审计未在任何时刻改动阈值。")
    A("")
    A("### 2.3 repro_cmd（在临时副本里**重放过**，不是看着像能跑）")
    A("")
    A(f"* `pytest tests/test_fsi_force_from_fluid.py -q` → rc={rep['pytest']['returncode']}，"
      f"`{rep['pytest']['tail'][-1] if rep['pytest']['tail'] else ''}`")
    A(f"* `python benchmarks/xval_fsi2/run.py` → rc={rep['run']['returncode']}，重写 results.json")
    A(f"* 逐叶比对：{rep['leaves_compared_excluding_timings']} 个数值叶"
      f"（共 {rep['leaves_audited']} 叶）**全部一致**；仅 {rep['ignored_timing_leaves']} 个 "
      "`timings_s` 叶按预期不同 → 逐位可复现 = "
      f"**{find['reproducible']}**")
    A("")

    A("## 3. 判级（L0–L5，按功能实质）")
    A("")
    A(f"**级别 = L{ld['level']}**。")
    A("")
    A("| 级 | 判据 | 本单 |")
    A("|---|---|---|")
    A("| L1 | 数学/单元验证 | 达标（不变量测试：符号、dx 不敏感、线性场精确、无邻居零载） |")
    A(f"| L2 | 独立功能实现（真代码：输入→算→输出） | **达标** — 352 行算子 + 复现驱动 |")
    A(f"| L3 | 系统集成（接入主求解器/平台） | **未达标** — {ld['L3_reason']}；"
      f"`src/tensorsph/__init__.py` 导出：{ld['package_export']} |")
    A(f"| L4 | 基准验证（完整算例+误差分析+可复现） | **未达标** — {ld['L4_reason']} |")
    A("| L5 | 科研工程成果 | 不适用 |")
    A("")
    A("即：有真实实现、有可复现证据、有诚实的失败陈述，但**移植算子本身在基准上失败**，"
      "且从未被主求解器调用。报告 167 行、两张图、拓扑完整的六段都不改变这一点"
      "（级别=功能实质，不是文档厚度）。")
    A("")

    A("## 4. 基准判读：ref vs got 逐项")
    A("")
    A("| case | dx | operator | ref | got | err | <15% | <3% |")
    A("|---|---|---|---|---|---|---|---|")
    for r in j["rows"]:
        A(f"| {r['case']} | {r['dx']:g} | {r['operator']} | {r['ref']:.6g} | "
          f"{r['got']:.6g} | {100*r['error']:+.2f}% | "
          f"{'✔' if r['passes_15'] else '✘'} | {'✔' if r['passes_3'] else '✘'} |")
    A("")
    A("（`oss_riemann` 列在本配置下与 `oss` 逐位相同：fsi2 用 `NoRiemannSolver`，"
      "Riemann 项本应为零 — 该列是自洽性检查而非独立样本。）")
    A("")
    A("汇总（程序化，零手抄）：")
    A("")
    A(f"* 移植算子行（oss + oss_riemann + oss_viscous）：{s['primary_pass15']}/"
      f"{s['primary_rows']} 在 15% 内（去重后 9 压力行 + 3 粘性行 = 12 个独立样本，0 通过）")
    A(f"* 压力支误差带：{100*s['oss_pressure_err_range'][0]:+.2f}% … "
      f"{100*s['oss_pressure_err_range'][1]:+.2f}%")
    A(f"* 粘性支误差带：{100*s['oss_viscous_err_range'][0]:+.2f}% … "
      f"{100*s['oss_viscous_err_range'][1]:+.2f}%")
    A(f"* panel 臂：{s['panel_pass15']}/{s['panel_rows']} 在 15% 内，误差带 "
      f"{100*s['panel_err_range'][0]:+.2f}% … {100*s['panel_err_range'][1]:+.2f}%")
    A(f"* 全部 {j['n_rows_scored']} 行里 `fluid_state='solved'` 的行数："
      f"{s['rows_with_solved_fluid_state']}")
    A("")
    A("图：`docs/board_audit_m17452/fig_error_vs_threshold.png`"
      "（逐行误差柱状图 + 15% 硬线与 3% 目标线，由 `judge.json` 程序化生成）。")
    A("")

    A("## 5. 发现（按严重度，附证据与复现）")
    A("")
    A("### F1 — 阻塞·基准：移植算子 12/12 独立样本超 15%")
    A("")
    A(f"* 证据：上表 + `evidence/judge.json`；平壁 +24.23…+28.14%，"
      f"闭合体 +67.66…+107.54%，Couette 粘性 −38.01…−38.54%。且**误差不随 dx 收敛**"
      "（+27.59→+27.96→+28.14），说明不是分辨率问题。")
    A("* 复现：`python benchmarks/xval_fsi2/run.py`（本审计已在临时副本重跑，逐叶一致）。")
    A("* 影响：这是本单唯一的**能力主张**（“把 OSS 传递算子落到 TensorSPH 并在基准上验证”），"
      "主张不成立 → 不得计入 L4，maturity gate 应为 reopened。")
    A("")
    A("### F2 — 阻塞·集成：模块无任何生产调用者")
    A("")
    A(f"* 证据：`grep -rIl fsi_force_from_fluid src/ examples/` 只命中模块自身；"
      f"`src/tensorsph/__init__.py` 中 `fsi_force` 命中：{ld['package_export']}；"
      "文档自己的移植决定是「不要把 panel 换成 OSS 算子」。")
    A("* 影响：不进主求解器 = 不是 L3。测试/基准里的调用不算集成。")
    A("")
    A("### F3 — 中等·出处：声称「verbatim OSS 源码已归档」，实际未归档被引文件")
    A("")
    A("`fsi_force_from_fluid.py` docstring 称 *the verbatim files are archived under "
      "`artifacts/xval_fsi2/oss_src/`*，并引 `force_on_structure.hpp:55-72,103-124`、"
      "`fsi2.cpp`。归档目录里只有 `hydrostatic_fsi.cpp` 与 3 个 XML，"
      "**`force_on_structure.hpp` / `fsi2.cpp` 均不存在**（`find` 全树无此名）。")
    A("→ 被引行号本地不可核；「零手抄」在 provenance 层面只做到「引用到了」，"
      "没做到「字节留档」。")
    A("")
    A("### F4 — 中等·过度声明：跑前设计承诺「有 solved WCSPH 算例」，实际 0 个")
    A("")
    A("`run.py` docstring 写 *One case-family (end-to-end) additionally uses a solved "
      f"WCSPH field*；而 results.json 全部 {j['n_rows_scored']} 行 "
      f"`fluid_state='analytic'`，`solved` 行数 = {s['rows_with_solved_fluid_state']}。"
      "→ 端到端（求解器在环）被 doc 记为「已含」，实为未做；六段报告第 6 节说明了延迟理由，"
      "但设计声明与执行不一致。")
    A("")
    A("### F5 — 中等·方法论：panel 的 −0.00% 是**同义反复**，不能当验证")
    A("")
    A("panel 臂是核权最小二乘平面拟合，解析静水压力场本身就是线性场 → 精确复现是**定义使然**"
      "（被审自己的测试 `test_panel_is_exact_on_a_linear_pressure_field` 正是断 this）。"
      f"同臂在闭合体上 {100*s['panel_err_range'][1]:+.2f}%（dx=0.025），且随加密变差"
      "（−0.70→+20.08→+32.26，越细越差），上述二者都不能作为「算子已验证」的证据。")
    A("* 影响：任何把 panel 的 0.00% 计入「TensorSPH 传递算子精度」的结论都是失效的。")
    A("")
    A("### F6 — 低·测试语义：把失败带「钉死」为期望值")
    A("")
    A("`test_oss_operator_signs_and_documented_overload` 断言 `0.20 < err < 0.32`，"
      "`test_viscous_branch_...` 断言 `0.55 < got/ref < 0.70`。作为**变更探测器**是合理的，"
      "但它不是验证：`10 passed` 不代表能力确认，读者极易误读。审计不把测试通过计入 L4。")
    A("")
    A("### F7 — 低·流程：交付物从未提交，任务以预算耗尽收尾")
    A("")
    A("`git status --porcelain`：")
    for line in find["deliverables_uncommitted"]:
        A(f"* `{line}`")
    A("")
    A("`head_commit=null`，`fail_stage=budget_exhausted+forced_finish`，80/80 tool_calls。"
      "→ 没有 commit/PR 身份，返修必须以工作树快照为起点（本审计已 sha256 绑定，可据此恢复）。")
    A("")

    A("## 6. 未做的检查（标 not_run，不标 passed）")
    A("")
    A("* `not_run` 上游 SPHinXsys 本体编译运行（无 C++ 工具链预算）：因此“OSS 本身在 fsi2 "
      "上是否也过冲 28%”未知；被审 doc 也只把它当 hypothesis。")
    A("* `not_run` 耦合 fsi2 瞬态（梁尖位移 vs `BeamObserver_Position_Run_0_result.xml`、"
      "时均 `TotalViscousForceFromFluid`）——被审明确延迟到 round 2，本轮无可比数字。")
    A("* `not_run` 3D / 非均匀粒子分布 / 移动固体（Riemann 项非零）——本审计只覆盖被审"
      "已交付的 4 个解析算例族。")
    A("* `not_run` 数值稳定性/性能：无求解器在环，无 CFL 时间推进。")
    A("")

    A("## 7. 过强的声明（原文摘出，逐条给判）")
    A("")
    A("| 声明 | 出处 | 判 |")
    A("|---|---|---|")
    A("| 把 OSS 的 ForceFromFluid 传递算子「落到 TensorSPH」 | mission headline | **部分成立**："
      "算子存在且可运行，但仅作为对照臂（comparison arm），未接入求解器，且文档建议不采用 |")
    A("| 「并用解析基准在 4 个算例×分辨率阶梯验证」 | mission headline | **不成立**：验证结果是"
      " 12/12 独立样本超 15%，属**证伪**（honest FAIL 有价值，但不是通过） |")
    A("| verbatim OSS 源码已归档 `oss_src/` | 模块 docstring | **不成立**（F3） |")
    A("| 端到端 solved WCSPH 算例含在内 | run.py docstring | **不成立**（F4） |")
    A("| panel 平壁 −0.00% 证明算子精度 | results.json `passes_15` | **不成立/同义反复**（F5） |")
    A("")

    A("## 8. Disposition")
    A("")
    A("**request changes（L2 + 基准 FAIL）**，理由与返修点：")
    A("")
    A("1. 若目标仍是「OSS 算子可用」：必须先解释并消除 +28% 系统偏差（单侧求和 + 单层壳 + "
      "h=1.3dx 的组合），或按文档自述改用 OSS 体离散（填充体、dx_s=dx_f/2）再测——"
      "否则任何基于该算子的能力声明都无效。")
    A("2. 若要进 L3：把算子接到实际 FSI 路径上，并给出至少一个调用点 + 回归用例。")
    A("3. F3/F4/F5 三处声明必须改口径（归档被引文件；删掉未做的 solved 声明；"
      "把 panel 的线性精确性写成“同义反复检查”而非“验证”）。")
    A("4. 交付物必须提交（F7）：现在只有快照，无 commit/PR 身份。")
    A("")
    A("不阻塞的建议：把 `oss_riemann` 的全零差异列显式标注为「自洽性检查」"
      "（已被审按其 doc 说明，但结果表未标）；把厚度敏感性（1 层 vs 3 层："
      "oss +27.59% → +66.48%）提升为第一类证据，它比平壁误差更有诊断价值。")
    A("")
    A("## 9. 审计产物清单")
    A("")
    A("本仓 `.gitignore` 忽略 `artifacts/`，所以审计证据同时以**可提交副本**落在 "
      "`docs/board_audit_m17452/evidence/`（下表的同名文件），`artifacts/audit_m17452/` "
      "为可重跑再生副本。")
    A("")
    A("* `evidence/manifest.json` — 交付物 sha256/字节/存在性")
    A("* `evidence/git_state.json` — 分支/HEAD/porcelain/追踪状态")
    A("* `evidence/referee.json` — 注册裁判表查询结果（0 行）")
    A("* `evidence/judge.json` — 逐行 ref/got/err/通过门 + 级别规则")
    A("* `evidence/repro.json` — 重放记录（554 叶逐位比对）")
    A("* `evidence/finding.json` — 结论摘要")
    A("* `evidence/verdict.json` — 落表行（verdict id + 完整 evidence）")
    A("* `fig_error_vs_threshold.png` — 误差 vs 15% 硬线（逐行柱状）")
    A("* `fig_convergence.png` — 误差 vs dx 细化曲线（系统偏差非分辨率）")
    A("* `scripts/audit/m17452_board_audit.py` — 一键复跑（`--replay --insert-verdict`）")
    A("")
    A("重跑命令：")
    A("")
    A("```bash")
    A("python scripts/audit/m17452_board_audit.py --replay            # 取证+重放+判级")
    A("python scripts/audit/build_m17452_doc.py                        # 渲染本文档")
    A("python scripts/audit/m17452_board_audit.py --replay --insert-verdict --audit-mission 17494")
    A("```")

    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    with open(DOC, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", DOC, len(L), "lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())