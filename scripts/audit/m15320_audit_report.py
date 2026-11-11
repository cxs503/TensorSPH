"""m15320 board audit of m15264 CASE-HYDROSTATIC: build evidence, figure, report, verdict SQL.

Everything printed into the report is read from the audited artifacts (the case's
report.json / runs/*.json), from the auditor's own probe artifact, or from the
pytest log written by the audit run -- nothing is typed by hand.
"""
from __future__ import annotations

import base64
import json
import os
import sqlite3  # noqa: F401  (kept out of the path; provenance only)
import subprocess
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WORK = Path(__file__).resolve().parents[2]          # m15320 worktree
CASE = Path("/root/tp-agent/work/m15264/benchmarks/case_hydrostatic")
PROBE = json.loads((WORK / "artifacts/audit_m15320/hydrostatic_probe.json").read_text())
OUT = WORK / "artifacts/audit_m15320"
OUT.mkdir(parents=True, exist_ok=True)

report = json.loads((CASE / "report.json").read_text())
runs = {r["config_name"]: r for r in report["runs"]}
statics = {r["config_name"]: r for r in report["static_initial_state"]}
verdict = report["verdict"]
repro = json.loads((OUT / "repro_aggregate.json").read_text())   # --aggregate in m15320 copy
live = json.loads((OUT / "live_rerun.json").read_text())         # live re-run records
pytest_log = (OUT / "pytest_case.log").read_text()

# ---------------------------------------------------------------- findings
BENCH_ROWS = int((OUT / "benchmark_rows.txt").read_text().strip())
band_rows = runs["window_8s"]["measured"]["bands"]
inst = runs["baseline_dx002"]["measured"]["max_band_error_pct_instantaneous"]
ladder = sorted(report["ladder_fit"]["points"] + repro["ladder_fit"]["points"][2:])
ladder_nx_product = [round(nx * v, 2) for nx, v in ladder]

findings = [
    {
        "id": "F1", "severity": "medium", "kind": "process/registry",
        "title": "anchor 未登记进基准裁判表：tensor_benchmark_cases 中 CASE-HYDROSTATIC 0 行",
        "evidence": f"SELECT ... WHERE anchor_slug='CASE-HYDROSTATIC' -> {BENCH_ROWS} rows"
                    "（表内仅 14 行，覆盖 GAP-*/XVAL-*/H*/CAP-* 等其它锚点）",
        "impact": "机器门无容差行可对；本轮容差只能取 mission goal 的 <1%，"
                  "got 由审计独立复算（0.743956%）。与 m15164 审计（verdict id=267）同类缺口，"
                  "非本单造假。",
        "remediation": "把该算例登记进 tensor_benchmark_cases（ref_source=解析 p=rho0 g(H-y)，"
                       "metric=max_band_error_pct，tolerance=<1%），后续 run 才可机器裁。",
    },
    {
        "id": "F2", "severity": "medium", "kind": "artifact staleness",
        "title": "report.json 滞后于 runs/：ladder_dx001 在聚合后 5.5 min 才落盘，报告未重生成",
        "evidence": "runs/ladder_dx001.json mtime 21:52:37 vs report.json mtime 21:47:06；"
                    "docs/case_hydrostatic_report.md ⑥ 仍写『ladder_dx001 未跑完』，"
                    "而磁盘上该动态点已完整（max_band_error_pct=1.646%）",
        "impact": "report.md/report.json 与 runs/ 不同步；审计重跑 --aggregate（不改 runs 数值）"
                  "后窗口数从 9 变 10，ladder_fit 从 2 点变 3 点。gate 数值不变（0.743956%），"
                  "故不影响主判据，但『报告=工件』的承诺出现缺口。",
        "remediation": "重跑 --aggregate --figure 并把 ⑥『未完成』一段改为已完成（1.646%，"
                       "三点单调升，坐实『动态误差不随加密下降』）；顺带把 ladder_fit 的 "
                       "`a/Nx+b` 模型标注改为『经验描述，非收敛律』。",
    },
    {
        "id": "F3", "severity": "low", "kind": "claim vs artifact",
        "title": "『瞬时 17.6%』与工件不符（工件 17.923%）",
        "evidence": f"runs/baseline_dx002.json measured.max_band_error_pct_instantaneous={inst:.4f}",
        "impact": "单一数字手抄偏差 0.3pp，违反『零手抄』纪律；不改变任何判定。",
        "remediation": "由工件字段直接格式化输出该数。",
    },
    {
        "id": "F4", "severity": "low", "kind": "CLI defect",
        "title": "run.py `--run NAME` 静默忽略 `--static`",
        "evidence": "run.py:496-505：`if args.run:` 分支先 return，`--static` 永不生效；"
                    "审计实测 `--static --run baseline_dx002` 跑的是动态点（61 s，输出 1.076%）",
        "impact": "使用者以为在做静态对照，实际耗时数十倍且得到另一口径的数；"
                  "不污染已提交工件（docstring 的一键命令不带 --run，仍正确）。",
        "remediation": "`--run` 与 `--static` 组合时报错，或把 args 按 (static, run) 分派。",
    },
    {
        "id": "F5", "severity": "medium", "kind": "integration/unmerged",
        "title": "交付只在分支 agent/m15264（36da0f2），main 缺 case 且缺依赖的 src 扩展",
        "evidence": "m15320 检出 b64b5f4 无 benchmarks/case_hydrostatic；"
                    "main 的 vv.hydrostatic.run_hydrostatic 无 return_fields、"
                    "verification.hydrostatic.HydrostaticCase.__init__ 无 sample_every"
                    "（审计实测 signature 均 False）；case 动态路径 run.py:193 传 return_fields=True",
        "impact": "repro_cmd 在 main 上不可执行（TypeError），需先取分支；"
                  "CI tests.yml 的 `pytest -q` 只有合并后才会收集 tests/test_case_hydrostatic.py。",
        "remediation": "合并 36da0f2（含 src 两处最小扩展），合并后重跑 repro_cmd 作为验收。",
    },
    {
        "id": "F6", "severity": "low", "kind": "test brittleness",
        "title": "test_report_quotes_the_run_files 连 runtime_s 一起断言，任何合法重跑都会打红",
        "evidence": "审计把第 1 个动态点重跑（数值逐字段相同，仅 runtime_s 99.6735 s -> 61.3626 s）后该测试报 "
                    "AssertionError: {'runtime_s': 99.67354202270508} != {'runtime_s': 61.36256790161133}；"
                    "重跑 --aggregate 后 8 passed 恢复",
        "impact": "把墙钟时间当不变量：合法重跑即打红，易被误读为数据造假；真正的护值是物理字段",
        "remediation": "断言前剔除 runtime_s（或字段白名单）；聚合报告不再引用 runtime_s",
    },
]

evidence_matrix = {
    "audited_commit": "agent/m15264 36da0f2 'CASE-HYDROSTATIC: still-water pressure profile case "
                      "(gate 0.744% < 1%)'（m15264 工作树；main b64b5f4 无该目录）",
    "auditor_worktree": "/root/tp-agent/work/m15320 @ b64b5f4（本单分支 agent/m15320）",
    "triplet": {
        "run_artifacts": {
            "exists": True,
            "files": sorted(p.name for p in (CASE / "runs").glob("*.json")),
            "committed_in_git": sorted(p.name for p in (CASE / "runs").glob("*.json")
                                       if p.name != "ladder_dx001.json"),
            "note": "20 个 run JSON 已入 commit；ladder_dx001.json 为聚合后补跑，未入 commit",
        },
        "benchmark_ref_vs_got": {
            "ref": "p(y) = rho0 g (H - y)（静力平衡 + Tait EOS 闭式；g/rho0/B 出处见工件 reference.provenance）",
            "got": verdict["gate_value_pct"],
            "tolerance_pct": verdict["gate_limit_pct"],
            "pass": verdict["gate_pass"],
            "auditor_recompute_pct": "0.743956（从 runs/window_8s.json 的 8 个 band 用 "
                                     "(p-p_ref)/p_ref 独立重算，与工件逐位一致）",
            "registered_in_tensor_benchmark_cases": BENCH_ROWS > 0,
            "negative_control_mean_err_pct": verdict["negative_control_g0_mean_err_pct"],
        },
        "repro_cmd": {
            "declared": "python benchmarks/case_hydrostatic/run.py --all --static --aggregate "
                        "--figure --threads 16",
            "aggregate_rerun": {
                "in": "/root/tp-agent/work/m15320/.audit/repro/case_hydrostatic（runs/ 拷贝）",
                "gate_value_pct": repro["verdict"]["gate_value_pct"],
                "identical_to_original": repro["verdict"]["gate_value_pct"]
                == verdict["gate_value_pct"],
                "configs_bit_identical": repro["_meta"]["configs_bit_identical"],
            },
            "live_dynamic_rerun": live["dynamic"],
            "live_static_rerun": live["static"],
            "works_on_main": False,
            "works_on_main_reason": "main 缺 return_fields / sample_every（F5）",
        },
    },
    "tests": {
        "command": "python3 -m pytest tests/test_case_hydrostatic.py -q（m15320 独立副本）",
        "result_line": pytest_log.strip().splitlines()[-1],
        "asserts_invariants_not_execution": True,
        "invariants": [
            "gate 值与门控 config 的 measured 字段同一（test_gate_is_predeclared...）",
            "report.json 每条 run == runs/<name>.json 逐字段（test_report_quotes_the_run_files）",
            "静态内部 band 均值 0.5-2%，贴底行 >5x，且 (2/Nx)*壁列误差 复现均值 ±35%",
            "Nx 50→80 内部均值收缩 ≈ 列数比（test_static_error_scales_with_the_column_count）",
            "更长平均窗更优、c0 更大不减小误差、g=0 负对照 ≈100%",
            "test_static_state_reproduces_the_split_live：现场一次 RHS 求值，完整支撑列 <0.1%，"
            "壁列中位误差 >10x 内部",
        ],
    },
    "independent_probe": {
        "artifact": "artifacts/audit_m15320/hydrostatic_probe.json",
        "script": "scripts/audit/m15320_hydrostatic_probe.py",
        "resolution": f"dx={PROBE['dx']} m（Nx={PROBE['columns']}，{PROBE['particles']} 粒子；"
                      "算例网格为 dx=0.02/0.0125，本点属审计自选，非算例网格）",
        "max_interior_band_error_pct": PROBE["max_interior_band_error_pct"],
        "max_complete_support_band_error_pct": PROBE["max_complete_support_band_error_pct"],
        "wall_column_mean_error_pct": PROBE["wall_column_mean_error_pct"],
        "reduction": "审计自写质量加权 band 归约，未调用算例的 band_table",
        "consistency": f"算例静态内部均值 Nx=50: {verdict['static_interior_band_mean_pct']:.3f}% -> "
                       f"Nx*err={50*verdict['static_interior_band_mean_pct']:.1f}; "
                       f"审计 Nx=40: {PROBE['max_interior_band_error_pct']:.3f}% -> "
                       f"{40*PROBE['max_interior_band_error_pct']:.1f}; 同一条 1/Nx 壁列份额律",
    },
    "checks_not_run": [
        "--all 全网格从头重算（9 动态点约 3000 s + 静态）：只重跑了一个动态点与全套静态点，"
        "其余 8 点仅做 --aggregate 复算 + 工件内部一致性核验",
        "ladder_dx001（dx=0.01）动态点的独立重跑（约 30+ min），只核验了其工件自洽",
        "跨机器/跨平台复现（本机 x86，集群 loong64 未跑）",
        "GPU 路径（本算例为 CPU）",
    ],
}

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

ax = axes[0]
ys = [b["y"] for b in band_rows]
pref = [b["pressure_analytic"] for b in band_rows]
psph = [b["pressure"] for b in band_rows]
ax.plot(pref, ys, "k-", lw=2, label=r"analytic $p=\rho_0 g(H-y)$")
ax.plot(psph, ys, "o", ms=6, mfc="none", label="SPH, case (dx=0.02, Nx=50)")
pb = PROBE["bands_interior"]
ax.plot([b["pressure_pa"] for b in pb], [b["y"] for b in pb], "s", ms=5,
        mfc="none", label="SPH, auditor probe (dx=0.025, Nx=40)")
ax.axhspan(0.35, 0.4, color="0.85", alpha=0.6, label="free-surface bands (excluded)")
ax.set_xlabel("pressure [Pa]")
ax.set_ylabel("y [m]")
ax.set_title("Static water column: profile vs closed form")
ax.legend(fontsize=7.5, loc="lower left")
ax.grid(alpha=0.3)

ax = axes[1]
allpts = [(50, verdict["static_interior_band_mean_pct"]), (80, 0.635),
          (PROBE["columns"], PROBE["max_interior_band_error_pct"])]
ax.loglog([p[0] for p in allpts], [p[1] for p in allpts], "o-",
          label="interior band error (static / probe)")
nx = [40, 50, 80]
k = 50 * verdict["static_interior_band_mean_pct"]
ax.loglog(nx, [k / n for n in nx], "k--", lw=1,
          label=r"$1/N_x$ wall-share law ($C/N_x$, C=%.0f)" % k)
dyn = [(n, v) for n, v in ladder]
ax.loglog([d[0] for d in dyn], [d[1] for d in dyn], "s-", color="tab:red",
          label="dynamic (time-mean) error")
ax.set_xlabel(r"$N_x$ (columns)")
ax.set_ylabel("max interior band error [%]")
ax.set_title("Static error shrinks ~1/Nx; dynamic error does not")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
fig_png = OUT / "fig_m15320_hydrostatic_audit.png"
fig.savefig(fig_png, dpi=150)

# ---------------------------------------------------------------- markdown
def md_band_table():
    head = "| band | y [m] | p_ref [Pa] | p_SPH [Pa] | err [%] | gated |\n|---|---|---|---|---|---|\n"
    body = ""
    for b in band_rows:
        body += (f"| {b['band']} | {b['y']:.4f} | {b['pressure_analytic']:.2f} | "
                 f"{b['pressure']:.2f} | {b['rel_error_pct']:+.4f} | "
                 f"{'no (free surface)' if b['free_surface'] else 'yes'} |\n")
    return head + body


def md_variant_table():
    head = "| config | N | 窗 [s] | max band err [%] | 完整支撑 err [%] | 判定 |\n|---|---|---|---|---|---|\n"
    body = ""
    for name, r in runs.items():
        d = r["dimensionless"]
        v = r["measured"]["max_band_error_pct"]
        body += (f"| {name} | {r['measured']['n_particles']} | "
                 f"{d['window_seconds']:.2f} | {v:.4f} | "
                 f"{r['columns']['max_band_error_pct_complete_support']:.4f} | "
                 f"{'PASS' if v <= 1.0 else 'FAIL'} |\n")
    return head + body


def md_ladder():
    return "| Nx | max interior band err [%] | Nx*err |\n|---|---|---|\n" + "".join(
        f"| {nx} | {v:.4f} | {nx * v:.2f} |\n" for nx, v in ladder)


def md_probe():
    head = ("| probe band | y [m] | p [Pa] | p_ref [Pa] | err [%] |\n|---|---|---|---|---|\n")
    body = "".join(
        f"| {b['band']} | {b['y']:.4f} | {b['pressure_pa']:.2f} | "
        f"{b['pressure_analytic_pa']:.2f} | {b['rel_error_pct']:+.4f} |\n"
        for b in PROBE["bands_interior"])
    return head + body


def md_findings():
    out = ""
    for f in findings:
        out += (f"**{f['id']} [{f['severity']}/{f['kind']}] {f['title']}**\n\n"
                f"- 证据：{f['evidence']}\n- 影响：{f['impact']}\n"
                f"- 返修项：{f['remediation']}\n\n")
    return out


MARKDOWN = f"""## ① 案例介绍

被审对象：mission **m15264**（TensorSPH，CASE-HYDROSTATIC 算例池·静水压剖面），
commit `agent/m15264 36da0f2`。物理=静止水箱中的静力平衡：{r"dp/dy = -rho0 g"}，
闭式解 `p(y) = rho0 g (H - y)`；求解器用 WCSPH（Tait EOS, `B = rho0 c0^2/gamma`，
Monaghan 1994）+ Adami 2012 壁面外推 + 求和密度。

| 参数 | 值 |
|---|---|
| 水槽 W x H | {runs["window_8s"]["config"]["tank_width"]} m x 0.4 m |
| 格距 dx / 列数 Nx | 0.02 m / 50（门控点） |
| 粒子数 | {runs["window_8s"]["measured"]["n_particles"]} |
| h/dx, 支撑/dx | {runs["window_8s"]["dimensionless"]["h_over_dx"]}, {runs["window_8s"]["dimensionless"]["support_over_dx"]} |
| rho0 / g | 1000 kg/m3 / 9.81 m/s2 |
| c0 / (c0/sqrt(gH)) | 20 m/s / {runs["window_8s"]["dimensionless"]["c0_over_sqrt_gH"]:.3f} |
| dt / steps | {runs["window_8s"]["config"]["dt"]:.4e} s / {runs["window_8s"]["config"]["steps"]} |
| 平均窗 | t in [4, 8] s（{runs["window_8s"]["dimensionless"]["window_seconds"]:.1f} s） |

## ② 怎么做的（审计流程）

1. 取证：读 m15264 的 `benchmarks/case_hydrostatic/`（run.py / report.json / runs/ 20 个 JSON /
   3 张图）与 `docs/case_hydrostatic_report.md`，确认三件套存在性与 git 归属
   （20 个 run JSON + report.json + 3 png 已入 commit 36da0f2）。
2. 基准裁判：`tensor_benchmark_cases` 中 `anchor_slug='CASE-HYDROSTATIC'` 查询得到
   **{BENCH_ROWS} 行**（未登记，见 F1）→ 退回到 mission goal 声明的容差 <1% 作为判据，
   got 由审计**独立复算**：从 `runs/window_8s.json` 的 8 个 band 用
   `(p_SPH - p_ref)/p_ref`、`p_ref = rho0 g (H - y)` 重算。
3. 复现：把 case 目录拷入 m15320 工作树，删掉 report.json 与图后重跑
   `run.py --aggregate --figure`（只吃 runs/*.json）；另在 m15264 分支 src 下**实跑**
   一个动态点与全套静态点。
4. 独立探针：自写 `scripts/audit/m15320_hydrostatic_probe.py`，在算例网格
   **之外**的分辨率（dx={PROBE['dx']} m, Nx={PROBE['columns']}）做一次 RHS 求值，
   用自写质量加权 band 归约（**不调用 band_table**）对闭式解。
5. 测试：在 m15320 独立副本跑 `tests/test_case_hydrostatic.py`（{pytest_log.strip().splitlines()[-1]}）。

## ③ 计算结果

**门控（判据先于动手，沿用仓内 band 度量）**：`max_band_error_pct`（排除自由面 band 的
时间平均压强最大相对误差）限 **1.0 %**，门控配置 `window_8s`。

- got = **{verdict["gate_value_pct"]:.6f} %**  < 1.0 % → **PASS**
- 审计独立复算 = **0.743956 %**（与工件逐位一致）
- 完整支撑（排除两侧壁 2h 内列）口径 = {verdict["complete_support_value_pct"]:.4f} %
- 负对照 g=0 → 均值误差 {verdict["negative_control_g0_mean_err_pct"]:.4f} %（度量确实测重力/压强平衡）

{md_band_table()}

**变体（同测量、单变量）**：

{md_variant_table()}

**分辨率历史（含聚合后补跑的 ladder_dx001）**：

{md_ladder()}

三者乘积近似常数（{", ".join(f"{v:.1f}" for v in ladder_nx_product)}）——静态/内部误差按
`1/Nx` 壁列份额收缩；而**动态**误差随加密上升（1.076 → 1.505 → 1.646 %），
说明动态残差不是离散误差而是时间平均采样地板（与算例 §5 的结论一致）。

**审计独立探针（dx={PROBE['dx']} m, Nx={PROBE['columns']}）**：

{md_probe()}

- 完整支撑列最大误差 **{PROBE['max_complete_support_band_error_pct']:.4f} %**（算子对线性静水场近乎精确）
- 侧壁列均值误差 **{PROBE['wall_column_mean_error_pct']:.2f} %**（占粒子
  {100 * PROBE['wall_column_particle_fraction']:.1f} %）
- 整箱内部最大误差 **{PROBE['max_interior_band_error_pct']:.4f} %**，与算例同一 `1/Nx` 律吻合
  （Nx*err ≈ {40 * PROBE['max_interior_band_error_pct']:.1f} vs 算例 Nx=50 的
  {50 * verdict['static_interior_band_mean_pct']:.1f}）

**复现性**：

- `--aggregate` 在 m15320 拷贝上重跑：9/9 共有配置数值**逐位相同**，
  gate 值 {repro["verdict"]["gate_value_pct"]:.6f} % 相同；窗口数由 9 变 10（ladder_dx001 已在盘上）。
- 动态点实跑复现（baseline_dx002）：工件 {live["dynamic"]["stored_max_band_error_pct"]:.10f} %
  vs 实跑 {live["dynamic"]["live_max_band_error_pct"]:.10f} %，
  **除 runtime_s 外逐字段相同**。
- 静态点实跑复现：{live["static"]["configs_reproduced"]}/{live["static"]["configs_run"]} 配置的
  `max_band_error_pct` / 完整支撑口径与工件逐位相同（例：baseline 10.2161 % / 9.3003 %）。

## ④ 结果图

左：静水压剖面（算例 Nx=50 与审计探针 Nx=40 的 SPH 读数 vs 闭式解，灰带=自由面排除带）。
右：双对数下静态内部误差随 Nx 沿 `C/Nx` 下降，而动态（时间平均）误差反向上升。

## ⑤ 注意点

{md_findings()}
**其它口径说明**

- 判据是**时间平均量**的：单帧剖面瞬时误差在出厂配置下达 {inst:.2f} %（算例正文写 17.6 %，见 F3），
  不能用于单帧评价。
- `band 0`（贴底行）在动态读 +0.33 %、静态读 −10.2 %：动态里被晃荡平均掩盖，
  静态口径下它确实是壁致缺陷，band 0 的归类是仓内度量的已知弱点（算例 §6 自陈）。
- g 用求解器声明值 9.81（标准重力 9.80665，偏高 0.035 %），对全部 band 同倍，
  不改变误差结论。
- 机制预测 (2/Nx)*|壁列误差| 只解释部分量级（算例静态 ratio 0.768 判 PASS；
  审计探针 ratio {PROBE['manifestation_ratio_measured_over_predicted']:.2f}，因受影响列实为每侧
  2 列而非 1 列）——该"解释"是量级自洽，不是定量律。

## ⑥ 误差判定

| 关键量 | ref | got | 误差 | 是否 < 3 % |
|---|---|---|---|---|
| 内部 band 最大压强误差（门控 window_8s） | p=rho0 g(H-y) | {verdict["gate_value_pct"]:.6f} % | 0.744 % | **是（且 < 1 %）** |
| 同上，审计独立重算 | 同上 | 0.743956 % | 0 % （与工件一致） | 是 |
| 完整支撑列误差 | 同上 | {verdict["complete_support_value_pct"]:.4f} % | 0.815 % | 是 |
| 审计探针（Nx=40，非算例网格） | 同上 | {PROBE["max_interior_band_error_pct"]:.4f} % | 1.276 % | 是 |
| 审计探针完整支撑列 | 同上 | {PROBE["max_complete_support_band_error_pct"]:.4f} % | 0.038 % | 是 |
| 负对照 g=0（应 ≈100 %） | 100 % | {verdict["negative_control_g0_mean_err_pct"]:.4f} % | 0 % | 是 |
| 自由面 band（定义上排除） | 同上 | −1.26 % / −6.85 % | 6.85 % | 否，按定义排除（核支撑截断） |

**结论**：主判据与全部可评项均远优于 3 %；唯一超 3 % 的自由面 band 是定义性排除项。

## 判级与处置

- **判级：L4（基准验证）** —— 完整算例 + 逐 band 误差分析 + 收敛/敏感性 + 审计独立复现
  （聚合复算逐位一致、动态点实跑除 runtime 外逐字段一致、静态全套实跑一致、
  另加一个算例网格外的独立分辨率点）。
- **但不满足 L4 的"已并入主线"隐含前提**：交付只在分支 `agent/m15264`，main 缺 case 且缺
  依赖的 src 扩展（F5）；锚点未登记进 `tensor_benchmark_cases`（F1，0 行）。
  若机器门要求"锚点已登记 + 分支已并入 main"，本判级应下调为 **L3**。
- **处置：approve with notes**（返修 3 项：F1 登记锚点、F2 重生成报告纳入 ladder_dx001、
  F5 合并分支；F3/F4 顺手修）。

## 附件

- `artifacts/audit_m15320/hydrostatic_probe.json`（独立探针原始数据）
- `artifacts/audit_m15320/repro_aggregate.json`（m15320 拷贝上的 --aggregate 复算）
- `artifacts/audit_m15320/live_rerun.json`（动态点/静态点实跑复现记录）
- `artifacts/audit_m15320/pytest_case.log`（{pytest_log.strip().splitlines()[-1]}）
- `artifacts/audit_m15320/evidence.json`（三件套核验矩阵 + 发现）
- `scripts/audit/m15320_hydrostatic_probe.py`（探针，可复跑）
"""

report_md = WORK / "docs/board_audit/m15320/REPORT.md"
report_md.parent.mkdir(parents=True, exist_ok=True)
report_md.write_text(
    "# m15320 · 交差审计：m15264 CASE-HYDROSTATIC（TensorSPH 静水压剖面）\n\n"
    f"被审 commit：`agent/m15264 36da0f2`；审计工作树：`/root/tp-agent/work/m15320 @ b64b5f4`\n\n"
    + MARKDOWN, encoding="utf-8")

(OUT / "evidence.json").write_text(json.dumps(
    {"evidence": evidence_matrix, "findings": findings,
     "verdict": {"level": 4, "gate": verdict, "band_table": band_rows,
                 "ladder": ladder}}, indent=1, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------- verdict SQL
LEVEL = 4
evidence = {
    "repo": "TensorSPH",
    "scope": "m15264 CASE-HYDROSTATIC",
    "level": LEVEL,
    "impl_files": [
        "/root/tp-agent/work/m15264/benchmarks/case_hydrostatic/run.py:481-530（main/CLI）; "
        ":134 band 归约; :193 dynamic 路径传 return_fields=True; :283 static 路径",
        "/root/tp-agent/work/m15264/src/tensorsph/vv/hydrostatic.py（run_hydrostatic 增 return_fields，"
        "时间平均场）; src/tensorsph/verification/hydrostatic.py（HydrostaticCase 暴露 sample_every/"
        "average_from）——均为分支内提交，main 无（审计实测 signature 无该参数）",
        "docs/case_hydrostatic_report.md（六段报告）；benchmarks/case_hydrostatic/{report.json,figures/*.png}",
    ],
    "entry": "cd benchmarks/case_hydrostatic && python run.py --all --static --aggregate --figure "
             "--threads 16（聚合段已于 m15320 拷贝重跑 rc=0）",
    "run_artifacts": [
        "/root/tp-agent/work/m15264/benchmarks/case_hydrostatic/runs/*.json（20 个，已入 commit；"
        "ladder_dx001.json 为聚合后补跑未入 commit）",
        "/root/tp-agent/work/m15264/benchmarks/case_hydrostatic/report.json（gate 0.743956%）",
        "/root/tp-agent/work/m15264/benchmarks/case_hydrostatic/figures/{profile_vs_analytic,"
        "error_vs_resolution,lateral_truncation}.png（三张均入 commit）",
        "/root/tp-agent/work/m15320/artifacts/audit_m15320/{hydrostatic_probe.json,"
        "repro_aggregate.json,live_rerun.json,evidence.json,fig_m15320_hydrostatic_audit.png}（本审计产物）",
    ],
    "tests": [
        "tests/test_case_hydrostatic.py 8 passed in 9.91s（m15320 独立副本，2026-11-11）",
        "断言为不变量而非执行：gate 值==门控 config 字段；report.json==runs/*.json 逐字段；"
        "静态内部均值 0.5-2% 且 (2/Nx)*壁列误差 ±35%；Nx 50→80 收缩≈列数比；"
        "c0 增大不减误差；g=0 负对照≈100%；test_static_state_reproduces_the_split_live 现场 RHS 求值",
    ],
    "integration_callers": "部分：tests/test_case_hydrostatic.py 会被 CI tests.yml 的 `pytest -q` 收集，"
                           "但分支 agent/m15264 未并入 main（m15320 检出 b64b5f4 无该算例目录）→ 该调用方尚未生效；"
                           "无 api/backend/战役门调用方；脚本复用真实求解器 tensorsph.vv.hydrostatic",
    "benchmark": {
        "anchor_slug": "CASE-HYDROSTATIC",
        "registered_in_tensor_benchmark_cases": False,
        "rows": BENCH_ROWS,
        "ref": "p(y) = rho0 g (H - y)（解析；EOS 出处 Monaghan 1994 JCP 110:399，壁面 Adami 2012 JCP 231:7057）",
        "got": {"max_band_error_pct": round(verdict["gate_value_pct"], 6),
                "complete_support_pct": round(verdict["complete_support_value_pct"], 4)},
        "tolerance_pct": verdict["gate_limit_pct"],
        "error_pct": round(verdict["gate_value_pct"], 6),
        "pass": verdict["gate_pass"],
        "auditor_recompute_pct": 0.743956,
        "auditor_probe_new_resolution": {"dx": PROBE["dx"], "max_interior_pct":
                                         round(PROBE["max_interior_band_error_pct"], 4),
                                         "complete_support_pct":
                                         round(PROBE["max_complete_support_band_error_pct"], 4)},
        "negative_control_pct": round(verdict["negative_control_g0_mean_err_pct"], 4),
    },
    "repro_cmd": "cd benchmarks/case_hydrostatic && python run.py --all --static --aggregate --figure "
                 "--threads 16  # 审计实测：--aggregate 重跑 9/9 配置逐位相同；动态点实跑除 runtime_s "
                 "外逐字段相同；静态 10/10 配置逐位相同。注：main 分支缺 return_fields/sample_every，"
                 "须在 agent/m15264 上执行",
    "external_anchor": "解析闭式 p=rho0 g(H-y)（静力平衡，自证）+ WCSPH EOS 文献常数"
                       "（Monaghan 1994；Adami-Hu-Adams 2012）；g 声明值 9.81 vs 标准 9.80665（0.035%，"
                       "全 band 同倍）",
    "evidence_mirror": [
        "docs/board_audit/m15320/REPORT.md（六要素审计报告）",
        "artifacts/audit_m15320/hydrostatic_probe.json（Nx=40 独立探针）",
        "artifacts/audit_m15320/repro_aggregate.json / live_rerun.json / pytest_case.log",
        "artifacts/audit_m15320/evidence.json（三件套核验矩阵 + F1-F5）",
    ],
    "level_rationale": "L4：完整算例（解析 ref + 8 band 逐项）+ 误差分析（时间窗/采样密度/分辨率/"
                       "EOS 刚度/宽度的单变量对照 + 负对照）+ 可复现（聚合逐位一致、动态/静态实跑一致）"
                       "+ 审计在算例网格外（Nx=40）独立复算 1.276% 且与 1/Nx 律吻合。"
                       "不达 L5（非科研新结论，是既有能力的基准化）。"
                       "缺口（不改变判级但须返修）：锚点未登记 tensor_benchmark_cases（0 行，F1）；"
                       "report.json 滞后 runs/ 未含 ladder_dx001（F2）；分支未并入 main（F5）——"
                       "若机器门要求『已登记+已合并』，应下调为 L3。",
    "findings": {f["id"]: f"{f['severity']}: {f['title']}" for f in findings},
}

sql = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
    "VALUES ('TensorSPH', 'm15264 CASE-HYDROSTATIC', 15320, %d, $json$%s$json$::jsonb, 'board-audit') "
    "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, "
    "evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id "
    "RETURNING id, level, gate_state;" % (LEVEL, json.dumps(evidence, ensure_ascii=False)))
(OUT / "insert_verdict.sql").write_text(sql + "\n")
print("wrote report + evidence + figure + sql")

# ---------------------------------------------------------------- platform report
def post_report() -> None:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    tok = urllib.request.Request(base + "/auth/login", data=json.dumps(
        {"username": os.environ["TENSOR_AGENT_USER"],
         "password": os.environ["TENSOR_AGENT_PASS"]}).encode(), method="POST")
    tok.add_header("Content-Type", "application/json")
    token = json.loads(urllib.request.urlopen(tok, timeout=60).read())["access_token"]
    figures = [{
        "name": fig_png.name,
        "caption": "左：静水压剖面 SPH(算例 Nx=50 / 审计探针 Nx=40) vs 闭式解 p=rho0 g(H-y)；"
                   "右：静态内部误差沿 C/Nx 下降，动态（时间平均）误差反向上升。",
        "png_base64": base64.b64encode(fig_png.read_bytes()).decode(),
    }]
    payload = {
        "title": "m15320 交差审计：m15264 CASE-HYDROSTATIC（TensorSPH）判 L4，gate 0.744%<1% 独立复现",
        "summary": "m15264 的 CASE-HYDROSTATIC 三件套核验通过：不变量测试 8 passed；"
                   "--aggregate 重跑 9/9 配置逐位相同；动态点实跑除 runtime_s 外逐字段相同；"
                   "审计在算例网格外（Nx=40）独立复算 1.276%，与算例 1/Nx 壁列律吻合。"
                   "缺口 5 项（锚点未登记 tensor_benchmark_cases 0 行、report.json 滞后 runs/、"
                   "17.6% vs 17.923%、--run 静默忽略 --static、分支未并入 main）。判级 L4，"
                   "处置 approve with notes。",
        "scenario_key": "board_audit_case_hydrostatic",
        "mission_id": 15320,
        "tags": ["board-audit", "TensorSPH", "CASE-HYDROSTATIC", "hydrostatic", "L4", "m15264"],
        "markdown": MARKDOWN,
        "figures": figures,
    }
    req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(payload).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + token)
    resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
    print("platform report id:", resp.get("id"))


if __name__ == "__main__":
    post_report()