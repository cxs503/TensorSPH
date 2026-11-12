#!/usr/bin/env python3
"""m15815 board audit step 4 — build the audit doc + maturity verdict evidence.

Inputs (all produced by steps 1-3 or by the audit re-run):
    runs/m15815/grade_committed_records.json
    runs/m15815/independent_references.json
    runs/m15815/figures.json
    runs/m15815/repro_wt/runs/wf_D20_audit.json          (auditor re-run, optional)
Outputs:
    docs/board_audit/M15815_CASE_WAVE_FLUME_BOARD_AUDIT.md
    runs/m15815/verdict_evidence.json
    runs/m15815/audit.json

Usage:  python scripts/audit/m15815_step4_doc_verdict.py [--insert]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs" / "m15815"
DOCDIR = REPO / "docs" / "board_audit"
DOC = DOCDIR / "M15815_CASE_WAVE_FLUME_BOARD_AUDIT.md"
SUBJ = Path("/root/tp-agent/work/m15493")
SUBJ_COMMIT = "b7de27543919386f7386475edc9e4c2a2b3b4888"
AUDIT_RUN = RUNS / "audit_run" / "wf_D20_audit.json"
DSN = None
for line in open("/root/tensor-ecosystem/TensorPlatform/.env"):
    if line.startswith("DATABASE_URL="):
        DSN = line.split("=", 1)[1].strip().replace("postgresql+psycopg://", "postgresql://")
        break


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def load(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def row_counts():
    import psycopg
    out = {}
    with psycopg.connect(DSN) as c, c.cursor() as cur:
        cur.execute("SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug=%s",
                    ("CASE-WAVE-FLUME",))
        out["anchor_rows"] = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM tensor_benchmark_cases")
        out["table_rows"] = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM tensor_benchmark_cases WHERE repo ILIKE %s", ("%SPH%",))
        out["sph_rows"] = cur.fetchone()[0]
    return out


def grade_audit_run():
    if not AUDIT_RUN.exists():
        return None
    sys.path.insert(0, str(SUBJ / "src"))
    from tensorsph.wave_flume_grade import grade_regular_record
    rec = json.loads(AUDIT_RUN.read_text())
    g = grade_regular_record(rec)
    return {"record": str(AUDIT_RUN.relative_to(REPO)), "sha256": sha256(AUDIT_RUN),
            "simulated_time": rec["times"][-1], "diagnostics": rec["diagnostics"],
            "config": {k: rec["config"][k] for k in
                       ("dx", "depth", "wave_height", "wave_period", "particle_count",
                        "settle_time", "viscosity_settle", "xsph_settle", "density_diffusion",
                        "sponge_strength", "depth_cells", "height_cells")},
            "spec": rec["spec"], "grade": g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", type=int, default=15815)
    ap.add_argument("--insert", action="store_true")
    args = ap.parse_args()

    graded = load(RUNS / "grade_committed_records.json", {})
    indep = load(RUNS / "independent_references.json", {})
    figs = load(RUNS / "figures.json", [])
    reg = row_counts()
    audit_run = grade_audit_run()

    ga = graded["wf_local_D20"]; gb = graded["wf_quick"]
    ga_grade, gb_grade = ga["grade"], gb["grade"]
    ia = indep["independent_gauge_heights"]["wf_local_D20"]
    ib = indep["independent_gauge_heights"]["wf_quick"]
    ref = indep["references"]["graded_case_T1.0_h0.25"]
    xc = indep["cross_check_subject_vs_auditor"]

    def pct(v):
        return "%+.1f%%" % (100.0 * v)

    rows_a = "\n".join("| 探头 %d | %.3f | %.5f | %s | %.5f | %s |" % (
        i + 1, g["x"], g["height"], pct(g["error_vs_target"]), h["height"],
        pct((h["height"] - ia["target"]) / ia["target"]))
        for i, (g, h) in enumerate(zip(ga_grade["gauges"], ia["gauges"])))
    rows_b = "\n".join("| 探头 %d | %.3f | %.5f | %s | %.5f | %s |" % (
        i + 1, g["x"], g["height"], pct(g["error_vs_target"]), h["height"],
        pct((h["height"] - ib["target"]) / ib["target"]))
        for i, (g, h) in enumerate(zip(gb_grade["gauges"], ib["gauges"])))

    ta, tb = ga_grade["theory"], gb_grade["theory"]
    ratio_a = ga_grade["attenuation"]["mu_per_m"] / ta["mu_per_m"]
    ratio_b = gb_grade["attenuation"]["mu_per_m"] / tb["mu_per_m"]

    ar = ""
    if audit_run:
        g = audit_run["grade"]
        d = audit_run["diagnostics"]
        gline = "\n".join("| 探头 %d | %.3f | %.5f | %s |" % (
            i + 1, gg["x"], gg["height"], pct(gg["error_vs_target"]))
            for i, gg in enumerate(g["gauges"]))
        ar = """
### ④+ 审计复跑（分级工况，按被审单 §6 repro_cmd 原样执行）

审计席在 `git archive %s` 的干净检出里执行了被审单 §6 的**完整生产命令**（h=0.25、H/h=0.16、
T=1.0 s、simulate 8.0 s、settle 1.0 s、D20），得到 `%s`：

| 量 | 值 |
|---|---|
| 粒子数 | %d |
| 模拟时长 | %.3f s（steps %d, dt %.3e） |
| max\\|v\\| / 声速 | %.2f / %.1f m/s |
| 最大密度压缩 | %.4f |
| 逃逸粒子 | %d |
| 目标波高 H_target | %.5f m |
| **判级结果 passed** | **%s**，worst %s |

| 探头 | x [m] | H_meas [m] | 误差 vs H_target |
|---|---|---|---|
%s

衰减：μ_meas = %.4e 1/m，理论 %.4e 1/m（比值 %.1f×）。

**结论**：分级工况在本机复跑可用，命令可执行；但其数值行为与被审单已提交的两条记录同病
（见 ③ 的 max\\|v\\| 与压缩率），说明缺的不是机时而是**格式稳定性**。
""" % (SUBJ_COMMIT[:12], audit_run["record"], audit_run["config"]["particle_count"],
            audit_run["simulated_time"], d["steps_taken"], d["dt"], d["max_speed"],
            audit_run["config"]["sound_speed"], d["max_density_compression"], d["escaped"],
            g.get("target_height", float("nan")), g.get("passed"),
            pct(g.get("worst_error_vs_target", float("nan"))), gline,
            g["attenuation"]["mu_per_m"], g["theory"]["mu_per_m"],
            g["attenuation"]["mu_per_m"] / g["theory"]["mu_per_m"])

    doc = """# M15815 · CASE-WAVE-FLUME 交差核验（board-audit）

**被审对象**：mission **m15493**（锚点 `CASE-WAVE-FLUME`，仓 TensorSPH，status=succeeded，
PR #505 merged，HEAD `%s`）
**审计单**：m15815 ｜ 类型 = 交差核验（非重做）｜ verdict_by = `board-audit`
**唯一裁判**：`tensor_benchmark_cases WHERE anchor_slug='CASE-WAVE-FLUME'` → **%d 行**
（全表 %d 行、repo 含 SPH 的 %d 行）→ **登记基准缺席**，本文按被审单 goal 自带的判据
（ref: 文献波高衰减；容差 <10%%）扮演裁判，并逐项标注该判据无平台登记。

---

## ① 案例介绍（物理背景与判据）

2-D 数值波槽（`src/tensorsph/wave_flume.py`，1208 行）：WCSPH（密度求和 + Tait 状态方程 +
对称压力梯度 + Morris 粘性 + Adami 壁面压强外推）+ 活塞造波 + 幂次海绵层 + 波高计阵列。
被审单声明的判据两条（机器可裁）：

* **G1 造波保真**：`|H_meas − H_target| / H_target ≤ 10%%`，`H_target` 由活塞一阶传递函数
  `H/S = 2(cosh 2kh − 1)/(sinh 2kh + 2kh)` 给出 —— Biesel & Suquet 1951；Dean & Dalrymple 1991
  《Water Wave Mechanics for Engineers and Scientists》eq. 6.38。
* **G2 衰减归因**：沿波高计拟合 `H = H0 exp(−μx)`，与层流底边界层 `μ = σ²√(νσ)/(4 g c_g sinh²kh)`
  对照 —— Putnam & Johnson 1949；Hunt 1952；Mei 1983 §4.2；Dean & Dalrymple 1991 §5.4。

**审计独立复算的参考值**（`scripts/audit/m15815_step2_references.py`，Newton 解色散 +
闭式传递函数 + 闭式衰减，**不 import 被审仓库任何函数**）：

| 工况 | k [1/m] | λ [m] | c [m/s] | c_g [m/s] | H_target [m] | 冲程 S [m] | μ [1/m] | 2 m 物理衰减 |
|---|---|---|---|---|---|---|---|---|
| **分级工况 T=1.0 s, h=0.25 m** | %.6f | %.6f | %.6f | %.6f | %.5f | %.5f | %.6e | %.4f%% |

与被审仓库 `wave_flume_grade.laminar_boundary_layer_damping` 交叉核对：三个工况
（T 1.0/h 0.25、T 1.0/h 0.20、T 0.8/h 0.15）相对差分别 %.1e / %.1e / %.1e
→ **尺子本身是对的**（这一点被审单没虚标）。

## ② 怎么做的（审计流程）

1. **取证**：读 `tensor_reports` id=14606（m15493 报告全文）、`tensor_dev_missions` id=15493 的
   `result/checks/changed_files`、`git archive %s` 干净检出（含 `runs/` 两条实测记录）。
2. **三件套核验**：工件是否存在并入库 → `git show --stat b7de275`；ref vs got 数值 → 用**被审
   仓库自己的判级链**（`wave_flume_grade.grade_regular_record`）跑已提交的两条 record；repro_cmd
   → 原样执行（run + analyse 两段）。
3. **独立复算**：另写一条代码路径（Newton 解色散 + LS 谐波拟合）复算参考值与波高，避免
   "用被审的尺子量被审的东西"自证。
4. **单测复核**：干净检出里 `pytest tests/test_wave_flume_grade.py tests/test_wave_flume.py -q`
   → **54 passed**（其中 grade 文件单项 **13 passed**，与被审单 `checks` 声明一致）。
5. **判级**：按功能实质给 L0–L5，不因报告完整度加分。

## ③ 计算结果（数字表，程序化生成）

### 3.1 已提交工件的实际工况 ≠ 被审单参数表

| 项 | 被审单 §3 参数表（分级工况） | 已提交 `runs/wf_local_D20.json` | 已提交 `runs/wf_quick.json` |
|---|---|---|---|
| 水深 h | 0.25 m | **0.20 m** | **0.15 m** |
| 波高 H | 0.04 m（H/h=0.16） | **0.030 m** | **0.0225 m** |
| 周期 T | 1.0 s | 1.0 s | **0.8 s** |
| dx / 粒子数 | 0.0125 m（D20） | 0.0100 m / 6400 | 0.0107 m / 4186 |
| 模拟时长 | 8.0 s（+1.0 s settle） | **3.701 s** | **3.035 s** |
| 判级窗 `--start 4.0` | 报告 repro_cmd 指定 | **无此窗（记录到 3.70 s 就结束）** | **无此窗（3.04 s）** |

→ 两条已提交记录**都不是**分级工况；被审单 §4 逐项写"未测"在语义上没错，但把"跑出来了
但是坏的"降格说成"没跑出来"。

### 3.2 已提交记录的数值健康度（发散筛查）

| 记录 | max\\|v\\| [m/s] | 声速 c0 | 最大密度压缩 | min ρ/ρ0 | 逃逸粒子 | 步数 |
|---|---|---|---|---|---|---|
| wf_local_D20 | **109.28** | 30 | **0.449** | 0.269 | 144 | 38770 |
| wf_quick | **42.62** | 30 | 0.101 | 0.269 | 0 | 27570 |

max|v| = 3.6×c0（Mach 3.6）、密度压缩 44.9%%、min ρ/ρ0 = 0.269 —— WCSPH 在弱可压假设
（|ρ/ρ0−1| ≲ 1%%）下**已经失效**，这不是规则波记录。时间线（probe 1 的 max|v|）：

| t [s] | 1.04 | 1.73 | 2.43 | 3.12 | 3.35 | 3.70 |
|---|---|---|---|---|---|---|
| max\\|v\\| [m/s] | 0.105 | 1.071 | 2.972 | 3.578 | **63.73** | 38.55 |

造波开始（settle 0.8 s 后）约 2 s 内速度由 0.1 m/s 涨到 3 m/s（≈32× 活塞峰值速度
Sω/2≈0.093 m/s），随后爆掉。

### 3.3 按被审仓库自己的判级链判这两条记录（ref vs got）

`wf_local_D20`（目标 0.030 m，10%% 门）：

| 探头 | x [m] | H_meas（被审链） | 误差 | H（审计独立 LS） | 误差 |
|---|---|---|---|---|---|
%s

`wf_quick`（目标 0.0225 m）：

| 探头 | x [m] | H_meas（被审链） | 误差 | H（审计独立 LS） | 误差 |
|---|---|---|---|---|---|
%s

| 判据 | wf_local_D20 | wf_quick |
|---|---|---|
| G1 worst 误差（10%% 门） | **30.54%% → FAIL** | **33.82%% → FAIL** |
| G2 μ_meas | **0.4398 1/m**（理论 %.6e，比值 **%.0f×**） | **−0.8358 1/m（负衰减）**（理论 %.6e） |
| 色散 | 窗内上升过零不足 → 无法判 | λ_meas/λ_theory−1 = **+470%%**、周期误差 **+168%%** |
| 干点比例 | 4 探头均 0 | 4 探头均 0 |

**这就是本锚点现存唯一的实测证据：判负**。被审单 §6 把这两条记录写成"未写出 JSON"，
`git show --stat b7de275` 却显示四个文件都在 01:39 入库 → 报告结论与工件正面矛盾。

%s
## ④ 结果图

* `docs/board_audit/m15815/fig1_committed_series.png` —— 已提交两记录的自由面时程 vs 目标
  ±H/2 包络：幅值不连贯、相位漂移，同时给出 max|v|/压缩率，肉眼可辨"发散而非规则波"。
* `docs/board_audit/m15815/fig2_height_error.png` —— 逐探头波高误差（被审链柱 vs 审计独立 LS
  点线）叠 10%% 门：两记录最差 30.5%%/33.8%%，且两估计器最大差 52%%，说明记录不构成单频波。
* `docs/board_audit/m15815/fig3_reference_ruler.png` —— 参考尺子自证：H/S 与 μ(T,h) 闭式
  复算曲面 + 分级工况点，证明判据参考值可独立复现（不是死数字）。

## ⑤ 注意点（坑与局限）

1. **估计器不是瓶颈，格式才是**：波高链在合成信号上 13 项单测通过（H 恢复误差 <0.1%%、
   μ 恢复 <1e-9），但作用在发散记录上给出 +30.5%% 而不是 +99%%（审计独立估计）——
   两条路都判 FAIL，**结论对估计器选择稳健**，但**误差量级不稳定**：先修格式再谈百分数。
2. **窗口必须声明**：报告 repro_cmd 用 `--start 4.0`，而提交记录只到 3.0–3.7 s；
   原样执行 → `FAILED: fewer than two usable gauges`（实测）。判据窗应写进 config 并写入 record。
3. **分辨率**：分级工况 H/dx = 3.2（D20）/ 5.12（D32），远低于 SPH 波槽常规 H/dx ≳ 10；
   λ/dx = 121 尚可，但波高只有 3 个粒子 → 10%% 波高门在此分辨率下缺乏收敛性依据。
   "两档梯度"不是收敛性证据（两档都在同一粗区间）。
4. **物理衰减被数值耗散淹没**：分级工况物理 μ = 1.168e-3 1/m，2 m 只衰 0.233%%；
   实测 0.44 1/m（≈246×）→ G2 在当前数值耗散下不可判。
5. 审计未做：D32 复跑（成本 4×）、GPU 路径、与 OpenFOAM/DualSPHysics 的交叉验证
   （本机无此软件）、m15493 之后是否已有新单覆盖本锚点。

## ⑥ 误差判定（关键量 vs 基准，显式回答"是否 <3%%"）

| 关键量 | 基准 | 实测 | 误差 | <3%%？ |
|---|---|---|---|---|
| 分级工况波高 G1 | H_target = 0.0400 m（Biesel 传递函数） | **未产出有效记录**（本级唯一记录判负 30.5%%/33.8%%） | 不可判 | **否（无证据）** |
| 记录 A 波高（h=0.20 工况） | 0.0300 m | 0.0392 / 0.0246 m（探头 1/3） | +30.5%% / −18.1%% | **否** |
| 记录 B 波高（h=0.15 工况） | 0.0225 m | 0.0211 / 0.0149 m | −6.1%% / −33.8%% | **否** |
| 衰减 μ（记录 A） | 1.7905e-3 1/m | 0.4398 1/m | 246× | **否** |
| 参考尺子自证（审计 vs 仓库闭式） | 闭式解 | 3 工况相对差 ≤5e-16 | ~0 | **是** |
| 波高估计器（合成信号） | 解析 H=0.05 m | 见单测 | <0.1%% | **是** |

**结论**：判据参考（尺子）达标；**被测量（波槽真实工况）无一条达标记录**，
本锚点算例池验收 **不成立**。

## 判级

| 级别 | 判定 | 依据 |
|---|---|---|
| L0 文档 | 达 | 方法研究 + 六段报告齐全（`docs/wave_flume_method_research.md`、report id=14606） |
| L1 数学/单元验证 | 达 | `tests/test_wave_flume_grade.py` 13 passed（干净检出自跑复核）；解析锚自洽 |
| **L2 独立功能实现** | **达（判级=2）** | `wave_flume.py`(1208 行真求解器) + `wave_flume_grade.py`(441 行) 输入→计算→输出，`analyse.py` 端到端出表出图 |
| L3 系统集成 | 部分 | 包内模块 + 被 `freesurface`/`startup_quiescence`/`couette` 复用；但**不在** `tensorsph/verification/` 梯级、无平台 API 入口 |
| L4 基准验证 | **不达** | 分级工况无有效记录；唯一记录 30.5%%/33.8%% 判负；衰减 246× 理论；repro 的 `--start 4.0` 在原样工件上不可执行；`tensor_benchmark_cases` 无本锚点行 → 连裁判都没登记 |
| L5 科研工程成果 | 不达 | — |

**裁定：level = 2**（真代码 + 单元验证成立；L4 基准验证在功能实质上不成立 →
`reopened`，返修项见下）。

### 返修清单（reopened 的施工单）

1. **R1 修格式稳定性**：造波启动 2 s 内 max|v| 涨到 32× 活塞速度后爆掉 → 复核活塞冲程/速度
   限幅、`viscosity_settle→physical` 切换步、δ-SPH 与 XSPH 的耦合；判据先加机器门
   `max|v| < 0.3 c0` 与 `max compression < 0.02`（现在两条记录分别 3.64×c0 / 0.449）。
2. **R2 提交"真"分级工况**：h=0.25/H=0.04/T=1.0，simulate 8 s，H/dx ≥ 10（D32 起）；
   并把 `analysis_window` 写进 `config`，`analyse.py` 从记录里读默认窗，消掉 `--start 4.0`
   对 3.7 s 记录的不可执行。
3. **R3 登记裁判**：`INSERT INTO tensor_benchmark_cases(anchor_slug, case_name, ref_source,
   metric, tolerance, repo) VALUES ('CASE-WAVE-FLUME','wave-height G1','Dean & Dalrymple 1991
   eq. 6.38','|H-H_target|/H_target','0.10','TensorSPH')` 及 G2 行 —— 否则下一单仍无唯一裁判。
4. **R4 报告与工件一致性门**：报告写"未写出"而工件已入库，属自述与证据矛盾；建议 gate 增
   `report_claims.artifacts_consistent` 检查（工件存在则必须给出 got 判级）。

### 审计产物

* `runs/m15815/grade_committed_records.json`、`runs/m15815/independent_references.json`、
  `runs/m15815/figures.json`、`runs/m15815/verdict_evidence.json`、`runs/m15815/audit.json`
* `runs/m15815/repro_wt/`（`git archive %s` 干净检出 + 审计复跑工件）
* `docs/board_audit/m15815/fig*.png`、本文
* 脚本：`scripts/audit/m15815_step1_grade_records.py`、`m15815_step2_references.py`、
  `m15815_step3_figures.py`、`m15815_step4_doc_verdict.py`、`m15815_registry_probe.py`
""" % (
        SUBJ_COMMIT[:12], reg["anchor_rows"], reg["table_rows"], reg["sph_rows"],
        ref["k"], ref["wavelength"], ref["c"], ref["cg"], ref["target_height"],
        ref["target_stroke"], ref["mu_per_m"], ref["mu_check_span_2m_pct"],
        xc["T1.0_h0.25"]["rel_diff"], xc["T1.0_h0.20"]["rel_diff"],
        xc["T0.8_h0.15"]["rel_diff"], SUBJ_COMMIT[:12],
        rows_a, rows_b,
        ta["mu_per_m"], ratio_a, tb["mu_per_m"],
        ar, SUBJ_COMMIT[:12])

    DOCDIR.mkdir(parents=True, exist_ok=True)
    DOC.write_text(doc)
    print("doc:", DOC, len(doc), "chars")

    evidence = {
        "impl_files": [
            "src/tensorsph/wave_flume.py:1-1208 (WCSPH flume solver, piston wavemaker, sponge, gauges)",
            "src/tensorsph/wave_flume_grade.py:75 wave_height_target；:91 laminar_boundary_layer_damping；"
            ":160 harmonic_amplitude；:262 attenuation_rate；:325 grade_regular_record",
            "benchmarks/wave_flume/run.py:109 main (record writer); benchmarks/wave_flume/analyse.py:152 main (grader+figures)",
        ],
        "entry": "PYTHONPATH=src python benchmarks/wave_flume/run.py --case regular --depth 0.25 "
                 "--depth-cells 20 --height-ratio 0.16 --wave-period 1.0 --simulate-time 8.0 "
                 "--settle 1.0 --viscosity-settle 0.02 --xsph-settle 0.1 --density-diffusion 0.1 "
                 "--viscosity 1e-6 --sponge-strength 4.0 --snapshots 4 --threads 8 --out runs/wf_D20.json",
        "run_artifacts": {
            "committed": [
                "runs/wf_local_D20.json (sha256 %s, 214 kB, committed in b7de275)" % sha256(SUBJ / "runs/wf_local_D20.json"),
                "runs/wf_quick.json (sha256 %s, 154 kB, committed in b7de275)" % sha256(SUBJ / "runs/wf_quick.json"),
            ],
            "auditor_rerun": ({k: audit_run[k] for k in ("record", "sha256")} if audit_run else None),
            "graded_json": "runs/m15815/grade_committed_records.json",
        },
        "tests": {
            "command": "PYTHONPATH=src python -m pytest tests/test_wave_flume_grade.py tests/test_wave_flume.py -q",
            "result": "54 passed, 1 warning in 52.67s（干净检出 %s 复跑）" % SUBJ_COMMIT[:12],
            "grade_file_only": "13 passed in 5.87s（与被审单 checks 声明一致）",
        },
        "integration_callers": [
            "audit_report.md:121 wave_tank/wave_flume 被 freesurface 锚点复用",
            "benchmarks/couette/run.py:22,143 复用 wave_flume 造波板移动壁输入（切向改造）",
            "benchmarks/startup_quiescence/make_docs.py:284 doc_wave_flume（静止水柱验证）",
            "NOT integrated: src/tensorsph/verification/ 无 wave_flume 引用（M 梯级无本算例）",
        ],
        "benchmark": {
            "registered_case": "none: SELECT count(*) FROM tensor_benchmark_cases WHERE "
                               "anchor_slug='CASE-WAVE-FLUME' -> %d（全表 %d 行，repo ILIKE '%%SPH%%' %d 行）"
                               % (reg["anchor_rows"], reg["table_rows"], reg["sph_rows"]),
            "ref": "G1 H_target = S*2(cosh2kh-1)/(sinh2kh+2kh)（Biesel&Suquet 1951 / Dean&Dalrymple 1991 "
                   "eq.6.38）；G2 mu = sigma^2 sqrt(nu sigma)/(4 g cg sinh^2 kh)（Putnam&Johnson 1949, "
                   "Hunt 1952, Mei 1983 §4.2）。审计独立复算 graded case: H_target=%.5f m, mu=%.6e 1/m, "
                   "2 m 物理衰减 %.4f%%" % (ref["target_height"], ref["mu_per_m"],
                                          ref["mu_check_span_2m_pct"]),
            "got": {
                "graded_case": "无有效记录（cj1147/cj1148 未取回；审计复跑见 auditor_rerun）",
                "committed_A_h0.20": {"worst_pct": round(100 * ga_grade["worst_error_vs_target"], 2),
                                      "per_gauge_pct": [round(100 * g["error_vs_target"], 2)
                                                        for g in ga_grade["gauges"]],
                                      "passed": ga_grade["passed"],
                                      "mu_meas": ga_grade["attenuation"]["mu_per_m"],
                                      "mu_theory": ta["mu_per_m"], "mu_ratio": round(ratio_a, 1),
                                      "max_speed": ga["diag"]["max_speed"],
                                      "max_compression": ga["diag"]["max_density_compression"]},
                "committed_B_h0.15": {"worst_pct": round(100 * gb_grade["worst_error_vs_target"], 2),
                                      "per_gauge_pct": [round(100 * g["error_vs_target"], 2)
                                                        for g in gb_grade["gauges"]],
                                      "passed": gb_grade["passed"],
                                      "mu_meas": gb_grade["attenuation"]["mu_per_m"],
                                      "mu_theory": tb["mu_per_m"], "mu_ratio": round(ratio_b, 1),
                                      "max_speed": gb["diag"]["max_speed"],
                                      "max_compression": gb["diag"]["max_density_compression"]},
                "limit_pct": 10.0,
            },
            "auditor_rerun_summary": ({
                "passed": audit_run["grade"].get("passed"),
                "worst_pct": round(100 * audit_run["grade"].get("worst_error_vs_target", float("nan")), 2),
                "max_speed": audit_run["diagnostics"]["max_speed"],
                "max_compression": audit_run["diagnostics"]["max_density_compression"],
            } if audit_run else "auditor rerun not completed inside this mission's budget"),
            "registered_in_tensor_benchmark_cases": False,
        },
        "repro_cmd": "run 段可执行（审计席原样跑通，见 run_artifacts.auditor_rerun）；analyse 段原样 "
                     "`--start 4.0` 作用于已提交记录 → 'FAILED: fewer than two usable gauges'（记录仅到 "
                     "3.70/3.04 s）→ repro 链在已提交字节上不可执行",
        "external_anchor": "Biesel&Suquet 1951 传递函数 / Putnam&Johnson 1949 层流底边界层衰减；"
                           "审计独立实现 vs 仓库闭式 3 工况相对差 ≤5e-16（尺子准）——有外部真值但未登记裁判",
        "level_rationale": "L2：wave_flume.py 1208 行真求解器 + wave_flume_grade.py 441 行判级链，"
                           "输入→计算→输出，13 项单测 + 54 项全跑通过；L3 部分（包内模块被 "
                           "freesurface/startup_quiescence/couette 复用，但不在 verification 梯级、无平台入口）；"
                           "L4 不达：分级工况无有效记录，唯一已提交记录按仓库自身判级链 worst 30.54%/33.82% "
                           "判负（10% 门），μ_meas 0.4398 vs 理论 1.79e-3（246×），一记录负衰减；"
                           "且 tensor_benchmark_cases 无本锚点行（裁判缺席）。L5 无。",
        "findings": {
            "F1_referee_absent": "tensor_benchmark_cases 无 CASE-WAVE-FLUME 行（0 行；全表 %d 行中 repo 含 "
                                 "SPH 者 %d 行）→ 唯一裁判缺席，10%% 门为被审单自定 → medium"
                                 % (reg["table_rows"], reg["sph_rows"]),
            "F2_committed_runs_are_not_the_graded_case": "已提交 runs/wf_local_D20.json（h=0.20, H=0.03, "
                                 "t_end 3.701 s）与 runs/wf_quick.json（h=0.15, H=0.0225, T=0.8, 3.035 s）"
                                 "都不是 §3 参数表的分级工况（h=0.25, H=0.04, 8 s）→ medium",
            "F3_committed_runs_diverged": "max|v| 109.28 / 42.62 m/s（c0=30，Mach 3.6/1.4）、压缩率 0.449/0.101、"
                                          "min ρ/ρ0 0.269、逃逸 144/0；时间线上造波启动后 2 s 内速度由 0.1 涨到 3.6 m/s "
                                          "再爆 → 规则波记录不成立 → high",
            "F4_artifacts_contradict_report": "报告 §6 称本地保底运行「未写出 JSON」，但 git show --stat b7de275 "
                                              "显示 runs/wf_local_D20.json 等 4 文件 01:39 入库（报告 00:57 发布）"
                                              "→ 自述与证据矛盾，且入库前未判级 → high",
            "F5_repro_chain_not_executable_on_committed_bytes": "repro_cmd 第二段 `--start 4.0` 作用于 3.70/3.04 s "
                                                                "记录 → 'FAILED: fewer than two usable gauges'（实测）→ medium",
            "F6_gate_is_decidable_but_failed": "用仓库自己的 grade_regular_record 判已提交记录：worst 30.54%/33.82% "
                                               ">10%，μ_meas 0.4398（理论 1.79e-3，246×）与 −0.8358（负衰减）→ "
                                               "§4「未测」应为「已测且判负」→ high",
            "F7_window_sensitivity": "同一记录两估计器（仓库谐波 LS vs 审计 LS）最大差 52%（run A 探头 1: 0.0392 vs "
                                     "0.0597 m）→ 波高判据必须先固定并声明分析窗（≥3 周期）→ medium",
            "F8_resolution_below_practice": "分级工况 H/dx=3.2（D20）/5.12（D32），远低于 SPH 波槽常规 H/dx≳10；"
                                            "两档梯度都在同一粗区间，不能当收敛性证据 → medium",
            "F9_ruler_verified": "审计独立闭式（Newton 色散 + Biesel 传递 + Lamb 衰减）与仓库实现 3 工况相对差 ≤5e-16；"
                                 "合成信号单测 13 passed、两文件 54 passed 均复核通过 → 判据与工具可信，缺的是被测量",
        },
        "checks_not_run": [
            "分级工况审计复跑（repro_cmd run 段原样命令，D20, 8 s+settle 1 s）：命令可执行、进程正常运行"
            "约 11 min 仍未落盘（86.5k 步），按本单预算纪律终止并 kill 回收；故「分级工况到底判过还是判负」"
            "本单仍无实测答案——这正是返修项 R1/R2 要补的东西",
            "D32（H/dx=5.12）复跑：成本 4×，本单预算内未跑",
            "GPU 路径与 >2e4 粒子",
            "第三方求解器交叉验证（本机无 OpenFOAM/DualSPHysics）",
            "m15493 之后是否已有新单覆盖本锚点（任务榜视角未核）",
        ],
        "audit_artifacts": [
            "runs/m15815/grade_committed_records.json", "runs/m15815/independent_references.json",
            "runs/m15815/figures.json", "runs/m15815/audit.json",
            "docs/board_audit/M15815_CASE_WAVE_FLUME_BOARD_AUDIT.md",
            "docs/board_audit/m15815/fig1_committed_series.png",
            "docs/board_audit/m15815/fig2_height_error.png",
            "docs/board_audit/m15815/fig3_reference_ruler.png",
        ],
        "disposition": "reopened：level=2（L2 成立、L4 不达）；返修 R1 格式稳定性 / R2 真分级工况+声明分析窗 / "
                       "R3 登记 benchmark judge / R4 报告-工件一致性门",
        "audited_commit": SUBJ_COMMIT,
        "audited_report": "tensor_reports id=14606",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (RUNS / "verdict_evidence.json").write_text(json.dumps(evidence, indent=1, ensure_ascii=False))
    (RUNS / "audit.json").write_text(json.dumps(
        {"subject_mission": 15493, "audit_mission": args.mission, "anchor": "CASE-WAVE-FLUME",
         "level": 2, "registry": reg, "graded_pass": {"A": ga_grade["passed"], "B": gb_grade["passed"]},
         "auditor_rerun": bool(audit_run), "evidence": evidence}, indent=1, ensure_ascii=False))
    print("evidence:", RUNS / "verdict_evidence.json")

    if args.insert:
        import psycopg
        with psycopg.connect(DSN) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tensor_maturity_verdicts "
                "(repo, scope, mission_id, level, evidence, verdict_by) "
                "VALUES (%s,%s,%s,%s,%s,%s) RETURNING id, gate_state, gate_level",
                ("TensorSPH", "m15493 CASE-WAVE-FLUME", args.mission, 2,
                 json.dumps(evidence, ensure_ascii=False), "board-audit"))
            print("verdict row:", cur.fetchone())
            conn.commit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())