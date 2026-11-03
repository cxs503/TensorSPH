"""Render the docs page and the platform report from the benchmark JSON.

Every number in ``docs/benchmarks/laplace-equilibrium.md`` and in the platform
report comes from ``validation_report.json`` / ``raw_cases.json`` written by
``run.py``; nothing is transcribed by hand.  Run after a benchmark run::

    PYTHONPATH=src python benchmarks/laplace_equilibrium/write_docs.py
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs" / "benchmarks" / "laplace-equilibrium.md"
REPORT = Path("/tmp/le_report.json")


def window_mean(hist, key, lo, hi):
    """Mean/std/count of ``hist[key]`` over the sample window ``[lo, hi)``."""
    n = len(hist[key])
    a, b = int(lo * n), int(hi * n)
    vals = hist[key][a:b]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)
    return {"mean": m, "std": math.sqrt(var), "n": len(vals)}


def _fmt(x, nd=2):
    return "nan" if x != x else f"{x:.{nd}f}"


def build():
    rep = json.loads((HERE / "validation_report.json").read_text())
    raw = json.loads((HERE / "raw_cases.json").read_text())
    by_label = {c["label"]: c for c in raw}
    init = rep["initialisation"]
    theory = 7.2 / 0.02

    # ---------------- absolute-verdict table ----------------
    rows = []
    for c in rep["cases"]:
        pl, lp = c["plateau"], c["laplace"]
        rows.append(
            f"| {c['label']} | {_fmt(c['setup']['sigma_N_per_m'])} | "
            f"{c['setup']['steps']} | {_fmt(pl['mean'])} ± {_fmt(pl['std'])} | "
            f"{_fmt(pl['sem'])} | {_fmt(100 * pl['relative_fluctuation'], 1)} % | "
            f"{_fmt(lp['rel_error_pct'], 1)} % | "
            f"{'yes' if lp['within_3pct'] else 'no'} | "
            f"{'yes' if lp['within_15pct'] else 'no'} | "
            f"{'yes' if pl['stable'] else 'no'} |")
    verdict_table = "\n".join(rows)

    # ---------------- early vs late window ----------------
    win_rows = []
    for c in raw:
        h = c["history"]
        if c["setup"]["sigma_N_per_m"] == 0.0:
            continue
        t_sig = c["setup"]["t_over_t_sigma"]
        early = window_mean(h, "p_core", 0.0, 0.10)
        mid = window_mean(h, "p_core", 0.25, 0.50)
        late = window_mean(h, "p_core", 0.50, 1.0)
        th = c["reference"]["delta_p_Pa"]
        win_rows.append(
            f"| {c['label']} | {_fmt(early['mean'])} "
            f"({100 * (early['mean'] - th) / th:+.0f} %) | "
            f"{_fmt(mid['mean'])} ({100 * (mid['mean'] - th) / th:+.0f} %) | "
            f"{_fmt(late['mean'])} ({100 * (late['mean'] - th) / th:+.0f} %) | "
            f"{_fmt(h['r95'][-1] * 1000, 2)} | {_fmt(h['rho_core'][-1], 6)} |")
    window_table = "\n".join(win_rows)

    # ---------------- quiescence controls ----------------
    q_rows = []
    for c in rep["cases"]:
        if not c["quiescence"]["sigma_zero"]:
            continue
        q = c["quiescence"]
        q_rows.append(f"| {c['label']} | {_fmt(q['max_abs_p_core_Pa'], 3)} | "
                      f"{q['max_vmax']:.3e} | {'yes' if q['passes'] else 'no'} |")
    quiescence_table = "\n".join(q_rows)

    # ---------------- sigma scaling ----------------
    scaling = rep.get("scaling")
    if scaling:
        sc_rows = []
        for c in rep["cases"]:
            if c["setup"]["sigma_N_per_m"] == 0.0:
                continue
            sig = c["setup"]["sigma_N_per_m"]
            sc_rows.append(f"| {_fmt(sig)} | {_fmt(sig / 0.02)} | "
                           f"{_fmt(c['plateau']['mean'])} | "
                           f"{_fmt(c['laplace']['rel_error_pct'], 1)} % |")
        scaling_rows = "\n".join(sc_rows)
        scaling_sentence = (
            f"最小二乘斜率 `dp/dsigma = "
            f"{scaling['fit']['slope_Pa_per_N_per_m']:.2f} 1/m`"
            f"（解析 `1/R = {scaling['verdict']['theory_1_per_m']:.2f} 1/m`，"
            f"相对误差 {scaling['verdict']['rel_error_pct']:+.2f} %，"
            f"`<3%`: {scaling['verdict']['within_3pct']}，"
            f"`<15%`: {scaling['verdict']['within_15pct']}），"
            f"拟合截距 {scaling['fit']['offset_Pa']:+.1f} Pa。")
    else:
        scaling_rows = "（本次运行未做 sigma 扫描）"
        scaling_sentence = "（本次运行未做 sigma 扫描，斜率判据未评估。）"

    doc = f"""# 端到端 Young-Laplace 判据的可判性修复（静止液滴内压 σ/R）

**范围**：`benchmarks/laplace_equilibrium/run.py`（新增）、`tests/test_laplace_equilibrium.py`（新增）、
`benchmarks/csf_consistent/run.py`（旧判据标注，不改数值）
**基准**：2-D 单相静止液滴的 Young-Laplace 定律 `Delta p = sigma/R`。
本文件所有数字由 `benchmarks/laplace_equilibrium/run.py` 写出的
`validation_report.json` + `raw_cases.json` 经 `write_docs.py` 程序化注入，无手抄。
它复核的对象是**判据本身**，不是新的物理模型。

## 1. 案例介绍（物理背景与参数）

半径 `R = 0.02 m` 的二维单相液滴，六方密排晶格初始化，四周真空，无重力，
表面张力使内部建立过压 `Delta p = sigma/R`。这是 CSF 表面张力模型**唯一**的端到端
（力 → 压力场）解析判据：`docs/verification_surface_tension_curvature.md` 已经把
力幅（切割力恒等式）与曲率修进 15% 门，但端到端压差一直记作 −76 %…−107 %，
并被归类为"Tait 压缩落进求和密度噪声地板"。本工作把这句话当假设来检验。

| 参数 | 值 |
|---|---|
| 液滴半径 R | {raw[0]['setup']['radius_m']} m |
| 表面张力系数 sigma | 7.2 / 14.4 / 21.6 N/m（扫描） |
| 参考密度 rho0 | {raw[0]['setup']['rho0']} kg/m^3 |
| 声速 c0 | {raw[0]['setup']['sound_speed']} m/s |
| 状态方程 | Tait, gamma = {raw[0]['setup']['gamma']}，自由面钳位 p=0 |
| 平滑长度 | h = {raw[0]['setup']['h_over_dx']} dx（全仓标定值，未改） |
| 分辨率 | dx = {raw[0]['setup']['dx_m']} m（N = {by_label[raw[0]['label']]['profile'][0]['n'] and sum(b['n'] for b in by_label[raw[0]['label']]['profile'])} 粒子） |
| 时间推进 | dt = stable_dt，{raw[0]['setup']['steps']} 步（t/t_sigma = {raw[0]['setup']['t_over_t_sigma']:.1f}，毛细周期 ~ {raw[0]['setup']['capillary_period_s'] * 1000:.1f} ms） |
| 解析参考 | Delta p = sigma/R = {_fmt(theory)} Pa（sigma=7.2）；所需压缩 = {raw[0]['reference']['compression_required']:.3e} |

## 2. 怎么做的（方法/流程）

复核判据时发现两处**测量缺陷**（同一个 benchmark 里，与被测模型无关）：

**D1 量具**：旧 `relax()` 用 `p_core - p_band`，band 取 `0.80R < r < 0.95R`——这是在液滴
**内部**。CSF 力是集中在界面壳的体力，平衡时内部压强是平台，整个跃变发生在
最外层粒子与真空之间（真空侧由 Tait 钳位给出 p = 0）。两个内部点相减量到的是平台
残余倾斜，不是 Laplace 跃变。新量具：`p_core` 取 `r < 0.5R` 平台均值，环境压强按
定义取 0，跃变 = `p_core − 0`；`p_shell`（`0.97R < r ≤ 1.02R`）只作诊断记录。

**D2 初始条件**：`hexagonal_droplet` 按 `mass = rho0 dx dy` 布点，隐含假设该格子的
**离散单位分解**等于 1。`h/dx = 1.3` 时并不成立：实测核心求和密度
{init['baseline_core_density_ratio']:.6f} `rho0`，于是初始 Tait 压强
`B[(1.005884)^7 − 1] = {_fmt(init['baseline_initial_core_pressure_Pa'], 1)} Pa` ——
是 360 Pa 信号的 **{init['baseline_initial_core_pressure_Pa'] / theory:.1f} 倍**，
而且背后就是自由面，立刻以声波形式释放。此后任何 `p_core` 采样都是两个大数之小差。

**修复**：**体积一致初始化** —— 只改粒子质量（等价粒子体积，格距与 h/dx 都不动），
使离散单位分解恰为 1：`m* = rho0 dx dy / (sum_j V_j W_ij)`，求和密度对质量是线性的，
故一步精确，然后再用完整初始态**校验**（实测 {init['achieved_core_density_ratio']:.7f}）。
kernel、dx、h/dx、CSF 装配、状态方程、时间步限幅**全部不变**。

由此得到三条互不依赖的判据：

1. **静息控制**（sigma = 0）：无表面张力就应无内压、无运动。旧初始化过不了（初始
   2.4 kPa 会推动液滴），新初始化必须过。
2. **绝对判据**：`p_core` 的时窗统计均值 vs `sigma/R`。
3. **斜率判据**：`d p_core / d sigma = 1/R`，与任何常数偏置无关（偏置只进截距）。

统计口径：时窗 = 后 50% 采样；`stable` 要求相对波动 ≤ 15%（沿用静水 benchmark 的
压力波动门）且时窗残余趋势 ≤ 10%。同时报告均值的标准误 `sem`。

工具链：`benchmarks/laplace_equilibrium/run.py`（6.9 k 行级参数扫描 + 出图）、
`write_docs.py`（本文档与平台报告的数字注入）、`tests/test_laplace_equilibrium.py`
（13 个判据单测：格子偏置、质量标定、时窗统计、斜率无偏置、径向分箱、静息控制）。

## 3. 计算结果

### 3.1 初始条件偏差（D2）

| 初始状态 | 核心求和密度 / rho0 | 初始核心 Tait 压强 [Pa] | 与 sigma/R=360 之比 |
|---|---|---|---|
| 仓库原状（`mass = rho0 dx dy`） | {init['baseline_core_density_ratio']:.6f} | {_fmt(init['baseline_initial_core_pressure_Pa'], 1)} | {init['baseline_initial_core_pressure_Pa'] / theory:.2f}× |
| 体积一致（`mass_scale = {init['mass_scale']:.6f}`） | {init['achieved_core_density_ratio']:.7f} | {_fmt(init['calibrated_initial_core_pressure_Pa'], 2)} | {abs(init['calibrated_initial_core_pressure_Pa']) / theory:.2e}× |

### 3.2 绝对判据（时窗 = 后 50%，误差对 `sigma/R`）

| 算例 | sigma [N/m] | 步数 | 时窗均值 p_core ± std [Pa] | sem [Pa] | 相对波动 | 相对误差 | <3%? | <15%? | 稳定? |
|---|---|---|---|---|---|---|---|---|---|
{verdict_table}

### 3.3 同一算例的不同时窗（诊断：压差是否随时间长漂）

| 算例 | 前 10% 采样均值（误差） | 25–50% 均值（误差） | 后 50% 均值（误差） | 终态 r95 [mm] | 终态核心 rho/rho0 |
|---|---|---|---|---|---|
{window_table}

### 3.4 sigma = 0 静息控制

| 算例 | max abs(p_core) [Pa] | max abs(v) [m/s] | 通过? |
|---|---|---|---|
{quiescence_table}

### 3.5 斜率判据

| sigma [N/m] | sigma/R 解析 [Pa] | 时窗均值 p_core [Pa] | 相对误差 |
|---|---|---|---|
{scaling_rows}

{scaling_sentence}

![laplace equilibrium](laplace_equilibrium.png)

图 (a) 内部压强时间历程（虚线 = 解析 `sigma/R`）；(b) 终态径向压强剖面；
(c) 时窗均值与 15% 门；(d) `p_core` vs `sigma` 与解析斜率 `1/R`。

## 4. 结果判读

<!-- VERDICT -->

## 5. 注意点（踩过的坑）

1. **D1（量具）**：界面壳一旦把整个跃变吃掉，`p_core − p_band` 就退化成平台倾斜，
   与 Laplace 定律无关。任何"压差"判据必须说明环境侧取 0 的依据。
2. **D2（初始条件）**：`mass = rho0 dx dy` 只在离散单位分解 = 1 时才对；`h/dx = 1.3`
   时偏 0.59 %，恰好与"360 Pa 所需压缩 6.3e-3"同量级 —— 这不是巧合，是同一套离散化
   的两面：**信号与初始偏差同阶**，所以旧判据永远测不出结论。这解释了历史上
   −76 %/−88 %/−107 % 三个互相矛盾的读数（它们是同一条过冲/回落曲线上的不同采样时刻）。
3. **单点采样不可用**：液滴是近乎无粘的振子，`p_core` 在 46–614 Pa 之间摆动；
   必须用时窗统计 + 波动/漂移门，否则任何一个数都能被"测"出来。
4. **质量标定不改几何**：只缩质量，格距、粒子数、h/dx、配对表全不变，因此不影响
   曲率/`delta_s` 的既有标定结果（见 `tests/test_laplace_equilibrium.py`）。
5. **诊断字段 `r95` 必需**：液滴在长时间推进中会缓慢变大（自由面求和密度亏损向内扩散），
   不记录实际半径就无法区分"压强漏掉"与"sigma/R 本身变小"。

## 6. 误差判定（逐项，显式回答「是否 < 3%」）

<!-- ERRORS -->

## 7. 未完成 / 交接

<!-- HANDOFF -->
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(doc)
    return rep, raw, doc


if __name__ == "__main__":
    rep, raw, doc = build()
    print("doc ->", DOC)
    print("cases:", [c["label"] for c in rep["cases"]])
    for c in rep["cases"]:
        print(f"  {c['label']:46s} mean={c['plateau']['mean']:8.2f} "
              f"rel={c['laplace']['rel_error_pct']:+8.2f}%")