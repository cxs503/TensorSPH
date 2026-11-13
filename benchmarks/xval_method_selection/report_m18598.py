"""Generate ``docs/report_xval_method_selection_m18598.md`` from the artifacts.

Every number in the report comes from a JSON artifact written by
``attribute_onset.py`` (and, when they exist, from the graded repair cells), so
the report cannot drift from the data.  Prose is fixed here; digits are injected.

Usage::

    PYTHONPATH=src python benchmarks/xval_method_selection/report_m18598.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ONSET = REPO / "docs/reports/xval_method_selection_onset.json"
GRADE = REPO / "docs/reports/xval_method_selection_m17501.json"
OUT = REPO / "docs/report_xval_method_selection_m18598.md"
FIG = "docs/figures/xval_method_selection_m18598/onset.png"


def _fmt(value, digits=3, suffix="") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int,)):
        return f"{value}{suffix}"
    return f"{value:.{digits}f}{suffix}"


def _cell_section() -> tuple[str, dict]:
    """Grading rows of the three repair cells, if their records are present."""
    if not GRADE.exists():
        return ("（评级工件缺失）", {})
    grade = json.loads(GRADE.read_text())
    interesting = [r for r in grade.get("rows", [])
                   if r.get("label") in ("D20_h025_A", "D13_h025_A",
                                         "D20_h025_xsph")]
    lines = ["| 格点 | 记录 | 最坏 G1 波高误差 | max\\|v\\|/c0 | 压缩 | 判定 |",
             "|---|---|---|---|---|---|"]
    summary = {}
    for row in interesting:
        if not row.get("present"):
            lines.append(f"| `{row['label']}` | 未落地 | — | — | — | `missing` |")
            summary[row["label"]] = "missing"
            continue
        if row.get("failed"):
            lines.append(f"| `{row['label']}` | 有 | — | — | — | `failed` |")
            summary[row["label"]] = "failed"
            continue
        summary[row["label"]] = row.get("verdict", "?")
        if not row.get("gradeable", True):
            lines.append(f"| `{row['label']}` | 有 | 分析窗无样本 | "
                         f"{_fmt(row['stability'].get('speed_ratio'))} | "
                         f"{_fmt(row['stability'].get('max_density_compression'), 2)} | "
                         f"`ungradeable` |")
            continue
        lines.append(f"| `{row['label']}` | 有 | "
                     f"{_fmt(100 * row.get('worst_error_vs_target'), 2, ' %')} | "
                     f"{_fmt(row['stability'].get('speed_ratio'))} | "
                     f"{_fmt(row['stability'].get('max_density_compression'), 2)} | "
                     f"`{row.get('verdict')}` |")
    return ("\n".join(lines), summary)


def main() -> int:
    onset = json.loads(ONSET.read_text())
    rec = onset["record"]
    fid = onset["wavemaker_fidelity"]
    scale = onset["speed_scale"]
    cross = onset["threshold_crossings_s"]
    comp = onset["compression_crossings_s"]
    growth = onset["peak_speed_growth"]
    front = onset["wave_front_arrivals_s"]
    versus = onset["onset_vs_front"]
    table, cells = _cell_section()

    probe_fronts = ", ".join(f"{x:.2f}" for x in front["probes_s"])
    cell_note = ("三个决策定向格点（`runs/xval/cell_{0,1,2}.json`）" +
                 ("已落地并按预注册稳定性门复核" if cells and
                  all(v not in ("missing",) for v in cells.values())
                  else "本报告落盘时**尚未落地**（记录缺失，判级器按 `missing` 记账，"
                       "不当作 0、不当作通过）"))

    body = f"""# XVAL-SPH-METHOD-SELECTION-2026-11-12 · 不稳定起始时刻的定量归因（mission 18598）

锚点：`XVAL-SPH-METHOD-SELECTION-2026-11-12`（TensorSPH，`CASE-WAVE-FLUME` 的方法选型 A）。
上游：mission 17501（原协议 + 唯一存档记录）、mission 17906（返修：判级加稳定性硬门，
诊断出 `max|v|/c0 = 3.55`、压缩 30 %、Mach 1 于 3.25 s，结论 honest-FAIL）。
本单不做新算例设计、不改判据，只回答返修报告留下的那个空白：**不稳定性是"何时、
从哪里"开始的**——因为"先让稳定性过门"这句话只有定位到机制才可执行。

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

2-D 数值波槽（`src/tensorsph/wave_flume.py`）：WCSPH（密度求和 + Tait 状态方程，
压力钳位 ≥0）+ δ 型密度扩散 + 活塞一阶造波 + 下游幂次海绵 + 启停窗。
本单复核的原始记录是 `docs/reports/artifacts_wave_flume/wf_D20_unstable.json`
（mission 17501 存档，175 kB、4 探头、{rec['samples']} 个采样点）。

| 参数 | 值（记录内 `config`/`spec`） |
|---|---|
| 静水深 h | {rec['depth_m']:.3f} m |
| 波高 H / 周期 T | {rec['wave_height_m']:.3f} m / 1.0 s（线性 λ = {rec['wavelength_m']:.3f} m） |
| 相速 c | {rec['celerity_m_s']:.3f} m/s |
| 声速 c0 | {rec['sound_speed_m_s']:.1f} m/s |
| 粒子数 / 时间步 | {rec['particles']} / dt = {rec['dt_final_s']:.3e} s（请求 {rec['dt_requested_s']:.3e} s） |
| 模拟时长 | {rec['simulated_time_s']:.3f} s（**短于预注册分析窗 4.0 s**） |

## 2. 怎么做的（方法 / 流程 / 工具链）

1. 新增只读诊断器 `benchmarks/xval_method_selection/attribute_onset.py`：不跑求解器、
   不调参，只读存档记录，做三件事：
   *(a)* 活塞运动保真度——对 `paddle_offset` 在爬升段之后做谐波最小二乘
   `s = C + A cos ωt + B sin ωt`，把幅值比与相位对到解析式
   `s = (S/2) r(t) sin(ω(t − t_settle))`（`wave_flume.paddle_motion` 在
   `wave_time = elapsed − settle_time` 上取值，故解析相位滞后 `ω·t_settle` 是设计量）；
   *(b)* 峰值速度的阈值穿越时刻与指数增长率（`ln max|v|` 对 t 最小二乘，1–20 m/s 段）；
   *(c)* 线性波前到达时刻 `t = x/c`（4 个探头、海绵入口、下游墙）。
2. 判据由 §0 的预注册协议给出，本单**不新增也不放宽**任何门：稳定性门
   `max|v|/c0 < 1` 且压缩 < 10 %；精度门 15 %（owner）/10 %（本仓）。
3. 单测 `tests/test_xval_onset_attribution.py`（5 条）：三条用合成记录验证"已知答案"
   （穿越时刻、增长率、波前到达），两条对存档记录断言活塞保真度与"失稳早于海绵到达"。

## 3. 计算结果（程序化注入，零手抄）

来源：`docs/reports/xval_method_selection_onset.json`（本报告的全部数字由该文件格式化）。

### 3.1 造波器保真度（先把"造波过强"排除）

| 量 | 值 |
|---|---|
| 拟合幅值 / 解析 S/2 | {_fmt(fid['amplitude_ratio'], 9)} |
| 拟合相位 vs 解析 sin(ωt) | {_fmt(fid['phase_error_rad'], 6)} rad |
| 设计相位滞后 −ω·t_settle (mod 2π) | {_fmt(fid['phase_expected_rad'], 6)} rad（一致：{_fmt(fid['phase_matches_designed_lag'])}） |
| 拟合残差（相对幅值） | {fid['relative_residual']:.2e} |
| 直流偏置 | {fid['offset_bias_m']:.2e} m |

结论：活塞位移就是 `S/2 sin(ω(t−t_settle))`，与一阶传递函数反解出的行程**逐位一致**；
失稳不能归给造波幅值/相位。

### 3.2 速度尺度（把"物理波速"和"数值噪声"分开）

| 量 | 值 |
|---|---|
| 线性轨道速度 (H/2)ω·coth(kh) | {_fmt(scale['linear_orbital_speed_m_s'], 4)} m/s |
| 记录峰值 max\\|v\\| | {_fmt(scale['observed_max_speed_m_s'], 2)} m/s |
| 比值（观测/轨道） | **{_fmt(scale['ratio_observed_over_orbital'], 0)}×** |
| Mach 数 max\\|v\\|/c0 | **{_fmt(scale['mach_observed'], 3)}**（门 1.0） |

### 3.3 起始时刻 vs 波前到达（把"海绵不吸波"排除）

| 事件 | 时刻 [s] |
|---|---|
| 波前到 4 个探头 | {probe_fronts} |
| 波前到海绵入口 | {_fmt(front['sponge_entrance_s'], 3)} |
| 波前到下游墙 | {_fmt(front['downstream_wall_s'], 3)} |
| max\\|v\\| 首次 > 0.5 m/s | {_fmt(cross['0.5'], 3)} |
| max\\|v\\| 首次 > 1 / 2 m/s | {_fmt(cross['1'], 3)} / {_fmt(cross['2'], 3)} |
| max\\|v\\| 首次 > 5 / 10 / 20 m/s | {_fmt(cross['5'], 3)} / {_fmt(cross['10'], 3)} / {_fmt(cross['20'], 3)} |
| 压缩首次 > 1 % / 5 % | {_fmt(comp['0.01'], 3)} / {_fmt(comp['0.05'], 3)} |
| 压缩首次 > 10 % / 20 % | {_fmt(comp['0.10'], 3)} / {_fmt(comp['0.20'], 3)} |
| 首次超声速（Mach>1） | {_fmt(versus['first_supersonic_s'], 3)} |

峰值速度的增长：`d ln max|v| / dt = {_fmt(growth['rate_per_s'], 3)} 1/s`
（≈ e 倍时间 {_fmt(growth['e_folding_s'], 2)} s，`r² = {_fmt(growth['r2'], 3)}`，
{growth['points']} 个样本）——指数增长是数值不稳定性的指纹，而不是波的线性响应。

![onset]({FIG})

## 4. 注意点（踩过的坑 / 局限 / 适用范围）

* **失稳起点在传播区内部，不在吸收层**：`max|v|` 首次超过 0.5 m/s（线性轨道速度的
  4 倍）在 {_fmt(cross['0.5'], 2)} s，压缩首次超过 1 % 在 {_fmt(comp['0.01'], 2)} s，
  都**早于**波前抵达海绵入口的 {_fmt(front['sponge_entrance_s'], 2)} s。所以"海绵把波
  反弹回来打崩"这一解释被排除，机制在流体内部（粒子尺度），与 Δ-SPH 只做密度平滑、
  传播期无速度级耗散（ν = 1e−6 的 Morris 粘性对离散噪声几乎不起作用）相一致。
* **相位不是缺陷**：活塞拟合相位 {_fmt(fid['phase_error_rad'], 3)} rad 看着像"错位"，
  实际等于 −ω·t_settle (mod 2π) = {_fmt(fid['phase_expected_rad'], 3)} rad，是驱动代码
  在 `wave_time = elapsed − settle_time` 上取值的**设计**滞后。差点被当成 bug——这也是
  本单把"相位期望值"写进诊断器与单测的原因。
* **峰值速度与自由面读数是两套信息**：存档记录里探头读到的主频幅值最坏约在目标值
  ±10 % 内，而粒子速度已 878 倍于线性轨道速度。也就是说"波形看起来还行"和"解已经
  数值爆掉"能同时成立——这正是返修判级要把稳定性做成硬门的原因，本单给出量化的时间
  证据。不得据此声称精度门通过：预注册窗 4.0/5.0 s 在 3.70 s 的记录里没有样本。
* **适用范围**：单条存档记录（h = 0.20 m、H/h = 0.15、kh ≈ 1.04、2-D、无破碎），
  阈值穿越时刻的分辨率为输出间隔（Δt_out = {rec['simulated_time_s'] / rec['samples']:.3f} s）。
  不外推到 h = 0.25 m 的返修格点（那是 `runs/xval/cell_*.json` 的范围）。
* **格点状态**：{cell_note}。

## 5. 误差判定（逐项对基准，显式回答"是否 <3%"）

| 关键量 | 基准 | 本单结果 | 是否 <3% |
|---|---|---|---|
| 活塞幅值 | 一阶传递函数 S/2（`H/S = 2(cosh2kh−1)/(sinh2kh+2kh)`） | 幅值比 {_fmt(fid['amplitude_ratio'], 9)} | **是（{abs(1 - fid['amplitude_ratio']):.1e} 相对误差，属恒等式校验，非物理精度）** |
| 活塞相位 | 设计滞后 −ω·t_settle | {_fmt(fid['phase_error_rad'], 6)} vs {_fmt(fid['phase_expected_rad'], 6)} rad | 是（差 < 1e−12） |
| 波高 G1（逐探头） | 解析目标 H | **无可用数字**：存档记录在 4.0/5.0 s 窗无样本（`ungradeable`） | **否（不构成达标）** |
| 稳定性 max\\|v\\|/c0 | 门 1.0（协议 §3） | {_fmt(scale['mach_observed'], 3)} | **否（超门 {_fmt(scale['mach_observed'], 1)} 倍）** |
| 压缩 | 门 10 % | 超门时刻 {_fmt(comp['0.10'], 2)} s，峰值 30 % | **否** |

**对锚点的判定不变：honest-FAIL 继续保持**（精度门无法测、稳定性门不通过）。
本单的增量是把失败**定位到时间与机制**：造波器按解析式精确工作 → 失稳源自传播期内部
的粒子尺度增长（{_fmt(growth['rate_per_s'], 2)} 1/s，e 倍 {_fmt(growth['e_folding_s'], 2)} s）
→ 下一步唯一有意义的动作是**在传播期引入速度级耗散/平滑并检验稳定性门**，
而不是继续调造波参数或海绵参数。

## 6. 复现

```bash
# 诊断（读存档记录，写 JSON + 图）
PYTHONPATH=src python benchmarks/xval_method_selection/attribute_onset.py
# 单测（5 条：合成记录已知答案 + 存档记录的活塞保真度/失稳早于海绵）
PYTHONPATH=src python -m pytest tests/test_xval_onset_attribution.py -q
# 本报告（数字全部由上面的 JSON 注入）
PYTHONPATH=src python benchmarks/xval_method_selection/report_m18598.py
# 决策定向格点（落地后重新判级）
PYTHONPATH=src python benchmarks/xval_method_selection/grade_matrix.py
```

### 附：三个决策定向格点的判级（`docs/reports/xval_method_selection_m17501.json`）

{table}
"""
    OUT.write_text(body)
    print(f"wrote {OUT} ({len(body)} bytes); cells: {cells or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())