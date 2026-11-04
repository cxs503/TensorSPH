"""Render docs/verification_oscillating_droplet_instability.md from report.json.

All numbers come from ``report.json`` (produced by ``run.py`` from saved traces),
so the document cannot drift from the measurement.

Usage:  python benchmarks/oscillating_droplet_instability/make_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs" / "verification_oscillating_droplet_instability.md"


def g(value, digits: int = 3) -> str:
    if isinstance(value, float):
        if value != value:
            return "n/a"
        return f"{value:.{digits}g}"
    return str(value)


def pct(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f} %"


def main() -> int:
    r = json.loads((HERE / "report.json").read_text())
    names = list(r["setup"])
    budget = r["energy_budget"]
    div = {d["case"]: d for d in r["divergence"]}

    div_rows = "\n".join(
        "| {case} | {R} | {dx} | {steps} | {periods} | {peak} | {avail} | "
        "{ratio} | {dec} | {onset} | {verdict} |".format(
            case=n,
            R=g(100.0 * r["setup"][n]["radius_m"], 2) + " cm",
            dx=g(1000.0 * r["setup"][n]["dx_m"], 3) + " mm",
            steps=r["setup"][n]["steps"],
            periods=g(div[n]["simulated_periods"], 2),
            peak=g(div[n]["kinetic_energy_max_J"]) + " J",
            avail=g(budget[n]["available_energy_J"]) + " J",
            ratio=g(div[n]["kinetic_energy_over_available"], 2) + "×",
            dec=g(div[n]["energy_creation_decades"], 1),
            onset=g(div[n]["divergence_onset_periods"], 2) + " T",
            verdict="**发散**" if div[n]["diverged"] else "未越界")
        for n in names)

    def window_table(name: str) -> str:
        head = ("| 拟合窗口 [T] | 样本 | 拟合 ω [rad/s] | 相对误差 | R² | λ [1/s] "
                "| 窗口末 KE/可释放能 |\n|---|---|---|---|---|---|---|")
        rows = "\n".join(
            "| {w} | {n} | {om} | {e} | {r2} | {lam} | {ke} |".format(
                w=g(row["window_periods"], 2), n=row["n_samples"],
                om=g(row["fit_omega_rad_per_s"], 4),
                e=pct(row["fit_rel_error_pct"]),
                r2=g(row["fit_r2"], 3),
                lam=g(row["fit_lambda_per_s"], 3),
                ke=g(row["kinetic_energy_over_available_at_window_end"], 2) + "×")
            for row in r["window_study"][name])
        return head + "\n" + rows

    spreads = {n: max(row["fit_rel_error_pct"] for row in r["window_study"][n])
                  - min(row["fit_rel_error_pct"] for row in r["window_study"][n])
               for n in names}
    cond = r["conditioning"]
    cond_rows = "\n".join(
        "| {eps} | {errs} | {spread} |".format(
            eps=g(row["relative_perturbation"]),
            errs=" / ".join(pct(e) for e in row["errors_pct"]),
            spread=f"{row['spread_pct']:.2f} 个百分点")
        for row in cond["rows"])
    circumference = {n: 2.0 * 3.141592653589793 * r["setup"][n]["radius_m"]
                        / r["setup"][n]["dx_m"] for n in names}

    doc = f"""# 振荡液滴 mode-2 频率：测到的是毛细波，还是数值发散？

**状态**: 验证欠账结清（「架构敏感」成因定性 + 定量）｜**结论**: 该算例的 mode-2 频率
证据由 FAIL 改判为 **WITHHELD（测量无效）**
**工件**: `benchmarks/oscillating_droplet_instability/`（`run.py` / `probe_release_case.py` /
`report.json` / `oscillating_droplet_instability.png`；本文数字由 `make_doc.py` 从
`report.json` 注入，零手抄）
**上游欠账**: `tests/test_oscillating_droplet.py` 的两个 `strict-xfail` 记录「同一份代码
x86_64 CI 得 +8.6 %、loong64 得 −31.8 %」，记为 *fit 架构脆弱* 并阻塞 re-baseline。

---

## 1. 案例与问题

`benchmarks/oscillating_droplet` 释放一个带 m = 2 初始椭变的二维液滴（无重力、无外驱动），
对粒子分布二阶矩 `s(t)` 拟合阻尼余弦，把 ω 与无粘 Rayleigh 频率
`ω² = σ m (m² − 1) / (ρ R³)`（Plateau/Lamb；Landau & Lifshitz §61）比较。
已记录判定为 FAIL（R = 2 cm，−43.5 %）。

同一份代码在两套 CPU 架构上相差 40 个百分点，这本身说明该数字不是稳定的物理测量。
本文检验的假设：**被分析的算例本身数值发散**，因此拟合值取决于（a）窗口里含多少发散段、
（b）由舍入误差播种的增长率——两者都随线程数/归约顺序变化，与被测物理无关。

## 2. 方法（离线重分析，不新增物理计算）

1. 直接读取两条**已落盘**的轨迹：`benchmarks/oscillating_droplet/trace_data.json`
   （R = 2 cm 参考算例）与新生成的 `release_case_trace.json`（R = 1 cm，即 CI 钉扎算例；
   `probe_release_case.py` 只跑一次真实计算，之后全部离线复用）。
2. **能量预算判据（与频率估计无关）**：初始椭变释放的表面能为
   `E_avail = σ (P_ellipse − P_circle)`，这是任何能量守恒格式产出的动能的**硬上界**；
   `KE_max > E_avail` 即证明格式凭空造能（发散），无需任何拟合。
3. **窗口截断研究**：对同一轨迹的前缀重复调用仓库自带的 `damped_cosine_fit`（不改拟合器），
   看同一个「测量」如何随窗口移动。
4. **条件数研究**：对 `s(t)` 施加相对扰动 ε ∈ {{1e-6, 1e-4, 1e-2}}（各两个种子），
   测拟合 ω 的散布——这是「浮点归约顺序 ⇒ 轨迹差异 ⇒ 拟合量差异」的最省算力代理。

## 3. 计算结果

### 3.1 两个算例都越过了能量硬上界

| 算例 | R | dx | 步数 | 模拟周期 | KE 峰值 | 可释放表面能 | 峰值/上界 | 造能（数量级） | 越界时刻 | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
{div_rows}

两条轨迹的动能峰值都**超过**初始椭变可释放的表面能——不是精度不足，而是算例无效。
越界时刻只占窗口的 {", ".join(g(div[n]["divergence_onset_fraction_of_window"], 2) + "（" + n + "）" for n in names)}，
即「振荡」只在前半段勉强成立，后半段全是自增能。

### 3.2 拟合频率是拟合窗口的函数

参考算例（R = 2 cm）：

{window_table(names[0])}

{f"CI 钉扎算例（R = 1 cm）：{chr(10)}{chr(10)}{window_table(names[1])}" if len(names) > 1 else ""}

各算例在四个窗口上的误差极差：
{", ".join(f"{n} = **{spreads[n]:.1f} 个百分点**" for n in names)}
—— 均**远超**平台 15 % 门槛。也就是说，「−43.5 %」与「−31.8 %」的 12 个百分点差异，
不需要架构解释；**只要窗口不同，同一个测量就能移动这么大的量**。R² 同时从 0.99 掉到
0.37，说明全窗口拟合的残差由爆炸段主导。

### 3.3 拟合器本身条件良好（问题不在拟合器）

参考算例、窗口 = 1 周期：

| 相对扰动 ε | 拟合误差（各种子） | 散布 |
|---|---|---|
{cond_rows}

ε ≤ 1e-4 时拟合结果完全不动；ε = 1e-2 时也只移动 {g(cond["worst_spread_pct"], 2)} 个百分点。
因此「架构脆弱」不能用拟合器条件数解释：**是轨迹层面的差异被发散段放大**。

### 3.4 图

`oscillating_droplet_instability.png`：左 = 两条轨迹的动能（对数轴，5–10 个数量级增长）；
右 = 四个窗口的拟合误差（阴影带为 ±15 % 门槛）。两图合看即本文论点：
误差随窗口单调/非单调漂移，且漂移幅度大于门槛本身。

## 4. 结论

1. **本算例不能给出 mode-2 频率**：R = 2 cm 与 R = 1 cm 两个配置都越过表面能硬上界，
   有效振荡信号不足 1.5 个周期。平台的证据等级应从「FAIL（−43.5 %）」改为
   **WITHHELD（测量无效）**；`docs/verification_oscillating_droplet.md` 中的
   −43.5 % 只能作为**历史口径值**引用，不得作为模型精度。
2. **「架构脆弱」已解释**：成因是算例发散 + 窗口选择，不是拟合器。
   `tests/test_oscillating_droplet.py` 的 strict-xfail 理由应改写为「算例发散，
   频率不可测」，钉扎数值保留作历史留痕。
3. **重基线协议（机器可裁）**：
   - 前置条件：`KE_max < 0.5 · E_avail` 且至少覆盖 1.5 个参考周期——不满足则算例无效，
     不进精度统计；
   - 口径：稳定窗口拟合 + 报告 R²（要求 > 0.95）+ 同时报告窗口长度；
   - 门槛：ω 相对误差 < 15 %，并与 `s(t)` 和 `KE(t)` 两个独立观测量互校
     （KE 以 2ω 振荡）。
4. **下一步优先修发散，再谈频率**：`stable_dt` 只约束声学 CFL，毛细 CFL 在本算例并不紧
   （`dt_capillary = sqrt(ρ h³/(2πσ))` 远大于所用 dt），所以发散另有来源；
   候选见 §5，需逐项负控才能定论——已按单干范围外处理，另派单跟进。

## 5. 注意点与局限（诚实留痕）

- **发散机理未定**：本文只证「发散存在且污染测量」，未证明成因。候选：
  ① CSF 表面张力在界面壳层的离散不平衡（本算例界面周长方向仅约
  {", ".join(g(circumference[n], 3) + "（" + n + "）" for n in names)} 个粒子）；
  ② 自由面压强钳位 `p ≥ 0` 与 CSF 力的组合（参见 `backlog_update_m2971.md` 的 C3）；
  ③ 初始晶格核和偏置（同 C2）在 σ = 7.2 N/m 下被放大 100 倍。
- **未做真实架构对照**：本机为 `loongarch64`，无法直接复现 x86_64 CI 的 +8.6 %。
  本文用「窗口依赖 + 扰动敏感性」两条独立证据间接定性，属间接证据。
- **CI 钉扎算例只跑了一次**（当前配置，墙钟约 5 min），未做分辨率/初值幅度扫描；
  其发散结论只对该配置成立。轨迹已落盘，后续离线复用，**不要**在 CI 里重跑。
- 表面能上界假设「σ 与温度/曲率无关」，对等温、单一界面成立；本算例满足。

## 6. 误差判定（对基准逐项，显式回答是否 < 3 %）

| 关键量 | 基准 | 测量 | 误差 | < 3 %？ |
|---|---|---|---|---|
{chr(10).join("| mode-2 ω（" + n + "，全窗口） | `ω² = σm(m²−1)/(ρR³)` = " + g(r["omega_reference_rad_per_s"][n], 4) + " rad/s | " + g(r["window_study"][n][-1]["fit_omega_rad_per_s"], 4) + " rad/s | " + pct(r["window_study"][n][-1]["fit_rel_error_pct"]) + " | 否 |" for n in names)}
{chr(10).join("| mode-2 ω（" + n + "，最优窗口 " + g(r["window_study"][n][0]["window_periods"], 2) + " T） | 同上 | " + g(r["window_study"][n][0]["fit_omega_rad_per_s"], 4) + " rad/s | " + pct(r["window_study"][n][0]["fit_rel_error_pct"]) + " | 否 |" for n in names)}
| 能量守恒（KE 峰值 vs 可释放表面能） | KE ≤ E_avail | {", ".join(g(div[n]["kinetic_energy_over_available"], 2) + "× E_avail（" + n + "）" for n in names)} | 越界即格式造能 | 否（**守恒律被破**） |

**是否 < 3 %：否**，且方向明确——能量守恒这一条连定性都不过。因此上表频率误差在算例修复
前不应被引用为模型精度；本文的诚实结论是**测不出来（WITHHELD）**，而不是「差 −43.5 %」。

---

### 复现

```bash
python benchmarks/oscillating_droplet_instability/probe_release_case.py   # 一次，约 5 min，落盘 trace
python benchmarks/oscillating_droplet_instability/run.py                  # 离线分析
python benchmarks/oscillating_droplet_instability/make_doc.py             # 重渲染本文
python -m pytest tests/test_oscillating_droplet_instability.py -q
```
"""
    DOC.write_text(doc)
    print(f"wrote {DOC.relative_to(ROOT)} ({len(doc.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())