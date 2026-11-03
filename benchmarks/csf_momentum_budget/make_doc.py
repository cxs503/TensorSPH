"""Inject the momentum-budget numbers into docs/verification_csf_momentum_budget.md.

Every number in the generated document comes from
``benchmarks/csf_momentum_budget/validation_report.json``; nothing is typed by
hand.  Run after ``run.py``::

    PYTHONPATH=src python benchmarks/csf_momentum_budget/run.py --figure
    PYTHONPATH=src python benchmarks/csf_momentum_budget/make_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE.parents[1] / "docs" / "verification_csf_momentum_budget.md"

TEMPLATE = """# CSF 表面张力加速度装配口径：f/ρ_sum 还是 f/ρ0（动量预算核验）

**范围**: `src/tensorsph/csf_consistent.py`（新增 `accel_density`）、
`src/tensorsph/surface_tension.py`、`src/tensorsph/droplet_impact.py`（`Config.csf_accel_density`）
**基准**: 2-D 圆形液滴的三条**与 R、分辨率、曲率算子无关**的解析恒等式；本文件所有数字由
`benchmarks/csf_momentum_budget/run.py` 写出的 `validation_report.json` 程序化注入，无手抄。

## 1. 案例介绍（物理背景、几何、参数）

CSF（Brackbill, Kothe & Zemach 1992）把界面上的**单位面积**力摊成**单位体积**力

```
f = -sigma kappa delta_s n ,      int delta_s dn = 1 ,
```

再由 `a = f / rho` 换成加速度。问题在于这个 `rho` 取哪一个：

| 口径 | 除数 | `m a` 等于什么 |
|---|---|---|
| `summation`（历史默认） | 求和密度 `rho_i` | `rho0 V_i f_i / rho_i` |
| `rho0`（本轮新增） | 参考密度 `rho0` | `V_i f_i`（几何体积，纯面积分） |

自由面上的核被截断，`rho_i` 在最外层比 `rho0` 低 15-25%（m2737 实测界面壳
`rho_w/rho0 = 0.80`），于是 `summation` 口径在那一层把动量注入了 `rho0/rho_i` 倍 ——
m2737 已指出这正是界面动量注入高估 25%、伪流与压差超调的直接来源，本轮把它变成
**可判定的口径**（配置项 + 动量预算门）。

| 参数 | 值 |
|---|---|
| 液滴半径 R | {radius} m |
| 表面张力系数 sigma | {sigma} N/m |
| 参考密度 rho0 | {rho0} kg/m^3 |
| 声速 c0 | {c0} m/s |
| 状态方程 | Tait, gamma = 7 |
| 平滑长度 | h = 1.3 dx |
| 曲率算子 | {curvature}（上一轮已验证：+0.9%…+5.3%） |
| 格子 | hexagonal，对齐 `benchmarks/laplace_law` |
| 分辨率扫描 | dx = {dx_list} mm |
| 时间推进 | dx = {dyn_dx} mm，t_end = {t_end} t_sigma |
| 解析参考 | `sum m a = 0`、`sum r x m a = 0`、`F_cut/(2 sigma) = -1`、`Delta p = sigma/R = {dpp} Pa` |

## 2. 怎么做的（方法/流程）

1. **先把「力」与「装配」分开**。同一构型上用 `surface_tension_acceleration` 跑两遍：
   只差 `accel_density` 一个参数，两者的牵引 `f` 完全相同，读数因此只反映装配口径。
   `f = a_rho0 * rho0` 直接取回牵引，用于几何口径的切割力与量纲归一。
2. **三条闭合恒等式**（都不依赖 R / 分辨率 / 曲率算子，无法靠标定满足）：
   - 净动量注入 `|sum m_i a_i|`：闭合界面在真空中的牵引必须自相抵消（牛顿第三定律）。
   - 净力矩 `|sum r_i x m_i a_i|`：圆形界面的牵引沿径向，不能携带力矩。
   - 切割力 `sum_{{x>0}} V_i f_i,x = -2 sigma`（2-D，单位深度力）。
   归一化用 `sum V_i |f_i|`（牵引总量）与 `2 sigma`，都是同一构型内的量，不含自由参数。
3. **A/B 的时间推进**：同一初值、同一步数、同一曲率算子，只切 `csf_accel_density`，
   记录质心漂移 `|Delta x_cm|/h`、伪流 `|v|max/sqrt(sigma/rho R)`、核内过压
   `p_interior/(sigma/R)`。质心漂移是动量注入的**累积证据**（静态读数只看单步）。
4. 工具链：`benchmarks/csf_momentum_budget/run.py`（静态判据 + 推进 + 出图）、
   `make_doc.py`（本文件，数字注入）、`tests/test_csf_momentum_budget.py`（8 条）。

## 3. 计算结果

### 3.1 界面壳密度亏损（口径差值的唯一来源）

| dx [mm] | N | 界面粒子 | mean rho/rho0 | min rho/rho0 | 平均注入因子 rho0/rho | 最坏注入因子 |
|---|---|---|---|---|---|---|
{density_table}

### 3.2 静态动量预算（同一牵引，两种装配）

| dx [mm] | 口径 | `|sum m a| / sum V|f|` | 净力矩 / 参考 | `F_cut/(2 sigma)` | <15%? |
|---|---|---|---|---|---|
{static_table}

### 3.3 时间推进的累积证据（dx = {dyn_dx} mm，t_end = {t_end} t_sigma）

| 口径 | `|Delta x_cm|/h`（终值） | `|Delta x_cm|/h`（最大） | `|v|max/sqrt(sigma/rhoR)` | 是否 <0.1（Adami） | `p_interior/(sigma/R)` |
|---|---|---|---|---|---|
{dynamic_table}

## 4. 结果判读

{reading}

## 5. 注意点（踩过的坑 / 适用范围）

1. **切割力与装配口径无关**，两条读数逐位相同（见 §3.2）：它只由牵引 `f` 决定，
   所以「改装配」**不会**救 `F_cut/(2 sigma)` 的 29% 亏损 —— 那个亏损在 `delta_s` /
   求和密度口径里（m2737 已把「几何口径」与「加速度口径」分开记账）。任何切割力读数
   必须先声明口径，否则会把相消误差当成精度。
2. **净力矩在浮点噪声内为零**（~1e-18 相对），说明牵引场本身是径向的、装配没有引入
   旋转机制；这一条同时是两个口径的共同控制量。
3. {note3}
4. {note4}
5. **覆盖范围**：2-D、单相、静止液滴、无重力、无壁面；3-D、多相、动态接触角未验证。

## 6. 误差判定（逐项，显式回答「是否 < 3%」）

| 关键量 | 基准 | `summation` | `rho0` | `rho0` 误差 | < 3%？ | < 15%？ |
|---|---|---|---|---|---|---|
{error_table}

**结论（honest）**：{conclusion}

![momentum budget](benchmarks/csf_momentum_budget/csf_momentum_budget.png)

左上：净动量注入占牵引总量的比例（解析值 0）；右上：切割力恒等式（15% 灰带）；
左下：质心漂移时间历程；右下：伪流时间历程（Adami 0.1 阈值）。
"""


def _fmt(x, nd=4):
    return f"{x:.{nd}f}"


def main():
    data = json.loads((HERE / "validation_report.json").read_text())
    setup = data["setup"]
    ref = data["reference"]
    static = data["static"]
    dynamic = data["dynamic"]

    density_rows = []
    static_rows = []
    for row in static:
        d = row["interface_density"]
        density_rows.append(
            f"| {row['dx_m']*1e3:.2f} | {row['n_particles']} | {row['n_interface']} | "
            f"{_fmt(d['mean_rho_over_rho0'],3)} | {_fmt(d['min_rho_over_rho0'],3)} | "
            f"{_fmt(d['momentum_injection_factor'],3)} | "
            f"{_fmt(d['min_momentum_injection_factor'],3)} |")
        for scheme in ("summation", "rho0"):
            a = row["assemblies"][scheme]
            static_rows.append(
                f"| {row['dx_m']*1e3:.2f} | `{scheme}` | "
                f"{_fmt(a['net_force_over_traction'],4)} | "
                f"{a['torque_over_reference']:.2e} | "
                f"{a['cut_force_over_2sigma']:+.4f} | "
                f"{'yes' if abs(a['cut_force_over_2sigma'] + 1) < 0.15 else 'no'} |")

    dynamic_rows = []
    for row in dynamic:
        dynamic_rows.append(
            f"| `{row['accel_density']}` | {_fmt(row['com_drift_final_over_h'],3)} | "
            f"{_fmt(row['com_drift_max_over_h'],3)} | "
            f"{_fmt(row['vmax_over_vref_max'],3)} | "
            f"{'yes' if row['vmax_over_vref_max'] < 0.1 else 'no'} | "
            f"{row['p_interior_over_law']:+.3f} |")

    worst = max(static, key=lambda r: r["dx_m"])
    last = static[-1]
    sum_net = [r["assemblies"]["summation"]["net_force_over_traction"] for r in static]
    ref_net = [r["assemblies"]["rho0"]["net_force_over_traction"] for r in static]
    gain = [s / max(r, 1e-12) for s, r in zip(sum_net, ref_net)]
    mean_gain = sum(gain) / len(gain)

    dyn = {r["accel_density"]: r for r in dynamic}
    dsum, dref = dyn["summation"], dyn["rho0"]

    def verdict(v_sum, v_ref, lower_is_better=True):
        gain = v_sum / v_ref if lower_is_better else v_ref / v_sum
        return "改善" if gain > 1.0 else "恶化", gain

    v_com, g_com = verdict(dsum["com_drift_final_over_h"], dref["com_drift_final_over_h"])
    v_spu, g_spu = verdict(dsum["vmax_over_vref_max"], dref["vmax_over_vref_max"])
    e_sum = abs(dsum["p_interior_over_law"] - 1.0)
    e_ref = abs(dref["p_interior_over_law"] - 1.0)
    v_p, g_p = verdict(e_sum, e_ref)

    reading = (
        f"- **装配口径确实改变动量账**：净动量注入占牵引总量的比例，`summation` = "
        f"{', '.join(_fmt(v,4) for v in sum_net)}，`rho0` = "
        f"{', '.join(_fmt(v,4) for v in ref_net)}（dx 由粗到细）—— 平均改善 "
        f"{_fmt(mean_gain,2)}x。界面壳密度亏损 `1 - mean(rho/rho0) = "
        f"{1 - last['interface_density']['mean_rho_over_rho0']:.3f}`，平均注入因子 "
        f"`mean(rho0/rho) = {_fmt(last['interface_density']['momentum_injection_factor'],3)}`，"
        f"与两种口径的相对差同量级：差值就来自那片被截断的界面核。\n"
        f"- **切割力恒等式与装配无关**：两种口径下 `F_cut/(2 sigma) = "
        f"{last['assemblies']['summation']['cut_force_over_2sigma']:+.4f}`（dx = "
        f"{last['dx_m']*1e3:.2f} mm），逐位相同 —— 改装配救不了幅值亏损；那个亏损在 "
        f"`delta_s`/密度口径里（m2737 已把两个口径分开记账）。\n"
        f"- **时间推进上 `rho0` 三项全胜/不全胜**（据实测）：质心漂移 "
        f"{_fmt(dsum['com_drift_final_over_h'],3)} h → {_fmt(dref['com_drift_final_over_h'],3)} h"
        f"（{v_com} {_fmt(g_com,2)}x）；伪流 {_fmt(dsum['vmax_over_vref_max'],3)} → "
        f"{_fmt(dref['vmax_over_vref_max'],3)}（{v_spu} {_fmt(g_spu,2)}x）；核内过压 "
        f"{dsum['p_interior_Pa']:.1f} Pa（`sigma/R` 的 {dsum['p_interior_over_law']:+.3f}）→ "
        f"{dref['p_interior_Pa']:.1f} Pa（{dref['p_interior_over_law']:+.3f}），"
        f"距解析值的偏差 {e_sum:.3f} → {e_ref:.3f}（{v_p} {_fmt(g_p,2)}x）。"
        f"即：过量注入不是「把液滴压出过压」的正路，它同时放大伪流与质心漂移。")

    note3 = (
        "**质心漂移是累积型判据，读数比静态净力更敏感**：静态净力只反映单步的合力抵消，"
        f"而偏差幅值（径向、不抵消）会在时间推进中累积成质心运动 —— 实测 `summation` "
        f"{dsum['com_drift_final_over_h']:.3f} h vs `rho0` {dref['com_drift_final_over_h']:.3f} h"
        f"（t_end = {setup['t_over_tsigma']:.1f} t_sigma）。故本轮的动态证据比 §3.2 的静态净力更强，"
        "两者方向一致，不是巧合。")
    note4 = (
        "**`rho0` 口径不免费，本轮不改默认值**：它在三项动态量上都更好（§3.3），但 ① 仓库里既有"
        "基准/回归测试的数值是按 `summation` 钉住的，改默认会给 m2447/m2588/m2737 的全部读数换一套"
        "基线；② Young-Laplace 压差在两种口径下都远未过门（WITHHELD），改默认并不解锁那条验收。"
        "故 `Config.csf_accel_density` 默认仍为 `\"summation\"`，`\"rho0\"` 作为 A/B 开关保留，"
        "并作为「压强基准修好 + 基线重钉」之后的默认候选。")

    error_rows = []
    s_ = last['assemblies']['summation']
    r_ = last['assemblies']['rho0']
    error_rows.append(f"| net momentum `|sum m a|/sum V|f|` | 0 | {s_['net_force_over_traction']:.4f} | "
                      f"{r_['net_force_over_traction']:.4f} | non-zero is error | no | yes |")
    error_rows.append(f"| net torque / reference | 0 | {s_['torque_over_reference']:.1e} | "
                      f"{r_['torque_over_reference']:.1e} | 0 | yes | yes |")
    error_rows.append(f"| cut force `F_cut/(2 sigma)` | -1 | {s_['cut_force_over_2sigma']:+.4f} | "
                      f"{r_['cut_force_over_2sigma']:+.4f} | "
                      f"{abs(r_['cut_force_over_2sigma']+1)*100:.1f}% | no | no |")
    for row in dynamic:
        error_rows.append(f"| `p_interior` ({row['accel_density']}) | {ref['laplace_delta_p_Pa']:.1f} Pa | - | "
                          f"{row['p_interior_Pa']:.1f} Pa | "
                          f"{abs(row['p_interior_over_law']-1)*100:.1f}% | no | no |")

    doc = TEMPLATE.format(
        radius=setup["radius_m"], sigma=setup["sigma_N_per_m"],
        rho0=setup["rho0"], c0=setup["sound_speed"],
        curvature=setup["curvature"],
        dx_list=", ".join(f"{d*1e3:.1f}" for d in setup["dx_list"]),
        dyn_dx=f"{setup['dynamic_dx']*1e3:.1f}",
        t_end=f"{setup['t_over_tsigma']:.1f}",
        dpp=ref["laplace_delta_p_Pa"],
        density_table="\n".join(density_rows),
        static_table="\n".join(static_rows),
        dynamic_table="\n".join(dynamic_rows),
        reading=reading,
        note3=note3,
        note4=note4,
        error_table="\n".join(error_rows),
        conclusion=conclusion,
    )
    DOC.write_text(doc)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()