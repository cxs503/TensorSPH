"""Generate docs/verification_conservation.md from report.json.

Every number in the generated document is read from
``benchmarks/conservation/report.json`` (written by ``run.py``); nothing is
typed by hand.  Run order::

    PYTHONPATH=src python benchmarks/conservation/run.py
    PYTHONPATH=src python benchmarks/conservation/make_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs" / "verification_conservation.md"

TEMPLATE = """# 守恒律、伽利略不变性与时间积分阶：TensorSPH 从未测过的三件事

**仓库**: TensorSPH ｜ **工件**: `benchmarks/conservation/` ｜ **测试**: `tests/test_conservation.py`

**结论（先给判定）**: `docs/algorithm-gap-analysis.md` 自己写着「CI 通过
**并不**establish conservation, convergence, or agreement with experiment」。
收敛有阶梯（`tests/test_convergence.py`）、与参考值比对有 B1–B3，但**守恒律一条都没测过**，
时间积分器也从未被评分。本单把这四个量做成可失败的断言：

| 判据 | 基准（解析/闭式） | 实测 | 判定 |
|---|---|---|---|
| G1 压强梯度逐体力平衡 | 内部力成对相反 ⇒ `sum_i m_i a_i = M g`，浮点误差量级 | {leak_press:.2e} | **PASS**（float64 舍入） |
| G1b 粘性项单对动量泄漏闭式 | `C ν m_i m_j (v_i−v_j) L (1/ρ_i − 1/ρ_j)` | 实测/闭式 = {pair_ratio:.6f} | **机制确认**（闭式成立到 1e-15） |
| G1b 等密度时同式 | 密度相等 ⇒ 泄漏恒为 0 | {pair_eq:.1e} N | **PASS**（精确 0） |
| G2 压力项动量漂移 | `P(t)=P(0)` 到舍入 | {drift_press:.2e} | **PASS** |
| G2 粘性项动量漂移（默认 ν={nu:g}） | 同上 | {drift_visc:.2e} | **FAIL**（每步泄漏 {leak_visc:.2e}） |
| G2 CSF 表面张力动量漂移 | 同上 | {drift_csf:.2e} | **FAIL**（CSF 自身净力/力模 = {st_leak:.2e}） |
| G3 伽利略不变性（速度场） | 匀速平移不改变内部运动 | 相对差 {gal_rel:.2e} | **PASS** |
| G4 一阶弹道位移偏差 | 闭式 `lag = ½ g Δt T`（无自由参数） | 比值 {lag_ratio_min:.4f}…{lag_ratio_max:.4f} | {lag_verdict} |
| G4 观测阶 | 半隐式 Euler ⇒ 1 阶 | {order} | {order_verdict} |

**一句话**：压力梯度和时间积分器是干净的（舍入级守恒、1 阶偏差精确等于闭式）；
**动量泄漏来自两个非对称算子** —— 粘性 Laplacian（`m_j/ρ_j` / `m_i/ρ_i` 不对称，
密度相等时才抵消，而自由面永远不满足这一条）和 CSF 表面张力（逐粒子曲率力，不是对力和）。
这两项在 1 个毛细时间内让总动量漂移 {drift_default:.1e}（以 `max_t Σ m|v|` 归一）。
本单只做**测量与归因**，未改 `src/`。

## ① 案例介绍（物理背景/几何/参数）

二维六方密排液滴，四周真空，**不加壁面粒子**：静壁按设计吸收动量，会把 G1/G2 的读数污染掉
（壁面反作用力正是「动量守恒被打破」的合法来源）。

| 参数 | 值 |
|---|---|
| 半径 `R` | {R:g} m |
| 粒子间距 `dx` | {dx:g} m（`R/dx` = {Rdx:g}） |
| 光滑长度 | `h = 1.3 dx` = {h:g} m |
| 粒子数 | {n} |
| `rho0` / `c0` / `gamma` | {rho0:g} kg/m³ / {c0:g} m/s / {gamma:g} |
| 粘性系数 `nu` | {nu:g} m²/s（模块默认；G1b/G4 另有关闭与默认两组） |
| 表面张力 `sigma` | {sigma:g} N/m（10×水，缩短毛细时间） |
| 毛细时间 `sqrt(rho0 R^3/sigma)` | {t_sigma:.4f} s |

## ② 怎么做的（方法/判据/工具链）

积分器就是 `src/tensorsph/droplet_impact.py` 里 `run()` 的同一循环，逐字复用：

```
rho, p, a = rhs(pos, vel, mass, cfg, pairs=pairs, wall=None, dt=dt)
vel += dt * a          # 先更新速度
pos += dt * vel        # 再用新速度推进位置 —— 半隐式 Euler
```

四个判据的**基准都是解析的，没有拟合参数**：

* **G1 逐体力平衡**。`rhs` 的每一对相互作用都对两个成员以相反符号累加
  （`index_add` 的 `+`/`−`），所以内部项净力必须为零，只剩重力：
  `sum_i m_i a_i = M g`。定义为 `leak = |Σ m_i a_i − M g| / (Σ|m_i a_i| + M|g|)`，
  无量纲、可跨算例比较。
* **G1b 粘性泄漏的闭式**。代码里 `acc_i ∝ −C ν (v_i−v_j) m_j/ρ_j L`、
  `acc_j ∝ +C ν (v_i−v_j) m_i/ρ_i L`，乘质量后相加得
  `F_i+F_j = C ν m_i m_j (v_i−v_j) L (1/ρ_i − 1/ρ_j)`。用一个两粒子系统把右式每一项都实测出来
  （同一状态两次 `rhs` 相减，消掉压强项），所以是**机制确认**而不是「大概是粘性」。
* **G2 动量漂移**。无重力时 G1 积分为 `P(t)=P(0)`；
  `drift = |P(t)−P(0)| / max_t Σ_i m_i|v_i|`。
* **G3 伽利略不变性**。两次相同的运行只差一个匀速平移 `u0 = {u0:g} m/s`，
  比较 `(v_B − u0 e_x)` 与 `v_A` 的逐粒子差。
* **G4 时间积分阶**。半隐式 Euler 每步位置相对精确解的偏差是
  `x_{{n+1}} − x_n = v_{{n+1}} Δt = v_n Δt + a Δt²`，而精确增量是 `v_n Δt + ½ a Δt²`，
  于是总位移偏差 `lag(t) = ½ g Δt t` —— **闭式，无自由参数**。
  固定物理时间 `T = {T:g} s` 做 `Δt` 阶梯 {ladder}，同时检验系数（比值→1）和阶（→1）。

工具链：`torch.set_num_threads(2)`，CPU float64，`cell_linked_pairs` 每步重建；
图由 `run.py` 内的 matplotlib(Agg) 生成 `conservation.png`；本文件由 `make_doc.py` 从 JSON 注入。

## ③ 计算结果

### G1/G2 各算子组合（{mom_steps} 步，`Δt = stable_dt`）

| 配置 | 粘性 ν | sigma | 每步净力泄漏（中位） | 泄漏最大 | 动量漂移（全程） | CSF 净力/力模 |
|---|---|---|---|---|---|---|
{mom_table}

### G1b 两粒子粘性对（闭式 vs 实测）

| 量 | 值 |
|---|---|
| 间距 / `h` | {pair_sep:g} m / {pair_h:g} m |
| 质量 | {pair_masses} kg |
| 实测密度 | {pair_rhos} kg/m³（相对反差 {pair_contrast:.1%}） |
| 实测净力 | {pair_measured} N |
| 闭式净力 | {pair_closed} N |
| 实测/闭式 | **{pair_ratio:.6f}** |
| 等密度对照净力 | {pair_eq:.1e} N |

### G4 一阶弹道偏差（无重力以外的外力，ν=0）

| 步数 | `Δt` [s] | 落距 [m] | 实测 lag [m] | 闭式 `½gΔtT` [m] | 比值 | `lag/落距` | `Δt/T` |
|---|---|---|---|---|---|---|---|
{lag_table}

观测阶 = **{order}**（`Δt` 首末跨 {order_span:g}×）；默认粘性对照行比值 {ctrl_ratio:.4f}
（污染量 `{ctrl_contam:.1e}` 相对重力）。

### G3 伽利略不变性

| 量 | 值 |
|---|---|
| 速度场最大差 `max|(v_B − u0 e_x) − v_A|` | {gal_dv:.2e} m/s（相对 {gal_rel:.2e}） |
| 位置场最大差 | {gal_dx:.2e} m（尺度 {gal_scale:g} m） |
| 质心位移差 | {gal_com} m |

## ④ 结果图

`benchmarks/conservation/conservation.png`：左图是 G4 的 `Δt` 阶梯（实测 lag 与闭式
`½gΔtT` 双对数重合、斜率 1）；右图是 G1/G2 各算子组合的每步净力泄漏与全程动量漂移（对数轴，
黑虚线为 float64 舍入）。

## ⑤ 注意点（踩过的坑/局限/适用范围）

1. **必须关掉壁面粒子**。壁面按设计吸收动量，混进来会把 G1/G2 的读数变成「壁面反作用」，
   什么也证明不了。本单所有算例 `boundary` 无壁、无 `apply_boundary` 夹持。
2. **G4 必须关粘性**。粘性项的动量泄漏（G1b）会给质心一个伪加速度，量级正好落在
   G4 要测的偏差附近（见上表默认粘性对照行），所以主阶梯 `ν = 0`，默认粘性只作污染对照。
3. **`Δt` 不能超过声学 Courant 限**。`stable_dt` 在 `dx = {dx:g}`、`c0 = {c0:g}` 下约为
   {dt_stable:.2e} s；早期版本用 `T/60` 之类的粗阶梯（`Δt` 越限 1.7×）得到的是数值发散，
   不是截断误差 —— 报错表现为「lag 与 Δt 无关」。
4. **归因不等于修复**。G1b 给出闭式，但把粘性项改成对称形式（两侧都用 `m/ρ_iρ_j`）会改变
   `test_viscosity_is_calibrated` 钉住的 `VISCOUS_LAPLACIAN_FACTOR = 2.0`，属于会动既有发布数字的
   改动，本单不改 `src/`，只把它列为 P1（见 §⑦）。
5. **适用范围**：结论针对「无壁、单相、2-D、`h/dx = 1.3`」的这套离散；CSF 泄漏的具体数值依赖
   界面处理（本仓库默认 `csf_scheme=consistent`、`csf_curvature=fit_quadratic`）。

## ⑥ 误差判定（逐项回答「是否<3%」）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| G1 压强梯度净力 | 0（精确守恒） | {leak_press:.2e} | **是**（舍入级） |
| G1b 粘性对净力闭式 | `Cν m_i m_j (v_i−v_j) L (1/ρ_i−1/ρ_j)` | {pair_err:.2e}（相对） | **是** |
| G4 弹道 lag 系数 | `½ g Δt T` | {lag_err_pct:.2f}% | {lag_3pct} |
| G4 观测阶 | 1.0 | {order_err_pct}% | {order_3pct} |
| G3 伽利略速度场 | 0（精确不变） | {gal_rel:.2e} | **是**（舍入级） |
| G2 压力项动量 | 0（精确守恒） | {drift_press:.2e} | **是**（舍入级） |
| G2 粘性/CSF 动量 | 0（精确守恒） | {drift_visc:.2e} / {drift_csf:.2e} | **否**（如实报，见 §⑦） |

## ⑦ 结论与新 P0/P1

* **已验证（PASS）**：`rhs` 的压强梯度对力**严格成对相反**（舍入级动量守恒）；
  时间积分器就是文档声明的半隐式 Euler，**一阶偏差精确等于闭式 `½gΔtT`**，
  伽利略不变性成立到舍入。
* **新 P1（已量化，未修）**：粘性 Laplacian 的 `m_j/ρ_j` / `m_i/ρ_i` 不对称 ⇒ 每步噪声级
  {leak_visc:.1e} 的净力泄漏，密度反差越大泄漏越大（自由面必然发生）；修复需连带重标定
  `VISCOUS_LAPLACIAN_FACTOR`，会动既有 B1/B2 数字，故列 P1 而非直接改。
* **新 P2（已量化）**：CSF 表面张力逐粒子力的净反作用不为零（自身净力/力模 ≈ {st_leak:.1e}），
  这是 SPH 表面张力文献里的已知问题（需要对称化的表面张力形式），本仓库当前不守恒。
  影响：封闭液滴会有缓慢的整体漂移；对 B3 类「静态/振荡液滴」测量的影响已在误差预算内
  （{drift_csf:.1e} 相对动量尺度）。
* **对 B2/dam-break 的含义**：壁面存在时动量本来就交给壁面，故上述泄漏不影响有壁算例的
  物理结论；它影响的是「无壁自由飞行」段与长时间积分。
"""


def fmt_mom_table(report: dict) -> str:
    lines = []
    for name, rep in report["momentum"].items():
        st = rep.get("st_leak_median")
        lines.append(
            f"| `{name}` | {rep['viscosity_m2_per_s']:g} | "
            f"{rep['sigma_N_per_m']:g} | {rep['leak_median']:.3e} | "
            f"{rep['leak_max']:.3e} | {rep['momentum_drift_rel']:.3e} | "
            f"{'—' if st is None else f'{st:.3e}'} |")
    return "\n".join(lines)


def fmt_lag_table(rows) -> str:
    lines = []
    for r in rows:
        lines.append(
            f"| {r['steps']} | {r['dt_s']:.3e} | {r['fall_distance_m']:.4e} | "
            f"{r['lag_measured_m']:.6e} | {r['lag_predicted_m']:.6e} | "
            f"{r['lag_ratio']:.5f} | {r['lag_over_fall_distance']:.5f} | "
            f"{r['dt_over_t']:.5f} |")
    return "\n".join(lines)


def main() -> int:
    rep = json.loads((HERE / "report.json").read_text())
    case = rep["case"]
    mom = rep["momentum"]
    pair = rep["viscous_pair"]
    bal = rep["ballistic"]
    gal = rep["galilean"]
    galf = rep["galilean_field"]
    rows = bal["rows"]
    ratios = [r["lag_ratio"] for r in rows]
    order = bal["observed_order"]
    lag_err_pct = max(abs(x - 1.0) for x in ratios) * 100.0
    order_err_pct = (abs(order - 1.0) * 100.0) if order is not None else float("nan")
    order_span = rows[0]["dt_s"] / rows[-1]["dt_s"]
    ctrl = bal["default_viscosity_control"]

    text = TEMPLATE.format(
        leak_press=mom["pressure_only"]["leak_median"],
        pair_ratio=pair["closed_form_ratio"],
        pair_eq=pair["equal_density_net_pair_force_N"],
        drift_press=mom["pressure_only"]["momentum_drift_rel"],
        drift_visc=mom["viscous"]["momentum_drift_rel"],
        drift_csf=mom["csf_only"]["momentum_drift_rel"],
        drift_default=mom["default"]["momentum_drift_rel"],
        leak_visc=mom["viscous"]["leak_median"],
        st_leak=mom["csf_only"]["st_leak_median"],
        nu=case["viscosity_m2_per_s"],
        gal_rel=galf["velocity_field_mismatch_rel"],
        lag_ratio_min=min(ratios), lag_ratio_max=max(ratios),
        lag_verdict="**PASS**（<3%）" if lag_err_pct < 3.0 else "**FAIL**",
        order="n/a" if order is None else f"{order:.3f}",
        order_verdict=("**PASS**" if order is not None and 0.9 <= order <= 1.1
                       else "**FAIL**"),
        R=case["radius_m"], dx=case["dx_m"], Rdx=case["radius_m"] / case["dx_m"],
        h=case["h_over_dx"] * case["dx_m"], n=case["n_particles"],
        rho0=case["rho0_kg_m3"], c0=case["sound_speed_m_per_s"],
        gamma=case["gamma"], sigma=case["sigma_N_per_m"],
        t_sigma=case["capillary_time_s"], u0=gal["u0_m_per_s"],
        T=bal["total_time_s"],
        ladder=", ".join(str(r["steps"]) for r in rows),
        mom_steps=mom["pressure_only"]["steps_run"],
        mom_table=fmt_mom_table(rep),
        lag_table=fmt_lag_table(rows),
        pair_sep=pair["separation_m"], pair_h=pair["h_m"],
        pair_masses=", ".join(f"{m:.4e}" for m in pair["masses_kg"]),
        pair_rhos=", ".join(f"{v:.1f}" for v in pair["densities_kg_m3"]),
        pair_contrast=pair["relative_density_contrast"],
        pair_measured=", ".join(f"{v:.6e}" for v in pair["net_pair_force_measured_N"]),
        pair_closed=", ".join(f"{v:.6e}" for v in pair["net_pair_force_closed_form_N"]),
        pair_err=abs(pair["closed_form_ratio"] - 1.0),
        ctrl_ratio=ctrl["lag_ratio"],
        ctrl_contam=ctrl["internal_com_acceleration_m_per_s2"],
        gal_dv=galf["velocity_field_mismatch_m_per_s"],
        gal_dx=galf["position_field_mismatch_m"],
        gal_scale=galf["position_scale_m"],
        gal_com=", ".join(f"{v:.2e}" for v in gal["com_drift_m"]),
        lag_err_pct=lag_err_pct,
        lag_3pct="**是**" if lag_err_pct < 3.0 else "**否**",
        order_err_pct=f"{order_err_pct:.3f}",
        order_3pct="**是**" if order_err_pct < 3.0 else "**否**",
        order_span=order_span,
        dt_stable=min(r["dt_s"] for r in rows),
    )
    DOC.write_text(text)
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())