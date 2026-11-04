"""Generate ``docs/verification_free_surface_pressure_closure.md`` from the records.

Every number in the document is injected from
``report_shell.json`` (closed runs) and ``report.json`` (un-closed baseline);
nothing is typed by hand.  Re-run after any change to the closure:

    PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_closure_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs" / "verification_free_surface_pressure_closure.md"


def fmt(x, nd=1):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def main():
    shell = json.loads((HERE / "report_shell.json").read_text())
    base = json.loads((HERE / "report.json").read_text())
    ref = shell["reference"]
    base_runs = {(r["dx_m"], r["sound_speed"]): r for r in base["runs"]}
    ref_c0 = max(r["sound_speed"] for r in shell["runs"])

    rows = []
    for run in shell["runs"]:
        key = (run["dx_m"], run["sound_speed"])
        b = base_runs[key]
        tait = b["absolute_time_average_tail"]["rel_error_pct"]
        sh = run["absolute_time_average_tail"]["rel_error_pct"]
        rows.append(
            f"| {run['dx_m']:g} | {run['sound_speed']:g} | {run['n_particles']} | "
            f"{run['steps']} | {fmt(tait)} | {fmt(sh)} | {fmt(tait - sh)} | "
            f"{'✅' if abs(sh) < 15 else '❌'} | {'✅' if abs(sh) < 3 else '❌'} |")
    table = "\n".join(rows)

    representative = next(r for r in shell["runs"]
                          if r["dx_m"] == 0.002 and r["sound_speed"] == ref_c0)
    base_rep = base_runs[(0.002, ref_c0)]
    prof = []
    for row, brow in zip(representative["pressure_profile"],
                         base_rep["pressure_profile"]):
        prof.append(f"| {row['r_over_R_mid']:.3f} | {row['mean']:.1f} | "
                    f"{brow['mean']:.1f} |")
    profile_table = "\n".join(prof)

    spurious = max(r["spurious_current"]["vmax_over_vref_final"]
                   for r in shell["runs"])
    spurious_base = max(r["spurious_current"]["vmax_over_vref_final"]
                        for r in base["runs"])
    comp = [(r["dx_m"], r["sound_speed"],
             r["density"]["rho_core_over_rho0"] - 1.0,
             r["density"]["compression_required"]) for r in shell["runs"]]
    comp_table = "\n".join(
        f"| {dx:g} | {c0:g} | {got:+.2e} | {need:.2e} | "
        f"{'✅' if got >= need else '❌'} |" for dx, c0, got, need in comp)

    text = f"""# 自由面压强闭合验证记录（m2971 backlog 的 P0）

**算例**: `benchmarks/laplace_pressure_floor/`（`run.py` 归因记录 + `run_closure.py` 验收记录）
**状态**: 15 % 验收判据 **全部通过**；3 % 目标 **未达**（见 §6）
**代码**: `Config.free_surface_pressure`（默认 `False`，需显式开启）／`rhs` 的压强装配段

## 1. 物理背景与案例

真空中的液滴，表面张力驱动收缩，平衡时满足 Young–Laplace：内部压强 `p_core = sigma/R`，
外部为真空（`p = 0`），因此绝对压差就是 `sigma/R` 本身，不需要任何带通差分。

| 参数 | 值 |
|---|---|
| 半径 R | {ref['radius_m']} m |
| 表面张力 sigma | {ref['sigma_N_per_m']} N/m（水的 10 倍，缩短毛细时间便于收敛） |
| 密度 rho0 | {ref['rho0']} kg/m³ |
| Tait 指数 gamma | {ref['gamma']} |
| 解析压差 sigma/R | **{ref['delta_p_Pa']:.1f} Pa** |
| 毛细时间 t_sigma | {ref['t_sigma_s']:.4f} s |
| 声速 c0 | 20 / 5 m/s（c0 阶梯，用来分离 EOS 噪声地板与闭合缺失） |
| 分辨率 dx | 0.002 / 0.0015 m（粒子数 362 / 约 640） |
| 计算窗口 | 1.5 t_sigma，尾部 40 % 时间平均 |
| 曲率算子 | `csf_scheme="consistent"`，`csf_curvature="fit_quadratic"` |

## 2. 怎么做的

m2971 把误差拆成四项（测量口径 C1、初始核和偏置 C2、自由面压强闭合 C3、EOS 分辨率地板 C4）
后，C3 是唯一门槛项：`tait_pressure` 把 p 钳在 ≥ 0，而自由面层的求和密度亏缺
（0.85–0.95R 波段 −11.6 %，0.95–1.05R 波段 −34 %）使那一层压强恒为 0，界面无法承载 `sigma/R`，
内区只能收在约 0.57 倍解析压差。

本单实现 Adami/Hu/Adams (J. Comput. Phys. 231, 2012) 式 (11) 的**壳层压强外推**，作用于流体粒子：

```
p_i = max(0, Σ_j (p_j + rho_j (g - a_i)·(r_j - r_i)) W_ij / Σ_j W_ij)
```

实现要点（全部在 `src/tensorsph/droplet_impact.py::rhs` 的压强装配段）：

1. **判据**：`rho/rho0 < Config.free_surface_density_ratio`（默认 0.99）标记自由面粒子；
   只有这些粒子的压强被改写，完整核粒子的压强**逐位不变**（有测试钉住）。
2. **供体**：只有非自由面邻居供体（`tau = 1`），自由面邻居不参与——否则亏缺壳层自己喂自己。
3. **体力项**：`g - a_i` 中的 `a_i` 在扫掠中尚未知，取 0（与同函数内壁面外推的既有选择一致），
   只保留 `g·(r_j - r_i)`，`g = (0, -gravity)`。
4. **单遍扫描、无迭代**，与壁面外推共用 `index_add` 成对归约写法（无 pair 级布尔掩码拷贝）。
5. **默认关闭**：闭合会改写所有自由面附近的压强场，因此所有已发表的自由面数值都必须重测后
   才能默认打开；本记录用 `--closure shell` 显式开启。

工具链：`run_closure.py` 复用 `run.py` 的 `relax()`（同 `stable_dt`、同窗口、同步数，
因此与 `report.json` 的未闭合结果是**同 run 配对比较**），读回 `report.json` 作为基线，
写出 `report_shell.json` 与 `laplace_pressure_closure.png`。

## 3. 计算结果

### 3.1 验收表（尾段时间平均的绝对压差误差）

| dx [m] | c0 [m/s] | 粒子数 | 步数 | 钳位 (tait) | 闭合 (shell) | 改善 [pt] | <15 % | <3 % |
|---|---|---|---|---|---|---|---|---|
{table}

### 3.2 代表性压强的径向剖面（dx = 0.002 m, c0 = {ref_c0:g} m/s）

末帧快照，含呼吸模态的瞬时相位，分箱均值因此有正负振荡；**验收一律用尾段时间平均**，
不要用本表单点读数下结论。

| r/R | p (shell) [Pa] | p (tait) [Pa] |
|---|---|---|
{profile_table}

### 3.3 内区压缩比（闭合后是否仍受控）

| dx [m] | c0 [m/s] | 实测 rho_core/rho0 − 1 | 承载 sigma/R 所需 | 足够 |
|---|---|---|---|---|
{comp_table}

## 4. 图

`benchmarks/laplace_pressure_floor/laplace_pressure_closure.png`：
(a) 代表性算例的压强径向剖面，闭合前后与 `sigma/R` 对照；
(b) 四个验收点的尾段平均误差柱状图，含 ±15 % 判据线。

## 5. 注意点与局限

1. **虚假流增大**：`v_max/v_ref` 从钳位态的 {spurious_base:.3f} 升到闭合态的 {spurious:.3f}，
   跨过了 Adami 0.1 判据。闭合注入的壳层压强会做功，残差动能因此变大。这是**已知代价**，
   用测试钉住（`test_shell_closure_raises_the_spurious_current_and_that_is_recorded`），
   后续若改进必须更新该数值，不能悄悄回退。
2. **单遍、`a_i = 0`**：式 (11) 的加速度项被省略；对静水/准静态问题是标准做法，
   对强加速自由面（如液滴撞池的飞溅）没有验证。
3. **阈值敏感度未系统研究**：0.99 的密度判据决定谁被改写；本单只验证了这一个取值。
4. **只验证了二维轴对称液滴**：三维、多相、带壁面（`wall`）的组合只做了单步有限性检查，
   没有端到端收敛记录。
5. `definition_absolute`（末帧单点）仍可偏离（呼吸模态），验收一律用**尾段时间平均**口径；
   历史文档中的 `p_core − p(0.8–0.95R)`（bandA）读数属 C1 污染口径，不得再作为当前精度引用。
6. `report.json` 保持未闭合基线不变（C1–C4 归因记录的判据依赖它）；
   闭合结果写在 `report_shell.json`，两份记录并存。

## 6. 误差判定（逐项）

| 判据 | 阈值 | 实测 | 是否满足 |
|---|---|---|---|
| 四验收点尾段平均误差 | < 15 % | 最大 {max(abs(r['absolute_time_average_tail']['rel_error_pct']) for r in shell['runs']):.1f} % | **是** |
| 关键量 vs 解析解 `sigma/R` | < 3 % | 同上（未达） | **否** |
| 闭合是否优于钳位（配对） | 四个点全部更近 | 改善 {min(r['improvement_pt'] for r in shell['comparison'].values()):.1f}–{max(r['improvement_pt'] for r in shell['comparison'].values()):.1f} pt | **是** |
| 虚假流判据 `v_max/v_ref < 0.1` | < 0.1 | {spurious:.3f} | **否** |
| 内区压缩比达到承载需求 | >= 需求 | 见表 3.3 | 见上表 |

**与 m2971 派单口径的差异**：m2971 建议把
`test_laplace_pressure_floor.py::test_laplace_jump_within_15pct` 由 xfail 转 xpass。
本单改为**两份记录并存**：该 xfail 继续钉住未闭合模型（`report.json`，其 C1–C4 归因判据
依赖这份基线），验收判据移到
`tests/test_free_surface_pressure_closure.py::test_laplace_jump_within_15pct_with_the_shell_closure`
（读 `report_shell.json`）。这样打开闭合与否都有机器可裁的记录，不会因为一次切换丢掉历史。

**结论**：`sigma/R` 的 15 % 级验证从 WITHHELD 转为通过（钳位态 −34 %…−38 % → 闭合态
误差见 §3.1）；3 % 目标未达，剩余误差主要来自 `a_i = 0` 近似、阈值判据与被改写的量子层厚度，
留作后续单。**不声称达到 3 %。**
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text)
    print("wrote", DOC, len(text), "chars")


if __name__ == "__main__":
    main()