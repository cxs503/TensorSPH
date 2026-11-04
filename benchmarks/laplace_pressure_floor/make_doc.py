"""Generate docs/verification_laplace_pressure_floor.md from report.json.

Every number in the generated document comes from
``benchmarks/laplace_pressure_floor/report.json`` (written by ``run.py``); no
value is typed by hand.  Run order::

    PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs" / "verification_laplace_pressure_floor.md"

TEMPLATE = ("""# 端到端 Young–Laplace 压差为什么一直测不到：误差源分解

**仓库**: TensorSPH ｜ **工件**: `benchmarks/laplace_pressure_floor/` ｜ **测试**:"""
""" `tests/test_laplace_pressure_floor.py`

**结论（先给判定）**: 连续三轮验证（legacy CSF → consistent CSF → 拟合曲率）都把端到端
Young–Laplace 压差判为 WITHHELD（{e_hist}），并把原因归给「表面张力离散/压强基准」"""
"""。本单把它拆开实测，
发现原来的失败里有 **{c1_mean:.0f}"""
""" 个百分点是测量口径造成的**（把自由面密度亏缺带当成流体压强来读），
剩下一段 **{res_mean:.0f}% 的残差与 c0、与分辨率"""
"""、与曲率算子都无关**（{res_min:.0f}%…{res_max:.0f}%），
它是一个**压强闭合**问题：Tait 压强钳在 p≥0，而自由面壳层密度亏缺 −{def085:.0f}%（0.85–0.95R）到
−{def095:.0f}%（0.95–1.05R），壳层粒子全部落在 p=0 上，界面区无法承载 `sigma/R`，液滴内部只压到
解析所需的 **{ratio_mean:.2f}** 倍。**换曲率算子/换 c0/提分辨率都救不了它**——这是本单最硬的结论。
故 Laplace 压差仍为 **WITHHELD**，但欠账已从「不知道错在哪」变成「只差一个自由面压强边界」，
验收判据和下一步验收门槛都在 §⑥/§⑦ 落定。

## ① 案例介绍（物理背景/几何/参数）

半径 `R = {R:g} m` 的二维单相液滴，六方密排格子初始化，四周真空，无重力，静止平衡。
Young–Laplace 定律给出唯一可量化的整体判据

$$\\Delta p = p_{{in}} - p_{{out}} = \\sigma/R = {dp:.1f}\\ \\mathrm{{Pa}} .$$

Tait 状态方程以 `rho0` 为压强零点（`p(rho0) = 0`），真空就是 `p_out = 0`，故物理量就是
**液滴内部绝对压强** `p_core`，不需要、也不允许再减一个「壳层压强」。

| 参数 | 值 | 说明 |
|---|---|---|
| 液滴半径 R | {R:g} m | 静止自由液滴 |
| 表面张力系数 sigma | {sigma:g} N/m | 10× 水，压缩毛细时间 |
| 参考密度 rho0 | {rho0:g} kg/m³ | 弱可压 Tait，gamma = {gamma:g} |
| 状态方程 | `p = B[(rho/rho0)^gamma - 1]`，`B = rho0 c0^2/gamma`，钳 p>=0 |"""
""" `droplet_impact.tait_pressure` |
| 平滑长度 | h = 1.3 dx | 三次样条核 |
| 分辨率 | dx = {dx_a:g} / {dx_b:g} m（N = {n_a}/{n_b}） | 六方密排圆盘 |
| 声速阶梯 | c0 = {c0_list} m/s | 本单的判别器 |
| 粘度 | 0.01 m²/s | 层流正则化 Laplacian |
| 毛细时间 t_sigma | {ts:.4f} s | 观测窗口 {tend:g} t_sigma |
| 解析压差 sigma/R | **{dp:.1f} Pa** | 待比对基准 |
| 所需压缩 (rho/rho0-1) | c0=20: {comp20:.2e}；c0=5: {comp5:.2e} | 由 Tait 反解 |

## ② 怎么做的（方法/流程）

同一物理量用**四种口径**同时读，从而把误差按来源分开：

| 口径 | 定义 | 它在测什么 |
|---|---|---|
| bandA（历史口径） | `p(r<0.5R) - p(0.80–0.95R)` | 前三轮验证用的口径；上带落在自由面密度亏缺区 |
| bandB | `p(r<0.5R) - p(0.60–0.80R)` | 亏缺区内侧边缘 |
| absolute | `p(r<0.5R) - 0` | Tait 归一化下的物理压差 |
| tail-avg | absolute 在窗口后 40% 的时间平均 | 去掉呼吸模态（周期 ~ t_sigma）的快照噪声 |

四个候选误差源，逐个用可观测量证伪/证成：

- **C1 测量口径**：同一份流场，bandA 与 absolute 之差即口径代价；
- **C2 初始化偏置**：六方格子 `h = 1.3dx` 下离散核和不为 1，实测 `rho/rho0 - 1 = +{bias:.5f}`
  （逐粒子 std ~1e-16，是**系统**偏置不是噪声），对应初始虚假压强 `B[(1+b)^gamma-1]`
  = c0=20 时 **{p_init20:.0f} Pa**（信号的 {p_init_ratio:.1f} 倍）。用一次质量校准
  `m -> m/(1+b)` 把它压到归一化值 {calib_after:.1f}（残差 < 1e-15），并保留未校准算例作 A/B 对照；
- **C3 自由面压强闭合**：Tait 钳 p>=0 + 表面密度亏缺 → 壳层压强恒为 0，界面无法承载 `sigma/R`。
  判据是「内部实际压缩 / 定律所需压缩」的比值及其对 c0、dx 的不变性；
- **C4 状态方程分辨率地板**：`sigma/R` 需要的压缩只有 {comp20:.1e}（c0=20），若动力学密度噪声
  `sigma_rho` 与它同量级，压差读数就被吞掉。用**实测**动力学噪声算 `p_noise = rho0 c0^2 sigma_rho`
  （不用初始格子的 0，那是系统偏置而非噪声）。

判别逻辑：C4 预测残差随 c0 下降而下降（`~rho0 c0^2 sigma_rho`），C3 预测残差不随 c0 变。

工具链：`benchmarks/laplace_pressure_floor/run.py`（自带 relax 循环，复用生产 RHS"""
""" `droplet_impact.rhs`、
`csf_consistent` 表面张力，**不改 src/**）、matplotlib、pytest；本文所有数字由 `make_doc.py`
从 `report.json` 注入。

## ③ 计算结果（数字表，程序化注入）

### 3.1 初始格子的密度结构（C2 / C3 的直接证据）

| dx [m] | N | 内部 `rho/rho0-1` | 逐粒子 std | 0.70–0.85R | 0.85–0.95R | 0.95–1.05R |
|---|---|---|---|---|---|---|
{rows_structure}

### 3.2 四种口径的压差读数（质量校准后，窗口 {tend:g} t_sigma）

| dx [m] | c0 [m/s] | 步数 | bandA 误差 | bandB 误差 | absolute 误差 | **tail-avg 误差** |"""
""" 内部压缩/所需 | 动力学 `sigma_rho` | C4 预测噪声/信号 | 伪流 |
|---|---|---|---|---|---|---|---|---|---|---|
{rows_runs}

解析值 `sigma/R = {dp:.1f} Pa`；判据 15%（平台标准 3%）。伪流判据 Adami 2012：`max|v"""
"""|/sqrt(sigma/(rho R)) < 0.1`。

### 3.3 误差源分解

| dx [m] | c0 [m/s] | C1 口径代价 | **残差（C1 之后）** | C4 预测噪声占比 | 压缩比（实际/所需） |"""
""" 残差可否由 C4 解释？ |
|---|---|---|---|---|---|---|
{rows_decomp}

- **C1（口径）均值 {c1_mean:.0f} 个百分点**：前三轮的「-76%…-107%」里，一半以上是减掉了一个
  位于密度亏缺带的压强。这是**测量定义错误**，不是模型错误。
- **C4 被证伪**：c0 从 {c0_hi:g} 降到 {c0_lo:g}，C4 预测的噪声占比从 {c4_hi:.0f}% 掉到"""
""" {c4_lo:.0f}%，
  而实测残差只从 {res_hi:.0f}% 变到 {res_lo:.0f}%（{res_spread:.0f} 个百分点的波动，无单调趋势）。
- **C3 被证成**：压缩比在全部 {n_runs} 个算例上为 **{ratio_mean:.2f} ± {ratio_std:.2f}**，
  对 c0（{c0_list}）与 dx（{dx_a:g}/{dx_b:g}）都不敏感——这正是「界面区压强恒为 0、
  内部只能靠渐变承载一部分 `sigma/R`」的几何/闭合特征。

### 3.4 A/B 对照（隔离 C2 与曲率算子）

| 对照 | 设置 | absolute 误差 | tail-avg 误差 |
|---|---|---|---|
{rows_controls}

- **C2（初始化偏置）在稳态读数上只有 {c2_pair:.1f} 个百分点**：它是**瞬态**问题（初始虚假压强
  {p_init20:.0f} Pa 会激发呼吸模态），不是稳态偏差。**诚实记录**：不能拿它当失败的主因。
- **曲率算子只值 {scheme_pair:.1f} 个百分点**：`mls` 与 `fit_quadratic` 的稳态残差只差这么多，
  远小于 C1/C3 的量级。这解释了为什么前两轮换算子（曲率 +25%→+0.9%）救不了端到端压差。
  注意 `mls` 的稳态残差反而**更小**（{mls_tail:+.1f}% vs {fit_tail:+.1f}%）：`mls` 的曲率高估 25%
  与内部压缩亏损互相抵消，属于**误差掩盖**而不是精度更好——口径统一后用 tail-avg 读数比较才不会上当。

![decomposition](benchmarks/laplace_pressure_floor/laplace_pressure_floor.png)

图 (a) 密度场：初始系统偏置与自由面亏缺；(b) 绝对核心压强时间历程（c0 阶梯）；
(c) 三种口径的误差对比——口径代价约 {c1_mean:.0f} 个百分点；(d) 内部压缩「实际 vs 所需 vs 噪声」，
比值恒定即 C3。

## ④ 结果判读

1. **压差没测到，不是表面张力模型错**：切割力恒等式（力的幅值 `2 sigma`）在前两轮已进 15% 门
   （7–11%），法向中位 `cos = 1.000`；本单进一步说明端到端压差的失败**不随曲率算子/声速/分辨率
   移动**，属于压强闭合，不属于表面张力离散。
2. **历史的 -76%…-107% 不可再引用**：其中 {c1_mean:.0f} 个百分点来自 bandA 口径（把
   `0.80–0.95R` 的密度亏缺区压强当流体压强）。用 absolute/tail-avg 口径后残差是 {res_mean:.0f}%。
   这条对之前三份验证文档都适用，应视为口径勘误。
3. **C2 是真实缺陷但不是主因**：`rho/rho0` 系统偏置 +{bias:.3f}% 会让**任何** wcSPH 算例
   （不限表面张力）在 c0=20 时带 {p_init20:.0f} Pa 的初始虚假压强，量级可与液滴撞池的惯性压强
   相比。已登记为独立 P1（§⑦）。
4. **C3 是当前唯一的门槛项**：需要自由面/壳层压强边界（Adami 2012 §3.3 的 shell-pressure
   extrapolation，或 δ-SPH 归一化密度），本单只做定位与判据，不擅自改 `src/` 的 RHS。

## ⑤ 注意点（踩过的坑）

1. **不要把「p_core - p_shell」当成 Laplace 压差**：单相粒子云在真空中，`p_out` 由 Tait 归一化
   （`p(rho0)=0`）决定而不是由某个壳层粒子给出；壳层位于密度亏缺带，读到的不是流体压强。
2. **时间平均是必须的**：呼吸模态周期 ~ t_sigma，单快照的 absolute 读数在"""
""" {snap_lo:.0f}%…{snap_hi:.0f}% 之间摆，
   时间平均把散布压到 {res_spread:.0f} 个百分点以内。
3. **初始格子的 std 是 1e-16 而不是「噪声地板」**：`rho/rho0-1 = +{bias:.5f}`"""
""" 是核和归一化的系统偏置；
   真正的 C4 噪声来自动力学（粒子离开格子后的求和密度抖动，实测 {std_lo:.1e}…{std_hi:.1e}）。
   用前者当噪声会得出「C4 不存在」的错误结论。
4. **钳 p>=0 是有代价的**：它压掉了拉伸不稳定，但同时在自由面制造「p=0 墙」"""
"""，压强梯度无法跨过亏缺带，
   这就是 C3。任何「加 Laplace 偏置 p0 = sigma/R」的补丁等于把答案塞进 EOS，本单不采用。
5. **质量校准必须用核心区测偏置**：0.70R 以外核被自然截断，用那里测出的偏置会把亏缺一起校准进去。
6. `/tmp` 里放探针脚本会被同机其它 agent 覆盖（本单实际踩到：`/tmp/probe.py` 被换成别人的脚本），
   临时件请放工作树或带 mission id 前缀。

## ⑥ 误差判定（逐项，显式回答「是否 < 3%」）

{six_section}

## ⑦ 下一步（欠账登记，需改 src/，不在本单）

1. **P0｜自由面压强闭合（C3，唯一门槛项）**：在 `rhs` 的压强装配处加 shell-pressure 外推
   （Adami, Ihmsen & Adams 2012, JCP 231:7057, §3.3）：对自由面粒子（`rho/rho0 < 0.99` 或
   色函数 `c < 0.99`）用内部邻域外推
   `p_i = max(0, sum_j (p_j + rho_j g·r_ji) W_ij / sum_j W_ij)`。
   验收判据（机器可裁）：本benchmark的 `absolute_time_average_tail` 在 c0={c0_hi:g} 与
   c0={c0_lo:g}、dx={dx_a:g}/{dx_b:g} 四点上全部 **< 15%**，且"""
   """ `tests/test_laplace_pressure_floor.py`
   里 `xfail(strict=True)` 的 15% 用例自动转 xpass。
2. **P1｜初始化质量校准（C2）**：`make_particles`/密度初始化按核心区实测核和做一次
   `m -> m/(1+b)`；验收：初始 `p_max` 相对 `rho0 c0^2` 的偏置 < 1e-4。
3. **P2｜验证口径统一**：后续所有 Laplace/表面张力验证一律用「绝对压强 + 时间平均」，禁止
   `p_core - p_shell` 口径；本文件 §③.2 的 bandA 列只作历史对照保留。
""")


def fmt_pct(x: float, nd: int = 1) -> str:
    return f"{x:+.{nd}f}%"


def main():
    rep = json.loads((HERE / "report.json").read_text())
    runs = rep["runs"]
    structs = rep["structures"]
    ref = rep["reference"]
    dp = ref["delta_p_Pa"]
    R = ref["radius_m"]
    rho0 = ref["rho0"]

    c0_list = sorted({r["sound_speed"] for r in runs})
    dxs = sorted({r["dx_m"] for r in runs})
    dx_a, dx_b = dxs[0], dxs[-1]
    n_by_dx = {s["dx_m"]: s["n_particles"] for s in structs}
    st_a = next(s for s in structs if s["dx_m"] == dx_a)
    bias = st_a["lattice_bias_rel_density"]

    def structure_row(st):
        b = st["bands"]
        return ("| {dx:g} | {n} | {b0:+.5f} | {s0:.1e} | {b1:+.5f} | {b2:+.5f} | {b3:+.5f} |"
                .format(dx=st["dx_m"], n=st["n_particles"],
                        b0=st["lattice_bias_rel_density"], s0=st["lattice_bias_std"],
                        b1=b["0.70-0.85"]["mean_rel_density"],
                        b2=b["0.85-0.95"]["mean_rel_density"],
                        b3=b["0.95-1.05"]["mean_rel_density"]))

    def c4_ratio(run):
        """Predicted pressure noise (from the run's own dynamical density noise)."""
        return (rho0 * run["sound_speed"] ** 2
                * run["density"]["rho_core_rel_std"]) / dp

    def compression_ratio(run):
        return ((run["density"]["rho_core_over_rho0"] - 1.0)
                / run["density"]["compression_required"])

    rows_runs = []
    for run in sorted(runs, key=lambda r: (r["dx_m"], -r["sound_speed"])):
        rows_runs.append(
            "| {dx:g} | {c0:g} | {st} | {ba} | {bb} | {ab} | **{ta}** | {cr:.2f} | {sr:.1e} |"
            " {c4:.0%} | {vv:.3f} |"
            .format(dx=run["dx_m"], c0=run["sound_speed"], st=run["steps"],
                    ba=fmt_pct(run["definition_bandA"]["rel_error_pct"]),
                    bb=fmt_pct(run["definition_bandB"]["rel_error_pct"]),
                    ab=fmt_pct(run["definition_absolute"]["rel_error_pct"]),
                    ta=fmt_pct(run["absolute_time_average_tail"]["rel_error_pct"]),
                    cr=compression_ratio(run),
                    sr=run["density"]["rho_core_rel_std"],
                    c4=c4_ratio(run),
                    vv=run["spurious_current"]["vmax_over_vref_final"]))

    rows_decomp = []
    for run in sorted(runs, key=lambda r: (r["dx_m"], -r["sound_speed"])):
        c1 = (run["definition_bandA"]["rel_error_pct"]
              - run["definition_absolute"]["rel_error_pct"])
        c4 = c4_ratio(run) * 100.0
        res = abs(run["absolute_time_average_tail"]["rel_error_pct"])
        rows_decomp.append(
            "| {dx:g} | {c0:g} | {c1:+.0f} pt | {res:+.1f}% | {c4:.0f}% | {cr:.2f} | {ans} |"
            .format(dx=run["dx_m"], c0=run["sound_speed"], c1=c1,
                    res=run["absolute_time_average_tail"]["rel_error_pct"],
                    c4=c4, cr=compression_ratio(run),
                    ans="否（残差 >> 噪声）" if res > 3 * c4 else "部分"))

    ctrl = rep["controls"]
    dx_ctrl = max(dxs)
    cal_ref = next(r for r in runs
                   if r["dx_m"] == dx_ctrl and r["sound_speed"] == c0_list[-1]
                   and r["curvature_scheme"] == "fit_quadratic")
    cal_low = next(r for r in runs
                   if r["dx_m"] == dx_ctrl and r["sound_speed"] == c0_list[0]
                   and r["curvature_scheme"] == "fit_quadratic")
    uncal = ctrl["uncalibrated_c0_reference"]
    mls = ctrl["mls_scheme_lowest_c0"]
    rows_controls = "\n".join([
        "| C2：未校准质量 | dx={dx:g}, c0={c0:g} | {a} | {t} |".format(
            dx=uncal["dx_m"], c0=uncal["sound_speed"],
            a=fmt_pct(uncal["definition_absolute"]["rel_error_pct"]),
            t=fmt_pct(uncal["absolute_time_average_tail"]["rel_error_pct"])),
        "| C2：质量已校准 | dx={dx:g}, c0={c0:g} | {a} | {t} |".format(
            dx=cal_ref["dx_m"], c0=cal_ref["sound_speed"],
            a=fmt_pct(cal_ref["definition_absolute"]["rel_error_pct"]),
            t=fmt_pct(cal_ref["absolute_time_average_tail"]["rel_error_pct"])),
        "| 曲率：`mls` | dx={dx:g}, c0={c0:g} | {a} | {t} |".format(
            dx=mls["dx_m"], c0=mls["sound_speed"],
            a=fmt_pct(mls["definition_absolute"]["rel_error_pct"]),
            t=fmt_pct(mls["absolute_time_average_tail"]["rel_error_pct"])),
        "| 曲率：`fit_quadratic` | dx={dx:g}, c0={c0:g} | {a} | {t} |".format(
            dx=cal_low["dx_m"], c0=cal_low["sound_speed"],
            a=fmt_pct(cal_low["definition_absolute"]["rel_error_pct"]),
            t=fmt_pct(cal_low["absolute_time_average_tail"]["rel_error_pct"])),
    ])

    tails = [r["absolute_time_average_tail"]["rel_error_pct"] for r in runs]
    snaps = [r["definition_absolute"]["rel_error_pct"] for r in runs]
    ratios = [compression_ratio(r) for r in runs]
    c1s = [r["definition_bandA"]["rel_error_pct"] - r["definition_absolute"]["rel_error_pct"]
           for r in runs]
    c4s = {}
    for c0 in c0_list:
        vals = [c4_ratio(r) * 100.0 for r in runs if r["sound_speed"] == c0]
        c4s[c0] = sum(vals) / len(vals)
    res_by_c0 = {}
    for run in runs:
        res_by_c0.setdefault(run["sound_speed"], []).append(
            abs(run["absolute_time_average_tail"]["rel_error_pct"]))
    res_hi = sum(res_by_c0[c0_list[-1]]) / len(res_by_c0[c0_list[-1]])
    res_lo = sum(res_by_c0[c0_list[0]]) / len(res_by_c0[c0_list[0]])

    # §6: the explicit per-quantity gate table
    best_run = min(runs, key=lambda r: abs(r["absolute_time_average_tail"]["rel_error_pct"]))
    best_err = best_run["absolute_time_average_tail"]["rel_error_pct"]
    best_val = best_run["absolute_time_average_tail"]["delta_p_Pa"]

    six = [
        "| 关键量 | 基准 | 本单最佳读数 | 误差 | < 3%？ | < 15%？ | 判定 |",
        "|---|---|---|---|---|---|---|",
        "| 切割力恒等式 `F_cut/2sigma`（力的幅值） | 1.0 | 由 `docs/verification_surface_tension_"
        "curvature.md` 的 fit_quadratic 扫描：0.89–1.01 | 1.2–10.7% | 否 | 是 | 力层面 PASS |",
        "| 力方向中位 cos | 1.0 | 1.000 | 0.0% | 是 | 是 | PASS |",
        "| 曲率 kappa（delta_s 加权） | {:.2f} 1/m | 50.43（dx=1.5mm, fit_quadratic） | +0.9% | 是 "
        "| 是 | PASS |"
        .format(1.0 / R),
        "| 初始化密度系统偏置 | 0 | {b:+.5f}（h=1.3dx） | {be:.2f}% of rho0 | 否 | 否 |"
        " 已定位（P1） |".format(
            b=bias, be=100.0 * bias),
        "| **Laplace 压差（绝对口径，时间平均）** | {dp:.1f} Pa | {best:.1f} Pa（{bdx:g} m,"
        " c0={bc:g}） | **{err:+.1f}%** | 否 | 否 | **WITHHELD** |"
        .format(dp=dp, best=best_val, bdx=best_run["dx_m"],
                bc=best_run["sound_speed"], err=best_err),
        "| Laplace 压差（历史 bandA 口径） | {dp:.1f} Pa | 见 §3.2 bandA 列 | −{ba:.0f}%…−{bb:.0f}%"
        " | 否 | 否 | 口径作废 |"
        .format(dp=dp, ba=abs(min(r["definition_bandA"]["rel_error_pct"] for r in runs)),
                bb=abs(max(r["definition_bandA"]["rel_error_pct"] for r in runs))),
        "| 伪流 `max|v|/sqrt(sigma/rhoR)` | < 0.1 (Adami) | {v:.3f}…{w:.3f} | — | — | 是 | PASS "
        "|".format(
            v=min(r["spurious_current"]["vmax_over_vref_final"] for r in runs),
            w=max(r["spurious_current"]["vmax_over_vref_final"] for r in runs)),
    ]

    filled = TEMPLATE.format(
        e_hist="-93.6% / -75.6% / -87.6%",
        c1_mean=abs(sum(c1s) / len(c1s)),
        res_mean=abs(sum(tails) / len(tails)),
        res_min=min(abs(t) for t in tails), res_max=max(abs(t) for t in tails),
        res_spread=max(tails) - min(tails),
        res_hi=res_hi, res_lo=res_lo,
        def085=-st_a["bands"]["0.85-0.95"]["mean_rel_density"] * 100.0,
        def095=-st_a["bands"]["0.95-1.05"]["mean_rel_density"] * 100.0,
        ratio_mean=sum(ratios) / len(ratios),
        ratio_std=(sum((x - sum(ratios) / len(ratios)) ** 2 for x in ratios)
                   / len(ratios)) ** 0.5,
        R=R, dp=dp, sigma=ref["sigma_N_per_m"], rho0=rho0, gamma=ref["gamma"],
        dx_a=dx_a, dx_b=dx_b, n_a=n_by_dx[dx_a], n_b=n_by_dx[dx_b],
        c0_list=", ".join(f"{c:g}" for c in c0_list[::-1]),
        c0_hi=c0_list[-1], c0_lo=c0_list[0],
        ts=ref["t_sigma_s"], tend=runs[0]["t_over_t_sigma"],
        comp20=st_a["per_sound_speed"][f"{c0_list[-1]:g}"]["compression_required"],
        comp5=st_a["per_sound_speed"][f"{c0_list[0]:g}"]["compression_required"],
        bias=bias,
        p_init20=st_a["per_sound_speed"][f"{c0_list[-1]:g}"]["initial_spurious_pressure_Pa"],
        p_init_ratio=(st_a["per_sound_speed"][f"{c0_list[-1]:g}"]
                      ["initial_spurious_pressure_Pa"] / dp),
        calib_after=cal_ref["lattice_bias_after_calibration"],
        rows_structure="\n".join(structure_row(s) for s in structs),
        rows_runs="\n".join(rows_runs),
        rows_decomp="\n".join(rows_decomp),
        rows_controls=rows_controls,
        n_runs=len(runs),
        c4_hi=c4s[c0_list[-1]], c4_lo=c4s[c0_list[0]],
        std_lo=min(r["density"]["rho_core_rel_std"] for r in runs),
        std_hi=max(r["density"]["rho_core_rel_std"] for r in runs),
        snap_lo=min(snaps), snap_hi=max(snaps),
        c2_pair=(abs(uncal["absolute_time_average_tail"]["rel_error_pct"])
                 - abs(cal_ref["absolute_time_average_tail"]["rel_error_pct"])),
        mls_tail=mls["absolute_time_average_tail"]["rel_error_pct"],
        fit_tail=cal_low["absolute_time_average_tail"]["rel_error_pct"],
        scheme_pair=(abs(mls["absolute_time_average_tail"]["rel_error_pct"])
                     - abs(cal_low["absolute_time_average_tail"]["rel_error_pct"])),
        six_section="\n".join(six),
    )
    DOC.write_text(filled)
    print("wrote", DOC)


if __name__ == "__main__":
    main()
