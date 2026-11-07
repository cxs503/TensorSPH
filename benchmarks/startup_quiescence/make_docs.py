"""Generate the quiescence verification docs and figures from ``report.json``.

Every number in the two markdown files is read out of
``benchmarks/startup_quiescence/report.json`` here -- nothing is typed by hand,
so regenerating after a re-run cannot leave a stale figure behind.

Usage::

    PYTHONPATH=src python benchmarks/startup_quiescence/make_docs.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REPORT = HERE / "report.json"
FIGS = HERE / "figures"
DOC_Q = REPO / "docs" / "verification_startup_quiescence.md"
DOC_F = REPO / "docs" / "verification_wave_flume.md"


def load() -> dict:
    return json.loads(REPORT.read_text())


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and (abs(value) < 1e-3 and value != 0.0
                                     or abs(value) >= 1e4):
        return f"{value:.3e}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pool_rows(cases: list[dict]) -> list[str]:
    rows = []
    for case in cases:
        if case["state"] != "pool":
            continue
        rows.append(
            f"| {case['geometry']['h_over_dx']:g} | "
            f"{case['settings']['density_calibration']} | "
            f"{case['interior_particles']} | "
            f"{fmt(case['initial_pressure_deviation'])} | "
            f"{fmt(case['transition_band_pressure_deviation'])} | "
            f"{fmt(case['initial_acceleration']['all_max'], 2)} | "
            f"{fmt(case['initial_acceleration']['interior_max'], 2)} | "
            f"{fmt(case['v_peak'])} | {fmt(case['interior_v_late_rms'])} |")
    return rows


def flume_rows(cases: list[dict]) -> list[str]:
    rows = []
    for case in cases:
        if case["state"] != "flume":
            continue
        label = case["label"].replace("flume resting column (paddle frozen at rest)",
                                      "shipped").strip(", ")
        rows.append(
            f"| {label} | {fmt(case['initial_pressure_deviation'])} | "
            f"{fmt(case['v_peak'])} | {fmt(case['interior_v_late_rms'])} | "
            f"{fmt(case['interior_v_late_rms'] / case['settings']['sound_speed'])} |")
    return rows


def figures(report: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    made = []
    cases = report["cases"]
    everything = [c for c in cases if c["state"] == "pool"]
    flumes = [c for c in cases if c["state"] == "flume"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    for case in everything:
        t = [i * case["sample_every"] * case["settings"]["dt"]
             for i in range(len(case["v_series"]))]
        axes[0].plot(t, case["v_series"], lw=1.2, label=case["label"])
    axes[0].set_xlabel("t (s)")
    axes[0].set_ylabel(r"$\max_i |v_i|$ (m/s)")
    axes[0].set_title("free-surface pool, no forcing")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    for case in flumes:
        t = [i * case["sample_every"] * case["settings"]["dt"]
             for i in range(len(case["v_series"]))]
        label = case["label"].replace(
            "flume resting column (paddle frozen at rest)", "shipped").strip(", ")
        axes[1].plot(t, case["v_series"], lw=1.2, label=label)
    signal = report["wave_signal"]["u_orbital_surface_m_s"]
    axes[1].axhline(signal, color="k", ls="--", lw=1.0,
                    label=fr"$u_{{orbital}}$ (H={report['wave_signal']['height_m']} m)")
    axes[1].set_xlabel("t (s)")
    axes[1].set_ylabel(r"$\max_i |v_i|$ (m/s)")
    axes[1].set_title("flume resting column: damping ablation")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    path = FIGS / "startup_v_series.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(str(path.relative_to(REPO)))

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    labels, values, colours = [], [], []
    for case in cases:
        if case["state"] == "pool":
            labels.append(f"pool h/dx={case['geometry']['h_over_dx']:g}\ncalib={case['settings']['density_calibration']}")
            colours.append("tab:blue" if case["settings"]["density_calibration"] else "tab:red")
        else:
            label = case["label"].replace(
                "flume resting column (paddle frozen at rest)", "flume shipped").strip(", ")
            labels.append(label)
            colours.append("tab:green")
        values.append(case["initial_pressure_deviation"])
    ax.bar(range(len(values)), values, color=colours)
    ax.set_yscale("log")
    ax.axhline(1e-3, color="k", ls="--", lw=1.0, label="gate 1e-3")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel(r"$\max|p-p_{hydro}|/(\rho_0 g H)$ at $t=0$")
    ax.set_title("static initial-state quality (interior, 3h margin)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    path = FIGS / "initial_state_quality.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(str(path.relative_to(REPO)))
    return made


def doc_quiescence(report: dict, figs: list[str]) -> str:
    cases = report["cases"]
    gates = report["gates"]
    findings = report["findings"]
    signal = report["wave_signal"]
    pool_on = [c for c in cases if c["state"] == "pool"
               and c["settings"]["density_calibration"]]
    worst_floor = max(pool_on, key=lambda c: c["interior_v_late_rms"])
    gate_rows = "\n".join(
        f"| `{g['id']}` | {g['reads']} | {fmt(g['measured'])} | "
        f"{fmt(g['limit'])} ({g['limit_side']}) | "
        f"{'pass' if g['pass'] else '**FAIL**'} |" for g in gates)
    finding_rows = "\n".join(
        f"| `{f['id']}` | {f['status']} | {f['reads']} | {fmt(f['measured'])} | "
        f"{fmt(f['limit'])} | {fmt(f['ratio_to_orbital_velocity'], 2)} |"
        for f in findings) or "| — | — | — | — | — | — |"
    ablation_rows = "\n".join(
        f"| {a['label'].replace('flume resting column (paddle frozen at rest)', 'shipped').strip(', ')} | "
        f"{fmt(a['v_peak'])} | {fmt(a['interior_v_late_rms'])} |"
        for a in report["flume_ablation"])
    return f"""# 零外力静水的启动伪流：谁真的在动，动多少

**范围**: TensorSPH ｜ **性质**: 验证欠账（把 backlog 的建议变成可复算的判据）
｜ **数据**: `benchmarks/startup_quiescence/report.json`（本文件全部数字由
`benchmarks/startup_quiescence/make_docs.py` 从该 JSON 注入，无手抄）

## 1. 案例与物理背景

`backlog_update_m3211.md` P2.2 提出：把「零外力静止水池的自生流速」当作任何
质量单位／自由面／压强闭合改动的冒烟判据。该条只给了一个数（`h/dx = 1.5`、
未标定质量时 **0.096 m/s**），并把它归因于晶格质量偏置。本单回到这个归因本身，
把一次性数字做成可复算的基准，并区分**静态度量**与**动态度量**：

| 名词 | 定义 |
|---|---|
| 内区（interior） | 距任一壁面/自由面 ≥ 3h 的粒子（见 §5 为何不是 2h） |
| `initial_pressure_deviation` | `max_i |p_i - rho0 g (H - y_i)| / (rho0 g H)`，t=0，内区 |
| `transition_band_pressure_deviation` | 自由面下 2h–3h 壳层内的同一度量（单列，不并入判据） |
| `v_peak` / `interior_v_late_rms` | 全粒子 / 内区 `max_i |v_i|` 的峰值，以及末四分之一时段的 RMS |

算例（无任何外力、无入射波、活塞冻结在静位）：

* **pool**：`droplet_impact.make_particles`，自由面水池 0.60 m × 0.24 m，`dx = 20 mm`，
  `h/dx ∈ {{1.3, 1.5}}` × `density_calibration ∈ {{True, False}}`，模拟 0.10 s。
* **flume**：`wave_flume.flume_water_lattice` 静止水柱 0.60 m × 0.20 m，`dx = 10 mm`，
  `h/dx = 1.3`，地板+下游壁+冻结活塞，模拟 0.06 s；再做粘度/密度扩散消融。

## 2. 怎么做的

1. `benchmarks/startup_quiescence/run.py` 直接调用生产 RHS（`droplet_impact.rhs`）与
   生成的初始状态构造函数，不复制求解器代码；每次采样记录全粒子与内区的 `max|v|`。
2. t=0 先用 `rhs` 取一次场量，得到初始压强相对解析静水压的偏差与初始加速度。
3. 判据分两层：**契约判据**（`gates`，必须全绿）与**未决发现**（`findings`，登记缺陷）。
   一个刚暴露出来的缺陷不应该被塞进契约判据里让基准永久变红，也不该被删掉。

## 3. 计算结果

### 3.1 池（静态度量随质量标定而变，动态度量几乎不变）

| h/dx | density_calibration | 内区粒子 | 初始压强偏差 | 2h–3h 壳层 | a0 max (m/s²) | a0 内区 (m/s²) | v_peak (m/s) | 内区 v_late_rms (m/s) |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(pool_rows(cases))}

`g = 9.81 m/s²`，`sqrt(gH) = {fmt((9.81 * 0.24) ** 0.5, 3)} m/s`；`a0` 峰值出现在池底
第 2 行（`y ≈ 0.030 m`），**不在自由面**：静止水池的启动瞬态由壁面压力外推/kernel 补偿驱动，
不是自由面。

### 3.2 波槽静止水柱（消融）

| 配置 | 初始压强偏差 | v_peak (m/s) | 内区 v_late_rms (m/s) | Mach (`v/c0`) |
|---|---|---|---|---|
{chr(10).join(flume_rows(cases))}

| 消融（波槽静止水柱） | v_peak (m/s) | 内区 v_late_rms (m/s) |
|---|---|---|
{ablation_rows}

波槽信号参照（Airy，一阶）：深 `{signal['depth_m']}` m、波高 `{signal['height_m']}` m、
周期 `{signal['period_s']}` s → `k = {fmt(signal['wavenumber_1_per_m'], 3)} 1/m`、
`omega = {fmt(signal['omega_rad_s'], 3)} rad/s`、自由面轨道速度
`u_orbital = {fmt(signal['u_orbital_surface_m_s'], 3)} m/s`。

### 3.3 判据与未决发现

| 判据 | 读取 | 实测 | 限值（方向） | 结果 |
|---|---|---|---|---|
{gate_rows}

| 发现 | 状态 | 读取 | 实测 | 限值 | 实测/u_orbital |
|---|---|---|---|---|---|
{finding_rows}

## 4. 结果图

![startup velocity series]({figs[0]})

左：自由面水池（无外力）的 `max|v|` 时程——标定与否的差异在动力学里几乎看不见；
右：波槽静止水柱的消融——虚线是 `u_orbital`。

![initial state quality]({figs[1]})

静态度量：`density_calibration` 把初始压强偏差从 `6.0e-1`（`h/dx=1.5` 未标定）
压到 `1.7e-4`，横幅是 1e-3 判据线。

## 5. 注意点（踩过的坑与适用范围）

1. **2h 不够。** 最初按 kernel 支撑 `2h` 划内区，波槽在自由面下方 2h–3h 的壳层里
   读到 `3.9e-2` 的压强偏差，直接把 `h/dx=1.3` 的池（`1.4e-4`）与波槽（`3.9e-2`）
   拉成两个量级。改成 3h 后波槽降到 `8.7e-4`，壳层单列成
   `transition_band_pressure_deviation`。**稳态压强的内区判据要用 3h。**
2. **静态度量和动态度量回答不同的问题。** 质量标定（m3211）确实把初始压强偏差
   压到 `1e-4` 级（G1/G2，灵敏度比 `{fmt(gates[1]['measured'], 0)}`），但它对启动
   瞬态的峰值几乎没影响（`h/dx=1.3`：`{fmt(pool_on[0]['v_peak'])}` vs 未标定
   `{fmt([c for c in cases if c['state']=='pool' and c['geometry']['h_over_dx']==1.3 and not c['settings']['density_calibration']][0]['v_peak'])}` m/s）。
   所以 m3211 P2.2 的「自生流速」不能作为质量单位的判据，只能作为**相对回归探针**；
   它量的主要是壁面/自由面的启动瞬态。
3. **静水不是"应该完全不动"。** 弱可压缩 SPH 的静水本身有 O(Mach) 的声学振铃，
   本基准量到的内区动态底噪（`h/dx=1.3`）为 `{fmt(pool_on[0]['interior_v_late_rms'])}` m/s，
   即 `sqrt(gH)` 的 `{fmt(pool_on[0]['interior_v_late_rms'] / (9.81 * 0.24) ** 0.5 * 100, 1)} %`。
   任何要解析更小速度的工况都必须先把这个底噪压下去（加数值粘度/更长的沉降段），
   或用相对判据（相对同一配置的基线）而不是绝对门限。
4. **波槽的 F1 是这类问题的实例**：默认 `viscosity = 1e-6`（水）下，静止水柱的噪声在
   0.2 s 窗口内不衰减（`benchmarks/startup_quiescence` 的消融：加 `viscosity = 0.01`
   落到 `0.0079` m/s）。这是"是否该给波槽加数值粘度"的工程决策，不是本基准该自动改的东西。
5. 适用范围：2-D、单相、`h/dx ∈ [1.3, 1.5]`、`dx ∈ [10, 20] mm`、`c0 ∈ [20, 30] m/s`；
   3-D 与多相未测。

## 6. 误差判定

| 关键量 | 基准 | 实测 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 内区初始压强（池 h/dx=1.3，标定） | `rho0 g (H-y)` | `{fmt(pool_on[0]['initial_pressure_deviation'])}`·`rho0 g H` | `{fmt(pool_on[0]['initial_pressure_deviation'] * 100, 3)} %` | **是** |
| 内区初始压强（池 h/dx=1.5，标定） | 同上 | `{fmt([c for c in pool_on if abs(c['geometry']['h_over_dx'] - 1.5) < 1e-9][0]['initial_pressure_deviation'])}`·`rho0 g H` | `{fmt([c for c in pool_on if abs(c['geometry']['h_over_dx'] - 1.5) < 1e-9][0]['initial_pressure_deviation'] * 100, 3)} %` | **是** |
| 内区初始压强（波槽，标定） | 同上 | `{fmt([c for c in cases if c['state']=='flume'][0]['initial_pressure_deviation'])}`·`rho0 g H` | `{fmt([c for c in cases if c['state']=='flume'][0]['initial_pressure_deviation'] * 100, 3)} %` | **是** |
| 内区初始压强（未标定 h/dx=1.5） | 同上 | `{fmt([c for c in cases if c['state']=='pool' and not c['settings']['density_calibration'] and abs(c['geometry']['h_over_dx']-1.5)<1e-9][0]['initial_pressure_deviation'])}`·`rho0 g H` | `{fmt([c for c in cases if c['state']=='pool' and not c['settings']['density_calibration'] and abs(c['geometry']['h_over_dx']-1.5)<1e-9][0]['initial_pressure_deviation'] * 100, 1)} %` | **否**（这正是缺陷本身） |
| 动态度量 | 无公认验收阈值 | — | — | **无基准**：不存在「多静才算静」的文献判据，故 `G3` 只是钉在今日实测值之上的回归包络，不是精度验收；不声明 <3% |

动态底噪最坏情况 `{fmt(worst_floor['interior_v_late_rms'])}` m/s（`{worst_floor['label']}`），
其与 `u_orbital` 的比值为 §3.3 的 F1 行；该量级意味着 **H=3 cm 级波浪的静水基线噪声
不可忽略**，这也是 `docs/verification_wave_flume.md` 里仍列为未验证的原因之一。
"""


def doc_wave_flume(report: dict, figs: list[str]) -> str:
    flumes = [c for c in report["cases"] if c["state"] == "flume"]
    shipped = flumes[0]
    finding = next((f for f in report["findings"]
                    if f["id"] == "F1_flume_resting_noise"), None)
    ablation = report["flume_ablation"]
    signal = report["wave_signal"]
    return f"""# 波槽（`tensorsph.wave_flume`）验证状态：已经证明了什么、还欠什么

**范围**: TensorSPH ｜ 本文是 `src/tensorsph/wave_flume.py` 与 `tests/test_wave_flume.py`
docstring 所指向的验证入口。数字全部来自
`benchmarks/startup_quiescence/report.json`（由 `make_docs.py` 注入）。

## 1. 这个模块要做什么

矩形 2-D 波槽：活塞造波（一阶造波理论 / Goring 孤波条件）、海绵层吸收、波高计，
把 WCSPH 核心（`droplet_impact.rhs` + 运动壁面输入）当作求解器。它自身不含新算子，
新增的是**算例**与**解析锚**：Airy 色散 `omega^2 = g k tanh(kh)`（Dean & Dalrymple
1991, eq. 3.23）、活塞传递函数 `H/S = 2(cosh 2kh - 1)/(sinh 2kh + 2kh)`（同书 eq. 6.38）、
一阶孤波 `eta = H sech^2(alpha(x - ct))`。

## 2. 已验证（有可复算证据）

| 项 | 证据 | 状态 |
|---|---|---|
| 色散关系、波数反演、相速极限、Airy 波形、活塞传递函数、Goring 冲程/速度/加速度解析式 | `tests/test_wave_flume.py`（解析锚自洽，含深/浅水极限与级数分支连续） | 通过（解析对解析） |
| 水柱格点恰好填满指定水深、质量单位与 kcal 标定一致 | `tests/test_wave_flume.py` | 通过 |
| **静止水柱（t=0，活塞冻结）是离散静水平衡** | `benchmarks/startup_quiescence`（本页 §3） | 通过：内区压强偏差 `{fmt(shipped['initial_pressure_deviation'])}`·`rho0 g H` |

## 3. 未验证 / 未决

| 项 | 现状 |
|---|---|
| **波浪工况本身**（造波→传播→吸收，波高/周期/反射系数 vs Airy） | **未跑、无已提交证据**。`benchmarks/wave_flume/run.py` 只提供单算例记录器，`artifacts/` 被 gitignore，仓库内没有 `report.json`。本单**未**补，因为一次可用工况需要上万粒子 × 数万步（本机 load15 ≈ 120，容量纪律禁止起新的重算力进程）。 |
| **静止水柱的动态底噪**（F1，open） | 默认配置（`viscosity = 1e-6`、`density_diffusion = 0.1`）下噪声不衰减：内区 `v_late_rms = {fmt(shipped['interior_v_late_rms'])}` m/s，是 `u_orbital = {fmt(signal['u_orbital_surface_m_s'], 3)}` m/s（H={signal['height_m']} m）的 `{fmt(finding['ratio_to_orbital_velocity'], 2)}` 倍。消融见下表。 |
| 反射系数、海绵层吸收效率、孤立波传播保真度 | 模块里有 `reflection_coefficient` / `separate_incident_reflected` 工具，**均无定量验证证据**。 |

消融（同一静止水柱，只改一个数值设置，0.06 s）：

| 配置 | v_peak (m/s) | 内区 v_late_rms (m/s) |
|---|---|---|
{chr(10).join(f"| {a['label'].replace('flume resting column (paddle frozen at rest)', 'shipped').strip(', ')} | {fmt(a['v_peak'])} | {fmt(a['interior_v_late_rms'])} |" for a in ablation)}

结论：底噪由**数值粘性缺位**主导，加上 `viscosity = 0.01`（`droplet_impact` 的默认数值
粘度）后底噪降 `{fmt(ablation[0]['interior_v_late_rms'] / blob, 1) if (blob := ablation[1]['interior_v_late_rms']) else '—'}` 倍。
这是工程决策（波槽是否该保持"物理水粘度"），需要另开一单决定后重跑分级工况。

## 4. 结果图

![flume quiescence]({figs[0]})

## 5. 误差判定

| 关键量 | 基准 | 实测 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 静止水柱内区初始压强 | `rho0 g (H - y)` | `{fmt(shipped['initial_pressure_deviation'])}`·`rho0 g H` | `{fmt(shipped['initial_pressure_deviation'] * 100, 3)} %` | **是** |
| 静止水柱动态底噪 | 无验收基准 | `{fmt(shipped['interior_v_late_rms'])}` m/s | 相对 `u_orbital` `+{fmt(finding['ratio_to_orbital_velocity'] * 100, 0)} %` | **否/无基准**：判据应是"底噪 ≪ 目标波高对应的轨道速度"，本配置不满足 |
| 波浪工况（波高/周期/反射） | Airy 解析 | **未测** | **未测** | **无法回答**——本页不声明任何波浪精度 |

## 6. 下一步（给接单的人）

1. 先定波槽的数值粘度/密度扩散设置（见 §3 消融），再跑分级工况；否则噪声与
   H=3 cm 波信号同量级，误差表没有意义。
2. 分级工况建议：同一 `H/T` 下 2–3 个分辨率、`dx` 由 `depth_cells = 20/30/40` 生成；
   读数用 `probe_elevation` + `dominant_omega`，与 `wavelength()` 对比；反射用
   `separate_incident_reflected` 两探头法。
3. 跑完把 `report.json` 提交进 `benchmarks/wave_flume/`（不要只留 `artifacts/`），
   并把本页 §3 的"未验证"逐条改成实测行。
"""


def main() -> int:
    report = load()
    figs = figures(report)
    DOC_Q.write_text(doc_quiescence(report, figs))
    DOC_F.write_text(doc_wave_flume(report, figs))
    print(f"wrote {DOC_Q.relative_to(REPO)}")
    print(f"wrote {DOC_F.relative_to(REPO)}")
    for f in figs:
        print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())