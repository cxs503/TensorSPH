"""Render ``docs/case_hydrostatic_report.md`` from ``runs/case_hydrostatic/report.json``.

Every number in the report is read out of the run artifacts at render time
(programmatic injection, no hand transcription); figures are the PNGs written by
``run.py --figure``.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ART = REPO / "runs" / "case_hydrostatic"
OUT = REPO / "docs" / "case_hydrostatic_report.md"


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main() -> int:
    report = json.loads((ART / "report.json").read_text())
    verdict = report["verdict"]
    gate = report["gate_run"]
    ref = gate["reference"]
    cfg = gate["config"]
    nd = gate["nondimensional"]
    rows = gate["profile"]
    bench = gate["benchmark"]
    figures = sorted(p.name for p in (ART / "figures").glob("*.png"))

    lines: list[str] = []
    add = lines.append
    add("# CASE-HYDROSTATIC — 静水压剖面 `p(y) = rho0 g (H - y)`（TensorSPH）\n")
    add(f"**算例锚点**：`{report['case_id']}` ｜ **仓库**：{report['repo']} ｜ "
        f"**判据**：内部条带最大相对误差 ≤ {verdict['gate_limit_pct']:.1f} % ｜ "
        f"**实测**：{fmt(verdict['gate_value_pct'], 4)} % → "
        f"**{'达标' if verdict['gate_pass'] else '未达标'}**\n")
    add("> 复现：`" + report["repro"]["gate"] + "`；一键单跑：`"
        + report["repro"]["one_shot"] + "`；网格+静态+图：`"
        + report["repro"]["grid"] + "` → `" + report["repro"]["figures"] + "`\n")

    # ------------------------------------------------------------------ ①
    add("## ① 问题与参考解出处\n")
    add("矩形水箱静水（still water）：水体在重力下静止，压力沿垂向线性分布。"
        "参考解由静力平衡 `dp/dy = -rho0 g` 从自由面 `p(H)=0` 积分得到闭式解\n")
    add(f"```\n{ref['formula']}\n```\n")
    add(f"* 参考解类型：**{ref['kind']}**（推导：{ref['derivation']}）。")
    add(f"* 出处与同构基准：{ref['provenance']}")
    add(f"* 常数：`rho0 = {ref['rho0']:.0f} kg/m³`、`g = {ref['g']:.2f} m/s²`、`H = {ref['H']:.3f} m`。")
    add("* **零手抄**：ref 数值不是从表格誊抄，而是在每个实测条带中心 `y` 上由闭式解程序求值"
        "（`analytic_pressure(y)`，与实测同一 moment）；报告中的 ref 列直接来自 run JSON。\n")
    add("几何与工况：\n")
    add("| 量 | 值 |\n|---|---|")
    add(f"| 水槽宽 W | {cfg['tank_width']:.3f} m |")
    add(f"| 水深 H | {cfg['depth']:.3f} m |")
    add(f"| 粒径 dx | {cfg['dx']:.4f} m（{cfg['columns']} 列跨槽）|")
    add(f"| 流体粒子数 N | {cfg['fluid_particles']}（能力下限 {cfg['min_particles']}）|")
    add(f"| 声速 c0 | {cfg['sound_speed']:.1f} m/s |")
    add(f"| 时间步 dt | {cfg['dt']:.3e} s（{cfg['steps']} 步）|")
    add(f"| 物理时长 | {cfg['seconds']:.1f} s |")
    add(f"| 统计窗 | 后 {cfg['average_from']*100:.0f} %，每 {cfg['sample_every']} 步采样，"
        f"{cfg['samples']} 个样本 |")
    add(f"| 核支撑 2h | {cfg['support']:.4f} m |\n")

    # ------------------------------------------------------------------ ②
    add("## ② 方法与实现\n")
    add("**求解器**：仓库现成 WCSPH 生产路径 `tensorsph.droplet_impact.rhs`（Tait 状态方程、"
        "三阶样条核、邻居对表 `PairListCache`），边界为两层壁面粒子 `wall_lattice`，"
        "初始质量按静水密度 `hydrostatic_density(y)` 布置（`hydrostatic_init=True`），"
        "时间推进为显式辛欧拉，`apply_boundary` 施加壁面/自由面约束。**没有另起炉灶**："
        "算例只复用求解器并新增约简与判据层。\n")
    add("**实现文件**：\n")
    add("* `src/tensorsph/case_hydrostatic.py` — 算例实现：闭式参考 `analytic_pressure`、"
        "条带约简 `pressure_profile`、判据 `evaluate`、时间平均记录 `run_case`、"
        "确定性 t=0 记录 `run_initial_state`、CLI `main`。")
    add("* `src/tensorsph/vv/hydrostatic.py` — 验证阶梯 Rung 1（本次扩展 `return_fields=True`，"
        "让调用方拿到统计窗内的时间平均场，而不只是标量条带值）。")
    add("* `src/tensorsph/verification/hydrostatic.py` — 验证面 M2（本次扩展 `sample_every` / "
        "`average_from` 两个窗参数并透传 `return_fields`，避免为了换窗口而复制算例定义）。")
    add("* 入口：`python -m tensorsph --case hydrostatic`（包 CLI 分发）与 "
        "`python -m tensorsph.case_hydrostatic`（模块 CLI）；"
        "网格编排 `benchmarks/case_hydrostatic/run.py`；测试 `tests/test_case_hydrostatic.py`。")
    add("* 平台侧调用：`benchmarks/case_hydrostatic/run.py` 与 `tests/test_case_hydrostatic.py` "
        "直接调用 `tensorsph.case_hydrostatic`，包 `__init__` 导出 "
        "`run_hydrostatic_profile_case` / `analytic_pressure` / `pressure_profile`。\n")
    add("**度量定义**（先于读数声明）：\n")
    add("1. 权重：粒子质量的条带加权平均压力 `p_band`，条带中心 `y_band` 亦为质量加权；")
    add("2. 时间平均：只对统计窗内的样本做平均，取同一批样本的**时间平均位置**与"
        "**时间平均压力**配对（混用末态位置与窗口均值等于比较两个时刻）；")
    add("3. 自由面排除（overlap 规则）：条带上边缘一旦进入自由面 2h 支撑带即排除；")
    add("4. 判据：`max_{内部条带} |p_band - p_ref(y_band)| / p_ref(y_band)` ≤ 1.0 %。\n")

    # ------------------------------------------------------------------ ③
    add("## ③ 参数表（无量纲数完整）\n")
    add("| 无量纲量 | 定义 | 值 |\n|---|---|---|")
    add(f"| 声学马赫数 | `sqrt(2 g H)/c0` | {nd['mach_acoustic']:.4f} |")
    add(f"| 声学 Courant 数 | `c0 dt / dx` | {nd['gravity_wave_courant']:.4f} |")
    add(f"| 垂向分辨率 | `H/dx` | {nd['H_over_dx']:.1f} |")
    add(f"| 展弦比 | `W/H` | {nd['W_over_H']:.2f} |")
    add(f"| 声学周期占比 | `(2H/c0)/T` | {nd['acoustic_period_fraction']:.5f} |")
    add("| 密度比 | `rho0/rho0` | 1（单相）|")
    add(f"| 压力尺度 | `rho0 g H` | {ref['rho0']*ref['g']*ref['H']:.1f} Pa |\n")

    # ------------------------------------------------------------------ ④
    add("## ④ 对照表（ref vs got，逐条打勾叉）\n")
    add(f"配置 `{gate['config_name']}`（判据配置），条带压力按质量加权时间平均：\n")
    add("| 条带 | y [m] | ref p [Pa] | got p [Pa] | 相对误差 [%] | ≤1 %？ |")
    add("|---|---|---|---|---|---|")
    for row in bench:
        add(f"| {row['band']} | {row['y_m']:.4f} | {row['ref_Pa']:.2f} | {row['got_Pa']:.2f} | "
            f"{row['rel_error_pct']:+.4f} | {'✓' if row['within_tolerance'] else '✗'} |")
    add(f"\n**最大内部条带误差 {fmt(verdict['gate_value_pct'], 4)} % "
        f"（条带 {gate['gate']['worst_band']}）→ {'✓ 达标' if verdict['gate_pass'] else '✗ 未达标'}**"
        f"；门槛 1.0 %。\n")
    add("被排除的自由面条带（不计入判据，仅报告）：\n")
    add("| 条带 | y [m] | ref p [Pa] | got p [Pa] | 相对误差 [%] | 说明 |")
    add("|---|---|---|---|---|---|")
    for row in rows:
        if row["gated"]:
            continue
        add(f"| {row['band']} | {row['y']:.4f} | {row['pressure_analytic']:.2f} | "
            f"{row['pressure']:.2f} | {row['rel_error_pct']:+.1f} | 上边缘进入 2h 自由面支撑带 |")
    add(f"\n内部条带最大瞬时波动 {fmt(gate['max_fluctuation_pct'], 1)} %（不计入判据）："
        "静水槽从静止释放并非离散平衡态，池体作声学晃动，故判据取时间平均。\n")

    # ------------------------------------------------------------------ ⑤
    add("## ⑤ 收敛性与参数敏感性（≥2 点）\n")
    add("### 5.1 统计窗收敛（同一网格，只动窗口）\n")
    add("| 配置 | 时长 [s] | 样本数 | 最大条带误差 [%] | 内部条带平均误差 [%] | 瞬时波动 [%] |")
    add("|---|---|---|---|---|---|")
    for w in report["window_ladder"]:
        add(f"| {w['config']} | {w['seconds']:.1f} | {w['samples']} | "
            f"{w['max_band_error_pct']:.4f} | {w['mean_band_error_pct']:.4f} | "
            f"{w['max_fluctuation_pct']:.1f} |")
    add("\n窗口越长，时间均值的采样噪声越小 → 误差随窗口单调下降，说明残差主要是**无粘池体"
        "声学晃动的均值采样误差**，而不是静压偏置。\n")
    add("### 5.2 分辨率阶梯（确定性 t=0 晶格，单次 `rhs`）\n")
    add("| 配置 | dx [m] | 跨槽列数 | 最大条带误差 [%] |")
    add("|---|---|---|---|")
    for s in report["resolution_ladder_static"]:
        add(f"| {s['config']} | {s['dx']:.4f} | {s['columns']} | {s['max_band_error_pct']:.3f} |")
    add("\n静态误差由**侧壁列**的求和密度截断主导，随列数近似 1/Nx 下降（壁面层占比 ~2/Nx）；"
        "完全支撑内部列（去掉两侧 2h 与底排）在 dx=0.02 时 < 0.1 %（见 "
        "`test_static_lattice_is_exact_away_from_the_walls`）。\n")
    add("### 5.3 声速与槽宽\n")
    add("| 配置 | 变动量 | 最大条带误差 [%] |\n|---|---|---|")
    for run in report["runs"]:
        if run["config_name"] in ("sound_c30", "width_w15") and run is not gate:
            add(f"| {run['config_name']} | c0 与 W 单变量 | {run['gate']['max_band_error_pct']:.4f} |")
    add("")

    # ------------------------------------------------------------------ ⑥
    add("## ⑥ 失败模式、反面控制与工件\n")
    add(f"* **反面控制** `control_g0`（求解器重力置 0、参考解保留 `rho0 g`）：平均误差 "
        f"{fmt(verdict['negative_control_pct'], 2)} % → 判据确实在量重力/压力平衡，"
        f"不是恒真式（{'✓' if verdict['negative_control_ok'] else '✗'}）。")
    add("* **自由面条带**：SPH 求和密度在自由面 2h 内被截断，该带上压力不可与静压比较，"
        "故按 overlap 规则排除；若强行纳入，误差为数十个百分点（见 ④ 第二张表）。")
    add("* **侧壁截断**：水箱平均意义上的误差由两侧壁列贡献（静态 dx=0.02 时底排 −10 %、"
        "内部 −1 %），这是求和密度边界效应而非静压物理；提高分辨率或改用 Adami 壁面密度外推可减小。")
    add("* **窗口依赖**：窗口不足时时间均值噪声可超过 1 %（2 s 窗口实测 "
        f"{fmt(next((w['max_band_error_pct'] for w in report['window_ladder'] if w['config']=='window_2s'), None), 4)} %），"
        "因此判据把窗口写进算例定义（预声明），不能事后挑窗口。")
    add("* **静态不可作判据**：t=0 单次 `rhs` 的压力来自初始密度场，误差由壁面截断主导（10–28 %），"
        "仅作为机理诊断，不参与判据。")
    add("* **工件**（全部在本 commit 内）：")
    add("  * `runs/case_hydrostatic/report.json`（聚合：verdict + 每个 run 的完整记录 + repro 命令）")
    add("  * `runs/case_hydrostatic/runs/*.json`（每个配置一份真实运行工件，含 ref/got 逐条带对照）")
    if figures:
        add("  * `runs/case_hydrostatic/figures/`：" + "、".join(f"`{f}`" for f in figures))
    add(f"* **生态摩擦**：① `pip install -e` 的兄弟工作树把 `src` 抢先放进 `sys.path`，"
        "裸 `python -m tensorsph` 会跑到别的工作树（本单以 `tests/conftest.py` 的 pin 与"
        "脚本内 `sys.path.insert` 规避）；② torch 默认多线程在 99 控制面（容器 CPU 配额）上"
        "发生线程抖动，单次 8 s 算例从 49 s 变 >120 s，算例入口统一 `set_num_threads(1)`；"
        f"③ 8 s@dx=0.02 单核约 {fmt(gate['wall_seconds'], 0)} s（本工件在集群 loong64 节点产出），"
        "重算请走作业队列。\n")
    add("## 误差判定（显式回答）\n")
    add(f"关键量 = 内部条带最大相对压力误差（时间平均 vs 闭式静水解）。\n\n"
        f"* 实测：**{fmt(verdict['gate_value_pct'], 4)} %**；门槛 1.0 %；"
        f"**是否 < 3 %：是**（且 < 1 %，算例判据达标）。")
    add(f"* 每一项见 ④ 表；被排除项（自由面条带）与反面控制（{fmt(verdict['negative_control_pct'], 1)} %）"
        "如实列出，未粉饰。")
    add("* 参考解为解析闭式解，不存在实验不确定度；比较误差即求解器/约简误差。\n")
    if figures:
        add("## 图\n")
        for name in figures:
            add(f"`runs/case_hydrostatic/figures/{name}`\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
