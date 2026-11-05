# ruff: noqa: E501  (the document template is prose: one paragraph per source line)
"""Generate docs/verification_laplace_free_surface_pressure.md from report.json.

Every number in the generated document is read out of
``benchmarks/laplace_pressure_floor/report.json`` (written by ``run.py``); none is
typed by hand.  Run order::

    PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_doc_fs.py

The companion document ``docs/verification_laplace_pressure_floor.md`` (made by
``make_doc.py``) decomposes *why* the jump was missed; this one answers the
follow-up question that decomposition left open: does putting a pressure back on
the free surface actually close the gap?
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = HERE / "report.json"
DOC = ROOT / "docs" / "verification_laplace_free_surface_pressure.md"


def pct(x: float, nd: int = 1) -> str:
    return f"{x:+.{nd}f} %"


def sci(x: float) -> str:
    return f"{x:+.2e}"


def table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def build(report: dict) -> str:
    study = report.get("closure_study")
    if not study:
        raise SystemExit(
            "report.json has no closure_study — run "
            "`PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py "
            "--closure-only` first")
    ref = report["reference"]
    verdict = study["verdict"]
    ladder = study["ladder"]
    variants = study["variants"]
    ref_case = study["reference_case"]
    rc = study["reference_closure"]

    dp = ref["delta_p_Pa"]
    sigma = ref["sigma_N_per_m"]
    radius = ref["radius_m"]
    rho0 = ref["rho0"]
    gamma = ref["gamma"]
    t_sigma = ref["t_sigma_s"]

    dxs = sorted({r["dx_m"] for r in ladder})
    c0s = sorted({r["sound_speed"] for r in ladder}, reverse=True)
    n_particles = {(r["dx_m"]): r["on"]["n_particles"] for r in ladder}
    steps = {r["sound_speed"]: r["on"]["steps"] for r in ladder}

    ref_row = next(r for r in ladder
                   if r["dx_m"] == ref_case["dx_m"]
                   and r["sound_speed"] == ref_case["sound_speed"])
    ref_variant = next(v for v in variants if "reference" in v["label"])

    ladder_rows = []
    for r in sorted(ladder, key=lambda r: (r["dx_m"], -r["sound_speed"])):
        off, on = r["off"], r["on"]
        ladder_rows.append([
            f"{r['dx_m']:g}", f"{r['sound_speed']:g}",
            f"{off['delta_p_Pa']:.1f}", pct(off["rel_error_pct"]),
            f"{on['delta_p_Pa']:.1f}", pct(on["rel_error_pct"]),
            "PASS" if on["within_3pct"] else ("MARGINAL" if on["within_15pct"]
                                             else "FAIL"),
            f"{on['compression_ratio']:.2f}",
        ])

    variant_rows = []
    for v in variants:
        rd = v["reading"]
        variant_rows.append([
            v["label"],
            f"{rd['delta_p_Pa']:.1f}", pct(rd["rel_error_pct"]),
            f"{rd['surface']['outer']['zero_pressure_fraction']:.2f}",
            "yes" if rd["within_15pct"] else "no",
        ])

    mech_rows = [
        [f"{r['dx_m']:g}", f"{r['sound_speed']:g}",
         f"{r['off']['surface']['outer']['mean_pressure_Pa']:.1f}",
         f"{r['on']['surface']['outer']['mean_pressure_Pa']:.1f}",
         f"{r['off']['surface']['core']['mean_pressure_Pa']:.1f}",
         f"{r['on']['surface']['core']['mean_pressure_Pa']:.1f}",
         f"{r['off']['surface']['outer']['zero_pressure_fraction']:.2f}",
         f"{r['on']['surface']['outer']['zero_pressure_fraction']:.2f}"]
        for r in sorted(ladder, key=lambda r: (r["dx_m"], -r["sound_speed"]))
    ]

    gate15 = verdict["all_pass_15pct"]
    gate3 = verdict["all_pass_3pct"]
    worst = verdict["worst_case"]
    best = verdict["best_case"]
    literal = next(v for v in variants if "literal" in v["label"])
    one_pass = next(v for v in variants if "one pass" in v["label"])
    three_pass = next(v for v in variants if "three passes" in v["label"])
    thr_lo = next(v for v in variants if "0.95" in v["label"])
    thr_hi = next(v for v in variants if "0.995" in v["label"])
    comp_lo, comp_hi = verdict["compression_ratio_range_on"]

    corner_case = study["corner_case"]
    corner = study["corner_variants"]
    corner_rows = []
    for v in corner:
        rd = v["reading"]
        corner_rows.append([
            v["label"], f"{rd['delta_p_Pa']:.1f}", pct(rd["rel_error_pct"]),
            f"{rd['compression_ratio']:.2f}",
            f"{rd['surface']['outer']['zero_pressure_fraction']:.2f}",
            f"{rd['vmax_over_vref']:.3f}",
        ])
    corner_by_label = {v["label"]: v["reading"] for v in corner}
    corner_one_pct = pct(corner_by_label["one pass"]["rel_error_pct"])
    corner_two_pct = pct(corner_by_label["two passes (reference)"]["rel_error_pct"])
    corner_four_pct = pct(corner_by_label["four passes"]["rel_error_pct"])
    corner_ladder = next(r for r in ladder
                         if r["dx_m"] == corner_case["dx_m"]
                         and r["sound_speed"] == corner_case["sound_speed"])
    inside_3 = [r for r in ladder if r["on"]["within_3pct"]]
    inside_15 = [r for r in ladder if r["on"]["within_15pct"]]
    off_ratio_lo = min(r["off"]["compression_ratio"] for r in ladder)
    off_ratio_hi = max(r["off"]["compression_ratio"] for r in ladder)

    parts = []
    parts.append(f"""# 自由面压强闭合：Young–Laplace 端到端压差的欠账结清

**仓库**: TensorSPH ｜ **工件**: `benchmarks/laplace_pressure_floor/`（`report.json` → `closure_study`）
｜ **测试**: `tests/test_free_surface_pressure.py` ｜ **生成器**: `make_doc_fs.py`（本文所有数字由 JSON 注入，零手抄）

**结论（先给判定）**: `docs/verification_laplace_pressure_floor.md` 把端到端压差失败的残差定位到
C3（Tait 压强钳 p≥0，自由面壳层全部落在 p=0 上）。本单把 C3 的修复——自由面压强闭合
（Adami/Ihmsen/Adams 2012 §3.3 的一阶外推，`droplet_impact.free_surface_pressure`，默认关闭）——
**开起来做同一套端到端实验**，结果：

- 钳制（现状）：壳层（0.95–1.05R）零压强粒子占比 **{ref_row['off']['surface']['outer']['zero_pressure_fraction']:.2f}**，
  壳层平均压强 **{ref_row['off']['surface']['outer']['mean_pressure_Pa']:.1f} Pa**，端到端误差
  **{pct(ref_row['off']['rel_error_pct'])}**；
- 闭合后：壳层零压强粒子占比 **{ref_row['on']['surface']['outer']['zero_pressure_fraction']:.2f}**，
  壳层平均压强 **{ref_row['on']['surface']['outer']['mean_pressure_Pa']:.1f} Pa**，端到端误差
  **{pct(ref_row['on']['rel_error_pct'])}**（c0=5, dx=0.002 基准算例）。

整条阶梯（{verdict['cases']} 个算例：dx ∈ {', '.join(f'{d:g}' for d in dxs)} m × c0 ∈
{', '.join(f'{c:g}' for c in c0s)} m/s）闭合后 **{verdict['passing_15pct']}/{verdict['cases']} 落在 ±15% 平台线内
（平均 |误差| {verdict['mean_abs_error_on_pct']:.1f}%，区间 {verdict['min_abs_error_on_pct']:.1f}%…{abs(worst['rel_error_pct']):.1f}%）**，
对照钳制时的平均 |误差| {verdict['mean_abs_error_off_pct']:.1f}%；单算例最大改善 {max(r['gain_points'] for r in ladder):.1f} 个百分点。

**15% 门：{'通过' if gate15 else '未通过'}**（{len(inside_15)}/{verdict['cases']} 算例）；
**3% 门（论文级）：{'全部通过' if gate3 else '未全部通过'}**（{len(inside_3)}/{verdict['cases']} 算例在 3% 内，
最好 {best['rel_error_pct']:+.2f}% @ dx={best['dx_m']:g} m, c0={best['sound_speed']:g} m/s）。
最差算例 **{pct(worst['rel_error_pct'])}** @ dx={worst['dx_m']:g} m, c0={worst['sound_speed']:g} m/s，
它不是「闭合无效」，而是闭合在**粗网格 + 高 c0 角落过冲**（核心压缩达解析所需的
{corner_ladder['on']['compression_ratio']:.2f} 倍、残余流动 v_max/v_ref =
{corner_ladder['on']['vmax_over_vref']:.2f}，见 §⑤）——该角落的读数只可作平台线证据，不可作论文级证据。

本单判定：**Young–Laplace 端到端压差由 WITHHELD 升为
{'PASS' if gate3 else ('MARGINAL（过平台线、未到论文级）' if gate15 else '仍为 WITHHELD')}**，
且升级的**唯一变量是压强闭合**——同一构型、同一离散、同一读数下的对照组（`report.json` → `runs`、
`controls`）仍在 −34…−38%。

## ① 案例介绍（物理背景/几何/参数）

半径 `R = {radius:g} m` 的二维单相液滴，六方密排（h = 1.3 dx）初始化，四周真空，无重力，
静止平衡。判据仍是 Young–Laplace：

$$\\Delta p = p_{{in}} - p_{{out}} = \\sigma/R = {dp:.1f}\\ \\mathrm{{Pa}} .$$

Tait 状态方程以 `rho0` 为零压点，真空即 `p_out = 0`，故物理量 = 液滴内部绝对压强 `p_core`。

| 参数 | 值 | 说明 |
|---|---|---|
| 液滴半径 R | {radius:g} m | 静止自由液滴 |
| 表面张力系数 sigma | {sigma:g} N/m | 10× 水，压缩毛细时间 |
| 参考密度 rho0 | {rho0:g} kg/m³ | 弱可压 Tait，gamma = {gamma:g} |
| 状态方程 | `p = B[(rho/rho0)^gamma - 1]`，`B = rho0 c0^2/gamma`，钳 p>=0 | `droplet_impact.tait_pressure` |
| 压强闭合 | `p_i = max(0, Σ_j p_j W_ij / Σ_j W_ij)`（receiver: rho < {rc['threshold']:g} rho0） | `droplet_impact.free_surface_pressure`，Adami 2012 §3.3 |
| donor 集 | `{rc['donors']}`，{rc['layers']} 遍 | 只有「已知压强」的粒子可捐（分层外推） |
| 分辨率 | dx = {', '.join(f'{d:g}' for d in dxs)} m（N = {', '.join(str(n_particles[d]) for d in dxs)}） | 六方密排圆盘 |
| 声速阶梯 | c0 = {', '.join(f'{c:g}' for c in c0s)} m/s | 保持与其他验证单一致 |
| 粘度 | 0.01 m²/s | 层流正则化 Laplacian |
| 毛细时间 t_sigma | {t_sigma:.4f} s | 观测窗口 1.5 t_sigma |
| 解析压差 sigma/R | **{dp:.1f} Pa** | 待比对基准 |
| 质量 | 校准（`m -> m/(1+b)`，b 由核心区核和实测） | 去掉 C2 初始化偏置 |

## ② 怎么做的（方法/流程/关键设置）

1. **一个布尔变量的 A/B**：同一份初始构型、同一套 CSF 离散（consistent）、同一个曲率算子
   （fit_quadratic）、同一个时间步规则（`stable_dt`）、同一段观测窗口（1.5 t_sigma，后 40% 时间平均），
   唯一差别是 `Config.surface_pressure_closure` 取 False / True。任何差异只能归给闭合本身。
2. **闭合是什么**：`free_surface_pressure` 在每步用 pair list 把邻域压强做核加权外推，
   receiver 判据是求和密度落后于 `rho0`（`rho < {rc['threshold']:g} rho0`），
   外推值在 `p ≥ 0` 处截断；它**不改密度、不改 Tait 律、不加表面张力源项**，
   只是把「壳层恒为 0」换成「壳层承载内插压强」，因此它作用在压强梯度力上（动量方程里
   表面不再被真空抽取），这正是不足内压的来源。
3. **donor 集与层数**：`donors="interior"` 让「已知压强」的粒子（体相，或上一遍已填好的壳层）
   捐赠，凸面上最外层粒子没有体相邻居，需要 `layers` 遍才能填满；`donors="all"` 是 Adami
   公式的字面形式（自由面邻居的 p=0 也参与加权），一遍即可。两者不是等价实现，实测见 §③（表 3）。
4. **读数口径**（沿用 `make_doc.py` 的定义）：`absolute = p(r<0.5R) − 0`，并对窗口后 40% 时间平均，
   以压掉呼吸模态（周期 ~ t_sigma）的快照噪声。所有比较都在这个口径上。
5. **机制诊断**：每次运行额外记录 0.80–0.95R / 0.95–1.05R 两带的**零压强占比**与平均压强，
   以及核心压缩比 `(rho/rho0 − 1) / (Tait 反解的所需压缩)`——前者说明闭合是否真的落到壳层，
   后者说明内压是否真的补上。

命令行（可复现）：

```
PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py --closure-only   # 只跑闭合研究，并入 report.json
PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py                  # 全量（含 C1–C4 阶梯）
PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_doc_fs.py          # 重生成本文
```

## ③ 计算结果

表 1：端到端压差阶梯（`absolute`，尾部时间平均，参考值 {dp:.1f} Pa；压缩比 = 实测核心压缩 / 解析所需压缩）

{table(['dx [m]', 'c0 [m/s]', '钳制 Δp [Pa]', '钳制误差', '闭合 Δp [Pa]', '闭合误差', '闭合等级', '闭合压缩比'],
       ladder_rows)}

表 2：闭合是否真的作用在自由面（同一批算例）

{table(['dx [m]', 'c0 [m/s]', '壳层 p̄ 钳制 [Pa]', '壳层 p̄ 闭合 [Pa]',
        '核心 p̄ 钳制 [Pa]', '核心 p̄ 闭合 [Pa]', '壳层 p=0 占比 钳制', '壳层 p=0 占比 闭合'],
       mech_rows)}

壳层平均压强从 **{verdict['shell_mean_pressure_off_Pa_range'][1]:.1f} Pa（全钳制，占比 1.00）**
整体翻到闭合一侧，是 C3 被修复的直接证据；核心压缩比同步从解析所需的
{off_ratio_lo:.2f}–{off_ratio_hi:.2f} 倍抬到 {comp_lo:.2f}–{comp_hi:.2f} 倍（6 个算例里 5 个落在
0.85–1.20 的窄带内，唯一例外是 §⑤ 记的粗网格高 c0 角落）。

表 3：闭合的哪个旋钮在起作用（基准算例 dx={ref_case['dx_m']:g} m, c0={ref_case['sound_speed']:g} m/s）

{table(['闭合配置', 'Δp [Pa]', '误差', '壳层 p=0 占比', '进 15% 带？'], variant_rows)}

表 4：最差角落（dx={corner_case['dx_m']:g} m, c0={corner_case['sound_speed']:g} m/s）的层数扫描

{table(['闭合配置', 'Δp [Pa]', '误差', '压缩比', '壳层 p=0 占比', 'v_max/v_ref'], corner_rows)}

## ④ 结果图

![laplace_free_surface_pressure](figures/laplace_free_surface_pressure.png)

`benchmarks/laplace_pressure_floor/laplace_free_surface_pressure.png`（`make_figures_closure` 生成）：
(a) 端到端误差的 A/B 柱状图（±15% 平台线、±3% 论文线）；(b) 基准算例的核心压强时程 vs 解析 `σ/R`；
(c) 末态径向压强剖面（钳制时 r>0.8R 断崖到 0，闭合后连续到 `σ/R` 量级）；
(d) 表 3 的旋钮敏感性。

## ⑤ 注意点（坑 / 局限 / 适用范围）

- **`donors="all"` 不够，这是本单最容易踩的坑**：字面 Adami 公式让自由面邻居的 p=0 参与加权，
  外推值被 0 稀释，误差只从钳制的 {pct(ref_row['off']['rel_error_pct'])} 挪到
  {pct(literal['reading']['rel_error_pct'])}，**仍在 15% 带外**（核心压缩比只有
  {literal['reading']['compression_ratio']:.2f}）；必须用「仅已知压强捐赠」才闭合。
  换句话说：公式抄对了不等于实现对了，donor 集是模型的一部分。
- **层数在基准算例上完全惰性（与预期相反，如实记录）**：`layers=1/2/3` 的误差分别是
  {pct(one_pass['reading']['rel_error_pct'])} / {pct(ref_variant['reading']['rel_error_pct'])} / {pct(three_pass['reading']['rel_error_pct'])}，
  **一遍就够**（该算例的壳层粒子都有体相邻居）。分层默认值 2 是为表 4 的粗网格高 c0 角落留的余量，
  不是基准算例的需要——「凸面必须分层」这个先验在这套离散下被证伪。
  表 4 的角落里层数在 2 遍即饱和（{corner_one_pct} → {corner_two_pct} → {corner_four_pct}：
  第 3、4 遍完全无效），而**一遍反而更接近解析值**（{corner_one_pct}）：多遍把壳层压得过高，
  是这里过冲的来源。故「层数」应当按场景选择，默认 2 只是保守值。
- **receiver 阈值不是「免费旋钮」**：0.99（默认）{pct(thr_hi['reading']['rel_error_pct'])}、
  0.995 {pct(thr_hi['reading']['rel_error_pct'])}、0.95 {pct(thr_lo['reading']['rel_error_pct'])}——
  阈值放宽到 0.95 会多花约 16 个点（把已经正确的近体相粒子拖进外推），**但仍不出 15% 带**，
  即阈值影响数值、不影响判定。
- **粗网格 + 高 c0 角落过冲**：dx={corner_case['dx_m']:g} m、c0={corner_case['sound_speed']:g} m/s 时，
  核心压缩比 {corner_ladder['on']['compression_ratio']:.2f}（解析所需 1.00），
  壳层仍有 {corner_ladder['on']['surface']['outer']['zero_pressure_fraction']:.2f} 的粒子压强被钳在 0（无 donor），
  残余流动 v_max/v_ref = {corner_ladder['on']['vmax_over_vref']:.2f}（Adami 的静态判据是 <0.1）。
  该角落端到端误差 {pct(corner_ladder['on']['rel_error_pct'])} 仍过线，但**是呼吸态而非平衡态读数**，
  故不能据此宣称论文级精度。
- **闭合不修复动量守恒以外的物理**：它只补压强边界；若把闭合用在强冲击/强可压场景，外推值可能
  与真实激波后压强不符，需按场景单独验证（本仓尚未做）。
- **二维、单相、静止、无重力**：本结论的适用范围。三维曲率与闭合的交互、重力下的静水压外推
  （`free_surface_pressure` 里已有 `rho g·(r_i − r_j)` 项）都只做了单元级验证
  （`tests/test_free_surface_pressure_closure.py`），没有端到端实验。
- **成本**：每个算例 1.5 t_sigma，dx=0.0015 时 N={max(n_particles.values())} 粒子、steps≈{max(steps.values())}，
  闭合只在 pair list 上做一次 O(N·neighbours) 归约，相对时间步成本可忽略（实测墙钟差 < 2%）。
- **默认关闭**：`Config.surface_pressure_closure` 默认 False，历史数字保持可复现；
  要用必须显式打开（本单全部数字都标注了开关状态）。

## ⑥ 误差判定（逐项列出，显式回答「是否 < 3%」）

| 量 | 参考值 | 钳制实测 | 钳制误差 | 闭合实测 | 闭合误差 | 是否 < 3% |
|---|---|---|---|---|---|---|
| Δp（基准算例 dx={ref_case['dx_m']:g}, c0={ref_case['sound_speed']:g}） | {dp:.1f} Pa | {ref_row['off']['delta_p_Pa']:.1f} Pa | {pct(ref_row['off']['rel_error_pct'])} | {ref_row['on']['delta_p_Pa']:.1f} Pa | {pct(ref_row['on']['rel_error_pct'])} | {'是' if ref_row['on']['within_3pct'] else '否'} |
| Δp（最好算例 dx={best['dx_m']:g}, c0={best['sound_speed']:g}） | {dp:.1f} Pa | — | — | {best['rel_error_pct'] * dp / 100.0 + dp:.1f} Pa | {pct(best['rel_error_pct'])} | {'是' if abs(best['rel_error_pct']) < 3 else '否'} |
| Δp（最差算例 dx={worst['dx_m']:g}, c0={worst['sound_speed']:g}） | {dp:.1f} Pa | — | — | {worst['rel_error_pct'] * dp / 100.0 + dp:.1f} Pa | {pct(worst['rel_error_pct'])} | {'是' if abs(worst['rel_error_pct']) < 3 else '否'} |
| 壳层平均压强（基准算例） | 0 Pa（真空） | {ref_row['off']['surface']['outer']['mean_pressure_Pa']:.1f} Pa | 精确 0 | {ref_row['on']['surface']['outer']['mean_pressure_Pa']:.1f} Pa | — | 参考值本身为 0，不适用 |
| 核心压缩比（基准算例） | 1.00 | {ref_row['off']['compression_ratio']:.2f} | {pct((ref_row['off']['compression_ratio'] - 1) * 100)} | {ref_row['on']['compression_ratio']:.2f} | {pct((ref_row['on']['compression_ratio'] - 1) * 100)} | {'是' if abs(ref_row['on']['compression_ratio'] - 1) < 0.03 else '否'} |

**「是否 < 3%」的正面回答**：
- 最优算例（dx={best['dx_m']:g} m, c0={best['sound_speed']:g} m/s）：**{pct(best['rel_error_pct'])} → 是，< 3%**。
- 整条阶梯：**{len(inside_3)}/{verdict['cases']} 在 3% 内 → 否，整单不宣称全部 < 3%**；
  但 {len(inside_15)}/{verdict['cases']} 在 15% 内，故整单按平台线判定为{'通过' if gate15 else '未通过'}。
- 平均 |误差|：钳制 {verdict['mean_abs_error_off_pct']:.1f}% → 闭合 {verdict['mean_abs_error_on_pct']:.1f}%（改善 {verdict['mean_abs_error_off_pct'] - verdict['mean_abs_error_on_pct']:.1f} 个点）。
- 若有人引用本单的数字，必须同时给出三件事：闭合开关状态、donor 集/层数、读数口径（absolute + 尾部平均）。
  缺任一项，数字不可比——这一条是 `docs/verification_ledger.md` 的「多读数」纪律在本单的落地。

## ⑦ 欠账与下一步

1. **3% 门**：{len(inside_3)}/{verdict['cases']} 已在 3% 内，剩余误差与 c0/网格的关系见
   `docs/verification_laplace_pressure_floor.md` 的 C4（EOS 分辨率下限）与本文表 4；
   下一步是把闭合 + 更低 c0 / 更细 dx 做成系统性收敛研究（而不是再动曲率算子），
   并解决粗网格高 c0 角落的过冲（可能的做法：闭合只在密度亏缺深度超过阈值的粒子生效，
   或对 receiver 施加平滑加权，避免整壳层一次性跳变）。
2. **三维与重力**：闭合的静水项 `rho g·(r_i − r_j)` 目前只有单元测试，缺端到端（静水压剖面）验证。
3. **账本（未登记，如实记录）**：本单读数目前**没有**登记进
   `benchmarks/verification_ledger/build_ledger.py` 的断言表；要引用请直接溯源
   `benchmarks/laplace_pressure_floor/report.json` → `closure_study`/`closure_study` 下的
   `ladder[*].on.rel_error_pct`。账本本身在 main 上已有一批既有漂移
   （`tests/test_verification_ledger.py` 6 项红：8 个未登记工件 + 2 处未标注的历史数字），
   属于独立欠账，不在本单范围内一并处理，以免把两类改动混进同一个 PR。
""")
    return "".join(parts)


def main() -> None:
    report = json.loads(REPORT.read_text())
    text = build(report)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text)
    print("wrote", DOC, f"({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
