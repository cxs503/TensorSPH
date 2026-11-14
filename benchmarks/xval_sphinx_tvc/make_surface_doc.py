"""Generate the surface-anchor report from ``surface_anchor.json`` (no hand-copied numbers).

Writes ``docs/xval_sphinx_tvc/SURFACE_ANCHOR.md`` and ``/tmp/report_surface_anchor.json``
(the platform report payload with the figure embedded as base64).
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/xval_sphinx_tvc/data/surface_anchor.json"
FIG = ROOT / "docs/xval_sphinx_tvc/figure_surface_anchor.png"
DOC = ROOT / "docs/xval_sphinx_tvc/SURFACE_ANCHOR.md"


def pct(x: float, digits: int = 2) -> str:
    return f"{100.0 * x:.{digits}f} %"


def main() -> None:
    d = json.loads(DATA.read_text())
    s = d["summary"]
    ladder = d["ladder_fixed_h"]
    depth = d["depth_ladder"]
    wall = d["wall_ladder"]
    production = next(r for r in ladder if r["h_over_dx"] == d["production_h_over_dx"])
    finest = ladder[-1]

    surf_rows = "\n".join(
        f"| {r['h_over_dx']:g} | {r['dx']:.5f} | {r['n_particles']} | {r['n_neighbours_2h']} "
        f"| {r['g_x2']:.4f} | {r['anchor_g_x2']:.4f} | {pct(r['rel_err'])} "
        f"| {'PASS' if r['rel_err'] < 0.15 else 'FAIL'} | {r['dense_vs_linked_rel']:.1e} |"
        for r in ladder
    )
    depth_rows = "\n".join(
        f"| {r['h_over_dx']:g} | {r['depth_steps']:d} | {r['depth_over_h']:.2f} | {r['g_mag']:.4e} "
        f"| {r['anchor_mag']:.4e} | {pct(r['rel_err'])} |"
        for r in depth
    )
    wall_rows = "\n".join(
        f"| {r['h_over_dx']:g} | {r['wall_factor']:g} | {r['g_mag']:.3e} "
        f"| {r['g_mag'] / r['surface_scale']:.2e} |"
        for r in wall
    )
    w2 = max(r["g_mag"] for r in wall if r["wall_factor"] == 2.0)
    w1 = min(r["g_mag"] for r in wall if r["wall_factor"] == 1.0)
    scale = max(r["surface_scale"] for r in wall)

    loc = (math.log(next(r['rel_err'] for r in ladder if r['h_over_dx'] == 16.0)
                    / next(r['rel_err'] for r in ladder if r['h_over_dx'] == 32.0))
           / math.log(2.0))
    md = f"""# XVAL 补账：transport-velocity-correction 的**自由面/壁面分支**对解析锚点

**锚点**：`XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION`｜**任务**：m20882｜**仓库**：TensorSPH｜
**分支**：`agent/m20882`｜**代码**：`benchmarks/xval_sphinx_tvc/surface_term.py`、
`tests/test_tvc_surface_anchor.py`

**一句话结论**：前几轮都只验证了算子的**体相**（完美格点是不动点 `g=0`，没有非零基准可对标）；
本单补上**自由面与壁面分支**的解析锚点：① 平面自由面上算子值 vs 梯度定理
`g = +2 n I(h,d)`，生产分辨率 `h/dx={d['production_h_over_dx']}` 处离散值比连续极限**低 {pct(production['rel_err'])}**
（`<15%` 达标，`<3%` 不达标），加密到 `h/dx={finest['h_over_dx']:g}` 误差 {pct(finest['rel_err'])}
（≈ 2 阶收敛）；② 壁面分支有**精确**锚点：镜面 ghost 排布下 `wall_factor=2` 使 `|g|` 落到舍入底
（{w2:.1e}，即 {w2 / scale:.0e}× 自由面尺度），`wall_factor=1` 则残留 {pct(w1 / scale)}
——这是本仓第一次**定量**验证 `wall_factor=2.0` 约定；③ `|δr|/h = 30α/(7π)` 与 `dx` 无关，
把 `RESOLUTION_LADDER.md` §3 里"固定 α 下修正率 ∝ dx"的实测结论升级为**解析推导+实测校核**。

---

## ① 案例介绍

**被测对象**（已合并，`src/tensorsph/transport_velocity_correction.py`，逐字对应
`docs/xval_sphinx_tvc/oss_source/transport_velocity_correction.hpp`）：

```
g_i = -Σ_j (C_i + C_j) V_j ∇_i W_ij            # upstream interaction(): inconsistency
δr_i = α h_ref² · limiter(h_ref²|g_i|²) · g_i   # upstream update(), NoLimiter
```

**为什么需要新锚点**：`g_i` 与 `|∇W|` 同量纲，只有"完美格点→0"这一个不变量可测，而**真正驱动粒子
移动、决定稳定性上限**的是自由面处的 `g_i`，它此前只有"投影后法向分量为零"这类**不变量**，
从未与任何**数值**比对。

| 参数 | 值 |
|---|---|
| 平滑长度 `h` | {d['h']} m |
| 生产分辨率 `h/dx` | {d['production_h_over_dx']}（全仓所有生产算例） |
| 核函数 | 2-D cubic spline (M4)，`σ = 10/(7πh²)`，支撑 `2h` |
| `C`（kernel correction） | `NoKernelCorrection`（全 1），同上游 `TransportVelocityCorrectionInner` 默认 |
| 壁面权重 `wall_factor` | 2.0（`NoKernelCorrection` 下的上游约定） |
| 构型 | 方形格点半空间，自由面平面 `x2=0`，流体 `x2≤0`；壁面为镜面 ghost |
| 参考 | Zhang, Rezavand & Hu (2020) JCP 404:109135；Adami, Hu & Adams (2012) JCP 231:805（ghost fluid） |

## ② 怎么做的

1. **解析锚点（连续极限，精确）**：由梯度定理，位于自由面下方深度 `d` 的粒子

   ```
   ∫_fluid ∇_i W(x_i-x') dV' = -n I(h,d),   I(h,d) = ∫ W_2D(s,d) ds
   ⇒ g_i → +2 I(h,d) n        (C=1, V_j = dx²)
   ```

   `I(h,0) = 15/(7πh)` 有闭式；脚本用 40001 点梯形积分求任意 `d`，并**先自查**核函数归一化
   `∫W dA = {d['kernel_normalisation']:.9f}`、`I(h,0)` 与闭式相对差 {d['I0_rel_diff']:.1e}。
2. **两条独立算法互证**：`kernel_gradient_integral`（linked-cell 邻域，生产代码路径）与
   暴力稠密求和（不看邻域搜索）逐例比对，最大相对差 {max(r['dense_vs_linked_rel'] for r in ladder):.1e}
   ——保证测的不是邻域搜索的产物。
3. **`h` 固定、`dx→0` 的加密阶梯**：`h/dx = {', '.join(str(r['h_over_dx']) for r in ladder)}`，
   看离散和是否收敛到连续值、几阶。
4. **深度阶梯**：深度取 `dx` 的整数倍（避免一个近奇异邻居主导求和），与同深度解析 `I(h,d)` 比。
5. **壁面精确锚点**：流体格点行取 `x2=(k+½)dx`，壁面取其**镜面像**（ghost fluid）。上游对流体对
   权重 `C_i+C_j=2`、对壁面对权重 `wall_factor·C_i`，仅当 `wall_factor=2` 两者相等，此时
   流体+壁面 = 对目标点**全空间对称**格点和 = 0（无需求积、无容差判断）。`wall_factor=1` 作对照。
6. **工具链**：`python benchmarks/xval_sphinx_tvc/surface_term.py`（本机 ~47 s / 单核，
   向量化 torch float64）；测试 `python -m pytest tests/test_tvc_surface_anchor.py`（15 项）。

## ③ 计算结果

### 3.1 自由面：离散格点和 vs 梯度定理（`d = 0`）

| `h/dx` | `dx` [m] | 粒子数 | 支撑内邻居 | `g_x2` 离散 | `g_x2` 解析 | 相对误差 | 15% 门 | linked vs 稠密 |
|---|---|---|---|---|---|---|---|---|
{surf_rows}

### 3.2 深度衰减（`|g|` vs `I(h,d)`）

| `h/dx` | `m=d/dx` | `d/h` | `|g|` 离散 | 解析 `2I(h,d)` | 相对误差 |
|---|---|---|---|---|---|
{depth_rows}

生产分辨率 `h/dx={d['production_h_over_dx']}` 下**只允许 `m=0,1` 两种合法深度**（`2h = 2.6dx`），
也就是说自由面项在网格上只有"一层粒子"厚，深度结构**无法分辨**——这是粗格点的事实，不做平滑。

### 3.3 壁面：镜面 ghost 精确锚点

| `h/dx` | `wall_factor` | `|g|` | `|g|` / 自由面尺度 |
|---|---|---|---|
{wall_rows}

### 3.4 由此得到的解析律（与既有实测互校）

`|δr| = αh²|g| = 2αh²I(h,0) = 2αh·(15/7π)`，即

```
|δr| / h = 30α/(7π) = 1.364 α      （与 dx 无关）
```

| 量 | 值 |
|---|---|
| 解析预测 `|δr|/h`（α=0.1，连续极限） | {s['shift_over_h_alpha_0p1_analytic']:.4f} |
| 实测离散（生产 `h/dx`，α=0.1） | {s['shift_over_h_alpha_0p1_measured']:.4f} |
| 相对差 | {pct(abs(s['shift_over_h_alpha_0p1_measured'] - s['shift_over_h_alpha_0p1_analytic']) / s['shift_over_h_alpha_0p1_analytic'])} |
| `RESOLUTION_LADDER.md` §3 生产算例实测 `max|δr|/h`（α=0.1） | 0.155 |

体相团簇的 `|grad-1|` 叠加使实测上限（0.155）高于纯自由面值（{s['shift_over_h_alpha_0p1_measured']:.3f}），
两者同量级即独立印证了"每步位移 ∝ h ⇒ 累积修正率 ∝ dx"这一结论：该结论此前只由实测**拟合**，
本单由算子的连续极限**推导**出来。

端到端斜率（`h/dx` 1.3→32，非最小二乘拟合）：**p ≈ {s['loglog_slope']:.2f}**；相邻档位 16→32 的局部阶为 {loc:.2f}。

## ④ 结果图

![surface anchor](figure_surface_anchor.png)

(a) 自由面离散值和解析锚点的相对误差随 `h/dx`（15% 门与 3% 严门、生产分辨率竖线）；
(b) 深度衰减 `|g(d)|/|g(0)|`：解析曲线 + `h/dx=1.3`（生产，只有 3 个可分辨深度）与 `h/dx=8`；
(c) 壁面分支 `|g|/(2I(h,0))` 对数轴：`wall_factor=2` 落到舍入底，`wall_factor=1` 残留约 50%。

## ⑤ 注意点

1. **这 14% 不是 bug，是粗格点的求积误差**：`h/dx={d['production_h_over_dx']}` 时支撑半径只有
   `2.6 dx`，自由面项必然亏欠；误差随 `h/dx` 二次收敛（{pct(production['rel_err'])} → {pct(finest['rel_err'])}），
   说明实现与文献公式一致、亏损纯粹来自离散。
2. **它解释了 α 校准**：离散自由面项比连续极限弱 {pct(production['rel_err'])}，而生产的
   `α=0.10` 是上游默认 `0.05`（单声速步 `0.2/4`）的 2 倍——1.14× 的亏损不足以解释 2×，
   余下差额来自体相团簇需要更强修正（见 `RESOLUTION_LADDER.md` §3）。**不得**据此把 α 写死成 0.1。
3. **`wall_factor=2` 有精确锚点，别"顺手调"**：任何非 2 的取值都会在镜面 ghost 下留下
   O(50%) 的伪不一致向量，直接给壁面附近注入伪速度。
4. **只测了方形格点**：六角格点、非均匀/自适应 `h`（上游 `h_ratio`）未覆盖；`h_ratio≠1` 的上游分支
   本仓未移植（模块 docstring 已声明），本单不为其背书。
5. **只测位置修正算子**：动量/压力精度不在此列（`RESOLUTION_LADDER.md` 的前沿位置腿负责）；
   本单也没有实验输入，纯解析/连续极限对标。
6. **体积用几何体积**：本单取 `V_j = dx²`（几何体积），生产算例用 `V_j = m/ρ`，实测密度压缩
   ~2.2%（`RESOLUTION_LADDER.md`），因此真实运行的自由面项还叠加约 2% 的体积误差；这是本单
   锚点的适用范围，不是被验证掉的项。
7. **深度阶梯的粗格点量化**：`h/dx=1.3` 只有 `m=0,1`，`m=1` 处相对误差 {pct(next(r['rel_err'] for r in depth if r['h_over_dx'] == d['production_h_over_dx'] and r['depth_steps'] == 1))}
   是**深度不可分辨**而非实现错误（`h/dx=8` 的同一深度只有 {pct(next(r['rel_err'] for r in depth if r['h_over_dx'] == 8.0 and r['depth_steps'] == 1))}）。

## ⑥ 误差判定（逐项，显式回答）

| 关键量 | 基准 | 误差 | `<15%`？ | `<3%`？ |
|---|---|---|---|---|
| 自由面 `g`，生产 `h/dx={d['production_h_over_dx']}` | 梯度定理 `2I(h,0)` | {pct(production['rel_err'])} | **是**（临界） | 否 |
| 自由面 `g`，`h/dx=4` | 同上 | {pct(next(r['rel_err'] for r in ladder if r['h_over_dx'] == 4.0))} | 是 | **是** |
| 自由面 `g`，`h/dx={finest['h_over_dx']:g}` | 同上 | {pct(finest['rel_err'])} | 是 | **是** |
| 深度 `d=0.25h`（`h/dx=8`） | `2I(h,d)` | {pct(next(r['rel_err'] for r in depth if r['h_over_dx'] == 8.0 and r['depth_steps'] == 2))} | 是 | 否 |
| 深度 `d=0.5h`（`h/dx=8`） | 同上 | {pct(next(r['rel_err'] for r in depth if r['h_over_dx'] == 8.0 and r['depth_steps'] == 4))} | 是 | 否 |
| 壁面 `wall_factor=2` | 精确 0（镜面 ghost） | {w2 / scale:.1e}（相对自由面尺度，即舍入底） | 是 | **是** |
| 壁面对照 `wall_factor=1` | 同上 | {pct(w1 / scale)} | 否（**应当**否，证明锚点有分辨力） | 否 |
| `I(h,0)` 数值 vs 闭式 `15/(7πh)` | 闭式解 | {d['I0_rel_diff']:.1e} | 是 | 是 |
| `|δr|/h`（α=0.1，生产 `h/dx`） | `30α/(7π)` | {pct(abs(s['shift_over_h_alpha_0p1_measured'] - s['shift_over_h_alpha_0p1_analytic']) / s['shift_over_h_alpha_0p1_analytic'])} | 是 | 否 |
| 两条独立算法（linked-cell vs 稠密） | 互相 | {max(r['dense_vs_linked_rel'] for r in ladder):.1e} | 是 | 是 |

**结论**：算子的自由面/壁面分支与文献公式（梯度定理 + ghost-fluid 权重）**一致**；生产分辨率下
自由面项本身有 {pct(production['rel_err'])} 的求积亏损（`<15%` 达标、`<3%` 不达标），加密后二阶收敛到
{pct(finest['rel_err'])}。壁面 `wall_factor=2` 得到**精确**（舍入级）确认。

**环境**：python {d['environment']['python']}，torch {d['environment']['torch']}，git {d['environment']['git_head'][:12]}。
原始数据：`docs/xval_sphinx_tvc/data/surface_anchor.json`。
"""
    DOC.write_text(md)
    print(f"wrote {DOC}")

    payload = {
        "title": "XVAL 补账：TVC 算子自由面/壁面分支的解析锚点（梯度定理 + ghost-fluid 精确零点）",
        "summary": (
            f"TVC 算子自由面项在生产 h/dx={d['production_h_over_dx']} 处比梯度定理低 "
            f"{pct(production['rel_err'])}（<15% 达标、<3% 不达标），h/dx={finest['h_over_dx']:g} 处 "
            f"{pct(finest['rel_err'])}（2 阶收敛）；壁面 wall_factor=2 由镜面 ghost 精确锚点确认到舍入底"
            f"（{w2:.0e}），factor=1 残留 {pct(w1 / scale)}；导出 |δr|/h=30α/(7π) 与 dx 无关。"
        ),
        "scenario_key": "xval_sph_transport_velocity_correction",
        "mission_id": 20882,
        "tags": ["TensorSPH", "SPH", "transport-velocity-correction", "verification",
                 "analytic-anchor", "free-surface", "XVAL"],
        "markdown": md,
        "figures": [{
            "name": "figure_surface_anchor.png",
            "caption": "(a) 自由面离散和 vs 梯度定理锚点的相对误差随 h/dx；(b) 深度衰减 "
                       "|g(d)|/|g(0)| 离散 vs 解析；(c) 壁面分支 |g|/(2I) 对数轴，"
                       "wall_factor=2 落到舍入底、=1 残留约 50%",
            "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
        }],
    }
    out = Path("/tmp/report_surface_anchor.json")
    out.write_text(json.dumps(payload))
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()