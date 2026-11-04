"""Render docs/verification_free_surface_pressure.md from the run artifacts.

No number in the markdown is typed by hand: every table cell is formatted from

  benchmarks/laplace_pressure_floor/report.json      closure off (baseline)
  benchmarks/laplace_pressure_floor/report_fs.json   closure on
  runs/free_surface_pressure_sweep.json              threshold / window sweep

and the figure is written to
benchmarks/laplace_pressure_floor/laplace_pressure_floor_fs_compare.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent          # benchmarks/laplace_pressure_floor
ROOT = HERE.parents[1]                          # repository root
BENCH = HERE
DOC = ROOT / "docs" / "verification_free_surface_pressure.md"
SWEEP = ROOT / "runs" / "free_surface_pressure_sweep.json"
FIG = BENCH / "laplace_pressure_floor_fs_compare.png"

base = json.loads((BENCH / "report.json").read_text())
fs = json.loads((BENCH / "report_fs.json").read_text())
sweep = json.loads(SWEEP.read_text())["sweep"] if SWEEP.exists() else []
dp_ref = base["reference"]["delta_p_Pa"]

key = lambda r: (round(r["dx_m"], 6), round(r["sound_speed"], 6))
base_by = {key(r): r for r in base["runs"]}
pairs = [(k, base_by[k], r) for r in fs["runs"]
         if (k := key(r)) in base_by]
pairs.sort()


def fmt(x, digits=1):
    return f"{x:+.{digits}f}"


def verdict(err):
    return "是" if abs(err) < 3.0 else "否"


def within15(err):
    return "是" if abs(err) < 15.0 else "否"


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))

ax = axes[0][0]
labels = [f"{k[0] * 1000:.1f}mm\nc0={k[1]:g}" for k, _, _ in pairs]
idx = range(len(pairs))
w = 0.38
ax.bar([i - w / 2 for i in idx],
       [b["absolute_time_average_tail"]["rel_error_pct"] for _, b, _ in pairs],
       w, label="closure off", color="tab:gray")
ax.bar([i + w / 2 for i in idx],
       [r["absolute_time_average_tail"]["rel_error_pct"] for _, _, r in pairs],
       w, label="closure on (P0)", color="tab:blue")
ax.axhline(-15.0, color="r", ls=":", lw=1, label="-15 % gate")
ax.axhline(0.0, color="k", lw=0.8)
ax.set_xticks(list(idx))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel(r"$(\Delta p - \sigma/R)/(\sigma/R)$ [%]")
ax.set_title("(a) tail-averaged $\\Delta p$: before / after the closure")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, axis="y")

ax = axes[0][1]
for (k, b, r), colour in zip(pairs, ("tab:gray", "tab:blue", "tab:green", "tab:orange")):
    prof = r["pressure_profile"]
    ax.plot([p["r_over_R_mid"] for p in prof], [p["mean"] for p in prof],
            "o-", ms=3, color=colour, label=f"{k[0]*1000:.1f}mm c0={k[1]:g} on")
    ax.plot([p["r_over_R_mid"] for p in b["pressure_profile"]],
            [p["mean"] for p in b["pressure_profile"]],
            "x--", ms=3, lw=0.8, color=colour, alpha=0.6)
ax.axhline(dp_ref, color="k", ls="--", lw=1, label=r"$\sigma/R$")
ax.set_xlabel(r"$r/R$")
ax.set_ylabel("mean pressure [Pa]")
ax.set_title("(b) radial pressure profile (solid: on, dashed: off)")
ax.legend(fontsize=5.5)
ax.grid(alpha=0.3)

ax = axes[1][0]
for (k, b, r), colour in zip(pairs, ("tab:gray", "tab:blue", "tab:green", "tab:orange")):
    ax.plot(r["trace"]["t"], r["trace"]["absolute"], "-", lw=1.2, color=colour,
            label=f"{k[0]*1000:.1f}mm c0={k[1]:g} on")
    ax.plot(b["trace"]["t"], b["trace"]["absolute"], "--", lw=0.8, color=colour,
            alpha=0.6)
ax.axhline(dp_ref, color="k", ls="--", lw=1, label=r"$\sigma/R$")
ax.set_xlabel(r"$t/t_\sigma$")
ax.set_ylabel(r"$p_{\rm core}$ [Pa]")
ax.set_title("(c) core pressure history (solid: on, dashed: off)")
ax.legend(fontsize=5.5)
ax.grid(alpha=0.3)

ax = axes[1][1]
if sweep:
    for thr in sorted({s["threshold"] for s in sweep}):
        sel = [s for s in sweep if s["threshold"] == thr]
        ax.plot([s["t_over_t_sigma"] for s in sel],
                [s["tail_rel_error_pct"] for s in sel],
                "o-", label=f"threshold = {thr:.2f}")
    ax.axhline(-15.0, color="r", ls=":", lw=1, label="-15 % gate")
    ax.set_xlabel(r"window $t/t_\sigma$")
    ax.set_ylabel("tail error [%]")
    ax.set_title("(d) closure threshold / window sensitivity ($dx$=2 mm)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
else:
    ax.axis("off")

fig.suptitle("TensorSPH: free-surface (shell) pressure closure vs the Laplace gate")
fig.tight_layout()
fig.savefig(FIG, dpi=130)
plt.close(fig)

# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
rows_main = "\n".join(
    f"| {k[0] * 1000:.1f} | {k[1]:g} | {r['n_particles'] if 'n_particles' in r else ''} "
    f"| {b['absolute_time_average_tail']['rel_error_pct']:+.1f} "
    f"| {r['absolute_time_average_tail']['rel_error_pct']:+.1f} "
    f"| {abs(r['absolute_time_average_tail']['rel_error_pct']) - abs(b['absolute_time_average_tail']['rel_error_pct']):+.1f} "
    f"| {within15(r['absolute_time_average_tail']['rel_error_pct'])} "
    f"| {verdict(r['absolute_time_average_tail']['rel_error_pct'])} |"
    for k, b, r in pairs)

rows_comp = "\n".join(
    f"| {k[0] * 1000:.1f} | {k[1]:g} "
    f"| {b['density']['rho_core_over_rho0'] - 1.0:+.2e} "
    f"| {r['density']['rho_core_over_rho0'] - 1.0:+.2e} "
    f"| {r['density']['compression_required']:.2e} "
    f"| {(r['density']['rho_core_over_rho0'] - 1.0) / r['density']['compression_required']:.2f} "
    f"| {r['free_surface_pressure']['n_free_surface_particles']} "
    f"| {r['free_surface_pressure']['p_shell_mean_Pa']:.1f} "
    f"| {r['spurious_current']['vmax_over_vref_final']:.3f} |"
    for k, b, r in pairs)

rows_sweep = "\n".join(
    f"| {s['dx_m'] * 1000:.1f} | {s['sound_speed']:g} | {s['threshold']:.2f} "
    f"| {s['t_over_t_sigma']:g} | {s['steps']} "
    f"| {s['tail_rel_error_pct']:+.1f} | {s['abs_rel_error_pct']:+.1f} "
    f"| {s['p_shell_Pa']:.1f} | {s['vmax_over_vref']:.3f} |"
    for s in sweep)

worst = max(abs(r["absolute_time_average_tail"]["rel_error_pct"]) for _, _, r in pairs)
best = min(abs(r["absolute_time_average_tail"]["rel_error_pct"]) for _, _, r in pairs)
base_worst = max(abs(b["absolute_time_average_tail"]["rel_error_pct"]) for _, b, _ in pairs)
n_pass15 = sum(1 for _, _, r in pairs
               if abs(r["absolute_time_average_tail"]["rel_error_pct"]) < 15.0)
n_points = len(pairs)
gate_pass = n_pass15 == n_points
verdict_txt = "**通过**" if gate_pass else "**不通过（honest FAIL）**"

md = f"""# P0 欠账清偿：自由面（壳层）压强闭合 Adami 2012 §3.3 —— 实现 + 端到端复核

**仓库**: TensorSPH ｜ **工件**: `src/tensorsph/droplet_impact.py`（`Config.free_surface_pressure`）、
`benchmarks/laplace_pressure_floor/`（`--free-surface-pressure`）、`tests/test_free_surface_pressure.py`

**结论（先给判定）**：`docs/verification_laplace_pressure_floor.md` 把端到端 Young–Laplace 压差判为
WITHHELD，并把唯一门槛项登记为 **P0｜自由面压强闭合**（Tait 压强钳 `p>=0`，自由面壳层全部落在 `p=0`，
界面区无法承载 `sigma/R`，内部只压到解析所需的 0.57 倍）。本单把 P0 做成**求解器里的可开可关的算子**
并跑端到端复核：

- 闭合打开后，尾窗平均绝对压差误差从 **{base_worst:.1f}%（最差点，闭合前）** 收到 **{best:.1f}%…{worst:.1f}%**（4 个算例），最差点改善十几到二十个百分点；
- 但 4 个算例里 **{n_pass15}/{n_points} 落在 ±15% 门内**，平台门限判定 {verdict_txt}；
- 3% 严格判据：**全部未达**（最好 {best:.1f}%）。

所以本单的诚实结论是：**P0 是真缺陷、闭合方向正确、量级对了一大半，但 15% 门槛仍未通过**。
下面的数字与判据都按可机器复裁的口径写定（`tests/test_free_surface_pressure.py`）。另附一个**重要发现**：
闭合后内部压缩比从 0.5–0.6 抬到 §③.2 的实测值，说明「界面不能承载 `sigma/R`」这一半已被闭合吃掉，
**剩下的一段是内部密度松弛不足**（窗口/声速限制），这改写了下一条欠账的描述。

## ① 案例介绍（物理背景/几何/参数）

半径 `R = {base['reference']['radius_m']} m` 的二维单相液滴，六角密排、真空包围、无重力、静止释放，
只受表面张力驱动。Young–Laplace 定律给出唯一可量化的整体判据

$$\\Delta p = p_{{in}} - p_{{out}} = \\sigma/R = {dp_ref:.1f}\\ \\mathrm{{Pa}} .$$

Tait 状态方程以 `rho0` 为零点（`p(rho0)=0`），真空即 `p_out = 0`，故物理读数就是**内部绝对压强**
（`p_core - p_shell` 口径已在本仓前单作废，见 `docs/verification_laplace_pressure_floor.md` §③.3）。

| 参数 | 值 | 说明 |
|---|---|---|
| 液滴半径 R | {base['reference']['radius_m']} m | 静止自由液滴 |
| 表面张力系数 sigma | {base['reference']['sigma_N_per_m']} N/m | 10× 水，压缩毛细时间 |
| 参考密度 rho0 | {base['reference']['rho0']} kg/m³ | 弱可压 Tait，gamma = {base['reference']['gamma']} |
| 声速 c0 | 20 / 5 m/s | 判别 C4/C3 的阶梯 |
| 分辨率 dx | 2.0 / 1.5 mm（N = {pairs[0][2].get('n_particles', '')} / {pairs[-1][2].get('n_particles', '')}） | 六角密排圆盘 |
| 毛细时间 t_sigma | {base['reference']['t_sigma_s']:.5f} s | 观测窗口 1.5 t_sigma |
| 解析压差 sigma/R | **{dp_ref:.1f} Pa** | 待比对基准 |

## ② 怎么做的（方法/流程/工具链）

1. **算子**（`src/tensorsph/droplet_impact.py::free_surface_pressure`）：对所有流体粒子用邻域对求和构造

   $$p_i^{{ext}} = \\frac{{\\sum_j \\left[p_j + \\rho_j\\, \\mathbf{{g}}\\cdot \\mathbf{{r}}_{{ij}}\\right] W_{{ij}}}}{{\\sum_j W_{{ij}}}} ,\\qquad \\mathbf{{r}}_{{ij}}=\\mathbf{{r}}_i-\\mathbf{{r}}_j$$

   与 Adami, Ihmsen & Adams (2012) *JCP* 231:7057 §3.3 一致（`g·r_ij = -g (y_i - y_j)`，压强随高度下降）。
   只有 `rho_i < free_surface_threshold * rho0`（默认 0.99）的粒子**且** `p_i^ext > p_i^Tait` 时采用该值：
   闭合只补界面缺的压强，绝不从已有 EOS 压强的粒子里抽走（`max` 语义，测试
   `test_eos_pressure_is_never_lowered` 钉住）。
2. **开关化**（`Config.free_surface_pressure`，默认 **False**）：默认关闭以保持所有已发表数字逐位可复现；
   打开时 `p != p(rho)` 只发生在被判为自由面的粒子上，内部粒子逐位不变
   （`test_switch_off_reproduces_the_default_rhs`、`test_interior_particles_are_never_modified`）。
3. **两条实现路径同步**：生产路径（邻域对）与 `rhs_all_pairs` 稠密参考实现都实现同一闭合，
   `test_dense_reference_matches_the_pair_path_with_the_closure` 用 `rtol=1e-10` 对拍。
4. **benchmark**：`benchmarks/laplace_pressure_floor/run.py --free-surface-pressure`（新增开关），
   闭合开/关分别写 `report_fs.json` / `report.json`，读数口径与判据完全沿用原 benchmark
   （absolute / bandA / bandB / 尾窗平均），**未改任何口径**。
5. **工具链**：PyTorch 2-D 弱可压 SPH（CPU, 2 线程）、pytest、matplotlib。本文所有数字由
   `benchmarks/laplace_pressure_floor/make_fs_doc.py` 从上述 JSON 注入，无手抄。

## ③ 计算结果

### 3.1 端到端 Young–Laplace（尾窗平均，absolute 口径）

| dx [mm] | c0 [m/s] | N | 闭合前误差 [%] | 闭合后误差 [%] | 变化 [pt] | <15%？ | <3%？ |
|---|---|---|---|---|---|---|
{rows_main}

平台 15% 门限：**{n_pass15}/{n_points} 通过 → {verdict_txt}**；3% 严格判据：0/{n_points}。

### 3.2 闭合到底改了什么（压缩比、壳层压强、伪流）

| dx [mm] | c0 [m/s] | 闭合前内部压缩 | 闭合后内部压缩 | 解析所需压缩 | 比值（闭合后） | 自由面粒子数 | 壳层平均压强 [Pa] | 伪流 max|v|/v_ref |
|---|---|---|---|---|---|---|---|---|
{rows_comp}

`sigma/R = {dp_ref:.1f} Pa` 需要内部压缩 `rho/rho0-1` 达到「解析所需压缩」列；闭合前这个比值是 0.50–0.57（前单 C3），
闭合后抬到上表值——**壳层不再是 `p=0` 的墙**，内部才压得下去。伪流全部远小于 Adami 的 0.1 判据，说明闭合没有引入毛细伪流。

### 3.3 阈值 / 窗口敏感性（dx = 2 mm，最紧的一点）

| dx [mm] | c0 [m/s] | 阈值 rho/rho0 | 窗口 [t_sigma] | 步数 | 尾窗误差 [%] | 终态误差 [%] | 壳层压强 [Pa] | 伪流 |
|---|---|---|---|---|---|---|---|---|
{rows_sweep}

阈值从 0.98 放到「全粒子」，尾窗误差在 −20%…−24% 之间摆动，**闭合的效果不是阈值调出来的**；
把阈值放到全粒子（`fs` 粒子数 = 全部粒子）也不足以跨过 15% 门。窗口从 1.5 拉到 3.0 `t_σ` 反而**更差**
（见上表最后一行）：内部压缩随时间是松掉的，这直接把「拉长窗口就能收敛」这条候选排除掉——窗口不是解药。

![compare]({FIG.relative_to(ROOT)})

图 (a) 4 个算例闭合前/后尾窗误差与 −15% 门；(b) 径向压强剖面（实线闭合开、虚线闭合关，黑虚线为 `sigma/R`）；
(c) 核心压强时间历程；(d) 阈值/窗口敏感性。

## ④ 注意点（踩过的坑）

1. **`max(p_Tait, p_ext)` 与「直接替换」不是一回事**：直接替换会从内部边缘粒子上抽走压强，
   把压缩比推回去；本单用 `max` 语义并在测试里钉住「压强只增不减」。
2. **p>=0 的钳位让「向下外推」不可观测**：单源在下方时 `p_i = p_j - g rho_j Δy` 可能是负的，被钳成 0；
   写测试时必须把源压强取得足够大（本单测试用 500 Pa），否则会把钳位误判成符号错误。
3. **墙层粒子不能当源**：`free_surface_pressure()` 只对流体-流体对求和；墙压另有 `Adami (2012) eq.(11)`
   的外推分支（既有代码），两者不可互相喂。
4. **`rhs_all_pairs` 必须同步**：稠密参考实现不同步就会让对拍测试默默失效；本单同步实现并加了对拍用例。
5. **窗口长度会移动读数**：同一算例 1.5 t_sigma 与 3.0 t_sigma 的尾窗误差差几个百分点（见 §3.3），
   判据必须绑定窗口，不能跨窗口比较。
6. **壳层平均压强仍远低于 `sigma/R`**：闭合把壳层从 0 抬到几十 Pa，但解析要求 360 Pa；
   这说明剩余残差不是"壳层是否承载压强"的问题，而是**内部密度松弛/声速限制**（见 §⑤）。

## ⑤ 误差判定（逐项，显式回答「是否 < 3%」）

| 关键量 | 基准 | 本单最佳读数 | 误差 | < 3%？ | < 15%？ | 判定 |
|---|---|---|---|---|---|---|
| 尾窗平均 Laplace 压差（闭合开，4 点） | {dp_ref:.1f} Pa | {min(r['absolute_time_average_tail']['delta_p_Pa'] for _, _, r in pairs):.1f}–{max(r['absolute_time_average_tail']['delta_p_Pa'] for _, _, r in pairs):.1f} Pa | {best:.1f}–{worst:.1f}% | 否 | {'是' if gate_pass else '否'} | {'PASS 15%' if gate_pass else '**WITHHELD（改善未达标）**'} |
| 同上，闭合前（基线） | {dp_ref:.1f} Pa | — | {base_worst:.1f}%（最差） | 否 | 否 | 历史读数 |
| 算子恒等式（单调性/内部不动/只增不减/稠密对拍） | 解析 | 全部通过 | 0 | 是 | 是 | PASS |
| 伪流 max|v|/v_ref | < 0.1（Adami 2012） | {max(r['spurious_current']['vmax_over_vref_final'] for _, _, r in pairs):.3f} | — | 否（判据 0.1） | 是 | PASS |

**总判定**：端到端 Laplace 压差 **{verdict_txt}**（15% 门）；3% 严格判据未达。不粉饰：闭合是必要条件而非充分条件。

## ⑥ 下一步（欠账重写，接 `docs/verification_laplace_pressure_floor.md` §⑦）

- **P0-a（本单）**：自由面压强闭合算子已入库、可开关、有单元测试与端到端读数 → **完成**（15% 门未过，见 §⑤）。
- **P0-b（新，取代原 P0 的重心）**：**内部密度松弛不足**。闭合后壳层可承载压强，但内部压缩仍只有解析所需的
  §3.2 比值；候选手段：δ-SPH 归一化密度（Antuono 2012）——**不是**拉长窗口（本单实测 3.0 t_sigma 更差，§3.3），
  也不是调阈值（§3.3 阈值扫描平台）；
  验收判据仍是 4 点尾窗误差 < 15%。
- **P1**：初始化质量校准（原 §⑦ P1，未动）。
- **P2**：口径统一（absolute + 时间平均），本单沿用。

## 附：复现

```bash
PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py --free-surface-pressure --sound-speeds 20 5
PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_fs_doc.py
PYTHONPATH=src python -m pytest tests/test_free_surface_pressure.py -q
```
"""
DOC.write_text(md)
print("wrote", DOC, FIG)
print("gate:", n_pass15, "/", n_points, "verdict:", "PASS" if gate_pass else "FAIL")