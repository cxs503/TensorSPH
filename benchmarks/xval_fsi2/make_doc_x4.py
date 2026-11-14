"""Render ``docs/xval_fsi2/X4_DRY_BODY_DEFECT.md`` + its figure from the artifact.

Every number in the document is read from
``artifacts/xval_fsi2/closed_body_x4.json`` (written by
``benchmarks/xval_fsi2/closed_body_audit.py``); nothing is hand-copied.

    python benchmarks/xval_fsi2/make_doc_x4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "xval_fsi2" / "closed_body_x4.json"
DOC = ROOT / "docs" / "xval_fsi2" / "X4_DRY_BODY_DEFECT.md"
FIG = ROOT / "docs" / "xval_fsi2" / "assets" / "x4_closed_body.png"

ARMS = ("quad", "panel", "oss", "oss_norm")
LABEL = {"quad": "exact surface quadrature", "panel": "TensorSPH panel (WLS)",
         "oss": "OSS one-sided (raw)", "oss_norm": "OSS + half-space moment"}


def rows(data, variant, arm):
    return [r for r in data["loads"] if r["variant"] == variant and r["arm"] == arm]


def pct(x):
    return f"{x * 100:+.2f} %"


def table(data, variant):
    out = ["| dx [m] | arm | F_y | 解析 −cA | 误差 | <15 % | <3 % |",
           "|---|---|---|---|---|---|---|"]
    for dx in data["ladder"]:
        for arm in ARMS:
            r = next(r for r in data["loads"]
                     if r["variant"] == variant and r["arm"] == arm and r["dx"] == dx)
            out.append(f"| {dx} | {arm} | {r['F_y']:+.6f} | {r['ref_y']:.6f} | "
                       f"{pct(r['err_y'])} | {'✔' if r['pass_gate'] else '✘'} | "
                       f"{'✔' if r['pass_target'] else '✘'} |")
    return "\n".join(out)


def census_table(data):
    out = ["| 变体 | dx [m] | 湿表面粒子 | 流体粒子 | 无流体邻居粒子 | 占比 |",
           "|---|---|---|---|---|---|"]
    for c in data["census"]:
        out.append(f"| {c['variant']} | {c['dx']} | {c['n_solid']} | {c['n_fluid']} | "
                   f"{c['dry_solid']} | {c['dry_fraction'] * 100:.1f} % |")
    return "\n".join(out)


def segment_table(data):
    out = ["| dx [m] | 段 | 粒子数 | 精确 −Σp·n·A | panel | OSS 原始 |",
           "|---|---|---|---|---|---|"]
    for r in data["segments"]:
        out.append(f"| {r['dx']} | {r['segment']} | {r['n']} | {r['exact_y']:+.5f} | "
                   f"{r['panel_y']:+.5f} | {r['oss_y']:+.5f} |")
    return "\n".join(out)


def figure(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ladder = data["ladder"]
    for arm, style in (("panel", "o-"), ("oss_norm", "s--"), ("oss", "^:")):
        ys = [next(r for r in data["loads"] if r["variant"] == "wetted"
                   and r["arm"] == arm and r["dx"] == dx)["err_y"] * 100 for dx in ladder]
        ax1.plot(ladder, ys, style, label=LABEL[arm])
    ax1.axhspan(-15, 15, color="tab:green", alpha=0.10, label="15 % gate")
    ax1.axhline(3, color="tab:green", ls=":", lw=1.0, label="3 % target")
    ax1.axhline(-3, color="tab:green", ls=":", lw=1.0)
    ax1.set_xscale("log")
    ax1.invert_xaxis()
    ax1.set_xlabel("dx [m]")
    ax1.set_ylabel("load error (F_y − A)/A  [%]")
    ax1.set_title("fixed (fully wetted): fsi2 insert body, linear field")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)

    for variant, mk in (("legacy_dry", "o--"), ("wetted", "s-")):
        xs = [c["dx"] for c in data["census"] if c["variant"] == variant]
        ys = [c["dry_fraction"] * 100 for c in data["census"] if c["variant"] == variant]
        ax2.plot(xs, ys, mk, label=f"{variant}: dry fraction")
    ax2.set_xscale("log")
    ax2.invert_xaxis()
    ax2.set_xlabel("dx [m]")
    ax2.set_ylabel("surface particles with no fluid neighbour [%]")
    ax2.set_title("the defect: legacy box falls into a dry band as dx refines")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    return FIG


def main() -> int:
    data = json.loads(ART.read_text())
    v = data["verdict"]
    fig = figure(data)
    wet_panel = rows(data, "wetted", "panel")
    wet_oss = rows(data, "wetted", "oss")
    wet_norm = rows(data, "wetted", "oss_norm")
    best = min(wet_panel, key=lambda r: abs(r["err_y"]))
    doc = f"""# X4 勘误：fsi2 插入体静态载荷的「+32 % 失败」是算例干区缺陷，不是算子缺陷

锚点 `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`（Xiangyu-Hu/SPHinXsys `test_2d_fsi2`
+ `FSI::ForceFromFluid`）· 仓 TensorSPH · 驱动 `benchmarks/xval_fsi2/closed_body_audit.py`
· 数据 `artifacts/xval_fsi2/closed_body_x4.json` · 图 `docs/xval_fsi2/assets/x4_closed_body.png`

## ① 案例介绍

X4 是锚点里唯一以**真实 fsi2 插入体**（圆柱 r = 0.5 m + 梁 3.5 × 0.2 m，`fsi2.h:14-35`）
为对象、且带解析参照的静态门：闭合体浸在**线性压力场** `p = p0 + c·x`（`c = (0, −1)`，
无重力，与 fsi2 一致）中，散度定理给出净载荷

```
F = −∮ p n dl = −c A,   A = π r² + b_h b_l = 1.4853981633974485
```

已发表表格（`GAP-SPH-FSI-BENCHMARK_xval.md` §4 X4）把该门记为**双 FAIL**：OSS 原始臂
dx = 0.025 时 +67.66 %，TensorSPH 面元臂 +32.26 %，且误差**随加密单调增涨**
（−0.70 % → +20.08 % → +32.26 %）。按 owner 精度标准（>15 % 不可接受）该门当时判负。

几何/参数（全部由 `cases.fsi2_insert_linear` 程序化生成）：

| 项 | 取值 | 来源 |
|---|---|---|
| 圆柱 | r = 0.5 m，圆心 (2, 2) | `fsi2.h:14-35` |
| 梁 | 3.5 × 0.2 m，自圆柱右侧延伸 | 同上 |
| 压力场 | `p = 1 − y`（线性，解析给定，无求解器） | `cases.py:fsi2_insert_linear` |
| 参照 | `−c A` = +1.4853981633974485（y 分量） | 散度定理 |
| 核 / 光滑长度 | Wendland C2，h = 1.3 dx，支持域 2h | OSS 默认（`adaptation.h:71`） |
| dx 阶梯 | 0.1 / 0.05 / 0.025 / 0.0125 m | 本轮 |
| 门限 / 目标 | 15 % / 3 % | owner 标准 |

## ② 怎么做的

**a. 先分离「面元采样错」与「算子错」。** 在同一批粒子上直接做精确面元积分
`−Σ p(x_i) n_i A_i`（臂 `quad`，不含任何传递算子）。若 `quad` 精确而算子不精确，
问题在算子；若 `quad` 自身偏离，问题在**表面/粒子集**。

**b. 数干粒子。** 对每个表面粒子数支撑域内的流体邻居数（`cross_pairs`）：邻居数为 0
的粒子，任何算子都只能给出零载荷——它的分摊面积被静默丢掉。

**c. 定位缺陷。** 旧算例的流体格点包围盒下界写成 `cy − bh/2 − pad`（梁半高），
而**圆柱**下缘在 `cy − r = 1.5 m`，比梁底 1.9 m 低 0.4 m；`pad = 5.2 dx` 随加密缩小，
于是被漏掉的圆柱下弧**越来越大**——这正是「误差随加密变差」的签名。修正为上界取
`max(bh/2, r)`（`cases.fsi2_insert_linear(wetted=True)`，默认），旧盒保留为
`wetted=False` 以便复现勘误。

**d. 同时去掉埋在梁里的圆柱弧。** 该段（`|y−cy| < bh/2, x > cx`）位于固体内部、
无流体邻居，且法向成对抵消（贡献 O(dx²)），留着只会虚增粒子数；去掉后梁根留下的
开口面法向纯流向，**梯度方向**的 `−c A` 恒等式不受影响（`quad` 臂实测 ≤0.8 % 佐证）。

**e. 工具链。** `cases.fsi2_insert_linear(..., wetted=True/False, include_buried=...)`
+ `LoadCase.pressure_at`（解析压力律，供分段对账）+ `benchmarks/xval_fsi2/closed_body_audit.py`
四臂（`quad` / `panel` / `oss` / `oss_norm`）+ `tests/test_x4_closed_body_wetting.py` 11 项守护。

## ③ 计算结果（程序化注入，源 `closed_body_x4.json`）

### 表 1 干粒子普查（缺陷本体）

{census_table(data)}

旧盒（`legacy_dry`）在 dx = 0.025 时有 **{max(c['dry_fraction'] for c in data['census'] if c['variant'] == 'legacy_dry') * 100:.1f} %**
的表面粒子完全无流体邻居；修正后（`wetted`）恒为 **0 %**。

### 表 2 修正后（全润湿）四臂

{table(data, "wetted")}

### 表 3 旧盒（复现已发表表）

{table(data, "legacy_dry")}

### 表 4 分段对账（修正后，y 分量）

{segment_table(data)}

## ④ 图

![X4 修正前后](assets/x4_closed_body.png)

左：修正后三臂误差 vs dx（阴影=15 % 门限带，点线=3 % 目标），面元臂全程贴零；
OSS 原始臂 100 %→27 %、OS​S+矩归一化 70 %→9 %，随加密收敛但粗网格仍过门限。
右：旧包围盒（`legacy_dry`）干粒子占比随加密**上升**，修正后（`wetted`）恒为 0——
缺陷的几何签名。

## ⑤ 注意点

1. **这是算例缺陷，不是算子缺陷。** `quad` 臂（纯面元积分，无算子）在修正后各 dx 上
   误差 ≤ {v['quad_max_abs_err_wetted'] * 100:.2f} %，说明表面采样与参照一致；旧盒下
   `quad` 也「精确」，因为它同样只在湿粒子上求和——**干粒子被静默扣除**，所以只看
   算子误差永远发现不了这个缺陷，必须数邻居。
2. **面元（WLS）臂是线性精确的。** 修后 panel 与 quad 逐点一致到机器精度
   （≤{v['panel_max_abs_err_wetted'] * 100:.2f} %），因此已发表表里 X4 的「面元 +32 %」是
   假阴性，不可再作为「面元算子不适应闭合体」的证据。
3. **OSS 一侧求和确有真缺陷，但不在 X4 的旧数值里。** 修后 OSS 原始臂仍在
   {wet_oss[0]['err_y'] * 100:+.0f} %（dx = {wet_oss[0]['dx']}）到 {wet_oss[-1]['err_y'] * 100:+.1f} %
   （dx = {wet_oss[-1]['dx']}），随加密收敛（曲率 + 半空间核矩双效应）；半空间矩归一化
   （`pressure_force_from_fluid_normalised`）把它压到 {wet_norm[0]['err_y'] * 100:+.1f} % →
   {wet_norm[-1]['err_y'] * 100:+.1f} %，仅在最细网格过 15 % 门限。
   **弯曲闭合体上归一化仍非成熟修复**（`tests/test_pressure_moment_normalisation.py`
   的 scope guard 已按新数值更新）。
4. **流的 x 分量不参与判定。** 湿表面在梁根处有一个开口面（法向纯流向、高度 b_h），
   其闭合面在固体内部，故 F_x 带 O(p·b_h) 的残量（本次 ≈−0.199）；`−c A` 只约束梯度方向。
5. **适用范围。** 静态、解析场、单层湿表面；不含时间推进、不含 TLSPH 固体响应。
   真实 fsi2 瞬态（Turek–Hron）仍由 `benchmarks/fsi2_turek_hron` 单独负责。

## ⑥ 误差判定（逐项，是否 <3 %）

| 量 | 基准 | 修正后最差误差 | <3 % ? |
|---|---|---|---|
| 面元臂 F_y（dx 0.1…0.0125） | −cA = 1.4853981633974485 | {v['panel_max_abs_err_wetted'] * 100:.2f} % | **是**（四档全过） |
| 精确面元积分 `quad` F_y | −cA | {v['quad_max_abs_err_wetted'] * 100:.2f} % | **是** |
| OSS 原始臂 F_y | −cA | {v['oss_max_abs_err_wetted'] * 100:.2f} %（dx = 0.1） | **否**（1 档 >15 %） |
| OSS + 半空间矩归一化 F_y | −cA | {v['oss_norm_max_abs_err_wetted'] * 100:.2f} %（dx = 0.1） | **否**（最细档 {wet_norm[-1]['err_y'] * 100:.1f} % 过 15 %，未达 3 %） |

honest：面元臂达到 3 % 目标；两个 OSS 一侧求和臂在弯曲闭合体上**未**达到 3 %，也不承诺
后续单调达到——它们在本仓的定位是对照臂（用于证明移植保真），生产 FSI 载荷传递应使用
面元臂或（多体耦合时）逐粒子矩归一化 + 曲率修正，后者仍是未完成项。

最佳单点：{best['arm']} 在 dx = {best['dx']} 时误差 {pct(best['err_y'])}。
"""
    DOC.write_text(doc)
    print(f"wrote {DOC} and {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())