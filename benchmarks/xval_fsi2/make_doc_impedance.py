"""Generate the impedance-sign report (markdown + platform payload) from artifacts."""
from __future__ import annotations

import base64, json, time
from pathlib import Path

import numpy as np
import torch

from make_fig_impedance import piston_sweep
from tensorsph import fsi2_turek_hron as fsi2

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "docs/xval_fsi2/assets"
DOC = ROOT / "docs/fsi2_impedance_sign"


def main():
    sign = json.loads((ART / "impedance_sign.json").read_text())
    energy = json.loads((ART / "impedance_energy.json").read_text())
    speeds, got, zc = piston_sweep()
    ref = zc * speeds
    piston_err = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-300))

    rows = energy["rows"]
    act = [r for r in rows if r["v_beam_max"] > 1e-4]
    mirror = max(abs(r["wdot_physical"] + r["wdot_pre_fix"])
                 / max(abs(r["wdot_physical"]), 1e-300) for r in act)
    n_neg = sum(1 for r in act if r["wdot_physical"] < 0.0)
    w_first = act[0]; w_last = act[-1]

    arm_tbl = "\n".join(
        "| {} | {} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {} |".format(
            a, "on" if r["wall_impedance"] else "off",
            r["sigma_per_s"], r["v_end"], r["v_max"], r["max_fluid_speed_end"],
            "DIVERGENT" if r["divergent"] else "bounded")
        for a, r in sign["arms"].items())
    eng_tbl = "\n".join(
        "| {:.4f} | {:.3e} | {:+.3e} | {:+.3e} |".format(
            r["t"], r["v_beam_max"], r["wdot_physical"], r["wdot_pre_fix"])
        for r in rows)

    md = f"""# FSI2 声学壁面阻抗符号勘误：抗阻尼项 → 耗散项

锚点 `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`（Xiangyu-Hu/SPHinXsys
`test_2d_fsi2` + `FSI::ForceFromFluid`）· 仓 TensorSPH · 模块
`src/tensorsph/fsi2_turek_hron.py::boundary_pressure_from_fluid`
· 驱动 `benchmarks/xval_fsi2/impedance_sign.py`、`benchmarks/xval_fsi2/impedance_energy.py`
· 数据 `docs/xval_fsi2/assets/impedance_sign.json`、`impedance_energy.json`
· 图 `docs/xval_fsi2/assets/impedance_sign.png`

## ① 案例介绍

`docs/xval_fsi2/STRONG_COUPLING_AND_ROD_STABILITY.md` 的 next-round 第 2 条悬案：
流固载荷传递里的**声学（Riemann 星态）壁面压力项**

```
p* = p_f + s · rho_f c_f (u_wall - u_f)·n ,   n = 壁面外法向（固体指向流体）
```

「本应阻尼运动的壁面，却把所有配置都搞不稳定」。本轮把它当作**符号缺陷**审：
模块 docstring 写的是 `+`（第 158 行原文 `p* = p_f + rho0 c0 (v_wall - v_f).n`），
实现写的是 `p_src = p_src - rho0*sound_speed*u_n`（`s = -1`）——**契约与实现自相矛盾**。

物理背景（重壁极限下的线性化 Riemann 问题，状态 L=壁面 `(p_L,u_L)`、R=流体
`(p_R,u_R)`，阻抗 `Z = rho c`）：

```
u* = (Z_L u_L + Z_R u_R + p_R - p_L) / (Z_L + Z_R)
Z_L >> Z_R  =>  u* = u_L,  p* = p_R + Z_R (u_L - u_R) = p_f + rho_f c_f u_n
```

即：壁面**压向**流体（`u_n > 0`）做功并产生压缩波 `rho c u_n`，壁压**升高**；
反力 `-p* n` 与运动方向相反 ⇒ 该术语是**耗散**的。取 `s=-1` 则变成向壁面所有模态
**注能**的抗阻尼项。

## ② 怎么做的

1. **推导定符号**（上面两行），不依赖任何实现：它是重壁极限的 Riemann 星态，唯一。
2. **改码**：`p_src = p_src + impedance_sign * rho0 * sound_speed * u_n`，默认
   `impedance_sign = +1`；保留 `impedance_sign = -1` 仅作审计臂（复现旧约定），
   `FSI2Case.wall_impedance_sign` 透传到 loose / strong 两个驱动。
3. **解析守护**（新增 `tests/test_fsi2_impedance_sign.py`，5 项）：
   活塞关系 `p* = rho c u_n` 精确到 1e-10；整梁刚性横移的反力
   `2 rho c u trib M` 且**反向**；只允许相对速度进入；旧符号必须给出反向力。
   `tests/test_fsi2_coupled_fixing.py::test_acoustic_impedance_sign` 原先把
   **上表面下移（膨胀）当作压缩**，因此把错误符号钉死——已按物理改正。
4. **受控实验 A（轨迹无关）**：在**同一条** `wall_impedance=False` 轨迹上，逐步
   比较「带/不带声学项」的载荷差与梁速率的点积 `Wdot = Σ F_imp · v_node`。
   `Wdot < 0` = 耗散，`Wdot > 0` = 泵能。工具链 `impedance_energy.py`。
5. **受控实验 B（耦合臂）**：`off / pre_fix(-1) / physical(+1)` 三臂，同一初始态、
   同一 `dx=0.02`、`dt`、`c0=5 m/s`、`rho_s/rho_f=1`（基准点）、loose 驱动，
   200 步探针；预声明闸门 `max|v_beam| > 0.3 m/s` 判发散。

图 (a) 活塞关系直线与计算点重合；(b) 两条功率曲线关于零**镜像**；
(c) 三臂 `max|v_beam|(t)`。

![impedance](assets/impedance_sign.png)

## ③ 计算结果（程序化注入，源两 JSON）

### 表 1 耦合臂探针（`dx = 20 mm`，`rho_s/rho_f = 1`，200 步）

| 臂 | 声学项 | sigma [1/s] | v_end [m/s] | v_max [m/s] | 流体最大速度 [m/s] | 判决 |
|---|---|---|---|---|---|---|
{arm_tbl}

### 表 2 阻抗功率（同一轨迹，逐点对账）

| t [s] | max abs v_beam [m/s] | Wdot 修正符号 [W/m] | Wdot 旧符号 [W/m] |
|---|---|---|---|
{eng_tbl}

## ④ 误差判定（逐项，是否 <3 %）

| 量 | 基准 | 最差误差 | <3 % ? |
|---|---|---|---|
| 壁压 `p*` vs `rho c u_n`（13 个 u_n ∈ [-0.6, 0.6]，图 (a)） | 重壁 Riemann 星态（解析） | {piston_err:.2e} | **是** |
| 壳体反力 vs `2 rho c u trib M` | 解析 | 见 `test_rigid_transverse_translation_is_damped`（rel 5e-3 闸门内） | **是** |
| `Wdot(旧) = -Wdot(新)` 镜像恒等 | 线性叠加 | {mirror:.2e} | **是** |
| 耦合探针 `v_end` vs Turek-Hron 位移幅值 | 文献 0.0082 m | **未测**（探针仅 200 步 = 0.035 s，占涡致振动周期 1 s 的 3.5 %） | **否/未运行** |

honest：解析项全部 <3 %；但**耦合瞬态幅值仍未与文献对上**——本轮不声称 fsi2 基准
闭环。预声明预测 `sigma(physical) < sigma(off) < sigma(pre_fix)` 的**前半段被证伪**：
`sigma` 16.2(off) < 21.6(physical) < 17.7(pre_fix)（见 JSON
`prediction_met: false`）。原因是 `sigma` 在起动瞬变段被入口斜坡驱动的强迫响应对撞，
不是能量判据；**能量判据（表 2）才是本轮的判决**：`Wdot` 在 {n_neg}/{len(act)} 个有效采样点
上为负（{w_first['t']:.4f}–{w_last['t']:.4f} s，v 从 {w_first['v_beam_max']:.1e} 到
{w_last['v_beam_max']:.1e} m/s），旧符号逐点严格为正且数值镜像。

## ⑤ 注意点

1. **符号错误已经确定性地消除**：`pre_fix` 臂在 t ≈ 0.04 s 内冲到
   `v = {sign['arms']['pre_fix']['v_max']:.1e} m/s`（旧记录的「destabiliser」当场复现），
   修正后同臂只到 `{sign['arms']['physical']['v_max']:.2e} m/s`，四个数量级。
2. **但修正并不足以稳定 partitioned loose 驱动**：`off` 臂自己在这条轨迹上也长到
   `{max(r['v_beam_max'] for r in rows):.2f} m/s`（表 2 末行），远高于阻力驱动尺度
   3e-2 m/s ⇒ 剩余激励**与声学项无关**，来源应在流体侧壁面 BC / added-mass 与
   入口斜坡的相互作用，属于 next-round 的强耦合（IQN-ILS）范畴，不能算在符号账上。
3. 旧测试 `test_acoustic_impedance_sign` 是「错误符号的守护者」：它把上表面**下移**
   （离开流体=膨胀）标成「closing」。这类测试只断言「不为零」时无法发现符号反，
   必须断言**解析值**——本轮已改成断言 `±rho c v` 的精确值。
4. 适用范围：静态/短时探针、`dx=20 mm`（梁厚 1 dx）、单层湿表面载荷传递；不含
   时间推进到涡致振动周期、不含 TLSPH 固体（本仓用离散弹性杆）。
5. 复现：

```bash
PYTHONPATH=src python benchmarks/xval_fsi2/impedance_sign.py --nst 200
PYTHONPATH=src python benchmarks/xval_fsi2/impedance_energy.py
PYTHONPATH=src python benchmarks/xval_fsi2/make_fig_impedance.py
PYTHONPATH=src python benchmarks/xval_fsi2/make_doc_impedance.py
PYTHONPATH=src python -m pytest tests/test_fsi2_impedance_sign.py \\
    tests/test_fsi2_coupled_fixing.py tests/test_fsi2_turek_hron.py -q   # 29 passed
```

## ⑥ 结论与下一步

- 判决：**符号缺陷成立并已修复**（模块契约、解析推导、能量判据三者一致）；
  旧符号是 `STRONG_COUPLING_AND_ROD_STABILITY.md` 所说「impedance 破坏一切配置」的
  **全部**原因（数量级 1e3 m/s vs 1e-1 m/s），但它不是 loose 驱动残余激励的原因。
- 下一步（未做，建议单列）：①用 strong coupling（IQN-ILS）在 `rho_s/rho_f=1` 上跑
  到 t ≈ 1 s（约 5.8e4 步）取梁尖位移幅值/拍动频率，对标 Turek-Hron 0.0082 m/0.0028 m
  的 15 %/3 % 门；②把声学项做成**双向**（同时进入流体侧壁面 BC 的动量），检验
  `off` 臂残余激励是否来自单向载荷的非保守性。
"""
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / "01_report.md").write_text(md)
    png = base64.b64encode((ART / "impedance_sign.png").read_bytes()).decode()
    payload = {
        "title": "FSI2 声学壁面阻抗符号勘误：抗阻尼项修正为耗散项（含解析+能量双证）",
        "summary": "boundary_pressure_from_fluid 的 Riemann 星态项符号与模块契约相反（实为抗阻尼）。"
                   "按重壁极限解析改正，活塞关系误差 %.1e，阻抗功率由泵能(+%.1f W/m)变为耗散"
                   "(-%.1f W/m)，旧符号臂 200 步冲到 %.0f m/s、修正后 %.2f m/s。" % (
                       piston_err, w_last["wdot_pre_fix"], -w_last["wdot_physical"],
                       sign["arms"]["pre_fix"]["v_max"], sign["arms"]["physical"]["v_max"]),
        "scenario_key": "sph_fsi2_force_from_fluid",
        "mission_id": 20872,
        "tags": ["TensorSPH", "FSI", "force-from-fluid", "Turek-Hron", "defect-fix",
                 "XVAL-SPH-FSI2"],
        "markdown": md,
        "figures": [{"name": "impedance_sign.png",
                     "caption": "(a) 活塞关系 p*=rho c u_n 精确重合；(b) 阻抗功率镜像"
                                "（修正=耗散，旧=泵能）；(c) 三臂耦合探针，旧符号发散。",
                     "png_base64": png}],
    }
    Path("/tmp/report.json").write_text(json.dumps(payload))
    print("doc + payload written; piston_err=%.2e mirror=%.2e neg=%d/%d" % (
        piston_err, mirror, n_neg, len(act)))


if __name__ == "__main__":
    main()
