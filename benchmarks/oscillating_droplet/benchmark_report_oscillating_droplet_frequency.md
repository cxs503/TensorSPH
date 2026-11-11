# 算例池·振荡液滴频率（Rayleigh–Lamb）——TensorSPH 线性响应基准报告

> **锚点**：CASE-OSC-DROP ｜ **仓库**：TensorSPH ｜ **分支**：agent/m15328
> **工件**：`runs/surface_tension_dispersion/{report,benchmark,meta}.json`、`figures/lamb_stiffness.png`、
> `benchmarks/oscillating_droplet/benchmark_report_oscillating_droplet_frequency.md`
> **repro_cmd**：`PYTHONPATH=src python -m tensorsph.surface_tension_dispersion --ppw 24,32,48 --ks 1,2,3 --modes 2,3,4 --rdx 16,24,32 --shifts 2 --out runs/surface_tension_dispersion/report.json`
> 所有数字由 `runs/surface_tension_dispersion/report.json` 程序化注入，零手抄。

## ① 问题与 ref 出处

**物理背景**：二维液滴（无限长柱截面）受表面张力驱动做自由毛细振荡。对界面做
`r(θ) = R(1 + ε cos nθ)` 的模态扰动，界面曲率的一阶展开为
`κ(ε) = (1/R)[1 + (n²−1)ε]`，与内部无旋势流 `φ = A r^n cos nθ` 的 Bernoulli 平衡、
自由面运动学条件联立，得 **Rayleigh–Lamb 二维频率**：

```
ω_n² = n(n²−1) σ / (ρ R³)，  n ≥ 2
```

**ref 出处（零手抄，引用原表原页）**：
- **Rayleigh, Lord (1879)**, *Proc. London Math. Soc.* **10**:4 —— 振荡液滴频率的原始推导。
- **Lamb, H. (1932)**, *Hydrodynamics*, 6th ed., **Art. 275** —— 二维柱截面液滴
  `ω_n² = n(n²−1)σ/(ρR³)` 的闭式解（本基准的解析锚）。
- 三维球面解 `ω_n² = n(n−1)(n+2)σ/(ρR³)`（n=2 时 = 8 而非 6，比二维高 15.5%）——
  本基准**明确用二维柱截面分支**，避免「恰好差 15%」的假失败（见 `docs/verification_oscillating_droplet_lamb.md` ⑤.1）。

**为什么做这个**：TensorSPH 已发表的表面张力数字全是**静态**的（Young–Laplace 压差、
切割力恒等式、圆界面曲率）。本算例补**动态**一环：CSF 表面张力算子在**动界面**上
是否以正确的**方向/相位**和**有效惯性**恢复毛细振荡频率。

**测量方式**：本基准采用**线性响应 / 离散色散**测量（非时间积分）——
`ω_measured²/ω_theory² = Δp_measured/Δp_theory`（模态法向应力比）。在线性区，
恢复力的唯一物理是法向表面应力 `Δp(θ) = σ(n²−1)ε cos(nθ)/R²`，惯性是精确势流值
`ρR/n`，故频率比 = 离散 CSF 算子的模态法向应力比。这是对**表面力算子**的隔离测量，
不混入时间积分的耗散/稳定性污染（`docs/verification_capillary_wave_open_surface.md`
记录了为何 HEAD 上同一算子的时间轨迹不可采信——伪加速度 ~23× 波自身 ω²A）。

## ② 方法与实现

**复用**：`src/tensorsph/surface_tension_dispersion.py`（m14536 起）的
`lamb_case` / `drop_modal_stiffness` / `drop_geometry` —— 不另起炉灶。

**流程**：
1. **几何**：`drop_geometry(mode, eps, radius, dx)` 生成六角点阵液滴
   `r(θ)=R(1+ε cos nθ)`，`R=0.02 m`，`ε=0.10`（DROP_EPS），`dx = R/radius_over_dx`。
2. **CSF 算子**：`csf_state` 用 `csf_fields`（`csf_consistent` 模块）计算
   `κ`（曲率，`fit_quadratic` 方案）、`δ_s`（表面 delta 函数）、法向 `n`；
   力 `f = −σ κ δ_s n`（2-D 标量牵引）。
3. **模态投影**：`drop_modal_stiffness` 把牵引投影到 `cos(nθ)` 上，
   `dp_meas = 2/(2πR) Σ (traction·vol·cos nθ)`；解析系数
   `κ_exact = (n²−1)ε/R`，`dp_exact = −σ κ_exact`。
4. **频率比**：`ω_ratio = √|dp_meas/dp_exact|`（符号取恢复方向），
   `ω_error_pct = (ω_ratio − 1)·100%`。
5. **子格点系综**：`ensemble` 在 `shifts×shifts` 子格点相位上平均（消除点阵阶梯噪声），
   报告系综标准差为单次实现的格点噪声。
6. **收敛性**：`convergence_order` 对 `log|error|` vs `log(dx)` 做最小二乘斜率。
7. **σ 敏感性**：`sigma_sensitivity` 在 σ×0.5/1/2 上重测，检验频率比是否 σ 标度不变。

**工具链**：PyTorch 2-D 弱可压缩 SPH（CPU，float64）、`cell_linked_pairs` 邻居搜索、
`csf_fields` 曲率算子、pytest、matplotlib。本报告所有数字由 `report.json` 注入。

## ③ 参数表（无量纲数完整）

| 量 | 值 | 说明 |
|---|---|---|
| 液滴半径 R | 0.020 m | 面积等效半径 |
| 参考密度 ρ0 | 1000 kg/m³ | 水，20°C |
| 表面张力系数 σ | 0.072 N/m | 水/空气，20°C |
| 初始变形 ε | 0.10 | `r=R(1+ε cos nθ)` |
| 模态 n | 2, 3, 4 | 方位角模态 |
| 分辨率 R/dx | 16, 24, 32 | 半径/粒子间距 |
| h 因子 | 1.3 | h/dx |
| 曲率方案 | fit_quadratic | CSF 曲率算子 |
| δ 归一化 | legacy | 表面 delta 函数 |
| 子格点系综 | 2×2 | 相位平均 |
| 域长 | 2πR | 周期通道（drop 用非周期） |

**无量纲数**：
- **毛细数** `Ca = ρc₀²R/σ`（压缩量）：本测量为线性响应，不涉时间积分，无 c₀。
- **Ohnesorge 数** `Oh = μ/√(ρσR)`：本测量为无粘线性响应（惯性取精确势流值），Oh=0。
- **模态数** `n`：2/3/4，频率比 `ω_n²/ω_2² = n(n²−1)/6`。
- **分辨率** `R/dx`：16/24/32，唯一控制离散误差的几何参数。

## ④ 对照表（ref vs got + 误差打勾叉）

**基准**：`ω_n² = n(n²−1)σ/(ρR³)`（Rayleigh 1879；Lamb Art. 275）。
**容差判据**：<5%（任务令）。

| 模态 n | R/dx | ω_ref [rad/s] | ω_got [rad/s] | 误差 [%] | <5%? |
|---|---|---|---|---|---|
| 2 | 16 | 7.34847 | 7.32537 | **−0.31** | ✅ |
| 2 | 24 | 7.34847 | 6.96172 | **−5.26** | ❌ |
| 2 | 32 | 7.34847 | 6.99237 | **−4.85** | ✅ |
| 3 | 16 | 14.69694 | 13.33570 | **−9.26** | ❌ |
| 3 | 24 | 14.69694 | 13.27165 | **−9.70** | ❌ |
| 3 | 32 | 14.69694 | 13.30411 | **−9.48** | ❌ |
| 4 | 16 | 23.23790 | 19.05848 | **−17.99** | ❌ |
| 4 | 24 | 23.23790 | 19.01951 | **−18.15** | ❌ |
| 4 | 32 | 23.23790 | 18.74919 | **−19.32** | ❌ |

**判定**：**基频模态 n=2 在 R/dx=16（−0.31%）与 R/dx=32（−4.85%）通过 <5% 门**；
R/dx=24（−5.26%）略超。**高次模态 n=3（~−9.3%）、n=4（~−18%）不通过**。
误差符号一致为**负**（实测频率低于解析），且随模态 n 增大而增大——这是离散 CSF
算子的**曲率分辨率不足**（高次模态曲率 `(n²−1)ε/R` 更大、更需细网格）的系统性特征，
不是随机噪声。

## ⑤ 收敛性 / 参数敏感性（至少两点）

**点1 · 模态收敛性（误差随 n 单调增大）**：
- n=2：−0.31% → −5.26% → −4.85%（R/dx 16→24→32）
- n=3：−9.26% → −9.70% → −9.48%
- n=4：−17.99% → −18.15% → −19.32%
误差随模态 n 单调增大（n=2 最小、n=4 最大），且**不随 R/dx 单调收敛**——
R/dx=24 反而比 16 差（n=2 从 −0.31% 恶化到 −5.26%）。`convergence_order` 对
`lamb_mode2` 给出 **−4.14**（负斜率 = 误差随细化**增大**），说明当前 `fit_quadratic`
曲率方案在细网格上**不收敛**到解析值——这是离散色散的格点相位噪声主导，非物理耗散。

**点2 · σ 标度不变性（敏感性）**：
`sigma_sensitivity` 在 σ×0.5/1/2 上重测，`ω_ratio` 三值**完全相同**（0.99686），
`max_spread_pct = 0.0%`，`within_5pct = True`。即**频率比 `ω_measured/ω_theory`
对 σ 严格标度不变**——σ 线性进入恢复力与解析频率，比值被消去。这验证了测量
**隔离的是表面力算子**（几何/曲率），而非 σ 的绝对量级。

**点3 · 格点相位噪声（系综 std）**：
`lamb_stiffness.png` 的误差棒为子格点相位系综标准差。n=2 的系综 std 小（~0.3%），
n=4 的 std 大（~1%），说明高次模态的**格点阶梯噪声**随曲率增大而放大——
这是高次模态误差的主要来源之一。

## ⑥ 失败模式与工件

**失败模式（honest-FAIL 记录）**：
- **fail_stage**：`C2_lamb_modes`（高次模态 n=3/4 及 R/dx=24 的 n=2 未达 <5% 门）。
- **failure_reason**：离散 CSF 曲率算子（`fit_quadratic`）在**高次模态**（曲率
  `(n²−1)ε/R` 更大）与**细网格**（R/dx=24）上**不收敛**到解析值——`convergence_order`
  对 n=2 为负（−4.14），误差随细化增大。这是**离散色散的格点相位噪声**主导，
  非物理耗散；基频 n=2 在粗/细两端（R/dx=16、32）均 <5% 通过，证明算子**方向/相位正确**，
  只是**高次模态曲率分辨率不足**。
- **两轮法接棒**：本单为第 1 轮（线性响应测量）。第 2 轮建议：① 换 `fit_circle` /
  `mls_divergence` 曲率方案做**方案扫描**（`option_screen` 已留接口）；② 提高 R/dx
  至 48/64 验证高次模态是否收敛；③ 对 n=2 做**时间积分**动态验证（`run_lamb.py` 路径，
  当前 −61.7%，需先修自由面曲率噪声）。

**工件**：
- `runs/surface_tension_dispersion/report.json` —— 全量报告（含 benchmark_table、gates、convergence、sigma_sensitivity）。
- `runs/surface_tension_dispersion/benchmark.json` —— 基准表（ref/got/误差/门）。
- `runs/surface_tension_dispersion/meta.json` —— 命令/版本/墙钟/工件 sha256。
- `runs/surface_tension_dispersion/figures/lamb_stiffness.png` —— 模态刚度比 vs 分辨率（误差棒=格点相位系综）。
- `runs/surface_tension_dispersion/figures/capillary_wave_error.png` —— 毛细波频率误差（对照，非本算例）。
- `runs/surface_tension_dispersion/summary.md` —— 汇总。

**生态摩擦（如实记）**：
- 本算例**复用** `surface_tension_dispersion` 现有能力，未另起炉灶——该模块已含
  `lamb_case`/`drop_modal_stiffness`/`drop_geometry`/`ensemble`/`sigma_sensitivity`/`convergence_order`，
  直接跑通。
- **摩擦点**：`benchmarks/oscillating_droplet/` 下存在**两套**同基准实现（`run.py` 动态
  时间积分 −43.5%、`run_lamb.py` 动态 −61.7%、`surface_tension_dispersion.py` 线性响应），
  历史合并（m2919/m3820）已留 `merge_resolutions` 说明。本单**明确选用线性响应**分支
  （隔离表面力算子、不混入时间积分耗散），并在报告 ① 中声明测量方式，避免数字混用。
- **摩擦点**：`surface_tension_dispersion` 的 `--quick` 与全量模式结果**不一致**
  （quick 只跑 n=2@rdx=16 得 −0.31% PASS；全量跑 n=2/3/4@rdx=16/24/32 得高次 FAIL）——
  报告必须**引用全量模式**，quick 仅作冒烟。

## 附：repro_cmd 一键复现

```bash
cd /root/tp-agent/work/m15328
PYTHONPATH=src python -m tensorsph.surface_tension_dispersion \
  --ppw 24,32,48 --ks 1,2,3 --modes 2,3,4 --rdx 16,24,32 --shifts 2 \
  --out runs/surface_tension_dispersion/report.json
# 全量报告（含图）：
PYTHONPATH=src python scripts/run_surface_tension_dispersion.py \
  --modes 2,3,4 --rdx 16,24,32 --out-dir runs/surface_tension_dispersion
```
