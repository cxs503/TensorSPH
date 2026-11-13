# 交叉对标（XVAL）：SPHinXsys 的 compression-based density regularization

**锚点**：`XVAL-SPH-COMPRESSION-BASED-DENSITY-REGULARIZATION`｜**任务**：m17364｜**仓库**：TensorSPH｜
**分支**：`agent/m17364`｜**日期**：2026-11-13

**一句话结论**：把 SPHinXsys 的 compression-based density regularization 逐行移植进 TensorSPH
（新模块 `src/tensorsph/compression_regularization.py` + `Config.density_regularization` 开关）后：
① **公式级等价**：向量化实现与 C++ 循环体的字面转写逐粒子一致到 `1.4e-16`（机器精度）；
② **物理级**：移植后（OSS 2026-06-19 版式，无参考均值）静水压误差 **1.43 %**（判据 <15 % **通过**，
且与 TensorSPH 原生求和密度 `1.40 %` 同量级 —— 二者本就等价）；
③ 但 OSS **master 版式**（2026-07-19 引入的 `1/K_ref` 参考均值归一化）在同一算例上把静水压系统性压低
**9.10 %**（判据仍 <15 %，但比本仓自己的 1 % 门限差 9 倍，也 >3 % 不可接受线）—— 该归一化把
静水压缩的均值（约 `0.95 %·½`）从密度场里减掉了，压力只剩梯度项。

---

## ① 问题与参考出处

静水/自由面问题里 WCSPH 的密度有两条互不兼容的路线：(a) 对连续性方程积分（本仓现役：求和密度 +
可选 δ-SPH 扩散率，`src/tensorsph/density_diffusion.py`）；(b) **每步用核内参考体积和覆盖密度**
（SPHinXsys 的 compression 路线）。本单把 (b) 移植进来并与解析静水场对照，回答两件事：移植是否忠实、
OSS 的做法在静水精度上表现如何。

**对标实现（零手抄，直接读源码）**

| 项 | 值 |
|---|---|
| 仓库 | `Xiangyu-Hu/SPHinXsys`（Apache-2.0，601★，default branch `master`） |
| 树 sha | `1612330e44c8947406f24b6756449e80dcd7158e`（`git/trees/master`，2026-11-13 取） |
| 文件 | `src/shared/shared_ck/particle_dynamics/fluid_dynamics/density_regularization.{h,hpp,cpp}` |
| 版式 A（无参考均值） | commit `d5c793e08c`（2026-06-19，「using compression formulation for fluid」） |
| 版式 B（master） | commit `375ec50b42`（2026-07-19，「only use compression sum larger than one to obtain average」）+ `708f01e970`（2026-07-25 revise） |
| 兄弟实现（对照读） | `src/shared/particle_dynamics/fluid_dynamics/density_summation.hpp`（自由面规则 `SMAX(rho_sum, rho0)`） |

取源码的方式：`api.github.com/repos/.../contents/<path>?ref=<sha>`（raw.githubusercontent 在本机被 reset，
见 ⑥ 摩擦 F2），公式全部来自上述 4 个文件，本报告不含任何转述数字。

**ORB 算法（逐阶段，按 C++ dynamics list 的执行顺序）**

1. `CompressionSummation<Inner>` + `<Contact>`：
   `S_i = W(0)·Vref_i + (1/Kref)·Σ_j W_ij·Vref_j`，`Vref = m/ρ0`（`VolumetricMeasureRef` 建体时复制一次，
   之后不变），自项**不乘** `1/Kref`（这是 OSS 的写法，本单原样保留并在差异清单里点出）。
2. `AverageCompression`（全局归约，只统计 `S_i > 1` 的粒子）：`Kref = 1 / mean{S_i : S_i>1}`。
3. `DensityRegularization<FlowType>`：`K_i = rule(S_i)`（`internal`=原值，`free_surface`=`max(S,1)`，
   `free_stream`=指示器位替换为 1），随后 `rho_i = K_i·ρ0_i`。
4. **一步滞后**：阶段 1 消费的是**上一时刻**的 `Kref`，阶段 2 之后才写新值 —— 本移植用
   `CompressionRegularization.inverse_reference_compression` 状态复现该顺序（`tests/...::test_two_stage_order_*`）。

## ② 方法与实现（最小改动）

| 文件 | 改动 |
|---|---|
| `src/tensorsph/compression_regularization.py`（新，236 行） | 移植：`reference_volumes` / `compression_summation` / `average_inverse_reference_compression` / `regularize` / `regularized_density` / `CompressionRegularization`（三阶段 + 状态） |
| `src/tensorsph/droplet_impact.py` | `Config` 新增 `density_regularization ∈ {none, compression}`、`compression_regularization_rule`、`compression_normalization ∈ {reference_mean, none}`（默认全 `none`/原值，**老算例逐位不变**）；`rhs(...)` 新增可选 `state`（跨步标量）；`density_regularization="compression"` 时**覆盖**求和密度，并与 `density_renormalization`/`density_diffusion` 互斥（显式 `ValueError`，不静默降级） |
| `tests/test_compression_regularization.py`（新，16 tests） | 公式等价/规则语义/解析语义/接线 |
| `benchmarks/xval_sphinxsys_compression/{run.py, figures.py}` | 受控对照跑批 + 出图/对照表 |
| `xval_output/sphinxsys_compression/` | `records_dx004_quick.json`、`figures/*.png` |

**受控对照设计**（experimental-design 技能）：唯一改变量=密度方案，其余（格点、质量、核、`h/dx`、`Δt`、
壁面层、边界处理、初速为零、采样窗口）全部固定；四臂：
`summation`（本仓现役基线）/ `compression_none`（OSS 版式 A）/ `compression_refmean`（OSS master）/
`compression_refmean_internal`（master + 无自由面裁剪）。**判据先于测量**：静水压带均值相对误差
<15 %（owner 精度标准），另并列报告本仓 1 % 门限与 3 % 线；自由面 2h 内的带**不计入**（截断支撑）。

## ③ 参数表（无量纲数完整）

算例：方形静水槽 `W=H=0.4 m`，格点 `dx=0.04 m`（10×10=100 流体粒子 + 32 壁面粒子），
`h=1.3dx`，`ρ0=1000 kg/m³`，`γ=7`，`c0=20 m/s`，`ν=0.01 m²/s`，`g=9.81 m/s²`，
`Δt=0.3h/c0`，总时长 0.4 s（`513` 步），均值窗口=后半程，每 2 步采样。

| 无量纲量 | 值 |
|---|---|
| `H/dx` | 10 |
| `W/H` | 1 |
| `h/dx` | 1.3 |
| 自由面波 Mach `√(2gH)/c0` | 0.140 |
| 声学 Courant `c0Δt/dx` | 0.390 |
| 重力波 Courant `√(gH)Δt/dx` | 0.0386 |
| 覆盖声学周期数 `(c0/2H)·T` | 10 |
| 静水压缩 `ρ(0)/ρ0 − 1`（解析） | 9.533e-03 |
| 核内参考体积标定残差 `dx²ΣW−1` | −5.32e-05（本仓已标定质量，故 S 内部=1；OSS 未标定，其内部读 `1+b`） |

## ④ 对照表（ref vs got，误差打勾/叉）

参考：解析静水 `p(y)=ρ0 g (H−y)`（无表格、无手抄；`case_hydrostatic.pressure_profile` 在带中心求值）。
门限 15 %；带 0/1 为内部带（带 2/3 上边缘进入自由面 2h，按规则不计入）。

| 臂 | 带 | y [m] | ref p [Pa] | got p [Pa] | 误差 [%] | <15 % | <3 % | <1 %(本仓门限) |
|---|---|---|---|---|---|---|---|---|
| summation（基线） | 0 | 0.056 | 3370.7 | 3323.4 | −1.40 | ✅ | ✅ | ❌ |
| summation | 1 | 0.153 | 2420.8 | 2404.3 | −0.68 | ✅ | ✅ | ✅ |
| compression_none（OSS 版式 A） | 0 | 0.056 | 3370.7 | 3322.6 | −1.43 | ✅ | ✅ | ❌ |
| compression_none | 1 | 0.153 | 2420.7 | 2403.3 | −0.72 | ✅ | ✅ | ✅ |
| compression_refmean（OSS master） | 0 | 0.056 | 3372.9 | 3066.0 | −9.10 | ✅ | ❌ | ❌ |
| compression_refmean | 1 | 0.153 | 2425.8 | 2212.3 | −8.80 | ✅ | ❌ | ❌ |
| compression_refmean_internal | 0 | 0.056 | 3372.9 | 3064.1 | −9.16 | ✅ | ❌ | ❌ |
| compression_refmean_internal | 1 | 0.153 | 2425.8 | 2211.1 | −8.85 | ✅ | ❌ | ❌ |

（不计入门限但列出：`compression_none` 带 2 −8.27 %、带 3 +3.05 %；`refmean` 带 2 −16.6 %、带 3 −11.2 %。）

汇总：

| 臂 | 最大内部带 p 误差 [%] | 最大内部带 ρ 误差 [%] | 时间平均 RMS 速度 [m/s] | `K_ref`(末) | `mean(ρ/ρ0)`(末) |
|---|---|---|---|---|---|
| summation | 1.40 | 0.013 | 0.0500 | 1.000000 | 0.98481 |
| compression_none | 1.43 | 0.013 | 0.0502 | 0.99480 | 1.00460 |
| compression_refmean | 9.10 | 0.076 | 0.0875 | 0.99383 | 1.00497 |
| compression_refmean_internal | 9.16 | 0.078 | 0.0875 | 0.99396 | 0.98520 |

**机制解释**（可复算）：`p = B((ρ/ρ0)^γ − 1)`、`B=ρ0c0²/γ`，故 `δp/p ≈ γδK/(K^γ−1)`；
本算例 `γ/(K^γ−1) ≈ 105`，实测密度偏低 0.076 % → 压力偏低约 8 %，与实测 9.10 % 的**同一数量级**：
`mean` 归一化把全场静水压缩 `≈0.95 %` 的均值从密度里减掉，只留梯度，等压线随之整体下移
（表面 `max(K,1)` 把顶部钉在 `ρ0`），因此误差是**系统性负偏**而非随机噪声。

**公式级等价（机器精度）**：向量化 `compression_summation` vs C++ 循环体的字面转写
（`literal_compression_sum`，含 `Inner`+`Contact`，`Kref∈{1, 0.987654321, 1.031}`）最大相对差
`1.4e-16`；`normalization="none" + rule="internal"` 与 `Σ_j m_j W_ij` 差 `2.2e-16`
（即版式 A ≡ 求和密度，移植是忠实的）。

## ⑤ 收敛性 / 参数敏感性（≥2 点）

| # | 变的是什么 | 结果 |
|---|---|---|
| 1 | 归一化：`none` → `reference_mean` | 静水压误差 1.43 % → 9.10 %（×6.4），ρ 误差 0.013 % → 0.076 %（×5.8），RMS 速度 0.050 → 0.0875 m/s（+75 %）。**这是两版式唯一差别，效应最大** |
| 2 | 自由面规则：`free_surface`(`max(K,1)`) → `internal` | 9.10 % → 9.16 %（+0.7 %，几乎无差）。说明 master 版的偏差主因是 `1/K_ref`，不是裁剪规则 |
| 3 | 压缩估计器解析精度（压缩量参数扫描，单元级，无算例依赖） | 各向同性压缩 `λ∈{1.000, 1.002, 1.005, 1.010}`：估计值 `1.000000045 / 1.0020216 / 1.0050541 / 1.0101109`，误差 `+0.0000 / +0.0022 / +0.0054 / +0.0110 %` —— 估计器本身远优于 15 % 判据，故 ④ 的 9 % 不能归因于估计器精度 |
| 4 | 分辨率（**未完成**）：`dx=0.02 / 0.015`、4 s | 命令见 `benchmarks/xval_sphinxsys_compression/run.py`；本单递交集群 job `cj1504`（排队 593 位，交付时未出结果）与本机后台跑（受控节点负载，仅完成第 1 臂）。**honest：多分辨率收敛曲线本单未取得**，判据只对 `dx=0.04` 成立 |

## ⑥ 失败模式、差异清单与工件

**本仓 vs OSS 做法差异清单（做法/参数/边界条件）**

| 维度 | TensorSPH（本仓） | SPHinXsys（OSS） | 移植取舍 |
|---|---|---|---|
| 密度来源 | 求和密度 `Σ m_j W_ij`（可 Shepard 归一化 `density_renormalization`） | 求压缩 `Σ W Vref` 后**覆盖** `ρ=K·ρ0` | 新增 `density_regularization="compression"` 覆盖路径；两者互斥并显式报错 |
| δ-SPH 扩散 | 有（Molteni / Antuono 形，`density_diffusion`） | 本文件不含（在 `density_diffusion`/黎曼求解器别处） | 移植项**不**叠加扩散项：一为「覆盖」一为「积分」，叠加无物理意义 |
| 参考体积 | `m/ρ0`，且质量已按格点核和标定（内部 S=1） | `VolumetricMeasure` 原值（未标定，内部 S=1+b） | 采用本仓已标定质量；差异体现在 K 的零点（本仓内部 K≡1，OSS 为 1+b） |
| 全局归一化 | 无 | master：`1/K_ref` 只乘邻域/接触项，自项不乘 | 两版式均提供（`compression_normalization`），默认 `reference_mean` 以忠实对标 master |
| 自由面处理 | 仅 `p=max(p,0)`（Tait 截断）+ 可选 Adami 压力外插 | `Regularization<FreeSurface>` 的 `max(K,1)`；另有按指示器的 `FreeStream` | 提供 `free_surface`/`internal`/`free_stream` 三种 rule |
| 核 | 2D cubic spline（`10/7πh²`），`h=1.3dx` | 默认 Wendland C2（`kernel_cubic_B_spline`/`kernel_quadratic` 等可选），`h=1.3dx` 量级 | 保留本仓核：核只改常数不改逻辑；同一核下对比才有意义 |
| 边界 | 固定壁面粒子 + Adami 壁压外插 | 壁面粒子 + `Contact` 参考体积计入压缩和 | 壁面 Vref 计入 `compression_summation`（与 OSS 一致） |
| 时间积分 | 固定步长显式（`0.3h/c0`） | 自适应步长（声学/力 CFL）常见 | 固定步长（可复现性优先），单列为差异 |
| 状态存放 | `AverageCompression` 的单变量 `InverseReferenceCompression` | 同（`registerSingleVariable`，默认 1） | `rhs(state=...)` 显式传字典，避免隐式全局状态 |

**移植取舍结论**：忠实移植版式 A（等价于求和密度，`1.43 %`）可放心用于本仓；master 的 `1/K_ref`
归一化**不建议**在需要静水压的场景启用（`9.10 %`，且误差随 `γ·静水压缩量` 放大，浅水/低 `c0` 更糟）。
`1/K_ref` 的原意（去掉全场平均压缩漂移）在自由面附近更有意义，但代价是丢失静水偏移 —— 这一点在
OSS 内部也并非无争议（`375ec50b42` 前一周才引入、次月即 "revise density regularization"）。

**失败模式**：4 个臂在 `dx=0.04` 全部稳定（无非有限加速度、无爆）；`compression_refmean_internal`
的 `mean(ρ/ρ0)=0.9852` 表明无裁剪时自由面粒子长期低于参考密度（负压被 Tait 截断，最多丢 1.5 % 平均
密度），这是 `free_surface` 规则存在的理由。

**生态摩擦（friction）**
- **F1** `raw.githubusercontent.com` 在本机连接被 reset（`Errno 104`），只能走 `api.github.com/contents`
  取源码；若将来 GitHub API 限流，OSS 对标将无源可查 → 建议平台侧镜像 OSS 源码入 `oss_solutions`。
- **F2** `run_shell` 的 120 s 组收割会把 `nohup &` 派生进程一起杀掉（`setsid` 才存活），
  而控制面 99 同时被集群/多任务占用，长跑单机 rsync 到超时 —— 这正是 `cluster_queue` 该承接的活，
  但 `cj1504` 排队 593 位 20 min 未派发，**长尾任务供给不足**，客观导致本单 ⑤-#4 未完成。

**工件**（worktree 内）
```
src/tensorsph/compression_regularization.py
src/tensorsph/droplet_impact.py                      (Config 3 字段 + rhs 覆盖分支 + state)
tests/test_compression_regularization.py             (16 passed)
benchmarks/xval_sphinxsys_compression/run.py
benchmarks/xval_sphinxsys_compression/figures.py
xval_output/sphinxsys_compression/records_dx004_quick.json
xval_output/sphinxsys_compression/figures/fig_{pressure_profile,band_error,time_series,compression_profile}.png
```
复算：`python3 benchmarks/xval_sphinxsys_compression/run.py --quick`（15 s，四臂）→
`python3 benchmarks/xval_sphinxsys_compression/figures.py --records <json>`。

**误差判定（逐项回答是否 <3 %）**：`summation` 与 `compression_none` 的最大内部带压力误差
1.40 %/1.43 % → **是**（<3 %，也 <15 %）；`compression_refmean`(+`internal`) 9.10 %/9.16 % →
**否**（<15 % 判据通过，但 >3 % 不可接受线，>本仓 1 % 门限 9 倍）。