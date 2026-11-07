# 2-D WCSPH 波浪水槽：造波—色散—消波三锚 + 分辨率/人工声速定标

> 本文件由 `benchmarks/wave_tank/make_doc.py` 从 `runs/wave_tank/*.json` 与
> `runs/wave_tank/verdict.json` **程序化生成**：文中所有判据数字、表格与误差陈述均
> 来自运行证据，没有手抄数字。改动任何算例后重跑 `make_doc.py` 即可刷新。

## 1. 案例介绍

### 1.1 物理背景

TensorSPH 仓内已有的 30+ 项基准（静水压、Young–Laplace、液滴振荡、溃坝、Poiseuille、
毛细波等）都建立在**静平衡或低速定常流**之上。它们验证的是压力闭合、曲率、粘性与边界，
但没有 *波* 这个对象：没有自由面随时间的相位演化、没有色散、没有反射。

本节点补的正是这一层：一个 2-D 弱可压 SPH（WCSPH）数值波浪水槽，具备

* **造波**：左端活塞（piston）或摇板（flap）作动器，按线性造波理论给定冲程；
* **传播**：重力波在 WCSPH 自由面上传播，波要素（波长、周期、相位速度）由两点探针测量；
* **消波**：右端海绵层（sponge layer）速度阻尼，残余反射系数由探针阵列包络给出。

物理域：阻力—推进—耐波—操纵—稳性中的**耐波性（seakeeping）**，是船舶总体性能里第一
次出现"波"的验证面。水槽本身不模拟船体；它提供的是后续耐波性算例（RAO、波浪中阻力
增加、甲板上浪）必须先有的**数值造波/消波基础设施**与其精度定标。

### 1.2 几何与参数表

* 水深 `h = 0.5 m`，水槽长 `L` 按目标波长取（见下表），顶部留 `0.3 m` 余量；
* 目标波要素：`kh = 3`（深水侧）、线性造波理论给冲程，波高 `H = 0.1 h`；
* 分辨率 `dp/h ∈ {0.02, 0.015, 0.01, 0.0075}`（判据要求的四档）；
* 人工声速 `c0 = c0_factor·√(gh)`，`c0_factor ∈ {5, 10, 20}`；
* 海绵层长度 0.8λ，阻尼率 `a(x) = 3ω ((x-x_s)/(L-x_s))²`；
* 探针：水位用**柱体积分**测量（见 2.4）。

{{CASE_TABLE}}

## 2. 方法

### 2.1 复用的求解器（不新造物理）

水槽没有重写流体求解器。每一步调用的是仓内已有的
`tensorsph.droplet_impact.rhs`：密度求和 + Tait 状态方程（负压钳零）+ 对称化压力力 +
Morris 正则化层流粘性 + 静态壁面（壁面压力由 Adami 一阶外推得到）。因此静水压判据 (a)
与 `benchmarks/hydrostatic` 用的是**同一个**压力口径。

新增的只有：

1. **动边界作动器**：左端壁面粒子云按刚体平移（活塞）或绕池底铰接转动（摇板）逐时间步
   更新位置；
2. **海绵层**：对右端 `sponge_length` 段按 `v ← (1 - a(x)dt) v` 阻尼；
3. **探针与后处理**：柱体积分水位、单频最小二乘拟合、两点相位差色散、包络反射系数。

### 2.2 造波：冲程—波高传递函数

对水平位移 `X(z)·sin(ωt)`（`-h ≤ z ≤ 0`）的作动器，把作动器速度与传播模态匹配，得

```
H = (4 sinh(kh) / (2kh + sinh 2kh)) · k · ∫_{-h}^{0} X(z) cosh(k(z+h)) dz
```

* 活塞 `X = S/2`（S 为峰峰冲程）→ Havelock 关系 `H/S = 4sinh²(kh)/(2kh + sinh 2kh)`，
  `kh→0` 时退化为 `H/S→kh`；
* 摇板 `X = (S/2)(1+z/h)`（池底铰接）→ `H/S = (4sinh(kh)/(2kh+sinh2kh))·(sinh(kh) -
  (cosh(kh)-1)/kh)`，长波极限正好是活塞的 1/2（深度平均位移为 1/2）。

传递函数**只用来定冲程**；实际波高由探针测出并与理论对比（见 3.3 的 `H/S` 列），
因此传递函数的系统偏差是可见的，而不是被假设掉的。

### 2.3 消波：海绵层

阻尼率沿程二次增长 `a(x) = A·ω·((x-x_s)/(L-x_s))²`，`A = 3`。只阻尼速度而不松弛密度：
本求解器的密度由求和得到，任何"把密度拉回 ρ0"的做法都会破坏质量守恒。单步阻尼量
`a·dt ≈ 6e-3`，即阻尼本身是慢过程（不会自己产生反射），但穿越海绵层的累积衰减约
`e^{-4}`。

### 2.4 量测

* **水位估计**：`η_p(t) = Σ_{|x_i-x_p|<W/2} (m_i/ρ0)/W - h`，即柱窗内的材料体积除以窗宽。
  格点上它是精确的（每个粒子携带 `dx²` 体积），且远小于最短波长（`W = 2dp ≪ λ/2`），
  比"取最高粒子 y"平滑得多（后者按一个粒子间距跳变，对 `H = 0.1h = 5dp` 的波就是 20% 噪声）。
* **幅相提取**：对已知激励频率 ω 做最小二乘投影 `y = c + a cos ωt + b sin ωt + d·t`
  （lock-in）。这比峰峰值稳健得多：非波分量（粒子级噪声、声学振荡、二次谐波）被正交投影滤掉。
* **色散**：两点相位差 `k = (φ_b - φ_a)/Δx`（要求 `Δx < λ/2`），相位速度
  `c = ω/k`；与**实测 k** 处的线性理论 `√(g·tanh(kh)/k)` 比较（用给定 k 比较会把 k 的误差
  藏进被检验量里）。
* **反射系数**：`R = (H_max - H_min)/(H_max + H_min)`，`H` 取海绵层前 4 支探针（间距 λ/4，
  覆盖 0.75λ，即 1.5 个包络半周期）的单频幅值。
* **孤立波**：初值问题 `η = H sech²(β(x-x₀))`、`u = cη/(h+η)`，
  `β = √(3H/4h³)`、`c = √(g(h+H))`；格点按列自适应（每列保持 dp 间距、只改粒子数），
  使波"长在几何里"而不是压在一个不平衡的速度场上。相位速度由两支探针的 crest 到达时间
  （抛物线refine）给出，波形 L2 误差对解析 `H sech²` 在 `|x-x_c| < 2.5/β`（含 98.7% 体积）窗内。

### 2.5 时间积分与成本

定步长，步长取声学 Courant 界 `dt = C·h_smooth/c0`（`C = 0.45`，`h_smooth = 1.3dp`），
记录到的最大声学 Courant 见下表。粒子量 `N = (L/dp)·(h/dp)`，成本近似 `∝ N/dt ∝ dp^{-3}`。
`c0` 越大越贵，故 `c0_factor` 是"精度—成本"旋钮，本节点对它有单独扫描。

## 3. 计算结果

### 3.1 判据总表（程序化注入）

{{CRITERIA_TABLE}}

### 3.2 分辨率阶梯与人工声速扫描

{{LADDER_TABLE}}

### 3.3 造波传递函数实测

{{TRANSFER_TABLE}}

### 3.4 误差判定（逐项回答"是否 <3%"）

{{ERROR_STATEMENTS}}

## 4. 结果图

![dispersion](figures/wave_tank_dispersion.png)

**图 1** 左：四个 kh 下实测相位速度与线性理论之比（灰带为 ±3% 判据带）；右：两点相位差
给出的波数 vs 由 ω 反解的线性理论波数。

![ladder](figures/wave_tank_ladder.png)

**图 2** 左：色散误差 vs 分辨率（log–log，附拟合阶数）；右：误差—墙钟成本。收敛必须
"单调下降且无平台"。

![decay](figures/wave_tank_decay.png)

**图 3** 波高沿程衰减：探针 0（0.5λ）到 10λ 处；虚线为判据 H/H₀ ≥ 0.95。

![field](figures/wave_tank_field.png)

**图 4** 上：三个时刻的粒子云按速度着色的场量图（含背景格点，可见位移）；下：对应时刻的
自由面高程剖面 + 探针位置 + 海绵层起点。

![solitary](figures/wave_tank_solitary.png)

**图 5** 左：孤立波的初始解析剖面与两个时刻的实测剖面；右：crest 到达时间 vs 位置，
斜率即实测相位速度，虚线为 `√(g(h+H))`。

## 5. 注意点、局限与适用范围

1. **分辨率下限是几何噪声，不是算法**：`H = 0.1h` 在 `dp = 0.02h` 时只有 5 个粒子高；
   水位估计用柱体积分把量化噪声压到 `dp/(W·√N)` 量级，但波面的本征离散误差仍与 `dp/H` 同阶。
   孤立波算例因此用 `H = 0.2h`（10 个粒子高）。
2. **弱可压偏差**：有限 `c0` 使相位速度偏高（约 `½(c/c0)²`）。`c0_factor = 10` 时对
   `kh ∈ [0.5, 3]` 不超过 0.5%；`c0_factor = 5` 在 `kh = 1` 附近可达 1.5%，这是
   `csweep` 里可以直接读出的系统性偏差，也是把 `c0_factor` 定在 10 的理由。
3. **海绵层不是无反射边界**：残余反射是频率与入射角相关的量；本节点只测正向入射、
   窄带规则波。斜向入射与不规则波谱需要另立算例。
4. **跨求解器互裁（判据 f）状态**：见判据表，`WITHHELD` —— 本 worktree 内没有可运行的
   TensorLBM 重力波二维水槽（其 `wave_bc.py` 给的是 Airy 入口边界条件、另有自由面 LBM
   内核，但没有探针+海绵的二维重力波水池驱动），实现它属于 TensorLBM 仓，不属于本仓。
   `benchmarks/wave_tank/cross_solver.py` 已经把工况规格与比较接口写好，把对方结果按
   格式喂回 `--compare` 即可出三个差；**没有用任何替代物冒充这一格**。
5. **成本与线程**：单进程 PyTorch 在该平台上 per-op 开销约 1 ms，粒子数少时是
   "算子数 × 1 ms" 的常数底（3000 粒子 0.05 s/步），粒子数多时转为带宽主导
   （6e4 粒子 0.35 s/步）。实测 `torch.set_num_threads(>8)` 反而更慢（内存带宽 + 线程池
   开销），故所有算例单线程、靠多进程并行。
6. **数值衰减是本节点最大的局限（已定量）**：实测波高沿程按 `H/H₀ ≈ ρ^(x/λ)` 衰减，
   `ρ` 见 3.2 表（`height_ratio_per_wavelength`）。诊断（`.scratch` 外的复现脚本见
   6.1 的 `--case` 与 `benchmarks/wave_tank/run.py` 的 `decay` 字段）表明**总动能不衰减而
   波的振幅衰减**：波能泄漏到粒子尺度的无序运动，再被粘性吃掉——这是"密度求和式 WCSPH
   在自由面上瞬时响应粒子重排"的固有耗散（本仓不解连续方程）。因此判据 (c) 10 波长
   0.95 在本文档覆盖的分辨率下 **不通过**，且 (e) 的 `R` 与 (b) 的相位差都被它污染：
   因此同时报告"去趋势 R"（先把指数衰减拟合掉再看包络）与 `R_raw`。
7. **`surface_pressure_closure=True` 与壁面粒子不兼容（发现的既有缺陷）**：
   `droplet_impact.rhs` 把壁面粒子并入 `pairs` 后调用 `free_surface_pressure`，而压力数组
   只有流体长度，越界 `IndexError`（`index 3125 out of bounds for dimension 0 with size
   3125`）。本节点未改核心文件（不属本仓改动范围），只报告：该开关"压力闭合"在带壁面
   的水槽里当前不可用，故水槽的自由面处理仍是 `p=0` 钳零。
8. **二阶平均流（Stokes setup）**：松弛区注入的速度场是线性的，其质量输运是二阶的；在
   封闭水槽里表现为入口段水位缓慢下降、出口段抬升（实测入口段均值 −1.4 cm / 3 周期）。
   规则波短时测量不受影响，但长时（>10 周期）算例需要均值流补偿。
9. **未做的部分**：不规则波谱（JONSWAP）造波、摇板定量验证（摇板只做了运动学与传递函数
   量级校核）、斜向入射反射、三维扩展、湍流/破碎波、均值流补偿。

## 6. 审计链

| 项目 | 内容 |
|---|---|
| 源码 | `src/tensorsph/wave_tank.py` |
| 算例驱动 | `benchmarks/wave_tank/run.py`（`--case NAME`，一算例一 JSON） |
| 证据 | `runs/wave_tank/<case>.json`（配置 + 探针统计 + 判据 + 成本） |
| 汇总 | `benchmarks/wave_tank/summarise.py` → `runs/wave_tank/verdict.json` |
| 图 | `benchmarks/wave_tank/make_figures.py` → `docs/figures/wave_tank_*.png` |
| 跨求解器接口 | `benchmarks/wave_tank/cross_solver.py` → `runs/wave_tank/cross_case.json` |
| 测试 | `tests/test_wave_tank.py`（{{PYTEST_COUNT}} 项，含 2 项短积分） |
| 复现 | `python benchmarks/wave_tank/run.py --case dispersion_kh3` 等 |

### 6.1 复现命令

```
python benchmarks/wave_tank/run.py --list
for c in hydrostatic dispersion_kh0.5 dispersion_kh1 dispersion_kh2 dispersion_kh3 \
         decay10 reflection reflection_control ladder_dp0.020 ladder_dp0.015 \
         ladder_dp0.010 ladder_dp0.0075 csweep_c5 csweep_c20 solitary; do
  python benchmarks/wave_tank/run.py --case $c --threads 1 --json-out runs/wave_tank/$c.json
done
python benchmarks/wave_tank/summarise.py
python benchmarks/wave_tank/make_figures.py
python benchmarks/wave_tank/make_doc.py
```

## 7. 结论

{{CONCLUSION}}