# Force from fluid：控制体积动量账 + 三条载荷路线自审（fsi2 刚性闸门几何）

锚点：`XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`（Xiangyu-Hu/SPHinXsys 的
`fsi2-benchmark-and-force-from-fluid` 做法）。本轮把「力从哪里来」变成可机器判定的量：
一个控制体积动量账 + 逐壁面粒子反作用力，用来裁决 SPH 里三条互不相同的载荷路线。

## ① 案例（物理背景 / 几何 / 参数）

我们自己的求解器里「流体对结构的作用力」有三条路线，此前从未互相校验过：

| 路线 | 做法 | 代码 |
|---|---|---|
| **pair** | 对每个流-壁核作用对取反作用（牛顿第三定律），逐壁面粒子给力 | 本轮新增，`rhs(split=True)["reaction"]` |
| **panel** | 壁面压力 `p_face`×面积元积分（Adami-Hu-Adams 2012 eq.11 的 `p_face`） | 本轮新增 `panel_load`；同族于 `fsi_elastic_gate.gate_load`（WLS 拟合版） |
| **momentum** | 控制体积恒等式：`d/dt Σ m u = Σ m g + F_wall→fluid`，不看壁面几何 | 本轮新增 `MomentumBudget` |

几何 = `fsi2` 弹性闸门溃坝的**刚性闸门版**（与本仓 `docs/fsi_elastic_gate/` 同一族案例）：
槽 1.0×0.6 m，水柱 0.2×0.4 m，闸门 x=0.21/0.23（两层壁面粒子，湿高 0.4 m）。
参数：dx=0.02 m，h=1.3dx，c0=20 m/s（弱可压），γ=7，ρ0=1000，ν=1e-6 m²/s，
dt=1e-4 s（声学 Courant 0.077），900 步 = 0.09 s，200 流体粒子 + 385 壁面粒子。
解析基准（纯几何，不抄任何文献数字）：
水平推力 `0.5 ρ g H² = 784.80 N/m`，竖向总反力 = 流体重量 `ρ g A = 784.80 N/m`。

## ② 怎么做的

1. `droplet_impact.rhs` 增加**默认关闭**的 `split=True` 出口：返回 `{reaction, face_pressure}`。
   关掉时 3 元组返回值与全部既有数字逐位不变（`tests/test_force_from_fluid.py::
   test_split_is_opt_in_and_does_not_move_the_solution` 断言 `torch.equal`）。
   反作用力 = `m_i m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W`（注意 `acc` 是加速度，必须乘回流体粒子质量；
   第一版漏乘，实测反作用力偏大 2.5 倍，被牛顿第三定律检验当场抓出）。
2. 新模块 `tensorsph/force_from_fluid.py`：`MomentumBudget`（左端点冲量，与半隐式 Euler 的
   速度更新严格对应，所以干净算例残差 = 舍入误差）、`gate_lattice`/`wall_groups`、
   `panel_load`、`hydrostatic_reference`、`run_gate_case`（固定步长驱动，逐壁面粒子记录）。
3. 新增护栏：闸门面与最后一列流体粒子的间距必须落在 `[0.2dx, 1.2dx]`，否则报错。
   dx=0.03 时该间距 1.67dx，闸门载荷**静默掉到解析值的 0.03%**——这是实测踩到的坑。
4. 基准：`benchmarks/force_from_fluid/run_budget.py`（可 `--cases` 分批、`--merge` 出图），
   工件 `artifacts/force_from_fluid/{budget_cases.json, partial_*.json, fig_budget.png}`。

## ③ 计算结果（程序化写入，零手抄）

**初始静水态（t=0，step 0）**

| 量 | 解析 | pair 路线 | panel 路线 | pair/panel |
|---|---|---|---|---|
| 闸门水平推力 | 784.80 N/m | 580.9（**0.7402**） | 583.5（**0.7435**） | 0.9956（−0.44%） |
| 槽底竖向反力 | 784.80 N/m | 577.5（0.7358） | 780.3（0.9943）※ | — |
| 竖向总计 | 784.80 N/m | 578.2（**0.7368**） | — | — |

※ panel 槽底偏高，是因为槽底「湿」粒子实际为 14 个，其中 4 个（x=−0.03/−0.01/0.21/0.23）
不在水柱正下方，却因侧向核支撑拿到 ~2.4 kPa；扣除后 ≈0.75W，与闸门一致。**这是 panel
路线在水线/端部需要额外掩码的证据，本轮未修**。

**分辨率阶梯（同一静水态）**

| dx [m] | pair 闸门/解析 | panel 闸门/解析 | 竖向总计/重量 |
|---|---|---|---|
| 0.020 | 0.7402 | 0.7435 | 0.7368 |
| 0.010 | 0.7630 | 0.7582 | 0.7574 |
| 0.005 | 0.7728 | 0.7640 | 0.7681 |

**瞬态动量账（0.09 s）**

| 配置 | 残差/已记账冲量 | 残差折合等效力 | 占静水推力 |
|---|---|---|---|
| conservative（ν=1e-6，无扩散，自由滑移） | **3.160e-2** | 0.216 N/m | 0.028% |
| inviscid（ν=0） | **6.487e-5** | 0.00064 N/m | 8e-7 |
| diffusion（Molteni δ=0.1） | 4.263e-2 | 0.257 N/m | 0.033% |
| friction（wall_friction=1） | 4.723e-2 | 0.221 N/m | 0.028% |
| friction + ν=0（对照） | 6.487e-5 | — | — |
| no_gate（无闸门普通溃坝） | 2.930e-3 | 1.58 N/m | — |

全部算例：质量漂移 0，边界钳制事件 0，闸门推力尾段均值/解析 = 0.9990（conservative）。

## ④ 图

`artifacts/force_from_fluid/fig_budget.png`（四联图）：(a) 闸门推力时程 pair vs panel vs 解析；
(b) 竖向总反力/重量时程（动量路线裁判）；(c) 各配置动量账残差（对数轴）；(d) 初始态 pair/panel 对比。

## ⑤ 注意点 / 局限

1. **动量账是「检测器」不是「拟合器」**：左端点冲量与本仓半隐式 Euler 严格对应，干净算例
   残差只可能是舍入；残差非零 ⇒ 有东西在力装配之外改速度或漏记力。
2. **唯一可识别的泄漏 = 流体-流体粘性 Laplacian**：`visc_i = −ν (m_j/ρ_j) ∇²…` 的权重不对称，
   `m_i m_j / ρ_j ≠ m_j m_i / ρ_i`，故不成对守恒。ν=0 时残差降到 6.5e-5（**两条 ν=0 算例同值**，
   其中一条带 no-slip 壁摩擦 ⇒ 壁摩擦项本身是守恒的，本轮新增的壁面粘性反作用力口径正确）。
   量级：折合 0.216 N/m ≈ 静水推力的 **0.028%**——对 FSI 影响可忽略，但它是结构性非守恒，
   且密度梯度越大越显著（diffusion 算例 4.26e-2 高于 conservative，本轮未分离该间接效应）。
3. **26% 静水支撑亏缺不是成对求和的错**：同一状态下 pair 与 panel 只差 0.44%，两者都比解析低
   26%。因此 `docs/fsi_elastic_gate/03_report.md` 里「pair 成对和在壁面半空间支撑下有 O(h) 亏缺
   （−28.4%），改用面板法修好（+0.24%）」的**归因不成立**：亏缺住在两条路线共读的壁面压力/支撑
   场里（阶梯 0.740→0.773 随 dx 缓慢上升 ⇒ O(h) 边界效应，不是归一化错误）。
   那轮把估计器从 Adami eq.11 换成 WLS 平面拟合确实把数拉到解析值，但它改变的是 **p_face 估计器
   与动力学**，不是「成对求和形式」。根因本轮**未定位**（诚实：只证伪了归因，没给出替代根因）。
4. **静水态不是平衡态**：t=0 竖向支撑只有 0.7368W ⇒ 流体净加速度 −0.263 g，水柱会先塌陷
   （与 `docs/verification_startup_quiescence.md` 记录的启动瞬态同源）。所以任何「静水载荷
   = 解析值」的静态验收都必须在**动量账闭合之后**做，或对 t=0 做口径声明。
5. 闸门必须与流体格点公度（见 ②.3）；否则载荷静默归零，这类坑比数值误差危险得多。
6. 未做：弹性耦合算例的动量账（`fsi_elastic_gate` 的 dx=12.5 mm 单算例 247 s，本机 load15≈120
   只做轻量活）；Antoci 2007 实验参照值仍 WITHHELD（沿 `docs/fsi_elastic_gate/03_report.md` 结论）。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 判定项 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| pair vs panel（同一载荷的两种离散） | 互比 | 0.44% | ✅ |
| 动量账闭合（inviscid） | 恒等式 | 6.49e-5 | ✅ |
| 动量账闭合（conservative，含粘性） | 恒等式 | 3.16e-2 | ❌（折合载荷 0.028%，见 ⑤.2） |
| 竖向总反力 vs 流体重量（t=0） | 解析 | **26.3%** | ❌ honest FAIL（O(h) 边界支撑亏缺，双侧共享） |
| 闸门推力尾段均值 vs `0.5ρgH²` | 解析 | 0.1% | ✅（但为振荡均值，非逐点） |
| 弹性闸门顶点位移 vs Antoci 2007 | 实验 | WITHHELD | 不可判 |

复现：
```
PYTHONPATH=src python -m pytest tests/test_force_from_fluid.py -q                # 9 项全绿，23 s
PYTHONPATH=src python benchmarks/force_from_fluid/run_budget.py --cases static,conservative,inviscid --steps 900
PYTHONPATH=src python benchmarks/force_from_fluid/run_budget.py --cases diffusion,friction,no_gate --steps 900
PYTHONPATH=src python benchmarks/force_from_fluid/run_budget.py --merge
```
---

## ⑦ 26 % 静水支撑亏缺：根因已定位并可修复（m20448）

⑤.3 当时诚实写「根因未定位，只证伪了归因」。本轮把它定位到**壁面粒子的求和密度**
（不是成对求和形式、不是自由面截断），并给出一个默认关闭的修复开关。全部数字由
`benchmarks/force_from_fluid/run_static_support.py` 程序化产出
（`artifacts/force_from_fluid/static_support.json`）。

### 机制（可复算的因果链）

1. 壁面粒子进流体的**求和密度**时用的是名义质量 `rho0 dx^2`，而紧贴壁面的流体已被
   压到 `rho0 (1 + rho0 g d / B)^(1/gamma)`（本例底部 1.00930）：界面上有一个
   **0.26 % 的密度台阶**。
2. Tait 律把这台阶**放大 ~100 倍**：`c0=20 m/s, γ=7 ⇒ B=57.1 kPa`，
   `dp/dρ ≈ γB/ρ0`，故 0.26 % 密度差 ⇒ **29 % 压力差**（实测底部第 0 行 `p=2708 Pa`
   vs 解析 `3826 Pa`，比值 0.708 = `(1.00664⁷−1)/(1.00930⁷−1)`，逐位吻合）。
3. 于是**底部两行的压力梯度反号**（`p(y=0.5dx) < p(y=1.5dx)`，比值 0.772–0.794），
   壁面邻近行被流体**向下压**：该两行净竖向力 = −1.06 / −1.11 倍自重，
   而全柱亏缺的 **78 %** 就出在这两行（按壁面邻接行/内部/顶部四行分带，见下表）。

### 分带账（dx=0.02，各行 `m(a_z+g)` 之和 / 全柱重量，逐行相加恒等于 floor/W）

| 配置 | 底部两行 | 内部 | 顶部四行 | Σ = floor/W | 行质量占比 |
|---|---|---|---|---|---|
| baseline | **−0.1087** | 0.6509 | 0.1911 | **0.7358** | 0.1004 / 0.7003 / 0.1992 |
| 修复开关 on | **+0.0364** | 0.6892 | 0.1986 | **0.9285** | 同上 |
| oracle（诊断上限） | +0.0997 | 0.7040 | 0.2007 | **1.0093** | 同上 |

底部两行从「净向下压 −0.11」翻成「净向上被托 +0.036」，压力反号从 0.79 收敛到 0.99。
内部/顶部基本不动（内部 −0.049 → −0.011），说明亏缺的位置与机制都是**壁面邻接**，
不是整体标定问题。

### 修复：`Config.boundary_consistent_density`（默认 False）

壁面粒子按**自己外推出的壁面压力**（Adami 2012 eq.11，本仓已有）取 EOS 密度
`rho_w = rho0 (1 + p_w/B)^(1/γ)`，把 `Δm = m_w(rho_w/rho0 − 1)` 通过**一次
index_add** 反馈进流体求和密度（求和密度对质量是线性的，故不需要第二次核扫描），
再用修正后的流体场重新外推一次壁面压力。实现见 `droplet_impact.rhs`；
与 `density_regularization='compression'` 组合会抛错（后者是从参考体积重写密度，
叠加会二次计数）。

分辨率阶梯（三条路线，误差 vs 解析闭式）：

| dx | baseline floor/W | 修复 floor/W | baseline gate(pair/panel) | 修复 gate(pair/panel) | oracle floor/W |
|---|---|---|---|---|---|
| 0.020 | 0.7358 | 0.9285 (−7.2 %) | 0.7402 / 0.7435 | 0.9124 / 0.9275 (−7.3 %) | 1.0093 |
| 0.010 | 0.7570 | 0.9392 (−6.1 %) | 0.7630 / 0.7582 | 0.9353 / 0.9397 (−6.0 %) | 1.0093 |
| 0.005 | 0.7678 | 0.9448 (−5.5 %) | 0.7728 / 0.7640 | 0.9449 / 0.9439 (−5.5 %) | 1.0094 |

`oracle` 行是**诊断**不是方案：它把壁面质量直接取解析静水密度，全柱配平到 0.5 % g
以内、floor/W=1.009（+0.9 %）、gate=0.974…1.001。它证明「壁面质量」这一项即可解释
**全部**亏缺；而可实现的自洽版（用外推压力回喂）只走到 −5.5 %，因为回喂的 `p_w`
本身继承流体场的残余亏缺（不动点有偏：0.9285 → 再迭代一次 0.9337 → 稳定在 ~0.93）。

### 误差判定（逐项，显式回答是否 <3 %）

| 判定项 | 基准 | 误差 | <3 %？ | <15 %？ |
|---|---|---|---|---|
| 修复后 floor/W（dx=0.005） | `ρgA`（闭式） | −5.52 % | ❌ | ✅ |
| 修复后 gate 推力 pair（dx=0.005） | `½ρgH²`（闭式） | −5.51 % | ❌ | ✅ |
| 修复后 gate 推力 panel（dx=0.005） | `½ρgH²`（闭式） | −5.61 % | ❌ | ✅ |
| oracle floor/W（dx=0.02） | `ρgA` | +0.93 % | ✅ | ✅ |
| oracle gate pair（dx=0.005） | `½ρgH²` | +0.14 % | ✅ | ✅ |
| pair vs panel（修复后，各分辨率） | 互比 | ≤1.7 % | ✅ | ✅ |
| 底部两行净竖向力方向 | 应 ≥0 | −1.06 → +0.036 | 定性修复 | — |

结论（honest）：**该机制把 26 % 的 FAIL 修到 5.5 %，跨过 owner 的 15 % 验收线，
但没到 3 %**；剩下的 5.5 % 属另一条已知缺陷链（自由面两行 `p=0`、壁面外推继承流体
残余亏缺），本仓已有的 `density_renormalization` / `surface_pressure_closure`
在本指标上**不动**（实测 0.7358 不变 / 0.9285 不变），故不是这两个开关的重复。

### 瞬态回归（dx=0.02，900 步，conservative，ν=0）

| 配置 | 质量漂移 | 动量账残差 | 闸门推力尾段均值/½ρgH² | 峰值压缩 |
|---|---|---|---|---|
| baseline | 0 | 6.49e−5 | 0.9386 | 1.33 % |
| 修复 on | 0 | 1.93e−5 | 0.8725 | 1.26 % |

开关打开后动量账仍闭合在舍入量级、质量严格守恒、边界钳制 0 次；推力尾段从 0.939
降到 0.873（同一物理量的不同初值场，非判据）。

复现：
```
PYTHONPATH=src python benchmarks/force_from_fluid/run_static_support.py --table
PYTHONPATH=src python -m pytest tests/test_boundary_consistent_density.py -q
PYTHONPATH=src python -m pytest tests/test_force_from_fluid.py -q
```
