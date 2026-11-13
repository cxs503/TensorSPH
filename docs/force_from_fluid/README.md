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