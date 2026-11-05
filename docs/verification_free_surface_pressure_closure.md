# 验证记录：自由面压强闭合（Adami et al. 2012 §3.3）对 Young–Laplace 压差的作用

**仓库**: TensorSPH ｜ **工件**: `benchmarks/laplace_pressure_floor/`（`report.json` = 默认 clamp 基线，
`report_closure.json` = shell 闭合实测）、`docs/verification_free_surface_pressure_closure.md`（本文件，
全部数字由 `make_closure_doc.py` 从 JSON 注入）、`tests/test_free_surface_pressure_closure.py`（闭合算子的
单元门）、`tests/test_shell_closure_report.py`（本文件数字的回归门）。
**开关**: `Config.surface_pressure_closure`（默认 `False`，既有全部数字逐位不变）、
`surface_pressure_donors`（`"interior"` = 从体相单向逐层外推，`"all"` = Adami 原始式一遍）、
`surface_pressure_layers`、`surface_pressure_threshold`（θ）。

**结论（先给判定）**：在已跑完的 2 个算例上，末 40 % 窗口时间平均的 Young–Laplace 压差误差由
clamp 的 -37.9 % / -34.0 % 变为 shell 闭合的
**+8.8 % / +4.9 %**（c0 由高到低）；
**全部落到平台 15 % 门限内（2/2）**，
3 % 严格门限 0/2 通过。伪流强度 $|v|_{max}/\sqrt{\sigma/(\rho R)}$
为 0.388, 0.104（Adami 2012 判据 < 0.1，0/2 通过，未过关）——
闭合把压强边界修正了，代价是自由面壳层的伪流放大（clamp 时为
0.030, 0.026），这是本轮的**新欠账**，见 ④⑤。
分辨率阶梯点 dx = 1.5 mm 未跑（算力预算，见 ⑤），因此这是 **2 点**的结论，不是收敛结论。

## ① 案例介绍

静止二维液滴（无限长柱截面），表面张力驱动，零重力，除 CSF 表面张力外无其它体积力。
解析基准为 Young–Laplace 定律的二维形式 $\Delta p = \sigma/R$：

| 参数 | 值 | 说明 |
|---|---|---|
| 面积等效半径 R | 20 mm | 液滴初始半径 |
| 表面张力系数 σ | 7.2 N/m | 10× 水，用于压缩毛细时间 |
| 参考密度 ρ0 | 1000 kg/m³ | Tait 状态方程，γ = 7 |
| 解析压差 σ/R | **360.0 Pa** | 比对基准量（二维柱面是 1/R，不是 2/R）|
| 毛细时间 t_σ | 33.33 ms | $\sqrt{\rho R^3/\sigma}$ |
| 重力 | 0 | 纯毛细平衡 |
| 曲率算子 | `fit_quadratic` | 圆上曲率误差 < 1 %（docs/verification_surface_tension_curvature.md）|
| 核 | 2-D cubic spline，h = 1.3 dx | 归一化 10/(7πh²) |
| 粒子质量 | 核心区核和校准 | 消除六角格点 +0.59 % 的初始密度偏置 |

初始状态是六角密排格点；粒子质量按核心区实测核和校准，使 $t=0$ 的 $\rho/\rho_0-1$ 在核心区为机器精度量级，
压差偏差不可能来自初始化预压缩（m2971 的 C2 结论）。

## ② 怎么做的

1. **缺陷定位（m2971 的 C3）**：密度求和按定义在自由面附近邻居数不足，$\rho$ 在 ~2h 厚的壳层内低于
   $\rho_0$；`tait_pressure` 把 $p$ 钳在 $\ge 0$，等于给内部压强的边界条件是 $p=0$ 而不是
   $p=\sigma/R$。实测内部只压到所需压缩比的 0.57 倍——这正是压差长期停在
   -37.9 % 量级的原因。
2. **闭合实现**（`src/tensorsph/droplet_impact.py::free_surface_pressure`，Adami, Ihmsen & Adams 2012
   §3.3）：对 $\rho_i < \theta\rho_0$ 的粒子令
   $$p_i=\max\Big(0,\ \frac{\sum_j (p_j+\rho_j\,\mathbf{g}\cdot(\mathbf{r}_i-\mathbf{r}_j))W_{ij}}{\sum_j W_{ij}}\Big),$$
   即把体相压强沿核权重一阶外推到自由面粒子。`donors="interior"` 时只有已知压强（体相或已填层）
   的粒子参与捐赠，并在 `surface_pressure_layers` 遍内逐层向外填；`donors="all"` 为字面公式（自由面
   邻居的 $p=0$ 也参与），一遍即可。
3. **不改状态方程、不改力项**：闭合只替换传给压强梯度的压强场，`rhs` 返回的求和密度不变
   （`test_closure_changes_only_the_surface_pressure`）；关闭时 `rhs` 的压强与 `tait_pressure(rho)`
   逐位相同（`test_closure_is_off_by_default`）。
4. **测量口径**：绝对压强 $p_{core}$（$r<0.5R$ 均值）减真空参考 $p=0$，取末 40 % 窗口时间平均
   （`absolute_time_average_tail`）。呼吸模态周期 ~$t_\sigma$，单帧快照不是平衡值，故必须时间平均。
5. **工具链**：PyTorch 二维 WCSPH（CPU，2 线程）、`tensorsph.neighbor_search.cell_linked_pairs`、
   pytest、matplotlib；`report_closure.json` 由
   `run.py --closure shell --no-controls --out report_closure.json` 产生，本文件由
   `make_closure_doc.py` 从该 JSON 程序化生成（零手抄）。

## ③ 计算结果

clamp ↔ shell 逐点对照（末 40 % 窗口时间平均误差；伪流取末态；最后一列为「实际压缩比 / 所需压缩比」）：

| dx | c0 [m/s] | N | 步数 | 窗口 [t_σ] | clamp 误差 | shell 误差 | clamp \|v\|max | shell \|v\|max | 压缩比 clamp | 压缩比 shell |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.0 mm | 20 | 362 | 1283 | 1.50 | -37.9 % | **+8.8 %** | 0.030 | **0.388** | 0.57 | **2.41** |
| 2.0 mm | 5 | 362 | 592 | 1.50 | -34.0 % | **+4.9 %** | 0.026 | **0.104** | 0.57 | **1.03** |

解析压差 360.0 Pa。误差判定（关键量 $\Delta p$ vs $\sigma/R$）：

| 算例 | shell 误差（时间平均） | < 15 % | < 3 % |
|---|---|---|---|
| dx=2.0 mm, c0=20 | +8.8 % | 是 | 否 |
| dx=2.0 mm, c0=5 | +4.9 % | 是 | 否 |

图 `free_surface_pressure_closure.png`：(a) 核心压强时间迹与解析值；(b) 末窗误差 vs 15 %/3 % 验收带；
(c) 内部实际压缩比 / Young–Laplace 所需压缩比（= 1 才算压强边界被正确施加），clamp 与 shell 对照。

## ④ 逐项误差判定与「是否 < 3 %」的显式回答

- dx=2.0 mm, c0=20 m/s：压差误差 **+8.8 %** → < 3 %：**否**；< 15 %：**是**。
- dx=2.0 mm, c0=5 m/s：压差误差 **+4.9 %** → < 3 %：**否**；< 15 %：**是**。
- 伪流强度（Adami 2012 判据 $|v|_{max}/v_{ref} < 0.1$）：shell
  0.388, 0.104 vs clamp
  0.030, 0.026 → **0/2 通过，未过关**。
  闭合没有抑制伪流，反而放大了壳层伪流：外推压强梯度与 CSF 力在自由面壳层不匹配，
  需要下一轮单独处理（例如外推时并入 CSF 加速度 $\mathbf{a}_i$，见 ⑤.4）。
- 最好点 +4.9 %（dx=2.0 mm, c0=5），
  最差点 +8.8 %（dx=2.0 mm, c0=20）。
  **严格 3 % 门限：0/2 通过 → 未通过**，
  平台 15 % 门限：2/2 通过。所有数字如实记录，不粉饰。

## ⑤ 注意点 / 局限 / 适用范围

1. **默认路径不变**：`Config.surface_pressure_closure = False` 是默认值；关闭时 `rhs` 返回的压强与
   `tait_pressure(rho)` 逐位相同，`report.json` 的全部既有数字因此可复现。
2. **不是收敛结论**：只测了 dx = 2 mm 的两个 c0。dx = 1.5 mm 阶梯点未跑（本机 load15 长期 > 150，
   按容量纪律不硬起重算）；不要把本文件当作网格收敛证据。
3. **阈值 θ = 0.99 未做敏感性扫描**（预算所限）：θ 只决定闭合作用在哪些粒子上，不影响默认路径；
   下一轮应先做 θ ∈ {0.97, 0.99, 1.00} 扫描并记录。
4. **公式与本实现的差异（诚实披露）**：Adami et al. 的体力项是
   $\rho_j(\mathbf{g}-\mathbf{a}_i)\cdot(\mathbf{r}_i-\mathbf{r}_j)$，其中 $\mathbf{a}_i$ 是**非压强**加速度，
   使表面张力造成的法向应力跳跃也由体力项承载。本实现只含重力项
   $\rho_j\mathbf{g}\cdot(\mathbf{r}_i-\mathbf{r}_j)$（零重力算例中为 0），**未并入 CSF 加速度**；
   对纯毛细平衡这是残余误差的候选来源之一，尚未分离验证。
5. **踩坑之一**：pair 列表是**无序**的（每个无序对只存一条），外推累加必须对两个成员各做一次
   `index_add`，否则一半粒子的外推为 0（`test_linear_hydrostatic_field_is_reproduced_exactly` 一开始
   就是这么红的）。
6. **踩坑之二**：`donors="all"` 的字面公式会被自由面邻居的 $p=0$ 拉低，单测
   `test_all_donors_is_diluted_by_free_surface_zeros` 记录的正是这个稀释效应，故默认取
   `"interior"` + 多遍（`test_layers_fill_the_outermost_shell` 钉住逐层填充行为）。
7. **超出本文件覆盖的算例**：液滴撞击（`src/tensorsph/droplet_impact.py::run`）也走同一闭合，
   但本文件只证明静止毛细平衡下的作用；带惯性/壁面的算例需另建基准。

## ⑥ 与既有验证记录的关系

`docs/verification_laplace_pressure_floor.md` §7 的 P0 门槛项（四点上 `absolute_time_average_tail`
全部 < 15 %）在本单**部分结清**：2/2 已跑点达标，dx = 1.5 mm 未测。
`tests/test_laplace_pressure_floor.py::test_laplace_jump_within_15pct` 仍以 `report.json`（clamp）为输入，
因此**保持 xfail**；shell 闭合的判定单独记在 `tests/test_shell_closure_report.py` → `report_closure.json`。
