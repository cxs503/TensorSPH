# TensorSPH 首单健康盘点与首吸收：CSF 颜色梯度算子的符号/体积缺陷

**任务**：m1617（生态跟进首单，TensorSPH 首次纳入派发面）
**分支**：`agent/m1617`　**基线提交**：`400db02`（Merge PR #14）
**日期**：2026-11-01　**环境**：CPU-only（torch 2.7.1，`torch.cuda.is_available() = False`），128 核 / 1 TB，Python 3.11

---

## 1. 案例介绍

### 1.1 被审计的库

TensorSPH 是 PyTorch 原生的弱可压 SPH（WCSPH）研究求解器。目录结构与 README 声明的功能面对照：

| 目录 / 文件 | 内容 | 与 README 声明的关系 |
|---|---|---|
| `src/tensorsph/` | 6 个模块，2062 行：`droplet_impact`(1181)、`dam_break`(283)、`surface_tension`(261)、`neighbor_search`(227)、`__init__`(63)、`__main__`(47) | 与 README 的求解器描述一致 |
| `tests/` | 9 个测试文件 | README 写「104 unit tests」，实际收集 **114**（漂移 10 条，非缺陷但在场性声明失真） |
| `benchmarks/` | 6 组：dam_break、hydrostatic、free_surface、neighbor_search、scaling、static_droplet | README 的 B1/B2 门禁表逐项对应 |
| `docs/` | 8 篇 md + `docs/benchmarks/` 5 篇 | 对应 README 的门禁证据链 ✓ |
| `examples/` | `droplet_impact_2d/run.py` | ✓ |

README「Current limitations」仍写着 **“No surface-tension model”**，而 `src/tensorsph/surface_tension.py`（CSF 模型，261 行）自 `4357c66` 起已在库内、被 `__init__.py` 导出、并被 10 条测试覆盖。这是本库最明显的文档—实现不一致（本次一并记录，未改 README 正文，改由本报告与后续单跟进）。

### 1.2 本单聚焦的物理问题

CSF（Brackbill, Kothe & Zemach 1992）把表面张力写成体积力

```
f = σ κ n̂ δ_s ,   κ = ∇·n̂ ,  n̂ = -∇c / |∇c| ,  δ_s = |∇c|
```

即「颜色函数 c 的梯度」定义了法向、曲率和表面 delta 函数三件东西。二维圆盘界面的解析参照是 Laplace 定律 **Δp = σ/R**。

几何与参数（程序化生成，见 `runs/make_metrics.py`）：

| 参数 | 值 | 说明 |
|---|---|---|
| 粒子间距 dx | 0.01 / 0.005 m | 两套分辨率 |
| 核半径 h | `h_factor · dx` | 库缺省 `Config.h_factor` |
| 参考密度 ρ0 | 1000 kg/m³ | 水 |
| 表面张力系数 σ | 0.072 N/m | 水-空气，`Config.surface_tension_coefficient` |
| 液滴半径 R | 0.05 / 0.10 m | 曲率 1/R = 20 / 10 m⁻¹ |
| 粒子排布 | ① 直角点阵切圆盘（stencil 对称，用于线性复现）② 同心环圆盘（用于曲率/Laplace） | |
| 重力 | 0（静态液滴） | 隔离表面张力通道 |

---

## 2. 怎么做的

### 2.1 判据先于动手（预注册）

在触碰源码前先为每一级找**解析参照**，避免「自洽即通过」：

| 编号 | 判据 | 解析答案 |
|---|---|---|
| G1 | 线性颜色场 c = x 的 SPH 梯度 | (1, 0)，逐粒子严格 |
| G2 | 线性场 c = y + 0.5x 的梯度（破坏对称方向） | (0.5, 1) |
| N1 | 圆盘旋涡界面上 CSF 法向方向 ⟨n̂·r̂⟩ | +1（外法向） |
| K1 | 给定解析径向法向，`compute_curvature` 的散度 | div(r̂) = 1/r，逐粒子 |
| C1 | 凸液滴 rim 上的曲率 | κ = +1/R > 0（库内 docstring 明确此约定） |
| L1 | 沿半径积分 ρ·a_r 得到的压差 | σ/R |

### 2.2 红—绿流程

1. 用独立探针脚本（`runs/probe_*.py`）量化现状；
2. 把判据写成 `tests/test_surface_tension.py` 的 5 条新测试，**先在未修改的源码上跑**，得到失败签名（`runs/surface_tension_RED.log`）；
3. 做最小修复；
4. 同一组测试转绿（`runs/surface_tension_GREEN.log`），再跑完整套件做回归门禁。

### 2.3 工具链与证据链

* 未修改版本的源码副本由 `git archive HEAD` 解到 `/tmp/pristine_tp`，用 `PYTHONPATH` 切换被测源码，保证红/绿两次跑的是**同一组测试、不同源码**。
* 所有数字由 `runs/make_metrics.py` 程序化产出 → `artifacts/m1617_metrics.json`，图由同一脚本生成；本报告不手抄任何数字。
* **环境陷阱（重要）**：本镜像里 `pip` 的可编辑安装把 `tensorsph` 指向**另一个 agent 的工作树** `/root/tp-agent/work/m507/src`。首次运行测试时被测源码并非本分支。发现后用 `PYTHONPATH=<本树>/src` 显式钉住，并 `diff` 确认 m507 的 `surface_tension.py` 与 `400db02` 逐字节一致，红/绿证据全部重新采集。所有跨 agent 的测试证据都受此风险影响。

---

## 3. 计算结果

### 3.1 缺陷定位

`compute_color_gradient` 的 SPH 离散写成

```
grad(c)_i = Σ_j V_j (c_j − c_i) ∇_i W_ij ,   V_j = m_j/ρ_j
```

配对列表每对只出现一次，因此粒子 `j` 必须从同一条记录里拿到**自己那一份**贡献。由 `∇_j W_ji = −∇_i W_ij` 可得正确写法是

```
grad(c)_j = V_i (c_j − c_i) ∇_i W_ij        （体积用 i 的，符号与 i 行同号）
```

而库内实现是 `index_add(j, −V_j (c_j − c_i) ∇W_ij)`：**体积错用 j、符号取反**。在对称 stencil 上两者逐对精确抵消，梯度恒为 0——这正是「线性场复现率 0%」的来源，且因为 `c ≡ const` 时 `dc = 0`，原有测试（只测均匀场）完全看不见它。

`compute_curvature` 的 `j` 行（`+V_i (n_j−n_i)·∇W`）是**正确的**——同一文件里两个算子写法不一致，只有前者出错（这也解释了为什么 K1 判据一直是绿的）。

### 3.2 表 1：线性场梯度复现（判据 G1，解析值 1.0）

| dx | 粒子/内部 | 修复前 | 修复前误差 | 修复后 | 修复后误差 |
|---|---|---|---|---|---|
| 0.01 | 305 / 237 | 0.00000 | **100.0 %** | 1.01054 | **1.05 %** |
| 0.005 | 1245 / 1053 | 0.00000 | **100.0 %** | 1.01054 | **1.05 %** |

残差 1.05 % 与 dx 无关，是核零阶矩（离散归一化）偏置，不是空间离散误差；两条判别方向（G2）同样复现到 <2 %。

### 3.3 表 2：rim 量（同心环圆盘，σ = 0.072 N/m）

| dx | R | rim 粒子 | ⟨n̂·r̂⟩ 修复前 | 修复后 | κ 修复前 [1/m] | 修复后 | 1/R |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.05 | 20 | −0.936 | **+0.968** | −86.33 | **+10.58** | 20.0 |
| 0.01 | 0.10 | 63 | −0.998 | **+0.998** | −91.33 | **+7.73** | 10.0 |
| 0.005 | 0.05 | 63 | −0.998 | **+0.998** | −182.65 | **+15.46** | 20.0 |
| 0.005 | 0.10 | 119 | −0.998 | **+0.998** | −181.45 | **+7.99** | 10.0 |

三点结论：① 修复前法向**朝内**（与 docstring 的外法向约定相反）；② 修复前 κ 的量级随 h 反比放大（−86 → −183，即 ∝1/h），是「法向在单层带外被置零」造成的带宽伪曲率，而不是界面曲率；③ 修复后符号与外法向约定一致，量级回到 0.53–0.8/R。

### 3.4 表 3：解析曲率估计算子本身（判据 K1，给定解析法向，误差 = |κ−1/r|/(1/r)）

| dx | 中位误差 | 最大误差 |
|---|---|---|
| 0.01 | 2.09 % | 4.28 % |
| 0.005 | 1.30 % | 1.81 % |

散度离散本身是好的——缺陷只在颜色梯度与「法向带被置零」的接口上。

### 3.5 表 4：Laplace 压力（判据 L1）

| dx | R | 修复前 Δp/(σ/R) | 修复后 Δp/(σ/R) | 与 1 的偏差（修复后） |
|---|---|---|---|---|
| 0.01 | 0.05 | 0.256 | 0.167 | −83 % |
| 0.01 | 0.10 | 0.583 | 0.425 | −57 % |
| 0.005 | 0.05 | 0.583 | 0.425 | −57 % |
| 0.005 | 0.10 | 1.063 | 0.593 | −41 % |

### 3.6 图

![color-gradient defect](figures/m1617-color-gradient-defect.png)

**图 1**　(a) 线性场梯度复现误差（对数轴）：修复前恒为 100 %（梯度被逐对抵消为 0），修复后 1.05 %。(b) rim 法向与径向单位矢量的内积：修复前 −0.94…−1.00（朝内，违反库内约定），修复后 +0.97…+1.00。(c) 回收的 Laplace 压差与 σ/R 之比：修复前 0.26–1.06、修复后 0.17–0.59，两者都远离 1（详见 §6）。

---

## 4. 测试基线（库健康盘点）

| 项 | 结果 | 证据 |
|---|---|---|
| 收集 | 114 条（README 声称 104） | `pytest --collect-only` |
| 基线（未修改） | **114 passed, 0 failed, 58.61 s** | `runs/pytest_baseline.log` |
| 未修改源码 + 新增 5 条测试 | `4 failed, 115 passed, 2 warnings in 68.99s (0:01:08)` → 4 条失败全部来自新增的解析判据，**既有 114 条全绿** | `runs/pytest_full_pristine_src.log` |
| 修复后完整套件 | `119 passed, 2 warnings in 82.66s (0:01:22)` | `runs/pytest_full_GREEN.log` |
| 环境红 | 无。CPU-only：CUDA 用例（`benchmarks/scaling`）不在测试套件内，未触发红 | `runs/env.txt` 性质的环境探针 |
| 静态检查 | `ruff check` 在本单触及的两个文件上**既有** 5 条（src）/ 28 条（tests）报错，修复前后计数不变 | 与 `git show HEAD:` 版本对比 |

基线签名：`test_adaptive_dt / boundary / dam_break / droplet_impact / free_surface / neighbor_search / pair_based_rhs / surface_tension / wall_pressure` 全绿，无可复现真红；本轮新增 5 条测试后为 119 条。

---

## 5. 修复

`src/tensorsph/surface_tension.py::compute_color_gradient`，两行：

```python
-    vol_j = mass[j] / density[j]
-    grad_c = grad_c.index_add(0, i,  vol_j[:, None] * dc[:, None] * grad_w)
-    grad_c = grad_c.index_add(0, j, -vol_j[:, None] * dc[:, None] * grad_w)
+    vol = mass / density
+    grad_c = grad_c.index_add(0, i, vol[j, None] * dc[:, None] * grad_w)
+    grad_c = grad_c.index_add(0, j, vol[i, None] * dc[:, None] * grad_w)
```

并补齐 docstring 里的推导。改动仅限该函数；`compute_curvature`、`capillary_dt_limit`、`rhs` 的调用方式均未动，物理标定未被改动（故不宣称精度提升，见 §6）。

新增测试（`tests/test_surface_tension.py`，5 条，全部程序化解析参照）：

| 测试 | 修复前 | 修复后 |
|---|---|---|
| `test_color_gradient_reproduces_linear_field` | FAIL（0.0 vs 1.0） | PASS（1.05 %） |
| `test_color_gradient_direction_is_from_low_to_high_colour` | FAIL | PASS |
| `test_surface_normal_points_outward_on_a_droplet` | FAIL（⟨n̂·r̂⟩ = 0.000） | PASS（+0.97） |
| `test_surface_tension_contracts_a_convex_interface` | FAIL（κ = −86） | PASS |
| `test_curvature_recovers_analytic_divergence_of_radial_normal` | PASS（2.09 %） | PASS（1.30–2.09 %） |

红：`4 failed, 11 passed, 2 warnings in 6.89s`（`runs/surface_tension_RED.log`）；绿：`15 passed, 2 warnings in 6.37s`（`runs/surface_tension_GREEN.log`）；全量套件 `119 passed, 2 warnings in 82.66s (0:01:22)`。

---

## 6. 误差判定（逐项，基准=解析解）

| 关键量 | 基准 | 修复前误差 | 修复后误差 | 是否 < 3 % |
|---|---|---|---|---|
| 线性场梯度 ∂c/∂x | 1.0 | 100 % | **1.05 %** | **是** |
| 线性场梯度 ∂c/∂y（斜置场） | 1.0 | 100 % | < 2 % | **是** |
| 曲率估计算子 (div r̂) | 1/r | 2.09 % / 1.30 % | 2.09 % / 1.30 % | **是** |
| rim 法向方向 ⟨n̂·r̂⟩ | +1 | 187 %（−0.94…−1.00，反向） | **0.2 %–3.2 %** | 边界（dx=0.01,R=0.05 为 3.2 %，其余 <0.3 %） |
| rim 曲率 κ | 1/R | 量级偏差 4–18 倍、符号反向 | 0.53–0.80 × (1/R)，即 −20 %…−47 % | **否** |
| Laplace 压差 Δp | σ/R | 0.26–1.06（分辨率不一致） | 0.17–0.59（−41 %…−83 %） | **否** |

**显式回答**：算子级三项关键量均 < 3 %；**界面曲率与 Laplace 压差两项不达标**（远超 15 % 不可接受线）。本单**不宣称**表面张力的物理精度提升。

对「修复前在 dx=0.005、R=0.1 上 Δp/(σ/R)=1.063（误差 6 %）」的说明：那是两个错误相乘的巧合——κ ∝ 1/h 被数值放大（−182 1/m，而物理值是 10 1/m），同时 δ_s 只有应有量级的 ~0.22 倍；一旦换分辨率或换半径就崩（同一公式在 dx=0.01、R=0.05 上给出 0.256）。修复前该指标**没有收敛性**，因此不能作为「模型曾是对的」的证据。

---

## 7. 注意点与局限

1. **环境陷阱（跨 agent 通用）**：`import tensorsph` 解析到 `/root/tp-agent/work/m507/src`。任何 agent 若不显式 `PYTHONPATH`，跑出来的红/绿都是别人的源码。本单已重采全部证据；建议生态侧给每个 worktree 统一 `pip install -e` 隔离或强制 `PYTHONPATH`。
2. **还缺的修复（后续单）**：`compute_surface_tension_force` 的颜色函数用「邻居计数 / 平均邻居数」，rim 上 c 只从 1.0 掉到 0.63–0.66，跨界面只变化 ~0.36，因此 `δ_s = |∇c|` 沿法向积分 ≈ 0.22（应有 1.0）；同时法向在带外被置零，使 κ 只到 0.53–0.8/R。两者叠加即 §6 的 −41 %…−83 %。建议改法：c_i = Σ_j V_j W_ij（核和指示函数）+ 对 δ_s 做离散归一化（Σ|Δc| 或跨越积分=1），并以 L1 判据做门禁。本单不实施，避免在无收敛证据时改物理标定。
3. **格点台阶效应**：在直角点阵切出的圆盘上，rim 是阶梯状，局部曲率符号会翻转，修复后**净**径向加速度在点阵几何上仍可为正（+0.077 m/s²），在同心环几何上为负（−0.24…−2.19，内向）。测试因此用同心环几何，并在报告里同时给出两个数，不做几何挑选式叙事。
4. **README 失真**：仍写「No surface-tension model」、测试数 104（实际 114/119）。未改 README，原因是该文档的 Verification 表由既有门禁维护，需与前一条「CSF 未验证」的结论一起改，留给后续单。
5. **ruff**：`select = E,F,W` 在 `surface_tension.py` 上既有 5 条、`test_surface_tension.py` 上既有 28 条报错，本单未新增任何一条（与 `git show HEAD:` 版本计数逐一对齐），也未顺手修，避免无关改动。
6. **未做的事**：未跑 GPU 基准（环境无 CUDA）；未做 B1/B2 门禁复跑（60 s+ 的案例，与本单算法缺陷无耦合）；未对 `rhs` 做端到端液滴仿真——Laplace 标定问题需要先解决第 2 条，届时再用端到端仿真复核。

---

## 8. 仓库状态（任务①）

* `git log -15` 主题：0.5.0→0.8.0 四个功能版本（自适应步长、壁压外推、自由面备选、力评估提速）+ 本派发面此前的三次 agent 贡献（首波审计 m239、GPU 标定 m276、表面张力 m277）+ 两次评审/合并（m281、m277）；本单是 TensorSPH 在 99 派发面的**第 0→1 单**。
* remote：`origin https://github.com/cxs503/TensorSPH.git`；`git rev-list --left-right --count origin/main...HEAD` = **0 / 0**（本地快照与上游 main 无领先/落后）。容器内不做 fetch，故「上游实时领先量」未测，标注为未验证。
* 工作规范：改动仅 `src/tensorsph/surface_tension.py` 与 `tests/test_surface_tension.py`，加 `docs/reports/`、`docs/figures/`、`artifacts/`、`runs/` 工件；未 push（推送归 `tp_pusher`）。

## 9. 复现

```bash
cd /root/tp-agent/work/m1617
export PYTHONPATH=$PWD/src OMP_NUM_THREADS=4
python3 runs/make_metrics.py                       # 数字 + 图 + artifacts/m1617_metrics.json
python3 -m pytest tests/test_surface_tension.py -q # 15 passed
python3 -m pytest tests -q                         # 全量
```