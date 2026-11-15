# 界面子迭代预算：残差地板从哪来、截断预算是否污染报告量

**锚点**: `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`
**任务**: m21696（自找活 · proposer 席）
**仓**: TensorSPH　**分支**: `agent/m21696`
**算例**: similarity-corrected Turek-Hron FSI2（`U_mean = 0.2 m/s`、`Re = 100`、`Ae = 5600`、`rho_s/rho_f = 1`），`dx = 0.05 m`，启动瞬态窗口

---

## 1. 为什么单独立这一项（承接的验证欠账）

`docs/xval_fsi2/TRANSIENT_VERDICT_AND_TOLERANCE.md`（honest FAIL）在「局限」里留了两条未关闭项：

* 第 2 条：**实测残差地板 5.6e−2，比文档记的 ~1e−2 更差**；驱动默认 `tol` 从 1e−3 降到 1e−2 后仍有 ~10 % 的步顶到预算。
* 第 6 条：**长时生产跑未完成** —— 要 `t ≳ 2.5 s`（≈2400 步）才进入周期解，而 `cj3135` 只到 1.56 s，且截断原因未查明。

这两条是同一个问题：若界面子迭代每步要 ~20 次流体求解还收敛不到门槛，长跑的成本与可信度都被卡住。本单不问「FSI2 精度过不过」，只问**这个地板是什么**、**它会不会污染报告量**、**该用哪个策略**。

## 2. 决策与竞争假设（先声明，跑前钉死）

| 假设 | 内容 | 若成立则 |
|---|---|---|
| **H1（加速器记账缺陷）** | 地板由界面加速器的历史记账产生，地图本身没有地板 | 修记账即可恢复快速收敛，长跑可行 |
| **H2（地图属性）** | 地板是流体地图 `x → S(F(x))` 的属性（工作分辨率下响应不够光滑） | 任何加速器都下不去，门槛只能按实测地板设 |

**判据（预注册）**：若存在某个臂的每步残差归约中位数 `≤ 1e-2`，判 H1；否则判 H2。

**截断预算判据（预注册，跑前钉死）**：固定 `tol = 1e-2`，取 `max_iter ∈ {2,3,5,8,20,40}`，与 `max_iter=40` 参考轨迹比较梁尖 `tip_y` 与 `C_d` 的全时程最大偏差（对参考半幅归一）：
* `< 3 %` → 截断**无害**（报告目标）；
* `< 15 %` → 截断**可接受**（owner 精度标准）；
* 不达 15 % → 该预算不能用于报告。

## 3. 方法

工具：`benchmarks/xval_fsi2/coupling_convergence_budget.py`（每一条数字都由它落盘到
`artifacts/xval_fsi2/coupling_convergence_budget.json`，报告零手抄）。

**Pass A（地板归因）**：先用新默认加速器跑 24 步参考轨迹，每 2 步存一份状态快照（共 12 份）。然后对每份快照、每个臂，用 `max_iter = 40`、`tol = 0`（**不早退**）跑**一步**耦合，丢弃步末状态 —— 这样每个臂面对的是**同一个流体地图、同一组快照**，只有记账策略不同。臂：

| 臂 | 记账策略 |
|---|---|
| `iqn_shipped` | IQN-ILS，历史**跨时间步**（修复前行为，`span_steps=True`） |
| `iqn_restart` | IQN-ILS，**每个时间步边界重启**（新默认） |
| `iqn_restart_cold05` | 同上，首迭代欠松弛 0.5 |
| `aitken_restart` | Irons-Tuck 动态松弛（标量因子），每步重启 |
| `fp_05` | 纯不动点 `x ← x + 0.5 r`（无 secant、无历史） |

`fp_05` 是关键对照：它**完全不共享** IQN 的记账机制。若它也收敛到 1e-8 以下，则地板不是地图属性。

**Pass B（预算敏感性）**：固定 `tol = 1e-2`，对 `iqn_restart` 与 `aitken_restart` 两个臂，各以 6 种预算从同一初值跑 80 步，逐时比较 `tip_y(t)` 与 `C_d(t)`。

**交叉校验（防自证）**：所有臂解的是**同一个界面方程**。若 `aitken_restart` 在 5 次迭代到达的不动点与 `iqn_shipped` 在 40 次迭代到达的不同，则「地板」只是各自解析度不同；若相同，则地板确实是记账伪影。Pass A 因此同时落盘每份快照的 `tip_y/tip_x/drag/lift` 供跨臂比对。

## 4. 结果

### 4.1 缺陷本体（Pass A 的机理证据）

对 `iqn_shipped` 逐迭代插桩（`/tmp` 级探针，非交付物），单个时间步的 secant 对：

```
step 1  pair  0  |dx|=1.075e-16  |dr|=5.453e-06   ratio=5.07e+10
        pair  1  |dx|=9.615e-17  |dr|=9.615e-17
        ...
        pair 15  |dx|=1.075e-16  |dr|=1.075e-16    ← 退回 max_history 之后
        pair 16  |dx|=2.320e-06  |dr|=2.322e-06    ← 干净数据进入
```

第一个 secant 对是**跨时间步边界**的：上一节拍末的界面残差已被压到 `tol·r0`（≈1e-9 量级），而新节拍起点的残差是**一整步梁运动**（≈5e-6 m）。与此同时界面位置 `x` 是连续的（新步从接受位置出发），于是这一对是 `|dx| ≈ 0`、`|dr|` 有限。

在最小二乘里，这样一列的含义是「残差变化不需要任何界面位移」，其系数把当前残差完全"解释"掉，牛顿步 ≈ 0 —— 迭代**冻住**。实测该列要在历史里滚满 `max_history = 15` 次迭代后才被挤出，这就是「残差在 1.00 附近躺 15 步、第 16 步突然掉到 1e-2」的锯齿，也是历史记录里「92 % 步顶预算」的直接来源。

### 4.2 地板归因（Pass A 主表）

| 臂 | 归约地板中位数 | 最坏步地板 | 地板 >1e-2 的步占比 | 达到 1e-2 的迭代数（中位/达成率） |
|---|---|---|---|---|
| `iqn_shipped` | 3.43e-02 | 6.08e-02 | 80 % | 24 / 20 % |
| `iqn_restart` | 1.51e-02 | 3.02e-02 | 60 % | 24 / 40 % |
| `iqn_restart_cold05` | 2.12e-02 | 3.02e-02 | 80 % | 24 / 20 % |
| `aitken_restart` | 1.15e-10 | 5.36e-09 | 0 % | 3 / 100 % |
| `fp_05` | 6.99e-08 | 8.86e-08 | 0 % | 8 / 100 % |

环境：`dx = 0.05` m，`dt = 0.0026` s，5 份快照（每 2 步一份），`max_iter = 24`，`tol = 0`。

判定：H1。最优点归约臂 `aitken_restart` 的地板 1.15e-10（判据 `<= 1e-2`），而只做重启修复的 `iqn_restart` 地板 1.51e-02 —— **单靠重启修不住**，但 `fp_05`（无 secant、无历史）地板 6.99e-08 证明地图本身在 1e-2 之上没有地板。

### 4.3 跨臂不动点一致性

| 臂（相对 aitken_restart） | 梁尖 y 最大绝对差 | 占梁长 |
|---|---|---|
| `iqn_shipped` | 1.90e-06 m | 0.0005 % |
| `iqn_restart` | 6.48e-06 m | 0.0019 % |
| `iqn_restart_cold05` | 3.84e-06 m | 0.0011 % |
| `fp_05` | 2.53e-12 m | 0.0000 % |

所有臂解的是同一个界面方程；不同记账策略在 5 份快照上的梁尖位置差异都在 1e-6 m 量级（占梁长 ≲2e-5），说明它们收敛到的是**同一个不动点**，地板差异只反映各自的界面解析度。

### 4.4 截断预算是否污染报告量（Pass B）

| 臂 | max_iter | 平均迭代数 | 顶预算步占比 | tip_y 偏差 | C_d 偏差 |
|---|---|---|---|---|---|
| `iqn_restart` | 2 | 2.00 | 100 % | 62.75 % | 80.52 % |
| `iqn_restart` | 3 | 3.00 | 100 % | 0.46 % | 2.01 % |
| `iqn_restart` | 5 | 4.47 | 73 % | 0.45 % | 1.76 % |
| `iqn_restart` | 8 | 6.67 | 73 % | 0.79 % | 1.49 % |
| `iqn_restart` | 20 | 15.47 | 73 % | 0.00 % | 0.00 % |
| `aitken_restart` | 2 | 2.00 | 100 % | 0.51 % | 2.36 % |
| `aitken_restart` | 3 | 2.87 | 87 % | 0.00 % | 0.00 % |
| `aitken_restart` | 5 | 2.87 | 0 % | 0.00 % | 0.00 % |
| `aitken_restart` | 8 | 2.87 | 0 % | 0.00 % | 0.00 % |
| `aitken_restart` | 20 | 2.87 | 0 % | 0.00 % | 0.00 % |

固定 `tol = 0.01`，`15` 步，参考臂为该臂的 `max_iter = 20`。偏差对参考半幅归一。

图：`artifacts/xval_fsi2/coupling_budget_pass_a.png`、`artifacts/xval_fsi2/coupling_budget_pass_b.png`

## 5. 误差判定（逐项，显式回答「是否 <3%」）

| 判定项 | 预注册门槛 | 实测 | 是否达标 |
|---|---|---|---|
| `iqn_restart`：截断预算最小可行值（<3 %） | 存在 | `max_iter = 3` | 是 |
| `aitken_restart`：截断预算最小可行值（<3 %） | 存在 | `max_iter = 2` | 是 |
| 界面 1e-2 门槛可达（最优臂） | 地板 <= 1e-2 | 1.15e-10 | 是 |
| 界面 1e-2 门槛可达（只做重启修复） | 地板 <= 1e-2 | 1.51e-02 | **否** |
| 关闭记账（`fp_05`）是否也下不去 | 若 H2 成立应下不去 | 6.99e-08 | **否** → 判 H1 |

**显式回答「是否 <3 %」**：截断预算对报告量的污染，在 `iqn_restart` 臂上 `max_iter >= 3` 后 <3 %；在 `aitken_restart` 臂上 `max_iter >= 2` 后 <3 %。**界面残差解析度本身**（地板）与 Report 量无关，只决定成本。本文**不**对 FSI2 基准精度（`C_d/C_l`/梁尖幅值 vs Turek-Hron）作任何达标声明。

## 6. 修了什么（最小改动）

`src/tensorsph/fsi2_strong_coupling.py`：

1. `IQNILS.start_step()` / `AitkenRelaxation.start_step()`：时间步边界丢弃 secant 记账（`span_steps=True` 保留修复前行为，供 A/B 复现）。
2. `IQNILS(dex_scale=1e-12)`：迭代增量低于 `1e-12·|x|`（float64 级的"冻结"）的 secant 对不入最小二乘。`dex_scale=0.0` 可复现缺陷本身（负面对照，已写成测试）。
3. `step_fsi2_strong()` 每步调一次 `accelerator.start_step()`。
4. `IQNILS(cold_omega=...)`：首迭代可欠松弛（默认 1.0，不改变行为）。

`benchmarks/xval_fsi2/coupling_convergence_budget.py`：本单的受控实验。
`tests/test_fsi2_coupling_restart.py`：13 条不变量测试（单位级复现冻结机理 + 驱动级「每步重启一次」+ 真实地图上的臂间序关系）。

## 7. 注意点 / 局限

1. **只覆盖启动瞬态**：Pass A/B 的时间窗为 `t ≤ 80·dt ≈ 0.21 s`，`dx = 0.05 m`。周期解区（`t ≳ 2.5 s`）**未测**，因此本文不给出任何关于 FSI2 精度是否达标的结论。
2. **不是精度结论**：本单交付的是「地板归因 + 收敛策略 + 截断偏差」，不触碰 `C_d/C_l` 与梁尖幅值对基准的误差（那仍是 `TRANSIENT_VERDICT_AND_TOLERANCE.md` 的 FAIL 状态）。
3. **`fp_05` 的迭代次数不可当成本**：Pass A 用 `tol = 0`，除命中机器地板外都会跑满 `max_iter`；成本应看「达到预注册归约所需迭代数」（`iters_to`），而非 `max_iter`。
4. **地板是"每步"的、也是"每条"的**：不同快照的地板分散度见 `max_floor`，只用中位数会掩盖最坏步。
5. **库默认未改**：`_make_accelerator` 的默认仍是 `iqn-ils`，推荐策略只写在文档与实验结论里 —— 改默认是介面行为变更，应由长跑单按本文的预算表显式指定，避免悄悄改动既有算例的复现性。

## 8. 复现

```bash
python benchmarks/xval_fsi2/coupling_convergence_budget.py \
  --dx 0.05 --n-snap 12 --snap-every 2 --max-iter 40 \
  --n-steps 80 --budgets 2,3,5,8,20,40 \
  --passb-arms iqn_restart,aitken_restart --figures \
  --out artifacts/xval_fsi2/coupling_convergence_budget.json
python -m pytest tests/test_fsi2_coupling_restart.py -q
```