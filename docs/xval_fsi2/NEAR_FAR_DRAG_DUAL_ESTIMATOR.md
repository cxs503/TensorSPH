# fsi2 刚体前驱：近场/远场双估计器与入流-出流质量账

**锚点** `XVAL-SPH-FSI2-BENCHMARK-AND-FROM-FLUID` · 仓 TensorSPH · 分支 `agent/m22888` · 域 SPH / FSI

## ① 案例介绍

Turek–Hron FSI2 的**刚体前驱**：通道 `L = 2.5 m × H = 0.41 m`，圆柱 `D = 0.1 m`（心 `(0.2, 0.2)`），
梁 `L_b = 0.35 m × t_b = 0.02 m` 挂在柱后，`Re = 100`（`nu = 2.0e-4 m²/s`），`rho = 1000 kg/m³`，
入流抛物 `U_max = 0.3 m/s`（`U_mean = 0.2 m/s`），`c0 = 5 m/s`。结构**全部冻结**（柱 + 梁边界零速、
节点不推进），流体算子就是生产算子，所以量到的只有流体那一半，界面子迭代与杆件都被排除。

为什么做这件事：耦合臂在本锚点读 `Cd_mean = 0.052` 对 Turek–Hron `3.2`（−98 %，
`VIV_PERIODIC_VERDICT.md`），而载荷通道本身在弯曲壁面上还有 `+31..36 %` 的偏置
（`CYLINDER_LOAD_AT_BENCHMARK_SCALE.md`）。两个误差源共处一条回路，谁也判不了谁——
`GAP-SPH-FSI-BENCHMARK_xval.md` 因此把「与文献 Cd/Cl 对比」记成 `not_run`。
本单给这件事补上**同一场上的第二个独立估计器**，并把验收判据写在跑之前。

## ② 怎么做的

1. **新增远场（控制体）读数**（`benchmarks/xval_fsi2/rigid_cfd_precursor.py`）：
   `momentum_plane_flux()` 在竖直平面 `x = const` 上取 `int (rho u_x^2 + p) dy`，用已验的
   `tensorsph.farfield_force.wake_plane_drag(u_inf=0, p_ref=0, reconstruct="fit")` 实现
   （`u_inf=0, p_ref=0` 时该函数正是 `int (-rho u_x^2 - p) dy`；求解器压力基准是
   `tait_pressure`，`rho = rho0` 时 `p = 0`，故 `p_ref = 0` 是正确基准，不需估计偏置）。
   `farfield_drag()` 取两平面之差，平面 `x = 0.10 m`（柱前）与 `x = 0.75 m`（梁尖后）；
   该 0.65 m 上无滑移壁面的层流摩阻约 `6 mu U_mean/H · 0.65 = 4e-4 N/m`，占载荷 ~6 N/m 的 0.007 %。
2. **判定逻辑前移**（`src/tensorsph/fsi2_near_far_verdict.py`）：门限在跑之前写死——
   近/远场时均一致性 `|Δ| <= 15 %`；归因带 `Cd_far / 3.2 in [1/3, 3]`（只判"流体这一半能否
   承载 O(1) 阻力"，**不作** <15 % 精度声明）；平稳窗用分块均值的相对展布 `<= 30 %` 判定，
   不平稳就 `withheld`，不允许"缩短窗口直到通过"。
3. **守护**：`tests/test_fsi2_near_far_verdict.py`（8 项）——门必须**能失败**（0.05 vs 3.0 判 fail、
   反号判 fail、边界 15 % 判 pass、1.2 倍门限判 fail）、平稳性检验必须拒绝漂移序列、
   Strouhal 只在 ≥3 个过零点时给数、`verdict()` 在漂移记录上必须 `withheld`。
4. **算力**：`dx = 0.02` 长时跑（24 000 步，t ≈ 25 s）已递交调度（作业 `cj4462`，32 核），
   本单交付时仍在排队；本文报告的是**本机可负担的粗档 `dx = 0.04`**（6 线程，100 s 墙钟
   上限，2 516 步 / t = 5.16 s），并在文中标明它只是**案例可行性档**，不是精度档。

## ③ 计算结果（`runs/m22888/rigid_cfd_dx004.json`，程序化注入）

| 量 | 值 |
|---|---|
| samples / t_end | 63 / 5.158 s |
| fluid mass first → last | 1043.3 → 771.2 kg (-26.1%) |
| particles first → last | 652 → 482 |
| injected / removed (total) | 0 / 0 |
| near-field Cd (median of window) | 79.9 |
| far-field Cd (median of window) | -35.6 |
| near/far rel. diff | -2 |
| near/far gate (@15%) | fail |
| order gate | not_run (ratio nan) |
| window accepted | False |
| verdict status | withheld |

读数（全部由工件与判据程序产出，无手抄）：

* **质量账先坏，阻力问题问不出来**：出流移除 `399` 个粒子、入流只补 `227` 个，流体质量
  `1043 → 771 kg`（**−26 %**），粒子数 `652 → 482`。这是逐行入流时钟的**中点求积欠交付**：
  `dx = 0.04` 时入口列只有 `H/dx ≈ 10` 行，`Σ_j u(y_j) ≈ U_mean H/dx` 的求积误差比 `dx = 0.02`
  （20 行）大 4 倍，且柱面遮蔽掉了整列粒子，而行累加器仍按几何行记账。
  同一代码 `dx = 0.02` 的 800 步记录是 `inj 240 / rem 234`、质量 `+0.23 %`（`ROW_CLOCK_DEFECT.md`），
  即**这个欠交付只在粗档暴露**——所以它是分辨率门槛，不是常数偏置。
* 近场 `Cd` 中位 `79.9`、远场 `-35.6`：两臂都在
  随质量流失漂移，且反号（`rel_diff = -2`）。
* 判据给出 `status = withheld`，`reason = "stationary window rejected: block means still drifting"`。**本单不发布任何精度判定**。

## ④ 结果图

`docs/xval_fsi2/assets/near_far_drag_dx004.png`：上=近场/远场 `Cd` 随时间的对比（虚线为 Turek–Hron
`3.2`）；下=流体质量（实线）与粒子数（虚线）。质量单调下降与 `Cd` 的单调增长同相，正是"载荷读的是
一个正在排空的系统"的直接证据。

## ⑤ 注意点 / 局限

1. **`dx = 0.04` 只有 2.5 个粒子跨柱径**（`D/h = 1.9`），几何与尾流都不可信；本文用它只回答
   "粗档上什么先坏"，答案就是质量账。
2. **远场缺非定常项**：`farfield_force` 不实现 `-d/dt int rho u dV`（2-D 中其均匀部分按 `R²`
   发散，必须与压力通量一起离散），故远场只在时均意义上有定义；本文两臂都在漂移，
   时均也没有意义 → `withheld`。
3. **壁面摩阻未单列**：平面差里含两平面之间壁面的摩阻（估 0.007 %），本文按可忽略处理并记录。
4. **未跑**：`dx = 0.02` 长时档（`cj4462` 排队中）；柔性梁耦合臂；与 Turek–Hron 的
   `Cd/Cl` <15 % 判定（需要柔性梁 + 长时积分）。
5. 远场读数用 `reconstruct="fit"`（线性场精确、去掉薄片平均的一阶残差），未跑 `mean` 控制臂。

## ⑥ 误差判定

| 判定项 | 基准 | 结果 | 是否 <15 % / <3 % |
|---|---|---|---|
| 近场 vs 远场 `Cd`（同一场上两个独立归约） | 互为基准 | `rel_diff = -2` | **否**（且反号，`fail`） |
| 流体这一半能否承载 O(1) 阻力 | `Cd_far / 3.2 in [1/3, 3]` | `not_run` | **未判定（not_run）** |
| `Cd` / `Cl` vs Turek–Hron | 3.2 / 0.22 | 未跑（需柔性梁） | **not_run**，不在本单声明范围内 |
| 流体质量守恒（自查） | `0 %` | `-26.1%` | **否**（≥3 % 不可接受） |

honest FAIL：本档位下**没有任何一项**达到 15 % 门或 3 % 门；阻塞项是逐行入流时钟在粗档的
质量欠交付，不是阻力算子。给下一轮的明确线索：先让 `dx = 0.02` 的长时档（`cj4462`）证明质量账
在该档闭合（`inj ≈ rem`、质量漂移 <1 %），再谈 `Cd` 归因；粗档若要复检，须给入口列加
"按实际可用行数归一"的通量修正，而不是加大 `dx`。

复现：

```bash
PYTHONPATH=src python benchmarks/xval_fsi2/rigid_cfd_precursor.py --dx 0.04 \
    --steps 6000 --sample-every 40 --threads 6 --max-seconds 100 \
    --out runs/m22888/rigid_cfd_dx004.json
PYTHONPATH=src python -m tensorsph.fsi2_near_far_verdict runs/m22888/rigid_cfd_dx004.json \
    --out runs/m22888/rigid_cfd_dx004_verdict.json
```
