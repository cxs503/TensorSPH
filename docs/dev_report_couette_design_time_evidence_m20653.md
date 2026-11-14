# Couette 壁面剪切证据的时程前提：把「启动瞬态读数」从精度声明里剔除（m20653）

**锚点**：`XVAL-SPH-METHOD-SELECTION-2026-11-12`（TensorSPH，method-selection 做法）
**类型**：验证欠账偿还 + 选择性缺陷修复（library: `tensorsph.method_selection`）

## 1 问题

方法选择器（`src/tensorsph/method_selection.py`）对 `wcsph_wall_no_slip` 的登记证据全部来自
`benchmarks/couette/out/results.json`，而该工件是 `run.py --quick` 跑出来的：
`DESIGN.md` 规定的终端时间 `t_end = 1.2 H²/ν = 0.60 s` 被 `--quick` 硬改成
`0.15 s = 0.3 H²/ν`（`run.py:715-718`）。0.3 个黏性时间处的流动仍在启动瞬态，
壁面剪切读数 `|1 - τ_SPH/τ_exact|` 因此是 **32.6 %**，工件自身 verdict = FAIL。

独立复核（`docs/audit_couette_wall_shear_reading.md`, m16438）已证明该亏缺来自瞬态而非壁面模型，
但**注册表从未吸收这条结论**：它至今把这 32.6 % 当作该方案的「最差证据」，
于是选择器对壁面有界算例只能给出 WITHHELD，而实际上同一求解器在设计时程下达标。

## 2 做法

1. **补测设计时程**（同求解器/同格子/同估计器，只把 `t_end` 放回 DESIGN.md 的值）：
   `benchmarks/couette/out_design/results.json`（`run.py --only ladder --resolutions 8 16`）。
2. **给证据记录加前提字段**：`Evidence.status ∈ {banked, refuted}` 与
   `Evidence.precondition`（`src/tensorsph/method_selection.py`）。
   * `refuted` 记录**保留在注册表里**（删掉证据正是坏读数被遗忘的方式），
     但**不得进入精度声明**；
   * `Decision.worst_measured_pct` 只在 `banked` 记录上取界，
     `worst_banked_pct` 仍对所有记录取最差（诚实对照）；
   * 若某方案的全部记录都被 refute，判 WITHHELD（无活证据，不给默认值）。
3. **重建注册表**（`benchmarks/method_selection/build_registry.py`）：`wcsph_wall_no_slip`
   增加 3 条设计时程记录（内区 L2 / 质量流 / 壁面剪切），原 `--quick` 三条标 `refuted`
   并写明 refute 理由与指向审计文档的指针。数量审计脚本
   `audit_quantity_coverage.py` 同步新增三条镜像量。
4. **粘性时间收敛扫描**（`benchmarks/couette/viscous_time_convergence.py`）：同样的读数在
   `t_end/(H²/ν) ∈ {0.6, 1.2, 2.4, 3.6}` 上重测，用来把「读数是否随跑长移动」写成可复算的量，
   而不是靠断言。该扫描已作为集群作业 `cj2857` 递交（队列中，未在本单终态）。

## 3 结果（数字均由工件/注册表程序化读出）

| 量 | `--quick`（0.3 H²/ν） | 设计时程（1.2 H²/ν，H/dx=16） | 门 | 15 % 标准 |
|---|---|---|---|---|
| 壁面剪切 `\|1-τ_SPH/τ_exact\|` | 32.65 %（已 refuted） | **0.48 %** | 5 % | 是 |
| 内区 L2 速度误差 | 5.65 %（已 refuted） | **2.93 %** | 5 % | 是 |
| 质量流 `\|1-Q_SPH/Q_exact\|` | 7.16 %（已 refuted） | **4.42 %** | 5 % | 是 |
| H/dx=8 壁面剪切 | 32.65 % | **4.34 %** | 5 % | 是 |

注册表规模 17 → 20 条；`methods_with_evidence` 不变（10），`missing_sources=[]`，
`unexpected_routes=[]`；测试 22 → 26 passed；`build_registry.py --check` 绿。

## 4 注意点 / 局限（不粉饰）

1. **设计时程运行自身的 `steady_drift` 门（1 %）仍未过**（H/dx=16 为 1.55e-2）。
   因此这次登记的正确解释是「设计时程读数」，不是「已证明定常」；
   收敛扫描（`viscous_time_convergence.py`，作业 `cj2857`）就是为了给出该读数的收敛包络，
   本单结束时该作业仍在排队，**故本单不主张稳态收敛结论**。
2. 覆盖审计脚本对新增量 `x_couette_design` 自报一条 `integrity_failures`：
   脚本内核的 `r_couette_l2` 与工件里被判门的内区 L2 定义不同（行 RMS vs 门控量），
   属于审计脚本自身的量定义差异，本单未改其定义，如实保留。
3. `--quick` 记录被 refute 而非删除，任何人仍可在注册表里看到 32.6 % 及原因；
   refute 的判据是「同格子、只改终端时间即改变 28 个百分点」这一可复算事实，不是意见。
4. 适用范围：单方案（`wcsph_wall_no_slip`）、单算例族（平面 Couette，Re=1）。
   refute 机制是通用的，但其它算例族尚未逐条复查前提字段。

## 5 复现

```bash
PYTHONPATH=src python benchmarks/couette/run.py --only ladder --resolutions 8 16 \
    --out benchmarks/couette/out_design --threads 4        # ~115 s
PYTHONPATH=src python benchmarks/method_selection/build_registry.py        # 重建注册表
PYTHONPATH=src python benchmarks/method_selection/build_registry.py --check
PYTHONPATH=src python benchmarks/method_selection/audit_quantity_coverage.py
PYTHONPATH=src python -m pytest tests/test_method_selection.py -q         # 26 passed
```
