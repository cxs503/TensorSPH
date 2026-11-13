# 调研②③：方法选型与迁移方案（anchor `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`，m17584）

## 1. 选型决策（对照调研①的三条路线）

| 环节 | 现状（m16352/m16454） | 本轮选择 | 理由（可复算） |
|---|---|---|---|
| 判据锚 | 闸门顶点时程 vs Antoci 2007 实验（ref WITHHELD ⇒ 判据不可裁） | **改成"力"尺度判据**：壁面合力 vs `½ρg h_s²`；杆大变形 vs Kirchhoff elastica | 两个参考都能独立复算（解析 / 自解 ODE），不依赖不可得文献；且"力"正是本锚点 `FORCE-FROM-FLUID` 的所指 |
| 结构时间积分 | 显式辛欧拉，杆与流体同一步长 ⇒ 刚度上升即失稳（`ω_ax·dt=12.6` 被 `validate()` 拦下，端到端 `not_run`） | **结构子循环**：`rod_substeps_effective()` 由 `ω_ax·dt ≤ 0.25` 自动定 n_sub，`step()` 内对杆做 n_sub 个子步 | 交错分区 FSI 的常规做法；杆的自由度只有 25×2，子循环代价可忽略，而稳定性约束被彻底解除 |
| 静力平衡求解 | L-BFGS-B 能量下降 | **保留 L-BFGS-B + 停滞护栏**（`nit ≤ 1` 且载荷非零即抛错）+ 迭代日志 `_ROD_SOLVE_LOG` | 本轮曾把求解器换成 torch 阻尼 Newton（`dΠ/du=0` 求根），在本环境里①`create_graph=True` 缺失导致雅可比恒零（静默不动）、②scipy `least_squares` 与 torch autograd 交错触发 `free(): invalid pointer` 崩溃；预算内改为"保留已验证下降 + 拦住静默失效"，根治（预条件 Newton/弧长延拓）列入迁移路线 |
| 固体模型 | 离散弹性杆（能量 + AD 力） | **不改**（不引入 TLSPH 超弹连续体） | 杆链已有 EB/elastica 两条独立参照，且大变形同帧误差 +2.548 %；TLSPH 需重写离散与本构，收益不成比例 |
| 载荷传递 | 面元压力外推（m16352 引入） | **不改**（`load_transfer="panel"`，`load_scale=1.0`） | 本轮两分辨率 +0.240 %/+0.089 %，无系统性亏缺；成对和只作 before/after 基线 |
| 基准交付形态 | 自备驱动 + 手写 summary | **TensorBench `json_stdout` workload** | 判据数字由平台从原始 stdout 按点指针复算，ref/got 不可由调用方自报 |

## 2. 迁移方案（已落地 / 未落地）

已落地（本轮 commit 内）：
1. `ElasticGateCase.{rod_substeps, rod_stability_target, max_rod_substeps}` + `rod_axial_frequency()`
   + `rod_substeps_effective()`；`validate()` 改为子步感知（不再一刀切拒绝刚性门）。
2. `step()` 内杆子循环（流体载荷在一个流体步内冻结，杆推进 n_sub 个 dt_rod）。
3. `rod_static_equilibrium()` 停滞护栏 + `_ROD_SOLVE_LOG` 诊断记录。
4. `benchmarks/fsi2_gate_load/run_fsi2_benchmark.py`：三条判据（两条静水力 + 一条 elastica 大变形）
   的机器可裁基准 workload；`docs/fsi2_gate_load/{01,02,03}`。

未落地（写清原因，不粉饰）：
1. **端到端静态判据**（SPH 载荷 → `W^T` → 杆 → 平衡 vs 梁 Green 函数）：刚性门静态求解仍会停滞，
   现阶段不可用；根治需预条件 Newton/弧长延拓。
2. **耦合时程两分辨率收敛**：dx=12.5 mm 单点约 250 s，预算内未跑第二分辨率。
3. **Antoci 2007 实验曲线**：三次任务一致 WITHHELD，本单不再作为判据。
4. **SPHinXsys `test_2d_fsi2`（圆柱后弹性梁流致振动）**：若 owner 认定这才是"FSI2"真身，
   需不可压/低频长时积分面 + TLSPH 固体，属独立节点。

## 3. 复算入口

```bash
python benchmarks/fsi2_gate_load/run_fsi2_benchmark.py \
    --out artifacts/fsi2_gate_load/bench.json          # 三判据，约 16 s（本机 1 核）
python -m pytest tests/test_fsi_elastic_gate.py -q    # 模块回归（含本单新增子步断言）
```