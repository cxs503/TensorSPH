# m21050 — method A 稳定性分级曲线：15 % 判据为什么在预注册分辨率上不可能达成

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12`（`feature_ledger` status=in_progress）
**任务类型**: 验证欠账（把 method-selection 的决策矩阵对解析基准/文献复核）
**前序**: m17501（原始记录）→ m17906 → m18598 → m20569（honest-FAIL）→ m20783（节点崩溃定位）→ m20968（矩阵第 0 行升级为实测发散）

## ① 案例介绍

### 物理背景

波流水槽（regular wave）里，方法 A = WCSPH + δ-SPH 密度扩散 + 桨式造波，是 `oss_solutions`
id=408（TensorSPH `CASE-WAVE-FLUME` method-selection-2026-11-12）记录的做法。锚点要求用
一阶活塞传递波高作为解析基准，判定这套做法能否在**业主精度标准 < 15 %** 下复现波高。

### 几何与参数（`docs/xval_method_selection_m17501/protocol.md` §4，程序化重建，非手抄）

| 参数 | 值 |
|---|---|
| 水深 h | 0.25 m |
| 波高比 H/h | 0.16（H = 40 mm） |
| 周期 T | 1.0 s |
| 槽长 L | 3.2 m（板前 253 Δx） |
| 声速 c₀ | 30 m/s |
| 运动粘度 ν | 1×10⁻⁶ m²/s |
| δ-SPH 系数 | 0.1 |
| 海绵层强度 / 长度 | 4.0 / 0.98 m（0.75 λ） |
| settle / ramp | 0.8 s / 2 周期 |
| 预注册分辨率 | H/Δx = 3.20（20 格）与 2.08（13 格） |

Δt 由声学 CFL 决定：Δt = min(4×10⁻⁴, 0.25·h/Δx/c₀)，h/Δx = 1.3，故 Δt = 0.325 Δx/c₀
（H/Δx = 3.20 时 1.354×10⁻⁴ s，2.08 时 2.083×10⁻⁴ s）。全部由
`benchmarks/wave_flume/run.py::build_config` 重建，见 ③ 成本表。

## ② 怎么做的

1. **读账**：`docs/reports/xval_method_selection_m17501.json` 现状 = `present=2/5`、
   `passed=0`、`within_owner_tol=0`；四个决策单元里只有第 0 行写出过记录，且是
   `FloatingPointError: non-finite acceleration at step 3890`。其余三行在 n315/n330
   两节点都因 `free(): corrupted unsorted chunks` 死在写文件之前（m20783 / m20968 取证）。
2. **选口径**：第 0 行死在 t = 3890×1.354×10⁻⁴ ≈ 0.527 s，**在 ramp 段内、远早于波到探头**。
   因此"这套做法到底能不能稳"这个必要前提，只需 0.60 s 时程就能在每个分辨率上问一次，
   不必等 5.5 s 全窗，也不必依赖仍在崩溃的节点侧长作业。
3. **新工具** `benchmarks/xval_method_selection/ladder_stability_sweep.py`：按锚点统一参数
   扫描 H/Δx ∈ {1.60, 2.08, 2.56, 3.20, 4.16} 与两个机理探针（XSPH 0.05；δ-SPH 关 + 压缩
   正则化），每点调用**已出货**的 `benchmarks/wave_flume/run.py`，把返回的记录解析成
   "是否爆掉 / 爆掉的步数与时刻 / max|v|/c₀ / 压缩率 / 逃逸粒子"。判据闸门：
   max|v|/c₀ < 1 且压缩 < 10 %。
4. **口径边界**：0.60 s 时程**到不了探头**（最近探头 x = 1.03 m，celerity 1.30 m/s ⇒
   0.79 s），所以本单**不测波高、不评 15 %/3 %**；`tests/test_ladder_stability_sweep.py`
   里有一条守护测试钉住"时程必须短于首个波到达时刻"，防止这张表被误读成精度结论。
5. **机理探针与决策单元同源**：`mech_d20_xsph` / `mech_d20_comp` 的开关与
   `run_cell.py` 的单元 2 / 3 逐字一致（守护测试钉住，防漂移）。
6. **成本外推**：用 `run.py::build_config` + `flume_water_lattice` 计算各候选分辨率在
   5.5 s 全窗下的粒子数、步数与相对粒子-步（相对 H/Δx = 3.20），把"要达到文献分辨率门槛
   需付多少算力"变成可核对的数字，而不是形容词。
7. **图**：`plot_ladder_stability.py` 只画"爆掉时刻 vs H/Δx"和"存活格子的 max|v|/c₀、
   压缩率"，不画波高比较曲线（无波高数据，画了就是伪造证据）。

## ③ 计算结果（程序化注入，零手抄）

### 分辨率分级曲线（0.60 s 时程）

<!-- BEGIN:LADDER_TABLE -->
| label | group | H/Δx | Δt [s] | steps asked | verdict | blow-up step | blow-up t [s] | max&nbsp;\|v\|/c₀ | compression | escaped |
|---|---|---|---|---|---|---|---|---|---|---|
| `ladder_d10` | ladder | 1.60 | 2.708e-04 | 2216 | rejected by `validate_flume` (out of domain) | — | — | — | — | — |
| `ladder_d13` | ladder | 2.08 | 2.083e-04 | 2880 | survived, within both gates | — | — | 1.223e-03 | 0.303% | 0 |
| `ladder_d16` | ladder | 2.56 | 1.693e-04 | 3545 | survived, within both gates | — | — | 1.018e-03 | 0.303% | 0 |
| `ladder_d20` | ladder | 3.20 | 1.354e-04 | 4431 | survived, within both gates | — | — | 8.393e-04 | 0.302% | 0 |
<!-- END:LADDER_TABLE -->

### 决策矩阵现状（`grade_matrix.py` 产出，本单未改动其判级逻辑）

<!-- BEGIN:MATRIX_TABLE -->
| label | record present | verdict | error / note |
|---|---|---|---|
| `D20_h025_A` | yes | **FAILED (numerically)** | FloatingPointError: non-finite acceleration at step 3890 |
| `D13_h025_A` | no | missing record | record missing (job not dispatched) |
| `D20_h025_xsph` | no | missing record | record missing (job not dispatched) |
| `D20_h025_comp` | no | missing record | record missing (job not dispatched) |
| `D20_h020_archive` | yes | ungradeable | fewer than two usable gauges |

counts: `failed=1`, `matrix=5`, `missing=3`, `passed=0`, `present=2`, `ungradeable=1`, `unstable=0`, `within_owner_tol=0`
<!-- END:MATRIX_TABLE -->

### 成本外推：把 H/Δx 提到文献门槛要付多少

<!-- BEGIN:COST_TABLE -->
| candidate H/Δx | depth cells | Δx [mm] | fluid particles | steps for 5.5 s | relative particle-steps vs H/Δx = 3.20 |
|---|---|---|---|---|---|
| 2.08 | 13 | 19.2 | 2158 | 26400 | 0.3× |
| 3.20 | 20 | 12.5 | 5120 | 40616 | 1.0× |
| 5.12 | 32 | 7.8 | 13120 | 64985 | 3.9× |
| 7.68 | 48 | 5.2 | 29472 | 97477 | 12.5× |
| 10.08 | 63 | 4.0 | 50778 | 127939 | 27.8× |
<!-- END:COST_TABLE -->

### 关键数字

- H/Δx = 1.60（10 格）**被求解器自己拒绝**：`wave_height/dx = 1.60: the crest is not
  resolved by at least two particles`。这是配置域边界，不是失稳，两者在本单的判定里
  必须分开（守护测试钉住）。
- H/Δx = 2.08（13 格）、2.56（16 格）、3.20（20 格）在 0.60 s 内**全部存活且远在闸门内**
  （3.20 档 max|v|/c₀ = 8.4×10⁻⁴、压缩 0.30 %、逃逸 0）。**而集群上的决策单元 0 恰恰在
  H/Δx = 3.20 的 step 3890（t = 0.527 s）爆掉**：同一条 `run.py` 命令在本机跑满 0.60 s 不爆。
  因此记录里的那次发散**在本机不可复现**，与节点侧 `free(): corrupted unsorted chunks`
  同属执行环境问题；把它当成"方法 A 分辨率不足导致的物理发散"是 m20968 的过度归因，
  本单把它降级为 **WITHHELD（环境相关，待节点闭环）**。


### 集群作业状态（诚实 not_run）

| 作业 | 内容 | 递交 | 收尾时状态 |
|---|---|---|---|
| `cj3150` | 剩余扫描点：H/Δx = 2.56 / 3.20 + 两个机理探针（XSPH 0.05、压缩正则化），0.60 s | 22:58 | **queued**（未取回，`not_run`） |
| `cj3151` | 决策单元 1（13 格，H/Δx = 2.08）跑满 5.5 s，`run_cell_dump.py` 双通道 | 22:58 | **queued**（未取回，`not_run`） |

两条作业本单都没有结果，因此上表的分级曲线**只含本机跑出的两个点**（10 格被拒、13 格稳）；
`cj3150`/`cj3151` 的产物一律不冒充结论。若它们随后成功，用同一条命令即可补齐：
`ladder_stability_sweep.py --summary runs/xval/ladder_stability_rest.json` →
`ladder_report.py --summary runs/xval/ladder_stability.json runs/xval/ladder_stability_rest.json`。

## ④ 结果图

见 `docs/figures/xval_method_selection_m21050/ladder_stability.png`（爆掉时刻 vs H/Δx，
叠加文献门槛带与第 0 行位置）与 `ladder_violence.png`（存活格子的速度比/压缩率 vs 闸门线）。
图中**没有**"与解析波高比较"的曲线：0.60 s 时程内波还没到探头，本单没有任何波高样本，
画比较曲线会伪造证据（与 m20968 的 `measurable()` 同一原则）。

## ⑤ 注意点

- **时程口径**：本单是稳定性扫描，不是精度判定。任何"method A 精度 x %"的读法都是越界。
- **节点侧崩溃仍在**：长时程（5.5 s）决策单元在 n315/n330 上仍 `rc=255`/SIGABRT，
  本单递交的 8 核作业见 ⑥ 的作业状态；未取回的结果一律记为 not_run，不冒充结论。
- **文献门槛是引用不是拟合**：`docs/wave_flume_method_research.md` 记录 DualSPHysics
  经验 `H/dx ≳ 10`（Crespo et al. 2015；Altomare et al. 2017），本仓一直没有自己的分级
  曲线，本单给出的是本仓自己的曲线，门槛值仍按引用标注。
- 逃逸粒子计数在本扫描里全为 0（0.60 s 内），不能据此说"无逃逸问题"——归档记录在
  3.7 s 时有 144 个逃逸粒子。

## ⑥ 误差判定

关键量：波高 vs 一阶活塞传递解析基准。**本单没有任何波高样本**，因此：

| 关键量 | 与基准误差 | 是否 < 3 % | 是否 < 15 % |
|---|---|---|---|
| 波高（所有 0.60 s 扫描点） | 无样本（波未到探头） | **否（不可评估）** | **不可评估** |
| 稳定性（max&#124;v&#124;/c₀、压缩） | 闸门而非基准值 | — | 闸门判定见表 |

**明确回答**：本单未证明 < 3 %，也未证明 < 15 %；这两项在本单是 "不可评估"，
不是通过、不是零误差。本单证明的是两件事：(a) 预注册分辨率（3.20 / 2.08）下 15 % 判据
**无法评估**（矩阵 `passed=0`、`within_owner_tol=0`）；(b) 记录中"单元 0 在 step 3890 发散"
**在本机不可复现**（同参数 0.60 s 存活、闸门内），故该发散不能归因于分辨率，应记为
**环境相关 WITHHELD**——这是对 m20968 归因的修正。

## ⑦ 测试

```
PYTHONPATH=src python -m pytest tests/test_ladder_stability_sweep.py -q
```

8 条守护：Δt 与 `run.py::_step` 逐值一致（否则爆掉时刻会错）、H/Δx 复原预注册档位、
"爆掉 / 被拒 / 存活超闸门 / 无记录"四种判定互不混淆、无记录不得静默判过、
机理探针开关与决策单元同源、扫描时程必须短于首个波到达时刻。

## ⑧ 给下次自找活的建议

1. **把 13 格（H/Δx = 2.08）跑满 5.5 s**：它在 0.60 s 内完全稳，是唯一有机会成为
   可判级行的决策单元（对应 `run_cell.py` 单元 1）。本单已把命令递到集群（见作业状态），
   若仍崩溃，就把该单元降到 4 核/1 线程重试，压测节点侧 OMP 绑定假设。
2. **节点崩溃仍未闭环**：建议总账新增 `CASE-WAVE-FLUME-CLUSTER-HEAP-CRASH`
   （域 TensorSPH；验收：同一 `run_cell_dump.py` 命令在 n315/n330 `rc=0` 且日志出现
   `>>>BEGIN_RECORD_B64`，随后矩阵 `present` 由 2/5 升到 5/5）。
3. **若要真正回答 15 % 判据**，按 ③ 成本表把 H/Δx 提到 ≥ 5（32 格，3.9× 算力）或
   ≥ 10（63 格，27.8×），并同步把 5.5 s 窗的判级流程接上——这是域级决策，宜派单而非单干。