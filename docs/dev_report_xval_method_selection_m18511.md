# XVAL-SPH-METHOD-SELECTION-2026-11-12 · 能力扩展：压缩正则化与传输速度修正

锚点：`XVAL-SPH-METHOD-SPH-METHOD-SELECTION-2026-11-12`（TensorSPH，CASE-WAVE-FLUME 的方法选型）。

## 做了什么

### 1. 波槽求解器新增 SPHinXsys 风格稳定化机制的 CLI 支持

**文件**：`benchmarks/wave_flume/run.py`

在波槽运行脚本中新增以下 CLI 参数，使求解器可以使用 SPHinXsys 风格的密度正则化和传输速度修正：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--density-regularization` | `none` | 密度正则化方式：`none`（δ-SPH）或 `compression`（SPHinXsys 压缩正则化） |
| `--compression-rule` | `free_surface` | 压缩正则化规则：`internal`/`free_surface`/`free_stream` |
| `--transport-velocity` | `off` | 传输速度修正（Zhang, Hu & Adams 2017）：`off`/`on` |
| `--transport-velocity-strength` | `1.0` | 传输速度修正强度系数 |

**关键约束**：`density_regularization="compression"` 与 `density_diffusion > 0` 互斥（求解器底层已强制，CLI 层也做了前置检查）。

### 2. 方法选型矩阵新增压缩正则化探针格点

**文件**：`benchmarks/xval_method_selection/run_cell.py`

新增 Cell 3（`D20_h025_comp`）：使用 SPHinXsys 压缩正则化替代 δ-SPH，测试是否能解决原方法 A 的数值发散问题。

| index | label | 问题 | 协议参考 |
|---|---|---|---|
| 0 | D20_h025_A | 方法 A 在足够长的记录下能否通过 15% 门？ | H1, G1 |
| 1 | D13_h025_A | 粗分辨率端（H/dx = 2.08）是否因分辨率导致失败？ | H1, ladder |
| 2 | D20_h025_xsph | XSPH 速度平滑能否消除拉伸/团聚失稳？ | H3（机制探针） |
| **3** | **D20_h025_comp** | **SPHinXsys 压缩正则化能否修复 δ-SPH 无法解决的发散？** | **H4（机制探针）** |

### 3. 矩阵评级驱动更新

**文件**：`benchmarks/xval_method_selection/grade_matrix.py`

MATRIX 新增 `D20_h025_comp` 行，指向 `runs/xval/cell_3.json`。

## 集群作业

| 作业号 | 格点 | 状态 |
|---|---|---|
| cj1948 | cell_0 (D20_h025_A) | 排队中（212 ahead） |
| cj1949 | cell_1 (D13_h025_A) | 排队中（213 ahead） |
| cj1950 | cell_2 (D20_h025_xsph) | 排队中（214 ahead） |
| **cj1953** | **cell_3 (D20_h025_comp)** | **排队中（211 ahead）** |

## 测试

```
tests/test_xval_method_selection_matrix.py::test_matrix_reports_the_known_height_offset_and_the_gap PASSED
tests/test_xval_method_selection_matrix.py::test_report_artifact_matches_a_fresh_regrade PASSED
tests/test_xval_method_selection_matrix.py::test_a_diverged_record_can_never_pass_the_gate PASSED
tests/test_xval_method_selection_matrix.py::test_compression_alone_blocks_the_pass PASSED
tests/test_xval_method_selection_matrix.py::test_a_record_too_short_to_grade_is_a_gap_not_a_zero PASSED
5 passed, 1 warning
```

## 物理背景与动机

原方法 A（WCSPH + δ-SPH 密度扩散 δ=0.1）在波槽算例中数值发散：
- 存档记录 `wf_D20_unstable.json`：max|v|/c0 = 3.55（门 1.0），压缩 29.9%（门 10%），逃逸 144 粒子
- 振幅包络 e 倍时间 0.35–0.37 s，Mach 1 时刻 t = 3.25 s

SPHinXsys 使用三种不同的稳定化机制：
1. **压缩正则化**（`CompressionSummation` + `AverageCompression` + `DensityRegularization`）：每步用压缩量重置密度，而非 δ-SPH 的梯度扩散
2. **Riemann 耗散**（声学 Riemann 解 + Mach 限幅器）：激波捕捉工具，规则波场景量级 ≤M²≈1e-3
3. **传输速度修正**（Transport-velocity correction）：限幅粒子移位，维持粒子规则性

本单实现的是 #1 和 #3 的 CLI 暴露，使波槽算例可以直接对比 δ-SPH 与压缩正则化的效果。

## 下一步

- 等待集群作业 cj1948–cj1950, cj1953 完成
- 用 `grade_matrix.py` 重生成发布工件
- 对比 cell_0（δ-SPH）与 cell_3（压缩正则化）的稳定性诊断量
- 若压缩正则化通过稳定性门，则波高精度可对标活塞一阶传递函数（解析基准）

## 误差判定

本单是**能力扩展**（新增 CLI 参数与格点），不产生新的波高测量值。误差判定待集群作业完成后由 `grade_matrix.py` 自动给出。
