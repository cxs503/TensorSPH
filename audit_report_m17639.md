# Audit Report: m17639 XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION

**Auditor:** board-audit (m17667)  
**Date:** 2026-11-13  
**Mission:** m17639 — Port SPHinXsys transport-velocity correction to TensorSPH  
**Report ID:** 16603  
**Status:** succeeded  

---

## ① 取证：三件套核验

### run_artifacts ✅ 存在
- `artifacts/xval_sphinx_tvc/grid_dx003.json` — dx=0.03 全变体结果（含 baseline, tvc_a005_all, tvc_a010_all, tvc_a005_bulk）
- `artifacts/xval_sphinx_tvc/grid_dx004.json` — dx=0.04 全变体结果（含 baseline, tvc_a005_all, tvc_a010_all, tvc_a020_all, tvc_a005_bulk, tvc_a005_all_c5, pst_lind）
- `artifacts/xval_sphinx_tvc/figure_xval_tvc.png` — 99KB 对比图

### benchmark ref vs got 数值 ✅ 可核验
参考数据：Martin & Moyce (1952) 数字化 Z/a(τ)，来源 Koshizuka & Oka (1996) 开放获取综述，附带 PROVENANCE.json 和 12 个上游源文件 sha256。

**报告声称 vs 工件实测（逐项核对）：**

| 变体 | 报告 MAE | 工件 MAE | 报告 rel% | 工件 rel% | 匹配？ |
|---|---|---|---|---|---|
| dx=0.04 baseline | 0.591 | 0.5909 | 46.7% | 46.74% | ✅ |
| dx=0.04 tvc_a010_all | 0.147 | 0.1468 | 17.0% | 16.96% | ✅ |
| dx=0.04 tvc_a020_all | 0.563 | 0.5630 | 51.3% | 51.27% | ✅ |
| dx=0.03 baseline | 0.621 | 0.6214 | 51.0% | 50.98% | ✅ |
| dx=0.03 tvc_a010_all | 0.143 | 0.1425 | 10.8% | 10.81% | ✅ |

所有数字匹配（舍入误差内）。

### repro_cmd ✅ 可执行
`benchmarks/xval_sphinx_tvc/run.py` 存在，接受 `--resolutions`, `--seconds`, `--variants`, `--out` 参数，输出 JSON 工件。

---

## ② 基准判据核验

门限值：`owner_rel = 0.15`（15% 相对误差），`ledger_abs = 0.15`（绝对 MAE）。

| 变体 | MAE | rel% | abs gate | rel gate |
|---|---|---|---|---|
| dx=0.04 tvc_a010_all | 0.1468 | 16.96% | ✅ PASS | ❌ FAIL |
| dx=0.03 tvc_a010_all | 0.1425 | 10.81% | ✅ PASS | ✅ PASS |

**结论：** dx=0.03 时 TVC α=0.10 同时通过两道门；dx=0.04 时仅通过绝对门。误差随加密减小（17.0% → 10.8%），趋势正确但仅 2 个分辨率层级，不足以声明收敛阶。

---

## ③ 报告六段完整性

| 要素 | 状态 | 说明 |
|---|---|---|
| ① 案例介绍 | ✅ | 溃坝物理背景、Martin & Moyce 1952 参考、几何参数表 |
| ② 方法 | ✅ | 从 SPHinXsys 逐字移植、9 项转录测试、符号校准过程 |
| ③ 计算结果 | ✅ | 全变体 MAE/rel% 表、程序化注入零手抄 |
| ④ 结果图 | ✅ | figure_xval_tvc.png（99KB），左：Z/a vs τ 曲线；右：误差柱状图 |
| ⑤ 注意点 | ✅ | 符号错误导致崩溃的首次失败记录、α=0.2 不可移植、BulkParticles 作用域不适用、知识引擎离线等摩擦 |
| ⑥ 误差判定 | ✅ | 逐变体 gate_abs_pass/gate_rel_pass 显式列出 |

---

## ④ 代码与测试核验

### 实现文件
- `src/tensorsph/transport_velocity_correction.py` — ~200 行，包含 `kernel_gradient_integral()` 和 `transport_velocity_correction()` 两个核心函数
- `src/tensorsph/dam_break.py` — 集成 TVC 配置项（tvc_every, tvc_alpha, tvc_scope, tvc_max_shift）

### 测试（9/9 通过，已复现）
```
test_default_coefficients_match_upstream_header — PASS
test_uniform_lattice_is_a_fixed_point — PASS
test_matches_literal_upstream_transcription — PASS
test_surface_correction_points_out_of_the_fluid — PASS
test_wall_contact_uses_upstream_factor_two — PASS
test_shift_law_and_scope_mask — PASS
test_net_translation_of_a_symmetric_blob_is_zero — PASS
test_disabled_correction_leaves_the_solver_bit_identical — PASS
test_correction_keeps_the_column_bounded_and_massive — PASS
```

### 上游源文件溯源
`docs/xval_sphinx_tvc/oss_source/` 含 12 个 SPHinXsys 源文件（含 sha256），commit `09e55cfdd23104e1b1db006a209a906218b3b14e`，通过 GitHub contents API 获取。

---

## ⑤ 成熟度判级

| 级别 | 条件 | 满足？ |
|---|---|---|
| L0 文档成果 | 有文档 | ✅ |
| L1 数学/单元验证 | 9 项单元测试通过 | ✅ |
| L2 独立功能实现 | 真代码：输入→计算→输出 | ✅ |
| L3 系统集成 | 接入 dam_break 求解器 | ✅ |
| L4 基准验证 | 完整算例+误差分析+可复现 | ✅（有条件） |
| L5 科研工程成果 | 多分辨率收敛+上游二进制对比 | ❌ |

**判定：L4（有条件）**

条件/缺口：
1. **无上游二进制对比**：镜像无 Boost/TBB，无法编译运行 SPHinXsys 原版，仅做到算子级精确转录
2. **仅 2 层分辨率**：dx=0.04 和 0.03，4 层收敛阶梯（cj1623/cj1624）在报告时未运行
3. **平台无 tensorbench_results 行**：L4 机器裁定为 CLAIMED_NOT_PROVEN（自声明非平台复算）

---

## ⑥ adversarial review 发现

### 严重性排序

**🟡 MEDIUM — L4 平台裁定冲突**
- 平台 L4 verdict API 返回 `CLAIMED_NOT_PROVEN`，原因是无 `tensorbench_results` 行
- 但实际工件完整（JSON 结果 + 图 + 可复现脚本），缺口是平台侧未录入
- 建议：补录 benchmark 结果到 tensorbench_results 表

**🟡 MEDIUM — 收敛声明不足**
- 报告称"error decreases with refinement"但仅 2 层，不能声明收敛阶
- 报告自身已诚实标注"trend, not demonstrated convergence order"
- 不影响 L4 判定（L4 不要求收敛阶证明），但 L5 需要

**🟢 LOW — 相对误差门在 dx=0.04 未通过**
- tvc_a010_all at dx=0.04: rel=16.96% > 15% gate
- 但 dx=0.03 时 rel=10.81% 通过，且趋势正确
- 可接受：加密后通过是合理行为

**🟢 LOW — 图文件 png_base64 为空**
- 报告 figures[0].png_base64 为空字符串
- 但 PNG 文件实际存在（99KB），可能是上传时未内嵌
- 不影响实质验证

---

## ⑦ 最终裁定

**Verdict: L4（有条件通过）**

m17639 交付物实质完整：
- 算子从 SPHinXsys 逐字移植，9 项测试覆盖转录正确性、符号、边界、对称性
- 基准对比 Martin & Moyce (1952)，dx=0.03 时 α=0.10 通过双门（MAE=0.143, rel=10.8%）
- 可复现脚本 + 完整 JSON 工件 + 对比图
- 诚实记录首次符号错误失败、α=0.2 不可移植、BulkParticles 不适用

缺口不影响 L4 核心要求（完整算例+误差分析+可复现），但阻止 L5。

**Disposition: confirmed（消账）** — 建议补录 tensorbench_results 行以消除平台 L4 CLAIMED_NOT_PROVEN 状态。
