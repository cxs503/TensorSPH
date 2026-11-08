# TensorSPH 验证账本维护报告（m6529）

**日期**: 2026-11-08  
**范围**: 验证断言账本 + 声明有效性审计的例行更新  
**类型**: 维护性增量（无 src/ 改动）

## 1. 做了什么

### 1.1 声明有效性审计更新（claim_validity）

`benchmarks/claim_validity/audit.json` 与 `docs/verification_claim_validity.md` 过期（stale），
运行 `benchmarks/claim_validity/validity.py` 重新生成：

- 审计条目数：31 claims（不变）
- 新增 3 条 run_validity finding：`young_laplace.closure.closure_on_interior_layers2.dx2mm_c20`、
  `dx2mm_c5`、`closure_on_interior_layers4.dx2mm_c20` — 这些运行的残余伪流超过 Adami 阈值，
  读数有效性被标记为 INVALID
- 测试 `tests/test_claim_validity.py`：14/14 PASS

### 1.2 验证断言账本更新（verification_ledger）

`benchmarks/verification_ledger/build_ledger.py` 重新生成：

- claims：36（不变）
- findings：8 → 11（新增 3 条 run_validity，与 claim_validity 同步）
- 新增 1 条 uncovered_benchmark finding → 已通过豁免修复

### 1.3 冰区附加质量基准豁免

`benchmarks/ice_floe_added_mass/` 有完整的验证文档
（`docs/verification_ice_floe_added_mass.md`，程序化注入），但未在账本中注册为 claim。
添加豁免（waiver）并附理由，消除 `uncovered_benchmark` finding。

- `benchmarks/verification_ledger/waivers.json`：新增 `benchmarks/ice_floe_added_mass/` 条目
- 测试 `tests/test_verification_ledger.py`：24/24 PASS（含 `test_every_waiver_has_a_reason`）

## 2. 验证结果

| 测试套件 | 通过 | 失败 | 状态 |
|---|---|---|---|
| `test_claim_validity.py` | 14 | 0 | ✅ |
| `test_verification_ledger.py` | 24 | 0 | ✅ |
| 合计 | 38 | 0 | ✅ |

## 3. 变更文件

- `benchmarks/claim_validity/audit.json` — 重新生成（3 条新 run_validity finding）
- `benchmarks/verification_ledger/ledger.json` — 重新生成（findings 8→11）
- `benchmarks/verification_ledger/waivers.json` — 新增 ice_floe_added_mass 豁免
- `docs/verification_claim_validity.md` — 更新（程序化注入）
- `docs/verification_ledger.md` — 更新（findings 分类变化）

## 4. 误差判定

本单为维护性增量，不涉及物理量测量。所有数字由脚本程序化注入，零手抄。

## 5. 后续建议

- 账本中仍有 9 条 FAIL 和 13 条 WITHHELD 声明，主要集中在表面张力（delta_s 归一缺口 25%、
  曲率误差 10-25%）和 dam break 实验对照（MAE 0.57 vs 限 0.15）
- P0 候选：CSF 力密度重整（同时修复 delta_s 归一与曲率估计），预期可将 6-8 条 WITHHELD
  升级为 MARGINAL 或 PASS
- 详见 `docs/verification_status_report.md` 的优先级排序
