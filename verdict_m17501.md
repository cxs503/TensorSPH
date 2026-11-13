# 交差审计报告：m17501 XVAL-SPH-METHOD-SELECTION-2026-11-12

**审计任务**：m17589（board-audit）
**审计对象**：m17501（XVAL-SPH-METHOD-SELECTION-2026-11-12，TensorSPH，状态 succeeded）
**审计日期**：2026-11-13

## 裁定结果：L2（独立功能实现）

### 三件套核验

| 验收项 | 声称 | 实际 | 判定 |
|---|---|---|---|
| run_artifacts | 7 个配置记录 | 仅 1 个存在（wf_D13_settle_d010.json, 175KB），其余 6 个缺失 | ⚠️ 部分存在 |
| benchmark ref vs got | 报告称"落地 0 个" | 实际 1 个落地但数值失稳（误差 97.4% >> 15%） | ❌ 未通过 |
| repro_cmd | grade_matrix.py 可执行 | 可执行但产生错误结果（报告 JSON 过期） | ⚠️ 可执行但结果不可信 |

### 关键发现

1. **报告 JSON 过期**：`docs/reports/xval_method_selection_m17501.json` 声称全部 7 条记录 `present: false`，但 `runs/local/wf_D13_settle_d010.json`（175KB）实际存在。报告是在运行完成前生成的。

2. **唯一落地算例数值失稳**：
   - 最坏波高误差：-97.44%（容差 15%）
   - max|v|/c0 = 4.479（应 <1.0）
   - 最大密度压缩：15.3%（应 <10%）
   - 逃逸粒子：82（应为 0）

3. **grade_matrix.py 设计缺陷**：仅检查 `record.get("failed")` 标记，不检查稳定性判据。数值失稳算例被标记为 `failed: false`。

4. **全部交付物未入库**：`git status` 显示所有新增文件均为 `??`（untracked），从未 commit。

5. **测试部分失败**：`test_report_artifact_matches_a_fresh_regrade` FAIL（报告过期导致断言失败）。

### 成熟度分级理由

- ✅ L0 文档成果：协议、报告、OSS 对照表齐全
- ✅ L1 数学/单元验证：合成信号测试通过
- ✅ L2 独立功能实现：grade_matrix.py 是完整输入→计算→输出管线
- ❌ L3 系统集成：未接入主求解器（standalone benchmark driver）
- ❌ L4 基准验证：7 算例仅 1 落地且数值失稳，误差 97.4% >> 15%
- ❌ L5 科研工程成果：基准验证未完成

### 落表

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorSPH', 'm17501 XVAL-SPH-METHOD-SELECTION-2026-11-12', 17589, 2, '<JSON>', 'board-audit');
```

已写入 tensor.db (id=20) 和 tensor_platform.db (id=5)。

### 处置建议

**reopened**（返修）：
1. 重新运行 grade_matrix.py 生成正确的报告 JSON
2. 修复 grade_matrix.py 的稳定性检查逻辑
3. 将全部交付物 commit 入库
4. 修复 test_report_artifact_matches_a_fresh_regrade 测试
5. 补跑缺失的 6 个算例（或诚实声明为 honest-FAIL）
