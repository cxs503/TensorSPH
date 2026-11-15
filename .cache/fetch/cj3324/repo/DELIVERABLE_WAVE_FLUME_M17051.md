# CASE-WAVE-FLUME 交付总结

**Mission**: m17051  
**日期**: 2026-11-13  
**锚点**: `CASE-WAVE-FLUME`（算例池·规则波生成）  
**判据**: `|H_meas - H_target| / H_target ≤ 10%`（逐探头）

---

## 做了什么

### 1. 全面审查现有实现
- 审查 `src/tensorsph/wave_flume.py`（1208 行）：WCSPH 求解器 + 活塞造波 + 海绵层 + 波高计
- 审查 `src/tensorsph/wave_flume_grade.py`（441 行）：谐波拟合/穿越法/谱分析/衰减拟合/反射系数
- 审查 `benchmarks/wave_flume/` 三件套：run.py → analyse.py → make_report.py
- 审查 `tests/test_wave_flume*.py` 测试套件

### 2. 失败模式诊断
从历史报告 `docs/reports/wave_flume_report.json` 提取数据，诊断出原运行（wf_D16, H/dx=2.6）失败根因：
- **分辨率不足**：H/dx=2.6，波峰仅 2-3 粒子分辨
- **无启停瞬态处理**：settle_time=0, viscosity_settle=0, xsph_settle=0
- **无反射控制**：active_absorption=0.0
- **结果**：波高误差 65.5%，反射系数 82%，逃逸粒子 99

### 3. 改进方案制定
- 推荐参数：depth-cells=40-50（H/dx=6.4-8.0），settle_time=1.0s，active_absorption=0.5
- 创建 `scripts/run_wave_flume_campaign.sh`：三工况收敛性阶梯（D50/D30/D50_T08）
- 已递交集群作业 cj1401（排队中）

### 4. 验证计划文档
- 创建 `docs/wave_flume_verification_plan.md`：根因分析 + 改进方案 + 验证计划 + 收敛性阶梯

### 5. 测试状态确认
- `test_wave_flume_grade.py`：13/13 通过 ✅（评级模块正确）
- `test_wave_flume_case_artifacts.py`：3 失败（缺运行记录 + 历史数据不达标）

---

## 未完成部分（阻塞于计算资源）

### 阻塞原因
- 本机 load15 > 140（nproc=128），无法本地运行（每次模拟需 100-300s）
- 集群队列 cj1401 排队 544 位，预计等待 >2 小时
- 测试要求 steps_taken > 20000 且 simulated_time > 3.0s，无法缩减

### 待完成项
1. **运行模拟**：等待 cj1401 完成或重新递交
2. **生成报告**：运行 analyse.py + make_report.py
3. **验证测试**：`test_wave_flume_case_artifacts.py` 全部通过
4. **更新 feature_ledger**：status → verified

---

## 证据指针

| 工件 | 路径 |
|------|------|
| 验证计划 | `docs/wave_flume_verification_plan.md` |
| 运行脚本 | `scripts/run_wave_flume_campaign.sh` |
| 集群作业 | cj1401（TensorSPH 仓） |
| 评级测试 | `tests/test_wave_flume_grade.py`（13/13 pass） |
| 历史报告 | `docs/reports/wave_flume_report.json` |

---

## 给下次自找活的建议

1. **取回 cj1401 工件**：`run_cluster action=fetch job_id=cj1401`
2. **若 cj1401 失败**：调整参数（增加 active_absorption 至 0.7，或提高 resolution 至 depth-cells=60）
3. **若通过**：运行 `benchmarks/wave_flume/analyse.py` 和 `make_report.py`，更新报告
4. **长期**：将 wave_flume 纳入 CI nightly（计算量大，不宜每 PR 运行）
5. **扩展**：可考虑添加不规则波（JONSWAP 谱）作为后续 case-pool 条目

---

*本单因计算资源限制未能完成端到端验证，但已交付完整的失败诊断、改进方案和验证计划。集群作业 cj1401 正在排队，取回工件后可继续完成验证。*
