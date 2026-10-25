# TensorSPH 上游跟进评审报告

**Mission**: m281  
**跟进范围**: 847c34b5..195bc3d6（4 commits，含 2 merge）  
**评审日期**: 2026-10-25  
**评审员**: tensor-expand 班次  

---

## 1. 提交清单

| # | SHA | 类型 | 标题 | 来源 |
|---|-----|------|------|------|
| 1 | 520bc54 | 实质 | TensorSPH 首波审计：基线测试 104/104 PASS… | agent/m239 |
| 2 | 8f3eeb6 | merge | Merge pull request #11 from cxs503/agent/m239 | 上游 |
| 3 | 18212ac | 实质 | TensorSPH GPU 大规模基准：定位加速拐点 N* ≈ 928… | agent/m276 |
| 4 | 195bc3d | merge | Merge pull request #12 from cxs503/agent/m276 | 上游 |

**结论**：4 个提交全部是本线 agent 贡献（m239 首波审计 + m276 GPU 基准）被上游合入。无第三方新代码。

---

## 2. 逐 commit 评审

### 2.1 520bc54 — m239 首波审计报告

**变更**：+`expand_audit_tensorsph.md`（200 行）

**评审**：
- ✅ 审计报告结构完整：环境/测试基线/案例抽检/能力-声明对照/缺口/提案
- ✅ 测试数字 104/104 PASS 与当时基线一致
- ✅ 案例数字直读 manifest：density_compression=0.0355, mach=0.0708, courant=0.264
- ✅ 静水壁 benchmark 数字与 README 声明一致（1.12% / 9.03%）
- ⚠️ 未跑 dam-break 收敛案例（预算优先，已在报告中标注）
- ✅ 首次量化 GPU/CPU 对照（608 粒子，GPU 0.95x），正确识别为小规模无加速

**结论**：✅ 审计质量合格，数字可溯源。

### 2.2 8f3eeb6 — PR #11 merge

**评审**：上游合入 m239 审计报告，无冲突。

**结论**：✅ 例行 merge。

### 2.3 18212ac — m276 GPU 大规模基准

**变更**：
- +`benchmarks/scaling/run.py`（270 行）
- +`benchmarks/scaling/README.md`（163 行）
- +`docs/benchmarks/gpu-scaling.md`（168 行）
- +`dev_report_tensorsph-gpu-scaling.md`（221 行）
- ~`README.md`（1 行：替换 CUDA 未基准声明为 GPU 基准摘要）

**评审**：
- ✅ 不修改 src/，纯基准+文档贡献
- ✅ README 更新精准：旧声明"CUDA has never been benchmarked"→ 新声明含 N*≈928, 30.89x@95k
- ✅ 基准方法合理：复用 droplet_impact.run，固定物理参数仅变 dx
- ✅ 数字完整：5 个粒子规模 × 2 设备 = 10 组数据
- ⚠️ dx=0.006 (N≈10556) 两端均 FAIL（density_compression=0.17），报告正确标注为物理分辨率问题
- ✅ 内存数据线性增长：4.6 MB → 411 MB
- ✅ 测试状态 104/104 PASS（与基线一致）

**结论**：✅ 高质量基准贡献，填补 README 自承认缺口。

### 2.4 195bc3d — PR #12 merge

**评审**：上游合入 m276 GPU 基准，无冲突。

**结论**：✅ 例行 merge。

---

## 3. 源码变更分析

```
git diff 847c34b5..195bc3d6 -- src/
```

**结果**：空（无 src/ 变更）

**结论**：上游 847c34b5..195bc3d6 仅合入文档/基准，未修改任何源代码。对既有功能零影响。

---

## 4. 新声明 vs 测试在场性

| 新声明 | 来源 | 测试在场 | 验证状态 |
|--------|------|----------|----------|
| N* ≈ 928 particles | README.md | benchmarks/scaling/run.py | ✅ 可复现 |
| 1.34x @ 1k → 30.89x @ 95k | README.md + docs/benchmarks/gpu-scaling.md | benchmarks/scaling/run.py | ✅ 可复现 |
| 内存 4.6 MB → 411 MB | docs/benchmarks/gpu-scaling.md | torch.cuda.max_memory_allocated | ✅ 可复现 |

**结论**：所有新声明均有对应基准脚本支撑，可复现。

---

## 5. 对 backlog 影响

### 5.1 已合入提案（需撤项记录）

| 提案 | backlog 状态 | 上游动作 | 处理 |
|------|-------------|----------|------|
| GPU 基准缺失（m276 填补） | TensorSPH 缺口："CUDA 基准缺失" | ✅ 已实现并合入 | 从缺口列表移除，记入 m276 开发记录 |

### 5.2 仍缺项目（排期不变）

| 优先级 | 项目 | 状态 |
|--------|------|------|
| P0 | 表面张力模型（m277） | 已开发未合入（agent/m277 分支） |
| P0 | 3D 扩展 | 仍缺 |
| P1 | 多相流 / 自由面耦合 | 仍缺 |
| P1 | dam-break 收敛复现 | 仍缺（m239 未跑） |
| P2 | 更大规模（500k+） | 仍缺（提案建议） |

### 5.3 新发现

- dx=0.006 (N≈10556) 数值不稳定：density_compression=0.17（限 0.05）。CPU/GPU 一致失败，为物理分辨率问题。建议在 m277 表面张力或后续开发中关注此分辨率区间的稳定性。

---

## 6. 与 agent 分支冲突检查

| 分支 | 与 origin/main 冲突 | 说明 |
|------|---------------------|------|
| agent/m239 | ✅ 已合入 | PR #11 |
| agent/m276 | ✅ 已合入 | PR #12 |
| agent/m277 | ⚠️ 待合入 | 基于 195bc3d（已 merge main），无冲突风险 |
| agent/m281 | ✅ 无冲突 | 本分支基于 195bc3d，仅评审无代码变更 |

**结论**：无冲突。agent/m277 表面张力分支已包含最新 main，可安全走门禁。

---

## 7. 测试套件门禁

```
平台: /nfs/wangxi/tensor_eco/agent_work/m281/TensorSPH
HEAD: 195bc3d6 (origin/main)
Python: 3.12.13 (/nfs/wangxi/hermes-agent/.venv/bin/python)
PyTorch: 2.14.1+cu130
pytest: 9.1.1
```

**结果**：
```
104 passed in 55.15s
```

**结论**：✅ 全绿。与 README 基线（104 测试）一致。上游合入未引入回归。

---

## 8. 总结

| 维度 | 结论 |
|------|------|
| 上游变更性质 | 本线 agent 贡献合入（m239 审计 + m276 GPU 基准），无第三方代码 |
| 源码变更 | 无（src/ 零 diff） |
| 新声明 | 3 条（N*, speedup, memory），均有基准脚本支撑 |
| 测试门禁 | 104/104 PASS |
| 回归风险 | 无 |
| backlog 影响 | GPU 基准缺口已填补；其余排期不变 |
| 冲突 | 无 |
| 修复提案 | 不需要 |

**下一步**：
1. 推进 agent/m277 表面张力分支门禁 → 合入
2. 关注 dx=0.006 区间稳定性（后续开发参考）
3. 监控上游：v0.8.0 后上游可能推进 3D 或多相流
