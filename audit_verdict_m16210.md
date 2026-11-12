# 交差审计报告：m16210 CASE-SPH-WEDGE-ENTRY-2D

**审计任务**: m16235 (board-audit)
**被审任务**: m16210 (CASE-SPH-WEDGE-ENTRY-2D·TensorSPH)
**审计日期**: 2026-11-12
**成熟度判定**: L4

---

## ① 取证结果

### 三件套核验
| 项目 | 状态 | 详情 |
|---|---|---|
| run_artifacts | ✅ 存在 | 2个case JSON + summary.json + table_cp.csv + refs_derived.json + 4张图 |
| benchmark ref vs got | ⚠️ 存在但FAIL | Wagner解析锚可用；Z&F(1993) WITHHELD；所有压力指标误差230-280% >> 15%阈值 |
| repro_cmd | ✅ 可执行 | `python benchmarks/case_sph_wedge_entry_2d/run.py --stage matrix --shard 0/5 && --stage accept` |

### 测试核验
- **声称**: 13 passed
- **实际**: 10 passed + 3 failed
- **失败原因**: `grade()` 函数 KeyError: `sound_speed_factor` — 合成测试用例 `_synthetic_case()` 缺少该 config 键

### 报告六段核验
| 段 | 状态 |
|---|---|
| ① 案例介绍 | ✅ 完整（物理背景/几何/参数表） |
| ② 方法 | ✅ 完整（WCSPH/Tait EOS/驱动链） |
| ③ 计算结果 | ✅ 完整（数字自工件注入） |
| ④ 结果图 | ✅ 4张图（profile/时程/收敛/湿润） |
| ⑤ 注意点 | ✅ 完整（压缩性噪声/奇点/能量） |
| ⑥ 误差判定 | ✅ 完整（逐项列出，显式回答否<15%） |

---

## ② 判级（L0-L5）

**判定: L4**

| 级别 | 要求 | 状态 |
|---|---|---|
| L0 文档成果 | ✅ | 有文档 |
| L1 数学/单元验证 | ✅ | Wagner解析机器精度验证通过 |
| L2 独立功能实现 | ✅ | WedgeImpact求解器+驱动完整 |
| L3 系统集成 | ✅ | 接入benchmark驱动链 |
| L4 基准验证 | ✅ | 完整算例+误差分析+可复现（结果FAIL但诚实报告） |
| L5 科研工程成果 | ❌ | 未达 |

**降级因素**:
1. 3/13测试实际失败（声称13 passed不实）
2. 所有压力指标对Wagner锚误差230-280%（阈值15%）
3. 缺失3/5计划算例（dx=1mm, 0.5mm, V=5）
4. Z&F(1993)参考WITHHELD（闭源）

---

## ③ 落表

```sql
INSERT INTO tensor_maturity_verdicts
  (repo, scope, mission_id, level, evidence, verdict_by)
VALUES
  ('TensorSPH', 'm16210 CASE-SPH-WEDGE-ENTRY-2D', 16235, 4, '<JSON>', 'board-audit');
```

已执行，row id=33。

---

## ④ 关键发现

### 根因量化
WCSPH压缩性噪声地板：Cp_budget = 2(c₀/V)²/γ·[(ρ/ρ₀)^γ - 1] = 42-43
目标信号：Cp_max,Wagner = 8.402
**噪声地板 >> 目标信号** → 压力精度从根本上不可达

### 改进方向
1. 提高c₀/V至50-100（但计算成本×6-25）
2. 使用δ-SPH密度扩散降低密度噪声
3. 使用不可压SPH（ISPH/PCISPH）替代WCSPH

---

**审计结论**: m16210交付了完整的算例基础设施（驱动/判据/工件/图），但数值结果未达标且测试有缺陷。成熟度L4（有完整benchmark链但结果FAIL）。建议reopened返修：修复测试bug + 提高c₀/V或改用δ-SPH。
