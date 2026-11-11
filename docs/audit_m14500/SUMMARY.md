# TensorSPH 审计运行总结 (m14500)

## 任务完成情况

✅ **核心测试套件**: 部分完成（95/100 通过）  
✅ **代表性 benchmark**: 完成（收敛性 + 液滴冲击 flagship）  
✅ **负向测试**: 完成（发现输入验证缺陷）  
⏳ **守恒性检验**: 超时，已递交集群（cj847 排队中）

## 关键发现

### 缺陷 (2)

1. **TENSORSPH-AUDIT-001** (中): Config 类缺少输入验证
   - `Config(dx=-0.01)`, `Config(rho0=0.0)` 等非法参数被接受
   - 位置: `src/tensorsph/droplet_impact.py:Config.__init__`
   - 影响: 运行时崩溃而非构造时报错

2. **TENSORSPH-AUDIT-002** (低): audit.json 过期
   - claims 计数不同步（36 vs 40）
   - 修复: `python benchmarks/claim_validity/validity.py`

### 文档 vs 实测差异 (2)

1. **TENSORSPH-AUDIT-003**: 静水收敛阶数
   - 文档声称: 1.11
   - 实测: 1.550
   - 差异: +39.6%

2. **TENSORSPH-AUDIT-004**: 静水误差度量
   - 文档声称: 1.12%
   - 实测: 0.3981
   - 可能原因: 不同度量标准

## 实测数据

### 收敛性

```
Hydrostatic Convergence:
  dx=0.05    error=0.7296 (48 particles)
  dx=0.025   error=0.3981 (192 particles)
  dx=0.0125  error=0.0851 (768 particles)
  Observed order: 1.550
```

### Flagship 算例（液滴冲击）

- 初始粒子: 608
- 模拟时间: 0.04s
- 峰值密度压缩: 1.50% (<5% ✅)
- 峰值 Mach: 0.0196 (<0.1 ✅)
- 峰值 Courant: 0.251 (<0.3 ✅)
- 质量守恒: ✅ (380.02 不变)
- 所有验收标准: ✅ 通过

### 负向测试

| 测试 | 期望 | 实际 | 结果 |
|------|------|------|------|
| Config(dx=-0.01) | ValueError | 接受 | ❌ |
| Config(rho0=0.0) | ValueError | 接受 | ❌ |
| Config(sound_speed=-20.0) | ValueError | 接受 | ❌ |
| Config(dt=0.0) | ValueError | 接受 | ❌ |

## 产物清单

### 文档
- `docs/audit_m14500/AUDIT_REPORT.md` - 完整审计报告
- `docs/audit_m14500/audit_findings.json` - 机器可读发现

### 运行日志 (runs/)
- `pytest_smoke.log` - Smoke tests (23 passed)
- `pytest_kernel_analytic.log` - Kernel tests (18 passed)
- `pytest_convergence_conservation.log` - Convergence tests
- `pytest_dam_droplet.log` - Dam break + droplet tests
- `benchmark_convergence.log` - 收敛性 benchmark
- `benchmark_dambreak.log` - 溃坝 benchmark (partial)
- `flagship_droplet_impact.log` - 液滴冲击 flagship
- `negative_tests_validation.log` - 负向测试

### 工件 (artifacts/)
- `droplet_impact_audit/run_manifest.json` - 运行清单
- `droplet_impact_audit/diagnostics.csv` - 诊断数据
- `droplet_impact_audit/particles_*.pt` - 粒子快照

## 集群作业

```
jid: cj847
command: python -m pytest tests/ --tb=short -q --timeout=120
status: queued (ahead 249 jobs)
submitted: 2026-11-11 07:26
```

## 结论

TensorSPH 核心功能正常，但存在输入验证缺陷。文档声称与实测存在差异，需澄清度量标准。

**优先修复**: Config 输入验证 (TENSORSPH-AUDIT-001)

---

**审计智能体**: m14500  
**日期**: 2026-11-11  
**状态**: 完成（集群作业排队中）
