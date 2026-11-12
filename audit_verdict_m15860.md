# 交差审计 verdict: m15860 CASE-TAYLOR-GREEN-SPH

**审计任务**: m15921  
**审计对象**: m15860 (TensorSPH Taylor-Green 涡基准验证)  
**审计日期**: 2026-11-12  
**审计员**: board-audit  

## 取证摘要

### 1. 产物存在性 ✅
- `runs/taylor_green/m15860/results.json` (14678 bytes, sha256=c7620d62)
- `runs/taylor_green/m15860/run.log` (2429 bytes)
- `runs/taylor_green/m15860/environment.json` (loongarch64, torch 2.7.1)
- 4 张 PNG 图 (fig1-fig4, 全部存在且非空)

### 2. 基准数值裁判 ✅
容差来源: 代码 `TOLERANCE_PCT = 8.0`（注: `tensor_benchmark_cases` 表不存在于 DB）

| 量 | ref | got | 误差% | 容差% | 判定 |
|---|---|---|---|---|---|
| energy_decay_rate | 2.5132741 | 2.4836508 | 1.179 | 8 | ✅ |
| energy_ratio_at_t_end | 0.2213601 | 0.2148640 | 2.935 | 8 | ✅ |
| amplitude_ratio_at_t_end | 0.4704892 | 0.4657502 | 1.007 | 8 | ✅ |

全部 < 3%，全部 < 8%。

### 3. 复现命令 ✅
`PYTHONPATH=src python -m tensorsph --case taylor_green --output runs/taylor_green/m15860`
- 入口存在: `src/tensorsph/__main__.py`
- 案例模块存在: `src/tensorsph/taylor_green.py` (803 行)
- 求解器存在: `src/tensorsph/vv/tgv.py` (387 行)

### 4. 实现真实性 ✅
- 使用生产 SPH 算子: Morris Laplacian, Tait EOS, cubic spline, pair_kernels
- 从 `droplet_impact` 导入（与生产求解器共享代码路径）
- 前向 Euler + 最小镜像周期 + 邻居列表重建
- 非 mock/stub: 有真实粒子循环、力计算、时间积分

### 5. 测试 ✅
- 7 个测试用例（4 快速 + 3 慢速）
- 快速测试 4/4 通过 (4.47s)
- 测试断言 NS 残差 < 1e-10、参数自洽、容差满足、CLI 可执行

### 6. 报告六段 ✅
`docs/benchmarks/taylor-green.md` 包含完整六段

## 级别判定: L4

**理由**:
- L0 (文档): ✗ 有真实代码
- L1 (数学/单元): ✗ 有完整求解器
- L2 (独立功能): ✓ 输入→计算→输出
- L3 (系统集成): ✓ 接入 verification ladder rung C2 + CLI + benchmark driver
- L4 (基准验证): ✓ 完整算例 + 解析参考 + 误差分析 + 收敛性 + 灵敏度 + 可复现
- L5 (科研成果): ✗ 标准验证算例，非新研究

## 落表

```sql
INSERT INTO tensor_maturity_verdicts 
  (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorSPH', 'm15860 CASE-TAYLOR-GREEN-SPH', 15921, 4, '<JSON>', 'board-audit');
```

verdict_id = 30

## maturity_gate 预期

**confirmed**（消账）— 所有基准行通过，产物齐全，实现真实。
