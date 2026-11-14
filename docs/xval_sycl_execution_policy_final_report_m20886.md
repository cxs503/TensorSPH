# XVAL: SPHinXsys sycl-execution-policy-not-transferable 做法验证 — 最终报告

**锚点**: `XVAL-SPH-SYCL-EXECUTION-POLICY-NOT-TRANSFERABLE`
**任务**: m20886 (verifier 席自找活)
**日期**: 2026-11-14
**平台报告 ID**: 18924
**Feature Ledger**: 状态已更新为 `verified`

## 执行摘要

对 SPHinXsys (Xiangyu-Hu/SPHinXsys) 的 SYCL 执行策略不可转移设计进行了完整的交叉验证。TensorSPH 端口实现了：

1. **不可复制的 ExecutionInstance 单例** — 拷贝/深拷贝/pickle 均拒绝，匹配 OSS 的 `= delete` 语义
2. **编译期策略冻结** — `MAIN_EXECUTION_POLICY` 在模块导入时固定，`set_main_execution_policy` 抛出异常
3. **填充启动几何** — `get_uniform_nd_range` 是 OSS `getUniformNdRange` 的逐字节副本
4. **尾掩码迭代器** — `iterate_blocks` 生成带掩码的填充工作块，掩码缺失可被测试检测

在 200 流体粒子 + 420 边界粒子的溃坝柱算例上：
- **填充掩码扫描与向量化参考在所有 work-group 尺寸下位精确一致**（最大相对误差 **0.0**）
- **负对照（移除尾掩码）误差达 43.2x**（local_size=64, 56 空闲工作项），证明掩码是负载关键的
- **第三方标量循环交叉验证**确认一致性在浮点舍入级别（5.7e-16）

所有门限通过：owner 标准 <15% ✅，工程目标 <1e-12 ✅，负对照 >15% ✅。

## 验证矩阵

| 检查项 | 方法 | 结果 |
|---|---|---|
| 单例/不可复制 | `copy.copy`, `copy.deepcopy`, `pickle.dumps` | 全部拒绝 ✅ |
| 策略冻结 | `set_main_execution_policy` | 抛出 RuntimeError ✅ |
| 填充几何 | 独立 ceil 重推导 vs OSS 公式 | 11×8=88 组参数全部匹配 ✅ |
| 块覆盖 | 每个逻辑索引恰好访问一次 | 全部覆盖 ✅ |
| 数值等价 | 11 种 work-group 尺寸 | 最大误差 0.0 ✅ |
| 负对照 | 移除尾掩码 | 误差 43.2x > 15% ✅ |
| 第三方意见 | 标量逐粒子循环 | 误差 5.7e-16 ✅ |
| 平滑长度约定 | 均匀 h vs 变化 h | 均匀时位精确，变化时用 row_h 参考 ✅ |

## 产物清单

| 文件 | 角色 |
|---|---|
| `src/tensorsph/execution_policy.py` | 执行策略端口（360 行） |
| `src/tensorsph/wcsph/swept.py` | 填充掩码扫描实现 |
| `tests/test_execution_policy.py` | 127 条测试断言 |
| `benchmarks/xval_sycl_execution_policy/run.py` | 测量工具 |
| `artifacts/xval_execution_policy/results.json` | 数值结果 |
| `artifacts/xval_execution_policy/fig_*.png` | 4 张结果图 |
| 平台报告 #18924 | 论文级验证报告 |

## 复现

```bash
cd /srv/tensor/agent/work/m20886
python -m pytest tests/test_execution_policy.py -q        # 127 passed
python benchmarks/xval_sycl_execution_policy/run.py --quick  # all_gates_pass: true
```
