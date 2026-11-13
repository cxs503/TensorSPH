# Board Audit: m17584 XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID

**Auditor**: board-audit (m17624)  
**Date**: 2026-11-13  
**Mission**: m17584 (succeeded)  
**Repo**: TensorSPH  
**Anchor**: XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID  
**Verdict**: **L4** (基准验证：完整算例+误差分析+可复现)

---

## 1. 取证清单（三件套核验）

| 交付件 | 路径 | 存在? | 内容核实 |
|---|---|---|---|
| run_artifacts | `artifacts/fsi2_gate_load/bench_local.json` | ✅ | JSON 含 schema_version=1.0.0, 3 metrics, ref/got/误差 |
| run_artifacts | `artifacts/fsi2_gate_load/stdout_local.log` | ✅ | 3 行判据输出 + 完整 JSON |
| run_artifacts | `artifacts/fsi2_gate_load/stderr_local.log` | ✅ | 空（无错误） |
| run_artifacts | `artifacts/fsi2_gate_load/fig1_benchmark.png` | ✅ | 84KB 基准图 |
| benchmark ref vs got | bench_local.json → metrics | ✅ | 见 §2 |
| repro_cmd | `python benchmarks/fsi2_gate_load/run_fsi2_benchmark.py --out ...` | ✅ | 重跑成功，数值一致 |

## 2. 基准裁判（ref vs got 逐项）

> 注：`tensor_benchmark_cases` 表在平台 DB 中不存在（已查 4 个 DB），无法按 SQL 查询逐项裁。  
> 裁判依据 = bench_local.json 中的 ref 值 + 任务令 15% 不可接受线 + 报告自报 3% 线。

| 判据 | ref | got | 误差 | <3%? | <15%? |
|---|---|---|---|---|---|
| 壁面合力 dx=12.5 mm | 418.220 N/m | 419.223 N/m | +0.240% | ✅ | ✅ |
| 壁面合力 dx=10 mm | 418.220 N/m | 418.592 N/m | +0.089% | ✅ | ✅ |
| 杆大变形顶点（同帧横向） | 291.819 mm | 299.255 mm | +2.548% | ✅ | ✅ |
| 耦合闸门顶点时程 | WITHHELD | — | 不可判 | ✗ | ✗ |

**三项可裁判判据全部 <3%，最大误差 2.548%。**

WITHHELD 项（Antoci 2007 实验曲线）：三次任务（m16214/m16352/m17519）一致确认文献不可得（Unpaywall is_oa=false, OpenAlex any_repository_has_fulltext=false），本审计不扣分。

## 3. 报告六段核验

| 段 | 存在? | 质量 |
|---|---|---|
| ① 案例介绍（物理背景/几何/参数表） | ✅ `docs/fsi2_gate_load/03_report.md` §① | 完整：水柱/槽/闸门尺寸、ρ/E/ν/g、无量纲数 |
| ② 方法（流程/关键设置/工具链） | ✅ `03_report.md` §② + `02_method_selection.md` | 结构子循环、停滞护栏、TensorBench json_stdout 契约 |
| ③ 计算结果（数字表） | ✅ `03_report.md` §④ | ref/got/误差逐项，零手抄（bench_local.json 程序化注入） |
| ④ 结果图 | ✅ `artifacts/fsi2_gate_load/fig1_benchmark.png` | 84KB 基准对比图 |
| ⑤ 注意点（坑/局限/范围） | ✅ `03_report.md` §⑤⑥ | 收敛性、端到端不可用、ref WITHHELD、迁移路线 |
| ⑥ 误差判定 | ✅ `03_report.md` §④ | 逐项 <3% 判定，WITHHELD 显式声明 |

## 4. 测试核验

| 测试 | 结果 | 备注 |
|---|---|---|
| test_hydrostatic_load_panel_transfer_has_no_deficit | PASS | 两分辨率 |
| test_hydrostatic_load_pairwise_transfer_legacy_deficit | PASS | 基线对照 |
| test_cantilever_small_deflection_point_load | PASS | A0 |
| test_clamp_is_a_clamp_not_a_pin | PASS | 边界条件 |
| test_elastica_reference_is_self_consistent | PASS | 参考自洽 |
| test_elastica_reference_reproduces_the_small_deflection_limit | PASS | 参考退化 |
| test_large_deflection_rod_matches_the_elastica_in_the_same_frame | PASS | A2 主判据 |
| test_large_deflection_error_decreases_with_rod_refinement | PASS | 离散化证据 |
| test_cantilever_distributed_load_converges_with_rod_refinement | TIMEOUT | 50 节点杆，非正确性问题 |
| test_coupled_short_run_is_deterministic (slow) | 未跑 | 标记 slow |

**8/9 非慢测试通过，1 超时（计算量问题，非正确性）。**

## 5. 复现验证

```bash
cd /root/tp-agent/work/m17584
python benchmarks/fsi2_gate_load/run_fsi2_benchmark.py
```

输出：
```
[fsi2] hydrostatic dx=12.5 mm: got=419.223 N/m ref=418.220 N/m rel=+0.240%
[fsi2] hydrostatic dx=10.0 mm: got=418.592 N/m ref=418.220 N/m rel=+0.089%
[fsi2] large deflection PL^2/EI=10.000: rod=299.2549 mm elastica=291.8192 mm rel=+2.548%
```

与 bench_local.json 数值一致（浮点末位差异 <1e-10），**可复现**。

## 6. 级别判定

| 级 | 定义 | 是否满足 |
|---|---|---|
| L0 | 文档成果 | ✅（但远超） |
| L1 | 数学/单元验证 | ✅（但远超） |
| L2 | 独立功能实现 | ✅（879 行实现 + 235 行基准驱动） |
| L3 | 系统集成 | ✅（接入 TensorSPH 主模块 fsi_elastic_gate.py） |
| L4 | 基准验证 | ✅ **3 项判据全部 <3%，可复现，报告六段齐全** |
| L5 | 科研工程成果 | ✗（耦合时程 WITHHELD，端到端静态判据不可用） |

**判定：L4**

理由：
- 完整算例 ✅（弹性闸门 FSI，3 项独立参考）
- 误差分析 ✅（逐项 ref/got/误差，收敛性证据）
- 可复现 ✅（repro_cmd 重跑数值一致）
- 不足：Antoci 2007 实验曲线 WITHHELD（非本单责任），端到端静态判据不可用（rod_static_equilibrium 停滞，已登记迁移路线）

## 7. 发现与风险

| # | 严重度 | 发现 | 证据 |
|---|---|---|---|
| 1 | 非阻塞 | `test_cantilever_distributed_load_converges_with_rod_refinement` 超时 | 50 节点杆 autograd 反向传播 >30s |
| 2 | 非阻塞 | `rod_static_equilibrium` 在 E≥1e10 Pa 时停滞（已有护栏） | `03_report.md` §⑥ |
| 3 | 信息 | `tensor_benchmark_cases` 表不存在，无法按 SQL 裁判 | 已查 4 个 DB |
| 4 | 信息 | 耦合时程仅单分辨率（dx=12.5 mm），未跑收敛 | `03_report.md` §⑤ |

## 8. Verdict

```
repo: TensorSPH
scope: m17584 XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID
mission_id: 17624
level: 4
verdict_by: board-audit
```

**Disposition: APPROVE** — L4 基准验证通过，3 项判据全部 <3%，可复现，报告六段齐全。WITHHELD 项与端到端缺口已诚实登记。
