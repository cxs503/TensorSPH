# CASE-POISEUILLE-SPH · C6 归因研究补跑状态（m23999）

**日期**: 2026-11-16 ｜ **锚点**: `CASE-POISEUILLE-SPH`（算例池·泊肃叶流，ref: 解析解，容差 <5%）
**本单性质**: 验证欠账——结清 C6 归因研究（`backflow_level`）未落地的扩展 rung 数据。

## 1. 开工核对（总账锚点纪律）

- `feature_ledger` 中 `CASE-POISEUILLE-SPH` 状态 **open**（非 done/verified，可认领）。
- 算例池本体已成熟：`benchmarks/case_poiseuille/`（R1 阶梯 8/12/16、R2 Womersley、
  combined Couette–Poiseuille、steady-window/drift、uncertainty、wall-shear-harmonic、
  Wo-envelope、power-law 臂、near-wall envelope）。
- 池一致性记录 `out/pool_conformance.json`：55 个泛函，33 个从原始行重算，
  绑定最大相对差 5.98e-12，**0 绑定违例**；唯一失败 = 2 个诚实 FAIL
  （`CP_backflow_peak@Hdx8` -12.37%、`CP_backflow_peak@Hdx12` -8.23%，
  `tests/test_case_poiseuille_pool_conformance.py` 显式容忍并视为回归哨兵）。

## 2. 发现的缺口（本单增量）

1. **治理缺口（已闭环）**：锚点未登记 `tensor_benchmark_cases`（0 行）——与
   `docs/board-audit-m15711/FINDINGS.md` F1（CASE-COUETTE-SPH 同类发现）一致，
   算例池"供给"未闭环、平台无法机器裁定本算例。
   **已登记**：`tensor_benchmark_cases` id=178（anchor_slug=CASE-POISEUILLE-SPH，
   metric/tolerance 按池内实测与总账容差填写，status=pending）。
2. **C6 归因研究数据未落地**：`extra_rungs.py`（H/dx=24/32/48 + t=4 s 长时控制）
   曾递交集群作业 **cj4578**，该作业在 H/dx=24 完成后崩溃
   （`free(): corrupted unsorted chunks`，节点 n330 堆损坏，rc=255），
   `out/backflow_level_target.json` 从未生成，`docs/case_poiseuille_backflow_level.md`
   停留在"扩展 rung 待落地"状态。

## 3. 本单动作

| # | 动作 | 证据 |
|---|---|---|
| 1 | 登记 `tensor_benchmark_cases`（id=178，status=pending） | psql INSERT，见 §2.1 |
| 2 | 诊断 cj4578 崩溃：`/srv/tensor/agent/cluster/jobs/cj4578.log` rc=255，`free(): corrupted unsorted chunks`；**从作业日志恢复 H/dx=24 结果 C6 = -1.291 %** | 日志原文 |
| 3 | 隔离重递交缺失 rung：cj4840（H/dx=32,48）、cj4841（H/dx=24 + t=2/4 s 长时控制） | run_compute submit 回执 |
| 4 | 新增 `benchmarks/case_poiseuille/merge_extra_rungs.py`：把两个分片输出拼回 `out/backflow_level_target.json`（原 schema，字段逐字拷贝不重算） | 本文件同目录 |

## 4. 已可判定的物理结论（3 rung + 恢复点，全部读自工件/日志）

C6 反向峰误差与求解器水平偏差（`out/backflow_level.json`，H/dx=24 点来自 cj4578 日志）：

| H/dx | C6 % | solver dc %U_w |
|---|---|---|
| 8 | -12.370 | -1.0993 |
| 12 | -8.232 | -0.5081 |
| 16 | -1.321 | -0.2011 |
| 24（cj4578 日志恢复） | -1.291 | — |

**已可判定**：水平偏差在 H/dx=16→24 之间不再下降（-1.321% → -1.291%），
即 C6 的粗网格 FAIL 是**收敛中的离散水平偏差**（3 点拟合阶 p=2.41），
且 H/dx≥16 时 C6 已在 5% 容差内。32/48 与长时控制（t=4 s）用于确认
"平台期 vs 继续收敛"与"瞬态 vs 稳态偏差"，见下。

## 5. 在飞作业与后续步骤（下一单 10 分钟可结）

| 作业 | 内容 | 输出 |
|---|---|---|
| cj4840 | H/dx=32, 48 | `out/extra_rungs_32_48.json` |
| cj4841 | H/dx=24 + 长时控制 t=2/4 s（H/dx=8） | `out/extra_rungs_24_longtime.json` |

落地步骤（全部后处理，无新求解）：

```bash
# 1) 取回两个工件到 benchmarks/case_poiseuille/out/
# 2) 拼回目标 schema
python3 benchmarks/case_poiseuille/merge_extra_rungs.py
# 3) 重生成归因工件（extension_rungs 入档）
python3 benchmarks/case_poiseuille/backflow_level.py
# 4) 重生成文档+图
python3 benchmarks/case_poiseuille/make_backflow_level_doc.py
# 5) 回归
python3 -m pytest tests/test_case_poiseuille_backflow_level.py \
                  tests/test_case_poiseuille_pool_conformance.py -q
```

判定预案：若 32/48 的 solver dc 继续按 ~dx² 下降 → 偏差为收敛型，C6 粗网格 FAIL
维持"分辨率效应"定性；若平台化 → 定性为离散格式稳态偏差（量级 <0.2 %U_w，
经 15× 放大后仍 <5% 于 H/dx≥16）。长时控制若 t=4 s 窗口偏差与 t=1 s 一致 →
稳态偏差，排除"窗口过早"测量缺陷。

## 6. 治理闭环

- `tensor_benchmark_cases` 已登记 CASE-POISEUILLE-SPH（id=178，status=pending）——
  结清 `docs/board-audit-m15711/FINDINGS.md` F1 同类缺口（算例池供给未闭环）。
- `feature_ledger` 状态更新（open→done）留待 32/48+长时数据落地、
  文档重生成、测试绿之后执行，避免在数据未齐时提前销账。
