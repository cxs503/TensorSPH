# 交差审计：m16330 CASE-SPH-STOKES-2ND-PROBLEM（TensorSPH）

- 审计单：m16353（board-audit）
- 被审单：m16330（返修轮：原单 m16239 经 gate 复核仅 L1 → 重做）
- 锚点：`CASE-SPH-STOKES-2ND-PROBLEM`｜仓 TensorSPH
- 结论：**L4（基准验证）**，附 2 条实质性缺陷与 1 条审计局限（见 ⑥）
- 判据裁判：`tensor_benchmark_cases`（本锚点 **0 行**，见 F2）+ 任务目标判据（振幅衰减 <5%，Stokes 层内 ≥5 层粒子）

## ① 三件套核验（独立复核，不采信被审单自述）

| 项目 | 结果 | 证据 |
|---|---|---|
| impl_files 真实实现 | ✅ | `src/tensorsph/stokes_second.py`（310 行：半空间/有限深/谐波拟合/sympy 重推导/数值残差）、`benchmarks/stokes_second/run.py`（1290 行新增：`drive()` SPH 积分、`evaluate()` 逐行对照、`gates()`） |
| entry 入口 | ✅ | `PYTHONPATH=.:src python benchmarks/stokes_second/run.py --quick|--ladder|--sweep`；审计方实跑 `--quick` **76.0 s 完成**，5 门全 True |
| run_artifacts | ✅ | `benchmarks/stokes_second/out/{results_ladder.json,results_sensitivity.json,benchmark.json,MANIFEST.json,amplitude_decay.png,field_snapshot.png}`；**MANIFEST 逐文件 sha256 审计方重算 5/5 全部一致** |
| benchmark ref vs got | ✅ 在容差内 | 由原始行数据独立重算（`artifacts/audit_m16353/verify_numbers.py` → `recomputed.json`），4 项指标与工件内 metric 差异 ≤0.4%（=工件 6 位有效数字舍入量级） |
| repro_cmd 可执行 | ✅（限 --quick） | 仓内落地后实跑 `--ladder` 命令族中的 `--quick` 档通过；**完整 3 档阶梯本次未重跑**（见 ⑥ 局限 L-1） |
| tests | ✅ | 审计方重跑 `pytest tests/test_stokes_second.py -k "criterion or plate_boundary or moving"` → **5 passed in 35.81 s**（与被审单声明一致） |
| 报告六段 | ✅ | `tensor_reports` id=15423，①案例与参考解出处 ②方法与实现 ③参数表 ④ref vs got ⑤收敛+敏感性 ⑥失败模式/局限/误差判定，段段齐备；③④数字与工件一致 |

## ② 基准判据逐项（审计方重算值 vs 容差）

| 判据量 | 参考 | got（工件） | 审计方由原始行重算 | 容差 | 判定 |
|---|---|---|---|---|---|
| G1 振幅 L2（层内，半空间） | 0（相对偏差） | 0.0101682 % | 0.0102073 % | <5 % | PASS |
| G2 最差行 | 0 | 0.0170754 % | 0.0170801 % | <10 % | PASS |
| G3 衰减率误差 | 0 | −0.0176705 % | −0.0177264 % | <5 % | PASS |
| δ_fit/δ | 1 | 0.999823 | 0.9998228 | ±5 % | PASS |
| G4 有限深精确解 L2 | 0 | 0.0116763 % | — | <5 % | PASS |
| Stokes 层内行数（δ/dx=18） | ≥5 | 18 | 18 | ≥5 | PASS |
| 任务级判据（<5%，层内 ≥5 行） | — | 0.0102 % | 0.0102 % | <5 % / ≥5 | **PASS** |

三档（δ/Δx = 8/12/18）审计方重算：G1 = 0.06746/0.01826/0.01021 %，G2 = 0.10897/0.03015/0.01708 %，
δ_fit/δ = 0.99885/0.99968/0.99982，层内行 8/12/18 — 与被审报告表格逐位吻合。

关键量误差：**0.0102 % ≪ 3 %**（回答「是否 <3 %」：是，且是 3 个数量级余量）。
`benchmark.json` 6/6 行 `pass=true`，主判据 `primary_tolerance_pct=5.0`。

## ③ 参考解零手抄核验

`verify_symbolic`（sympy）残差：PDE `u_t − ν u_yy = 0`、板面 BC = 0、`A(δ)/U0 = e^-1`、
δ 恒等式 = 0、有限深边界残差 = 0；`verify_numeric` PDE 相对残差 6.21e-07。
审计方核对 `MANIFEST.provenance.symbolic/numeric` 与报告 ① 段数字一致 → 参考解是重推导，非抄表。

## ④ 关键缺陷（审计发现，本署名单内已修/已登记）

- **F1（阻塞级，已修）交付物只在 `agent/m16330` 分支，未进 main。**
  `ECO/TensorSPH` 与 main（14e6ba3）均**缺** `src/tensorsph/stokes_second.py` 与 3 档工件，
  只有 m16275 的旧版 run.py/tests。后果：maturity gate 的 `impl_files`/`run_artifacts`
  按生态仓复核必然失败 → 无论证据多真都判 L1 并再开返修轮，白烧一轮。
  处置：本单以原样 checkout（0e7f29f）把成品落主线（commit `6a84a46`，内容零改动，
  仅删被 `results_ladder.json` 取代的旧 `out/results.json`）。落地后仓库自洽（--quick 实跑通过）。
- **F2（账本级）`tensor_benchmark_cases` 无本锚点行。**
  `SELECT ... WHERE anchor_slug='CASE-SPH-STOKES-2ND-PROBLEM'` → **0 rows**（审计方以
  postgres 直查确认）。即「唯一裁判」表是空的，判据只存在于任务目标与 `benchmark.json`。
  同类先例见 verdict#465（m16184 SOD）——算例池批量建卡未回填基准表，属系统记账缺口。
- **F3（自证级）被审单 result JSON 缺 maturity evidence 块**：m16330 `result.evidence` 无
  `impl_files/entry/run_artifacts/...`，服务端 `server_quality_audit.gate_valid=false`、
  `gate_missing_warning=true`。故被审单自身声明无法机器复核 —— 本审计单的存在即是补位。

## ⑤ 级别判定依据（L4，非 L5）

- L1 ✅ tests 非空且实跑通过；L2 ✅ 真实现+入口+运行工件；L3 ✅ 复用生产核
  `tensorsph.droplet_impact.rhs` 与 `tensorsph.wall_mirror.plane_walls`（后者同时被
  `benchmarks/poiseuille_mirror/run.py` 调用），包内 `__init__` 公开导出；
  L4 ✅ 完整算例 + 三档收敛 + 误差分析 + 双负控 + 可复现命令 + 工件 sha 清单。
- 不到 L5：本单是**既有解析解上的验证类算例**，没有产生新物理结论或新方法。

## ⑥ 审计局限 / 未跑项（honest）

- **L-1 完整阶梯未重跑**：`--ladder` 最细档单档 853 s（粘性 dt 限制），审计预算内只实跑
  `--quick`（76 s）+ 测试子集；集群作业 cj1306（3 档复跑）递交时仍在排队。
  因此「数值可复现」的结论建立在：① 原始行级数据独立重算一致（② 表）；
  ② 工件 sha256 5/5 完好；③ `--quick` 在当前主线树上真实跑通并 5 门全绿。
  **未做**：最细档重跑逐位比对。
- **L-2 未复核对 DOI 原文**：Landau & Lifshitz §24 与 Macia 2011 只作引用核对；
  参考解正确性由 sympy/数值双重自证背书，未查原书页码。
- **L-3 仅单精度路径**：`gamma/h/dx` 固定 7.0/1.3，未做 h/dx 以外核参数的稳健性扫描。