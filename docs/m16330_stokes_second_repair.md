# m16330 — 返修收口：`CASE-SPH-STOKES-2ND-PROBLEM`（振荡平板 Stokes 第二问题）

上一轮 m16239 的独立复核（`docs/audit_m16239_CASE-SPH-STOKES-2ND-PROBLEM.md`）判 L4
但有返修项 F1–F8。本单逐项收口，并在交付**入库前**修掉，避免"证据只在工作树里"。

## 交付面（L2/L3/L4 要件 → 路径）

| 要件 | 路径 | 说明 |
|---|---|---|
| 实现（非占位） | `src/tensorsph/stokes_second.py` | 半空间/有限深解析解、谐波拟合、sympy 再推导 + 数值再校验 |
| 入口 | `benchmarks/stokes_second/run.py` | `--ladder` / `--sweep` / `--probe` / `--quick` / `--sensitivity-only` |
| 运行工件 | `benchmarks/stokes_second/out/` | `results_ladder.json`(1.3 MB)、`results_sensitivity.json`(101 KB)、`benchmark.json`、`MANIFEST.json`(逐工件 SHA-256)、`amplitude_decay.png`、`field_snapshot.png` |
| 基准对照 | `benchmarks/stokes_second/out/benchmark.json` | `anchor`、`primary_metric`、`primary_value_pct`、逐行 ref/got/tolerance/pass |
| 复现命令 | 同上 `repro_cmd` 字段；DESIGN §7 | `PYTHONPATH=.:src python benchmarks/stokes_second/run.py --ladder --c0 4 --periods 6 --measure-periods 2` |
| 集成调用点（生产侧） | `src/tensorsph/wall_mirror.py` `plane_vel`；`src/tensorsph/__init__.py` 导出；`benchmarks/stokes_second/run.py:192` 调用 `plane_walls`、`:196` 调用 `droplet_impact.rhs` | 生产边界核新增逐面速度，`w=0` 退化为旧行为，旧数字逐位不变 |
| 文档 | `docs/verification_stokes_second.md`（由 `make_doc.py` 从工件生成）、本文件 | 六段式 |

## 复核意见逐项收口

| # | 上轮问题 | 本单处置 | 证据 |
|---|---|---|---|
| F1 | "收敛阶梯"n=1 空转，G6 在 n=1 下为真 | 阶梯跑满 3 档；`convergence.complete` 进判据，缺档即 `all_gates=false`；n<2 时 `G6_not_growing=false` 并写 `G6_reason`；另记观测阶 | `results_ladder.json:convergence` = `resolutions [8,12,18]`, `l2_by_resolution [0.0674, 0.0182, 0.0102] %`, `observed_order [3.228, 1.439]`, `complete true` |
| F2 | 工件被门禁滤除（4.2/10.4 MB） | 持久化瘦身：仅最细网格保留时空场且时间抽 1/8、数值 4 位有效；敏感度块丢场/轨迹。结果 JSON 1.26 MB / 101 KB，`MANIFEST.json` 记每个文件 SHA-256 | `benchmarks/stokes_second/out/MANIFEST.json` |
| F3 | `make_doc.py` 语法错、docs 与平台报告双缺 | 修 f-string 前缀；docs 页由工件程序化生成；本单报告已发（`tensor_reports` id 15423） | `docs/verification_stokes_second.md`；`scripts/post_report_m16330.py` |
| F4 | `provenance.case.c0` 记常量 10，实跑 4 | `provenance.case` 由解析后的实参回填：`c0`、`periods`、`measure_periods`、`ladder`、`mach`、`Re_delta`、求解器名 | `results_ladder.json:provenance.case` = `c0 4.0, periods 6.0, measure_periods 2.0` |
| F5 | 波峰位置断言错（2 倍）且红着交付 | 期望改为 `delta(omega t - pi/4)`（驻定峰），另加相位速度上界断言；新增 G1 判据量回归守卫 `test_criterion_metric_stays_inside_the_g1_gate`（dx=6 两周期，实测 0.16 %） | `tests/test_stokes_second.py`，`pytest -k "criterion or plate_boundary or moving"` 5 passed |
| F6 | 无机器裁判行 | `benchmark.json` 提供 `primary_metric`+`tolerance`+逐行 pass，并可由 `MANIFEST.json` 校验哈希 | 同上 |
| F7 | 未进 main | 本分支交付，工件随实现一起入库 | 本分支 |
| F8 | 文件名易混（`3p5p` 缺 c0） | 本次只发布 ladder（`--c0 4 --periods 6`），probe 名不再混入 out/ | `ls benchmarks/stokes_second/out` |

## 结果（ref = 解析 `U0 exp(-y/delta)`；got = SPH 行平均振幅谐波拟合）

| δ/Δx | 流体粒子 | 层内行 | G1 L2 % | G2 最差行 % | G3 衰减率误差 % | δ_fit/δ | G4 有限深 L2 % | 墙钟 s |
|---|---|---|---|---|---|---|---|---|
| 8 | 288 | 8 | 0.0674 | 0.1091 | -0.1155 | 0.99885 | 0.0689 | 132 |
| 12 | 432 | 12 | 0.0182 | 0.0302 | -0.0315 | 0.99969 | 0.0197 | 271 |
| 18 | 648 | 18 | 0.0102 | 0.0171 | -0.0177 | 0.99982 | 0.0117 | 853 |

负控（同网格 δ/Δx=8）：自由滑移运动壁 `A_max/U0 = 0.0000`；静止壁 `A_max/U0 = 0.0000`——驱动的确来自壁面剪切，不是别处。`U0` 减半 → G1 0.0673 %（线性）；`h/Δx` 1.3→1.9 → G1 升到 0.6471 %（分辨率/核宽方向正确）。

判据：<5 % 门槛下最差 0.0674 %（**<3 % 成立**）；δ_fit/δ 三档最小 0.99885。

## 生态摩擦（如实记）

1. **工件体积门槛**：4.2 MB/10.4 MB 的结果 JSON 会被交付门禁滤除（上轮已发生一次），
   而"真实工件入库"是 L2 硬要件——结果文件需要自带瘦身策略（本次内置 `_slim_payload`）。
2. **np.bool_ 混入 gates 导致收尾 `json.dumps` 崩**（本单实遇）：三档全跑完、工件已落盘，
   仅在末尾打印 gates 时抛 `TypeError`，进程退出码 1。修法：打印与落盘统一走
   `default=_jsonable`。教训：跑完 ≠ 成功，入口的最后一段也要覆盖。
3. **dt 限制切换**：δ/Δx=18 时粘性限 `0.125 h²/ν` 接管声学限，步数从 1.3e4 涨到 2.9e4，
   墙钟 853 s（≈δ/Δx=12 的 3.1 倍），阶梯最细档是阶跃式变贵，预算要按此估。
4. **默认线程数**：这批小张量上 torch 默认线程池比单线程慢约 6 倍；单测里不加
   `torch.set_num_threads(1)` 会让 30 s 的用例超 120 s。
