# m15320 · 交差审计：m15264 CASE-HYDROSTATIC（TensorSPH 静水压剖面）

被审 commit：`agent/m15264 36da0f2`；审计工作树：`/root/tp-agent/work/m15320 @ b64b5f4`

## ① 案例介绍

被审对象：mission **m15264**（TensorSPH，CASE-HYDROSTATIC 算例池·静水压剖面），
commit `agent/m15264 36da0f2`。物理=静止水箱中的静力平衡：dp/dy = -rho0 g，
闭式解 `p(y) = rho0 g (H - y)`；求解器用 WCSPH（Tait EOS, `B = rho0 c0^2/gamma`，
Monaghan 1994）+ Adami 2012 壁面外推 + 求和密度。

| 参数 | 值 |
|---|---|
| 水槽 W x H | 1.0 m x 0.4 m |
| 格距 dx / 列数 Nx | 0.02 m / 50（门控点） |
| 粒子数 | 1000 |
| h/dx, 支撑/dx | 1.3, 2.6 |
| rho0 / g | 1000 kg/m3 / 9.81 m/s2 |
| c0 / (c0/sqrt(gH)) | 20 m/s / 10.096 |
| dt / steps | 3.9000e-04 s / 20513 |
| 平均窗 | t in [4, 8] s（4.0 s） |

## ② 怎么做的（审计流程）

1. 取证：读 m15264 的 `benchmarks/case_hydrostatic/`（run.py / report.json / runs/ 20 个 JSON /
   3 张图）与 `docs/case_hydrostatic_report.md`，确认三件套存在性与 git 归属
   （20 个 run JSON + report.json + 3 png 已入 commit 36da0f2）。
2. 基准裁判：`tensor_benchmark_cases` 中 `anchor_slug='CASE-HYDROSTATIC'` 查询得到
   **0 行**（未登记，见 F1）→ 退回到 mission goal 声明的容差 <1% 作为判据，
   got 由审计**独立复算**：从 `runs/window_8s.json` 的 8 个 band 用
   `(p_SPH - p_ref)/p_ref`、`p_ref = rho0 g (H - y)` 重算。
3. 复现：把 case 目录拷入 m15320 工作树，删掉 report.json 与图后重跑
   `run.py --aggregate --figure`（只吃 runs/*.json）；另在 m15264 分支 src 下**实跑**
   一个动态点与全套静态点。
4. 独立探针：自写 `scripts/audit/m15320_hydrostatic_probe.py`，在算例网格
   **之外**的分辨率（dx=0.025 m, Nx=40）做一次 RHS 求值，
   用自写质量加权 band 归约（**不调用 band_table**）对闭式解。
5. 测试：在 m15320 独立副本跑 `tests/test_case_hydrostatic.py`（8 passed, 1 warning in 9.58s）。

## ③ 计算结果

**门控（判据先于动手，沿用仓内 band 度量）**：`max_band_error_pct`（排除自由面 band 的
时间平均压强最大相对误差）限 **1.0 %**，门控配置 `window_8s`。

- got = **0.743956 %**  < 1.0 % → **PASS**
- 审计独立复算 = **0.743956 %**（与工件逐位一致）
- 完整支撑（排除两侧壁 2h 内列）口径 = 0.8145 %
- 负对照 g=0 → 均值误差 99.9999 %（度量确实测重力/压强平衡）

| band | y [m] | p_ref [Pa] | p_SPH [Pa] | err [%] | gated |
|---|---|---|---|---|---|
| 0 | 0.0243 | 3685.73 | 3697.89 | +0.3298 | yes |
| 1 | 0.0741 | 3197.29 | 3219.48 | +0.6939 | yes |
| 2 | 0.1252 | 2695.88 | 2715.94 | +0.7440 | yes |
| 3 | 0.1748 | 2208.73 | 2219.48 | +0.4867 | yes |
| 4 | 0.2248 | 1718.25 | 1723.41 | +0.3002 | yes |
| 5 | 0.2747 | 1229.34 | 1227.10 | -0.1828 | yes |
| 6 | 0.3224 | 761.33 | 751.70 | -1.2649 | no (free surface) |
| 7 | 0.3705 | 289.04 | 269.23 | -6.8530 | no (free surface) |


**变体（同测量、单变量）**：

| config | N | 窗 [s] | max band err [%] | 完整支撑 err [%] | 判定 |
|---|---|---|---|---|---|
| baseline_dx002 | 1000 | 1.00 | 1.0755 | 1.0824 | FAIL |
| gate_dx00125 | 2560 | 1.00 | 1.5052 | 1.6141 | FAIL |
| width_w15 | 1500 | 1.00 | 1.0489 | 1.0569 | FAIL |
| sound_c30 | 1000 | 1.00 | 1.4934 | 1.5220 | FAIL |
| sound_c40 | 1000 | 1.00 | 1.6386 | 1.9553 | FAIL |
| window_4s | 1000 | 2.00 | 0.6889 | 0.6653 | PASS |
| window_8s | 1000 | 4.00 | 0.7440 | 0.8145 | PASS |
| window_dense | 1000 | 1.00 | 1.0714 | 1.0785 | FAIL |
| control_g0 | 1000 | 1.00 | 100.0000 | 100.0000 | FAIL |


**分辨率历史（含聚合后补跑的 ladder_dx001）**：

| Nx | max interior band err [%] | Nx*err |
|---|---|---|
| 50 | 1.0755 | 53.78 |
| 80 | 1.5052 | 120.42 |
| 100 | 1.6460 | 164.60 |


三者乘积近似常数（53.8, 120.4, 164.6）——静态/内部误差按
`1/Nx` 壁列份额收缩；而**动态**误差随加密上升（1.076 → 1.505 → 1.646 %），
说明动态残差不是离散误差而是时间平均采样地板（与算例 §5 的结论一致）。

**审计独立探针（dx=0.025 m, Nx=40）**：

| probe band | y [m] | p [Pa] | p_ref [Pa] | err [%] |
|---|---|---|---|---|
| 0 | 0.0250 | 3129.26 | 3678.79 | -14.9378 |
| 1 | 0.0750 | 3148.06 | 3188.29 | -1.2615 |
| 2 | 0.1250 | 2663.74 | 2697.79 | -1.2621 |
| 3 | 0.1750 | 2179.52 | 2207.29 | -1.2578 |
| 4 | 0.2250 | 1695.12 | 1716.79 | -1.2618 |
| 5 | 0.2750 | 1210.88 | 1226.29 | -1.2561 |
| 6 | 0.3125 | 847.42 | 858.38 | -1.2764 |


- 完整支撑列最大误差 **0.0384 %**（算子对线性静水场近乎精确）
- 侧壁列均值误差 **-10.23 %**（占粒子
  12.2 %）
- 整箱内部最大误差 **1.2764 %**，与算例同一 `1/Nx` 律吻合
  （Nx*err ≈ 51.1 vs 算例 Nx=50 的
  50.1）

**复现性**：

- `--aggregate` 在 m15320 拷贝上重跑：9/9 共有配置数值**逐位相同**，
  gate 值 0.743956 % 相同；窗口数由 9 变 10（ladder_dx001 已在盘上）。
- 动态点实跑复现（baseline_dx002）：工件 1.0755191398 %
  vs 实跑 1.0755191398 %，
  **除 runtime_s 外逐字段相同**。
- 静态点实跑复现：11/11 配置的
  `max_band_error_pct` / 完整支撑口径与工件逐位相同（例：baseline 10.2161 % / 9.3003 %）。

## ④ 结果图

左：静水压剖面（算例 Nx=50 与审计探针 Nx=40 的 SPH 读数 vs 闭式解，灰带=自由面排除带）。
右：双对数下静态内部误差随 Nx 沿 `C/Nx` 下降，而动态（时间平均）误差反向上升。

## ⑤ 注意点

**F1 [medium/process/registry] anchor 未登记进基准裁判表：tensor_benchmark_cases 中 CASE-HYDROSTATIC 0 行**

- 证据：SELECT ... WHERE anchor_slug='CASE-HYDROSTATIC' -> 0 rows（表内仅 14 行，覆盖 GAP-*/XVAL-*/H*/CAP-* 等其它锚点）
- 影响：机器门无容差行可对；本轮容差只能取 mission goal 的 <1%，got 由审计独立复算（0.743956%）。与 m15164 审计（verdict id=267）同类缺口，非本单造假。
- 返修项：把该算例登记进 tensor_benchmark_cases（ref_source=解析 p=rho0 g(H-y)，metric=max_band_error_pct，tolerance=<1%），后续 run 才可机器裁。

**F2 [medium/artifact staleness] report.json 滞后于 runs/：ladder_dx001 在聚合后 5.5 min 才落盘，报告未重生成**

- 证据：runs/ladder_dx001.json mtime 21:52:37 vs report.json mtime 21:47:06；docs/case_hydrostatic_report.md ⑥ 仍写『ladder_dx001 未跑完』，而磁盘上该动态点已完整（max_band_error_pct=1.646%）
- 影响：report.md/report.json 与 runs/ 不同步；审计重跑 --aggregate（不改 runs 数值）后窗口数从 9 变 10，ladder_fit 从 2 点变 3 点。gate 数值不变（0.743956%），故不影响主判据，但『报告=工件』的承诺出现缺口。
- 返修项：重跑 --aggregate --figure 并把 ⑥『未完成』一段改为已完成（1.646%，三点单调升，坐实『动态误差不随加密下降』）；顺带把 ladder_fit 的 `a/Nx+b` 模型标注改为『经验描述，非收敛律』。

**F3 [low/claim vs artifact] 『瞬时 17.6%』与工件不符（工件 17.923%）**

- 证据：runs/baseline_dx002.json measured.max_band_error_pct_instantaneous=17.9234
- 影响：单一数字手抄偏差 0.3pp，违反『零手抄』纪律；不改变任何判定。
- 返修项：由工件字段直接格式化输出该数。

**F4 [low/CLI defect] run.py `--run NAME` 静默忽略 `--static`**

- 证据：run.py:496-505：`if args.run:` 分支先 return，`--static` 永不生效；审计实测 `--static --run baseline_dx002` 跑的是动态点（61 s，输出 1.076%）
- 影响：使用者以为在做静态对照，实际耗时数十倍且得到另一口径的数；不污染已提交工件（docstring 的一键命令不带 --run，仍正确）。
- 返修项：`--run` 与 `--static` 组合时报错，或把 args 按 (static, run) 分派。

**F5 [medium/integration/unmerged] 交付只在分支 agent/m15264（36da0f2），main 缺 case 且缺依赖的 src 扩展**

- 证据：m15320 检出 b64b5f4 无 benchmarks/case_hydrostatic；main 的 vv.hydrostatic.run_hydrostatic 无 return_fields、verification.hydrostatic.HydrostaticCase.__init__ 无 sample_every（审计实测 signature 均 False）；case 动态路径 run.py:193 传 return_fields=True
- 影响：repro_cmd 在 main 上不可执行（TypeError），需先取分支；CI tests.yml 的 `pytest -q` 只有合并后才会收集 tests/test_case_hydrostatic.py。
- 返修项：合并 36da0f2（含 src 两处最小扩展），合并后重跑 repro_cmd 作为验收。

**F6 [low/test brittleness] test_report_quotes_the_run_files 连 runtime_s 一起断言，任何合法重跑都会打红**

- 证据：审计把第 1 个动态点重跑（数值逐字段相同，仅 runtime_s 99.6735 s -> 61.3626 s）后该测试报 AssertionError: {'runtime_s': 99.67354202270508} != {'runtime_s': 61.36256790161133}；重跑 --aggregate 后 8 passed 恢复
- 影响：把墙钟时间当不变量：合法重跑即打红，易被误读为数据造假；真正的护值是物理字段
- 返修项：断言前剔除 runtime_s（或字段白名单）；聚合报告不再引用 runtime_s


**其它口径说明**

- 判据是**时间平均量**的：单帧剖面瞬时误差在出厂配置下达 17.92 %（算例正文写 17.6 %，见 F3），
  不能用于单帧评价。
- `band 0`（贴底行）在动态读 +0.33 %、静态读 −10.2 %：动态里被晃荡平均掩盖，
  静态口径下它确实是壁致缺陷，band 0 的归类是仓内度量的已知弱点（算例 §6 自陈）。
- g 用求解器声明值 9.81（标准重力 9.80665，偏高 0.035 %），对全部 band 同倍，
  不改变误差结论。
- 机制预测 (2/Nx)*|壁列误差| 只解释部分量级（算例静态 ratio 0.768 判 PASS；
  审计探针 ratio 2.49，因受影响列实为每侧
  2 列而非 1 列）——该"解释"是量级自洽，不是定量律。

## ⑥ 误差判定

| 关键量 | ref | got | 误差 | 是否 < 3 % |
|---|---|---|---|---|
| 内部 band 最大压强误差（门控 window_8s） | p=rho0 g(H-y) | 0.743956 % | 0.744 % | **是（且 < 1 %）** |
| 同上，审计独立重算 | 同上 | 0.743956 % | 0 % （与工件一致） | 是 |
| 完整支撑列误差 | 同上 | 0.8145 % | 0.815 % | 是 |
| 审计探针（Nx=40，非算例网格） | 同上 | 1.2764 % | 1.276 % | 是 |
| 审计探针完整支撑列 | 同上 | 0.0384 % | 0.038 % | 是 |
| 负对照 g=0（应 ≈100 %） | 100 % | 99.9999 % | 0 % | 是 |
| 自由面 band（定义上排除） | 同上 | −1.26 % / −6.85 % | 6.85 % | 否，按定义排除（核支撑截断） |

**结论**：主判据与全部可评项均远优于 3 %；唯一超 3 % 的自由面 band 是定义性排除项。

## 判级与处置

- **判级：L4（基准验证）** —— 完整算例 + 逐 band 误差分析 + 收敛/敏感性 + 审计独立复现
  （聚合复算逐位一致、动态点实跑除 runtime 外逐字段一致、静态全套实跑一致、
  另加一个算例网格外的独立分辨率点）。
- **但不满足 L4 的"已并入主线"隐含前提**：交付只在分支 `agent/m15264`，main 缺 case 且缺
  依赖的 src 扩展（F5）；锚点未登记进 `tensor_benchmark_cases`（F1，0 行）。
  若机器门要求"锚点已登记 + 分支已并入 main"，本判级应下调为 **L3**。
- **处置：approve with notes**（返修 3 项：F1 登记锚点、F2 重生成报告纳入 ladder_dx001、
  F5 合并分支；F3/F4 顺手修）。

## 附件

- `artifacts/audit_m15320/hydrostatic_probe.json`（独立探针原始数据）
- `artifacts/audit_m15320/repro_aggregate.json`（m15320 拷贝上的 --aggregate 复算）
- `artifacts/audit_m15320/live_rerun.json`（动态点/静态点实跑复现记录）
- `artifacts/audit_m15320/pytest_case.log`（8 passed, 1 warning in 9.58s）
- `artifacts/audit_m15320/evidence.json`（三件套核验矩阵 + 发现）
- `scripts/audit/m15320_hydrostatic_probe.py`（探针，可复跑）
