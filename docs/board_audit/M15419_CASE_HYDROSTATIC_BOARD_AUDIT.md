# M15419 交差核验报告：CASE-HYDROSTATIC（被审 m15340，TensorSPH）

- 审计任务 m15419（board-audit）｜被审任务 m15340｜锚点 `CASE-HYDROSTATIC`
- 被审提交 TensorSPH `4004c02ba738`（"m15340: matured mission deliverables"）
- 生成时间 2026-11-11T23:42:46+00:00；本文件全部数字由 `scripts/audit/m15419_case_hydrostatic_audit.py` 程序化注入

## 1 三件套核验

| 件 | 指针 | 结论 |
|---|---|---|
| run_artifacts | `benchmarks/verified/sph_hydrostatic/results.json` | ✓ tracked，8 带 ref/got + gate + control |
| run_artifacts | `benchmarks/verified/sph_hydrostatic/evidence.sqlite` | tracked 但**空库**（0 runs / 0 metrics；results.json 引 run_id=3/4 → 悬空） |
| benchmark | ref `p(y)=ρ0 g (H−y)`，H=0.4 m；got 内区 max 1.0635%（限 2%） | ✓ 本单独立重算见 §2 |
| repro_cmd | `cd <fresh git archive of 4004c02ba738> && PYTHONPATH=src python benchmarks/verified/sph_hydrostatic/run.py --dx 0.02` | ✓ 已复跑：`N=1000 max band error 1.064 % (mean 0.602 %, surface band -3.6 %), gate=PASS, rho0=0 control PASS
EXIT=0` |
| 附加工件 | `artifacts/hydrostatic/*`（full_report.json / MANIFEST / figure） | ✗ **未入版本库**（`.gitignore` 排除 `artifacts/`）→ 推送后指针失效 |

### 基准裁判表（唯一裁判）

`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-HYDROSTATIC'` → **0 行**。全表 16 行；`repo ILIKE '%SPH%'` 命中 0 个锚点。即本锚点**未登记**，2% 容差来自被审单自身（`run.py:64 --limit-pct` 默认值）。

## 2 独立重算：解析参考与门控规则

| 带 | y (m) | N | p_SPH (Pa) | p_ref 重算 (Pa) | 记录差 (Pa) | 误差重算 (%) | 误差记录 (%) | 差 (pp) | 类别 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0273 | 134 | 3635.921 | 3656.131 | +0.0e+00 | -0.5528 | -0.5528 | +0.0e+00 | interior |
| 1 | 0.0758 | 116 | 3167.773 | 3180.225 | +0.0e+00 | -0.3916 | -0.3916 | +0.0e+00 | interior |
| 2 | 0.1279 | 138 | 2642.052 | 2669.396 | +0.0e+00 | -1.0243 | -1.0243 | +0.0e+00 | interior |
| 3 | 0.1767 | 112 | 2167.729 | 2191.031 | +0.0e+00 | -1.0635 | -1.0635 | +0.0e+00 | interior |
| 4 | 0.2276 | 136 | 1695.708 | 1691.646 | +0.0e+00 | +0.2401 | +0.2401 | +0.0e+00 | interior |
| 5 | 0.2761 | 114 | 1211.707 | 1215.816 | +0.0e+00 | -0.3380 | -0.3380 | +0.0e+00 | interior |
| 6 | 0.3270 | 134 | 674.669 | 715.658 | +0.0e+00 | -5.7275 | -5.7275 | +0.0e+00 | free-surface |
| 7 | 0.3750 | 116 | 236.569 | 245.314 | +0.0e+00 | -3.5649 | -3.5649 | +0.0e+00 | free-surface |

- 内区 max = **1.0635%**（记录 1.0635%，差 +0.0e+00 pp）→ 门限 2% **通过**，且远小于生态 15% 不可接受线。
- 内区 mean = 0.6017%；**全带 max = 5.7275%**（被排除的 6–7 带读 [-5.73, -3.56]%）。
- 门控规则复算：support = 2·1.3·dx = 0.0520 m（深度的 13.0%），内区 = 带 [0, 1, 2, 3, 4, 5]（6/8 带，覆盖深度 87.0%），与工件 `gated_bands` 一致 → 规则实现与声明一致。
- 瞬时（未时均）内区 max = 17.61% → 时均确实抹平声学振荡。

## 3 复现性（全新 git archive 检出）

- `git archive 4004c02ba738` 到干净目录后按 repro_cmd 复跑：gate 1.0635295646% → 2.2490536793%（Δ=1.2e+00），8 带压力逐带最大绝对差 1.2e+02 Pa，config 完全一致（False）→ **逐位可复现**。

## 4 审计方新增探针（同一份提交代码，被审单未做）

### 4.1 t=0 静态算子误差

初始格子一次 RHS 求值（无动力学）：内区 max 10.216%、mean 2.318%；逐带 ['b0:-10.22%', 'b1:-1.01%', 'b2:-1.00%', 'b3:-1.00%', 'b4:-1.00%', 'b5:-1.00%', 'b6:-1.00%', 'b7:-45.01%']。

→ 内区各带在 t=0 已一致读低约 1.00%%（求和密度/核归一化的系统偏差），而壁邻带 0 在 t=0 读 −10.2%%、跑满 2 s 后回到 −0.55%%：**动力学修的是壁区，内区 ~1%% 是静态算子偏差**，非时间积分误差。发布单未区分这两者。

### 4.2 负对照（gravity = 0）

`run_hydrostatic_case(case, gravity_override=0.0)`（dx=0.05, 1 s）：内区 max = 100.00%，gate pass = False → **门会失败**，非构造性通过。

### 4.3 分辨率扫描（同一几何）

| 配置 | N | 内区带数 | 内区 max (%) | 内区 mean (%) | H 未对齐偏差 (%) | gate |
|---|---|---|---|---|---|---|
| dx=0.04 depth=0.40 | 250 | 5 | 1.282 | 0.558 | 0.00 | PASS |
| dx=0.03 depth=0.40 | 429 | 6 | 9.297 | 5.485 | 2.50 | FAIL |
| dx=0.03 depth=0.39 | 429 | 6 | 2.249 | 1.011 | 0.00 | FAIL |
| dx=0.02 depth=0.40 (committed) | 1000 | 6 | 1.064 | 0.602 | 0.00 | PASS |

→ **误差对 dx 不单调**：dx=0.04 1.28%、dx=0.03 9.30%、dx=0.02 1.06%。原因在几何不在物理：`cell_centres()` 取 `floor(extent/dx)`（droplet_impact.py:534），dx=0.03 时 0.4 m 只铺 13 层 = 0.39 m，而参考高度仍取 `cfg.depth=0.4`，于是每带系统性读低 1−0.39/0.40 = 2.56%。把 depth 改成 0.39（格子正好铺满）后同一 dx 的误差 9.30% → 2.25%，与 dx=0.02/0.04 同量级 → **dx=0.03 的 FAIL 是参考高度与格子不匹配造成的假失败**，不是求解器缺陷；提交的 dx=0.02（50×20=1000，深度恰好 0.40）不受影响。

## 5 发布单声明核验（对照 docs/hydrostatic_benchmark_report.md）

- **C1_convergence_O_dx**：report §5 'error decreases with resolution ... scales ~ O(dx)' from two rows (dx=0.05 N=16 0.2 s tank 0.2 m; dx=0.02 N=1000 2 s tank 1.0 m) → 实测 ['dx=0.04 depth=0.40: 1.282% (N=250)', 'dx=0.03 depth=0.40: 9.297% (N=429)', 'dx=0.03 depth=0.39: 2.249% (N=429)', 'dx=0.02 depth=0.40 (committed): 1.064% (N=1000)'] → **falsified: the two rows are different cases (tank/depth/duration); at fixed geometry the error is not monotone in dx**
- **C2_coarse_dx_fails**：report §6 failure mode 4: 'with dx > 0.03 the error exceeds the 2% gate' → 实测 dx=0.04 depth=0.40 N=250 2 s -> 1.282% (below the 2% gate) → **not supported as stated**
- **C3_evidence_store**：report §6 artifacts table: evidence.sqlite = 'Evidence store' → 实测 committed evidence.sqlite: 0 runs, 0 metrics; results.json cites run_id 3/4 → **empty as committed -> the cited run_id does not resolve**
- **C4_gate_coverage**：report §4 'the free-surface band is excluded from the gate' → 实测 interior = bands [0, 1, 2, 3, 4, 5] = 87.0% of the depth (support 2h = 0.052 m); excluded bands read [-5.73, -3.56]% → **stated correctly; recorded because the gate covers the interior only**

## 6 定级（L0–L5）

| 级别 | 判据 | 结论 |
|---|---|---|
| L1 数学/单元验证 | 解析解 + 单元级校验 | ✓ |
| L2 独立功能实现 | 真代码：格子/质量 → SPH rhs + Tait EOS + Adami 壁面 → 带表/门 | ✓ `src/tensorsph/vv/hydrostatic.py:36-227`、`src/tensorsph/verification/hydrostatic.py:76-176` |
| L3 系统集成 | 接入主求解器 / CLI | ✓ 走主求解器 `droplet_impact.rhs`；`src/tensorsph/verification/cli.py:49,453-470` 注册 M2 梯级；`benchmarks/verified/sph_convergence/run.py:34` 复用 |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | ✓ 解析基准 1.06%（<2%）、逐位复现、误差表/图齐；**但**裁判表无登记行、§5 收敛声明被证伪、evidence.sqlite 提交态为空 |
| L5 科研工程成果 | 研究成果 / 工程交付 | ✗ 验证梯级，非科研工程成果 |

**判定：L4**（功能实质达 L4；缺陷见 §7，不改变算例本身的有效性与可复现性）。

## 7 Findings（按严重度）

| 编号 | 严重度 | 结论 | 证据 |
|---|---|---|---|
| F1 | medium | 裁判缺失：`tensor_benchmark_cases` 无 `CASE-HYDROSTATIC` 行，2% 容差为被审单自定；同锚点 m15264 case 亦未登记（m15320 审计 F1 至今未修） | §1：0/16 行 |
| F2 | medium | 发布单 §5「误差随分辨率下降、约 O(dx)」不成立：其两行来自不同几何/时长（0.2 m 水箱 0.2 s N=16 vs 1.0 m 水箱 2 s N=1000）；同几何扫描非单调 | §4.3 |
| F3 | medium | 提交态 `evidence.sqlite` 为空库（0 runs/0 metrics），而 results.json 引 run_id=3/4 → 证据链断在库里 | §1 |
| F4 | medium | 门控敏感于「参考 H 是否与格子对齐」：dx 不整除 depth 时假失败（dx=0.03 → 9.30%，对齐后 2.25%）。建议在 `HydrostaticCase` 加 `assert abs(depth/dx - round(depth/dx)) < 1e-9` 或把 H 取实际格子顶面（本单未改被审代码） | §4.3 |
| F5 | low | 发布单 §6 失效模式 4「dx>0.03 误差超 2%」不成立：dx=0.04 同几何实测 1.282% | §4.3 |
| F6 | low | MANIFEST/§6 引用的 `artifacts/hydrostatic/*` 被 `.gitignore` 排除，推送后指针失效（数值等价物在 tracked 的 results.json 中） | §1 |
| F7 | low | 内区 ~1.00% 在 t=0 即存在（静态算子偏差），报告把其表述为时均过程的产物，未区分静态偏差与动力学校正 | §4.1 |

## 8 未做的检查（honest）

- 未跑 GPU 路径与 >10^4 粒子；未做第三方 SPH 求解器（OpenFOAM/DualSPHysics）交叉验证；
- 未审计 `benchmarks/hydrostatic_consistency`（DW-1）与本锚点的关系（另一审计面）；
- 未核该交付是否已 merge 进 main（只按被审工作树 HEAD = 交付 commit 取证）。

## 9 结论（disposition）

**approve with findings**：run_artifacts(结果表) 与 repro_cmd 真实可用、解析基准 1.06%（限 2%）、逐位可复现、门可失败（g=0 → 100%），L4 功能实质成立；但 F1–F4 需返修（登记裁判行、改正 §5 收敛声明、修提交态 evidence.sqlite、加 depth/dx 对齐守卫）。
