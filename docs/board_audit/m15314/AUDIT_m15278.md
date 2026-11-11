# 交差审计：m15278 — CASE-LAKE-AT-REST-SPH（TensorSPH 静水湖）

`verdict_by=board-audit` · 审计单 m15314 · 被审单 m15278（succeeded, task_key=case-lake-at-rest-sph-r2, claimed_by=tp-ds-7）

## 0. 结论先行（disposition: request changes）

* **级别 = L3（系统集成），不授 L4**：功能实质是真的（well-balanced 压力力写进生产 RHS `droplet_impact.rhs`，被 4 个求解器入口共用），但算例池判据 **压力漂移 < 0.5 %** **未达标**——最好一例 `wb dx=20mm c0=20 nu=0.01` = **3.887 %**，超判据 **7.77×**；`shipped` 模式最大 **45.367 %**（超 90.7×）。
* 被审单自报的正是 honest FAIL（`criterion_met=false`），本审计**逐位复现**了它：7/7 算例漂移与原工件**逐位相同**（max |Δ| = 0.0e+00），复跑耗时 ~22 s，命令即交付的 repro_cmd。
* 交付三件套实体存在（工件/数值/复现命令全部可核），**但一件都没进 git**：`head_commit=null`，`git ls-files` 对这些路径为空，改动只存在于工作树（未 track）。→ 产品（TensorSPH main）**没有** `equilibrium_level`（生态仓 grep 0 命中）；审计已把 patch 与工件逐字节快照进本分支 `docs/board_audit/m15314/refs/` 作保命。
* 基准裁判**空转**：`tensor_benchmark_cases` 对 `anchor_slug='CASE-LAKE-AT-REST-SPH'` **返回 0 行**（全表 14 行、无一含 LAKE）→ 机器裁判不存在，唯一可援引判据是算例池 `tensor_case_pool#67 tol_policy='压力漂移<0.5%'`（且该行 `status='dispatched'` 仍挂在失败前轮 m15241 上，未随 m15278 更新）。

## 1. 取证范围与独立性边界

| 项 | 值 |
|---|---|
| 被审工作树 | `/root/tp-agent/work/m15278`（base `ae61a2b`，`dirty=true`） |
| 交付工件 | `benchmarks/lake_at_rest/{run.py,report.json,make_docs.py}`, `docs/verification_lake_at_rest.md`, `docs/figures/lake_at_rest_{drift,profile}.png`, `tests/test_lake_at_rest.py` |
| 审计快照（逐字节 + sha256） | `docs/board_audit/m15314/refs/` + `refs/artifacts/droplet_impact_wellbal.patch` |
| 平台报告 | `tensor_reports#14367`（6 段齐全，附 2 图，图可取回 200/58 616 B、45 243 B PNG） |
| 独立性边界 | 同一物理 CPU、同一 torch 2.7.1/py3.11.6；复跑用交付自带的 `run.py`（未改一行）→ 验证的是**确定性与数值可重放**，不是独立实现比对；t=0 探针 `probe_t0.py` 是审计自写，独立于 run.py 的聚合逻辑 |

## 2. 三件套核验

| 三件套 | 结论 | 证据 |
|---|---|---|
| run_artifacts 文件 | **PASS（在盘）** | `report.json` 36 455 B：7 算例 × 16 采样点，含 `pressure_drift_max/mean`、`v_max`、`rho_rel_error_max`、`acc_max_at_rest`；判据字段 `acceptance_tolerance=0.005` |
| benchmark ref vs got | **逐项核，全部 FAIL** | ref = 精确解 `p=rho0 g (H-y), u=0`；got 见表 §3；`criterion_met=False`, `wb_criterion_met=False` |
| repro_cmd 可执行 | **PASS** | `PYTHONPATH=src python3 benchmarks/lake_at_rest/run.py` 审计实跑：7/7 算例漂移逐位相同（max |Δ| = 0.0e+00） |
| 单测 | **PASS** | `PYTHONPATH=src pytest tests/test_lake_at_rest.py -q` → **3 passed in 7.81 s**（审计实跑） |
| 进仓（git） | **FAIL** | `head_commit: null`；`git ls-files benchmarks/lake_at_rest …` 空；`git log --all -- benchmarks/lake_at_rest docs/verification_lake_at_rest.md` 空 |
| 基准登记（裁判） | **FAIL（系统级）** | `tensor_benchmark_cases` 该锚点 0 行 |

## 3. 基准判据逐项（唯一裁判 = 算例池 `压力漂移<0.5%`）

ref = 0（精确静水解，无实验不确定度）；误差定义 `max_{内区粒子,采样} |p−rho0 g (H−y)| / (rho0 g H)`，内区 = 距壁/自由面 ≥ 3h = 78 mm。

| # | mode | dx [mm] | c0 [m/s] | nu [m²/s] | 粒子/内区 | drift max | **< 0.5 %?** | 超判据倍数 |
|---|---|---|---|---|---|---|---|---|
| 1 | shipped | 25 | 20 | 0.010 | 560/192 | 32.002 % | FAIL | 64.0× |
| 2 | wb | 25 | 20 | 0.010 | 560/192 | 3.997 % | FAIL | 8.0× |
| 3 | shipped | 20 | 20 | 0.010 | 850/420 | 26.290 % | FAIL | 52.6× |
| 4 | wb | 20 | 20 | 0.010 | 850/420 | 3.887 % | FAIL | 7.8× |
| 5 | wb | 20 | 40 | 0.010 | 850/420 | 9.464 % | FAIL | 18.9× |
| 6 | shipped | 20 | 40 | 0.010 | 850/420 | 45.367 % | FAIL | 90.7× |
| 7 | wb | 20 | 20 | 0.000 | 850/420 | 7.758 % | FAIL | 15.5× |

* `shipped`（仓默认初始化）漂移 45.367 %，且**加密不收敛**（25 → 20 mm：32.002 % → 26.290 %）→ 缺陷在边界离散，不在插值。
* `wb`（well-balanced）最好 3.887 %（dx=20 mm，c0=20），最大 9.464 %（ν=0）→ 改善 **6.76×**，但距判据仍差 **7.77×**。

## 4. 独立 t=0 探针（审计自算，核报告 §4.2 的缺陷定位数字）

| 量 | shipped | wb | 报告自报 | 判定 |
|---|---|---|---|---|
| 内区 t=0 偏差 max | 0.01185 % | 0.01185 % | 0.012 % | **一致** |
| 内区 t=0 偏差 mean | 0.00478 % | 0.00478 % | — | 补充 |
| 底行(t=0) 偏差 max | 42.39 % | 0.009 % | −24.8 %（单列切片） | 同量级；审计给出整行 max 42.39 %，比切片更差 |
| max\|a\| (t=0) [m/s²] | 19.04 | 4.65 | 19.04 / 4.65 | **逐位一致**（1.94 g / 0.47 g） |

## 5. 发现（按严重度排序）

**B1（阻断，判级）判据未达标 → 不授 L4。** 最好 wb 3.887 % vs 0.5 % = 7.77× 超标；L4「基准验证」要求被验证能力通过基准，本算例的诚实结论是 FAIL（交付方也如此声明，无粉饰）。

**B2（阻断，交付完整性）全部交付物未进 git。** `head_commit=null`、路径 `untracked`、生态仓 `grep equilibrium_level src/` 0 命中 → 产品侧不存在该能力；报告 §6 写「commit `ae61a2b2ce7a`」实为**基线提交**（工作树 base），构成 provenance 误导。审计已把 diff（154 行 patch）与全部工件快照进本分支，避免工作树回收即丢。

**B3（系统级，非本单可控）基准裁判空转。** `SELECT … FROM tensor_benchmark_cases WHERE anchor_slug='CASE-LAKE-AT-REST-SPH'` → 0 行；算例池 `#67` 仍 `status='dispatched'`、`mission_id=15241`（已 failed 的前轮），未随 m15278 更新。与 m15240（F2）、m15235（F1_referee_missing）审计交叉一致：**算例池锚点普遍未登记判据**，「基准是唯一裁判」这条纪律目前**无机器裁判可依**，只能援引 tol_policy 文本。

**C1（中）判据与算式的自指耦合。** wb 算子把解析静水压 `p_h=rho0 g (H−y)`（H 由用户经 `Config.equilibrium_level` 传入）减出后积分，drift 又用**同一** `p_h` 度量 → 3.887 % 是 well-balanced 性质的度量，**不能**当作一般自由面精度证据；且要求先验已知平衡水位，对真实未知水面问题不可直接使用（交付文档 §2 已如实说明出处，但报告标题的 6.8× 易被读成普遍精度提升）。

**C2（中）默认模式仍是缺陷态。** `equilibrium_level=None` 时走原路径，底行压强亏空 42.4 %、虚假加速度 1.94 g、漂移 26.29 % 全部保留 → 修的是新开关，不是产品默认行为；该项须作为缺陷单跟进（壁面质量按静水剖面是通用修法，可无条件生效，未必需要新开关）。

**D1（低）报告证据链小瑕。** `tensor_reports#14367.figures[].png_base64` 为空串，但 `/tensor/reports/14367/figures/<name>` 取回 200 + 真实 PNG → 图**在**，只是行内未内嵌；不计缺陷，登记以免下游误判为无图。

## 6. 未跑的检查（unknown / not_run，不得当作 passed）

* 与**独立实现**的横比（ISPH/DFSPH/其它开源 SPH 的同一静水算例）——未做，本审计只做同码复现。
* 算例池判据 0.5 % 的**可达性**审查（WCSPH 在 c0/√(gH)=10.79 下压强漂移 0.5 % 等价于密度一致到 ~4e-5，是否该由 ISPH/更高 c0 承担）：只做了量纲推断，未做可达性数值实验。
* 三维/长时（>0.05 s）/更大规模网格上的 wb 行为：未跑。
* `make_docs.py` 的图管线本身未逐行审（只核 report.json → md 的数字一致性）。

## 7. 复现命令（审计实跑过的）

```
cd /root/tp-agent/work/m15278
PYTHONPATH=src python3 benchmarks/lake_at_rest/run.py --out /tmp/m15278_rerun.json   # 7/7 逐位复现
PYTHONPATH=src pytest tests/test_lake_at_rest.py -q                                  # 3 passed
PYTHONPATH=src:benchmarks/lake_at_rest python3 <审计>/probe_t0.py --out probe_t0.json  # t=0 探针
python3 docs/board_audit/m15314/render_audit.py                                      # 生成本文+两图
```

图：`refs/artifacts/figures/audit_drift_vs_criterion.png`（7 算例 vs 0.5 % 判据，log 轴）、`audit_drift_vs_time.png`（dx=20 mm/c0=20 漂移史）。

---

**落表**：`tensor_maturity_verdicts#276`（repo=TensorSPH, scope=`m15278 CASE-LAKE-AT-REST-SPH`, mission_id=15314, level=3, verdict_by=board-audit）；平台报告 `tensor_reports#14397`。
场图：`refs/artifacts/figures/audit_pressure_field_t0.png`（审计自算 t=0 压力偏差场，shipped 底行最大 42.4 
---

**落表**：`tensor_maturity_verdicts#276`（repo=TensorSPH, scope=`m15278 CASE-LAKE-AT-REST-SPH`, mission_id=15314, level=3, verdict_by=board-audit）；平台报告 `tensor_reports#14397`。
场图：`refs/artifacts/figures/audit_pressure_field_t0.png`（审计自算 t=0 压力偏差场：shipped 底行最大 42.4 %，wb 降至 0.009 %）。
