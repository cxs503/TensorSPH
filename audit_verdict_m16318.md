# 交差核验：m16244 CASE-SPH-RAYLEIGH-TAYLOR-MP（TensorSPH，多相 RT 线性增长率）

**审计任务**: m16318（board-audit）
**被审任务**: m16244（TensorSPH，状态 succeeded，PR #543 merged，head=`852e941`）
**审计日期**: 2026-11-12
**成熟度判定**: **L3**（落表 id=464，gate_state=pending→自动裁定）
**机器裁判**: `GET /api/v1/tensor/bench/l4-verdict/16244` → `status=CLAIMED_NOT_PROVEN`, `l4_benchmark_pass=false`, `rows=0`
**处置**: **request changes（返修）**——见 §6 返修清单；L2/L3 部分为真实实现，不需重做。

---

## ① 取证：三件套逐项核验

| 三件套 | 结论 | 证据 |
|---|---|---|
| run_artifacts | **真实存在** | `benchmarks/rayleigh_taylor/runs/` 8 个算例 JSON + `reference.json`/`table.json`/`damping.json`/`spurious.json`；每个含 config/derived(粒子数、步数、dt)/fit(R²)/series(t,η) /fields(rho,p,φ) /wall_seconds |
| benchmark ref vs got | **存在但超容差** | ref=√(Agk)=5.551488 s⁻¹；best got=4.1923 s⁻¹ @dx=0.0125 → **-24.48 %**，判据 **<10 %** → FAIL；4/4 完成档全 FAIL；`docs/benchmarks/rayleigh-taylor-v2.md` ④ 表 3 行仍为「集群计算中/待填/待判」 |
| repro_cmd | **可执行（部分核验）** | `PYTHONPATH=src python benchmarks/rayleigh_taylor/run.py --grid 0.02 --time-integrator rk2 --density-diffusion 0.05`：CLI 解析通过、RK2+δ-SPH 分支确实在跑（本地 99 s 推进到 t=0.175/0.45，被 120 s shell 上限收割）；全时长复跑已递交集群 **cj1301（排队中）**，审计窗口内 not_run |

### 基准判据溯源（唯一裁判）

- `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-SPH-RAYLEIGH-TAYLOR-MP'` → **0 行**（全表仅 30 行，无任何 RT 条目）。即本锚点在基准登记表中**没有登记规格**，容差只能取自任务书（<10 %，指标=早期指数增长率 vs √(Agk)）。
- `SELECT count(*) FROM tensorbench_results WHERE mission_id=16244` → **0**。平台侧没有任何被机器重算的 ref/got 行，故 L4 只能判 `CLAIMED_NOT_PROVEN`。
- `SELECT count(*) FROM tensor_reports WHERE mission_id=16244` → **0**：本单内容未发 `tensor_reports` 报告（报告纪律缺口；六段只存在于仓内 md，且 ④ 未填完）。

### 主张 vs 实测（adversarial checks）

| 被审主张 | 审计实测 | 结论 |
|---|---|---|
| "10 passed" | 审计方重跑 `pytest tests/test_rayleigh_taylor.py -q` → **10 passed in 60.97 s** | ✅ 属实 |
| "RK2 + δ-SPH 已实现" | `src/tensorsph/rayleigh_taylor.py:131-134,468-493,742-753` 真实代码，非桩 | ✅ 属实 |
| "σ_num 数值可信" | 审计方**独立**从 `series.t/amplitude` 重做最小二乘拟合（不复用存储的 fit）→ 4.1923 / 3.0952 / 3.9713 / 2.1252 **逐位重合**，R²≥0.998 | ✅ 工件非伪造 |
| "ref √(Agk)=5.5515" | 独立重算 A·g·k=0.5×9.81×2π，√=5.5514884429，与 `runs/reference.json` 相对差 **0.0** | ✅ 锚点正确 |
| "返修轮改进后接近 <10 %" | RK2/δ-SPH 一行数据都没有（cj1287 审计时仍 `queued`，已排队 29 min）；改进未被测量 | ❌ 未验证（not_run） |
| "工件齐全可复现" | `runs/summary.json` **只含 1/8 条**记录（`run.py` 每次运行重写 summary，末次即全量） | ⚠️ 聚合件失真 |
| 图件 | RT 算例目录下**无任何 png**（`fields` 已存，可出图但未出） | ⚠️ 缺图 |

---

## ② 判级（L0-L5）

| 级别 | 要求 | 判定 | 依据 |
|---|---|---|---|
| L0 文档 | 文档成果 | ✅ | `docs/benchmarks/rayleigh-taylor.md`（v1）+ `-v2.md` |
| L1 数学/单元验证 | 单元+解析自洽 | ✅ | 10 项测试含：数密度=lattice 和、静水零净加速度、界面压强连续、动量守恒、量计对已知界面 5 % 回收、理论极限（有限深/黏性）、拟合 1e-6 回收指数、A=0 不增长 |
| L2 独立功能实现 | 真代码 输入→计算→输出 | ✅ | 两相 WCSPH + Grenier 界面压强 + 双量计 + 增长率拟合，`run_case` 完整管线 |
| L3 系统集成 | 接入主求解器/平台 | ✅ | 独立包模块 `tensorsph.rayleigh_taylor` + 基准驱动 `benchmarks/rayleigh_taylor/run.py` + 测试；**非**主求解器耦合（本算例本身是独立 case，合理） |
| L4 基准验证 | 完整算例+误差分析+可复现 | ❌ | 误差 -24.48 % ≫ 10 %；改进档 0/3 完成；平台重算行 0（CLAIMED_NOT_PROVEN）；无图 |
| L5 科研工程成果 | — | ❌ | 无 |

**级别 = L3。** 附加降级因素：
1. **容差未过**：4/4 完成档 -24.5 %…-61.7 %，最好档 -24.48 %，离 10 % 判据差 14.5 个百分点。
2. **改进未被测量**：v2 文档承诺的 dx=0.02/0.0125(rk2)、dx=0.008(rk2+δ-SPH) 三档全空，理由（集群排队）成立但不是交付物。
3. **收敛未闭环**：σ/σ_th 从 0.383(dx=0.04) → 0.755(dx=0.0125)，dx 减半仅改善 4 个百分点（0.715→0.755），**系统偏低、非单调收敛**。
4. **度量口径不稳定**（关键）：速度模态量计在同档给出 -4.88 %@dx=0.02（表观 PASS）、-11.84 %@dx=0.0125，但在其他档散布到 +16.4 %/-41.1 %/-41.1 %/+13.5 %。即同档两量计最多相差 55 个百分点的 σ/σ_th（dx=0.025, ν=0：质心 0.583 vs 速度 1.135；ν=0.01 档相差 48 点），且跨档无单调收敛；**若仅摘取 dx=0.02 速度量计的 -4.88 % 报 PASS，属事后挑选量计（metric shopping）**，不予采信。
5. **报告体例不全**：v2 文档 ④ 表空 3 行、⑥ 只写「工件与复现+生态摩擦」，未写失败模式分析（v1 文档有）；无图；`tensor_reports` 0 篇。

**与先例的一致性说明**：m16210 交差把「结果 FAIL 但诚实报告」判为 L4。本单不沿用该口径，理由有二：一、本任务书的裁判规则明确「基准判据是唯一裁判」；二、机器裁判对本单直接返回 `l4_benchmark_pass=false / CLAIMED_NOT_PROVEN / rows=0`。若后续平台统一 L4 口径（"有误差分析即 L4" vs "容差通过才 L4"），本判定应随口径重判——已在此留痕。

---

## ③ 落表

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorSPH','m16244 CASE-SPH-RAYLEIGH-TAYLOR-MP',16318,3,'<8 键 JSON>','board-audit');
-- 实际执行：audit/insert_verdict_m16318.py → id=464, gate_state=pending（唯一键 (repo,scope,verdict_by) 幂等 upsert）
```

evidence 八键（值=真实指针，见 `audit/m16318_verdict_evidence.json`）：
`impl_files` / `entry` / `run_artifacts` / `tests` / `integration_callers` / `benchmark` / `repro_cmd` / `external_anchor`。

---

## ④ 发现清单（按严重度）

**阻断级（判定 L4 不成立）**
1. 容差未过：-24.48 % vs <10 %（`runs/table.json`，独立复算一致）。
2. 机器裁判无重算行：`tensorbench_results(m16244)=0` → CLAIMED_NOT_PROVEN。
3. 改进档 0/3 交付：v2 表 3 行空；cj1287 审计时仍排队。

**重要级（方法论/工件质量）**
4. 同档两量计 σ/σ_th 最大相差 55 个百分点（dx=0.025, ν=0：质心 0.583 vs 速度 1.135），且跨档不收敛 → 不能用任一量计单点达标宣称精度。
5. `run.py` 每次运行重写 `runs/summary.json`，仓库里只剩 1/8 行 → 聚合工件失真（复现性缺陷，易误导下游消费者）。**修复建议**：merge-by-tag 写回，或按 `--out` 隔离。
6. dx=0.05 档 `sigma=NaN` 已入 `table.json`（无有效拟合窗口），却在 `summary.json` 里以 NaN 形式流通；建议显式 `status="no_fit"`。
7. `tensor_reports` 0 篇 + 无图：报告六段未按平台口径交付（仓内 md 只是草稿）。

**次要级**
8. `src/tensorsph/__init__.py` 未导出 `rayleigh_taylor`（仅子模块导入可用），包面一致性小瑕疵。
9. 测试未覆盖新增 RK2 / `density_diffusion≠0` 分支——v2 的核心改动无回归测试。
10. 锚点重复登记风险：既有 `CASE-RAYLEIGH-TAYLOR`（m15436，TensorLBM Boussinesq+相场）在**同一线性指标上 6/6 PASS（-2.7 %…-9.9 %）**，而本 SPH 版 -24.5 %。平台已有达标实现，说明 10 % 判据并非不可达；同时也提示算例池在登记 RT 线性增长率时应标注求解器族，避免「同名不同实现」重复铺张。

---

## ⑤ 未执行的检查（honest not_run）

- 全时长复跑 repro_cmd（cj1301 排队中，未取回工件）→ RK2/δ-SPH 实际精度**未测**。
- dx=0.008 档（更细网格）未跑，界面数值阻尼的收敛率（是否真为 O(dx)）未独立确认。
- 未运行全仓回归套件（仅跑 `tests/test_rayleigh_taylor.py`）。

---

## ⑥ 返修清单（给接棒单）

1. 取回/重跑 cj1287 或 cj1301，把 v2 表三行填实数；若 RK2+δ-SPH 仍 >10 %，如实报 FAIL 并给出阻尼模型（A=0 控制组已在 `damping.json`，但只覆盖 dx=0.05/0.025，需补 dx=0.0125）。
2. 定死**单一**公示量计（建议：界面几何量计=质心模态），速度量计只作侧证，禁止事后择量计。
3. 修 `run.py` 的 summary 覆盖写缺陷（merge-by-tag）。
4. 出图（`fields` 已存）：界面场快照 + σ(dx) vs √(Agk) ±10 % 带 + ln η 拟合图。
5. 发 `tensor_reports`（mission_id=16244）六段报告；把本算例登记进 `tensor_benchmark_cases`（目前 0 行，机器无法裁）。
6. 补 `density_diffusion≠0` 与 `time_integrator="rk2"` 的单元/回归测试。

---

## 附：审计可复现命令

```bash
python3 audit/audit_rt_m16318.py          # 取 mission/artifacts/reports 证据 → audit/m16318_evidence.json
python3 audit/audit_rt_referee_m16318.py  # 机器裁判 l4-verdict → audit/m16318_referee.json
python3 audit/independent_refit_m16318.py # 从原始时程独立复算 σ → audit/m16318_independent_refit.json
python3 scripts/m16318_audit_figure.py    # 图 → docs/benchmarks/rayleigh-taylor-mp-audit/fig1_growth_audit.png
python3 audit/insert_verdict_m16318.py    # 落表（幂等）→ tensor_maturity_verdicts id=464
python3 -m pytest tests/test_rayleigh_taylor.py -q   # 10 passed
```
---

## ⑦ 门禁机器复核预演（dry-run）与一处生态组件失真

用平台守护 `maturity_gate.judge()` 对本人 evidence 做只读预演（未触发任何动作）：

```text
judge(claimed=3, evidence, repo='TensorSPH', audit_mid=16318) -> (2, ['L3 缺: integration_callers(grep 无引用)'])
```

**根因不是证据虚标，而是生态镜像滞后**：gate 的 L3 判据在 `/root/tensor-ecosystem/TensorSPH` 里 grep 调用方，
而该镜像 HEAD = `d9a513d`（Merge #533 / m16079），**比被审交付合入后的 `852e941`（Merge #543 / m16244）落后 9 个 PR**，
里面根本没有 `src/tensorsph/rayleigh_taylor.py` / `tests/test_rayleigh_taylor.py` / `benchmarks/rayleigh_taylor/`。
故本 scope 的 gate_level 在镜像同步前必然封顶 L2（L2 判据只看审计工作树，已全绿）。

**结论**：审计声明级别 **L3 不变**（依据=PASS 级实现/入口/工件/测试 + 独立复算），gate 侧 `gate_level=2` 记为
「镜像滞后导致的机械降级」，待 push-merge → 镜像同步重跑即应回到 3。此差异已在 evidence.defects 留痕，
不作为被审单返修理由（被审单返修理由仍是 §⑥ 的精度与交付缺口）。

**附带发现（平台侧，交生态维护）**：`/root/tensor-ecosystem/TensorSPH` 落后上游 9 个 PR，凡依赖该镜像 grep 的门禁
（maturity_gate L3+、claim 复核）都会系统性低估近期合入的能力——建议镜像同步纳入 push-merge 后置动作。
