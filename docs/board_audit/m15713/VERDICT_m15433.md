# 交差核验报告 · m15433 / CASE-TAYLOR-GREEN-SPH（Taylor-Green 涡 · TensorSPH）

> 审计单 m15713（board-audit）· 被审单 m15433（TensorSPH，branch agent/m15433 commit 3843c1d，status=succeeded）
> 判级：**L2**（独立功能实现）· 基准裁判：`tensor_benchmark_cases` 该 anchor **0 行**（表内 22 例）

## 0 结论先行（TL;DR）

| 项 | 结论 | 证据 |
|---|---|---|
| 三件套·run_artifacts | **真**（report.json 9888 B + 4 PNG 51251/54187/40985/43318 B，均在 m15433 工作树，gitignored 未入库） | §3.1 |
| 三件套·benchmark ref-vs-got | **数字在容差内**（decay_rate 2.483650793 vs ref 2.513274123，误差 1.1787% < 8% 任务判据） | §3.2 |
| 三件套·benchmark 正式裁判 | **缺位**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-TAYLOR-GREEN-SPH'` → **0 行** | §3.3 |
| 三件套·repro_cmd | **可执行且逐位复现**：本单实跑 rc=0，main err=1.18% 与原报告逐位一致（decay_rate 2.4836507931478486 完全相同） | §3.4 |
| 测试 | 本单独立实跑 `pytest tests/test_taylor_green_benchmark.py -v` → **2 passed in 22.41s** | §3.5 |
| 报告六段 | **齐全**（tensor_reports id=14720：1 Problem/ref 2 Method 3 Parameters 4 ref-vs-got 5 Convergence/sensitivity 6 Failure modes/artifacts + ecosystem friction） | §3.6 |
| 判级 | **L2**（非 L3：新驱动无调用方；非 L4：基准注册缺位，正式裁判无法机器判；实质达 L4 边界） | §4 |

## 1 案例介绍（①物理背景/几何/参数）

2-D 周期 Taylor-Green 涡：u=U₀e^(−2νk²t)sin(kx)cos(ky)，v=−U₀e^(−2νk²t)cos(kx)sin(ky)，
p=(ρU₀²/4)e^(−4νk²t)(cos2kx+cos2ky)——不可压 Navier-Stokes 的**精确非定常解**（Taylor & Green 1937,
Proc. R. Soc. Lond. A 158, 499-521）。动能必按 E(t)=E(0)e^(−4νk²t) 衰减，故门控量=能量衰减率
s=−d ln E/dt vs 解析指数 4νk²。参数：L=1.0 m，k=2π，U₀=0.2 m/s，ν=1/(10·2π)=0.015915 m²/s，
c₀=10 m/s，ρ₀=1000，γ=7，t_end=0.6 s → Re=U₀/(νk)=2.0，M=U₀/c₀=0.02，h/dx=1.3，CFL≤0.3。

## 2 怎么做的（②方法/流程/工具链）

1. 取证：读 m15433 工作树（/root/tp-agent/work/m15433）三交付文件 + artifacts/taylor_green/；
   读 mission 记录（tensor_dev_missions id=15433）与 tensor_reports id=14720 全文。
2. 基准裁判 SQL：`tensor_benchmark_cases` 全表 22 行枚举，无 TAYLOR/GREEN 匹配（ILIKE 双查）。
3. 独立复跑（不采信原单 stdout）：`PYTHONPATH=src python benchmarks/taylor_green/run.py --out /tmp/tgv_audit_repro`
   （m15433 工作树，setsid 后台，wall≈300 s）→ 与 artifacts/taylor_green/report.json 逐字段比对。
4. 独立 pytest：`python3 -m pytest tests/test_taylor_green_benchmark.py -v`（m15433 工作树）。
5. 集成调用方 grep：`run_tgv|vv.tgv` 于 src/tests/scripts/benchmarks 全仓。
6. 零手抄核对：报告 markdown 数字 vs report.json（main/convergence/mach/re 四组全对）。

## 3 计算结果（③数字表，程序化注入零手抄）

### 3.1 run_artifacts 存在性（m15433 工作树）

| 工件 | 大小 [B] | 状态 |
|---|---|---|
| artifacts/taylor_green/report.json | 9888 | 在 |
| fig1_tgv_velocity_field.png | 51251 | 在 |
| fig2_tgv_energy_decay.png | 54187 | 在 |
| fig3_tgv_convergence.png | 40985 | 在 |
| fig4_tgv_mach_window.png | 43318 | 在 |

### 3.2 ref vs got（本单独立复跑 vs 原报告）

| 量 | ref（解析） | got（SPH 复跑） | 误差 [%] | 判据 <8% |
|---|---|---|---|---|
| 能量衰减率 4νk² [1/s] | 2.5132741228718345 | 2.4836507931478486 | **1.1787** | ✅ |
| 能量比 E(t_end)/E(0) | 0.22136010399060624 | 0.21486398271427973 | 2.93 | ✅ |
| 收敛 dx=0.05/0.025/0.0125 | — | err 2.9998/1.1787/0.7499 | — | ✅ 单调降 |
| Mach 0.002/0.005/0.01/0.02 | — | err 2.45/2.11/2.18/3.00 | — | ✅ 平台期 |
| Mach 0.05 | — | err 173.58 | — | 失效窗（如实报） |
| Re=2/5/10 | — | err 3.00/1.44/12.65 | — | Re=10 超差（如实报） |

### 3.3 基准裁判记录

```sql
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-TAYLOR-GREEN-SPH'
```
返回 **0 行**。表内 22 个 anchor_slug（GAP-*/H*/CAP-*/XVAL-*/REPRO-*/CASE-*）无一匹配 TAYLOR/GREEN。
→ 正式基准裁判**不存在**该锚点判据行；本轮以任务榜判据（<8%）+ 解析解为替代裁判，如实标注非正式基准。

### 3.4 repro_cmd 复现性

原单命令 `PYTHONPATH=src python benchmarks/taylor_green/run.py --out artifacts/taylor_green`，本单实跑
（输出改道 /tmp/tgv_audit_repro）：rc=0，main_case.error_pct 复跑=1.1786748391033548 与原工件
**逐位相同**；convergence/mach/reynolds 三组敏感性行全部一致 → 数字非伪造，确定性可复现。

### 3.5 测试与集成

- pytest（本单实跑）：`2 passed, 1 warning in 22.41s`（warning=未注册 slow 标记，非失败）。
- 集成调用方 grep：新驱动 `benchmarks/taylor_green/run.py` **零调用方**；`run_tgv` 仅被既有
  `src/tensorsph/vv/ladder.py:137,202`（验证梯级，m7114 前作）与测试引用 → 新代码不达 L3。

### 3.6 报告溯源

tensor_reports id=14720（published，mission_id=15433，scenario_key=CASE-TAYLOR-GREEN-SPH，4 图元数据）；
六段齐全；报告数字与 report.json 逐项一致（零手抄）。

## 4 结果图（④）

fig2（SPH 动能衰减 vs 精确 exp(−4νk²t)，主算例 dx=0.025）与 fig3（衰减率误差 vs dx 收敛）内嵌于
API 报告（png_base64）；fig1/fig4 见 m15433 工作树 artifacts/taylor_green/。

## 5 注意点（⑤踩坑/局限/适用范围）

| # | 发现 | 严重度 | 证据 |
|---|---|---|---|
| F1 | `tensor_benchmark_cases` 无 CASE-TAYLOR-GREEN-SPH 行 → 正式基准裁判缺位，L4 无法机器判；建议补注册（metric=能量衰减率相对误差，tolerance=<8%，ref=4νk²=2.5132741228718345） | 中（阻塞 L4，不阻塞 L2） | §3.3 |
| F2 | 交付物仅在分支 agent/m15433（3843c1d），**未并入 main**（merge-base 校验 NOT IN MAIN）；artifacts/* 被 gitignore，复现须回工作树 | 中 | §3.1/§6 |
| F3 | 新驱动零集成调用方（standalone driver）→ L3 不达 | §3.5 |
| F4 | 复现依赖 m15433 工作树（分支未并 main，main 上无 benchmarks/taylor_green/） | 中 | §3.4 |
| F5（正） | 数字逐位可复现、报告六段齐全、零手抄、失效模式（M>0.05 崩 173.6%、Re=10 超差 12.65%）如实申报 | 正 | §3.2/§3.4 |

## 6 误差判定（⑥）

- 关键量=能量衰减率：主算例误差 **1.1787% < 8%（任务榜判据）→ 达标**；是否 <3%？**是**（1.18% < 3%）。
- 能量比 E(t_end)/E(0)：误差 2.93% < 3% → 达标。
- 收敛阶：误差 3.00→1.18→0.75%（dx 减半），与粘性算子 L2 误差（2.65→0.68%）同量级，二阶量级。
- 无外部实验基准（解析解为唯一 ref，属精确解基准，非文献数值表）——已声明。
- honest 申报：Re=10 行 12.65% 超 8%（拟合窗过短，非求解器缺陷，原单已如实报）；M>0.05 失效窗 173.6%（弱可压 SPH 物理局限，原单已如实报）。

## 7 判级与落表

L1✅（2 测试 passed）/ L2✅（真代码 316 行驱动：参数输入→SPH 计算→report.json+4 图输出，逐位复现）/
L3❌（无调用方）/ L4❌（tensor_benchmark_cases 无该 anchor 行，正式裁判缺位；算例完整度/误差分析/可复现
三要素实质齐备，达 L4 边界）→ **裁定 L2**。落表 tensor_maturity_verdicts（verdict_by='board-audit'），
maturity_gate 自动裁定。
