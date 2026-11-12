# 交差核验报告 · CASE-COUETTE-SPH / m15412（TensorSPH）

**审计单**：m15711（board-audit）· **被审对象**：m15412（status=succeeded, PR #497 merged 2026-11-12T00:02Z）
**性质**：交差核验（非重做）。**基准裁判**：`tensor_benchmark_cases`（结果：该锚点 **0 行**，无法裁→回落仓内预声明 `benchmarks/couette/DESIGN.md`）
**结论**：**L3（系统集成级）**，L4 基准验证**不予认定**——预声明门在 H/dx=16 未过（steady_drift 1.55% ≥ 1%），且锚点未登记进平台基准表。**原单无虚标**：其 headline/result/报告三处都自报 FAIL，与本次独立重跑完全一致。

## 0. 核验方法（可复现）

全部结论由 `docs/board-audit-m15711/verify_audit.py` 程序化产出 → `check_results.json`（零手抄）：

```bash
# 1) 独立重跑（交付的命令行，一条不删）
PYTHONPATH=src python benchmarks/couette/run.py --out runs/couette_repro \
  --resolutions 10 16 24 --speeds 0.4 --viscosities 0.04 --frictions 0.0
# 2) 测试
PYTHONPATH=src python -m pytest tests/test_couette.py -q      # 9 passed in 120.13s
# 3) 三件套机器复核（重跑 vs 交付、报告再生、基准表、门限逐项、工件入库）
python3 docs/board-audit-m15711/verify_audit.py
python3 docs/board-audit-m15711/make_figures.py
```

## 1. 三件套核验结果

| 项 | 交付事实 | 核验结论 |
|---|---|---|
| **run_artifacts** | `m15412:runs/couette_full/{results.json,4×png}`（269 KB / 2 图内嵌报告） | 文件真存在、内容真（与本次重跑 **0 差异字段**，排除 wall_clock）；**但 `runs/` 从未入库**（m15412 交付时 `git_status: ?? runs/`；main 的 `git ls-files` 只有 code+report+tests）→ 第三方无法从分支直接取工件，只能重跑（4 分钟，确定性已验证） |
| **benchmark ref vs got** | 剖面 L2 / 通量比 / 壁剪切比 / 稳态漂移逐项 | **数值全复现**（下节表）；四项门逐项机器判定成立 |
| **repro_cmd 可执行** | 报告 §复现 三条命令 | **实测可执行**：`run.py --resolutions 10 16 24 --speeds 0.4 --viscosities 0.04 --frictions 0.0` 在本机 4m20s 跑完 6 个 case；`make_doc.py` 再生 md 与交付 md **逐字节相同**（5491 B） |

**确定性**：独立重跑 results.json 与交付 results.json 规范化后 **diff 行数 = 0**（唯一的非确定性字段 wall_clock_s 已排除）。这同时证明"数字不是手抄/手调"——`make_doc.py` 由工件生成，逐字节一致。

## 2. 基准裁判（唯一裁判）

```
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-COUETTE-SPH';
-- (0 rows)     表内总行数 22
```

**基准登记缺失**。平台侧没有本锚点的 metric/tolerance 行 ⇒ 任务书要求的"对照基准表逐项核"无行可核。实际裁判回落到仓内预声明门（DESIGN.md，随交付一次入库），其门限与 results.json 的 `gates` 字段一致（0.05/0.05/0.05/0.01/0.2）。

## 3. 预声明门 × got 数值（逐项）

| 门 | 阈值 | r10 | **r16（受门）** | **r24（受门）** | 判定 |
|---|---|---|---|---|---|
| interior L2 | <5% | 2.98% | **2.93%** ✅ | **2.95%** ✅ | 过 |
| \|flux ratio −1\| | <5% | 4.26% | **4.42%** ✅ | **4.37%** ✅ | 过 |
| \|wall shear −1\|（下壁） | <5% | 3.24% | **0.48%** ✅ | **1.60%** ✅ | 过 |
| steady drift | <1% | 3.18% | **1.55% ❌** | **0.815%** ✅ | **r16 未过** |
| 负控 flux（f=0） | <0.2 | — | 0.1363 ✅ | — | 过 |
| 分辨率响应（r24 不劣于 r10） | ≤ | — | 2.95% ≤ 2.98% ✅ | — | 过 |

`results.json.verdict = FAIL`，理由 `ladder-r16: gate 'steady_drift' missed`。**DESIGN 的 PASS 规则是 r16 与 r24 同时全过**，故 FAIL 成立，不可能靠 r24 单独通过翻案。

**生态精度标准**：剖面 2.93%、通量 4.42%、壁剪切 0.48% 均远优于 15% 线（图 `fig_couette_audit.png` 面板 b）。失分项是**自设的稳态门**（声学振铃未耗散的数值伪影），不是精度门——两者不能混为一谈，也不能互相顶账。

## 4. 报告六段

① 问题与参考解（Batchelor 1967 §4.2 闭式，无自由参数）② 方法（复用 `tensorsph.droplet_impact.rhs` 的 `wall_vel`+`wall_friction`，无新物理）③ 参数表+无量纲数（Re=1.0、Ma=0.0100、H/dx=16）④ ref vs got 对照表（3 阶梯 + 逐行剖面 + 负控）⑤ 收敛与三轴敏感性 ⑥ 局限与工件。**六段齐全**，且由 `make_doc.py` 从工件生成（再生逐字节相同）。

## 5. 发现（按严重度）

| # | 严重度 | 发现 | 证据 | 影响 |
|---|---|---|---|---|
| F1 | **高（治理）** | 锚点 `CASE-COUETTE-SPH` **未登记** `tensor_benchmark_cases`（0 行）→ 算例池"供给"未闭环，基准裁判空转 | 上述 SQL；`check_results.json.benchmark_registry` | 平台无法机器裁定本算例；同类算例池单可能批量复现此缺口 |
| F2 | **高（可信度）** | 预声明门 FAIL（r16 steady_drift 1.55%≥1%）；即"SPH no-slip 移动壁在 H/dx=16 达到定常"这一能力**未被验证**，只在 H/dx=24 成立 | results.json.verdict；独立重跑同判 | 阻止 L4；后续任何"Couette 已验证"的引用必须带 r24 前提 |
| F3 | **中（可复现性）** | 交付 run 工件 `runs/` 未入 git 树 | m15412 `git_status: ?? runs/`；`git ls-files \| grep '^runs/'` 空 | 分支带不走工件；靠重跑补证（确定性已证，代价 4 min） |
| F4 | **中（度量独立性）** | 壁剪切比 M3 由**首两行剖面的有限差分**再乘 μ 得到，不是 SPH 壁面真实作用力 → M3 与 M1 同源，不构成独立证据；DESIGN 声称 M3 检验壁面条件的说法偏强 | `run.py:397-400`（`slope_of(0,1)` → `rho0*viscosity*slope`） | 门槛判读时 M3 不能当"独立第二证"；独立验证需取壁-流相互作用应力 |
| F5 | **低（解读瑕疵）** | nu 敏感性表里 `wall_shear_ratio=0.9179`（8.2% 偏差）报告列出但未解释；该偏差随 nu 变化恰是"粘性算子误差混入近壁梯度"的信号，报告 ⑥ 未登记 | results.json.viscosity_sweep；报告 5.3 | 结论不受影响（该 case 不受门约束），但读者可能漏掉这条诊断线索 |
| F6 | **低（测试强度）** | 真跑用例阈值宽（flux 0.6~1.2、L2<20%），且 `test_free_slip_wall_does_not_drive_the_fluid` 末句 `flux_ratio < 3.0*0.0+0.2` 与前句重复（恒等式），冗余无效断言 | `tests/test_couette.py` | 不影响门级结论（锐判在 ladder），但该行应删 |
| F7 | **信息** | 交付为 `budget_exhausted+forced_finish`（80 tool_calls），PR #497 由 merge-master 合并、`head_commit=null` | 任务 result JSON | 交付完整性靠磁盘 diff 兜底，非 commit 指纹 |

| F8 | **中（平台机制）** | `maturity_gate.judge()` 的 L4 判据只查 `benchmark` 键"有没有数字"（`has_numbers`），查不到"预声明门是否真的过了"——若审计者声明 4，一个自判 FAIL 的算例仍会铸成 L4。本单靠**声明级 3 封顶**挡住，机制层未挡住 | `maturity_gate.py:134-170`；实测 `judge(4, ev)` → 4（无理由） | 建议 gate 读 `benchmark.case_verdict/status`：非 PASS 则 L4 不成立 |

## 6. 未做/未核（诚实标注）

* 未做**收敛阶**测量（r10→r24 的 L2 几乎不动：2.98%→2.95%，说明误差已被壁面滑移/近壁处理主导，不是格式阶收敛；本单只核交付，不追加新算例）。
* 未在 loong64 集群复核（本单为核验，CPU 单机 4m20s 足够；架构差异未评估）。
* 未核对 4 张 png 内部数据（只核 results.json 与报告 md；图由同文件生成，路径见 `run.py:figure`）。
* 未评估物理适用性外延（转捩/Taylor–Couette/三维），报告 ⑥ 已自我限定。

## 7. 处置

**approve with notes**：交付真实、无造假、无虚标（自报 FAIL 与核验一致），功能实质达 **L3**（真代码 + 真求解器 API 调用 + CI pytest 消费 + 真工件可复现）；**L4 拒绝**（自设门未过 + 锚点未登记）。建议后续：① 把该锚点登记进 `tensor_benchmark_cases`（metric=剖面L2/通量比/壁剪切比/稳态漂移，tolerance=5%/5%/5%/1%）；② run 工件入库或登记到 run 存档；③ M3 改为壁面相互作用应力或明确标注"非独立";④ 放宽/重定义 steady_drift 门（改为分辨率依赖声明）后重开一轮以争取 L4。